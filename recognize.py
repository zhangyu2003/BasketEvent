import json, cv2
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import os, torch, math, re
from PIL import Image, ImageDraw, ImageFont
import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from collections import defaultdict

def crop_by_bbox_xywh(img_bgr: np.ndarray, bbox_xywh: List[int], pad_ratio: float = 0.2) -> Optional[np.ndarray]:
    H, W = img_bgr.shape[:2]
    x, y, w, h = map(int, bbox_xywh)

    if w <= 1 or h <= 1:
        return None

    pad_w = int(w * pad_ratio)
    pad_h = int(h * pad_ratio)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(W, x + w + pad_w)
    y2 = min(H, y + h + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None

    return img_bgr[y1:y2, x1:x2].copy()


def _uniform_sample_indices(n, k: int = 10) -> List[int]:
    if n == 0:
        return []
    if n == 1:
        return [0] * k
    
    pos = np.linspace(0, n - 1, num=k)
    idx = np.rint(pos).astype(int)
    idx = np.clip(idx, 0, n - 1)
    return idx


def build_pid_to_crops(
    video_path: str,
    bbox_json_path: str,
    k: int = 10,
    frame_stride: int = 1,
    pad_ratio: float = 0.0,
    rec_type: str = "player",
) -> Dict[str, List[Image.Image]]:

    with open(bbox_json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    pid_to_frames: Dict[str, List[int]] = defaultdict(list)
    for pid in ann.keys():
        if pid.startswith(rec_type):
            pid_to_frames[pid] = ann[pid]["trajectory"]
            num_frames = len(pid_to_frames[pid])

    idx = _uniform_sample_indices(num_frames, k=k)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    pid_to_crops: Dict[str, List[Image.Image]] = {pid: [] for pid in pid_to_frames.keys()}

    for fi in sorted(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        for pid in pid_to_frames.keys():
            bbox_xywh = ann[pid]["trajectory"][fi]
            if bbox_xywh is None:
                continue
            crop = crop_by_bbox_xywh(frame, bbox_xywh, pad_ratio=pad_ratio)
            if crop is None or crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb).convert("RGB")
            pid_to_crops[pid].append(pil_image)

    cap.release()

    for pid in list(pid_to_crops.keys()):
        imgs = pid_to_crops[pid]
        if len(imgs) == 0:
            pid_to_crops.pop(pid, None)
            continue
        if len(imgs) < k:
            imgs.extend([imgs[-1]] * (k - len(imgs)))
        elif len(imgs) > k:
            pid_to_crops[pid] = imgs[:k]

    return pid_to_crops

def build_roster_text(roster_json_path: str) -> str:
    with open(roster_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_color = {k: str(v).lower() for k, v in data.get("jersey_color", {}).items()}
    players = data.get("players", [])

    # color -> number -> [names]
    color_map = defaultdict(lambda: defaultdict(list))
    for p in players:
        team = p.get("team_name")
        color = team_color.get(team, None)
        jersey = str(p.get("jersey")).strip()
        name = str(p.get("name")).strip()
        if color and jersey and name:
            color_map[color][jersey].append(name)

    lines = []
    lines.append(f"Team jersey colors in this game: {team_color}")
    lines.append("Roster (use this table to map jersey number + jersey color to player name):")
    for color in sorted(color_map.keys()):
        lines.append(f"- Color: {color}")
        # number sorted numerically when possible
        def _num_key(x):
            try: return int(x)
            except: return 10**9
        for num in sorted(color_map[color].keys(), key=_num_key):
            names = color_map[color][num]
            # 如果同色同号理论上不该有多个，但以防万一保留列表
            lines.append(f"  {num}: {', '.join(names)}")
    return "\n".join(lines)

def build_onepass_prompt(roster_text: str) -> str:
    return f"""You are given 10 cropped images that are supposed to show the SAME tracked entity from a basketball game (same player ID), but the tracking/bounding boxes may be noisy. Some crops may contain:
- referees (officials),
- bench players / people sitting on the sideline,
- coaches or staff,
- audience,
- or completely irrelevant/empty crops.

Your job is to decide whether this ID corresponds to a VALID on-court player from the roster.
If it is NOT a valid on-court player, you MUST say so and DO NOT guess a roster name.

Task (single pass):
1) Validate the ID:
   - Determine if the 10 crops mostly show a real on-court player wearing a team jersey.
   - If the crops mostly show a referee / audience / bench / staff / irrelevant content, mark it as invalid.
2) If valid, determine:
   - jersey_number (1–2 digits, 0–99) from the images,
   - jersey_color (choose only from colors present in the roster info),
   - then map (jersey_color, jersey_number) to a player_name using the roster table below.

Rules:
- Use cross-image consistency: not every crop is clear; ignore unreadable or irrelevant crops.
- Only output a player_name if the ID is valid AND (jersey_color, jersey_number) matches the roster.
- If jersey number/color cannot be determined reliably, return player_name as null.
- NEVER output a name that is not present in the roster.
- If invalid, set jersey_number, jersey_color, and player_name to null.

=== ROSTER INFO ===
{roster_text}
=== END ROSTER INFO ===

Return STRICT JSON only (no extra text):
{{
  "is_valid_player": true/false,
  "invalid_reason": "<short reason or null>",
  "jersey_number": <number or null>,
  "jersey_color": "<color or null>",
  "player_name": "<name or null>",
  "confidence": "low|medium|high",
  "evidence": "briefly state which images (1-10) were most informative and why"
}}

Guidance for validity:
- Likely VALID on-court player: visible basketball uniform/jersey, player body proportion, court context, consistent across multiple crops.
- Likely INVALID: referee uniform (often gray/black with stripes), whistle, no jersey number anywhere across all crops, seated bench context, crowd faces, empty/background, heavy blur for most crops, or different people across the 10 crops.
"""

def parse_vlm_json(text: str) -> Dict[str, Any]:
    if isinstance(text, list):
        text = text[0]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "is_valid_player": False,
            "invalid_reason": f"JSON parse error: {str(e)}",
            "jersey_number": None,
            "jersey_color": None,
            "player_name": None,
            "confidence": "low",
            "evidence": "Model output could not be parsed as JSON",
            "_raw_output": text,
        }

def rec_one_video(model, processor, video_path, bbox_json_path, roster_json, device):
    pid2crops = build_pid_to_crops(
        video_path=video_path,
        bbox_json_path=bbox_json_path,
        k=10,
        frame_stride=1,
        pad_ratio=0.0,
        rec_type="player",
    )

    roster_text = build_roster_text(roster_json)
    prompt = build_onepass_prompt(roster_text)

    results = {}

    for pid, images in pid2crops.items():
        print(f"Inferencing PID {pid}")

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        results[str(pid)] = parse_vlm_json(output_text)

    return results


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def collect_ball_tracks(
    ann: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Optional[Tuple[float, float, float, float]]]], List[str]]:
    tracks_by_frame: Dict[int, Dict[str, Optional[Tuple[float, float, float, float]]]] = defaultdict(dict)
    candidate_ids = []

    for obj_id, payload in ann.items():
        obj_id = str(obj_id)
        if not obj_id.startswith("ball"):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("trajectory"), list):
            continue

        has_bbox = False
        for frame_id, bbox in enumerate(payload["trajectory"]):
            if bbox is None:
                continue
            if len(bbox) != 4:
                raise ValueError(f"Invalid ball bbox at frame {frame_id}, id {obj_id}: {bbox}")
            tracks_by_frame[frame_id][obj_id] = tuple(float(v) for v in bbox)
            has_bbox = True

        if has_bbox:
            candidate_ids.append(obj_id)

    return dict(tracks_by_frame), candidate_ids


def collect_by_id(
    tracks_by_frame: Dict[int, Dict[str, Optional[Tuple[float, float, float, float]]]]
) -> Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]]:
    by_id: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = defaultdict(list)
    for frame_id, obj_dict in tracks_by_frame.items():
        for obj_id, bbox in obj_dict.items():
            if bbox is not None:
                by_id[str(obj_id)].append((frame_id, bbox))
    return {k: sorted(v, key=lambda x: x[0]) for k, v in by_id.items()}


def segment_lengths(frames: List[int]) -> List[int]:
    if not frames:
        return []

    segs = []
    start = prev = frames[0]
    for fr in frames[1:]:
        if fr == prev + 1:
            prev = fr
        else:
            segs.append(prev - start + 1)
            start = prev = fr
    segs.append(prev - start + 1)
    return segs


def compute_ball_track_stats(
    tracks_by_frame: Dict[int, Dict[str, Optional[Tuple[float, float, float, float]]]]
) -> List[Dict[str, Any]]:
    by_id = collect_by_id(tracks_by_frame)
    total_frames = max(tracks_by_frame.keys(), default=-1) + 1
    rows = []

    for obj_id, items in by_id.items():
        frames = [f for f, _ in items]
        b = np.array([bbox for _, bbox in items], dtype=np.float32)
        x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx, cy = x + w / 2.0, y + h / 2.0
        area = w * h

        speeds = []
        for i in range(1, len(frames)):
            gap = frames[i] - frames[i - 1]
            if gap <= 0:
                continue
            dist = math.hypot(float(cx[i] - cx[i - 1]), float(cy[i] - cy[i - 1]))
            speeds.append(dist / gap)

        segs = segment_lengths(frames)
        row = {
            "id": str(obj_id),
            "n_frames": len(frames),
            "start_frame": min(frames),
            "end_frame": max(frames),
            "coverage": len(frames) / max(total_frames, 1),
            "segments": len(segs),
            "max_segment_len": max(segs) if segs else 0,
            "mean_w": float(np.mean(w)),
            "mean_h": float(np.mean(h)),
            "mean_area": float(np.mean(area)),
            "mean_aspect": float(np.mean(w / np.maximum(h, 1e-6))),
            "median_speed_px_per_frame": float(np.median(speeds)) if speeds else 0.0,
            "move_range": float(math.hypot(float(np.max(cx) - np.min(cx)), float(np.max(cy) - np.min(cy)))),
        }
        row["heuristic_score"] = heuristic_ball_score(row)
        rows.append(row)

    return sorted(rows, key=lambda r: r["heuristic_score"], reverse=True)


def heuristic_ball_score(row: Dict[str, Any]) -> float:
    mean_w = row["mean_w"]
    mean_h = row["mean_h"]
    mean_area = row["mean_area"]
    aspect = row["mean_aspect"]

    size_score = 1.0
    if mean_w < 6 or mean_h < 6:
        size_score *= 0.2
    if mean_w > 50 or mean_h > 50:
        size_score *= 0.15
    if mean_area > 2500:
        size_score *= 0.05

    aspect_score = min(1.0, max(0.0, 1.0 - abs(aspect - 1.0)) + 0.4)
    motion_score = min(1.0, row["move_range"] / 500.0)
    speed_score = min(1.0, row["median_speed_px_per_frame"] / 10.0)
    continuity_score = min(1.0, row["max_segment_len"] / 80.0)
    return float(
        0.30 * size_score
        + 0.20 * aspect_score
        + 0.20 * motion_score
        + 0.20 * speed_score
        + 0.10 * continuity_score
    )


def choose_sample_items(items: List[Tuple[int, Tuple[float, float, float, float]]], k: int = 8):
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [items[i] for i in idx]


def crop_ball_with_context(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 3.5,
) -> Optional[np.ndarray]:
    H, W = frame.shape[:2]
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return None

    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(max(w, h) * pad_ratio, 112)
    x1 = clamp(int(round(cx - side / 2)), 0, W - 1)
    y1 = clamp(int(round(cy - side / 2)), 0, H - 1)
    x2 = clamp(int(round(cx + side / 2)), 0, W)
    y2 = clamp(int(round(cy + side / 2)), 0, H)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()
    bx1 = int(round(x - x1))
    by1 = int(round(y - y1))
    bx2 = int(round(x + w - x1))
    by2 = int(round(y + h - y1))
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    return crop


def build_ball_contact_sheet(
    video_path: str,
    tracks_by_frame: Dict[int, Dict[str, Optional[Tuple[float, float, float, float]]]],
    candidate_ids: List[str],
    samples_per_id: int = 8,
    cell_size: int = 180,
    pad_ratio: float = 3.5,
) -> Image.Image:
    by_id = collect_by_id(tracks_by_frame)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    font = ImageFont.load_default()
    rows = []

    for obj_id in candidate_ids:
        cells = []
        for frame_id, bbox in choose_sample_items(by_id.get(str(obj_id), []), k=samples_per_id):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            crop = crop_ball_with_context(frame, bbox, pad_ratio=pad_ratio)
            if crop is None or crop.size == 0:
                continue

            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).convert("RGB")
            pil = pil.resize((cell_size, cell_size), Image.BICUBIC)
            draw = ImageDraw.Draw(pil)
            draw.rectangle([0, 0, cell_size, 20], fill=(255, 255, 255))
            draw.text((4, 4), f"id={obj_id} f={frame_id}", fill=(0, 0, 0), font=font)
            cells.append(pil)

        if not cells:
            continue

        row = Image.new("RGB", (cell_size * samples_per_id, cell_size + 34), color=(245, 245, 245))
        draw = ImageDraw.Draw(row)
        draw.text((4, cell_size + 10), f"candidate ball id = {obj_id}", fill=(0, 0, 0), font=font)
        for i, cell in enumerate(cells):
            row.paste(cell, (i * cell_size, 0))
        rows.append(row)

    cap.release()

    if not rows:
        raise RuntimeError("No valid ball candidate crops generated.")

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 44
    sheet = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "Find the real basketball track id. Red boxes are bboxes.", fill=(0, 0, 0), font=font)

    y = 44
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def build_ball_prompt(stats_rows: List[Dict[str, Any]]) -> str:
    stats_text = json.dumps(stats_rows, ensure_ascii=False, indent=2)
    return f"""You are a basketball video tracking reviewer. Each row in the image shows several cropped frames for one candidate ball track id. The red boxes show the bbox for that id.

Task: determine which track id is most likely the real basketball.

Rules:
1. The real basketball must be a physical basketball in play: held, passed, dribbled, shot, or flying.
2. Exclude floor logos, court patterns, scoreboard graphics, ball-shaped ads, audience decorations, and broadcast overlays.
3. Exclude player bodies, shoes, jerseys, arms, rims, nets, timers, and spectators.
4. A real basketball is usually orange/brown, roughly round, small, and has similar bbox width and height.
5. Judge by consistency across multiple frames, not one frame.
6. Very large boxes or nearly fixed positions are likely false targets.
7. Choose exactly one candidate id if there is a real basketball; otherwise return null.

Track statistics:
{stats_text}

Return STRICT JSON only:
{{
  "real_ball_id": "candidate id string, or null if uncertain",
  "confidence": 0.0,
  "reason": "one sentence",
  "ranking": [
    {{"id": "candidate id", "score": 0.0, "comment": "short reason"}}
  ]
}}"""


def parse_ball_vlm_json(text: str) -> Dict[str, Any]:
    if isinstance(text, list):
        text = text[0]
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    return {
        "real_ball_id": None,
        "confidence": 0.0,
        "reason": "Model output could not be parsed as JSON",
        "ranking": [],
        "_raw_output": text,
    }


def rec_ball_one_video(
    model,
    processor,
    video_path: str,
    bbox_json_path: str,
    device: str,
    topk_candidates: int = 8,
    samples_per_id: int = 8,
) -> Dict[str, Any]:
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    tracks_by_frame, ball_ids = collect_ball_tracks(ann)
    stats_rows = compute_ball_track_stats(tracks_by_frame)
    if not stats_rows:
        return {
            "selected_ball_id": None,
            "qwen_json": {
                "real_ball_id": None,
                "confidence": 0.0,
                "reason": "No non-null ball candidate was found.",
                "ranking": [],
            },
        }

    candidate_ids = [r["id"] for r in stats_rows[:topk_candidates]]
    if len(candidate_ids) == 1:
        selected_id = candidate_ids[0]
        return {
            "selected_ball_id": selected_id,
            "qwen_json": {
                "real_ball_id": selected_id,
                "confidence": 1.0,
                "reason": "Only one non-null ball candidate exists.",
                "ranking": [{"id": selected_id, "score": 1.0, "comment": "single candidate"}],
            },
            "stats": stats_rows,
        }

    sheet = build_ball_contact_sheet(
        video_path=video_path,
        tracks_by_frame=tracks_by_frame,
        candidate_ids=candidate_ids,
        samples_per_id=samples_per_id,
    )
    prompt = build_ball_prompt([r for r in stats_rows if r["id"] in set(candidate_ids)])
    messages = [{"role": "user", "content": [{"type": "image", "image": sheet}, {"type": "text", "text": prompt}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    qwen_json = parse_ball_vlm_json(output_text)
    selected_id = qwen_json.get("real_ball_id")
    if selected_id is not None:
        selected_id = str(selected_id)
    if selected_id not in set(candidate_ids):
        selected_id = None

    return {
        "selected_ball_id": selected_id,
        "candidate_ids": candidate_ids,
        "qwen_json": qwen_json,
        "qwen_raw_output": output_text,
        "stats": stats_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--bbox_json_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720_raw.json",
        help="Path to the input bbox JSON file",
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.json",
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--roster_json",
        type=str,
        default="examples/players_info_0022500006.json",
        help="Path to the roster JSON file",
    )
    parser.add_argument(
        "--gpus_to_use",
        type=str,
        default="6",
        help="GPU ids, e.g. '0' or '0,1,2'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video_path
    bbox_json_path = args.bbox_json_path
    json_save_path = args.json_save_path
    ROSTER_JSON = args.roster_json
    gpus_to_use = args.gpus_to_use
    device = f"cuda:{gpus_to_use.split(',')[0]}" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen2.5-VL-7B-Instruct", torch_dtype="auto"
    ).to(device)
    processor = AutoProcessor.from_pretrained("Qwen2.5-VL-7B-Instruct")

    results = rec_one_video(model, processor, video_path, bbox_json_path, ROSTER_JSON, device=device)
    # print(results)
    with open(bbox_json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    valid_results = {}
    valid_count = 0
    for pid, out in results.items():
        if out.get("is_valid_player") is True:
            valid_results[f"player_{valid_count}"] = {
                "jersey_number": out.get("jersey_number"),
                "jersey_color": out.get("jersey_color"),
                "player_name": out.get("player_name"),
                "trajectory": ann.get(pid, {}).get("trajectory"),
            }
            valid_count += 1

    ball_result = rec_ball_one_video(
        model=model,
        processor=processor,
        video_path=video_path,
        bbox_json_path=bbox_json_path,
        device=device,
    )
    selected_ball_id = ball_result.get("selected_ball_id")
    if selected_ball_id is not None and selected_ball_id in ann:
        valid_results["ball"] = {
            "trajectory": ann.get(selected_ball_id, {}).get("trajectory"),
        }

    with open(json_save_path, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, ensure_ascii=False, indent=2)

    

if __name__ == "__main__":
    main()
