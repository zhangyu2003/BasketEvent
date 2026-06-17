import os, json, pickle
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
import torch.nn.functional as F

# -------------------------
# cache utils
# -------------------------

def save_index_cache(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_index_cache(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

LABEL_MAP = {
   "blank": 0,
   "Missed Shot": 1,
   "Made Shot": 2,
   "Free Throw": 3,
   "Foul": 4,
   "Turnover": 5,
   "Jump Ball": 6,
   "Rebound": 7,
   "steal": 8,
   "block": 9,
   "ast": 10
}

# -------------------------
# helpers: bbox loader (single pid) with resize
# -------------------------

def load_bbox_from_json_resized_onepid(
    bbox_info: Dict[str, Any],
    person_id: str,
    kept_indices: List[int],
    scale_x: float,
    scale_y: float,
    to_xyxy: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    bbox_info: 已经 json.load 的 dict（避免每次读文件）
    return:
      bboxes: (T,4)
      valid: (T,)
    """
    pid_key = str(person_id)
    if pid_key not in bbox_info:
        T = len(kept_indices)
        return torch.zeros((T, 4), dtype=dtype), torch.zeros((T,), dtype=dtype)

    traj = bbox_info[pid_key].get("trajectory", [])
    total = len(traj)

    T = len(kept_indices)
    bboxes = torch.zeros((T, 4), dtype=dtype)
    valid = torch.zeros((T,), dtype=dtype)

    for i, fi in enumerate(kept_indices):
        if total <= 0:
            continue
        if fi < 0:
            fi = 0
        elif fi >= total:
            fi = total - 1

        b = traj[fi]
        if b is None or (not isinstance(b, list)) or len(b) != 4:
            continue

        x, y, w, h = map(float, b)
        x *= scale_x; w *= scale_x
        y *= scale_y; h *= scale_y

        if to_xyxy:
            bboxes[i] = torch.tensor([x, y, x + w, y + h], dtype=dtype)
        else:
            bboxes[i] = torch.tensor([x, y, w, h], dtype=dtype)

        valid[i] = 1.0

    return bboxes, valid


def load_ball_from_json_resized(
    bbox_info: Dict[str, Any],
    kept_indices: List[int],
    scale_x: float,
    scale_y: float,
    to_xyxy: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load ball trajectory from json top-level "ball" item.
    return:
      ball_boxes: (T,4)
      valid: (T,)
    """
    ball_info = bbox_info.get("ball", {})
    traj = ball_info.get("trajectory", [])
    total = len(traj)

    T = len(kept_indices)
    ball_boxes = torch.zeros((T, 4), dtype=dtype)
    valid = torch.zeros((T,), dtype=dtype)

    for i, fi in enumerate(kept_indices):
        if total <= 0:
            continue
        if fi < 0:
            fi = 0
        elif fi >= total:
            fi = total - 1

        b = traj[fi]
        if b is None or (not isinstance(b, list)) or len(b) != 4:
            continue

        x, y, w, h = map(float, b)
        x *= scale_x; w *= scale_x
        y *= scale_y; h *= scale_y

        if to_xyxy:
            ball_boxes[i] = torch.tensor([x, y, x + w, y + h], dtype=dtype)
        else:
            ball_boxes[i] = torch.tensor([x, y, w, h], dtype=dtype)

        valid[i] = 1.0

    return ball_boxes, valid

# -------------------------
# Bag dataset (fixed M clips per bag)
# -------------------------
class VideoBagClipsDataset(Dataset):
    """
    一个视频一个 bag；bag 内固定 M 个 clips（bag_clips）。
    clip 采样：
      - 原视频 nominal fps = fps_in
      - 目标 fps = fps_out => sample_stride_frames = round(fps_in / fps_out)
      - clip_len 帧 => clip 覆盖原始帧跨度 clip_span = (clip_len-1)*stride + 1
    bag 内 M 个 clip 起点：
      - max_start = max(0, nums - clip_span)
      - starts = linspace(0, max_start, M)  (短视频 max_start=0 => 全 0)
    """

    def __init__(
        self,
        clip_len: int = 24,
        fps_in: int = 25,
        fps_out: int = 5,
        bag_clips: int = 6,
        size: int = 224,

        bbox_dir: str = "",
        video_dir: str = "",
        cache_path: str = "",
        rebuild_cache: bool = False,
        add_blank: bool = False,
        require_ball: bool = False,
    ):
        self.clip_len = int(clip_len)
        self.fps_in = int(fps_in)
        self.fps_out = int(fps_out)
        self.bag_clips = int(bag_clips)
        self.size = int(size)

        self.bbox_dir = bbox_dir
        self.video_dir = video_dir

        stride_frames = max(1, int(round(self.fps_in / self.fps_out)))
        self.sample_stride_frames = stride_frames

        self.clip_offsets = torch.arange(self.clip_len, dtype=torch.long) * self.sample_stride_frames
        self.clip_span = int(self.clip_offsets[-1].item()) + 1

        self.cache_path = cache_path
        self.index: List[Dict[str, Any]] = []

        self.add_blank = add_blank
        self.require_ball = require_ball

        if not rebuild_cache:
            cache = load_index_cache(self.cache_path)
            if cache is not None and cache.get("meta", {}).get("version") == 2:
                meta = cache["meta"]
                ok = (
                    meta.get("clip_len") == self.clip_len
                    and meta.get("fps_in") == self.fps_in
                    and meta.get("fps_out") == self.fps_out
                    and meta.get("bag_clips") == self.bag_clips
                    and meta.get("size") == self.size
                )
                if ok:
                    print(f"[BagDataset] Loaded cache: {self.cache_path}")
                    self.index = cache["data"]

        if len(self.index) == 0:
            print("[BagDataset] Building bag index (fixed M clips with ball)...")
            self.index = []

            for game in tqdm(os.listdir(self.bbox_dir), desc="Indexing games"):
                bbox_game_dir = os.path.join(self.bbox_dir, game)
                if not os.path.isdir(bbox_game_dir):
                    continue

                for fname in os.listdir(bbox_game_dir):
                    if not fname.endswith(".json"):
                        continue

                    video_name = fname[:-5]
                    bbox_path = os.path.join(bbox_game_dir, fname)
                    # print(bbox_path)
                    with open(bbox_path, "r", encoding="utf-8") as f:
                        info = json.load(f)

                    ball_info = info.get("ball", {})
                    ball_traj = ball_info.get("trajectory")
                    if self.require_ball and ball_traj is None:
                        continue

                    nums = None
                    events = []
                    for pid, pdata in info.items():
                        if pid == "ball":
                            continue

                        traj = pdata.get("trajectory", None)
                        if traj is None:
                            continue
                        if nums is None:
                            nums = len(traj)

                        ev = pdata.get("event", None)
                        if self.add_blank == False:
                            if ev is not None:
                                label = LABEL_MAP.get(ev.get("actionType"), None)
                                if label is not None:
                                    events.append((pid, label))
                        else:
                            if ev is not None:
                                label = LABEL_MAP.get(ev.get("actionType"), None)
                                if label is not None:
                                    events.append((pid, label))
                            else:
                                events.append((pid, 0))

                    if nums is None or nums <= 0 or len(events) == 0:
                        continue

                    max_start = max(0, int(nums) - self.clip_span)
                    if self.bag_clips == 1:
                        starts = [int(max_start // 2)]
                    else:
                        starts = torch.linspace(0, max_start, self.bag_clips).round().long().tolist()

                    if len(starts) != self.bag_clips:
                        starts = (starts + [starts[-1]] * self.bag_clips)[:self.bag_clips]

                    self.index.append({
                        "game": game,
                        "video_name": video_name,
                        "nums": int(nums),
                        "events": events,
                        "starts": starts,
                    })

            cache = {
                "meta": {
                    "version": 2,
                    "clip_len": self.clip_len,
                    "fps_in": self.fps_in,
                    "fps_out": self.fps_out,
                    "bag_clips": self.bag_clips,
                    "size": self.size,
                },
                "data": self.index,
            }
            save_index_cache(self.cache_path, cache)
            print(f"[BagDataset] Saved cache: {self.cache_path}")

    def __len__(self):
        return len(self.index)

    def _build_clip_indices(self, start: int, nums: int) -> torch.Tensor:
        idx = start + self.clip_offsets
        idx = torch.clamp(idx, 0, nums - 1)
        return idx

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.index[i]
        game = item["game"]
        video_name = item["video_name"]
        nums = item["nums"]
        events = item["events"]
        starts = item["starts"]

        video_path = os.path.join(self.video_dir, game, video_name + ".mp4")
        bbox_path  = os.path.join(self.bbox_dir, game, video_name + ".json")

        video_all, _, _ = read_video(video_path, pts_unit="sec")
        total_frames = int(video_all.shape[0])
        if total_frames <= 0:
            raise ValueError(f"No frames read from video: {video_path}")

        with open(bbox_path, "r", encoding="utf-8") as f:
            bbox_info = json.load(f)

        orig_h = int(video_all.shape[1])
        orig_w = int(video_all.shape[2])
        scale_x = float(self.size) / float(orig_w)
        scale_y = float(self.size) / float(orig_h)

        labels = torch.tensor([lab for (_, lab) in events], dtype=torch.long)
        person_ids = [pid for (pid, _) in events]
        N = len(events)
        M = self.bag_clips
        T = self.clip_len

        clips_video = torch.zeros((M, 3, T, self.size, self.size), dtype=torch.float32)
        clips_bboxes = torch.zeros((M, N, T, 4), dtype=torch.float32)
        clips_bbox_mask = torch.zeros((M, N, T), dtype=torch.float32)
        clips_ball = torch.zeros((M, T, 4), dtype=torch.float32)
        clips_ball_mask = torch.zeros((M, T), dtype=torch.float32)
        clips_kept_indices: List[List[int]] = []

        for mi, s in enumerate(starts):
            idx = self._build_clip_indices(int(s), int(nums))
            idx = torch.clamp(idx, 0, total_frames - 1)

            kept = idx.tolist()
            clips_kept_indices.append(kept)

            frames = video_all[idx].float() / 255.0
            frames = frames.permute(0, 3, 1, 2)
            frames = F.interpolate(frames, size=(self.size, self.size), mode="bilinear", align_corners=False)

            mean = torch.tensor([0.45, 0.45, 0.45], dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.225, 0.225, 0.225], dtype=frames.dtype, device=frames.device).view(1, 3, 1, 1)
            frames = (frames - mean) / std

            frames = frames.permute(1, 0, 2, 3).contiguous()
            clips_video[mi] = frames

            for ni, pid in enumerate(person_ids):
                b, m = load_bbox_from_json_resized_onepid(
                    bbox_info=bbox_info,
                    person_id=pid,
                    kept_indices=kept,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    to_xyxy=True,
                    dtype=torch.float32,
                )
                clips_bboxes[mi, ni] = b
                clips_bbox_mask[mi, ni] = m

            ball_bboxes, ball_mask = load_ball_from_json_resized(
                bbox_info=bbox_info,
                kept_indices=kept,
                scale_x=scale_x,
                scale_y=scale_y,
                to_xyxy=True,
                dtype=torch.float32,
            )
            clips_ball[mi] = ball_bboxes
            clips_ball_mask[mi] = ball_mask

        return {
            "clips_video": clips_video,
            "clips_bboxes": clips_bboxes,
            "clips_bbox_mask": clips_bbox_mask,
            "clips_ball": clips_ball,
            "clips_ball_mask": clips_ball_mask,
            "labels": labels,
            "person_ids": person_ids,
            "meta": {
                "game": game,
                "video": video_name,
                "nums": nums,
                "starts": starts,
                "kept_indices": clips_kept_indices,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "fps_in": self.fps_in,
                "fps_out": self.fps_out,
                "sample_stride_frames": self.sample_stride_frames,
                "total_frames_video": total_frames,
            }
        }

# -------------------------
# collate (fixed M -> stack to B,M,...)
# -------------------------

def bag_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    clips_video = torch.stack([x["clips_video"] for x in batch], dim=0)
    B, M = clips_video.shape[0], clips_video.shape[1]

    idx = torch.tensor([x["meta"]["kept_indices"] for x in batch], dtype=torch.long)
    nums = torch.tensor([x["meta"]["nums"] for x in batch], dtype=torch.long)

    bboxes_list = []
    masks_list = []
    labels_list = []
    ball_list = []
    ball_mask_list = []
    metas = []
    person_ids_list = []

    for b in range(B):
        metas.append(batch[b]["meta"])
        labels_list.append(batch[b]["labels"])
        person_ids_list.append(batch[b]["person_ids"])
        cb = batch[b]["clips_bboxes"]
        cm = batch[b]["clips_bbox_mask"]
        ball_list.append(batch[b]["clips_ball"])
        ball_mask_list.append(batch[b]["clips_ball_mask"])

        for m in range(M):
            bboxes_list.append(cb[m])
            masks_list.append(cm[m])

    return {
        "clips_video": clips_video,
        "idx": idx,
        "nums": nums,
        "bboxes": bboxes_list,
        "bbox_masks": masks_list,
        "clips_ball": torch.stack(ball_list, dim=0),
        "clips_ball_mask": torch.stack(ball_mask_list, dim=0),
        "labels": labels_list,
        "metas": metas,
        "person_ids": person_ids_list,
    }

