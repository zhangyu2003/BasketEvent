import os
import json
import argparse
import numpy as np
from sam3.sam3.model_builder import build_sam3_video_predictor
from sam3.sam3.visualization_utils import prepare_masks_for_visualization
import pandas as pd
import torch
import gc

def mask_to_bbox(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def get_frame_dict(outputs_per_frame, frame_idx):
    if frame_idx in outputs_per_frame:
        return outputs_per_frame[frame_idx]
    if str(frame_idx) in outputs_per_frame:
        return outputs_per_frame[str(frame_idx)]
    return {}


def collect_object_ids(outputs_per_frame):
    obj_ids = set()
    for _, obj_dict in outputs_per_frame.items():
        obj_ids.update(obj_dict.keys())
    return sorted(int(x) for x in obj_ids)


def build_trajectory_json(player_outputs, ball_outputs, json_path, num_frames=None):
    """
    Output format:
    {
        "player_0": {
            "trajectory": [bbox_or_None, bbox_or_None, ...]
        },
        "player_1": {
            "trajectory": [...]
        },
        "ball_1": {
            "trajectory": [...]
        }
    }

    bbox format: [x, y, w, h]
    """

    player_ids = collect_object_ids(player_outputs)
    ball_ids = collect_object_ids(ball_outputs)

    all_frame_ids = set()
    all_frame_ids.update(int(k) for k in player_outputs.keys())
    all_frame_ids.update(int(k) for k in ball_outputs.keys())

    if num_frames is None:
        if len(all_frame_ids) == 0:
            num_frames = 0
        else:
            num_frames = max(all_frame_ids) + 1

    result = {}

    # 1. 保存 player trajectory
    # 这里把 SAM3 的原始 player id 重新映射成 player_0, player_1, ...
    for new_pid, raw_pid in enumerate(player_ids):
        object_name = f"player_{new_pid}"
        trajectory = []

        for frame_idx in range(num_frames):
            frame_dict = get_frame_dict(player_outputs, frame_idx)
            mask = frame_dict.get(raw_pid, None)

            if mask is None:
                mask = frame_dict.get(str(raw_pid), None)

            bbox = mask_to_bbox(mask) if mask is not None else None
            trajectory.append(bbox)

        result[object_name] = {
            "trajectory": trajectory
        }

    # 2. 保存 ball trajectory
    # 这里把 ball 命名为 ball_1, ball_2, ...
    for new_bid, raw_bid in enumerate(ball_ids, start=1):
        object_name = f"ball_{new_bid}"
        trajectory = []

        for frame_idx in range(num_frames):
            frame_dict = get_frame_dict(ball_outputs, frame_idx)
            mask = frame_dict.get(raw_bid, None)

            if mask is None:
                mask = frame_dict.get(str(raw_bid), None)

            bbox = mask_to_bbox(mask) if mask is not None else None
            trajectory.append(bbox)

        result[object_name] = {
            "trajectory": trajectory
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def propagate_in_video(predictor, session_id):
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def run_text_prompt(predictor, session_id, prompt_text, frame_index=0):
    predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_index,
            text=prompt_text,
        )
    )

    outputs_per_frame = propagate_in_video(predictor, session_id)
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    return outputs_per_frame

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 video segmentation and export bbox jsons"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--json_save_path",
        type=str,
        default="examples/4712c593-1cd3-fc7f-be55-1b967fadac0f_1280x720_raw.json",
        help="Path to save the output JSON file",
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
    gpus_to_use = [int(x) for x in args.gpus_to_use.split(",")]
    video_path = args.video_path
    json_save_path = args.json_save_path

    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    if not os.path.exists(video_path):
        print(f"[ERROR] video not found, skip: {video_path}")
        return

    session_id = None
    try:
        response = predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = response["session_id"]

        player_outputs = run_text_prompt(
            predictor,
            session_id,
            prompt_text="basketball player on the court",
            frame_index=0,
        )

        ball_outputs = run_text_prompt(
            predictor,
            session_id,
            prompt_text="basketball",
            frame_index=0,
        )

        build_trajectory_json(
            player_outputs=player_outputs,
            ball_outputs=ball_outputs,
            json_path=json_save_path,
        )

    except RuntimeError as e:
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "cuda" in err_msg:
            print(f"[OOM] skip video due to CUDA OOM: {video_path}")
        else:
            print(f"[ERROR] skip video {video_path}: {e}")

    finally:
        # 一定要 close session
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception:
                pass

        # 强制释放显存 & Python 对象
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
