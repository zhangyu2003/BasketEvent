import sys

import os
import argparse
import csv
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.dataset import VideoBagClipsDataset, bag_collate_fn, LABEL_MAP
from src.model import PlayerEventModel


# =========================================================
# DDP setup
# =========================================================
def setup_ddp():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# =========================================================
# load ckpt
# =========================================================
def load_ckpt(ckpt_path: str, model_ddp: DDP, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_ddp.module.load_state_dict(ckpt["model"])
    return ckpt.get("epoch", None), ckpt.get("global_step", None)


# =========================================================
# helper: slice one video from collated batch
# =========================================================
def slice_video_from_collated_batch(batch, b: int, device: torch.device):
    clips_video = batch["clips_video"].to(device, non_blocking=True)  # (B,M,C,T,H,W)
    idx = batch["idx"].to(device, non_blocking=True)                  # (B,M,T)
    nums = batch["nums"].to(device, non_blocking=True)                # (B,)

    B, M = clips_video.shape[:2]
    assert 0 <= b < B

    video_b = clips_video[b]          # (M,C,T,H,W)
    idx_b = idx[b]                    # (M,T)
    nums_b = nums[b].repeat(M)        # (M,)

    base = b * M
    bboxes_b = [
        batch["bboxes"][base + m].to(device, non_blocking=True)
        for m in range(M)
    ]
    masks_b = [
        batch["bbox_masks"][base + m].to(device, non_blocking=True)
        for m in range(M)
    ]

    clips_ball_b = batch["clips_ball"][b].to(device, non_blocking=True)       # (M,T,4)
    clips_ball_mask_b = batch["clips_ball_mask"][b].to(device, non_blocking=True)  # (M,T)

    labels_b = batch["labels"][b].to(device, non_blocking=True)  # (N,)

    meta_b = {}
    if "metas" in batch:
        meta_b = batch["metas"][b]
    elif "meta" in batch:
        meta_b = batch["meta"][b]

    return video_b, idx_b, nums_b, bboxes_b, masks_b, clips_ball_b, clips_ball_mask_b, labels_b, meta_b


# =========================================================
# top-k correctness per person
# =========================================================
@torch.no_grad()
def topk_correct_counts(logits: torch.Tensor, target: torch.Tensor, ks=(1, 3, 5)):
    """
    logits: (N,C)
    target: (N,)
    Returns:
      correct_k: dict{k: correct_count}
      total: N
    """
    N, C = logits.shape
    total = int(N)

    maxk = min(max(ks), C)
    topk = torch.topk(logits, k=maxk, dim=1).indices  # (N,maxk)
    tgt = target.view(-1, 1)                          # (N,1)

    correct = (topk == tgt)  # (N,maxk) bool

    out = {}
    for k in ks:
        kk = min(k, C)
        out[k] = int(correct[:, :kk].any(dim=1).sum().item())
    return out, total


# =========================================================
# args
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--bag_clips", type=int, default=12)
    p.add_argument("--clip_len", type=int, default=8)
    p.add_argument("--fps_in", type=int, default=25)
    p.add_argument("--fps_out", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cache_name", type=str, default="")
    p.add_argument("--rebuild_cache", action="store_true")

    # eval
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=3, help="MIL internal topk arg (kept for API compatibility)")

    # ckpt / metric
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--bg_id", type=int, default=0, help="background / blank class id")
    p.add_argument("--time_csv", type=str, required=True)
    p.add_argument("--strict_gate", action="store_true", help="raise error if model output has no gate_weights")

    return p.parse_args()


# =========================================================
# helper: per-video record build
# =========================================================
@torch.no_grad()
def build_clip_ranges(idx_b: torch.Tensor, fps_in: float):
    """
    idx_b: (M,T)
    returns:
      clip_ranges_frame: list[[start_f, end_f]] len=M
      clip_ranges_sec:   list[[start_sec, end_sec]] len=M
    """
    idx_cpu = idx_b.detach().cpu()
    clip_ranges_frame = []
    clip_ranges_sec = []

    M = idx_cpu.shape[0]
    for m in range(M):
        start_f = int(idx_cpu[m, 0].item())
        end_f = int(idx_cpu[m, -1].item())
        clip_ranges_frame.append([start_f, end_f])
        clip_ranges_sec.append([start_f / float(fps_in), end_f / float(fps_in)])

    return clip_ranges_frame, clip_ranges_sec


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def normalize_video_name(x) -> str:
    return os.path.basename(str(x))


def make_time_key(game_id, video_name, person_id) -> str:
    return f"{str(game_id)}||{normalize_video_name(video_name)}||{str(person_id)}"


def load_time_annotations(csv_path: str):
    annotations = {}
    if not csv_path or not os.path.exists(csv_path):
        return annotations

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = make_time_key(row["game_id"], row["videoname"], row["person_id"])
                annotations[key] = {
                    "starttime": float(row["starttime"]),
                    "endtime": float(row["endtime"]),
                }
            except Exception:
                continue
    return annotations


def extract_top1_pred(record: dict):
    return safe_int(record.get("pred_top1", None))


def extract_person_id(record: dict):
    for key in ("person_id", "pid", "player_id", "track_id", "person_video_id"):
        if key in record:
            return str(record[key])
    return None


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    len_a = max(0.0, a_end - a_start)
    len_b = max(0.0, b_end - b_start)
    denom = min(len_a, len_b)
    return inter / denom if denom > 0 else 0.0


def extract_pred_time_segment(record: dict):
    gate_weights = record.get("gate_weights", None)
    clip_ranges = record.get("clip_ranges_sec", None)
    if not gate_weights or not clip_ranges:
        return None

    max_idx = max(range(len(gate_weights)), key=lambda i: float(gate_weights[i]))
    if max_idx >= len(clip_ranges):
        return None

    pred_start, pred_end = clip_ranges[max_idx]
    pred_start = float(pred_start)
    pred_end = float(pred_end)
    return pred_start, pred_end, (pred_start + pred_end) / 2.0, max_idx


def keep_record(record: dict, bg_id):
    gt = safe_int(record.get("gt", None))
    if gt is None:
        return False
    return bg_id is None or gt != int(bg_id)


def compute_topk_acc(records, bg_id=None):
    kept = [r for r in records if keep_record(r, bg_id)]
    total = len(kept)
    if total == 0:
        return {1: 0.0, 3: 0.0, 5: 0.0}, 0
    return {
        k: sum(1 for r in kept if bool(r.get(f"correct{k}", False))) / total
        for k in (1, 3, 5)
    }, total


def compute_mean_recall_topk(records, k: int, bg_id=None):
    total_per_class = {}
    hit_per_class = {}

    for r in records:
        if not keep_record(r, bg_id):
            continue
        gt = int(r["gt"])
        total_per_class[gt] = total_per_class.get(gt, 0) + 1
        if bool(r.get(f"correct{k}", False)):
            hit_per_class[gt] = hit_per_class.get(gt, 0) + 1

    if not total_per_class:
        return 0.0

    recalls = [
        hit_per_class.get(c, 0) / total
        for c, total in total_per_class.items()
        if total > 0
    ]
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_macro_prf1(records, bg_id=None):
    y_true = []
    y_pred = []

    for r in records:
        if not keep_record(r, bg_id):
            continue
        pred = extract_top1_pred(r)
        if pred is None:
            continue
        y_true.append(int(r["gt"]))
        y_pred.append(pred)

    if not y_true:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    classes = sorted(set(y_true) | set(y_pred))
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for c in classes:
        tp = sum(1 for gt, pred in zip(y_true, y_pred) if gt == c and pred == c)
        fp = sum(1 for gt, pred in zip(y_true, y_pred) if gt != c and pred == c)
        fn = sum(1 for gt, pred in zip(y_true, y_pred) if gt == c and pred != c)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    n = len(classes)
    return {
        "precision": precision_sum / n,
        "recall": recall_sum / n,
        "f1": f1_sum / n,
    }


def compute_temporal_hits(records, time_annotations, bg_id=0):
    tiou_list = []

    for r in records:
        gt = safe_int(r.get("gt", None))
        if gt is None or gt == int(bg_id):
            continue

        game = r.get("game", r.get("game_id", None))
        video = r.get("video", r.get("videoname", r.get("video_name", None)))
        person_id = extract_person_id(r)
        if game is None or video is None or person_id is None:
            continue

        key = make_time_key(game, video, person_id)
        if key not in time_annotations:
            continue

        pred_seg = extract_pred_time_segment(r)
        if pred_seg is None:
            continue

        pred_start, pred_end, _, _ = pred_seg
        pred = extract_top1_pred(r)
        if pred is None or pred != gt:
            tiou_list.append(0.0)
            continue

        gt_start = float(time_annotations[key]["starttime"])
        gt_end = float(time_annotations[key]["endtime"])
        tiou_list.append(temporal_iou(gt_start, gt_end, pred_start, pred_end))

    total = len(tiou_list)
    if total == 0:
        return {"num_annotated_samples": 0, "Hit@0.1": 0.0, "Hit@0.3": 0.0, "Hit@0.5": 0.0}

    return {
        "num_annotated_samples": total,
        "Hit@0.1": sum(1 for x in tiou_list if x >= 0.1) / total,
        "Hit@0.3": sum(1 for x in tiou_list if x >= 0.3) / total,
        "Hit@0.5": sum(1 for x in tiou_list if x >= 0.5) / total,
    }


def compute_metrics(records, time_annotations, bg_id=0):
    acc_with_bg, total_with_bg = compute_topk_acc(records, bg_id=None)
    acc_no_bg, total_no_bg = compute_topk_acc(records, bg_id=bg_id)
    recall_no_bg = {
        k: compute_mean_recall_topk(records, k, bg_id=bg_id)
        for k in (1, 3, 5)
    }
    prf1_no_bg = compute_macro_prf1(records, bg_id=bg_id)
    temporal = compute_temporal_hits(records, time_annotations, bg_id=bg_id)

    return {
        "total_with_bg": total_with_bg,
        "total_no_bg": total_no_bg,
        "acc_with_bg": acc_with_bg,
        "acc_no_bg": acc_no_bg,
        "recall_no_bg": recall_no_bg,
        "precision_no_bg": prf1_no_bg["precision"],
        "f1_no_bg": prf1_no_bg["f1"],
        "temporal": temporal,
    }


def print_metrics(ckpt_path: str, epoch, global_step, metrics: dict):
    temporal = metrics["temporal"]
    print("=" * 100)
    print(f"ckpt: {ckpt_path}")
    print(f"epoch={epoch} global_step={global_step}")
    print(f"total_with_bg={metrics['total_with_bg']} total_no_bg={metrics['total_no_bg']}")
    print(
        "top135 acc no_bg: "
        f"{metrics['acc_no_bg'][1]:.6f}, {metrics['acc_no_bg'][3]:.6f}, {metrics['acc_no_bg'][5]:.6f}"
    )
    print(
        "top135 recall no_bg: "
        f"{metrics['recall_no_bg'][1]:.6f}, {metrics['recall_no_bg'][3]:.6f}, {metrics['recall_no_bg'][5]:.6f}"
    )
    print(f"precision no_bg: {metrics['precision_no_bg']:.6f}")
    print(f"F1 no_bg: {metrics['f1_no_bg']:.6f}")
    print(
        f"Hit@0.1/0.3/0.5: "
        f"{temporal['Hit@0.1']:.6f}, {temporal['Hit@0.3']:.6f}, {temporal['Hit@0.5']:.6f} "
        f"(N={temporal['num_annotated_samples']})"
    )
    print(
        "top135 acc with_bg: "
        f"{metrics['acc_with_bg'][1]:.6f}, {metrics['acc_with_bg'][3]:.6f}, {metrics['acc_with_bg'][5]:.6f}"
    )
    print("=" * 100)


# =========================================================
# main
# =========================================================
@torch.no_grad()
def main():
    args = parse_args()
    rank, world_size, local_rank, device = setup_ddp()

    if is_main(rank):
        print(f"[DDP Eval] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")
        print(f"[Eval] ckpt={args.ckpt} MIL_topk_arg={args.topk}")
        print(f"[Metric] bg_id={args.bg_id} time_csv={args.time_csv}")

    obj = [args.ckpt]
    dist.broadcast_object_list(obj, src=0)
    ckpt_path = obj[0]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    time_annotations = load_time_annotations(args.time_csv)
    if is_main(rank):
        print(f"[Metric] loaded time annotations: {len(time_annotations)}")

    # -------------------------
    # dataset / sampler / loader
    # -------------------------

    cache_name = f"bag_clip{args.clip_len}_fps{args.fps_out}_M{args.bag_clips}_test.pkl"
    cache_path = os.path.join(args.cache_dir, cache_name)


    dataset = VideoBagClipsDataset(
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        bag_clips=args.bag_clips,
        bbox_dir=args.test_dir,
        video_dir=args.video_dir,
        size=args.img_size,
        cache_path=cache_path,
        add_blank=True,
        rebuild_cache=args.rebuild_cache,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=bag_collate_fn,
        drop_last=False,
    )

    # -------------------------
    # model
    # -------------------------
    num_classes = len(LABEL_MAP)
    model = PlayerEventModel(
        num_classes=num_classes,
        use_clip_relation=True,
        use_actor_global=True,
        use_person_relation=True,
    ).to(device)

    model_ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ks = (1, 3, 5)

    ep, gs = load_ckpt(ckpt_path, model_ddp, device=device)
    model_ddp.eval()

    if is_main(rank):
        print(f"\n[Eval] ckpt={ckpt_path} epoch={ep} global_step={gs}")

    total_people = torch.tensor(0.0, device=device)
    correct1 = torch.tensor(0.0, device=device)
    correct3 = torch.tensor(0.0, device=device)
    correct5 = torch.tensor(0.0, device=device)
    local_records = []

    pbar = loader
    if is_main(rank):
        pbar = tqdm(loader, desc=f"Evaluating {os.path.basename(ckpt_path)}", ncols=120)

    for batch in pbar:
        B = batch["clips_video"].shape[0]

        for b in range(B):
            video_b, idx_b, nums_b, bboxes_b, masks_b, clips_ball_b, clips_ball_mask_b, labels_b, meta_b = \
                slice_video_from_collated_batch(batch, b, device)

            if labels_b.numel() == 0:
                continue

            out = model_ddp.forward(
                clips_video=video_b,
                idx=idx_b,
                nums=nums_b,
                bboxes=bboxes_b,
                bbox_masks=masks_b,
                clips_ball=clips_ball_b,
                clips_ball_mask=clips_ball_mask_b,
                fps_in=float(args.fps_in),
                topk=int(args.topk),
                return_weights=True,
            )

            logits_person = out["logits_person"]  # (N,C)
            if logits_person.numel() == 0:
                continue

            if "gate_weights" in out:
                gate_weights_cpu = out["gate_weights"].detach().cpu()  # (M,N)
            else:
                if args.strict_gate:
                    raise KeyError(
                        "Model output has no 'gate_weights'. Please modify PlayerEventModel.forward_mil_one_video() "
                        "to return gate_weights / gate_logits."
                    )
                gate_weights_cpu = None

            counts, tot = topk_correct_counts(logits_person, labels_b, ks=ks)
            total_people += float(tot)
            correct1 += float(counts[1])
            correct3 += float(counts[3])
            correct5 += float(counts[5])

            N, C = logits_person.shape
            ksave = min(5, int(C))
            topk_idx = torch.topk(logits_person, k=ksave, dim=1, largest=True, sorted=True).indices
            game = meta_b.get("game", "")
            video = meta_b.get("video", "")

            pid = meta_b.get("pids", None)
            if torch.is_tensor(pid):
                pid = pid.detach().cpu().tolist()

            if isinstance(pid, (list, tuple)) and len(pid) == N:
                person_ids = list(pid)
            elif pid is not None:
                person_ids = [pid] * N
            else:
                person_ids = list(range(N))

            labels_cpu = labels_b.detach().cpu().tolist()
            topk_cpu = topk_idx.detach().cpu().tolist()
            clip_ranges_frame, clip_ranges_sec = build_clip_ranges(idx_b, fps_in=float(args.fps_in))

            for i in range(N):
                gt_i = int(labels_cpu[i])
                pred_i = int(topk_cpu[i][0])

                rec = {
                    "game": game,
                    "video": video,
                    "person_id": person_ids[i],
                    "gt": gt_i,
                    "pred_top1": pred_i,
                    "topk_idx": [int(x) for x in topk_cpu[i]],
                    "correct1": bool(gt_i in topk_cpu[i][:min(1, C)]),
                    "correct3": bool(gt_i in topk_cpu[i][:min(3, C)]),
                    "correct5": bool(gt_i in topk_cpu[i][:min(5, C)]),
                    "clip_ranges_frame": clip_ranges_frame,
                    "clip_ranges_sec": clip_ranges_sec,
                }

                if gate_weights_cpu is not None:
                    rec["gate_weights"] = [float(x) for x in gate_weights_cpu[:, i].squeeze(-1).tolist()]

                local_records.append(rec)

        if is_main(rank):
            denom = max(1.0, float(total_people.item()))
            pbar.set_postfix(
                top1=f"{float(correct1.item())/denom:.4f}",
                top3=f"{float(correct3.item())/denom:.4f}",
                top5=f"{float(correct5.item())/denom:.4f}",
            )

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records)

    if is_main(rank):
        all_records = []
        for part in gathered:
            if part:
                all_records.extend(part)

        metrics = compute_metrics(
            records=all_records,
            time_annotations=time_annotations,
            bg_id=args.bg_id,
        )
        print_metrics(ckpt_path, ep, gs, metrics)

    cleanup_ddp()


if __name__ == "__main__":
    main()
