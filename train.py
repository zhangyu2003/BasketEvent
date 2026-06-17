import sys

import os
import time
import argparse
import random
import numpy as np
import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.dataset import VideoBagClipsDataset, bag_collate_fn, LABEL_MAP
from src.model import PlayerEventModel


# =========================================================
# DDP utils
# =========================================================
def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=2),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# save / load ckpt
# =========================================================
def save_ckpt(save_dir: str, epoch: int, global_step: int, model_ddp: DDP, optimizer):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model": model_ddp.module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
    latest_path = os.path.join(save_dir, "latest.pt")
    torch.save(state, path)
    torch.save(state, latest_path)
    print(f"[CKPT] saved: {path}")
    print(f"[CKPT] saved: {latest_path}")


def load_ckpt(ckpt_path: str, model_ddp: DDP, optimizer=None, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_ddp.module.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    return start_epoch, global_step


def broadcast_resume_state(rank: int, device: torch.device, start_epoch: int, global_step: int):
    t = torch.tensor([start_epoch, global_step], device=device, dtype=torch.long)
    dist.broadcast(t, src=0)
    return int(t[0].item()), int(t[1].item())


# =========================================================
# Batch slicing
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
    person_ids_b = batch["person_ids"][b]
    return video_b, idx_b, nums_b, bboxes_b, masks_b, clips_ball_b, clips_ball_mask_b, labels_b, person_ids_b


# =========================================================
# reduce metrics
# =========================================================
@torch.no_grad()
def ddp_all_reduce_mean(value: float, device: torch.device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.item())


# =========================================================
# Main
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--bag_clips", type=int, default=4)
    p.add_argument("--clip_len", type=int, default=8)
    p.add_argument("--fps_in", type=int, default=25)
    p.add_argument("--fps_out", type=int, default=4)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--rebuild_cache", action="store_true")
    p.add_argument("--bbox_dir", type=str, required=True)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, required=True)

    # train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--clip_aux_weight", type=float, default=0.2)
    p.add_argument("--clip_soft_tau", type=float, default=0.5)

    # io
    p.add_argument("--save_dir", type=str, required=True)

    # resume
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_ckpt", type=str, default="")

    return p.parse_args()


def main():
    args = parse_args()

    rank, world_size, local_rank, device = setup_ddp()
    seed_everything(args.seed, rank)

    if is_main_process(rank):
        print(f"[DDP] world_size={world_size} rank={rank} local_rank={local_rank} device={device}")

    # -------------------------
    # dataset / sampler / loader
    # -------------------------

    cache_dir = args.cache_dir
    cache_path = os.path.join(cache_dir, f"bag_clip{args.clip_len}_fps{args.fps_out}_M{args.bag_clips}_train.pkl")

    dataset = VideoBagClipsDataset(
        bbox_dir=args.bbox_dir,
        video_dir=args.video_dir,
        clip_len=args.clip_len,
        fps_in=args.fps_in,
        fps_out=args.fps_out,
        bag_clips=args.bag_clips,
        size=args.img_size,
        cache_path=cache_path,
        rebuild_cache=args.rebuild_cache,
        add_blank=True,
        require_ball=True,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=bag_collate_fn,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # -------------------------
    # model / optimizer / ddp
    # -------------------------
    num_classes = len(LABEL_MAP)
    model = PlayerEventModel(
        num_classes=num_classes,
        roi_out_size=(1, 1),
        use_clip_relation=True,
        use_actor_global=True,
        use_person_relation=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_ddp = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # -------------------------
    # train loop
    # -------------------------
    start_epoch = 1
    global_step = 0

    label_smoothing = 0.1
    bootstrap_alpha = 0.8
    ignore_thresh = 0.8
    blank_class = 0

    if args.resume:
        ckpt_path = args.resume_ckpt if args.resume_ckpt else os.path.join(args.save_dir, "latest.pt")
        if is_main_process(rank):
            if os.path.exists(ckpt_path):
                start_epoch, global_step = load_ckpt(ckpt_path, model_ddp, optimizer, device=device)
                print(f"[Resume] loaded: {ckpt_path} (start_epoch={start_epoch}, global_step={global_step})")
            else:
                print(f"[Resume] ckpt not found: {ckpt_path}, start from scratch")
        start_epoch, global_step = broadcast_resume_state(rank, device, start_epoch, global_step)
    else:
        start_epoch, global_step = broadcast_resume_state(rank, device, start_epoch, global_step)

    class_weight = torch.ones(num_classes, device=device)
    class_weight[0] = 0.1

    for epoch in range(start_epoch, args.epochs + 1):
        model_ddp.train()
        sampler.set_epoch(epoch)

        epoch_loss_sum = 0.0
        epoch_steps = 0
        t0 = time.time()

        pbar = loader
        if is_main_process(rank):
            pbar = tqdm(loader, desc=f"Epoch [{epoch}/{args.epochs}]", ncols=120)

        for batch in pbar:
            B = batch["clips_video"].shape[0]
            optimizer.zero_grad(set_to_none=True)

            batch_loss = 0.0
            valid_videos = 0

            for b in range(B):
                video_b, idx_b, nums_b, bboxes_b, masks_b, clips_ball_b, clips_ball_mask_b, labels_b, person_ids_b = \
                    slice_video_from_collated_batch(batch, b, device)

                if labels_b.numel() == 0:
                    continue

                out = model_ddp(
                    clips_video=video_b,
                    idx=idx_b,
                    nums=nums_b,
                    bboxes=bboxes_b,
                    bbox_masks=masks_b,
                    clips_ball=clips_ball_b,
                    clips_ball_mask=clips_ball_mask_b,
                    fps_in=float(args.fps_in),
                    topk=int(args.topk),
                )

                logits_person = out["logits_person"]
                logits_clip = out["logits_clip"]
                if logits_person.numel() == 0:
                    continue

                probs = torch.softmax(logits_person, dim=-1)
                log_probs = torch.log_softmax(logits_person, dim=-1)

                C = logits_person.shape[1]

                with torch.no_grad():
                    one_hot = F.one_hot(labels_b, num_classes=C).float()
                    smooth_target = one_hot * (1 - label_smoothing) + label_smoothing / C

                with torch.no_grad():
                    bootstrap_target = bootstrap_alpha * smooth_target + (1 - bootstrap_alpha) * probs

                with torch.no_grad():
                    max_prob, pred = probs.max(dim=-1)
                    ignore_mask = torch.ones_like(labels_b, dtype=torch.bool)
                    blank_mask = (labels_b == blank_class)
                    high_conf_non_blank = (max_prob > ignore_thresh) & (pred != blank_class)
                    ignore_mask[blank_mask & high_conf_non_blank] = False

                sample_weight = class_weight[labels_b]
                loss_vec = -(bootstrap_target * log_probs).sum(dim=-1)
                loss_vec = loss_vec * sample_weight
                loss_vec = loss_vec[ignore_mask]

                if loss_vec.numel() == 0:
                    continue

                loss_person = loss_vec.mean()

                log_probs_clip = torch.log_softmax(logits_clip, dim=-1)
                probs_clip = torch.softmax(logits_clip, dim=-1)

                pos_mask = (labels_b != blank_class)
                neg_mask = (labels_b == blank_class)
                clip_loss_terms = []

                if pos_mask.any():
                    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
                    pos_labels = labels_b[pos_idx]

                    pos_probs_clip = probs_clip[:, pos_idx, :]
                    pos_log_probs_clip = log_probs_clip[:, pos_idx, :]

                    with torch.no_grad():
                        gt_scores = pos_probs_clip.gather(
                            dim=-1,
                            index=pos_labels.view(1, -1, 1).expand(logits_clip.shape[0], -1, 1)
                        ).squeeze(-1)
                        clip_weights = torch.softmax(gt_scores / args.clip_soft_tau, dim=0)
                        pos_one_hot = F.one_hot(pos_labels, num_classes=C).float()
                        pos_one_hot = pos_one_hot.unsqueeze(0).expand(logits_clip.shape[0], -1, -1)
                        pos_smooth_target = pos_one_hot * (1 - label_smoothing) + label_smoothing / C
                        pos_bootstrap_target = bootstrap_alpha * pos_smooth_target + (1 - bootstrap_alpha) * pos_probs_clip

                    pos_loss = -(pos_bootstrap_target * pos_log_probs_clip).sum(dim=-1)
                    pos_loss = pos_loss * class_weight[pos_labels].unsqueeze(0)
                    pos_loss = pos_loss * clip_weights
                    clip_loss_terms.append(pos_loss.sum(dim=0).mean())

                if neg_mask.any():
                    neg_idx = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
                    neg_probs_clip = probs_clip[:, neg_idx, :]
                    neg_log_probs_clip = log_probs_clip[:, neg_idx, :]

                    with torch.no_grad():
                        blank_targets = torch.full(
                            (logits_clip.shape[0], neg_idx.numel()),
                            blank_class,
                            device=device,
                            dtype=torch.long,
                        )
                        neg_one_hot = F.one_hot(blank_targets, num_classes=C).float()
                        neg_smooth_target = neg_one_hot * (1 - label_smoothing) + label_smoothing / C
                        neg_bootstrap_target = bootstrap_alpha * neg_smooth_target + (1 - bootstrap_alpha) * neg_probs_clip

                    neg_loss = -(neg_bootstrap_target * neg_log_probs_clip).sum(dim=-1)
                    neg_loss = neg_loss * class_weight[blank_class]
                    clip_loss_terms.append(neg_loss.mean())

                if len(clip_loss_terms) > 0:
                    loss_clip = torch.stack(clip_loss_terms).mean()
                else:
                    loss_clip = torch.zeros((), device=device)

                loss_b = loss_person + args.clip_aux_weight * loss_clip
                batch_loss += loss_b
                valid_videos += 1

            if valid_videos == 0:
                batch_loss = torch.zeros((), device=device, requires_grad=True)
            else:
                batch_loss = batch_loss / valid_videos

            batch_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), args.grad_clip)

            optimizer.step()

            global_step += 1
            epoch_loss_sum += float(batch_loss.item())
            epoch_steps += 1

            if is_main_process(rank):
                M = batch["clips_video"].shape[1]
                pbar.set_postfix(loss=f"{epoch_loss_sum / max(1,epoch_steps):.4f}", B=B, M=M, topk=args.topk)

        local_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        epoch_loss_mean = ddp_all_reduce_mean(local_epoch_loss, device)

        if is_main_process(rank):
            print(f"[Epoch {epoch}] loss={epoch_loss_mean:.6f} time={time.time()-t0:.1f}s")
            save_ckpt(args.save_dir, epoch, global_step, model_ddp, optimizer)

    cleanup_ddp()
    if is_main_process(rank):
        print("[Done]")


if __name__ == "__main__":
    main()
