import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from typing import Optional, Dict, Any, Tuple, List


# =========================================================
# 1) BBox embedding
# =========================================================
class BBoxEmbedding(nn.Module):
    """
    bbox 序列 (N,T,4) + mask (N,T) -> (N,T,C)
    输入 bbox: xyxy in [0, image_size]
    特征包含：
      - 几何位置: x1,y1,x2,y2,w,h,cx,cy,area,aspect_ratio
      - 运动信息: dx,dy,dw,dh
    """
    def __init__(
        self,
        out_dim: int,
        image_size: int = 224,
        use_motion: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = float(image_size)
        self.use_motion = use_motion

        in_dim = 10 + (4 if use_motion else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, bboxes_xyxy: torch.Tensor, bbox_mask: torch.Tensor) -> torch.Tensor:
        """
        bboxes_xyxy: (N,T,4) float
        bbox_mask:   (N,T) float/bool, 1=valid
        return:      (N,T,out_dim)
        """
        b = bboxes_xyxy.to(torch.float32)
        b = torch.clamp(b, 0.0, self.image_size)

        x1, y1, x2, y2 = b.unbind(dim=-1)
        w = (x2 - x1).clamp_min(1.0)
        h = (y2 - y1).clamp_min(1.0)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        area = (w * h) / (self.image_size * self.image_size)
        ar = (w / h).clamp(0.0, 10.0)

        s = self.image_size
        x1n, y1n, x2n, y2n = x1 / s, y1 / s, x2 / s, y2 / s
        wn, hn = w / s, h / s
        cxn, cyn = cx / s, cy / s

        feats = [x1n, y1n, x2n, y2n, wn, hn, cxn, cyn, area, ar]

        if self.use_motion:
            dx = torch.zeros_like(cxn)
            dy = torch.zeros_like(cyn)
            dw = torch.zeros_like(wn)
            dh = torch.zeros_like(hn)
            dx[:, 1:] = cxn[:, 1:] - cxn[:, :-1]
            dy[:, 1:] = cyn[:, 1:] - cyn[:, :-1]
            dw[:, 1:] = wn[:, 1:] - wn[:, :-1]
            dh[:, 1:] = hn[:, 1:] - hn[:, :-1]
            feats += [dx, dy, dw, dh]

        feat = torch.stack(feats, dim=-1)  # (N,T,in_dim)
        m = bbox_mask.to(dtype=feat.dtype).unsqueeze(-1)
        feat = feat * m

        emb = self.mlp(feat)
        emb = emb * m
        return emb


# =========================================================
# 2) Time scale embedding
# =========================================================
class TimeScaleEmbedding(nn.Module):
    """
    idx + nums -> (B,T,D)
    feats: t_norm, t_sec, dt_sec
    """
    def __init__(self, d_model: int, use_dt: bool = True):
        super().__init__()
        self.use_dt = use_dt
        in_dim = 2 + (1 if use_dt else 0)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, idx: torch.Tensor, nums: torch.Tensor, fps: float = 25.0) -> torch.Tensor:
        """
        idx:  (B,T)
        nums: (B,)
        return: (B,T,D)
        """
        idx_f = idx.to(torch.float32)
        nums_f = nums.to(idx.device).to(torch.float32).clamp_min(2.0)

        t_norm = idx_f / (nums_f[:, None] - 1.0)
        t_sec = idx_f / float(fps)

        feats = [t_norm, t_sec]
        if self.use_dt:
            dt = torch.zeros_like(t_sec)
            dt[:, 1:] = t_sec[:, 1:] - t_sec[:, :-1]
            feats.append(dt)

        feat = torch.stack(feats, dim=-1)
        return self.mlp(feat)


# =========================================================
# 3) TimeSformer backbone
# =========================================================
class TimeSformerBackbone(nn.Module):
    """
    输入 video: (B,C,T,H,W) 或 (B,T,C,H,W)
    输出 featmap: (B,D,T,H',W')

    注意：HF TimeSformer 默认输出 token 序列。
    这里去掉 CLS token 后 reshape 成时空 feature map。
    """
    def __init__(
        self,
        pretrained_name: str = "facebook/timesformer-base-finetuned-k400",
        use_pretrained: bool = False,
    ):
        super().__init__()
        try:
            from transformers import TimesformerModel, TimesformerConfig
        except Exception as e:
            raise ImportError("请先 pip install transformers，且版本需包含 TimesformerModel。原始错误: " + str(e))

        if use_pretrained:
            self.model = TimesformerModel.from_pretrained(pretrained_name)
        else:
            # 如果 pretrained_name 是本地 config 或模型名，这里仍可用 config 初始化。
            cfg = TimesformerConfig.from_pretrained(pretrained_name)
            self.model = TimesformerModel(cfg)

        self.hidden_dim = self.model.config.hidden_size

        image_size = getattr(self.model.config, "image_size", 224)
        patch_size = getattr(self.model.config, "patch_size", 16)
        self.grid_h = image_size // patch_size
        self.grid_w = image_size // patch_size


    @staticmethod
    def _ensure_b_t_c_h_w(video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 5:
            raise ValueError(f"video must be 5D, got {tuple(video.shape)}")
        # Accept both (B,C,T,H,W) and (B,T,C,H,W).
        # Prefer interpreting ambiguous small-T inputs as channel-first because the
        # training pipeline in this repo constructs clips as (B,C,T,H,W).
        if video.shape[1] in (1, 3):
            return video.permute(0, 2, 1, 3, 4).contiguous()
        return video

    def forward(
        self,
        video: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        nums: Optional[torch.Tensor] = None,
        fps_in: float = 25.0,
    ) -> Dict[str, Any]:
        x = self._ensure_b_t_c_h_w(video)  # (B,T,C,H,W)
        B, T, C, H, W = x.shape

        # HF TimeSformer layers read config.num_frames during forward, so keep it
        # in sync with the current clip length. Its embedding table is resized
        # internally when T differs from the pretrained checkpoint setting.
        self.model.config.num_frames = T
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state  # (B,1+T*P,D)
        tokens = tokens[:, 1:, :]       # (B,T*P,D)

        P = self.grid_h * self.grid_w
        if tokens.shape[1] != T * P:
            raise ValueError(
                f"Token length mismatch: got {tokens.shape[1]}, expected {T}*{P}={T*P}. "
                f"Check image_size/patch_size or input size."
            )

        fmap = tokens.view(B, T, P, self.hidden_dim).view(
            B, T, self.grid_h, self.grid_w, self.hidden_dim
        )  # (B,T,H',W',D)


        fmap = fmap.permute(0, 4, 1, 2, 3).contiguous()  # (B,D,T,H',W')

        return {
            "featmap": fmap,
            "aux": {
                "hidden_dim": self.hidden_dim,
                "grid_h": self.grid_h,
                "grid_w": self.grid_w,
            },
        }


# =========================================================
# 4) Temporal ROIAlign
# =========================================================
class TemporalROIAlign(nn.Module):
    """
    对每一帧 feature map 做 ROIAlign。

    Inputs:
      featmap:    (B, C, T, Hf, Wf)
      bboxes:     list length B; each (Ni, T, 4) xyxy@image_size
      bbox_mask:  list length B; each (Ni, T) 1/0

    Outputs:
      person_feats:  (sum Ni, T, C) when output_size=(1,1)
                     (sum Ni, T, C*oh*ow) when output_size != (1,1)
      person_mask:   (sum Ni, T)
      person_splits: list length B, each = Ni
    """
    def __init__(
        self,
        image_size: int = 224,
        output_size: Tuple[int, int] = (1, 1),
        aligned: bool = True,
    ):
        super().__init__()
        self.image_size = float(image_size)
        self.output_size = output_size
        self.aligned = aligned

    def forward(
        self,
        featmap: torch.Tensor,
        bboxes: List[torch.Tensor],
        bbox_mask: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        device = featmap.device
        dtype = featmap.dtype

        if featmap.dim() != 5:
            raise ValueError(f"featmap must be (B,C,T,Hf,Wf), got {tuple(featmap.shape)}")

        B, C, T, Hf, Wf = featmap.shape
        if len(bboxes) != B or len(bbox_mask) != B:
            raise ValueError(f"Length of bboxes/mask list must equal B={B}, got {len(bboxes)} and {len(bbox_mask)}")

        out_h, out_w = self.output_size
        pooled_dim = C * out_h * out_w

        person_splits = [int(bb.shape[0]) for bb in bboxes]
        total_persons = sum(person_splits)

        if total_persons == 0:
            return (
                torch.zeros((0, T, pooled_dim), device=device, dtype=dtype),
                torch.zeros((0, T), device=device, dtype=torch.float32),
                person_splits,
            )

        person_feats = torch.zeros((total_persons, T, pooled_dim), device=device, dtype=dtype)
        person_mask = torch.zeros((total_persons, T), device=device, dtype=torch.float32)

        feat_2d = featmap.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, Hf, Wf)

        sx = float(Wf) / self.image_size
        sy = float(Hf) / self.image_size

        rois = []
        roi_person_index = []
        roi_t_index = []

        cursor = 0
        for b in range(B):
            bb = bboxes[b].to(device=device, dtype=torch.float32)
            mk = bbox_mask[b].to(device=device, dtype=torch.float32)
            Ni = int(bb.shape[0])
            if Ni == 0:
                continue

            person_mask[cursor:cursor + Ni] = mk

            for pi in range(Ni):
                for t in range(T):
                    if mk[pi, t].item() <= 0:
                        continue
                    x1, y1, x2, y2 = bb[pi, t].tolist()

                    x1 *= sx
                    x2 *= sx
                    y1 *= sy
                    y2 *= sy

                    x1 = max(0.0, min(x1, Wf - 1.0))
                    x2 = max(0.0, min(x2, Wf - 1.0))
                    y1 = max(0.0, min(y1, Hf - 1.0))
                    y2 = max(0.0, min(y2, Hf - 1.0))

                    if x2 <= x1:
                        x2 = min(Wf - 1.0, x1 + 1.0)
                    if y2 <= y1:
                        y2 = min(Hf - 1.0, y1 + 1.0)

                    frame_index = b * T + t
                    rois.append([frame_index, x1, y1, x2, y2])
                    roi_person_index.append(cursor + pi)
                    roi_t_index.append(t)

            cursor += Ni

        if len(rois) == 0:
            return person_feats, person_mask, person_splits

        rois = torch.tensor(rois, device=device, dtype=torch.float32)
        pooled = roi_align(
            input=feat_2d,
            boxes=rois,
            output_size=self.output_size,
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=self.aligned,
        )  # (R,C,oh,ow)

        pooled = pooled.flatten(1)  # (R,C*oh*ow)

        roi_person_index = torch.tensor(roi_person_index, device=device, dtype=torch.long)
        roi_t_index = torch.tensor(roi_t_index, device=device, dtype=torch.long)
        person_feats[roi_person_index, roi_t_index] = pooled.to(dtype)

        return person_feats, person_mask, person_splits


# =========================================================
# 5) Utility blocks
# =========================================================
class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TypeEmbedding(nn.Module):
    """
    给 person / ball 加不同类型 embedding。
    type_id: 0=person, 1=ball
    """
    def __init__(self, dim: int, num_types: int = 2):
        super().__init__()
        self.emb = nn.Embedding(num_types, dim)

    def forward(self, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """
        x:        (..., C)
        type_ids: broadcastable to x.shape[:-1]
        """
        return x + self.emb(type_ids.to(x.device).long())


# =========================================================
# 6) Actor-Global Interaction
# =========================================================
class ActorGlobalCrossAttention(nn.Module):
    """
    第 1 阶段：每个球员 token 与对应帧的全局 feature map 交互。

    输入：
      person_feats: (M, N, T, C)
      featmap:      (M, C, T, Hf, Wf)
      person_mask:  (M, N, T), 1=valid

    输出：
      enhanced person_feats: (M, N, T, C)

    实现逻辑：
      对每个 (m,t)，有 N 个 person queries；
      对应全局 feature map F[m,:,t,:,:] 展成 Hf*Wf 个 global tokens；
      Q = person token, K/V = global tokens。
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        add_spatial_pos: bool = True,
        grid_h: int = 14,
        grid_w: int = 14,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.add_spatial_pos = add_spatial_pos
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

        if add_spatial_pos:
            self.spatial_pos = nn.Parameter(torch.zeros(1, grid_h * grid_w, dim))
            nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(
        self,
        person_feats: torch.Tensor,
        featmap: torch.Tensor,
        person_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        if person_feats.dim() != 4:
            raise ValueError(f"person_feats must be (M,N,T,C), got {tuple(person_feats.shape)}")
        if featmap.dim() != 5:
            raise ValueError(f"featmap must be (M,C,T,H,W), got {tuple(featmap.shape)}")

        M, N, T, C = person_feats.shape
        Mf, Cf, Tf, Hf, Wf = featmap.shape
        if (M, T, C) != (Mf, Tf, Cf):
            raise ValueError(
                f"Shape mismatch: person_feats={(M,N,T,C)}, featmap={(Mf,Cf,Tf,Hf,Wf)}"
            )

        # global tokens: (M,T,H*W,C) -> (M*T, H*W, C)
        global_tokens = featmap.permute(0, 2, 3, 4, 1).contiguous().view(M * T, Hf * Wf, C)

        if self.add_spatial_pos:
            if Hf == self.grid_h and Wf == self.grid_w:
                pos = self.spatial_pos
            else:
                # 若输入尺寸改变，插值 spatial pos。
                pos_2d = self.spatial_pos.view(1, self.grid_h, self.grid_w, C).permute(0, 3, 1, 2)
                pos_2d = F.interpolate(pos_2d, size=(Hf, Wf), mode="bilinear", align_corners=False)
                pos = pos_2d.permute(0, 2, 3, 1).contiguous().view(1, Hf * Wf, C)
            global_tokens = global_tokens + pos.to(global_tokens.dtype)

        # queries: (M,T,N,C) -> (M*T,N,C)
        q = person_feats.permute(0, 2, 1, 3).contiguous().view(M * T, N, C)

        q_norm = self.q_norm(q)
        kv_norm = self.kv_norm(global_tokens)

        attn_out, attn_weights = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            need_weights=return_attn,
        )

        out = q + self.dropout(attn_out)
        out = self.out_norm(out)
        out = self.ffn(out)

        out = out.view(M, T, N, C).permute(0, 2, 1, 3).contiguous()  # (M,N,T,C)

        if person_mask is not None:
            out = out * person_mask.to(dtype=out.dtype).unsqueeze(-1)

        if return_attn:
            # attn_weights: (M*T, N, Hf*Wf) -> reshape to (M, T, N, Hf, Wf)
            attn_weights_reshaped = attn_weights.view(M, T, N, Hf, Wf)
            return out, attn_weights_reshaped
        return out


# =========================================================
# 7) Temporal pooler inside each clip
# =========================================================
class PersonFeaturePooler(nn.Module):
    """
    对每个 person 在一个 clip 内的 T 帧特征做 temporal attention pooling。
    输入:  (N,T,C)
    输出:  (N,C)
    """
    def __init__(self, in_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(in_dim)
        self.ffn = MLPBlock(in_dim, mlp_ratio=4.0, dropout=dropout)
        self.out_norm = nn.LayerNorm(in_dim)

    def forward(self, person_feats: torch.Tensor, person_mask: torch.Tensor) -> torch.Tensor:
        """
        person_feats: (N,T,C)
        person_mask:  (N,T)
        return:       (N,C)
        """
        if person_feats.numel() == 0:
            return torch.zeros((0, person_feats.shape[-1]), device=person_feats.device, dtype=person_feats.dtype)

        n, _, c = person_feats.shape
        valid_mask = person_mask > 0

        cls_token = self.cls_token.expand(n, -1, -1)
        temporal_tokens = torch.cat([cls_token, person_feats], dim=1)  # (N,1+T,C)

        cls_valid = torch.ones((n, 1), dtype=torch.bool, device=person_feats.device)
        attn_valid_mask = torch.cat([cls_valid, valid_mask], dim=1)
        key_padding_mask = ~attn_valid_mask

        attn_out, _ = self.temporal_attn(
            temporal_tokens,
            temporal_tokens,
            temporal_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        temporal_tokens = self.temporal_norm(temporal_tokens + attn_out)
        temporal_tokens = self.ffn(temporal_tokens)

        pooled = self.out_norm(temporal_tokens[:, 0, :])
        return pooled


# =========================================================
# 8) Person-Person Interaction within each clip
# =========================================================
class PersonRelationBlock(nn.Module):
    """
    第 2 阶段：同一 clip 内，球员之间、球员与 ball token 之间做 self-attention。

    输入：
      x:          (M,N,C)
      valid_mask: (M,N), 1=valid
    输出：
      x:          (M,N,C)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.numel() == 0:
            return x

        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~(valid_mask > 0)
            # 防止某一个 clip 内所有 token 都无效导致 attention NaN。
            all_invalid = key_padding_mask.all(dim=1)
            if all_invalid.any():
                key_padding_mask[all_invalid, 0] = False

        h = self.norm1(x)
        ctx, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm2(x + self.dropout(ctx))
        x = self.ffn(x)

        if valid_mask is not None:
            x = x * valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        return x


# =========================================================
# 9) Clip-Clip Interaction across clips for each player
# =========================================================
class ClipRelationBlock(nn.Module):
    """
    第 3 阶段：同一个球员/ball 在不同 clip 之间做 self-attention。

    输入：
      x:          (M,N,C)
      valid_mask: (M,N), 1=该 clip 内该 token 有效
    输出：
      x:          (M,N,C)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MLPBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.numel() == 0:
            return x

        # (M,N,C) -> (N,M,C): 对每个球员沿 clip 维度交互
        x_in = x.permute(1, 0, 2).contiguous()

        key_padding_mask = None
        if valid_mask is not None:
            mask_in = valid_mask.permute(1, 0).contiguous()  # (N,M)
            key_padding_mask = ~(mask_in > 0)
            all_invalid = key_padding_mask.all(dim=1)
            if all_invalid.any():
                key_padding_mask[all_invalid, 0] = False
        else:
            mask_in = None

        h = self.norm1(x_in)
        ctx, _ = self.attn(
            h,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x_out = self.norm2(x_in + self.dropout(ctx))
        x_out = self.ffn(x_out)

        if mask_in is not None:
            x_out = x_out * mask_in.to(dtype=x_out.dtype).unsqueeze(-1)

        return x_out.permute(1, 0, 2).contiguous()


# =========================================================
# 10) Heads and pooling
# =========================================================
class PersonEventClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, person_feats: torch.Tensor, person_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        person_feats = self.norm(person_feats)
        logits = self.mlp(person_feats)
        if person_mask is not None:
            person_mask = person_mask.to(dtype=logits.dtype).unsqueeze(-1)
            logits = logits * person_mask
        return logits


class GatedClipPooling(nn.Module):
    """
    对每个 person 在 M 个 clip 上做 gated pooling。

    输入：
      x:          (M,N,C)
      valid_mask: (M,N), 1=valid
    输出：
      pooled:     (N,C)
      gate_logits:(M,N)
      gate_weights:(M,N,1)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ):
        gate_logits = self.gate(x).squeeze(-1)  # (M,N)

        if valid_mask is not None:
            mask = valid_mask > 0
            # 对无效 clip 置为极小值，避免参与 softmax。
            gate_logits = gate_logits.masked_fill(~mask, -1e4)
            # 若某 person 所有 clip 都无效，给第一个 clip 一个安全位置，避免 NaN。
            all_invalid = (~mask).all(dim=0)
            if all_invalid.any():
                gate_logits[:, all_invalid] = -1e4
                gate_logits[0, all_invalid] = 0.0

        gate_weights = torch.softmax(gate_logits, dim=0).unsqueeze(-1)  # (M,N,1)
        pooled = torch.sum(x * gate_weights, dim=0)  # (N,C)

        if return_weights:
            return pooled, gate_logits, gate_weights
        return pooled


class TopKClipPooling(nn.Module):
    """
    备用：对每个 person 在 clip 维度做 top-k pooling。
    注意：top-k 根据每个 clip 的 gate score 选 clip，再平均特征。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        topk: int = 2,
        return_weights: bool = False,
    ):
        M, N, C = x.shape
        scores = self.score(x).squeeze(-1)  # (M,N)
        if valid_mask is not None:
            scores = scores.masked_fill(~(valid_mask > 0), -1e4)

        k = max(1, min(int(topk), M))
        topk_scores, topk_idx = torch.topk(scores, k=k, dim=0)  # (k,N)

        gather_idx = topk_idx.unsqueeze(-1).expand(k, N, C)
        selected = torch.gather(x, dim=0, index=gather_idx)  # (k,N,C)
        pooled = selected.mean(dim=0)  # (N,C)

        if return_weights:
            # 构造一个稀疏 weights，方便和 gated pooling 输出接口统一。
            weights = torch.zeros((M, N, 1), device=x.device, dtype=x.dtype)
            weights.scatter_(0, topk_idx.unsqueeze(-1), 1.0 / float(k))
            return pooled, scores, weights
        return pooled


# =========================================================
# 11) PlayerEventModel: G -> P -> C -> Pooling
# =========================================================
class PlayerEventModel(nn.Module):
    """
    重构版：
      1. 每个球员 / ball token 与全局特征图交互：ActorGlobalCrossAttention
      2. 同一 clip 内球员间交互：PersonRelationBlock
      3. 同一球员跨 clip 交互：ClipRelationBlock
      4. clip-level gated/top-k pooling
      5. person-level classification

    Ball 被加入为 virtual token，参与交互，但最终不分类。
    """
    def __init__(
        self,
        num_classes: int,
        pretrained_name: str = "/GPFS/rhome/yuzhang/.cache/huggingface/hub/models--facebook--timesformer-base-finetuned-k400/snapshots/f300f6ac53f51b74e7f691877142ac426ce800ad",
        roi_out_size: Tuple[int, int] = (1, 1),
        roi_out_dim: Optional[int] = None,
        image_size: int = 224,
        add_bbox_embedding: bool = True,
        add_type_embedding: bool = True,
        mil_attn_heads: int = 8,
        mil_attn_dropout: float = 0.1,
        use_actor_global: bool = True,
        use_person_relation: bool = True,
        use_clip_relation: bool = True,
        pooling_mode: str = "gated",  # "gated" or "topk"
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.add_bbox_embedding = bool(add_bbox_embedding)
        self.add_type_embedding = bool(add_type_embedding)
        self.use_actor_global = bool(use_actor_global)
        self.use_person_relation = bool(use_person_relation)
        self.use_clip_relation = bool(use_clip_relation)
        self.pooling_mode = str(pooling_mode)

        self.backbone = TimeSformerBackbone(
            pretrained_name=pretrained_name,
            use_pretrained=True,
        )

        C = self.backbone.hidden_dim
        self.backbone_dim = C

        self.roi = TemporalROIAlign(
            image_size=image_size,
            output_size=roi_out_size,
            aligned=True,
        )

        roi_flat_dim = C * roi_out_size[0] * roi_out_size[1]
        self.mil_feat_dim = roi_out_dim if roi_out_dim is not None else C

        # 如果 ROIAlign 不是 1x1，先投影回 mil_feat_dim。
        if roi_flat_dim != self.mil_feat_dim:
            self.roi_proj = nn.Sequential(
                nn.LayerNorm(roi_flat_dim),
                nn.Linear(roi_flat_dim, self.mil_feat_dim),
            )
        else:
            self.roi_proj = nn.Identity()

        if self.add_bbox_embedding:
            self.bbox_emb = BBoxEmbedding(
                out_dim=self.mil_feat_dim,
                image_size=image_size,
                use_motion=True,
                hidden_dim=256,
                dropout=0.1,
            )

        if self.add_type_embedding:
            self.type_emb = TypeEmbedding(dim=self.mil_feat_dim, num_types=2)

        if self.use_actor_global:
            # 若 mil_feat_dim 与 backbone_dim 不同，需要把 featmap 投影到 mil_feat_dim。
            if C != self.mil_feat_dim:
                self.featmap_proj = nn.Conv3d(C, self.mil_feat_dim, kernel_size=1, bias=False)
            else:
                self.featmap_proj = nn.Identity()

            self.actor_global = ActorGlobalCrossAttention(
                dim=self.mil_feat_dim,
                num_heads=mil_attn_heads,
                dropout=mil_attn_dropout,
                add_spatial_pos=True,
                grid_h=self.backbone.grid_h,
                grid_w=self.backbone.grid_w,
                mlp_ratio=4.0,
            )
        else:
            self.featmap_proj = nn.Identity()
            self.actor_global = None

        self.feature_pooler = PersonFeaturePooler(
            in_dim=self.mil_feat_dim,
            num_heads=mil_attn_heads,
            dropout=mil_attn_dropout,
        )

        if self.use_person_relation:
            self.person_relation = PersonRelationBlock(
                dim=self.mil_feat_dim,
                num_heads=mil_attn_heads,
                dropout=mil_attn_dropout,
                mlp_ratio=4.0,
            )
        else:
            self.person_relation = None

        if self.use_clip_relation:
            self.clip_relation = ClipRelationBlock(
                dim=self.mil_feat_dim,
                num_heads=mil_attn_heads,
                dropout=mil_attn_dropout,
                mlp_ratio=4.0,
            )
        else:
            self.clip_relation = None

        if self.pooling_mode == "gated":
            self.clip_pool = GatedClipPooling(self.mil_feat_dim)
        elif self.pooling_mode == "topk":
            self.clip_pool = TopKClipPooling(self.mil_feat_dim)
        else:
            raise ValueError(f"pooling_mode must be 'gated' or 'topk', got {pooling_mode}")

        self.clip_head = PersonEventClassifierHead(
            in_dim=self.mil_feat_dim,
            num_classes=num_classes,
            dropout=0.1,
            hidden_dim=512,
        )
        self.person_head = PersonEventClassifierHead(
            in_dim=self.mil_feat_dim,
            num_classes=num_classes,
            dropout=0.1,
            hidden_dim=512,
        )

    @torch.no_grad()
    def _assert_aligned_person_count(self, bboxes: List[torch.Tensor]) -> int:
        if len(bboxes) == 0:
            return 0
        N = int(bboxes[0].shape[0])
        for bb in bboxes:
            if int(bb.shape[0]) != N:
                raise RuntimeError(
                    f"MIL requires aligned persons across clips, but got person counts "
                    f"{[int(x.shape[0]) for x in bboxes]}"
                )
        return N

    @staticmethod
    def _build_extended_boxes_with_ball(
        bboxes: List[torch.Tensor],
        bbox_masks: List[torch.Tensor],
        clips_ball: torch.Tensor,
        clips_ball_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        M = len(bboxes)
        extended_bboxes = []
        extended_bbox_masks = []
        for m in range(M):
            person_bb = bboxes[m]
            person_mask = bbox_masks[m]

            ball_bb = clips_ball[m:m + 1, :, :].to(device=person_bb.device, dtype=person_bb.dtype)
            ball_mask = clips_ball_mask[m:m + 1, :].to(device=person_mask.device, dtype=person_mask.dtype)

            extended_bboxes.append(torch.cat([person_bb, ball_bb], dim=0))
            extended_bbox_masks.append(torch.cat([person_mask, ball_mask], dim=0))

        return extended_bboxes, extended_bbox_masks

    def _add_bbox_and_type_embedding(
        self,
        person_feats_flat: torch.Tensor,
        extended_bboxes: List[torch.Tensor],
        extended_bbox_masks: List[torch.Tensor],
        M: int,
        N_plus_1: int,
        T: int,
    ) -> torch.Tensor:
        """
        person_feats_flat: (M*(N+1),T,C)
        return:            (M*(N+1),T,C)
        """
        device = person_feats_flat.device

        if self.add_bbox_embedding and person_feats_flat.numel() > 0:
            bbox_cat = torch.cat(
                [bb.to(device=device, dtype=torch.float32) for bb in extended_bboxes],
                dim=0,
            )  # (M*(N+1),T,4)
            mask_cat = torch.cat(
                [mk.to(device=device, dtype=torch.float32) for mk in extended_bbox_masks],
                dim=0,
            )  # (M*(N+1),T)
            bbox_feat = self.bbox_emb(bbox_cat, mask_cat)
            person_feats_flat = person_feats_flat + bbox_feat

        if self.add_type_embedding and person_feats_flat.numel() > 0:
            # 对每个 clip: 前 N 个是 person，最后 1 个是 ball。
            type_ids_one_clip = torch.zeros((N_plus_1,), device=device, dtype=torch.long)
            type_ids_one_clip[-1] = 1
            type_ids = type_ids_one_clip.repeat(M)  # (M*(N+1),)
            type_ids = type_ids[:, None].expand(M * N_plus_1, T)  # (M*(N+1),T)
            person_feats_flat = self.type_emb(person_feats_flat, type_ids)

        return person_feats_flat

    def extract_roi_debug_features(
        self,
        clips_video: torch.Tensor,
        idx: torch.Tensor,
        nums: torch.Tensor,
        bboxes: List[torch.Tensor],
        bbox_masks: List[torch.Tensor],
        fps_in: float = 25.0,
    ) -> Dict[str, torch.Tensor]:
        """
        调试/可视化接口：
          - 直接从模型内部拿 backbone 全局特征图
          - 直接拿 ROIAlign 输出的人物帧级特征

        返回：
          featmap:               (M,C,T,Hf,Wf) backbone 原始输出
          featmap_for_global:    (M,C',T,Hf,Wf) 若存在 1x1 conv 投影，则为投影后的全局特征
          person_feats_raw:      (M,N,T,roi_flat_dim) ROIAlign 原始输出
          person_feats_projected:(M,N,T,mil_feat_dim) 经过 roi_proj 的人物特征
          person_mask:           (M,N,T)
        """
        device = clips_video.device
        M = int(clips_video.shape[0])
        N = self._assert_aligned_person_count(bboxes)

        backbone_out = self.backbone(
            clips_video,
            idx=idx,
            nums=nums,
            fps_in=fps_in,
        )
        featmap = backbone_out["featmap"]  # (M,C,T,Hf,Wf)

        person_feats_flat, person_mask_flat, _ = self.roi(
            featmap,
            bboxes,
            bbox_masks,
        )

        _, _, T, _, _ = featmap.shape
        roi_flat_dim = int(person_feats_flat.shape[-1]) if person_feats_flat.numel() > 0 else self.backbone_dim
        person_feats_raw = person_feats_flat.view(M, N, T, roi_flat_dim)
        person_mask = person_mask_flat.view(M, N, T)

        person_feats_projected = self.roi_proj(person_feats_flat)
        person_feats_projected = person_feats_projected.view(M, N, T, self.mil_feat_dim)

        featmap_for_global = self.featmap_proj(featmap) if self.use_actor_global else featmap

        return {
            "featmap": featmap,
            "featmap_for_global": featmap_for_global,
            "person_feats_raw": person_feats_raw,
            "person_feats_projected": person_feats_projected,
            "person_mask": person_mask,
            "num_persons": torch.tensor([N], device=device, dtype=torch.long),
        }

    def forward_mil_one_video(
        self,
        clips_video: torch.Tensor,         # (M,C,T,H,W)
        idx: torch.Tensor,                 # (M,T)
        nums: torch.Tensor,                # (M,)
        bboxes: List[torch.Tensor],        # list length M; each (N,T,4)
        bbox_masks: List[torch.Tensor],    # list length M; each (N,T)
        clips_ball: torch.Tensor,          # (M,T,4)
        clips_ball_mask: torch.Tensor,     # (M,T)
        fps_in: float = 25.0,
        topk: int = 2,
        return_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = clips_video.device
        M = int(clips_video.shape[0])

        if M == 0:
            return {
                "pooled_feats_flat": torch.zeros((0, self.mil_feat_dim), device=device),
                "pooled_feats_clip": torch.zeros((0, 0, self.mil_feat_dim), device=device),
                "pooled_feats_person": torch.zeros((0, self.mil_feat_dim), device=device),
                "logits_person": torch.zeros((0, self.num_classes), device=device),
                "logits_clip": torch.zeros((0, 0, self.num_classes), device=device),
            }

        N = self._assert_aligned_person_count(bboxes)
        N_plus_1 = N + 1

        # person_valid: (N,), 某个球员只要任一 clip/frame 有 bbox 即有效。
        if N > 0:
            person_valid = torch.stack(
                [mask.to(device=device).any(dim=1) for mask in bbox_masks], dim=0
            ).any(dim=0)  # (N,)
        else:
            person_valid = torch.zeros((0,), device=device, dtype=torch.bool)

        # 加 ball token。
        extended_bboxes, extended_bbox_masks = self._build_extended_boxes_with_ball(
            bboxes=bboxes,
            bbox_masks=bbox_masks,
            clips_ball=clips_ball,
            clips_ball_mask=clips_ball_mask,
        )

        # 1. backbone global featmap。
        backbone_out = self.backbone(
            clips_video,
            idx=idx,
            nums=nums,
            fps_in=fps_in,
        )
        featmap = backbone_out["featmap"]  # (M,C,T,Hf,Wf)
        _, _, T, _, _ = featmap.shape

        # 2. ROIAlign 得到每个 token 的帧级特征。
        person_feats_flat, person_mask_flat, person_splits = self.roi(
            featmap,
            extended_bboxes,
            extended_bbox_masks,
        )  # (M*(N+1),T,roi_dim), (M*(N+1),T)

        if N == 0:
            return {
                "pooled_feats_flat": torch.zeros((0, self.mil_feat_dim), device=device),
                "pooled_feats_clip": torch.zeros((M, 0, self.mil_feat_dim), device=device),
                "pooled_feats_person": torch.zeros((0, self.mil_feat_dim), device=device),
                "logits_person": torch.zeros((0, self.num_classes), device=device),
                "logits_clip": torch.zeros((M, 0, self.num_classes), device=device),
            }

        # ROI projection。
        person_feats_flat = self.roi_proj(person_feats_flat)

        # 3. bbox embedding + type embedding。
        person_feats_flat = self._add_bbox_and_type_embedding(
            person_feats_flat=person_feats_flat,
            extended_bboxes=extended_bboxes,
            extended_bbox_masks=extended_bbox_masks,
            M=M,
            N_plus_1=N_plus_1,
            T=T,
        )

        # reshape to (M,N+1,T,C)
        C = self.mil_feat_dim
        person_feats = person_feats_flat.view(M, N_plus_1, T, C)
        person_mask = person_mask_flat.view(M, N_plus_1, T)

        # 4. Actor-Global Cross Attention：每个人物/球 token 与全局特征交互。
        attn_weights_actor_global = None
        if self.use_actor_global:
            featmap_for_global = self.featmap_proj(featmap)
            forward_result = self.actor_global(
                person_feats=person_feats,
                featmap=featmap_for_global,
                person_mask=person_mask,
                return_attn=return_weights,
            )
            if return_weights and isinstance(forward_result, tuple):
                person_feats, attn_weights_actor_global = forward_result
            else:
                person_feats = forward_result  # (M,N+1,T,C)

        # 5. Person-Person Interaction：先在每一帧上做球员/ball 之间交互，再沿时间聚合。
        if self.use_person_relation:
            person_feats_frame = person_feats.permute(0, 2, 1, 3).contiguous().view(
                M * T, N_plus_1, C
            )  # (M*T,N+1,C)
            person_mask_frame = person_mask.permute(0, 2, 1).contiguous().view(
                M * T, N_plus_1
            )  # (M*T,N+1)
            person_feats_frame = self.person_relation(
                person_feats_frame,
                valid_mask=person_mask_frame,
            )
            person_feats = person_feats_frame.view(M, T, N_plus_1, C).permute(0, 2, 1, 3).contiguous()
            person_feats = person_feats * person_mask.to(dtype=person_feats.dtype).unsqueeze(-1)

        # 6. clip 内 temporal pooling：每个 clip 内，T 帧 -> 1 个 token。
        pooled_feats_flat = self.feature_pooler(
            person_feats.view(M * N_plus_1, T, C),
            person_mask.view(M * N_plus_1, T),
        )  # (M*(N+1),C)
        pooled_feats_clip = pooled_feats_flat.view(M, N_plus_1, C)  # (M,N+1,C)

        # clip-token valid mask: (M,N+1)，只要该 token 在该 clip 任一帧有效。
        clip_token_valid = person_mask.any(dim=2)  # (M,N+1)

        # 7. Clip-Clip Interaction：同一球员跨 clip 交互。
        if self.use_clip_relation:
            pooled_feats_clip = self.clip_relation(
                pooled_feats_clip,
                valid_mask=clip_token_valid,
            )

        # 8. clip-level logits：只对 person 分类，ball 不分类。
        logits_clip_all = self.clip_head(pooled_feats_clip.view(M * N_plus_1, C)).view(
            M, N_plus_1, self.num_classes
        )
        logits_clip = logits_clip_all[:, :N, :]

        # 9. 去掉 ball，只对 person 做 MIL pooling。
        pooled_feats_clip_persons = pooled_feats_clip[:, :N, :]  # (M,N,C)
        person_clip_valid = clip_token_valid[:, :N]              # (M,N)

        if return_weights:
            if self.pooling_mode == "topk":
                pooled_feats_person, gate_logits, gate_weights = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    topk=topk,
                    return_weights=True,
                )
            else:
                pooled_feats_person, gate_logits, gate_weights = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    return_weights=True,
                )
        else:
            if self.pooling_mode == "topk":
                pooled_feats_person = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    topk=topk,
                    return_weights=False,
                )
            else:
                pooled_feats_person = self.clip_pool(
                    pooled_feats_clip_persons,
                    valid_mask=person_clip_valid,
                    return_weights=False,
                )

        # 10. person-level classification。
        logits_person = self.person_head(pooled_feats_person, person_valid)

        out = {
            "pooled_feats_flat": pooled_feats_flat,              # (M*(N+1),C), includes ball
            "pooled_feats_clip_all": pooled_feats_clip,          # (M,N+1,C), includes ball
            "pooled_feats_clip": pooled_feats_clip_persons,      # (M,N,C), persons only
            "pooled_feats_person": pooled_feats_person,          # (N,C)
            "logits_person": logits_person,                      # (N,num_classes)
            "logits_clip": logits_clip,                          # (M,N,num_classes)
            "person_valid": person_valid,                        # (N,)
            "person_clip_valid": person_clip_valid,              # (M,N)
        }

        if return_weights:
            out["gate_logits"] = gate_logits       # (M,N)
            out["gate_weights"] = gate_weights     # (M,N,1)
            if attn_weights_actor_global is not None:
                out["attn_weights_actor_global"] = attn_weights_actor_global  # (M,T,N,Hf,Wf)

        return out

    def forward(
        self,
        clips_video: torch.Tensor,
        idx: torch.Tensor,
        nums: torch.Tensor,
        bboxes: List[torch.Tensor],
        bbox_masks: List[torch.Tensor],
        clips_ball: torch.Tensor,
        clips_ball_mask: torch.Tensor,
        fps_in: float,
        topk: int,
        return_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_mil_one_video(
            clips_video=clips_video,
            idx=idx,
            nums=nums,
            bboxes=bboxes,
            bbox_masks=bbox_masks,
            clips_ball=clips_ball,
            clips_ball_mask=clips_ball_mask,
            fps_in=fps_in,
            topk=topk,
            return_weights=return_weights,
        )
