"""Relative-position temporal multi-person SMPL decoder.

This is the full-scene, multi-query counterpart of the badminton temporal SMPL
head.  The VGGT aggregator runs independently for every timestep; this head then
jointly decodes all T frames.  A learned slot embedding keeps P candidate people
distinct, temporal self-attention is restricted to one slot's trajectory, and
all slots cross-attend to the shared spatio-temporal image context.

For compatibility with the existing framewise SMPL loss API, outputs are
flattened from ``[B,T,P,...]`` to ``[B*T,P,...]`` at the module boundary.  The
joint temporal computation happens before that flattening.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from vggt.heads.pose_transformer_rel import MultiPersonRelTemporalTransformer


@dataclass(frozen=True)
class SMPLMultiQueryTemporalRelConfig:
    transformer_depth: int = 6
    transformer_heads: int = 8
    transformer_mlp_dim: int = 1024
    transformer_dim_head: int = 64
    transformer_dropout: float = 0.0
    transformer_dim: int = 1024
    ief_iters: int = 1
    max_T: int = 128
    mean_params_path: Optional[str] = None
    rope_base: float = 10000.0
    rel_num_buckets: int = 32
    rel_max_distance: int = 31


def _default_mean_params_path() -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "tram" / "data" / "smpl" / "smpl_mean_params.npz"
    return str(candidate) if candidate.is_file() else None


def _load_mean_params(path: Optional[str]):
    path = path or _default_mean_params_path()
    if path is None:
        return np.zeros(72, np.float32), np.zeros(10, np.float32)
    data = np.load(path)
    pose = data.get("pose")
    shape = data.get("shape")
    if pose is None or shape is None:
        return np.zeros(72, np.float32), np.zeros(10, np.float32)
    return (
        pose.astype(np.float32).reshape(-1)[:72],
        shape.astype(np.float32).reshape(-1)[:10],
    )


class SMPLMultiQueryTransRotTemporalRelDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        num_people: int,
        smpl_cfg: Optional[SMPLMultiQueryTemporalRelConfig] = None,
    ):
        super().__init__()
        self.cfg = smpl_cfg or SMPLMultiQueryTemporalRelConfig()
        self.num_people = int(num_people)
        query_dim = int(self.cfg.transformer_dim)

        init_pose, init_betas = _load_mean_params(self.cfg.mean_params_path)
        # Keep the same names and shapes as SMPLRotTransformerDecoderHead so a
        # static checkpoint warm-starts the slot and output projections.
        self.register_buffer("init_pose", torch.from_numpy(init_pose).view(1, 1, 72))
        self.register_buffer("init_betas", torch.from_numpy(init_betas).view(1, 1, 10))
        self.person_queries = nn.Parameter(
            torch.randn(1, self.num_people, query_dim) * 0.02
        )
        self.transformer = MultiPersonRelTemporalTransformer(
            dim=query_dim,
            depth=int(self.cfg.transformer_depth),
            heads=int(self.cfg.transformer_heads),
            dim_head=int(self.cfg.transformer_dim_head),
            mlp_dim=int(self.cfg.transformer_mlp_dim),
            dropout=float(self.cfg.transformer_dropout),
            context_dim=dim_in,
            rope_base=float(self.cfg.rope_base),
            num_buckets=int(self.cfg.rel_num_buckets),
            max_distance=int(self.cfg.rel_max_distance),
        )
        self.decpose = nn.Linear(query_dim, 72)
        self.decshape = nn.Linear(query_dim, 10)
        self.dectrans = nn.Linear(query_dim, 3)
        self.decpresence = nn.Linear(query_dim, 1)
        for layer in (self.decpose, self.decshape, self.dectrans, self.decpresence):
            nn.init.xavier_uniform_(layer.weight, gain=0.01)

    def forward(self, context: torch.Tensor):
        # context: (B,T,M,C)
        B, T, M, C = context.shape
        if T < 1 or T > int(self.cfg.max_T):
            raise ValueError(f"Temporal SMPL T={T} is outside [1,{self.cfg.max_T}]")
        context_flat = context.reshape(B, T * M, C)
        queries = self.person_queries[:, :, None].expand(B, -1, T, -1).to(
            device=context.device, dtype=context.dtype
        )
        pred_pose = self.init_pose[:, None].expand(B, T, self.num_people, -1).to(context.dtype)
        pred_beta = self.init_betas[:, None].expand(B, T, self.num_people, -1).to(context.dtype)
        pred_translate = torch.zeros(
            B, T, self.num_people, 3, device=context.device, dtype=context.dtype
        )
        pred_pose_0 = pred_pose
        token_out = queries
        presence_logits = torch.zeros(
            B, T, self.num_people, device=context.device, dtype=context.dtype
        )

        for iteration in range(int(self.cfg.ief_iters)):
            # transformer returns person-major (B,P,T,D)
            token_out = self.transformer(queries, context_flat, T=T, M=M)
            frame_tokens = token_out.permute(0, 2, 1, 3).contiguous()
            pred_pose = pred_pose + self.decpose(frame_tokens)
            pred_beta = pred_beta + self.decshape(frame_tokens)
            pred_translate = pred_translate + self.dectrans(frame_tokens)
            presence_logits = self.decpresence(frame_tokens).squeeze(-1)
            if iteration == 0:
                pred_pose_0 = pred_pose

        frame_tokens = token_out.permute(0, 2, 1, 3).contiguous()
        return pred_pose, pred_beta, pred_translate, presence_logits, pred_pose_0, frame_tokens


class SMPLMultiQueryTransRotTemporalRelHead(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        num_people: int,
        smpl_cfg: Optional[SMPLMultiQueryTemporalRelConfig] = None,
        context_pool: str = "flatten",
    ):
        super().__init__()
        self.context_pool = str(context_pool)
        self.decoder = SMPLMultiQueryTransRotTemporalRelDecoder(
            dim_in=dim_in, num_people=num_people, smpl_cfg=smpl_cfg
        )

    @staticmethod
    def _metadata_value(smpl_inputs, key: str, default: int) -> int:
        value = smpl_inputs.get(key)
        if value is None:
            return int(default)
        if torch.is_tensor(value):
            flat = value.reshape(-1)
            if flat.numel() == 0:
                return int(default)
            if not torch.all(flat == flat[0]):
                raise ValueError(f"All samples must share {key}, got {flat.tolist()}")
            return int(flat[0].item())
        return int(value)

    def forward(
        self,
        aggregated_tokens_list,
        patch_start_idx: int,
        smpl_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        smpl_inputs = smpl_inputs or {}
        tokens = aggregated_tokens_list[-1]
        patch_tokens = tokens[:, :, patch_start_idx:, :]
        BT, V_in, N, C = patch_tokens.shape
        T = self._metadata_value(smpl_inputs, "temporal_num_frames", 1)
        V = self._metadata_value(smpl_inputs, "views_per_frame", V_in)
        if V != V_in or BT % T:
            raise ValueError(
                f"Folded temporal tokens {tuple(tokens.shape)} do not match T={T}, V={V}"
            )
        B = BT // T
        patch_tokens = patch_tokens.reshape(B, T, V, N, C)
        if self.context_pool == "mean":
            context = patch_tokens.mean(dim=3)
        elif self.context_pool == "flatten":
            context = patch_tokens.reshape(B, T, V * N, C)
        else:
            raise ValueError(f"Unknown context_pool: {self.context_pool}")

        (
            pred_pose,
            pred_beta,
            pred_translate,
            presence_logits,
            pred_pose_0,
            frame_tokens,
        ) = self.decoder(context)
        flat = lambda value: value.reshape(B * T, *value.shape[2:])
        pred_pose_flat = flat(pred_pose)
        return {
            "smpl_pose": pred_pose_flat,
            "smpl_beta": flat(pred_beta),
            "mesh_translate": flat(pred_translate),
            "mesh_rot": pred_pose_flat[..., :3],
            "smpl_presence_logits": flat(presence_logits),
            "pred_pose_0": flat(pred_pose_0),
            "person_tokens": flat(frame_tokens),
        }


__all__ = [
    "SMPLMultiQueryTemporalRelConfig",
    "SMPLMultiQueryTransRotTemporalRelDecoder",
    "SMPLMultiQueryTransRotTemporalRelHead",
]
