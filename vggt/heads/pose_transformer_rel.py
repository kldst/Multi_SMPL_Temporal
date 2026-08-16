"""Relative-position temporal decoder blocks shared by temporal SMPL heads.

The temporal axis is encoded without an absolute embedding:

* query self-attention applies 1-D RoPE along time;
* query-to-image cross-attention adds a learned T5-style bias based on
  ``context_time - query_time``.

``MultiPersonRelTemporalTransformer`` keeps person slots independent in
self-attention (the same role played by the two player groups in the badminton
model) while sharing one full-scene spatio-temporal image context.  This avoids
materialising one copy of all image keys/values per person slot.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _split_heads(value: torch.Tensor, heads: int) -> torch.Tensor:
    B, N, channels = value.shape
    if channels % heads:
        raise ValueError(f"Channels {channels} are not divisible by heads {heads}")
    return value.reshape(B, N, heads, channels // heads).permute(0, 2, 1, 3)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    B, heads, N, channels = value.shape
    return value.permute(0, 2, 1, 3).reshape(B, N, heads * channels)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


def build_rope_cache(seq_len: int, dim: int, base: float, device, dtype):
    if dim % 2 != 0:
        raise ValueError(f"RoPE needs an even head dimension, got {dim}")
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            / dim
        )
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.einsum("i,j->ij", positions, inv_freq)
    embedding = torch.cat([angles, angles], dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


class RoPESelfAttention(nn.Module):
    """Self-attention over one person's T query tokens."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rope_base=10000.0):
        super().__init__()
        if dim_head % 2 != 0:
            raise ValueError(f"RoPE needs an even head dimension, got {dim_head}")
        inner_dim = heads * dim_head
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.scale = dim_head**-0.5
        self.rope_base = float(rope_base)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*P, T, D)
        T = x.shape[1]
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda value: _split_heads(value, self.heads), (q, k, v))
        cos, sin = build_rope_cache(
            T, self.dim_head, self.rope_base, x.device, x.dtype
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        weights = self.dropout(
            self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        )
        out = torch.matmul(weights, v)
        return self.to_out(_merge_heads(out))


def relative_position_bucket(
    relative_position: torch.Tensor,
    *,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """Bidirectional T5 log-bucketing of signed temporal distances."""
    if num_buckets < 4 or num_buckets % 2:
        raise ValueError("num_buckets must be an even integer >= 4")
    half_buckets = num_buckets // 2
    buckets = (relative_position > 0).to(torch.long) * half_buckets
    distance = relative_position.abs()
    max_exact = half_buckets // 2
    is_small = distance < max_exact
    safe_distance = distance.float().clamp(min=max_exact)
    denominator = math.log(max(float(max_distance) / max_exact, 1.000001))
    large = max_exact + (
        torch.log(safe_distance / max_exact)
        / denominator
        * (half_buckets - max_exact)
    ).to(torch.long)
    large = torch.minimum(
        large, torch.full_like(large, half_buckets - 1)
    )
    return buckets + torch.where(is_small, distance, large)


class SharedContextRelBiasCrossAttention(nn.Module):
    """Cross-attend P*T queries to one shared T*M image-token context."""

    def __init__(
        self,
        dim,
        context_dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = int(heads)
        self.scale = dim_head**-0.5
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.rel_bias = nn.Embedding(self.num_buckets, self.heads)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        )

    def _bias(self, T: int, M: int, P: int, device) -> torch.Tensor:
        query_time = torch.arange(T, device=device)
        # Queries are flattened in person-major order: (P, T).
        query_time = query_time.repeat(P)[:, None]
        key_time = torch.arange(T, device=device).repeat_interleave(M)[None]
        buckets = relative_position_bucket(
            key_time - query_time,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        bias = self.rel_bias(buckets)  # (P*T, T*M, heads)
        return bias.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        T: int,
        M: int,
    ) -> torch.Tensor:
        # queries: (B,P,T,D), context: (B,T*M,C)
        B, P, T_in, _ = queries.shape
        if T_in != T or context.shape[1] != T * M:
            raise ValueError(
                "Temporal cross-attention shape mismatch: "
                f"queries={tuple(queries.shape)}, context={tuple(context.shape)}, "
                f"T={T}, M={M}"
            )
        q = self.to_q(queries.reshape(B, P * T, -1))
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda value: _split_heads(value, self.heads), (q, k, v))
        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        logits = logits + self._bias(T, M, P, queries.device).to(logits.dtype)
        weights = self.dropout(self.attend(logits))
        out = torch.matmul(weights, v)
        out = _merge_heads(out)
        return self.to_out(out).reshape(B, P, T, -1)


class MultiPersonRelDecoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        context_dim,
        rope_base,
        num_buckets,
        max_distance,
    ):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attention = RoPESelfAttention(
            dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attention = SharedContextRelBiasCrossAttention(
            dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            num_buckets=num_buckets,
            max_distance=max_distance,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, context, *, T, M):
        B, P, _, D = x.shape
        self_input = self.norm_self(x).reshape(B * P, T, D)
        x = x + self.self_attention(self_input).reshape(B, P, T, D)
        x = x + self.cross_attention(
            self.norm_cross(x), context, T=T, M=M
        )
        return x + self.ff(self.norm_ff(x))


class MultiPersonRelTemporalTransformer(nn.Module):
    """Temporal decoder for P persistent person slots and shared image context."""

    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        context_dim,
        rope_base=10000.0,
        num_buckets=32,
        max_distance=128,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MultiPersonRelDecoderLayer(
                    dim,
                    heads,
                    dim_head,
                    mlp_dim,
                    dropout,
                    context_dim,
                    rope_base,
                    num_buckets,
                    max_distance,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, queries, context, *, T, M):
        x = queries
        for layer in self.layers:
            x = layer(x, context, T=T, M=M)
        return x


__all__ = [
    "MultiPersonRelTemporalTransformer",
    "RoPESelfAttention",
    "SharedContextRelBiasCrossAttention",
    "relative_position_bucket",
]
