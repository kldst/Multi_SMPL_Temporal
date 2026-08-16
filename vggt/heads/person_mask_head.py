"""Per-person, per-view segmentation-mask head (MaskFormer-style).

Each person query already localises "its" person for the SMPL head; this head
makes that localisation *explicit and dense* by predicting, for every person and
every view, a soft occupancy mask over the aggregator patch grid.  Supervised by
the instance mask shipped with the raw data (``*.mask.jpg``, pixel value ==
person_idx + 1, so it is occlusion-aware), it pushes each person token to attend
to exactly one body's pixels -- the same identity-disentangling pressure as the
dense-landmark head, but expressed directly in image space.

Mechanism
---------
The mask for (person p, view s) is the scaled dot product between a projected
person token and every patch token of that view:

    logits[b, s, p, n] = <W_q · person_token[b, p], W_k · patch_token[b, s, n]> / sqrt(D)

which is one attention map per (person, view) -- cheap, and it reuses the very
tokens the SMPL head binds to.  Output is patch-resolution logits; the loss
compares them to the down-sampled instance mask.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt.heads.dpt_head import DPTHead


class PersonMaskHead(nn.Module):
    """Predict per-person per-view patch-grid mask logits.

    Args:
        query_dim: person-token channel dim (SMPL head query dim).
        context_dim: aggregator patch-token channel dim (2*embed_dim).
        embed_dim: shared dot-product embedding dim.
    """

    def __init__(self, *, query_dim: int = 1024, context_dim: int = 2048, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.q_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, embed_dim),
        )
        self.k_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, embed_dim),
        )
        # Learnable temperature (log-scale) on the dot product.
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        person_tokens: torch.Tensor,     # (B, P, C_q)
        patch_tokens: torch.Tensor,      # (B, S, N_patch, C_ctx)
        patch_hw: Optional[tuple] = None,
    ) -> torch.Tensor:
        B, P, _ = person_tokens.shape
        _, S, N, _ = patch_tokens.shape

        q = self.q_proj(person_tokens)                       # (B, P, D)
        k = self.k_proj(patch_tokens)                        # (B, S, N, D)

        # logits[b, s, p, n] = q[b, p] . k[b, s, n]
        logits = torch.einsum("bpd,bsnd->bspn", q, k)
        logits = logits * (self.embed_dim ** -0.5) * self.log_scale.exp()

        if patch_hw is not None:
            ph, pw = patch_hw
            if ph * pw != N:
                raise ValueError(
                    f"patch_hw {patch_hw} does not match {N} patch tokens"
                )
            logits = logits.reshape(B, S, P, ph, pw)
        return logits                                        # (B,S,P,N) or (B,S,P,H,W)


class PersonMaskDPTHead(nn.Module):
    """Pixel-level per-person mask head (Mask2Former-style dot product on a DPT
    per-pixel embedding map).

    The 37x37 patch-grid ``PersonMaskHead`` is too coarse to separate two people
    in contact -- a contact boundary spans only a couple of patches.  This head
    runs a DPT trunk ONCE over the aggregator's multi-layer tokens to produce a
    dense per-pixel embedding map at ``H/down_ratio`` resolution, then dots it
    with a projection of each person token:

        logits[b, s, p, y, x] = <W_q person_token[b, p], pixel_embed[b, s, :, y, x]>

    so P people cost one trunk pass + P cheap dot products (NOT P DPT passes).
    Supervised by the pixel-level instance mask (``*.mask.jpg``) area-pooled to
    the same resolution.

    Args:
        dim_in: aggregator token channel dim (2*embed_dim).
        query_dim: person-token channel dim (SMPL head query dim).
        embed_dim: shared dot-product embedding dim.
        features: DPT trunk fusion channels.
        out_channels: DPT per-layer projection channels.
        intermediate_layer_idx: aggregator layers feeding the trunk (same
            convention as the depth/point heads).
        down_ratio: output stride vs. the input image (2 -> 259x259 for 518).
    """

    def __init__(
        self,
        *,
        dim_in: int = 2048,
        query_dim: int = 1024,
        embed_dim: int = 128,
        features: int = 256,
        out_channels: Optional[List[int]] = None,
        intermediate_layer_idx: Optional[List[int]] = None,
        down_ratio: int = 2,
        patch_size: int = 14,
        frames_chunk_size: int = 8,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.down_ratio = int(down_ratio)
        # feature_only=True: the trunk returns the fused per-pixel feature map
        # (B, S, features, H/down_ratio, W/down_ratio) without an activation head.
        # Defaults mirror the depth head so its pretrained trunk weights can be
        # copied in for a warm start (see trainer's mask-head init option).
        self.trunk = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            out_channels=out_channels or [256, 512, 1024, 1024],
            intermediate_layer_idx=intermediate_layer_idx or [4, 11, 17, 23],
            feature_only=True,
            down_ratio=self.down_ratio,
            frames_chunk_size=int(frames_chunk_size),
        )
        self.pixel_proj = nn.Conv2d(features, self.embed_dim, kernel_size=1)
        self.q_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, self.embed_dim),
        )
        # Learnable temperature (log-scale) on the dot product.
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        person_tokens: torch.Tensor,               # (B, P, C_q)
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,                      # (B, S, 3, H, W)
        patch_start_idx: int,
    ) -> torch.Tensor:
        B, P, _ = person_tokens.shape
        S = images.shape[1]
        q = self.q_proj(person_tokens)                           # (B, P, D)

        def decode_chunk(q_chunk, tokens, start=None, end=None):
            if start is None:
                feat = self.trunk(
                    tokens, images=images, patch_start_idx=patch_start_idx
                )
            else:
                # Stream high-channel DPT features directly into low-channel
                # person logits. Calling _forward_impl avoids DPTHead.forward's
                # full-resolution torch.cat over all view chunks.
                feat = self.trunk._forward_impl(
                    tokens,
                    images,
                    patch_start_idx,
                    frames_start_idx=start,
                    frames_end_idx=end,
                )
            chunk_views = feat.shape[1]
            Hm, Wm = feat.shape[-2:]
            pix = self.pixel_proj(
                feat.reshape(B * chunk_views, *feat.shape[2:])
            )
            pix = pix.reshape(B, chunk_views, self.embed_dim, Hm, Wm)
            logits = torch.einsum(
                "bpd,bsdhw->bsphw", q_chunk, pix.to(q_chunk.dtype)
            )
            return logits * (self.embed_dim ** -0.5) * self.log_scale.exp()

        chunk_size = self.trunk.frames_chunk_size
        if chunk_size is None or int(chunk_size) >= S:
            return decode_chunk(q, aggregated_tokens_list)

        logits_chunks = []
        for start in range(0, S, int(chunk_size)):
            end = min(start + int(chunk_size), S)
            if self.training and torch.is_grad_enabled():
                # Return only logits from the checkpoint. Backward recomputes
                # the much larger DPT feature, rather than retaining one
                # full-resolution 256-channel tensor per view.
                def checkpointed(q_input, *token_inputs, _s=start, _e=end):
                    return decode_chunk(q_input, list(token_inputs), _s, _e)

                logits = checkpoint(
                    checkpointed,
                    q,
                    *aggregated_tokens_list,
                    use_reentrant=False,
                )
            else:
                logits = decode_chunk(q, aggregated_tokens_list, start, end)
            logits_chunks.append(logits)
        return torch.cat(logits_chunks, dim=1)                    # (B, S, P, H', W')


__all__ = ["PersonMaskHead", "PersonMaskDPTHead"]
