"""RGB/depth token fusion modules.

All fusers map (rgb_tokens, depth_tokens, depth_cues) -> fused tokens, shape (B, N, D).
They are *residual on the RGB stream* so the RGB pathway survives unreliable depth;
the attention-based ones start as the identity (zero-initialised output gate).

`depth_cues` is (B, N, 2): [local depth-gradient magnitude, validity] on the token grid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSION_TYPES = ("none", "concat", "cross_attn", "gated", "reliability")


class ConcatFusion(nn.Module):
    """1) concatenation + projection, added residually to the RGB tokens."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.norm_r, self.norm_d = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.proj[-1].weight), nn.init.zeros_(self.proj[-1].bias)

    def forward(self, rgb, depth, cues):
        return rgb + self.proj(torch.cat([self.norm_r(rgb), self.norm_d(depth)], -1))


class _CrossAttn(nn.Module):
    """Multi-head cross-attention: queries from RGB, keys/values from depth."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.h = heads
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)

    def forward(self, rgb, depth):
        B, N, D = rgb.shape
        q = self.q(self.norm_q(rgb)).view(B, N, self.h, -1).transpose(1, 2)
        k, v = self.kv(self.norm_kv(depth)).view(B, -1, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4).unbind(0)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        return self.out(o)


class CrossAttnFusion(nn.Module):
    """2) cross-attention with a learnable scalar tanh gate initialised at 0."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = _CrossAttn(dim, heads)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, rgb, depth, cues):
        return rgb + torch.tanh(self.alpha) * self.attn(rgb, depth)


class GatedFusion(nn.Module):
    """3) cross-attention output blended in through a learned per-token, per-channel sigmoid gate."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = _CrossAttn(dim, heads)
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.constant_(self.gate.bias, -2.0)  # start mostly closed: output ~= RGB

    def forward(self, rgb, depth, cues):
        a = self.attn(rgb, depth)
        g = torch.sigmoid(self.gate(torch.cat([rgb, a], -1)))
        return rgb + g * a


class ReliabilityFusion(nn.Module):
    """4) gated cross-attention whose gate is scaled by a predicted per-token depth reliability r in [0,1].

    r is estimated from RGB tokens, depth tokens, their disagreement and hand-crafted
    depth cues (gradient magnitude, validity). An invalid depth map forces r -> 0.
    `last_reliability` (B, N, 1) is kept for inspection / over-reliance analysis.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = _CrossAttn(dim, heads)
        self.rel = nn.Sequential(nn.Linear(3 * dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.gate = nn.Linear(2 * dim, dim)
        nn.init.constant_(self.gate.bias, -2.0)
        self.last_reliability: torch.Tensor | None = None

    def forward(self, rgb, depth, cues):
        a = self.attn(rgb, depth)
        r = torch.sigmoid(self.rel(torch.cat([rgb, depth, (rgb - depth).abs(), cues], -1)))
        r = r * cues[..., 1:2]  # hard zero when depth is invalid
        self.last_reliability = r
        g = torch.sigmoid(self.gate(torch.cat([rgb, a], -1)))
        return rgb + r * g * a


def build_fuser(kind: str, dim: int, heads: int) -> nn.Module:
    return {"concat": ConcatFusion, "cross_attn": CrossAttnFusion, "gated": GatedFusion,
            "reliability": ReliabilityFusion}[kind](dim, heads)
