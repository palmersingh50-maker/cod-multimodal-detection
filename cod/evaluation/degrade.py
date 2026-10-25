"""Controlled corruptions of (estimated) depth maps for robustness / over-reliance studies."""
from __future__ import annotations

import torch
import torch.nn.functional as F

MODES = ("noise", "blur", "shuffle", "constant", "invert", "quantize")


def degrade_depth(depth: torch.Tensor, mode: str, severity: float = 1.0, gen: torch.Generator | None = None) -> torch.Tensor:
    """depth: (B,1,H,W) in [0,1]. `severity` scales the corruption strength (noise std = 0.25*severity, etc.)."""
    if mode == "noise":
        n = torch.randn(depth.shape, generator=gen) * 0.25 * severity
        return (depth + n.to(depth.device)).clamp(0, 1)
    if mode == "blur":
        k = int(2 * round(15 * severity) + 1)
        return F.avg_pool2d(F.pad(depth, (k // 2,) * 4, mode="replicate"), k, 1)
    if mode == "shuffle":  # depth from a different image: spatially plausible but wrong for this scene
        return depth.roll(1, 0) if depth.shape[0] > 1 else torch.flip(depth, (3,))
    if mode == "constant":
        return torch.full_like(depth, 0.5)
    if mode == "invert":
        return 1 - depth
    if mode == "quantize":
        levels = max(2, int(round(16 / max(severity, 1e-3))))
        return torch.round(depth * (levels - 1)) / (levels - 1)
    raise ValueError(f"unknown degradation {mode!r}; choose from {MODES}")
