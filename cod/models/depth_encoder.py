from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ONNX has no antialiased resize in torch's exporter; scripts/export_onnx.py flips this flag (and reports the deviation).
ONNX_COMPAT = False


def to_grid(x: torch.Tensor, grid) -> torch.Tensor:
    """Antialiased resize to the token grid (adaptive pooling is unsupported for non-divisible sizes on MPS)."""
    if tuple(x.shape[-2:]) == tuple(grid):
        return x
    return F.interpolate(x, size=tuple(grid), mode="bilinear", antialias=not ONNX_COMPAT, align_corners=False)


def _cbr(i, o, s):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.GroupNorm(8, o), nn.GELU())


class DepthEncoder(nn.Module):
    """Lightweight CNN: (depth, |grad depth|, validity) -> feature map on the ViT token grid.

    Swappable: any module with `forward(depth, valid, grid) -> (B, C, h, w)` and attribute
    `out_channels` can replace it (register in `DEPTH_ENCODERS`).
    """

    def __init__(self, channels: int = 48):
        super().__init__()
        self.out_channels = channels
        self.net = nn.Sequential(_cbr(3, channels // 2, 2), _cbr(channels // 2, channels, 2), _cbr(channels, channels, 2),
                                 _cbr(channels, channels, 1))
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        self.register_buffer("sobel", torch.stack([k, k.t()])[:, None])

    def cues(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        g = F.conv2d(F.pad(depth, (1, 1, 1, 1), mode="replicate"), self.sobel)
        grad = g.pow(2).sum(1, keepdim=True).add(1e-8).sqrt()
        v = valid.view(-1, 1, 1, 1).expand_as(depth)
        return torch.cat([depth * v, grad * v, v], 1)

    def forward(self, depth, valid, grid):
        x = self.net(self.cues(depth, valid))
        return to_grid(x, grid)


DEPTH_ENCODERS = {"cnn": DepthEncoder}
