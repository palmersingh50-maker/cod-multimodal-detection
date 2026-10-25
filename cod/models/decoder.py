from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def cbg(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, 1, k // 2, bias=False), nn.GroupNorm(8, o), nn.GELU())


class MultiScaleDecoder(nn.Module):
    """Top-down decoder over a 4-level pyramid built from single-stride ViT features,
    plus a high-resolution convolutional stem on the raw image for thin structures / boundaries.

    Pyramid sizes: H/4 (shallow block) ... H/32 (deep block); logits at H/2 refined with
    the stem then bilinearly upsampled to full resolution.
    """

    def __init__(self, dim: int, boundary: bool = False):
        super().__init__()
        self.stem2 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.GroupNorm(8, 32), nn.GELU())  # H/2
        self.stem4 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.GroupNorm(8, 64), nn.GELU())  # H/4
        self.lat = nn.ModuleList([cbg(dim, dim, 1) for _ in range(4)])
        self.smooth = nn.ModuleList([cbg(dim, dim) for _ in range(3)])
        self.fuse4 = cbg(dim + 64, dim)
        self.fuse2 = nn.Sequential(cbg(dim + 32, dim // 2), cbg(dim // 2, dim // 2))
        self.head = nn.Conv2d(dim // 2, 1, 1)
        self.boundary = nn.Sequential(cbg(dim, dim // 2), nn.Conv2d(dim // 2, 1, 1)) if boundary else None

    def forward(self, feats: list[torch.Tensor], image: torch.Tensor) -> dict:
        H, W = image.shape[-2:]
        sizes = [(H // 4, W // 4), (H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
        p = [F.interpolate(l(f), size=s, mode="bilinear", align_corners=False) for l, f, s in zip(self.lat, feats, sizes)]
        x = p[3]
        for i in (2, 1, 0):  # top-down
            x = self.smooth[i](p[i] + F.interpolate(x, size=sizes[i], mode="bilinear", align_corners=False))
        s2 = self.stem2(image)
        s4 = self.stem4(s2)
        x4 = self.fuse4(torch.cat([x, s4], 1))
        x2 = self.fuse2(torch.cat([F.interpolate(x4, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        out = {"logits": F.interpolate(self.head(x2), size=(H, W), mode="bilinear", align_corners=False)}
        if self.boundary is not None:
            out["boundary"] = F.interpolate(self.boundary(x4), size=(H, W), mode="bilinear", align_corners=False)
        return out
