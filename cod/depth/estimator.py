"""Monocular depth estimation (Depth Anything V2) and depth normalisation.

IMPORTANT: outputs are *estimated relative inverse depth* (larger = nearer),
produced by a network from the RGB image alone. They are not sensor
measurements and not ground-truth geometry.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def normalize_depth(depth: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0, eps: float = 1e-6):
    """Robust per-image min-max normalisation to [0, 1].

    Returns (depth, valid). `valid` is False when the prediction is non-finite
    or (near-)constant; the returned depth is then all zeros so downstream
    code can treat it as missing.
    """
    d = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(d)
    if finite.mean() < 0.5:
        return np.zeros_like(d), False
    d = np.where(finite, d, np.nanmedian(d[finite]))
    lo, hi = np.percentile(d, [lo_pct, hi_pct])
    if hi - lo < eps:
        return np.zeros_like(d), False
    return np.clip((d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32), True


class DepthEstimator:
    """Wraps a Hugging Face Depth Anything V2 model.

    Depth is predicted at `input_size` (multiple of 14) and bilinearly resized
    back to the original image size, so it stays pixel-aligned with the RGB.
    """

    def __init__(self, model_name: str = DEFAULT_DEPTH_MODEL, device: torch.device | str = "cpu",
                 input_size: int = 518):
        from transformers import AutoModelForDepthEstimation

        self.device = torch.device(device)
        self.input_size = input_size
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()
        self.mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def predict_raw(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Raw (un-normalised) relative inverse depth at each image's own size."""
        outs = []
        for im in images:  # one at a time: sizes differ, and depth runs once per image
            w, h = im.size
            x = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0)
            x = x.permute(2, 0, 1)[None].to(self.device)
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
            x = ((x.clamp(0, 1) - self.mean) / self.std)
            d = self.model(pixel_values=x).predicted_depth  # (1,H,W)
            d = F.interpolate(d[None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            outs.append(d.float().cpu().numpy())
        return outs

    def predict(self, image: Image.Image) -> tuple[np.ndarray, bool]:
        """Normalised depth in [0,1] (H,W float32) and validity flag."""
        return normalize_depth(self.predict_raw([image])[0])


def save_depth_png(path, depth: np.ndarray, valid: bool) -> None:
    """uint16 PNG cache. Invalid depth is stored as all zeros (also flagged by value 0 everywhere)."""
    arr = np.round(np.clip(depth, 0, 1) * 65535).astype(np.uint16)
    Image.fromarray(arr).save(path)


def load_depth_png(path) -> tuple[np.ndarray, bool]:
    arr = np.asarray(Image.open(path), dtype=np.float32) / 65535.0
    valid = bool(arr.max() > 0)
    return arr, valid
