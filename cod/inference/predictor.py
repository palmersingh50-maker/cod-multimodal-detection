"""Single-image inference: preprocessing -> (cached) depth -> model -> postprocessing, with per-stage timings."""
from __future__ import annotations

import hashlib
import io
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cod.data.dataset import MEAN, STD
from cod.depth.estimator import DepthEstimator
from cod.models import CODModel
from cod.utils import get_device, timer


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[CODModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = CODModel(cfg, pretrained=False)  # weights come from the checkpoint
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


class Predictor:
    """Holds one trained COD model; shares a depth estimator (and depth cache) with other predictors."""

    def __init__(self, checkpoint: str | Path, device: str | torch.device = "auto",
                 depth_estimator: DepthEstimator | None = None, depth_cache_size: int = 32):
        self.device = get_device(device) if isinstance(device, str) else device
        self.model, self.cfg = load_checkpoint(checkpoint, self.device)
        self.size = self.cfg["data"]["image_size"]
        self.fusion = self.cfg["model"]["fusion"]
        self.depth_estimator = depth_estimator
        self._cache: OrderedDict[str, tuple[np.ndarray, bool]] = OrderedDict()
        self._cache_size = depth_cache_size

    @property
    def uses_depth(self) -> bool:
        return self.fusion != "none"

    def estimate_depth(self, image: Image.Image, key: str | None = None) -> tuple[np.ndarray, bool]:
        """Estimated (not measured) depth; memoised by content hash so repeated requests skip the depth network."""
        if self.depth_estimator is None:
            raise RuntimeError("no depth estimator configured")
        key = key or hashlib.sha1(image.tobytes()).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        res = self.depth_estimator.predict(image)
        self._cache[key] = res
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return res

    @torch.inference_mode()
    def predict(self, image: Image.Image, use_depth: bool | None = None, depth: tuple[np.ndarray, bool] | None = None) -> dict:
        """Returns {"prob": HxW float32 in [0,1] at original resolution, "timings_ms": {...}, "used_depth": bool,
        "reliability": optional HxW}. use_depth=None -> use depth iff the model fuses it."""
        use_depth = self.uses_depth if use_depth is None else use_depth
        if use_depth and not self.uses_depth:
            raise ValueError("this checkpoint is RGB-only and cannot use depth")
        t: dict[str, float] = {}
        image = image.convert("RGB")
        w, h = image.size
        with timer(self.device, t, "preprocess_ms"):
            arr = np.asarray(image)
            x = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_AREA if min(h, w) > self.size else cv2.INTER_LINEAR)
            x = torch.from_numpy(((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))[None].to(self.device)
        d = v = None
        if use_depth:
            with timer(self.device, t, "depth_ms"):
                dmap, ok = depth if depth is not None else self.estimate_depth(image)
                d = torch.from_numpy(cv2.resize(dmap, (self.size, self.size), interpolation=cv2.INTER_LINEAR))[None, None].to(self.device)
                v = torch.tensor([float(ok)], device=self.device)
        with timer(self.device, t, "model_ms"):
            out = self.model(x, d, v)
        with timer(self.device, t, "postprocess_ms"):
            prob = torch.sigmoid(out["logits"].float())
            prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            rel = None
            if "reliability" in out:
                rel = F.interpolate(out["reliability"].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        t["total_ms"] = sum(t.values())
        return {"prob": prob, "reliability": rel, "timings_ms": t, "used_depth": bool(use_depth and d is not None)}


def to_mask(prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return ((prob >= threshold) * 255).astype(np.uint8)


def make_overlay(image: Image.Image, prob: np.ndarray, threshold: float = 0.5, opacity: float = 0.5,
                 color=(255, 64, 64)) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = (prob >= threshold)[..., None]
    tint = np.array(color, dtype=np.float32)
    out = np.where(m, base * (1 - opacity) + tint * opacity, base)
    edge = cv2.morphologyEx(m[..., 0].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    out[edge] = tint
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def png_bytes(arr_or_img) -> bytes:
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
