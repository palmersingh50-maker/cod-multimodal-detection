"""Parameters, checkpoint size, latency (RGB-only vs multimodal, incl. depth), throughput and memory."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from cod.depth.estimator import DepthEstimator
from cod.inference.predictor import Predictor
from cod.utils import dump_json, get_device, hardware_info, peak_memory_mb, sync


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True, help="representative test image")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = get_device(a.device)
    p = Predictor(a.checkpoint, dev)
    img = Image.open(a.image).convert("RGB")
    rep = {"checkpoint": a.checkpoint, "checkpoint_mb": Path(a.checkpoint).stat().st_size / 2**20,
           "params_m": sum(x.numel() for x in p.model.parameters()) / 1e6, "image_size": p.size,
           "device": str(dev), "hardware": hardware_info(), "fusion": p.fusion}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def bench(fn):
        for _ in range(a.warmup):
            fn()
        ts, parts = [], []
        for _ in range(a.runs):
            sync(dev)
            t = time.perf_counter()
            r = fn()
            sync(dev)
            ts.append((time.perf_counter() - t) * 1000)
            parts.append(r["timings_ms"])
        ts = np.array(ts)
        return {"mean_ms": ts.mean(), "p50_ms": np.median(ts), "p95_ms": np.percentile(ts, 95), "fps": 1000 / ts.mean(),
                "stages_mean_ms": {k: float(np.mean([q[k] for q in parts])) for k in parts[0]}}

    rep["rgb_only"] = bench(lambda: p.predict(img, use_depth=False))
    if p.uses_depth:
        p.depth_estimator = DepthEstimator(device=dev)

        def uncached():
            p._cache.clear()  # force a fresh depth-network pass every call
            return p.predict(img, use_depth=True)

        rep["multimodal_depth_uncached"] = bench(uncached)
        p.estimate_depth(img)
        rep["multimodal_depth_cached"] = bench(lambda: p.predict(img, use_depth=True))
    rep["peak_memory_mb"] = peak_memory_mb(dev)
    for k in ("rgb_only", "multimodal_depth_uncached", "multimodal_depth_cached"):
        if k in rep:
            print(k, {x: (round(y, 1) if isinstance(y, float) else y) for x, y in rep[k].items() if x != "stages_mean_ms"})
    if a.out:
        dump_json(rep, a.out)


if __name__ == "__main__":
    main()
