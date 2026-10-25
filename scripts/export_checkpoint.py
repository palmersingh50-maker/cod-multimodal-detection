"""Strip optimizer/scheduler state from a training checkpoint -> small inference checkpoint (weights + config)."""
import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
a = ap.parse_args()
ck = torch.load(a.src, map_location="cpu", weights_only=False)
out = {"model": {k: v for k, v in ck["model"].items()}, "config": ck["config"],
       "epoch": ck["epoch"], "best": ck["best"]}
Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
torch.save(out, a.dst)
print(f"{a.dst}: {Path(a.dst).stat().st_size / 2**20:.0f} MB (from {Path(a.src).stat().st_size / 2**20:.0f} MB)")
