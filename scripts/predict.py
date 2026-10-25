"""Run inference on one image and write mask + overlay PNGs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from cod.depth.estimator import DepthEstimator
from cod.inference.predictor import Predictor, make_overlay, to_mask
from cod.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rgb-only", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    dev = get_device(a.device)
    p = Predictor(a.checkpoint, dev)
    use_depth = p.uses_depth and not a.rgb_only
    if use_depth:
        p.depth_estimator = DepthEstimator(device=dev)
    img = Image.open(a.image).convert("RGB")
    r = p.predict(img, use_depth=use_depth)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.image).stem
    Image.fromarray(to_mask(r["prob"], a.threshold)).save(out / f"{stem}_mask.png")
    make_overlay(img, r["prob"], a.threshold).save(out / f"{stem}_overlay.png")
    print({k: round(v, 1) for k, v in r["timings_ms"].items()}, "depth used:", r["used_depth"])


if __name__ == "__main__":
    main()
