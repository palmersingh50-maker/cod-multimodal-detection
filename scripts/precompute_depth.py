"""Pre-compute and cache estimated depth maps (Depth Anything V2) for dataset splits."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from tqdm import tqdm

from cod.config import load_config
from cod.data.dataset import build_samples, depth_path
from cod.depth.estimator import DepthEstimator, save_depth_png
from cod.utils import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=None, help="only first N per split (smoke tests)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dcfg = cfg["data"]
    cache = Path(dcfg["depth_cache"])
    est = DepthEstimator(cfg["depth"]["model"], get_device(a.device), cfg["depth"]["input_size"])
    seen, invalid = set(), 0
    for split in a.split:
        samples = build_samples(dcfg, split, cfg["experiment"]["seed"])[: a.limit]
        (cache / "cod10k").mkdir(parents=True, exist_ok=True)
        for s in tqdm(samples, desc=split):
            if s.name in seen:
                continue
            seen.add(s.name)
            p = depth_path(cache, "cod10k", s.name)
            if p.exists():
                continue
            d, ok = est.predict(Image.open(s.image).convert("RGB"))
            invalid += not ok
            save_depth_png(p, d, ok)
    print(f"done; {invalid} invalid/constant depth maps stored as zeros")


if __name__ == "__main__":
    main()
