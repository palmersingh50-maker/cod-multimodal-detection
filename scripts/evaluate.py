"""Evaluate a checkpoint on COD10K test (and any extra folder datasets) with the full COD metric suite."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cod.data.dataset import CODDataset, build_samples, pair_folder
from cod.evaluation.degrade import MODES
from cod.evaluation.evaluate import evaluate
from cod.inference.predictor import load_checkpoint
from cod.utils import dump_json, get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--extra", nargs=3, action="append", metavar=("NAME", "IMAGE_DIR", "MASK_DIR"), default=[],
                    help="extra dataset (CAMO/CHAMELEON/NC4K...); depth for these must be cached under data.depth_cache/NAME")
    ap.add_argument("--no-depth", action="store_true", help="withhold depth (RGB path of a fusion model)")
    ap.add_argument("--degrade", choices=MODES, help="corrupt the estimated depth before fusion")
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--size", type=int, help="override eval resolution (multiple of 14)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", help="write results JSON here")
    a = ap.parse_args()
    dev = get_device(a.device)
    model, cfg = load_checkpoint(a.checkpoint, dev)
    use_depth = cfg["model"]["fusion"] != "none" and not a.no_depth
    d, size = cfg["data"], a.size or cfg["data"]["image_size"]
    sets = {"cod10k": build_samples(d, a.split, cfg["experiment"]["seed"])[: a.limit]}
    for name, im, mk in a.extra:
        sets[name] = pair_folder(Path(im), Path(mk))[: a.limit]
    results = {}
    for name, samples in sets.items():
        ds = CODDataset(samples, name, size, train=False, use_depth=use_depth, depth_cache=d["depth_cache"])
        results[name] = evaluate(model, ds, dev, use_depth=use_depth, threshold=cfg["eval"]["threshold"],
                                 degradation=a.degrade, severity=a.severity)
        print(name, {k: round(v, 4) for k, v in results[name].items()})
    if a.out:
        dump_json({"checkpoint": a.checkpoint, "experiment": cfg["experiment"]["name"], "fusion": cfg["model"]["fusion"],
                   "depth_used": use_depth, "degrade": a.degrade, "severity": a.severity, "size": size,
                   "split": a.split, "results": results}, a.out)


if __name__ == "__main__":
    main()
