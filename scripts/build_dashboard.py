"""Collect real experiment outputs into dashboard/data.json (static site; nothing is invented, missing = null)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["rgb", "reliability", "concat", "cross_attn", "gated"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


models = {}
for v in ORDER:
    run = next(iter(sorted((ROOT / "experiments").glob(f"abl_{v}_e*"))), None)
    R = ROOT / "experiments" / "results"
    models[v] = {
        "history": load(run / "history.json") if run else None,
        "summary": load(run / "summary.json") if run else None,
        "test": (load(R / f"{v}_test.json") or {}).get("results", {}).get("cod10k"),
        "test_nodepth": (load(R / f"{v}_test_nodepth.json") or {}).get("results", {}).get("cod10k"),
        "degrade": {m: (load(R / f"{v}_test_degrade_{m}.json") or {}).get("results", {}).get("cod10k") for m in ("shuffle", "noise", "constant")},
        "bench": load(R / f"{v}_bench.json"),
    }
(ROOT / "dashboard" / "data.json").write_text(json.dumps({"models": models, "note": "single seed, 4-epoch ViT-S/392 pilot"}, default=str))
print("wrote dashboard/data.json")
