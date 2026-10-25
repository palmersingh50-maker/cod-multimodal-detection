#!/usr/bin/env bash
# Evaluate every finished ablation run on COD10K test and benchmark it.
# Primary table: full test split. Robustness probes: evenly strided subset of PROBE_LIMIT images.
# Usage: scripts/run_eval_all.sh EPOCHS [PROBE_LIMIT]
set -uo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-4}"; PL="${2:-500}"
IMG="$HOME/Downloads/COD10K-v3/Test/Image/COD10K-CAM-1-Aquatic-1-BatFish-2.jpg"
mkdir -p experiments/results
for v in rgb reliability concat cross_attn gated; do
  d="experiments/abl_${v}_e${EPOCHS}"; ck="$d/best.pt"
  [ -f "$ck" ] && [ -f "$d/summary.json" ] || { echo "skip $v (not finished)"; continue; }
  run() { out="$1"; shift; [ -f "experiments/results/$out.json" ] || .venv/bin/python scripts/evaluate.py --checkpoint "$ck" --out "experiments/results/$out.json" "$@" 2>&1 | grep -E "^cod10k|Error"; }
  run "${v}_test"
  if [ "$v" != rgb ]; then
    run "${v}_test_nodepth" --no-depth --limit "$PL"
    for m in shuffle noise constant; do run "${v}_test_degrade_${m}" --degrade "$m" --limit "$PL"; done
  fi
  [ -f "experiments/results/${v}_bench.json" ] || .venv/bin/python scripts/benchmark.py --checkpoint "$ck" \
    --image "$IMG" --out "experiments/results/${v}_bench.json" 2>&1 | grep -E "^(rgb|multi)|Error"
done
