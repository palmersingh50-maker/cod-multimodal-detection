#!/usr/bin/env bash
# Evaluate every finished ablation run on the COD10K test split (+ RGB-path / degraded-depth probes) and benchmark it.
# Usage: scripts/run_eval_all.sh EPOCHS [LIMIT]
set -uo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-4}"; LIMIT=(${2:+--limit "$2"})
mkdir -p experiments/results
for v in rgb reliability concat cross_attn gated; do
  d="experiments/abl_${v}_e${EPOCHS}"; ck="$d/best.pt"
  [ -f "$ck" ] && [ -f "$d/summary.json" ] || { echo "skip $v (not finished)"; continue; }
  run() { out="$1"; shift; [ -f "experiments/results/$out.json" ] || .venv/bin/python scripts/evaluate.py --checkpoint "$ck" "${LIMIT[@]}" --out "experiments/results/$out.json" "$@" 2>&1 | grep -E "^cod10k|Error"; }
  run "${v}_test"
  if [ "$v" != rgb ]; then
    run "${v}_test_nodepth" --no-depth
    for m in shuffle noise constant; do run "${v}_test_degrade_${m}" --degrade "$m"; done
  fi
  [ -f "experiments/results/${v}_bench.json" ] || .venv/bin/python scripts/benchmark.py --checkpoint "$ck" \
    --image "$HOME/Downloads/COD10K-v3/Test/Image/COD10K-CAM-1-Aquatic-1-BatFish-2.jpg" --out "experiments/results/${v}_bench.json" 2>&1 | grep -E "^(rgb|multi)|Error"
done
