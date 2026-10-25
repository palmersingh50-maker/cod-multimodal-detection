#!/usr/bin/env bash
# Sequentially train the fusion ablation under one shared, budget-limited protocol.
# Usage: scripts/run_ablation.sh [EPOCHS] [extra overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-4}"; shift || true
for v in rgb reliability concat cross_attn gated; do
  name="abl_${v}_e${EPOCHS}"
  mkdir -p "experiments/$name"
  [ -f "experiments/$name/summary.json" ] && { echo "skip $name (done)"; continue; }
  .venv/bin/python scripts/train.py --config "configs/vits_${v}.yaml" --resume \
    experiment.name="$name" train.epochs="$EPOCHS" "$@" 2>&1 | grep --line-buffered -v -E "Loading weights|Warning|warn" | tr '\r' '\n' | grep --line-buffered -E "^(train_|resumed|early|WARNING)|Error|Traceback" >> "experiments/$name/train.log"
done
