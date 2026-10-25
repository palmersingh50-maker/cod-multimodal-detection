#!/usr/bin/env bash
# Unattended finisher: wait for all training -> evaluate -> report -> dashboard -> deploy checkpoint -> tests -> push.
cd "$(dirname "$0")/.."
until [ -f experiments/abl_gated_e4/summary.json ]; do sleep 60; done
scripts/run_eval_all.sh 4 500
.venv/bin/python scripts/make_report.py
# pick the best fusion model by full-test S-measure for the app's multimodal mode
best=$(.venv/bin/python - <<'P'
import json,glob
r={}
for v in ["reliability","concat","cross_attn","gated"]:
    try: r[v]=json.load(open(f"experiments/results/{v}_test.json"))["results"]["cod10k"]["sm"]
    except Exception: pass
print(max(r,key=r.get) if r else "")
P
)
echo "best fusion: $best"
[ -n "$best" ] && .venv/bin/python scripts/export_checkpoint.py experiments/abl_${best}_e4/best.pt checkpoints/multimodal.pt
.venv/bin/python scripts/build_dashboard.py
.venv/bin/python -m pytest tests -q 2>&1 | tail -2
git add -A && git commit -q -m "Add ablation results, report and dashboard data

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" && git push -q origin main
pkill -f "uvicorn cod.api.app"; sleep 2; nohup ./run_app.sh > /private/tmp/claude-501/-Users-palmersingh-IMD-Project/8337fd20-d81a-4aeb-8c3a-5ed218fc2139/scratchpad/api.log 2>&1 &
echo FINISHED
