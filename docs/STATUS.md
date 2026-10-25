# Implementation status

| Area | State |
|---|---|
| Dataset validation, splits, depth cache | done (COD10K only) |
| RGB baseline, depth pipeline, 4 fusion variants, boundary head | done, unit-tested |
| Trainer (resume, early stop, NaN guard, progressive unfreeze, TB) | done, smoke-tested |
| Metrics | done, cross-checked vs PySODMetrics |
| FastAPI + React UI | done; verified in browser against a real (epoch-1) checkpoint |
| ONNX export | done, parity-checked on random-init tiny models (RGB and multimodal) |
| Ablation training (4 epochs × 5 variants) | **running in background** → `experiments/abl_*_e4/` |
| Test-set evaluation, degradation probes, benchmarks | queued: `scripts/run_eval_all.sh 4` |
| CAMO / CHAMELEON / NC4K results | not run (data not downloaded) |
| ViT-B/518, resolution & boundary ablations, multi-seed | not run (compute) |
| Docker image | written, never built (no Docker here) |

**Next action:** when `experiments/abl_*_e4/summary.json` exist for all five variants:
`scripts/run_eval_all.sh 4 && python scripts/make_report.py`, then export the chosen checkpoints with
`scripts/export_checkpoint.py` into `checkpoints/` and read `docs/RESULTS.md` before writing any conclusions.
