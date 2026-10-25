# Multimodal Camouflaged Object Detection (COD)

Pixel-level segmentation of camouflaged objects with a **DINOv2** RGB branch, optional **estimated depth**
(Depth Anything V2) and four RGB↔depth fusion strategies, plus a FastAPI backend and a React/TypeScript UI.

> **Research question:** can reliability-aware fusion of RGB features and *estimated* monocular depth improve COD
> over a strong RGB-only foundation-model baseline without unacceptable overhead?
> Depth here is **predicted from the RGB image by a network** — it is not sensor depth and not ground truth.
> Measured results live in [`docs/RESULTS.md`](docs/RESULTS.md) (auto-generated) and are discussed, with caveats,
> in [`docs/REPORT.md`](docs/REPORT.md). Nothing in this repo claims state-of-the-art performance.

## Layout

```
cod/                 python package
  data/              COD10K + folder datasets, augmentation, depth-aligned samples
  depth/             Depth Anything V2 wrapper, robust normalisation, uint16 PNG cache
  models/            DINOv2 backbone, depth encoder, fusion modules, multiscale decoder, CODModel
  training/          losses, trainer (resume, early stop, NaN guard, warmup+cosine, progressive unfreezing)
  evaluation/        S/Fβw/E/MAE/mIoU/Dice, evaluation loop, depth degradations
  inference/         Predictor (per-stage timings, depth memoisation), overlay/mask helpers
  api/               FastAPI app + settings (env-driven)
configs/             YAML experiment configs (`_base_` inheritance, dotted CLI overrides)
scripts/             train / evaluate / predict / benchmark / precompute_depth / check_dataset / export_onnx / ...
frontend/            Vite + React + TypeScript UI
tests/               pytest suite (metrics, data, model, losses, inference, API)
docs/                REPORT.md, RESULTS.md (generated), STATUS.md
```

## Setup

Tested: macOS 15 arm64 (M4, 24 GB), Python 3.14, PyTorch 2.14 (MPS). Node ≥ 20 for the UI.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # add requirements-dev.txt for tests
cd frontend && npm install && cd ..
```

Device selection is automatic (`cuda` → `mps` → `cpu`; override with `--device` / `COD_DEVICE`).
On CUDA install the matching torch wheel from pytorch.org first; training uses bf16 autocast there
(`train.amp: auto`) and fp32 on MPS/CPU. Pretrained weights (`facebook/dinov2-*`,
`depth-anything/Depth-Anything-V2-*-hf`) download from Hugging Face on first use.

## Data

COD10K v3 must be downloaded by you from the [official project page](https://dengpingfan.github.io/pages/COD/)
and its usage terms followed; set `data.root` (default `~/Downloads/COD10K-v3`). Only the camouflaged images
(`COD10K-CAM-*`; 3 040 train / 2 026 test) are used; NonCAM images are excluded. COD10K has no official
validation split, so 10 % of train is held out with a seeded shuffle. CAMO, CHAMELEON and NC4K are *not* bundled or
downloaded: pass them to `scripts/evaluate.py --extra NAME IMAGE_DIR MASK_DIR` after obtaining them from their
authors (and cache depth for them under `data.depth_cache/NAME`).

```bash
python scripts/check_dataset.py             # pairing, corrupt files, empty masks, cross-split near-duplicates
python scripts/precompute_depth.py          # estimated-depth cache (uint16 PNG, ~10 img/s on M4)
```

## Train / evaluate / infer

```bash
python scripts/train.py --config configs/vits_rgb.yaml train.epochs=30            # RGB-only baseline
python scripts/train.py --config configs/vits_reliability.yaml train.epochs=30    # reliability-aware fusion
python scripts/train.py --config configs/vits_gated.yaml model.boundary_head=true loss.boundary=0.5
python scripts/train.py --config configs/vits_rgb.yaml --resume                   # resume from experiments/<name>/last.pt
python scripts/run_ablation.sh 4 && scripts/run_eval_all.sh 4 && python scripts/make_report.py

python scripts/evaluate.py --checkpoint experiments/NAME/best.pt --out res.json   # [--no-depth] [--degrade shuffle|noise|...]
python scripts/predict.py image.jpg --checkpoint checkpoints/mm.pt                # mask + overlay PNGs
python scripts/benchmark.py --checkpoint ... --image ...                          # params / latency / memory
python scripts/export_checkpoint.py experiments/NAME/best.pt checkpoints/mm.pt    # strip optimiser state
python scripts/export_onnx.py --checkpoint checkpoints/mm.pt --out mm.onnx        # + onnxruntime parity check
tensorboard --logdir experiments
```

Fusion types (`model.fusion`): `none` (RGB only), `concat`, `cross_attn`, `gated`, `reliability`.
Every run writes `config.yaml`, `env.json` (hardware), `history.json`, `summary.json`, `last.pt`, `best.pt`
(selected by `train.val_metric`) under `experiments/<name>/`. Starting values for all hyper-parameters follow the
project brief (AdamW, backbone 1e-5 / head 1e-4, wd 1e-4, warmup + cosine, BCE + Dice, grad-clip 1.0); they are
configuration, not tuned optima.

## Run the application

```bash
export COD_RGB_CHECKPOINT=checkpoints/rgb.pt COD_MM_CHECKPOINT=checkpoints/mm.pt   # either or both (.env.example)
uvicorn cod.api.app:create_app --factory --port 8000
cd frontend && npm run dev            # http://localhost:5173 (proxies /api to :8000)
# or: (cd frontend && npm run build) && COD_FRONTEND_DIR=frontend/dist uvicorn cod.api.app:create_app --factory
```

* `GET /health`, `GET /api/v1/status` → `{ready, device, modes}`
* `POST /api/v1/predict` multipart `file`, `mode` ∈ `rgb|multimodal|both`, `threshold`, `opacity` →
  base64 PNG `mask_png`, `overlay_png`, `probability_png` + per-stage `timings_ms` per mode.
* Errors are always `{"error": {"code", "message"}}`; no paths or stack traces leave the server.
* The server refuses to start predicting without a checkpoint (503) — there are no placeholder predictions.
  `rgb` mode with only a fusion checkpoint loaded runs that model with depth withheld.
* Safeguards: content-based format check (JPEG/PNG/WEBP/BMP), size cap (`COD_MAX_UPLOAD_MB`), pixel/dimension caps,
  EXIF-orientation normalisation, nothing written to disk, uploads not retained, CORS allow-list from env.
* Models load once per worker; inference is serialised by a lock. Run one uvicorn worker per GPU.

## Docker / CUDA

`docker compose up --build` builds the UI and serves everything on :8000 with `./checkpoints` mounted read-only
(CPU image; **not exercised in this repo's development environment — Docker was not installed**). For NVIDIA GPUs use a
CUDA base image and CUDA torch wheels, plus the NVIDIA container runtime. Full training of ViT-B at 518 px is
intended for a cloud GPU; copy the resulting `best.pt` through `scripts/export_checkpoint.py` into `checkpoints/`.

## Tests

```bash
pip install -r requirements-dev.txt pysodmetrics   # pysodmetrics enables the metric cross-check test
pytest -q
```

## Troubleshooting

* *`Adaptive pool MPS … not implemented`* — fixed in code (antialiased bilinear is used); avoid adaptive pooling on MPS.
* *`cached depth maps missing`* — run `scripts/precompute_depth.py` for the splits/dataset you train or evaluate on.
* *ONNX export* uses a non-antialiased token-grid resize (documented deviation ≈ 6e-4 max prob. difference).
* *MPS memory* — lower `train.batch_size`, raise `train.accum_steps`, or reduce `data.image_size`.
* *API says `model_unavailable`* — check `COD_*_CHECKPOINT` paths and the server log.

## Known limitations

See [`docs/REPORT.md`](docs/REPORT.md). Notably: short training budget and a single seed (differences between
fusion variants may be within noise), ViT-S backbone at 392 px rather than ViT-B at 518 px, only COD10K evaluated
locally, memory figures on MPS are allocated-memory readings not true peaks, Docker untested.
