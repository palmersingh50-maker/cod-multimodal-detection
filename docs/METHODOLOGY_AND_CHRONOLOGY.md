# Methodology and Chronology Report
Multimodal Camouflaged Object Detection (COD) — RGB + estimated depth

Written 25 Oct 2026 while the fusion ablation was still training. Numbers shown here are only ones that had been measured at that time; the final test-set table is generated into [`RESULTS.md`](RESULTS.md) and is not repeated here.

---

## 1. What the project is
A system that finds objects that blend into their surroundings and outlines them pixel by pixel, plus a web application where a user uploads a photo and sees the detected object. It also asks one research question: *does adding depth predicted from the image itself, fused in a reliability-aware way, improve segmentation over an RGB-only model, at acceptable cost?*

Important framing: the depth is **estimated by a neural network from the same RGB image**. It is not sensor depth and not ground truth, so any gain would come from the depth network's learned priors, not from new information about the scene.

## 2. Technology used

| Layer | Technology | Role |
|---|---|---|
| Language / DL | Python 3.14, PyTorch 2.14 (MPS backend on Apple M4) | models, training, inference |
| RGB backbone | DINOv2 ViT-S/14 (`facebook/dinov2-small`, Hugging Face Transformers 5.20) | visual features from blocks 3, 6, 9, 12 |
| Depth | Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) | monocular depth estimates, cached |
| Image handling | OpenCV, Pillow, NumPy, SciPy | resizing, masks, distance transforms |
| Augmentation | Albumentations 2.0 | flips, affine, colour jitter, blur/noise |
| Tracking | TensorBoard, JSON/YAML run files | losses, metrics, configs, hardware |
| Metrics | custom NumPy/SciPy implementations, cross-checked against PySODMetrics 1.6.2 | S, weighted-F, E, MAE, mIoU, Dice |
| Backend | FastAPI 0.162, Uvicorn, python-multipart | inference API |
| Frontend | React 18 + TypeScript 5.6, Vite 5 | upload / visualise UI |
| Export | ONNX 1.23 + onnxruntime | portable model, parity-checked |
| Testing | pytest, FastAPI TestClient | 42 unit/integration tests |
| Packaging | Dockerfile, docker-compose (written, never built — no Docker on the machine) | deployment |
| Source control | git, GitHub CLI (`gh`), GitHub (private repo) | versioning |
| Hosting | Vercel (via its MCP connector and CLI) | static results dashboard |
| Dataset | COD10K v3 (camouflaged subset) | training / validation / test |

Hardware: Apple M4, 24 GB unified memory, no NVIDIA GPU.

## 3. Methodology

### 3.1 Data
* COD10K v3, camouflaged images only (`COD10K-CAM-*`): 3,040 official train images and 2,026 official test images. NonCAM images (empty masks) are excluded.
* COD10K has no official validation split, so 10 % of train (304 images) was held out with a seeded shuffle (seed 42); 2,736 remain for training.
* Validation script (`scripts/check_dataset.py`): 0 corrupt files, 0 empty masks, image/mask sizes consistent. A perceptual-hash scan found 1 near-duplicate train/val pair and 11 train/test pairs — a property of the official split, and a small leakage risk for every model equally.
* CAMO, CHAMELEON and NC4K loaders exist but the datasets were not downloaded, so no results on them exist.

### 3.2 Depth pipeline
Depth Anything V2 predicts relative inverse depth at 518 px, resized back to the image size so it stays pixel-aligned. Values are percentile-clipped (1–99) and min-max normalised per image. Constant or non-finite outputs are stored as zeros and flagged invalid. Maps are cached as 16-bit PNGs (about 10 images/s on the M4), so depth is never recomputed during training or evaluation; at inference an in-memory cache keyed by image hash avoids repeat work.

### 3.3 Model
1. **RGB branch:** DINOv2 tokens from four depths of the network, each normalised and projected to 128 channels.
2. **Depth branch:** a small CNN over three cues (depth, depth-gradient magnitude, validity flag) producing features on the same token grid.
3. **Fusion** (applied to the RGB token stream at selected levels), four alternatives:
   * `concat` — concatenate and project, added residually;
   * `cross_attn` — RGB queries attend to depth keys/values, with a tanh gate initialised at zero;
   * `gated` — a learned per-token, per-channel sigmoid gate on the attention output;
   * `reliability` — the gated attention scaled by a predicted per-token depth reliability (computed from RGB, depth, their disagreement and the depth cues; forced to zero when depth is invalid). This is the proposed variant and is an untested hypothesis until results say otherwise.
   All variants start at or near the identity, and every model can run with depth withheld.
4. **Decoder:** top-down multiscale decoder over a feature pyramid (1/4 to 1/32), plus a convolutional stem on the raw image at 1/2 and 1/4 resolution for thin structures, producing full-resolution logits; an optional boundary head exists (off by default).

### 3.4 Training
AdamW; backbone LR 1e-5, decoder/fusion LR 1e-4; weight decay 1e-4; 1 warm-up epoch then cosine decay; gradient clipping 1.0; loss = BCE-with-logits + Dice. The trainer supports seeds, resume, best-checkpoint selection (by validation S-measure), early stopping, non-finite-loss detection, progressive backbone unfreezing, gradient accumulation and bf16 autocast on CUDA (fp32 on MPS).

### 3.5 Evaluation
Predictions are resized to each ground-truth mask's original size. Metrics: S-measure, weighted F, mean E-measure (256 thresholds), MAE, mIoU and Dice (threshold 0.5). Robustness probes: depth withheld; depth replaced by another image's depth ("shuffle"); Gaussian noise; constant depth. Efficiency: parameters, checkpoint size, latency with RGB only / cached depth / fresh depth.

### 3.6 Experimental budget (important caveat)
The brief's setting (ViT-B at 518 px, 30–50 epochs) would take days on this machine. The ablation therefore uses **ViT-S/14, 392 px, 4 epochs, batch size 4, one seed per variant**. Models are under-trained, differences of a few thousandths may be inside run-to-run noise, and no hyper-parameter tuning or confidence intervals were done. Backbone-size, resolution and boundary-branch ablations are supported in code but were not run.

### 3.7 Application
FastAPI loads models once, validates uploads by content (JPEG/PNG/WEBP/BMP, size and dimension caps, EXIF orientation), runs inference in a worker thread under a lock, and returns base64 mask, overlay and probability PNGs with per-stage timings. Errors use one JSON format with no paths or stack traces. The React UI supports drag-and-drop, model and threshold selection, overlay-opacity slider, side-by-side original/overlay/mask, and downloads. It returns an error (503) rather than fake output if no checkpoint is loaded.

## 4. Chronology (25 Oct 2026, IST)

| Time | Step |
|---|---|
| ~13:30 | Inspected environment: M4 / 24 GB, Python 3.14, no Docker, COD10K already in Downloads. Created virtualenv; installed PyTorch (MPS available) and the rest of the stack. |
| 13:35 | Wrote config system, dataset loader, depth estimator; started caching depth for all 5,066 images in the background (~8 min). |
| 13:36–13:45 | Wrote backbone, depth encoder, four fusion modules, decoder, full model, losses, metrics, trainer, evaluation and degradation tools. Forward/backward smoke test passed for all five variants. |
| 13:40–13:50 | Fixed bugs found by testing: MPS does not support non-divisible adaptive pooling (replaced by antialiased resize); weighted-F had an inverted `min` (0.47 error → 1e-5 after fix, verified against PySODMetrics). |
| 13:47 | Started the RGB baseline training run; first epoch took ~9 min (val S-measure 0.7255). |
| 13:50–14:00 | Wrote inference module, FastAPI backend, 42 tests (all passing), dataset checker, benchmark, ONNX export (RGB and multimodal parity ≈ 4e-7; antialiased resize not exportable → documented 6e-4 deviation). |
| 14:00 | Built the React/TypeScript frontend; tested upload→detect→overlay end to end in the browser against a real early checkpoint. |
| 14:04 | Wrote README, research report, status page. |
| 14:07–14:30 | Built a static results dashboard; connected Vercel, deployed it; created a local git repo; installed GitHub CLI, you authenticated; created the private GitHub repo and pushed; linked Vercel to GitHub (needed a Login Connection you added) so pushes redeploy. |
| 14:36 | RGB baseline finished 4 epochs (val S-measure 0.7255 → 0.8058 → 0.8396 → 0.8481). Exported `checkpoints/rgb.pt`, launched the app (`run_app.sh`), verified it segments a green parrot hidden in foliage. |
| 14:45 | Started an unattended finisher job: waits for all training, runs evaluation, builds the results table and dashboard data, installs the best fusion checkpoint, runs tests, pushes to GitHub. |
| (pending) | Reliability, concat, cross-attention and gated runs train sequentially (~35–40 min each). |

## 5. Problems met and how they were handled
* **Vercel page not loading on your machine** — the site returned HTTP 200 from the dev Mac and the deployment was READY; the timeout was a network path issue on the client side.
* **GitHub MCP unavailable** — none existed in the connector registry, so GitHub CLI with a browser device-code login was used instead.
* **Vercel connector 403** on a personal scope — worked around with the Vercel CLI after a device login.
* **Premature "training finished" instruction** — checked actual files first and reported that only one of five runs was done, rather than producing an empty table.

## 6. Current status and honest limitations
* Complete: data validation, depth pipeline, RGB baseline and four fusion models implemented, training/evaluation framework, metrics (verified), tests (42 passing), API, UI, ONNX export, docs, git/GitHub/Vercel wiring, working local app.
* In progress: remaining fusion training runs, then test-set evaluation (automated).
* Not done: CAMO / CHAMELEON / NC4K evaluation; ViT-B and 518 px runs; multi-seed statistics; backbone/resolution/boundary ablations; Docker build; a systematic literature review (the related-work list in `REPORT.md` is a starting point, and no novelty is claimed).
* Known model limits: trained on COD10K only; it will outline the most object-like region even when nothing is camouflaged; probabilities are uncalibrated.

## 7. Repository map
`cod/` package (data, depth, models, training, evaluation, inference, api) · `configs/` · `scripts/` · `frontend/` · `dashboard/` · `tests/` · `docs/` · `run_app.sh` · `README.md`.
GitHub: private repo `palmersingh50-maker/cod-multimodal-detection`. Dashboard: `cod-dashboard` on Vercel.

## 8. References (starting list — verify before publication)
Fan et al., *Camouflaged Object Detection*, CVPR 2020 (COD10K) · Le et al., *Anabranch Network for Camouflaged Object Segmentation*, CVIU 2019 (CAMO) · Lv et al., CVPR 2021 (NC4K) · Oquab et al., *DINOv2*, TMLR 2024 · Yang et al., *Depth Anything V2*, NeurIPS 2024 · Ranftl et al., *Vision Transformers for Dense Prediction*, ICCV 2021 · Alayrac et al., *Flamingo*, NeurIPS 2022 · Fan et al., S-measure (ICCV 2017) and E-measure (IJCAI 2018) · Margolin et al., weighted F-measure (CVPR 2014).
