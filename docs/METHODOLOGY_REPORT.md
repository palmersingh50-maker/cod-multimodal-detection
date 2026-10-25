# Methodology Report
Multimodal Camouflaged Object Detection (COD): RGB plus estimated depth

## 1. Objective
Build a system that finds objects that blend into their surroundings and outlines them pixel by pixel, with a web application for uploading photos. The research question: does adding depth predicted from the image itself, fused in a reliability-aware way, improve segmentation over an RGB-only model at acceptable cost?

The depth is **estimated by a neural network from the same RGB image**. It is not sensor depth and not ground truth, so it adds no new measurements of the scene.

## 2. Technology

| Layer | Technology |
|---|---|
| Deep learning | Python 3.14, PyTorch 2.14 on Apple M4 GPU (MPS) |
| RGB backbone | DINOv2 ViT-S/14 (`facebook/dinov2-small`) via Hugging Face Transformers |
| Depth estimation | Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`) |
| Image processing | OpenCV, Pillow, NumPy, SciPy |
| Augmentation | Albumentations |
| Experiment tracking | TensorBoard, JSON/YAML run files |
| Metrics | Custom implementations, checked against PySODMetrics |
| Backend | FastAPI, Uvicorn |
| Frontend | React 18 + TypeScript, Vite |
| Export | ONNX / onnxruntime |
| Testing | pytest (42 tests) |
| Deployment / hosting | Dockerfile (written, not built), GitHub, Vercel (results dashboard) |

All training ran on one MacBook (Apple M4, 24 GB). No cloud GPU was used.

## 3. Methodology

### 3.1 Data
COD10K v3, camouflaged images only: 3,040 official training images and 2,026 official test images. Images without a camouflaged object are excluded. Because COD10K has no validation split, 304 training images (10 %, seeded shuffle) were held out for validation, leaving 2,736 for training.

Dataset checks found no corrupt files, no empty masks and no image/mask size mismatches. A perceptual-hash scan found 1 near-duplicate train/validation pair and 11 train/test pairs. These come from the official split and may slightly flatter every model equally. CAMO, CHAMELEON and NC4K were not downloaded, so no results on them exist.

### 3.2 Depth pipeline
Depth Anything V2 predicts relative depth at 518 px, resized back to the image size so it stays aligned with the photo. Each map is clipped to its 1st–99th percentile and scaled to 0–1. Constant or invalid outputs are stored as zeros and flagged as invalid. Maps for all 5,066 images were cached to disk once, so depth is never recomputed during training or evaluation.

### 3.3 Model
1. **RGB branch:** DINOv2 features taken from four depths of the network and projected to 128 channels.
2. **Depth branch:** a small CNN over three cues (depth, depth edges, validity flag), producing features on the same grid as the RGB features.
3. **Fusion:** four alternatives, each added to the RGB features so RGB always remains usable:
   - *Concat:* join RGB and depth features and mix them.
   - *Cross-attention:* RGB features query the depth features through a gate that starts closed.
   - *Gated:* same, with a learned per-feature gate.
   - *Reliability-aware (proposed):* the gated output is scaled by a predicted per-location reliability of the depth, which drops to zero when depth is invalid.
4. **Decoder:** top-down multiscale decoder with an image-resolution stem for thin structures, producing a full-size mask.

Every model can also run with depth switched off.

### 3.4 Training
AdamW; learning rate 1e-5 for the backbone and 1e-4 for the new layers; weight decay 1e-4; one warm-up epoch then cosine decay; gradient clipping 1.0; loss = binary cross-entropy + Dice. Input 392 × 392, batch size 4, seed 42. The best checkpoint is chosen by validation S-measure. The trainer also supports resume, early stopping, non-finite-loss detection and progressive unfreezing of the backbone.

### 3.5 Evaluation
Predictions are resized to each ground-truth mask's original size. Metrics: S-measure, weighted F-measure, E-measure, MAE, mIoU and Dice (threshold 0.5). S, weighted-F, E and MAE agree with PySODMetrics to within 6e-4. Extra tests probe depth reliance: depth removed, shuffled between images, noisy, or constant. Efficiency is measured as parameters, checkpoint size and latency.

### 3.6 Application
A FastAPI backend validates each upload (format, size, dimensions), runs the model and returns the mask, an overlay and timings. A React/TypeScript page supports drag-and-drop upload, model and threshold choice, an overlay-opacity slider, side-by-side original/overlay/mask and downloads. The server loads the models once and refuses to answer if no trained model is loaded.

### 3.7 Experimental budget
The original brief (ViT-B, 518 px, 30–50 epochs) would take days on this laptop. The experiments therefore use **ViT-S/14, 392 px, 4 epochs, one training run per model**. This limits how much the results can show.

## 4. Results

### 4.1 Dataset
| Split | Images | Use |
|---|---|---|
| Train | 2,736 | training |
| Validation (held out from train) | 304 | checkpoint selection |
| Test (official) | 2,026 | final evaluation |

### 4.2 Training
Validation S-measure after each of the 4 epochs, and training loss at the last epoch:

| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Final train loss | Params (M) | Train time (min) |
|---|---|---|---|---|---|---|---|
| RGB-only | 0.7255 | 0.8058 | 0.8396 | 0.8481 | 0.342 | 23.14 | 36 |
| Reliability-aware | 0.7469 | 0.7984 | 0.8360 | 0.8517 | 0.336 | 23.71 | 40 |
| Concat | 0.7591 | 0.8201 | 0.8542 | 0.8603 | 0.323 | 23.41 | 41 |
| Cross-attention | 0.7334 | 0.8114 | 0.8388 | 0.8524 | 0.330 | 23.48 | 53 |
| Gated | 0.7587 | 0.8067 | 0.8412 | 0.8489 | 0.339 | 23.61 | 58 |

All five were still improving at epoch 4, so they are under-trained. Training times were longer for the later runs partly because other jobs ran on the machine at the same time.

### 4.3 Test set (COD10K, all 2,026 images)

| Model | S↑ | Fβw↑ | E↑ | MAE↓ | mIoU↑ | Dice↑ |
|---|---|---|---|---|---|---|
| RGB-only | 0.8398 | 0.6624 | 0.9104 | 0.0328 | 0.6904 | 0.7898 |
| Reliability-aware | 0.8428 | 0.6675 | 0.9086 | 0.0324 | 0.6925 | 0.7908 |
| Concat | 0.8467 | 0.6840 | 0.9117 | 0.0308 | 0.6942 | 0.7923 |
| Cross-attention | 0.8437 | 0.6741 | 0.9129 | 0.0313 | 0.6927 | 0.7918 |
| Gated | 0.8417 | 0.6680 | 0.9102 | 0.0324 | 0.6907 | 0.7898 |

### 4.4 Depth reliance (S-measure, 500-image evenly spaced test subset)

| Model | Depth removed | Shuffled depth | Noisy depth | Constant depth |
|---|---|---|---|---|
| Reliability-aware | 0.8469 | 0.8428 | 0.8459 | 0.8447 |
| Concat | 0.8433 | 0.8368 | 0.8476 | 0.8447 |
| Cross-attention | 0.8497 | 0.8497 | 0.8497 | 0.8497 |
| Gated | 0.8436 | 0.8439 | 0.8439 | 0.8439 |

These use a subset, so they are not directly comparable with the table in 4.3.

### 4.5 Efficiency (Apple M4, one image at a time)

| Setting | Time per image |
|---|---|
| RGB-only | 43 ms |
| Multimodal, depth already cached | 48–49 ms |
| Multimodal, depth computed fresh | 116–120 ms |

Fusion adds 0.3–0.6 M parameters. Depth Anything V2 is a separate network on top of that.

## 5. Discussion
- **Depth fusion did not show a real improvement.** The best model, concat, beats the RGB-only baseline by 0.007 S-measure. With one run per model and 4 epochs, that is not distinguishable from run-to-run variation. Neither hypothesis (depth helps; reliability weighting helps) is supported by this pilot.
- **Cross-attention and gated models ignore depth.** Their scores barely change whether depth is real, removed, shuffled, noisy or constant, so they behave as RGB-only models.
- **Reliability-aware is also mostly RGB-driven,** and is not better than the simpler variants.
- **Concat is the only model that visibly uses depth,** and it gets worse when the depth is wrong.
- **Cost:** fusion itself is cheap (about 5 ms with cached depth). Estimating depth is what makes multimodal inference about 2.7 times slower.
- **Deployed models:** the app uses the RGB-only checkpoint for "RGB only" and the concat checkpoint for the depth modes.

## 6. Limitations
Small backbone, short training, one run per model, COD10K only, and depth that comes from the same photo. The models also outline the most object-like region even when nothing is camouflaged, and their probabilities are not calibrated. A fair test of depth would need longer training, several runs, depth dropout during training, and the larger model on a GPU. The related-work list in `REPORT.md` is a starting point only, and no novelty is claimed.

## 7. Where things are
Code in `cod/`, `scripts/`, `frontend/`, `dashboard/`; full tables in `docs/RESULTS.md`; detailed report in `docs/REPORT.md`; GitHub repository `palmersingh50-maker/cod-multimodal-detection`.
