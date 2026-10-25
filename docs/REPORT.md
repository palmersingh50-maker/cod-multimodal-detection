# Reliability-aware RGB + estimated-depth fusion for camouflaged object detection — working report

Status: **methodology and implementation complete; experimental results are generated into
[`RESULTS.md`](RESULTS.md) by `scripts/make_report.py` and are not restated here.** Anything below that is not
tied to a number in RESULTS.md is a design rationale or a hypothesis, not a finding.

## 1. Problem and hypothesis
Camouflaged object detection (COD) segments objects whose appearance matches their background. Hypothesis (H1):
monocular depth *estimated* from the same RGB image adds a complementary cue (surface discontinuities, figure/ground
layering) that a fusion module can exploit; H2: gating that cue by a learned reliability makes the model robust when
the estimate is wrong. Both are tested against an RGB-only model of identical architecture. Because the depth network
sees only the RGB image, it adds no new sensor information — any gain would come from its learned priors
(trained on other data), and that caveat applies to every claim here.

## 2. Related work (starting points — verify before publication)
* COD10K / SINet: Fan et al., *Camouflaged Object Detection*, CVPR 2020. CAMO: Le et al., *Anabranch Network for
  Camouflaged Object Segmentation*, CVIU 2019. NC4K: Lv et al., *Simultaneously Localize, Segment and Rank the
  Camouflaged Objects*, CVPR 2021. CHAMELEON: Skurowski et al., *Animal camouflage analysis: Chameleon database* (2018).
* DINOv2: Oquab et al., TMLR 2024 (arXiv:2304.07193). Depth Anything V2: Yang et al., NeurIPS 2024 (arXiv:2406.09414).
  DPT-style multiscale decoding: Ranftl et al., ICCV 2021.
* Gated cross-attention with a zero-initialised tanh gate: Alayrac et al., *Flamingo*, NeurIPS 2022.
* Structure loss: Wei et al., *F3Net*, AAAI 2020. Metrics: S-measure (Fan et al., ICCV 2017), E-measure (Fan et al.,
  IJCAI 2018), weighted F (Margolin et al., CVPR 2014).
* Depth-assisted COD has prior work; **this repo has not performed a systematic literature search, so no novelty is claimed.**
  A proper survey of RGB-D/depth-aided COD must precede any publication.

## 3. Data
COD10K v3, camouflaged subset: 3 040 train images → 2 736 train / 304 val (seeded 90/10 hold-out), 2 026 test
(official). `scripts/check_dataset.py` found 0 corrupt files, 0 empty masks, image/mask sizes consistent. A 64-bit dHash
scan (Hamming ≤ 2) found **1 near-duplicate train/val pair and 11 train/test pairs** — the latter is a property of the
official split and a small leakage risk to keep in mind. Depth maps are Depth-Anything-V2-Small predictions at 518 px,
resized to the image, percentile-clipped (1–99) and min-max normalised per image, stored as uint16 PNG; constant or
non-finite outputs are stored as zeros and flagged invalid. Only COD10K is available locally; CAMO, CHAMELEON and NC4K
loaders exist (`--extra`) but **no results on them exist yet**.

## 4. Model
DINOv2 (HF weights) → tokens from blocks {3,6,9,12} (ViT-S) → per-level norm+1×1 projection (D=128) →
[optional fusion at selected levels] → pyramid (H/4…H/32) top-down decoder + convolutional stem at H/2,H/4 →
full-resolution logits (+ optional boundary head). Depth path: 3-channel cues (depth, |∇depth|, validity) →
4-layer CNN → token grid → per-level projection.

| `model.fusion` | Operation on RGB tokens *r*, depth tokens *d* | Status |
|---|---|---|
| `none` | – | RGB-only baseline |
| `concat` | *r* + MLP([LN r; LN d]) (last layer zero-init) | established-style |
| `cross_attn` | *r* + tanh(α)·MHA(q=r, kv=d), α=0 at init | Flamingo-style gate |
| `gated` | *r* + σ(W[r;a])·a, bias −2 | |
| `reliability` | *r* + ρ·σ(W[r;a])·a, ρ=σ(MLP([r;d;\|r−d\|;cues]))·valid | **the proposed variant** (modification, untested claim) |

All attention variants start (near) identical to the RGB path, and the model runs depth-free (`depth=None`) so RGB
inference stays available. `depth_dropout` can randomly invalidate depth during training. Loss: BCE + Dice
(weights in config); structure and boundary losses are implemented but **off** by default and unvalidated.

## 5. Evaluation protocol
Predictions are bilinearly resized to the original GT size; S-measure, weighted-F, mean E-measure (256 thresholds),
MAE, mIoU and Dice (threshold 0.5, per-image means). S, Fβw, E and MAE were cross-checked against PySODMetrics 1.6.2 on
10 COD10K test masks with synthetic predictions: max |Δ| = 6e-4 (S, ddof difference), 1e-5 (Fβw), 0 (E), 4e-6 (MAE);
the cross-check also runs as a unit test. Robustness probes (`--degrade`): shuffled (other image's depth), Gaussian
noise, constant, inverted, blurred, quantised depth; and `--no-depth` for the RGB path of fusion models.

## 6. Experimental budget and honesty notes
Hardware: Apple M4 (24 GB, MPS). The brief's 30–50 epoch ViT-B/518 setting would take on the order of days here, so the
ablation uses **ViT-S/14, 392 px, 4 epochs, batch 4, seed 42, one run per variant** (`scripts/run_ablation.sh 4`).
Consequences: models are under-trained; with one seed, differences of a few thousandths are likely inside run-to-run
noise; no hyper-parameter tuning or multi-seed confidence intervals were done. Ablations for backbone size, input
resolution and the boundary branch are supported by configuration but **not run**. Treat RESULTS.md as a pilot study.

## 7. Efficiency
`scripts/benchmark.py` reports parameters, checkpoint size, RGB-only latency, multimodal latency with cached and with
freshly estimated depth (including preprocessing, depth, model, postprocessing) and memory. On MPS, memory is the
allocator's current reading, not a true peak. No real-time claim is made.

## 8. Limitations / failure cases
* Estimated depth can be confidently wrong on camouflaged scenes, exactly where it would matter; the shuffled/constant
  depth probes quantify sensitivity but not all failure modes. Qualitative failure analysis is not yet written.
* Train/test near-duplicates (§3) may inflate all models equally.
* The ONNX export differs slightly (non-antialiased resize); Docker and CUDA paths are untested here.
* Probabilities are uncalibrated; no confidence/uncertainty output is provided.

## 9. Future work
Longer training and ViT-B/518 on a cloud GPU; multiple seeds; CAMO/CHAMELEON/NC4K generalisation; boundary/structure
loss ablations; true-depth (sensor) experiments if aligned RGB-D COD data is obtained; systematic literature review.

## 10. Findings (from the measured tables in RESULTS.md; single seed, 4 epochs, ViT-S/392)
* **No evidence that depth fusion helps.** Full COD10K test (N=2026) S-measure: RGB-only 0.8398; reliability 0.8428; concat 0.8467; cross-attention 0.8437; gated 0.8417. The largest gap to the baseline is 0.007 (concat). With one seed and 4 epochs this is not distinguishable from run-to-run noise, so H1 is **not supported** by this pilot, and H2 (reliability helps) is **not supported** either: reliability-aware is not better than the simpler variants.
* **Cross-attention and gated models ignore depth.** Their scores are identical to 3–4 decimals whether depth is real, withheld, shuffled, noisy or constant (e.g. cross-attention S = 0.8497 in all four conditions; gated 0.8436–0.8439). The learned gate stayed (near) closed — these models behave as RGB-only models with extra parameters. Their primary-table differences from the baseline therefore reflect training variance, not depth.
* **Reliability-aware is nearly insensitive to depth** (S 0.8469 withheld vs 0.8428–0.8459 under degradation, N=500 subset), i.e. also mostly RGB-driven.
* **Concat is the only variant that visibly uses depth:** shuffled depth lowers it (S 0.8368 vs 0.8433 withheld), which also shows it can be hurt by wrong depth.
* **Subset caveat:** the depth-withheld/degraded tables use an evenly strided 500-image subset, so they are not directly comparable with the full-test table.
* **Cost (M4/MPS, batch 1, clean run, nothing else running):** RGB-only 43 ms; multimodal with cached depth 48–49 ms; with fresh depth estimation 116–120 ms. Fusion adds 0.3–0.6 M parameters; Depth Anything V2-S adds its own network on top. The efficiency table was re-run after the first benchmark was found to be contaminated by concurrent training.
* **Deployed models:** the app serves the RGB-only checkpoint for "RGB only" and the concat checkpoint (highest full-test S, +0.007 over baseline, not significant) for the multimodal modes.
* **What would settle the question:** longer training (the baseline itself was still improving at epoch 4), several seeds, depth dropout during training so the network is forced to use depth, and ViT-B/518 on a GPU.
