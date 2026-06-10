# Vista-UMAD: Unsupervised Video Anomaly Detection for Autonomous Driving

**Project goal.** Adapt the UMAD approach (Bogdoll et al., 2024) by replacing the MUVO world model with Vista (Gao et al., NeurIPS 2024) and evaluating on the AnoVox benchmark.

**Core idea.** UMAD's pipeline is pixel-space throughout: difference metrics between a model's frame estimate and the real frame, optionally refined by instance masks. Whether the model estimate comes from "reconstruct current" (MUVO) or "predict next" (Vista) is a framing choice, not an algorithmic one. We use Vista in predict-and-compare mode: condition on frames `t-k … t-1`, predict frame `t`, compare against the real frame `t` using UMAD's difference metrics (ABS, MSE, SSIM, perceptual), then refine with unsupervised instance masks.

**Why AnoVox.** It is the exact benchmark UMAD evaluated on, so reported numbers from the paper are a direct reference point. It provides multimodal sensor data plus ego-vehicle action data (steering, speed) — useful because Vista supports action-conditioned generation, opening a configuration MUVO-UMAD couldn't exploit. AnoVox also contains temporal behavioral anomalies in addition to static ones; UMAD's paper only evaluated on the static subset, so there is room to extend the evaluation.

---

## Architecture overview

```
Frames t-k … t-1  ──► Vista ──► predicted frame t̂
    (+ actions)              │
                                      │
Real frame t  ───────────────────────► │ ──► difference metrics
                                      │      (ABS, MSE, SSIM, perceptual)
                                      ▼
                              pixel-wise anomaly map
                                      │
Real frame t  ──► U2Seg / SAM ──► instance masks ──► mask-level refinement
                                      ▼
                          per-instance anomaly scores
```

Temporal-difference branch (from UMAD): generate multiple predictions for the same target frame `t` (different noise seeds), compute pixelwise variance, fuse with the visual-difference map via weighted sum.

---

## Phase overview

| Phase | Title | Goal |
|-------|-------|------|
| 1 | Vista standup | Confirm Vista runs end-to-end on its native nuScenes data |
| 2 | AnoVox generation + Vista CARLA probe | Generate the evaluation subset, test whether Vista can predict CARLA video, decide on extraction strategy. Go/no-go gate. |
| 3 | Anomaly map pipeline | Port UMAD difference metrics + weighted fusion onto Vista output |
| 4 | Full inference + mask refinement | Dataloader, full inference loop, U2Seg / SAM mask aggregation |
| 5 | Evaluation | Run AP / FPR95 / AUROC across the test scenarios |
| 6 | Writeup | Results table, qualitative analysis, temporal-anomaly extension |

---

## Phase 1 — Vista standup

Goal: get Vista running on its native data distribution. No AnoVox yet, no anomaly logic yet. Just confirm the environment, checkpoints, and inference work.

### 1.1 Hardware reality check

Vista is an SVD-XT finetune at 576×1024 resolution. Plan for:

- **GPU**: at minimum 24 GB VRAM for default sampling (e.g. A5000 / 3090 / 4090). A100 40 GB+ is much more comfortable, especially once Phase 2 starts multi-sample temporal-uncertainty runs.
- **Disk**: ~50 GB for checkpoints (`svd_xt.safetensors` ~9 GB, `vista.safetensors`, nuScenes mini split for testing, plus the generated AnoVox subset in Phase 2).
- **CUDA**: ≥ 11.8 with matching PyTorch build.

If only smaller GPUs are available, plan for `xformers` attention and CPU offload from the start.

### 1.2 Repository and environment

```bash
git clone https://github.com/OpenDriveLab/Vista.git
cd Vista
conda create -n vista python=3.9 -y
conda activate vista
pip install -r requirements.txt
pip install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
```

Then read `docs/INSTALL.md` end-to-end. The `sdata` submodule install above is the most common point of failure.

### 1.3 Checkpoints

Place two files in `ckpts/`:

- `svd_xt.safetensors` from `stabilityai/stable-video-diffusion-img2vid-xt` on Hugging Face (SVD-XT base weights, required because Vista's checkpoint only contains the finetuned delta).
- `vista.safetensors` from `OpenDriveLab/Vista` on Hugging Face.

### 1.4 First inference — sanity check on nuScenes

Run `sample.py` unmodified on a few nuScenes mini-split frames. Place the translated action JSONs from the Vista repo's `annos/` directory in the right location. Goal: confirm Vista produces visually coherent future-frame predictions from a real nuScenes context clip.

Common failure modes (most → least likely):
1. Missing `svd_xt.safetensors` — Vista loads SVD as backbone.
2. `xformers` / PyTorch version mismatch.
3. Action-annotation JSONs not in `annos/`.

### 1.5 Phase 1 exit criteria

- [ ] `sample.py` runs end-to-end on nuScenes mini split.
- [ ] Vista is callable from a Python script (not just CLI), returning predicted frames as tensors.
- [ ] Predicted frames are visually plausible compared to ground truth nuScenes continuations.

---

## Phase 2 — AnoVox generation + Vista CARLA probe

This is the project's go/no-go gate. Generate the AnoVox evaluation subset, then test whether Vista — trained on real-world nuScenes and OpenDV-YouTube — can produce coherent predictions on CARLA-rendered imagery at all. If it can't, the rest of the project is at risk.

### 2.1 Generate AnoVox evaluation subset

AnoVox provides a CARLA-based generation framework. Following UMAD's protocol, generate 16 abnormal driving scenarios with 200 frames each, sampling every 10th frame for evaluation. Vary towns, weather, and time of day. Include static anomalies for the main evaluation; optionally generate a temporal-anomaly subset for the Phase 6 extension.

Recorded modalities to retain:
- Front camera (required, the only modality Vista uses).
- Panoptic segmentation maps (for ground truth).
- Ego-vehicle actions (steering, throttle) — needed if §2.3 decides to use Vista's action conditioning.
- LiDAR — optional, not used in the base pipeline, but worth keeping for potential extensions.

Also generate a few short clips with no anomalies (10–20 frames each) for the cross-domain probe in §2.2.

### 2.2 Cross-domain probe — can Vista predict CARLA video?

The largest open risk in the project: Vista was trained on real-world driving footage, AnoVox is CARLA-rendered. Take the anomaly-free clips from §2.1 and feed them to Vista in predict-and-compare mode (frames `t-k … t-1` in, predict `t`, compare to real `t`). The goal is *not* good output — it is to learn:

- Does Vista crash, produce garbage, or produce something coherent on CARLA-rendered imagery?
- What does the failure mode look like? (Color shift, texture artifacts, hallucinated real-world details?)
- Is the failure consistent across the clip (suggesting systematic domain shift) or variable (suggesting random noise we can average out via multi-sampling)?

**Go/no-go decision**: if Vista produces unusable output on CARLA, that is itself a finding. Possible pivots:
- Train a Vista-on-CARLA finetune (significant scope expansion, but tractable with the CARLA generation framework).
- Switch benchmark to a real-world video dataset (e.g. LostAndFound — would require a separate plan).
- Abandon the project as scoped.

Do not proceed to §2.3 without an explicit call on this.

### 2.3 Anomaly signal sanity check

On a small AnoVox clip containing a known static anomaly, run predict-and-compare end-to-end: feed Vista frames `t-k … t-1`, run the full diffusion sampling process to produce predicted frame `t̂`, compute `|t - t̂|`. Inspect the difference map qualitatively. Does it light up on the anomaly more than on normal moving objects?

This is a sanity check, not an ablation. The full metric comparisons happen in Phase 5. The only purpose here is to confirm that the predict-and-compare pipeline produces *some* visible anomaly signal before investing in Phase 3.

**Expected complication**: moving cars and pedestrians will always have prediction error — that is noise to suppress in Phase 3 via the perceptual difference (semantic, less position-sensitive) and the mask-level refinement in Phase 4.

### 2.4 Prediction horizon

Multi-frame rollout: condition once on `t-k … t-1`, predict frames `t, t+1, …, t+n`, evaluate each against ground truth. One Vista call covers multiple evaluation frames, which is the cheapest option.

Known tradeoff: prediction error compounds with rollout depth, so frames late in the rollout will have elevated baseline error independent of any anomaly. Accepted for the first run; revisit if Phase 5 results look suspicious (e.g. anomaly scores correlating with rollout position rather than anomaly content).

ΔTD is computed naturally from this setup: prior predictions of frame `t` made from earlier rollouts (starting at `t-1`, `t-2`, etc.) are compared to the prediction of `t` from its own rollout, per UMAD eq. 5.

### 2.5 Phase 2 exit criteria

- [ ] AnoVox subset generated and stored: 16 scenarios × 200 frames + anomaly-free probe clips.
- [ ] §2.2 produces a defensible go/no-go decision on Vista output quality for CARLA imagery.
- [ ] §2.3 confirms predict-and-compare produces visible anomaly signal on at least one clip.
- [ ] Documented decision on context window length `k` (likely 3 or 5 frames given Vista's training setup).
- [ ] Documented decision on context source (eval-frame subset vs. raw video stream — see Phase 4.1).
- [ ] Decision on whether to use Vista's action conditioning (AnoVox provides ego-actions; Vista supports them; CARLA actions may differ in distribution from nuScenes, so this needs empirical testing).

---

## Phase 3 — Anomaly map pipeline

Port UMAD's four pixel-space difference metrics to operate on Vista's predicted frame `t̂` vs. real frame `t`:

- **Absolute error** ΔABS — mean absolute difference across RGB channels.
- **Squared error** ΔMSE — mean squared difference across RGB channels.
- **SSIM** — sliding-window structural similarity. Use `kornia.losses.ssim_loss` or `pytorch-msssim` for GPU implementation.
- **Perceptual difference** ΔPD — L1 distance between VGG-ImageNet feature maps of real and predicted frame, summed across layers as in UMAD eq. 4.
- **Temporal difference** ΔTD — mean absolute error between prior predictions of frame `t` (from rollouts starting at earlier context windows) and the current prediction of `t`. UMAD's eq. 5 formulation, computed naturally from the overlapping rollouts produced by Phase 2.4's setup.

**Weighted fusion**. Normalize each map to `[0, 1]`, then linearly combine with weights summing to 1. UMAD reports `(SSIM, PD)` and `(MSE, SSIM, PD, TD)` as competitive combinations to start from.

**Implementation note**: keep all metric implementations as standalone callables `f(real, pred) -> [H, W]` so the fusion module can compose them arbitrarily. Don't bake weights into the metric functions.

---

## Phase 4 — Full inference + mask refinement

### 4.1 Dataloader

The unit of work is a *rollout*, not a single evaluation frame. For each rollout:
- Return `k` context frames as Vista conditioning input.
- Return the `n` target frames following the context for ground-truth comparison.
- Return the panoptic anomaly masks for all `n` target frames.
- Return ego-actions for the window if action conditioning is in use.

Open decision (Phase 2.5): context frames are sampled either from the every-10th evaluation subset (UMAD-style, ~1s gaps between context frames) or from the raw video stream immediately preceding the rollout start (Vista-native, ~0.1s gaps). The latter is likely closer to Vista's training distribution; the former matches UMAD's setup directly.

### 4.2 Full inference loop

Run Vista in rollout mode across the test scenarios. Save:
- Predicted frames `t̂, t̂+1, …, t̂+n` per rollout (for debugging, may be discarded).
- Pixel-wise anomaly maps per evaluation frame (raw float, from Phase 3 fusion).
- Per-frame metadata (scenario ID, frame index, rollout depth at which this frame was predicted, inference time).

Expected scale: 16 scenarios × ~20 evaluation frames per scenario ≈ 320 frames covered by roughly 320/n Vista calls (n = rollout length). With n=10 that's ~32 Vista calls at ~30 seconds each on an A100 — under 20 minutes of pure inference. Checkpoint the loop anyway.

### 4.3 Mask-level refinement

Run U2Seg (unsupervised, matches UMAD's primary configuration) and SAM (supervised baseline, matches UMAD's strongest configuration) on each real frame `t`. For each generated mask, aggregate the pixel-wise anomaly scores into a per-instance score using the mean.

**Expected domain-shift issue**: U2Seg's released checkpoint was trained on ImageNet, not on driving imagery. UMAD acknowledges this as a limitation on CARLA imagery specifically. Expect SAM to be the stronger configuration but U2Seg to be the meaningful unsupervised result.

---

## Phase 5 — Evaluation

Use the AnoVox evaluation framework with the metrics UMAD reports:

- **Average Precision (AP)**.
- **FPR95** — False Positive Rate at 95% True Positive Rate.
- **AUROC** — Area under the Receiver Operating Characteristic curve.

Compute per-scenario and aggregated numbers. Report both the unsupervised configuration (Vista + U2Seg) and the SAM configuration.

---

## Phase 6 — Writeup

Results table mirroring UMAD's reporting structure, plus qualitative comparison figures showing: input frame, Vista prediction, difference map, mask-refined anomaly map, ground truth.

**Stretch contribution**: evaluate on AnoVox's temporal-anomaly subset, which UMAD did not. Because Vista is a video predictor, the temporal-difference branch (Phase 3, ΔTD) is principled for behavioral anomalies in a way MUVO's frame-reconstruction setup wasn't. If results are positive, this is the strongest argument for the Vista substitution.

---

## Risks and open decisions

| Risk | Mitigation |
|------|-----------|
| Vista produces poor output on CARLA imagery (real→synthetic shift) | §2.2 is the explicit go/no-go checkpoint; budget time to inspect output qualitatively before committing to Phase 3 |
| Vista produces high prediction error on normal moving objects, drowning the anomaly signal | §2.3 quantifies this; perceptual difference and mask refinement should help |
| U2Seg's ImageNet checkpoint produces poor masks on CARLA imagery | Compare U2Seg and SAM in Phase 4; report both honestly; same limitation UMAD reported |
| Rollout drift inflates baseline error at deeper rollout positions | Accepted for the first run; if Phase 5 shows scores correlating with rollout depth rather than anomaly content, revisit with per-depth normalization or shorter rollouts |
| Action conditioning helps in nuScenes but hurts on CARLA actions (different distribution) | Decide in §2.4 based on §2.3 output; reasonable to skip action conditioning if it degrades quality |

---

## Open decisions (resolve as you go)

- **Context window length `k`**: 3, 5, or 7 frames? Decide in Phase 2.4.
- **Context source**: every-10th evaluation subset (UMAD-style) or raw video stream (Vista-native)? Decide in Phase 2.5.
- **Rollout length `n`**: how many frames forward to predict per Vista call? Decide in Phase 2.4 based on sanity-check output quality.
- **Action conditioning**: use AnoVox ego-actions as Vista input, or skip? Decide in Phase 2.5.
- **Temporal anomaly extension**: include in main results, or keep as a clearly-marked extension in Phase 6? Decide after Phase 5 results.

---

## References

- Bogdoll et al., *UMAD: Unsupervised Mask-Level Anomaly Detection for Autonomous Driving*, BMVC 2024.
- Bogdoll et al., *AnoVox: A Benchmark for Multimodal Anomaly Detection in Autonomous Driving*, arXiv:2405.07865, 2024.
- Gao et al., *Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability*, NeurIPS 2024. [Code](https://github.com/OpenDriveLab/Vista)
- Bogdoll et al., *MUVO: A Multimodal Generative World Model for Autonomous Driving*, arXiv:2311.11762, 2023.
- Niu et al., *Unsupervised Universal Image Segmentation* (U2Seg), CVPR 2024.
- Kirillov et al., *Segment Anything*, 2023.
- Dosovitskiy et al., *CARLA: An Open Urban Driving Simulator*, CoRL 2017.
