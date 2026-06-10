# Phase 3 — Anomaly map pipeline

Status: **implemented**. Ports UMAD's pixel-space difference metrics and weighted
fusion onto Vista's predict-and-compare output.

## What was built

| File | Role |
|------|------|
| [src/vista_umad/metrics.py](../src/vista_umad/metrics.py) | UMAD difference metrics (Eq. 1–5) as standalone callables |
| [src/vista_umad/fusion.py](../src/vista_umad/fusion.py) | Per-map normalization + weighted fusion |
| [src/vista_umad/pipeline.py](../src/vista_umad/pipeline.py) | `AnomalyMapPipeline` — end-to-end Phase 3 stage |
| [src/vista_umad/image_io.py](../src/vista_umad/image_io.py) | Frame loading + anomaly-map visualization |
| [src/vista_umad/runtime.py](../src/vista_umad/runtime.py) | Shared-server GPU selection + CPU-thread limits |
| [scripts/run_phase3.py](../scripts/run_phase3.py) | Demo / integration check on existing Vista output |
| [tests/test_metrics.py](../tests/test_metrics.py) | 41 unit tests (all passing) |
| [.gitlab-ci.yml](../.gitlab-ci.yml) | Manual CI job template for the edward server |

The metric functions are standalone callables `f(real, pred) -> [H, W]` (the plan's
implementation note); weights are **not** baked in — fusion composes them.

## UMAD equation mapping

Equations are from Bogdoll et al., *UMAD* (BMVC 2024, arXiv:2406.06370).

| UMAD | Implementation | Notes |
|------|----------------|-------|
| Eq. 1 ΔABS | `metrics.abs_error` | mean over RGB of \|real−pred\| |
| Eq. 2 ΔMSE | `metrics.mse_error` | mean over RGB of (real−pred)² |
| Eq. 3 ΔSSIM | `metrics.ssim_difference` | `1 − SSIM` (kornia sliding window), so high = anomalous |
| Eq. 4 ΔPD | `metrics.VGGPerceptualExtractor` | L1 of VGG-16 ImageNet features at `relu1_2/2_2/3_3/4_3`, channel-averaged, upsampled, summed across layers |
| Eq. 5 ΔTD | `metrics.temporal_difference` | mean ΔABS between prior predictions of frame *t* and its own prediction |

`metrics.prediction_variance` is also provided — the multi-seed temporal-uncertainty
variant from the architecture overview — as an alternative `td` signal.

**Fusion**: each map is min-max normalized to `[0, 1]`, then linearly combined with
weights renormalized to sum to 1. Presets in `fusion.PRESETS`: `ssim_pd` and
`mse_ssim_pd_td` are UMAD's reported competitive combinations.

## Running it

The runtime is the Vista virtualenv (has torch/torchvision/kornia):

```bash
# unit tests
external/Vista/.venv/bin/python tests/test_metrics.py

# demo on a Vista rollout (defaults to outputs/vista-sample, rollout 0)
external/Vista/.venv/bin/python scripts/run_phase3.py --rollout 0 --weights ssim_pd
external/Vista/.venv/bin/python scripts/run_phase3.py --weights mse_ssim_pd_td \
    --real outputs/carla-probe/real/images --virtual outputs/carla-probe/virtual/images \
    --tag carla-probe_r00
```

Per rollout the demo writes to `outputs/phase3-demo/<tag>/`: raw float fused maps
(`fused/*.npy`, the Phase 4 input), comparison panels, `scores.csv`, `summary.png`.
The VGG-16 ImageNet weights (~528 MB) download once to the torch hub cache.

Throughput: ~0.2 s/frame at 576×1024 on one A40.

## Resource confinement (shared server — edward, 2× A40)

GPU choice follows the server documentation: the `CUDA_VISIBLE_DEVICES` environment
variable, set before the process starts. `scripts/run_phase3.py` supports both:

```bash
CUDA_VISIBLE_DEVICES=0 external/Vista/.venv/bin/python scripts/run_phase3.py   # first A40
external/Vista/.venv/bin/python scripts/run_phase3.py --gpu 1                  # second A40
```

`--gpu N` simply sets `CUDA_VISIBLE_DEVICES=N` before CUDA initializes; it errors if
it would conflict with an already-set variable. Once the variable is set PyTorch sees
exactly one GPU, addressed as `cuda:0` whichever physical card it is — **the project
never uses more than one GPU**. With more than one GPU still visible the demo prints a
warning. Every run prints the resolved runtime (visible GPUs, device in use, threads).

`--threads N` (default 4) caps the PyTorch CPU thread pool, which otherwise grabs one
thread per core (64 on edward). `--gpu-mem-fraction F` optionally caps GPU memory.

[src/vista_umad/runtime.py](../src/vista_umad/runtime.py) centralizes this:
`resolve_device`, `cap_cpu_threads`, `set_gpu_memory_fraction`, `multi_gpu_warning`,
`describe_runtime`. Resource confinement is an entry-point concern — the library
modules just accept a `device`; `run_phase3.py` (and any future entry point) does the
confining.

### CI

[.gitlab-ci.yml](../.gitlab-ci.yml) is a manual-trigger CI template for edward. It
declares `CUDA_VISIBLE_DEVICES` and the thread caps as job `variables`, so GitLab
exports them into the environment before Python starts — the correct place to confine
a process. Edit `GPU_ID` / `CPU_THREADS`, set the runner tag, and confirm the
interpreter path before use.

## Library notes

- SSIM uses `kornia.metrics.ssim` (kornia 0.6.9). `pytorch-msssim` is not installed
  and is not needed.
- Perceptual difference uses `torchvision.models.vgg16(IMAGENET1K_V1)`; no `lpips`
  dependency.
- `normalize_map` defaults to per-frame min-max (the plan's spec). A consequence:
  the **fused** map's mean is a relative within-frame statistic, not an absolute
  magnitude — every frame's fused map spans `[0, 1]`. For absolute prediction-error
  magnitude use the raw `abs`/`mse` maps. `normalize="none"` / `mode="none"` is
  available if Phase 5 wants global normalization across frames instead.

---

## Findings while reviewing Phases 1–2

These do not block Phase 3 (the pipeline is data-agnostic) but affect what Phase 3
output *means* on the currently bundled data, and should be resolved in Phase 4.

1. **`real` frames in `outputs/` are static repeats, not true futures.** Phases 1–2
   ran Vista's `sample.py` with `--dataset IMG`, which repeats a single seed image
   for all 25 conditioning slots. Verified: within any rollout, `real[0]==real[24]`
   exactly. Vista's `virtual` frames do drift (mean abs diff ≈ 13–26 on a 0–255
   scale). So `diff(real_t, virtual_t)` currently measures *Vista's predicted drift
   from the seed*, not anomaly. A true predict-and-compare needs distinct ground-truth
   future frames — i.e. Phase 4.1's multi-frame dataloader. The Phase 3 demo is
   therefore an **integration check**, not yet a quantitative anomaly result.

2. **Phase 2.3 ("anomaly signal sanity check") cannot be satisfied with current data.**
   It requires `|t − t̂|` against a real future `t`. With static `real` frames there
   is no such comparison. Recommend folding §2.3 into the start of Phase 4 once the
   dataloader exists.

3. **AnoVox generation is incomplete.** The plan asks for 16 scenarios × 200 frames;
   `data/anovox/Outputs/` contains **1** scenario (`Scenario_e2f08f8f…`, 200 frames,
   with RGB / ANOMALY / INSTANCE / SEMANTIC / ACTION / LiDAR all present). The
   remaining 15 scenarios still need to be generated with CARLA before Phase 5.

4. **`carla-probe` ran at 320×512, not Vista's native 576×1024.** `vista-sample` used
   the native 576×1024 (correct). 320×512 is off Vista's training resolution and may
   degrade quality; likely a VRAM trade-off. Worth standardizing on 576×1024 for the
   Phase 4 full inference run, or recording it as a deliberate decision.

## Open decisions still owed by Phase 2.5 (needed before Phase 4)

The plan lists these as Phase 2.5 exit criteria; they are not yet recorded anywhere:
context window length `k`, context source (eval-subset vs raw stream), rollout
length `n`, and whether to use Vista action conditioning. Phase 4.1's dataloader
depends on all four.
