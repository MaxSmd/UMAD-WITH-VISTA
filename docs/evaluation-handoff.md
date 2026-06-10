# Evaluation handoff — context for the next session

Compressed state + decision space for completing the Vista-UMAD evaluation.
Read this plus [docs/anovox.md](anovox.md), [docs/phase3.md](phase3.md), and
[vista_umad_implementation_plan-3.md](../vista_umad_implementation_plan-3.md)
and you have everything.

---

## Where we are

- **AnoVox dataset**: 16 viable scenarios unified at
  [data/anovox/Outputs/Final_Output_2026_05_26-21_18/](../data/anovox/Outputs/Final_Output_2026_05_26-21_18/)
  (11 native + 5 symlinks from a second batch). Per-town:
  Town01 3 / Town02 2 / Town03 3 / Town04 1 / Town05 2 / Town10HD 5 = **16**.
- **Phase 3 + 4.3 + 5 pipeline code**: complete and unit-tested.
- **Stage A (Vista inference)**: works on **one scenario** via
  [scripts/run_inference_anovox.py](../scripts/run_inference_anovox.py).
- **Stage B (anomaly pipeline)**: works on **one inference output** via
  [scripts/run_anomaly_pipeline.py](../scripts/run_anomaly_pipeline.py).
  Single-scenario test on the original AnoVox scenario produced
  AUROC ≈ 0.50 pooled — see
  [memory/project_single_scenario_eval_findings.md](../../.claude/projects/-home-ge95yem-umad-with-vista/memory/project_single_scenario_eval_findings.md)
  for why (one frame dominated the pool; SAM p99 hurt on small-object anomalies).
- **CARLA / generation processes**: shut down. No background jobs running.

---

## What's missing for headline results

Two wrappers + one launch:

| | Status | What's needed |
|---|---|---|
| Stage A multi-scenario driver | not written | bash or python loop calling [scripts/run_inference_anovox.py](../scripts/run_inference_anovox.py) per scenario dir, with `--scenario` and per-scenario `--out` |
| Stage B multi-scenario aggregator | not written | wrapper calling [scripts/run_anomaly_pipeline.py](../scripts/run_anomaly_pipeline.py) per inference dir, plus a final pooled-metrics step (concat all `scores`/`labels`, compute one project-wide `AnomalyScores`) |
| Stage A launch | pending | ~10 h on GPU 1 (sequential) or ~5 h split across both GPUs (GPU 0 is shared — riskier) |

Stage B itself is fast (~1–2 min per scenario × 16 ≈ 25 min). So once Stage A
is done, full evaluation finishes in well under an hour.

---

## Stage A — the settings, locked in

Defaults in [scripts/run_inference_anovox.py](../scripts/run_inference_anovox.py)
that we deliberately chose:

| Flag | Value | Reason |
|---|---|---|
| `--n-frames` | 25 | Hard-coded by Vista's architecture (25 temporal slots) |
| `--n-conds` | 3 | Vista's max conditioning slots; using fewer wastes capacity |
| `--rollout-stride` | 12 | Strict ΔTD requirement: must be < 22 (predicted span). 12 = ~2 predictions per evaluable frame. Lower = exponential compute. |
| `--height 512 --width 832` | 512×832 | Native (576×1024) OOMs at 48 GB. 832×512 ≈ 72 % of native pixels, fits in ~31 GB with low-vram. |
| `--low-vram` | on | Without it Vista doesn't fit at any usable resolution on a 48 GB GPU |
| `--n-steps 50`, `--cfg-scale 2.5`, `--seed 23` | Vista defaults | Kept unchanged |
| action conditioning | **off** | AnoVox is CARLA; Vista was trained on nuScenes; distribution mismatch risk. Defensible to revisit (see Ablations). |

**Output per scenario**:
`outputs/anovox-inference/<scenario_tag>/{real, pred/rollout_NN, manifest.json}`.

Wall-clock per scenario from previous single-scenario run: **~37 min**
(16 rollouts × ~2 min 20 s).

---

## Stage B — the settings, locked in

Defaults in [scripts/run_anomaly_pipeline.py](../scripts/run_anomaly_pipeline.py):

| Flag | Value | Reason |
|---|---|---|
| `--weights` | `mse_ssim_pd_td` | UMAD's reported best 4-metric combination |
| `--refine` | `sam` | UMAD's mask-level refinement (Phase 4.3) |
| `--sam-model-type` | `vit_b` | smallest SAM checkpoint (~358 MB), best speed |
| `--aggregate` | `p99` | robust 99th-percentile-within-mask. **Caveat: hurt single-scenario AUROC on small-object anomalies. Worth comparing `max` and `none`.** |
| `--eval-stride` | 10 | UMAD's every-10th-frame evaluation protocol |
| current pooling | **per scenario** | needs extension to pool across all 16 scenarios for the headline number |

---

## Ablations worth running (Stage B is cheap)

| Fusion weights | Refinement | Why |
|---|---|---|
| `mse_ssim_pd_td` | `sam --aggregate p99` | Headline candidate; UMAD-style |
| `mse_ssim_pd_td` | `sam --aggregate max` | p99 hurt on small objects in 1-scenario test |
| `mse_ssim_pd_td` | `none` | baseline (raw pixel maps) |
| `ssim_pd` | `none` | UMAD's other reported combination, lighter |

Stage B takes ~25 min per ablation across 16 scenarios. Running all 4 takes
roughly an evening.

---

## Optional re-runs that could improve numbers

| Change | Cost | Possible benefit |
|---|---|---|
| Enable action conditioning (Vista's `speed`/`angle`/`goal`/`trajectory`) | Re-run Stage A (~10 h) | Sharper predictions if CARLA actions don't break Vista's nuScenes-trained head. Worth an ablation. |
| Native resolution 576×1024 on an H100 (no `--low-vram`) | requires bigger GPU | Restores Vista's intended quality; throughput ~30 % faster |
| Lower Town04 `npc_vehicle_amount` to 100 and regenerate Town04 | ~15 min | Brings Town04 from 1 → 2-3 scenarios for better per-town balance |
| Multi-seed prediction variance (alternative ΔTD signal) | 3-5× Stage A compute | Different anomaly signal; UMAD specifies cross-rollout ΔTD, this is the variant |

---

## Environment & gotchas

- **GPUs**: edward server has 2× A40 (48 GB each). Pin Vista to GPU 1 with
  `--gpu 1`. GPU 0 is shared — another user often has 5–20 GB resident there
  and ~70–90 % compute utilization. The standing rule is *"if both GPUs are
  maxed out, stop"* — GPU 0 alone is not a stop signal.
- **Venv**: [external/Vista/.venv/](../external/Vista/.venv/) (Python 3.9.25,
  torch 2.0.1+cu118, kornia, segment-anything). All scripts in this project
  use this venv. Invocation: `external/Vista/.venv/bin/python scripts/...`.
- **AnoVox venv** (only relevant if regenerating scenarios):
  [external/anovox/.venv/](../external/anovox/.venv/) (Python 3.8.20, carla 0.9.14).
  Different venv on purpose — carla wheel is cp38-only.
- **CARLA libvulkan workaround**: must export
  `LD_LIBRARY_PATH=$PWD/external/lib/extracted/usr/lib/x86_64-linux-gnu`
  before `external/carla/CarlaUE4.sh` (system has no libvulkan). Already
  baked into [/tmp/anovox_run_carla.sh](../external/anovox/) if it still
  exists; otherwise re-create from [docs/anovox.md §7](anovox.md).
- **SAM checkpoint**: [data/sam-checkpoints/sam_vit_b_01ec64.pth](../data/sam-checkpoints/)
  (375 MB) already downloaded.
- **Detach-safe long jobs**: wrap in tmux session, optionally also nohup.
  `tmux new -d -s <name>; tmux send-keys -t <name> '<cmd>' C-m`. Verified
  to survive laptop close + SSH disconnect (PPID → 1).

---

## Quick recipes

```bash
# Stage A on ONE scenario (the existing single-scenario script, for spot-checking)
external/Vista/.venv/bin/python scripts/run_inference_anovox.py \
    --gpu 1 --low-vram --height 512 --width 832 --n-conds 3 --rollout-stride 12 \
    --scenario data/anovox/Outputs/Final_Output_2026_05_26-21_18/Scenario_<UUID>

# Stage B on ONE inference output (the existing single-scenario script)
PYTHONPATH=src external/Vista/.venv/bin/python scripts/run_anomaly_pipeline.py \
    --inference-dir outputs/anovox-inference/Scenario_<tag> \
    --gpu 1 --weights mse_ssim_pd_td --refine sam --aggregate p99

# Inventory: count viable scenarios in the unified dir
for s in data/anovox/Outputs/Final_Output_2026_05_26-21_18/Scenario_*-*; do
  n=$(ls "$s"/RGB-CAM*/*.png 2>/dev/null | wc -l)
  [ "$n" -ge 200 ] && echo "$(basename "$s")"
done | wc -l   # → 16
```

---

## First task for the next session

Write the **Stage A multi-scenario wrapper**. Roughly:

```bash
#!/bin/bash
# scripts/run_stage_a_all.sh  (or similar)
SCENS_DIR=data/anovox/Outputs/Final_Output_2026_05_26-21_18
OUT_BASE=outputs/anovox-inference
for s in $SCENS_DIR/Scenario_*-*; do
    [ -d "$s" ] || continue
    n=$(ls "$s"/RGB-CAM*/*.png 2>/dev/null | wc -l)
    [ "$n" -ge 200 ] || continue            # skip failed/partial
    tag=$(basename "$s" | cut -d- -f1)      # Scenario_<8 hex>
    out=$OUT_BASE/$tag
    [ -f "$out/manifest.json" ] && continue  # already done
    external/Vista/.venv/bin/python scripts/run_inference_anovox.py \
        --gpu 1 --low-vram --height 512 --width 832 \
        --n-conds 3 --rollout-stride 12 \
        --scenario "$s" --out "$OUT_BASE"
done
```

Launch in tmux, then walk away for ~10 hours.

---

## Standing user preferences

- Default to single-GPU sequential execution unless explicitly told to
  parallelize.
- *"If both GPUs are maxed out, stop"* — applies to any GPU-consuming launch.
- Long jobs go in tmux so they survive laptop close + reconnect.
- Save data analysis findings to memory, not transient files.
- Terse responses; no trailing summaries when work is shown by diff/output.
