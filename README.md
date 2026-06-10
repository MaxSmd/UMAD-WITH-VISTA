# Vista-UMAD

Unsupervised video anomaly detection for autonomous driving: the UMAD pipeline
(Bogdoll et al., 2024) with the MUVO world model replaced by **Vista** (Gao et al.,
NeurIPS 2024), evaluated on the **AnoVox** benchmark.

See [vista_umad_implementation_plan-3.md](vista_umad_implementation_plan-3.md) for the
full project plan.

## Layout

```
src/vista_umad/      Phase 3 anomaly-map pipeline (metrics, fusion, pipeline, io)
src/vista_umad/coda_clips/   CODA-clips benchmark builder (see docs/coda-clips.md)
scripts/             Runnable entry points (run_phase3.py, build_coda_clips.py)
configs/             Build configs (coda_clips.yaml)
tests/               Unit tests
docs/                Per-phase notes (docs/phase3.md, docs/coda-clips.md)
external/Vista/      Vista world model (upstream clone, .venv = project runtime)
external/anovox/     AnoVox benchmark generation framework
data/                Checkpoints, nuScenes samples, AnoVox output, CARLA probe seeds
data/coda-clips/     CODA-clips benchmark store (manifest, clips, masks, datasheet)
outputs/             Vista sample/probe output and phase3-demo results
```

## Status

| Phase | State |
|-------|-------|
| 1 — Vista standup | done (via `--dataset IMG` single-image mode; see docs/phase3.md) |
| 2 — AnoVox gen + CARLA probe | partial — 1 of 16 scenarios generated; probe done |
| 3 — Anomaly map pipeline | **done** — see [docs/phase3.md](docs/phase3.md) |
| 4 — Full inference + mask refinement | not started |
| 5 — Evaluation | not started |
| 6 — Writeup | not started |
| CODA-clips benchmark | **built** — 1057 ONCE + 309 KITTI scenes; nuScenes gated. See [docs/coda-clips.md](docs/coda-clips.md) |

## Runtime

All code runs in the Vista virtualenv (torch 2.0.1+cu118, torchvision, kornia):

```bash
external/Vista/.venv/bin/python tests/test_metrics.py
external/Vista/.venv/bin/python scripts/run_phase3.py --rollout 0 --weights ssim_pd
```

### Shared-server GPU selection (edward, 2× A40)

Pick the GPU via `CUDA_VISIBLE_DEVICES` (per the server docs) or the `--gpu` flag —
the project confines itself to **one** GPU and a capped CPU-thread pool:

```bash
CUDA_VISIBLE_DEVICES=0 external/Vista/.venv/bin/python scripts/run_phase3.py   # first A40
external/Vista/.venv/bin/python scripts/run_phase3.py --gpu 1 --threads 4      # second A40
```

See [docs/phase3.md](docs/phase3.md#resource-confinement-shared-server--edward-2-a40)
and [.gitlab-ci.yml](.gitlab-ci.yml) (manual CI job template).
