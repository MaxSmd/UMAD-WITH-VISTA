# CODA-clips benchmark

Turns CODA's single-frame `base-val-1500` corner-case benchmark into a **clip**
benchmark for the VISTA driving world model. Per scene we reconstruct the short clip
leading up to the corner-case (anchor) frame, letterbox to VISTA's 1024×576 / 10 fps
geometry **without cropping**, transform the boxes, and emit a license-clean
manifest + media store. Downstream task: corner-case detection via VISTA prediction
error (score error inside vs. outside the CODA boxes on the predicted anchor frame).

Spec: [PLAN.md](../PLAN.md). Code: [src/vista_umad/coda_clips/](../src/vista_umad/coda_clips/).
Config: [configs/coda_clips.yaml](../configs/coda_clips.yaml). Store: `data/coda-clips/`.

## Pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| A | [index.py](../src/vista_umad/coda_clips/index.py) | Parse CODA COCO + side files → `scenes_index.json`; assert 1057/309/134. |
| B | [sources/](../src/vista_umad/coda_clips/sources/) | Resolve + fetch source clips (once / kitti / nuscenes). |
| C | [sources/base.py](../src/vista_umad/coda_clips/sources/base.py) | Sample each sequence onto the 10 fps grid (context + history pool). |
| D | [letterbox.py](../src/vista_umad/coda_clips/letterbox.py) | Pad-not-crop to 1024×576 + matching box transform + valid mask. |
| E | [manifest.py](../src/vista_umad/coda_clips/manifest.py), [datasheet.py](../src/vista_umad/coda_clips/datasheet.py) | Write frames, `manifest.jsonl`, datasheet, `reconstruct.py`. |
| F | [eval_harness.py](../src/vista_umad/coda_clips/eval_harness.py) | Prediction-error detection: wrap VISTA, masked per-box AUROC/AP. |

## Running

```bash
P="PYTHONPATH=src external/Vista/.venv/bin/python"

# Stage A only — no downloads, asserts the 1057/309/134 composition
$P scripts/build_coda_clips.py index

# ONCE anchors end-to-end (history gated), validate + overlays on 50 scenes
$P scripts/build_coda_clips.py build --sources once
$P scripts/build_coda_clips.py validate

# KITTI full clips — downloads only the 12 needed raw drives (image_02 only)
$P scripts/build_coda_clips.py build --sources kitti

# Finalize the store (datasheet, reconstruct.py, config copy)
$P scripts/build_coda_clips.py finalize
```

`--limit N` caps scenes per source for quick validation; `--no-fetch` skips downloads.

## What each source actually yields

- **ONCE** (1057): anchor pixels ship inside CODA and are **redistributable** → every
  anchor present with no download. The clip *history* needs ONCE's gated per-split cam03
  tars (the 1057 scenes span **350 sequences across all ONCE splits** — no per-sequence
  download), so ONCE clips are anchor-only (`history_unavailable`, `truncated`) until the
  tars are dropped at `once.raw_cache/<sequence_id>/cam03/<timestamp_ms>.jpg`.
- **KITTI** (309): fully reconstructible. `kitti_indices` → object-devkit 1-based mapping
  → raw drive; **12 unique drives over 4 dates**, all 309 map cleanly (no empty/OOB lines).
  We fetch only those drives' `image_02` (~1.5 GB zip each, lidar discarded). KITTI raw is
  10 fps so the grid aligns natively. Aspect caveat: ~1242×375 ⇒ ~45% letterbox bars.
- **nuScenes** (134): built via **targeted CAM_FRONT fetch** — `scripts/fetch_nuscenes_camfront.py`
  pulls only the keyframes (anchors) + surrounding 12 fps sweeps each clip needs (~2.5k
  files / ~350 MB) by HTTP range reads from the HF CAM_FRONT mirror, not the ~400 GB
  blobs. 12→10 fps by nearest-tick; context picks >½ frame off their 100 ms tick are
  flagged `context_off_grid`. 124/134 are runnable (3 off-grid, 7 sequence-start). **In
  VISTA's training set** (`in_vista_train_domain`) and exactly 16:9 (no bars) — its numbers
  are optimistic and are never pooled with ONCE/KITTI. (Requires nuScenes `*_meta.tgz` at
  `nuscenes.dataroot` for the prev/next linkage.)

## Eval harness (Stage F)

Condition VISTA on the 3 letterboxed context frames; the anchor is generated frame index
`n_conds-1 + round(horizon·fps)` (= 12 for 1.0 s, 10 fps, 3 conds). Error map = pixel L2
or VGG-perceptual (reuses [`metrics.py`](../src/vista_umad/metrics.py)). The letterbox
**bars are masked out via `valid_mask` before any aggregation** (else they inflate
background). Per-box mean error vs. background → AUROC/AP via
[`evaluation.py`](../src/vista_umad/evaluation.py), broken out per source with an
ONCE+KITTI macro; K rollouts aggregate as mean / min-over-K / cross-sample variance.

## Validation & tests

- [tests/test_coda_clips.py](../tests/test_coda_clips.py) — letterbox/box transform, the
  10 fps grid + context indices, KITTI mapping, masked per-box scoring, no-pooling.
- `validate` stage checks counts, exactly-3 context spaced ~100 ms, all boxes in frame,
  and writes overlay PNGs (boxes on letterboxed anchors) under `data/coda-clips/overlays/`.

## Gotchas handled (PLAN.md checklist)

1. KITTI mapping is 1-based (`mapping[rand[i]-1]`). 2. Empty mapping lines flagged/skipped.
3. fps spacing verified per source; off-grid scenes flagged. 4. Box transform mirrors the
letterbox exactly. 5. Bars masked out of the error metric. 6. Truncated clips keep the
flag (filter, don't pad). 7. Black bars are OOD (accepted cost; `pad_color` ablatable).
8. nuScenes never pooled into the headline. 9. Counts asserted after Stage A; drop counts
reported in `build_report.json`.
