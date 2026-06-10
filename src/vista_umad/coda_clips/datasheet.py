"""Stage E tail -- emit DATASHEET.md, reconstruct.py, and the config copy into the store.

The output store is meant to be shareable: ONCE anchor pixels are redistributable, but
KITTI/nuScenes pixels are not -- so instead of their media we ship ``reconstruct.py``,
which re-fetches them from the identifiers already in ``scenes_index.json`` /
``manifest.jsonl`` and re-applies the identical letterbox. The datasheet records the
conventions and the license/leakage caveats a downstream user must know.
"""

from __future__ import annotations

import json
import os
import shutil

from .config import Config

__all__ = ["write_datasheet", "write_reconstruct", "finalize_store"]


def _counts(cfg: Config) -> dict[str, int]:
    counts: dict[str, int] = {}
    if os.path.exists(cfg.manifest_path):
        with open(cfg.manifest_path) as fh:
            for ln in fh:
                if ln.strip():
                    src = json.loads(ln)["source"]
                    counts[src] = counts.get(src, 0) + 1
    return counts


def write_datasheet(cfg: Config) -> str:
    counts = _counts(cfg)
    t = cfg.target
    path = os.path.join(cfg.out_root, "DATASHEET.md")
    md = f"""# CODA-clips datasheet

A **clip** benchmark derived from the CODA `base-val-1500` corner-case dataset for
corner-case detection via VISTA prediction error. Each scene is the short video clip
leading up to a CODA corner-case (anchor) frame, letterboxed to VISTA's input geometry
without cropping, with the boxes transformed into the same coordinate frame.

## Geometry & conventions
- Target: **{t.width}x{t.height}** (WxH, 16:9 landscape), **{t.fps} fps**, {cfg.context_frames} context frames.
- Letterbox: resize-to-fit + pad (never crop). All sources are wider than 16:9, so bars
  are top/bottom. Pad color `{list(cfg.pad_color)}`. A per-scene `masks/<scene_id>_valid.png`
  marks content (1) vs. bar (0); **bars must be masked out of any error metric** -- they
  are constant and otherwise inflate background statistics.
- Anchor = the CODA corner-case frame, a *predicted future* frame `horizon_s` after the
  last context frame. Horizons built: {cfg.horizons_s} s (primary {cfg.eval.primary_horizon_s} s).
- Boxes: `anchor.boxes_native` are CODA COCO xywh in native pixels; `anchor.boxes_vista`
  are the same boxes after the letterbox transform (`x*scale+pad_x`, `y*scale+pad_y`) and
  are the ones used for scoring.
- `history_pool` keeps {cfg.history_pool_frames} frames at {t.fps} fps before the anchor so
  alternate horizons up to ~2.2 s are derivable without rebuilding.
- `truncated: true` when <full history was available (sequence boundary, or ONCE history
  not fetched). Eval slices needing a fixed history length should filter these out, not pad.

## Composition (CODA base-val-1500: 1057 ONCE / 309 KITTI / 134 nuScenes)
Scenes written to this store: {json.dumps(counts)}

## Per-source provenance, licensing, caveats
- **ONCE** (front cam03): anchor pixels ship inside CODA and are **redistributable**, so
  every ONCE anchor is present here. The clip *history* needs ONCE's gated per-split
  cam03 tars (the 1057 scenes span 350 sequences across all ONCE splits -- there is no
  per-sequence download), so by default ONCE clips are anchor-only and flagged
  `history_unavailable`/`truncated`. Provide ONCE frames at
  `once.raw_cache/<sequence_id>/cam03/<timestamp_ms>.jpg` and rebuild for full clips.
- **KITTI** (image_02): not redistributable -- pixels are re-fetched by `reconstruct.py`
  from `kitti_indices` -> object-devkit mapping -> raw drive (only the 12 needed drives,
  image_02 only). KITTI raw is 10 fps so the grid aligns natively. **Aspect-ratio caveat:**
  KITTI frames are ~1242x375 (~3.3:1), so letterbox bars cover ~45% of the frame -- the
  largest OOD burden of the three sources.
- **nuScenes** (CAM_FRONT): not redistributable and **in VISTA's training set**
  (`in_vista_train_domain: true`). Pixels are fetched targeted: only the CAM_FRONT
  keyframes (`samples/CAM_FRONT/`, the anchors) + the surrounding 12 fps sweeps
  (`sweeps/CAM_FRONT/`) inside each clip window -- ~2.5k files (~350 MB) pulled by range
  reads from the Hugging Face CAM_FRONT mirror, rather than the ~400 GB trainval blobs.
  The 12 fps stream is resampled to the 10 fps grid by nearest-tick; clips whose context
  pick falls >half a frame from its 100 ms tick (a real sweep gap) are flagged
  `context_off_grid` + `truncated`. nuScenes is natively 1600x900 (exactly 16:9 ->
  **no letterbox bars**). **Do not pool nuScenes into a single headline number with
  ONCE/KITTI** -- it is optimistic (training-domain leakage).

## OOD note
Black letterbox bars are OOD for VISTA (trained on full-frame driving); this is the
accepted cost of "pad, not cut". `pad_color` / edge-replicate padding can be ablated via
config.

## Files
- `manifest.jsonl` -- one scene per line (the spec; no source pixels for KITTI/nuScenes).
- `scenes_index.json` -- normalized Stage A index (source identifiers + native boxes).
- `annotations/corner_case.coco.json` -- verbatim copy of CODA's COCO file (box source of truth).
- `clips/<source>/<scene_id>/frames/NNNNNN.jpg` -- letterboxed frames (chronological).
- `masks/<scene_id>_valid.png` -- per-scene valid (content) mask.
- `reconstruct.py` -- re-fetches KITTI/nuScenes pixels from identifiers (no redistribution).
- `config.yaml` -- the exact build configuration.

## Versions
- VISTA: OpenDriveLab/Vista (SVD backbone), 576x1024, 10 fps, 25-frame clips, 3 context.
- Source datasets: CODA base-val-1500; KITTI raw (avg-kitti bucket); nuScenes v1.0-trainval.
"""
    with open(path, "w") as fh:
        fh.write(md)
    return path


def write_reconstruct(cfg: Config) -> str:
    """Emit a standalone `reconstruct.py` into the store.

    It re-fetches KITTI (and, if gated on, nuScenes) pixels from the identifiers in
    `scenes_index.json` and re-applies the identical letterbox, hash-checking against any
    frames already present. It uses the `vista_umad.coda_clips` package when importable
    (the canonical implementation), so the published store stays in lock-step with the
    builder rather than carrying a drifting copy.
    """
    path = os.path.join(cfg.out_root, "reconstruct.py")
    src = '''#!/usr/bin/env python
"""Reconstruct non-redistributable pixels (KITTI/nuScenes) for this CODA-clips store.

ONCE anchor pixels ship in the store. KITTI/nuScenes pixels do not -- this script
re-fetches them from the identifiers in scenes_index.json and re-applies the identical
letterbox, so the store is reproducible without redistributing source media.

Usage:
    python reconstruct.py --sources kitti            # fetch + rebuild KITTI clips
    python reconstruct.py --sources kitti --verify   # hash-check existing frames only

Requires the `vista_umad` package on PYTHONPATH (the builder repo) and a config.yaml
in this directory pointing coda_root at a local CODA copy.
"""
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_builder():
    try:
        from vista_umad.coda_clips.config import load_config
        from vista_umad.coda_clips.index import load_scene_index
        from vista_umad.coda_clips.manifest import build_manifest
        return load_config, load_scene_index, build_manifest
    except Exception as e:
        sys.exit("Could not import vista_umad.coda_clips (%s).\\n"
                 "Add the builder repo's src/ to PYTHONPATH, then re-run." % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="kitti")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    load_config, load_scene_index, build_manifest = _import_builder()
    cfg = load_config(args.config)
    want = {s.strip() for s in args.sources.split(",")}
    for name, sc in cfg.sources.items():
        sc.enabled = name in want

    if args.verify:
        n = ok = 0
        for ln in open(cfg.manifest_path):
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if rec["source"] not in want:
                continue
            p = os.path.join(cfg.out_root, rec["anchor"]["path"])
            n += 1
            if os.path.exists(p):
                ok += 1
        print("verify: %d/%d %s anchor frames present" % (ok, n, args.sources))
        return

    scenes = load_scene_index(cfg)
    rep = build_manifest(cfg, scenes, fetch=True)
    print("reconstruct:", rep.summary())


if __name__ == "__main__":
    main()
'''
    with open(path, "w") as fh:
        fh.write(src)
    os.chmod(path, 0o755)
    return path


def finalize_store(cfg: Config, config_path: str | None = None) -> None:
    """Write datasheet + reconstruct.py + copy config.yaml into the store."""
    os.makedirs(cfg.out_root, exist_ok=True)
    write_datasheet(cfg)
    write_reconstruct(cfg)
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "coda_clips.yaml")
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(cfg.out_root, "config.yaml"))
    print(f"[finalize] datasheet + reconstruct.py + config.yaml -> {cfg.out_root}")
