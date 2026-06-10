#!/usr/bin/env python
"""Build the CODA-clips benchmark (see PLAN.md).

Runs the assembly stages against ``configs/coda_clips.yaml``:

    A  index      parse CODA -> scenes_index.json (+ count assertions)   [no downloads]
    BCDE build     resolve + fetch + 10 fps clips + letterbox + manifest
    V  validate    invariants + overlay PNGs

Examples
--------
    # Stage A only (no downloads), assert 1057/309/134:
    python scripts/build_coda_clips.py index

    # ONCE anchors end-to-end (history gated), validate on 50 scenes:
    python scripts/build_coda_clips.py build --sources once --limit 50 && \
        python scripts/build_coda_clips.py validate

    # KITTI full clips (downloads the 12 needed raw drives):
    python scripts/build_coda_clips.py build --sources kitti
"""

from __future__ import annotations

import argparse
import os
import sys

# Make ``src`` importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vista_umad.coda_clips import validate as validate_mod  # noqa: E402
from vista_umad.coda_clips.config import load_config  # noqa: E402
from vista_umad.coda_clips.datasheet import finalize_store  # noqa: E402
from vista_umad.coda_clips.index import build_scene_index, load_scene_index  # noqa: E402
from vista_umad.coda_clips.manifest import build_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["index", "build", "validate", "finalize", "all"])
    p.add_argument("--config", default=None, help="path to coda_clips.yaml")
    p.add_argument("--sources", default=None, help="comma list overriding which sources are enabled")
    p.add_argument("--limit", type=int, default=None, help="cap scenes per source (validation runs)")
    p.add_argument("--no-fetch", action="store_true", help="do not download raw media")
    p.add_argument("--overlays", type=int, default=25, help="number of overlay PNGs in validate")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.sources:
        want = {s.strip() for s in args.sources.split(",")}
        for name, sc in cfg.sources.items():
            sc.enabled = name in want

    if args.stage in ("index", "all"):
        scenes = build_scene_index(cfg, write=True)
        print(f"[index] wrote {len(scenes)} scenes -> {cfg.scenes_index_path}")

    if args.stage in ("build", "all"):
        scenes = load_scene_index(cfg)
        rep = build_manifest(cfg, scenes, fetch=not args.no_fetch, limit=args.limit)
        print(f"[build] {rep.summary()}")
        print(f"[build] manifest -> {cfg.manifest_path}")

    if args.stage in ("validate", "all"):
        validate_mod.check_manifest(cfg)
        validate_mod.render_overlays(cfg, n=args.overlays)

    if args.stage in ("finalize", "all"):
        finalize_store(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
