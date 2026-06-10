"""Validation suite for the CODA-clips store (PLAN.md "Validation suite to write").

Checks the invariants that silently break: per-source counts, exactly-3 context frames
spaced ~100 ms, every ``boxes_vista`` inside the frame, and renders overlay PNGs (boxes
drawn on the letterboxed anchor) for a random sample so the box transform is verifiable
by eye.
"""

from __future__ import annotations

import json
import os
import random

import cv2
import numpy as np

from .config import Config

__all__ = ["check_manifest", "render_overlays"]


def _load_manifest(cfg: Config) -> list[dict]:
    with open(cfg.manifest_path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def check_manifest(cfg: Config, fps_tol: float = 0.03) -> dict:
    """Validate the written manifest; returns a report dict and prints failures."""
    recs = _load_manifest(cfg)
    report: dict = {"n": len(recs), "by_source": {}, "issues": []}
    for r in recs:
        report["by_source"][r["source"]] = report["by_source"].get(r["source"], 0) + 1

        ctx = sorted(r["context_frames"], key=lambda c: c["rel_t"])  # oldest -> newest
        if not r["truncated"]:
            if len(ctx) != 3:
                report["issues"].append((r["scene_id"], f"context!=3 ({len(ctx)})"))
            else:
                # Each context frame must sit near its intended 10 fps tick. We check
                # distance-to-tick, not inter-frame gaps: a denser native source (e.g.
                # nuScenes 12 fps) yields uneven gaps yet each pick is still the nearest
                # frame to its 100 ms tick, which is what conditioning actually needs.
                h = r["horizon_s"]
                dt = 1.0 / cfg.target.fps
                intended = [-(h + 2 * dt), -(h + dt), -h]
                devs = [round(abs(ctx[i]["rel_t"] - intended[i]), 3) for i in range(3)]
                if max(devs) > dt / 2 + fps_tol:
                    report["issues"].append((r["scene_id"], f"context off-grid, dev={devs}"))

        W, H = cfg.target.width, cfg.target.height
        for b in r["anchor"]["boxes_vista"]:
            x, y, w, h = b
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                report["issues"].append((r["scene_id"], f"box out of frame {b}"))

        # every referenced frame file exists
        for fr in r["context_frames"] + r["history_pool"] + [r["anchor"]]:
            if not os.path.exists(os.path.join(cfg.out_root, fr["path"])):
                report["issues"].append((r["scene_id"], f"missing frame {fr['path']}"))

    report["ok"] = len(report["issues"]) == 0
    print(f"[validate] {report['n']} scenes, by_source={report['by_source']}, "
          f"{len(report['issues'])} issues")
    for sid, msg in report["issues"][:20]:
        print(f"  - {sid}: {msg}")
    return report


def render_overlays(cfg: Config, n: int = 25, seed: int = 0) -> str:
    """Draw boxes_vista on letterboxed anchors for a random sample -> overlays/ PNGs."""
    recs = _load_manifest(cfg)
    rng = random.Random(seed)
    sample = rng.sample(recs, min(n, len(recs)))
    out_dir = os.path.join(cfg.out_root, "overlays")
    os.makedirs(out_dir, exist_ok=True)
    for r in sample:
        anchor = cv2.imread(os.path.join(cfg.out_root, r["anchor"]["path"]))
        if anchor is None:
            continue
        # dim the letterbox bars using the valid mask so they are visible in the overlay
        if r.get("valid_mask"):
            m = cv2.imread(os.path.join(cfg.out_root, r["valid_mask"]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                bars = m == 0
                anchor[bars] = (anchor[bars] * 0.4).astype(np.uint8)
        for b in r["anchor"]["boxes_vista"]:
            x, y, w, h = [int(round(v)) for v in b]
            cv2.rectangle(anchor, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(anchor, r["scene_id"], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(out_dir, f"{r['scene_id']}.png"), anchor)
    print(f"[validate] wrote {len(sample)} overlays -> {out_dir}")
    return out_dir
