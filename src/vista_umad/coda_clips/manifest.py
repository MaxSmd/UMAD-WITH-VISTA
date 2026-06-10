"""Stage C+D+E orchestration -- build clips, letterbox, write the manifest store.

For each scene: resolve its source sequence (Stage B), sample the 10 fps clip
(Stage C), letterbox every available frame + transform the boxes (Stage D), and write
frames / valid mask / one ``manifest.jsonl`` line (Stage E). Frame files are numbered
chronologically (``000000.jpg`` = oldest), so the anchor lands at the highest index;
``context_frames`` and ``anchor`` reference those files by index. History deep enough for
the largest configured horizon is retained as ``history_pool`` so alternate horizons are
derivable without rebuilding. Scenes that can't resolve are logged with a reason, never
crash the run (gotchas #2, #6, #9).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2

from . import letterbox as lb
from .config import Config
from .index import Scene
from .sources import kitti, nuscenes, once
from .sources.base import Clip, build_clip, context_indices

__all__ = ["RESOLVERS", "build_manifest", "BuildReport"]

RESOLVERS = {"once": once.resolve, "kitti": kitti.resolve, "nuscenes": nuscenes.resolve}


@dataclass
class BuildReport:
    written: dict[str, int] = field(default_factory=dict)        # source -> scenes written
    dropped: dict[str, list[tuple[str, str]]] = field(default_factory=dict)  # source -> [(scene_id, reason)]
    truncated: dict[str, int] = field(default_factory=dict)      # source -> count flagged truncated

    def drop(self, source: str, scene_id: str, reason: str) -> None:
        self.dropped.setdefault(source, []).append((scene_id, reason))

    def summary(self) -> dict:
        return {
            "written": self.written,
            "truncated": self.truncated,
            "dropped_counts": {k: len(v) for k, v in self.dropped.items()},
        }


def _write_frames(cfg: Config, scene: Scene, clip: Clip):
    """Letterbox + write every clip frame that has pixels on disk.

    Returns ``(file_index_by_grid, transform, valid_mask_rel, boxes_vista)`` or ``None``
    if even the anchor has no pixels. Frame files are numbered in chronological order.
    """
    chrono = sorted(clip.frames, key=lambda g: g.rel_t)        # oldest -> anchor
    out_dir = cfg.clip_dir(scene.source, scene.scene_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cfg.masks_dir, exist_ok=True)

    transform = None
    valid_mask_rel = None
    file_index: dict[int, int] = {}     # grid_index -> written file index
    fi = 0
    for g in chrono:
        if g.src.path is None or not os.path.exists(g.src.path):
            continue
        img = cv2.imread(g.src.path)
        if img is None:
            continue
        out, mask, t = lb.letterbox(img, cfg.target.width, cfg.target.height, cfg.pad_color)
        if transform is None:
            transform = t
            mask_name = f"{scene.scene_id}_valid.png"
            cv2.imwrite(os.path.join(cfg.masks_dir, mask_name), mask * 255)
            valid_mask_rel = os.path.join("masks", mask_name)
        cv2.imwrite(os.path.join(out_dir, f"{fi:06d}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        file_index[g.grid_index] = fi
        fi += 1

    if transform is None or 0 not in file_index:   # anchor (grid_index 0) must be written
        return None
    boxes_vista = lb.boxes_to_vista(scene.boxes_native, transform)
    return file_index, transform, valid_mask_rel, boxes_vista


def _frame_record(cfg: Config, scene: Scene, clip: Clip, grid_index: int, file_index: dict[int, int]):
    g = next(g for g in clip.frames if g.grid_index == grid_index)
    fi = file_index[grid_index]
    rel = os.path.join("clips", scene.source, scene.scene_id, "frames", f"{fi:06d}.jpg")
    return {"index": fi, "path": rel, "rel_t": g.rel_t}


def build_manifest(cfg: Config, scenes: list[Scene], fetch: bool = True, limit: int | None = None) -> BuildReport:
    """Resolve, fetch (if enabled), build, and write manifest.jsonl for ``scenes``.

    Args:
        scenes: Stage A scene index (or a subset for validation).
        fetch: if True, fetch needed raw media for enabled sources before building.
        limit: optional cap on scenes processed per source (for quick validation runs).
    """
    os.makedirs(cfg.out_root, exist_ok=True)
    report = BuildReport()

    enabled = {n for n, sc in cfg.sources.items() if sc.enabled}
    work = [s for s in scenes if s.source in enabled]
    if limit is not None:
        by_src: dict[str, int] = {}
        capped = []
        for s in work:
            if by_src.get(s.source, 0) >= limit:
                continue
            by_src[s.source] = by_src.get(s.source, 0) + 1
            capped.append(s)
        work = capped

    if fetch and "kitti" in enabled:
        kitti.fetch(cfg, work)

    primary_h = cfg.eval.primary_horizon_s
    max_h = max(cfg.horizons_s)
    with open(cfg.manifest_path, "w") as mf:
        for scene in work:
            seq = RESOLVERS[scene.source](cfg, scene)
            if seq is None:
                report.drop(scene.source, scene.scene_id, "unresolvable_source_mapping")
                continue
            clip = build_clip(seq, cfg.target.fps, cfg.history_pool_frames)
            written = _write_frames(cfg, scene, clip)
            if written is None:
                report.drop(scene.source, scene.scene_id, "no_pixels_for_anchor")
                continue
            file_index, transform, valid_mask_rel, boxes_vista = written

            rec = _assemble_record(
                cfg, scene, seq, clip, file_index, transform, valid_mask_rel, boxes_vista, primary_h, max_h
            )
            mf.write(json.dumps(rec) + "\n")
            report.written[scene.source] = report.written.get(scene.source, 0) + 1
            if rec["truncated"]:
                report.truncated[scene.source] = report.truncated.get(scene.source, 0) + 1

    with open(os.path.join(cfg.out_root, "build_report.json"), "w") as fh:
        json.dump({"summary": report.summary(), "dropped": report.dropped}, fh, indent=1)
    return report


def _assemble_record(cfg, scene, seq, clip, file_index, transform, valid_mask_rel, boxes_vista, primary_h, max_h):
    fps = cfg.target.fps
    # context frames for the primary horizon, if all 3 grid frames were written
    ctx_idx = context_indices(primary_h, fps)
    have_ctx = all(i in file_index for i in ctx_idx)
    context = [_frame_record(cfg, scene, clip, i, file_index) for i in ctx_idx] if have_ctx else []
    # renumber context list as 0,1,2 (oldest->newest) for convenience
    for j, c in enumerate(context):
        c["context_pos"] = j

    history_pool = [
        _frame_record(cfg, scene, clip, g.grid_index, file_index)
        for g in sorted(clip.frames, key=lambda g: g.rel_t)
        if g.grid_index in file_index and g.grid_index != 0
    ]

    # Context frames must each sit near their intended 10 fps tick. With a denser native
    # source (nuScenes 12 fps) inter-frame gaps are uneven but each pick is still nearest
    # its tick; a pick far off its tick means a real sweep gap -> flag off-grid so a clean
    # eval slice can drop it (gotcha #3: never silently feed a broken triplet).
    dt = 1.0 / fps
    context_off_grid = False
    if have_ctx:
        intended = [-(primary_h + 2 * dt), -(primary_h + dt), -primary_h]
        ctx_sorted = sorted(context, key=lambda c: c["rel_t"])
        context_off_grid = max(abs(ctx_sorted[i]["rel_t"] - intended[i]) for i in range(3)) > dt / 2 + 0.02
    if context_off_grid and "context_off_grid" not in clip.notes:
        clip.notes.append("context_off_grid")

    # enough depth for the largest horizon? need context_indices(max_h) all present
    deep_enough = all(i in file_index for i in context_indices(max_h, fps))
    truncated = clip.truncated or (not have_ctx) or (not deep_enough) or context_off_grid

    anchor_fi = file_index[0]
    return {
        "scene_id": scene.scene_id,
        "source": scene.source,
        "in_vista_train_domain": scene.source == "nuscenes",
        "redistributable": scene.source == "once",
        "source_ids": scene.source_ids,
        "coco_image_id": scene.coco_image_id,
        "native_size": list(scene.native_size),
        "native_fps_estimate": clip.native_fps_estimate,
        "letterbox": transform.to_json(cfg.pad_color),
        "context_frames": context,
        "horizon_s": primary_h,
        "anchor": {
            "path": os.path.join("clips", scene.source, scene.scene_id, "frames", f"{anchor_fi:06d}.jpg"),
            "rel_t": 0.0,
            "coco_image_id": scene.coco_image_id,
            "boxes_native": scene.boxes_native,
            "boxes_vista": [[round(v, 2) for v in b] for b in boxes_vista],
            "category_ids": scene.category_ids,
        },
        "history_pool": history_pool,
        "valid_mask": valid_mask_rel,
        "truncated": truncated,
        "notes": clip.notes,
    }
