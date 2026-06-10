"""Stage A -- parse CODA into a normalized scene index.

CODA's ``corner_case.json`` is a COCO file covering all 1500 scenes. Each scene's
source is recoverable from its ``file_name``:

* ``^\\d+_\\d+\\.jpg$``  -> **once** (``sequence_id``, ``frame_id`` = ms timestamp);
  the pixels ship inside CODA's ``images/``.
* ``kitti_NNNNNN.png``  -> **kitti**; ``kitti_indices.json`` maps the file_name to a
  zero-padded index into the KITTI object-detection *training* set (0..7480).
* ``nuscenes_NNNNNN.jpg`` -> **nuscenes**; ``nuscenes_indices.json`` maps it to a
  nuScenes ``sample_token``.

(The side files are dicts keyed by COCO ``file_name`` -- verified against the actual
release, not assumed.) This stage emits one :class:`Scene` per image and asserts the
canonical 1057/309/134 composition before anything downstream runs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config

__all__ = ["Scene", "EXPECTED_COUNTS", "build_scene_index", "load_scene_index"]

# Canonical CODA base-val-1500 composition (README + PLAN.md). Load-bearing.
EXPECTED_COUNTS = {"once": 1057, "kitti": 309, "nuscenes": 134}

_ONCE_RE = re.compile(r"^(?P<seq>\d+)_(?P<frame>\d+)\.jpg$")


@dataclass
class Scene:
    """Normalized per-scene record produced by Stage A.

    ``source_ids`` is source-specific:
      once    -> {"sequence_id", "frame_id"}      (frame_id is a ms timestamp)
      kitti   -> {"odet_index"}                   (string index into the 7481 training set)
      nuscenes-> {"sample_token"}
    """

    scene_id: str
    source: str
    coco_image_id: int
    file_name: str
    native_size: tuple[int, int]          # [W, H]
    source_ids: dict[str, str]
    boxes_native: list[list[float]]       # COCO xywh, native pixels
    category_ids: list[int]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["native_size"] = list(self.native_size)
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Scene":
        return cls(
            scene_id=d["scene_id"],
            source=d["source"],
            coco_image_id=int(d["coco_image_id"]),
            file_name=d["file_name"],
            native_size=tuple(d["native_size"]),  # type: ignore[arg-type]
            source_ids=dict(d["source_ids"]),
            boxes_native=[list(b) for b in d["boxes_native"]],
            category_ids=list(d["category_ids"]),
        )


def _scene_id(source: str, source_ids: dict[str, str]) -> str:
    if source == "once":
        return f"once_{source_ids['sequence_id']}_{source_ids['frame_id']}"
    if source == "kitti":
        return f"kitti_{source_ids['odet_index']}"
    if source == "nuscenes":
        return f"nuscenes_{source_ids['sample_token']}"
    raise ValueError(f"unknown source {source!r}")


def build_scene_index(cfg: Config, write: bool = True) -> list[Scene]:
    """Parse CODA -> list[Scene]; assert composition; optionally persist.

    Args:
        cfg: builder config (uses ``coda_root``/``out_root``).
        write: if True, write ``out_root/scenes_index.json`` and copy the COCO file
            into ``out_root/annotations/`` (source of truth for boxes, untouched).
    """
    with open(cfg.coda_coco_path) as fh:
        coco = json.load(fh)

    images = {im["id"]: im for im in coco["images"]}

    # Aggregate annotations per image (COCO bbox is xywh in native pixels).
    boxes: dict[int, list[list[float]]] = {iid: [] for iid in images}
    cats: dict[int, list[int]] = {iid: [] for iid in images}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        boxes[iid].append([float(v) for v in ann["bbox"]])
        cats[iid].append(int(ann["category_id"]))

    kitti_idx = _load_side(os.path.join(cfg.coda_root, "kitti_indices.json"))
    # CODA ships nuScenes ids as nuscenes_indices.json (README still calls it
    # nuscenes_sample_tokens.json); accept either filename.
    nusc_idx = _load_side(
        os.path.join(cfg.coda_root, "nuscenes_indices.json"),
        os.path.join(cfg.coda_root, "nuscenes_sample_tokens.json"),
    )

    scenes: list[Scene] = []
    for iid, im in images.items():
        fn = im["file_name"]
        native = (int(im["width"]), int(im["height"]))
        m = _ONCE_RE.match(fn)
        if m and os.path.exists(os.path.join(cfg.coda_images_dir, fn)):
            source = "once"
            sids = {"sequence_id": m.group("seq"), "frame_id": m.group("frame")}
        elif fn.startswith("kitti_"):
            source = "kitti"
            if fn not in kitti_idx:
                raise KeyError(f"{fn} missing from kitti_indices.json")
            sids = {"odet_index": str(kitti_idx[fn])}
        elif fn.startswith("nuscenes_"):
            source = "nuscenes"
            if fn not in nusc_idx:
                raise KeyError(f"{fn} missing from nuscenes index file")
            sids = {"sample_token": str(nusc_idx[fn])}
        else:
            raise ValueError(f"cannot assign source for file_name {fn!r}")

        scenes.append(
            Scene(
                scene_id=_scene_id(source, sids),
                source=source,
                coco_image_id=int(iid),
                file_name=fn,
                native_size=native,
                source_ids=sids,
                boxes_native=boxes[iid],
                category_ids=cats[iid],
            )
        )

    _assert_counts(scenes)

    if write:
        os.makedirs(cfg.out_root, exist_ok=True)
        with open(cfg.scenes_index_path, "w") as fh:
            json.dump([s.to_json() for s in scenes], fh, indent=1)
        os.makedirs(cfg.annotations_dir, exist_ok=True)
        with open(os.path.join(cfg.annotations_dir, "corner_case.coco.json"), "w") as fh:
            json.dump(coco, fh)
    return scenes


def _load_side(*candidates: str) -> dict[str, str]:
    for path in candidates:
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    raise FileNotFoundError(f"none of these side files exist: {candidates}")


def _assert_counts(scenes: list[Scene]) -> None:
    counts: dict[str, int] = {}
    for s in scenes:
        counts[s.source] = counts.get(s.source, 0) + 1
    if counts != EXPECTED_COUNTS:
        raise AssertionError(
            f"CODA composition mismatch: got {counts}, expected {EXPECTED_COUNTS}. "
            "Stage A loaders are out of sync with the actual release."
        )


def load_scene_index(cfg: Config) -> list[Scene]:
    """Load a previously written ``scenes_index.json``."""
    with open(cfg.scenes_index_path) as fh:
        return [Scene.from_json(d) for d in json.load(fh)]
