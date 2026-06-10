"""AnoVox scenario helpers -- frame discovery and ground-truth anomaly masks.

An AnoVox scenario directory holds one sub-directory per sensor. The two this module
touches are the front RGB camera (``RGB-CAM*``) and its semantic camera
(``SEMANTIC-CAM*``). The semantic camera is a single-channel class-id map; anomalous
objects carry dedicated class ids (29-34, plus the generic ``anomaly`` id 100 -- see
``external/anovox/Definitions.py``). The per-pixel anomaly ground truth for a frame is
therefore ``semantic_map in ANOMALY_CLASS_IDS``.

Vista's ``load_img`` center-crops each RGB frame to the model's aspect ratio before
resizing. To compare an anomaly map (Vista resolution) against the ground truth, the
semantic map must undergo the *same* crop+resize -- :func:`load_gt_anomaly_mask` does
that with nearest-neighbour resampling so class ids are never interpolated.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

__all__ = [
    "ANOMALY_CLASS_IDS",
    "find_rgb_frames",
    "find_semantic_frames",
    "anovox_frame_id",
    "vista_crop_box",
    "load_gt_anomaly_mask",
    "eval_frame_indices",
]

# AnoVox semantic class ids that denote an anomaly (Definitions.py: home/animal/
# nature/special/falling/airplane = 29-34; generic "anomaly" = 100).
ANOMALY_CLASS_IDS: tuple[int, ...] = (29, 30, 31, 32, 33, 34, 100)


def _find_sensor_dir(scenario_dir: str, prefix: str) -> str:
    matches = [d for d in os.listdir(scenario_dir) if d.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"no {prefix}* directory in {scenario_dir}")
    return os.path.join(scenario_dir, sorted(matches)[0])


def anovox_frame_id(path: str) -> int:
    """Extract the integer AnoVox frame id from a sensor filename (``..._<id>.png``)."""
    return int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1])


def find_rgb_frames(scenario_dir: str) -> list[str]:
    """Return the sorted list of front-RGB-camera frame paths."""
    rgb_dir = _find_sensor_dir(scenario_dir, "RGB-CAM")
    frames = sorted(
        (os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".png")),
        key=anovox_frame_id,
    )
    if not frames:
        raise FileNotFoundError(f"no PNG frames in {rgb_dir}")
    return frames


def find_semantic_frames(scenario_dir: str) -> dict[int, str]:
    """Return a ``{frame_id: semantic_camera_path}`` map for the scenario."""
    sem_dir = _find_sensor_dir(scenario_dir, "SEMANTIC-CAM")
    return {
        anovox_frame_id(f): os.path.join(sem_dir, f)
        for f in os.listdir(sem_dir)
        if f.endswith(".png")
    }


def vista_crop_box(ori_w: int, ori_h: int, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Replicate the center-crop ``Vista/sample.py:load_img`` applies before resizing.

    Returns the crop box ``(left, top, right, bottom)``. The crop trims whichever axis
    is too long for the target aspect ratio; the result is then resized to the target.
    """
    if ori_w / ori_h > target_w / target_h:
        tmp_w = int(target_w / target_h * ori_h)
        left = (ori_w - tmp_w) // 2
        return left, 0, (ori_w + tmp_w) // 2, ori_h
    if ori_w / ori_h < target_w / target_h:
        tmp_h = int(target_h / target_w * ori_w)
        top = (ori_h - tmp_h) // 2
        return 0, top, ori_w, (ori_h + tmp_h) // 2
    return 0, 0, ori_w, ori_h


def load_gt_anomaly_mask(
    semantic_path: str,
    target_hw: tuple[int, int],
    anomaly_ids: tuple[int, ...] = ANOMALY_CLASS_IDS,
) -> np.ndarray:
    """Load a binary anomaly ground-truth mask aligned to a Vista-resolution frame.

    Args:
        semantic_path: an AnoVox ``SEMANTIC-CAM`` PNG (single-channel class-id map).
        target_hw: ``(height, width)`` of the Vista anomaly map to align to.
        anomaly_ids: semantic class ids treated as anomalous.

    Returns:
        A boolean ``[H, W]`` array, ``True`` where the frame is anomalous, cropped and
        nearest-neighbour resized to match Vista's ``load_img`` geometry.
    """
    target_h, target_w = target_hw
    semantic = Image.open(semantic_path)
    if semantic.mode not in ("L", "I", "P"):
        semantic = semantic.convert("L")
    ori_w, ori_h = semantic.size

    box = vista_crop_box(ori_w, ori_h, target_w, target_h)
    semantic = semantic.crop(box)
    # Nearest-neighbour: class ids must never be blended.
    semantic = semantic.resize((target_w, target_h), resample=Image.NEAREST)

    ids = np.asarray(semantic)
    return np.isin(ids, np.asarray(anomaly_ids))


def eval_frame_indices(n_total: int, stride: int = 10) -> list[int]:
    """Indices of the frames to evaluate -- every ``stride``-th frame (UMAD protocol)."""
    return list(range(0, n_total, stride))
