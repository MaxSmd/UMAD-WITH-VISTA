"""Phase 4.3 -- mask-level refinement of pixel anomaly maps.

UMAD's classification head turns a pixel-wise anomaly map into an *object-level*
prediction: instead of "this pixel is anomalous", "this segment is anomalous". A
class-agnostic instance segmenter produces candidate masks, and each mask's score
is the aggregate (default: 99th percentile) of the underlying pixel scores.
Why aggregate, not pixel-thresholding? UMAD reports it suppresses small
prediction-error islands inside ordinary objects (cars, road) and keeps the
anomaly signal *coherent* across an unfamiliar object.

The :class:`Segmenter` interface is intentionally segmenter-agnostic -- the
default backend is SAM (``segment_anything.SamAutomaticMaskGenerator``) but
U2Seg / Mask2Former / any class-agnostic source of ``[N, H, W]`` bool masks can
be dropped in. :func:`refine_with_masks` is the pure aggregation step; it does
not require SAM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import torch

__all__ = [
    "Segmenter",
    "SamSegmenter",
    "AGGREGATIONS",
    "refine_with_masks",
    "background_score",
]


# --- aggregation operators ----------------------------------------------------
# UMAD's paper picks "max" but reports the high-quantile variant as more robust
# to noisy single pixels. The dict makes this an entry-point choice.
AGGREGATIONS = {
    "max": lambda v: float(v.max()) if v.size else 0.0,
    "mean": lambda v: float(v.mean()) if v.size else 0.0,
    # 99th percentile -> robust max; ignores 1% extreme outliers per mask.
    "p99": lambda v: float(np.quantile(v, 0.99)) if v.size else 0.0,
    "p95": lambda v: float(np.quantile(v, 0.95)) if v.size else 0.0,
}


class Segmenter(Protocol):
    """A class-agnostic instance segmenter for a single RGB frame.

    ``segment(image)`` takes an ``[H, W, 3]`` uint8 RGB image and returns a list
    of ``[H, W]`` boolean masks -- one per candidate object/region. The masks do
    not have to be disjoint; pixels outside every mask are treated as background.
    """

    def segment(self, image: np.ndarray) -> list[np.ndarray]:  # pragma: no cover
        ...


@dataclass
class _SamMaskRecord:
    """Raw record returned by SAM's automatic mask generator (the fields we use)."""

    mask: np.ndarray  # [H, W] bool
    area: int        # pixel count


class SamSegmenter:
    """Class-agnostic mask source backed by Meta's Segment Anything model.

    Loads SAM lazily on first ``segment()`` call so that constructing the object
    is cheap. The model lives on the same device used for the rest of the
    Phase 3 pipeline -- by default whatever the caller passes in (typically
    cuda:0 after CUDA_VISIBLE_DEVICES masking).

    Args:
        checkpoint: path to a SAM ``.pth`` checkpoint (e.g.
            ``data/sam-checkpoints/sam_vit_b_01ec64.pth``).
        model_type: SAM backbone -- ``vit_b`` (~358 MB), ``vit_l``, or ``vit_h``.
        device: torch device for SAM.
        points_per_side: SAM grid density. Lower -> fewer masks, faster.
        min_mask_region_area: drop masks smaller than this (px). Suppresses noise.
        pred_iou_thresh / stability_score_thresh: SAM's internal mask quality
            thresholds. Defaults match the SAM repo's
            ``SamAutomaticMaskGenerator`` defaults except as noted.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str | torch.device = "cuda",
        points_per_side: int = 32,
        min_mask_region_area: int = 100,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
    ) -> None:
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.checkpoint = checkpoint
        self.model_type = model_type
        self.device = torch.device(device)
        self.points_per_side = points_per_side
        self.min_mask_region_area = min_mask_region_area
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self._generator = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._generator is not None:
            return
        # Import lazily so the module is importable without segment_anything.
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        self._generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def segment(self, image: np.ndarray) -> list[np.ndarray]:
        """Return SAM masks for ``image`` (``[H, W, 3]`` uint8)."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"SamSegmenter expects [H,W,3] uint8 RGB; got {image.shape} {image.dtype}"
            )
        self._ensure_loaded()
        records = self._generator.generate(image)  # type: ignore[union-attr]
        # Sort by area descending so the largest segments come first -- handy when
        # downstream code wants to inspect "the background" cheaply.
        records.sort(key=lambda r: r["area"], reverse=True)
        return [r["segmentation"].astype(bool) for r in records]


def background_score(
    anomaly_map: np.ndarray,
    masks: Iterable[np.ndarray],
    aggregate: str = "p99",
) -> float:
    """Aggregate score over the pixels covered by *no* mask.

    SAM does not always tile the full image; the leftover pixels are reported
    separately so a missing-coverage anomaly cannot quietly disappear.
    """
    coverage = np.zeros(anomaly_map.shape, dtype=bool)
    for mask in masks:
        coverage |= mask
    leftover = anomaly_map[~coverage]
    return AGGREGATIONS[aggregate](leftover)


def refine_with_masks(
    anomaly_map: np.ndarray | torch.Tensor,
    masks: list[np.ndarray],
    aggregate: str = "p99",
) -> tuple[np.ndarray, list[float]]:
    """Turn a pixel anomaly map into a mask-level anomaly map.

    Each input mask is assigned a single score by aggregating the pixel-wise
    anomaly values inside it. The output map paints every pixel of a mask with
    that mask's score; uncovered pixels keep their original anomaly value (so
    the refined map remains a strict ``[H, W]`` heatmap usable in Phase 5).

    Args:
        anomaly_map: ``[H, W]`` float array/tensor of pixel anomaly scores.
        masks: list of ``[H, W]`` bool masks (output of a :class:`Segmenter`).
        aggregate: key in :data:`AGGREGATIONS` -- ``"p99"`` (default), ``"max"``,
            ``"mean"``, ``"p95"``.

    Returns:
        ``(refined_map, per_mask_scores)``:
            * ``refined_map`` is ``[H, W]`` float32 with each mask filled to its
              aggregate score; uncovered pixels carry the input value.
            * ``per_mask_scores`` is the parallel list of scores (one per input
              mask, in the same order).
    """
    if aggregate not in AGGREGATIONS:
        raise KeyError(f"unknown aggregate {aggregate!r}; valid: {sorted(AGGREGATIONS)}")
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().float().cpu().numpy()
    anomaly_map = anomaly_map.astype(np.float32, copy=True)

    # The output keeps the *background* pixel score (helpful when SAM under-segments).
    refined = anomaly_map.copy()
    per_mask: list[float] = []
    aggr = AGGREGATIONS[aggregate]
    for mask in masks:
        if mask.shape != anomaly_map.shape:
            raise ValueError(
                f"mask shape {mask.shape} != anomaly_map shape {anomaly_map.shape}"
            )
        values = anomaly_map[mask]
        score = aggr(values)
        per_mask.append(score)
        # Overwrite this mask's pixels with its single score. If masks overlap,
        # later (smaller-area) masks win because SamSegmenter sorts descending.
        refined[mask] = score
    return refined, per_mask
