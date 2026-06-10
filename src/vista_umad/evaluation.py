"""Phase 5 -- pixel-level anomaly-detection metrics.

UMAD's evaluation protocol (Bogdoll et al. 2024, sec. 4.2) reports three numbers
per dataset:

* **AP / AUPR** -- area under the precision-recall curve. The headline metric:
  on heavily imbalanced data (anomaly pixels are rare) PR-AUC is more
  informative than ROC.
* **AUROC** -- area under the ROC curve. Reported for comparison with prior work.
* **FPR95** -- false-positive rate at 95% true-positive rate. A direct readout
  of how usable the detector is at a near-perfect-recall operating point.

These are straightforward wrappers around scikit-learn that take a *single*
flat (scores, labels) pair pooled across the whole evaluation set, matching the
AnoVox benchmark's reference implementation
(`external/anovox/benchmark/eval/metrics.py`). We re-implement them here so the
project does not depend on importing from the AnoVox tree.

The functions accept torch tensors or numpy arrays in any shape; ``labels`` is
booleanized internally (so the standard ``np.isin(ids, ANOMALY_CLASS_IDS)`` mask
from :mod:`vista_umad.anovox` flows through directly).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

__all__ = [
    "AnomalyScores",
    "compute_metrics",
    "fpr_at_tpr",
    "best_f1",
]


def _to_flat_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach + move to CPU + flatten -> float32 1-D numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, tpr_target: float = 0.95) -> float:
    """FPR at the smallest threshold achieving ``tpr >= tpr_target``.

    Returns ``nan`` if no threshold reaches ``tpr_target`` (e.g. all labels are
    one class).
    """
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    # roc_curve sorts thresholds descending -- find the first point at or above
    # the recall target and read its FPR.
    above = tpr >= tpr_target
    if not above.any():
        return float("nan")
    return float(fpr[np.argmax(above)])


def best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over the PR curve, plus the score threshold that achieves it."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one extra (precision=1, recall=0) sentinel
    # at the end and thresholds is shorter by 1; align them.
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return float(f1.max()) if len(f1) else 0.0, 0.0
    idx = int(np.argmax(f1[:-1]))
    return float(f1[idx]), float(thresholds[idx])


@dataclass
class AnomalyScores:
    """Bundle of headline anomaly-detection metrics."""

    auroc: float
    ap: float          # average precision == AUPR
    fpr95: float
    f1: float
    f1_threshold: float
    n_pixels: int
    n_anomaly_pixels: int
    anomaly_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)

    def as_table(self) -> str:
        """Return a 1-row human-readable summary."""
        return (
            f"AUROC={self.auroc:.4f}  AP={self.ap:.4f}  FPR95={self.fpr95:.4f}  "
            f"F1={self.f1:.4f} (@{self.f1_threshold:.3f})  "
            f"N={self.n_pixels:,} pos={self.n_anomaly_pixels:,} "
            f"({self.anomaly_fraction:.4%})"
        )


def compute_metrics(
    scores: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> AnomalyScores:
    """Compute the headline pixel-level anomaly metrics.

    Args:
        scores: per-pixel anomaly scores (higher = more anomalous). Any shape;
            flattened internally.
        labels: per-pixel ground truth. Booleanized via ``!= 0``; same shape as
            ``scores`` (or any broadcast-compatible flattening).

    Returns:
        :class:`AnomalyScores` bundling AUROC, AP, FPR95, best F1 and counts.

    Notes:
        If ``labels`` contains only one class, AUROC/AP/FPR95 are ``nan`` (they
        are undefined). This lets the caller pool across many frames without
        having to filter empty ones.
    """
    s = _to_flat_numpy(scores).astype(np.float64)
    y = _to_flat_numpy(labels)
    y = (y != 0).astype(np.uint8)
    if s.shape != y.shape:
        raise ValueError(f"scores/labels shape mismatch: {s.shape} vs {y.shape}")

    n = int(s.size)
    n_pos = int(y.sum())
    frac = n_pos / n if n else 0.0

    if n_pos == 0 or n_pos == n:
        nan = float("nan")
        return AnomalyScores(nan, nan, nan, nan, nan, n, n_pos, frac)

    auroc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    fpr95 = fpr_at_tpr(y, s, tpr_target=0.95)
    f1, thr = best_f1(y, s)
    return AnomalyScores(auroc, ap, fpr95, f1, thr, n, n_pos, frac)
