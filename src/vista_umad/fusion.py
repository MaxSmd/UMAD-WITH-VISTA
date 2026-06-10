"""Normalization and weighted fusion of UMAD difference maps.

UMAD computes several difference maps, normalizes each to a common range, and fuses
them with per-map weights ``w_i in [0, 1]``::

    fused = sum_i  w_i * normalize(map_i)

This module keeps fusion separate from the metric definitions in
:mod:`vista_umad.metrics`, so any subset of metrics can be composed with any weights.
"""

from __future__ import annotations

from typing import Mapping

import torch

__all__ = ["normalize_map", "fuse", "PRESETS"]


# UMAD reports (SSIM, PD) and (MSE, SSIM, PD, TD) as competitive combinations.
# Weights default to uniform within a preset; tune them in Phase 5.
PRESETS: dict[str, dict[str, float]] = {
    "ssim_pd": {"ssim": 0.5, "pd": 0.5},
    "mse_ssim_pd_td": {"mse": 0.25, "ssim": 0.25, "pd": 0.25, "td": 0.25},
    "all": {"abs": 0.2, "mse": 0.2, "ssim": 0.2, "pd": 0.2, "td": 0.2},
    "abs_only": {"abs": 1.0},
}


def normalize_map(
    anomaly_map: torch.Tensor,
    mode: str = "minmax",
    eps: float = 1e-8,
    clip_quantiles: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Normalize an anomaly map to ``[0, 1]``.

    Args:
        anomaly_map: ``[H, W]`` or ``[B, H, W]`` tensor. For a batch, each sample is
            normalized independently.
        mode: ``"minmax"`` rescales ``[min, max] -> [0, 1]``; ``"none"`` passes the
            map through unchanged (useful when a metric is already bounded).
        eps: guards against division by zero on a constant map.
        clip_quantiles: optional ``(low, high)`` quantile pair (e.g. ``(0.0, 0.99)``)
            applied per sample before min-max, making normalization robust to a few
            extreme pixels.

    Returns:
        A tensor of the same shape as ``anomaly_map``.
    """
    if mode not in ("minmax", "none"):
        raise ValueError(f"unknown normalization mode: {mode!r}")
    if mode == "none":
        return anomaly_map

    squeeze = anomaly_map.dim() == 2
    x = anomaly_map.unsqueeze(0) if squeeze else anomaly_map
    if x.dim() != 3:
        raise ValueError(f"expected [H,W] or [B,H,W], got {tuple(anomaly_map.shape)}")

    x = x.float()
    flat = x.reshape(x.shape[0], -1)
    if clip_quantiles is not None:
        low_q, high_q = clip_quantiles
        lo = torch.quantile(flat, low_q, dim=1, keepdim=True)
        hi = torch.quantile(flat, high_q, dim=1, keepdim=True)
        flat = torch.maximum(torch.minimum(flat, hi), lo)

    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    out = ((flat - lo) / (hi - lo + eps)).reshape(x.shape)
    return out[0] if squeeze else out


def fuse(
    maps: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    normalize: str = "minmax",
    clip_quantiles: tuple[float, float] | None = None,
    renormalize: bool = False,
) -> torch.Tensor:
    """Fuse named difference maps into a single anomaly map by weighted sum.

    Args:
        maps: name -> anomaly map. Maps must all share a shape.
        weights: name -> weight. Every weighted name must be present in ``maps``.
            Weights need not sum to 1 -- they are renormalized to sum to 1, matching
            UMAD's convention -- but they must be non-negative and not all zero.
        normalize: normalization ``mode`` applied to each map before weighting.
        clip_quantiles: forwarded to :func:`normalize_map`.
        renormalize: if ``True``, the fused map is min-max normalized to ``[0, 1]``
            as a final step.

    Returns:
        The fused anomaly map, same shape as the input maps.
    """
    if not weights:
        raise ValueError("fuse() needs at least one weighted map")

    missing = [name for name in weights if name not in maps]
    if missing:
        raise KeyError(f"weighted maps not found in `maps`: {missing}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"weights must be non-negative, got {dict(weights)}")

    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError(f"weights must not all be zero, got {dict(weights)}")

    ref_shape = maps[next(iter(weights))].shape
    fused: torch.Tensor | None = None
    for name, weight in weights.items():
        if weight == 0:
            continue
        amap = maps[name]
        if amap.shape != ref_shape:
            raise ValueError(
                f"map {name!r} shape {tuple(amap.shape)} != {tuple(ref_shape)}"
            )
        contribution = (weight / total_weight) * normalize_map(
            amap, mode=normalize, clip_quantiles=clip_quantiles
        )
        fused = contribution if fused is None else fused + contribution

    assert fused is not None  # guaranteed: total_weight > 0 implies one nonzero weight
    if renormalize:
        fused = normalize_map(fused, mode="minmax")
    return fused
