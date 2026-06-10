"""End-to-end Phase 3 anomaly-map pipeline.

:class:`AnomalyMapPipeline` ties the UMAD difference metrics (:mod:`vista_umad.metrics`)
to the weighted fusion stage (:mod:`vista_umad.fusion`). Given a real frame ``t`` and
Vista's predicted frame ``t-hat`` it produces:

* a dict of individual difference maps (``abs``, ``mse``, ``ssim``, ``pd``, ``td``), and
* a single fused pixel-wise anomaly map.

The fused map is the raw float output consumed by Phase 4's mask-level refinement.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from . import fusion, metrics

__all__ = ["AnomalyMapPipeline", "METRIC_KEYS"]

# Canonical metric names used as keys throughout the project.
METRIC_KEYS = ("abs", "mse", "ssim", "pd", "td")


def _resolve_weights(weights: str | Mapping[str, float]) -> dict[str, float]:
    """Resolve a preset name or explicit weight mapping to a validated weight dict."""
    if isinstance(weights, str):
        if weights not in fusion.PRESETS:
            raise KeyError(
                f"unknown weight preset {weights!r}; available: {sorted(fusion.PRESETS)}"
            )
        return dict(fusion.PRESETS[weights])
    resolved = {str(k): float(v) for k, v in weights.items()}
    unknown = [k for k in resolved if k not in METRIC_KEYS]
    if unknown:
        raise KeyError(f"unknown metric names in weights: {unknown}; valid: {METRIC_KEYS}")
    return resolved


class AnomalyMapPipeline:
    """Compute UMAD difference maps from Vista output and fuse them.

    Args:
        weights: a :data:`vista_umad.fusion.PRESETS` name (e.g. ``"ssim_pd"``) or an
            explicit ``{metric: weight}`` mapping. Determines which metrics the fused
            map combines.
        device: torch device for the metric computations. Defaults to CUDA if available.
        ssim_window: sliding-window size for the SSIM metric.
        vgg_layers: VGG-16 feature layers for the perceptual metric.
        normalize: per-map normalization mode passed to the fusion stage.
        clip_quantiles: optional robust-normalization quantiles (see
            :func:`vista_umad.fusion.normalize_map`).
        compute_all: if ``True`` (default) every metric is computed for inspection /
            ablation, even ones with zero fusion weight. If ``False`` only weighted
            metrics are computed, which skips loading VGG when ``pd`` is unused.
    """

    def __init__(
        self,
        weights: str | Mapping[str, float] = "ssim_pd",
        device: str | torch.device | None = None,
        ssim_window: int = 11,
        vgg_layers: Sequence[int] = (3, 8, 15, 22),
        normalize: str = "minmax",
        clip_quantiles: tuple[float, float] | None = None,
        compute_all: bool = True,
    ) -> None:
        self.weights = _resolve_weights(weights)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ssim_window = ssim_window
        self.normalize = normalize
        self.clip_quantiles = clip_quantiles
        self.compute_all = compute_all

        # VGG is loaded lazily: only when the perceptual map is actually needed.
        self._needs_pd = compute_all or self.weights.get("pd", 0.0) > 0.0
        self._vgg: metrics.VGGPerceptualExtractor | None = None
        self._vgg_layers = tuple(vgg_layers)

    @property
    def vgg(self) -> metrics.VGGPerceptualExtractor:
        """The (lazily constructed) VGG perceptual extractor."""
        if self._vgg is None:
            self._vgg = metrics.VGGPerceptualExtractor(
                layers=self._vgg_layers, device=self.device
            )
        return self._vgg

    def compute_maps(
        self,
        real: torch.Tensor,
        pred: torch.Tensor,
        prior_preds: Sequence[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the individual difference maps for one frame pair.

        Args:
            real: ground-truth frame ``t``, ``[3, H, W]`` or ``[B, 3, H, W]`` in ``[0, 1]``.
            pred: Vista's predicted frame ``t-hat``, same shape as ``real``.
            prior_preds: predictions of the *same* frame ``t`` from earlier overlapping
                rollouts, used for the temporal-difference (``td``) map. When omitted,
                ``td`` is simply absent from the returned dict.

        Returns:
            A dict mapping metric name -> anomaly map. ``td`` is included only when
            ``prior_preds`` is provided.
        """
        real = real.to(self.device)
        pred = pred.to(self.device)

        maps: dict[str, torch.Tensor] = {}

        def wanted(name: str) -> bool:
            return self.compute_all or self.weights.get(name, 0.0) > 0.0

        if wanted("abs"):
            maps["abs"] = metrics.abs_error(real, pred)
        if wanted("mse"):
            maps["mse"] = metrics.mse_error(real, pred)
        if wanted("ssim"):
            maps["ssim"] = metrics.ssim_difference(real, pred, window_size=self.ssim_window)
        if wanted("pd") and self._needs_pd:
            maps["pd"] = self.vgg(real, pred)
        if prior_preds is not None and len(prior_preds) > 0 and wanted("td"):
            priors = [p.to(self.device) for p in prior_preds]
            maps["td"] = metrics.temporal_difference(pred, priors)

        return maps

    def fuse(self, maps: Mapping[str, torch.Tensor], renormalize: bool = False) -> torch.Tensor:
        """Fuse pre-computed maps using this pipeline's weights.

        Metrics that are weighted but missing from ``maps`` (typically ``td`` when no
        prior rollouts exist) are dropped and the remaining weights are renormalized.
        """
        active = {name: w for name, w in self.weights.items() if name in maps and w > 0}
        if not active:
            raise ValueError(
                f"none of the weighted metrics {sorted(self.weights)} are present in "
                f"maps {sorted(maps)}"
            )
        return fusion.fuse(
            maps,
            active,
            normalize=self.normalize,
            clip_quantiles=self.clip_quantiles,
            renormalize=renormalize,
        )

    def __call__(
        self,
        real: torch.Tensor,
        pred: torch.Tensor,
        prior_preds: Sequence[torch.Tensor] | None = None,
        renormalize: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute every difference map and the fused anomaly map for a frame pair.

        Returns:
            ``(fused_map, maps)`` where ``fused_map`` is the weighted anomaly map and
            ``maps`` is the dict of individual difference maps.
        """
        maps = self.compute_maps(real, pred, prior_preds=prior_preds)
        fused = self.fuse(maps, renormalize=renormalize)
        return fused, maps
