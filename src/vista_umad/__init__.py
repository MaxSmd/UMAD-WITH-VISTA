"""Vista-UMAD: unsupervised video anomaly detection for autonomous driving.

Phase 3 -- anomaly map pipeline. This package ports UMAD's pixel-space difference
metrics onto Vista's predict-and-compare output and fuses them into a single
anomaly map.

Public API
----------
* :mod:`vista_umad.metrics`  -- UMAD difference metrics (Eq. 1-5).
* :mod:`vista_umad.fusion`   -- normalization and weighted fusion of difference maps.
* :mod:`vista_umad.pipeline` -- :class:`AnomalyMapPipeline`, the end-to-end Phase 3 stage.
* :mod:`vista_umad.image_io` -- frame loading and anomaly-map visualization helpers.
* :mod:`vista_umad.runtime`  -- shared-server GPU selection and CPU-thread limits.
"""

from . import fusion, image_io, metrics, pipeline, runtime
from .pipeline import METRIC_KEYS, AnomalyMapPipeline

__all__ = [
    "metrics",
    "fusion",
    "pipeline",
    "image_io",
    "runtime",
    "AnomalyMapPipeline",
    "METRIC_KEYS",
]
