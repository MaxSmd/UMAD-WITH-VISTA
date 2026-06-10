"""Configuration loading for the CODA-clips benchmark builder.

A single YAML file (``configs/coda_clips.yaml``) holds every knob; see PLAN.md
section 2. This module parses it into a typed :class:`Config` so the rest of the
package never touches raw dict access. Relative paths are resolved against the
repository root (the directory two levels above this file's package), so the
builder behaves identically regardless of the working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

__all__ = ["Config", "Target", "SourceCfg", "EvalCfg", "load_config", "REPO_ROOT"]

# src/vista_umad/coda_clips/config.py -> repo root is four parents up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _abspath(path: str) -> str:
    """Resolve ``path`` against the repo root unless already absolute."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(REPO_ROOT, path))


@dataclass
class Target:
    width: int = 1024
    height: int = 576
    fps: int = 10


@dataclass
class SourceCfg:
    enabled: bool = False
    raw_cache: str | None = None
    dataroot: str | None = None
    mode: str | None = None


@dataclass
class EvalCfg:
    k_samples: int = 4
    primary_horizon_s: float = 1.0
    metrics: list[str] = field(default_factory=lambda: ["auroc_box", "ap_box", "fvd", "lpips"])


@dataclass
class Config:
    coda_root: str
    out_root: str
    target: Target
    context_frames: int
    horizons_s: list[float]
    history_pool_frames: int
    pad_color: tuple[int, int, int]
    sources: dict[str, SourceCfg]
    vista_repo: str
    eval: EvalCfg
    raw: dict[str, Any] = field(default_factory=dict)

    # --- derived paths into the output store -------------------------------
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.out_root, "manifest.jsonl")

    @property
    def scenes_index_path(self) -> str:
        return os.path.join(self.out_root, "scenes_index.json")

    @property
    def annotations_dir(self) -> str:
        return os.path.join(self.out_root, "annotations")

    @property
    def clips_dir(self) -> str:
        return os.path.join(self.out_root, "clips")

    @property
    def masks_dir(self) -> str:
        return os.path.join(self.out_root, "masks")

    def clip_dir(self, source: str, scene_id: str) -> str:
        """Per-scene frame directory: ``clips/<source>/<scene_id>/frames``."""
        return os.path.join(self.clips_dir, source, scene_id, "frames")

    @property
    def coda_coco_path(self) -> str:
        return os.path.join(self.coda_root, "corner_case.json")

    @property
    def coda_images_dir(self) -> str:
        return os.path.join(self.coda_root, "images")


def load_config(path: str | None = None) -> Config:
    """Load and normalize the builder config.

    Args:
        path: YAML path. Defaults to ``configs/coda_clips.yaml`` under the repo root.
    """
    if path is None:
        path = os.path.join(REPO_ROOT, "configs", "coda_clips.yaml")
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    t = raw.get("target", {})
    target = Target(width=int(t.get("width", 1024)), height=int(t.get("height", 576)), fps=int(t.get("fps", 10)))

    sources: dict[str, SourceCfg] = {}
    for name, sc in (raw.get("sources") or {}).items():
        sc = sc or {}
        sources[name] = SourceCfg(
            enabled=bool(sc.get("enabled", False)),
            raw_cache=_abspath(sc["raw_cache"]) if sc.get("raw_cache") else None,
            dataroot=_abspath(sc["dataroot"]) if sc.get("dataroot") else None,
            mode=sc.get("mode"),
        )

    ev = raw.get("eval", {})
    eval_cfg = EvalCfg(
        k_samples=int(ev.get("k_samples", 4)),
        primary_horizon_s=float(ev.get("primary_horizon_s", 1.0)),
        metrics=list(ev.get("metrics", ["auroc_box", "ap_box", "fvd", "lpips"])),
    )

    return Config(
        coda_root=_abspath(raw["coda_root"]),
        out_root=_abspath(raw["out_root"]),
        target=target,
        context_frames=int(raw.get("context_frames", 3)),
        horizons_s=[float(h) for h in raw.get("horizons_s", [1.0, 2.0])],
        history_pool_frames=int(raw.get("history_pool_frames", 30)),
        pad_color=tuple(int(c) for c in raw.get("pad_color", [0, 0, 0])),  # type: ignore[arg-type]
        sources=sources,
        vista_repo=_abspath(raw.get("vista_repo", "external/Vista")),
        eval=eval_cfg,
        raw=raw,
    )
