"""Stage F -- prediction-error corner-case detection (wraps VISTA).

The pipeline per scene: condition VISTA on the 3 letterboxed context frames, roll out
to the anchor (``primary_horizon_s`` later = generated frame index ``n_conds-1 +
round(h*fps)``), and score prediction error *inside* the CODA boxes vs. outside. Two
things make or break the numbers and are handled here explicitly:

* **Mask the letterbox bars** out of every error map before aggregation -- the bars are
  constant and would otherwise inflate background statistics (gotcha #5).
* **Never pool nuScenes** with ONCE/KITTI into one headline (it is in VISTA's training
  domain; gotcha #8) -- metrics are broken out per source, plus an ONCE+KITTI macro.

This module owns the parts that are pure and testable (anno generation, masked error
maps, per-box scoring, K-aggregation, metric aggregation) and shells out to the official
``external/Vista/sample.py`` for the actual rollouts. The conditioning is the repo's own
3-frame latent injection; we do not reimplement it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import evaluation, metrics
from .config import Config

__all__ = [
    "anchor_frame_index",
    "build_vista_anno",
    "run_vista",
    "error_map",
    "score_boxes",
    "aggregate_k",
    "evaluate",
]


def anchor_frame_index(horizon_s: float, fps: int, n_conds: int) -> int:
    """VISTA rollout frame index of the anchor.

    Context frames occupy indices ``0..n_conds-1``; the last (index ``n_conds-1``) is
    ``horizon_s`` before the anchor, so the anchor is ``round(horizon_s*fps)`` further on.
    For h=1.0 s, 10 fps, 3 conds -> index 12.
    """
    return (n_conds - 1) + round(horizon_s * fps)


# --------------------------------------------------------------------------- anno
def _evaluable(rec: dict, horizon_s: float, fps: int = 10) -> bool:
    """A scene is evaluable iff it has 3 context frames each near their 10 fps tick.

    The tick-distance gate (not inter-frame spacing) drops scenes with a real sweep gap
    in the context window -- e.g. a few nuScenes clips -- while keeping the many whose
    12 fps picks land within half a frame of each 100 ms tick.
    """
    if rec["horizon_s"] != horizon_s:
        # context_frames in the manifest are for the primary horizon; for other horizons
        # the eval would re-derive from history_pool. Keep it simple: primary only here.
        return False
    ctx = sorted(rec["context_frames"], key=lambda c: c["rel_t"])
    if len(ctx) != 3:
        return False
    dt = 1.0 / fps
    intended = [-(horizon_s + 2 * dt), -(horizon_s + dt), -horizon_s]
    return max(abs(ctx[i]["rel_t"] - intended[i]) for i in range(3)) <= dt / 2 + 0.02


def build_vista_anno(cfg: Config, out_path: str, horizon_s: float | None = None, sources=None) -> list[dict]:
    """Write a VISTA-compatible annotation JSON from the manifest.

    Each entry mirrors VISTA's NUSCENES schema: a ``frames`` list of ``num_frames`` paths
    whose first ``n_conds`` are the real context frames. The remaining slots are padded
    with the real anchor (so every path exists and the anchor lands at
    :func:`anchor_frame_index`); padding never feeds conditioning, so it does not affect
    the prediction -- but it does invalidate FVD, which needs the full real clip (only
    KITTI/nuScenes with fetched pixels can supply that; flagged in the datasheet).
    """
    horizon_s = horizon_s or cfg.eval.primary_horizon_s
    n_conds = cfg.context_frames
    num_frames = 25  # VISTA clip length
    a_idx = anchor_frame_index(horizon_s, cfg.target.fps, n_conds)
    assert a_idx < num_frames, f"anchor index {a_idx} >= {num_frames}; horizon too long"

    entries: list[dict] = []
    with open(cfg.manifest_path) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if sources and rec["source"] not in sources:
                continue
            if not _evaluable(rec, horizon_s, cfg.target.fps):
                continue
            ctx = sorted(rec["context_frames"], key=lambda c: c["rel_t"])  # oldest->newest
            anchor_path = rec["anchor"]["path"]
            frames = [os.path.join(cfg.out_root, c["path"]) for c in ctx]
            frames += [os.path.join(cfg.out_root, anchor_path)] * (num_frames - len(frames))
            entries.append(
                {
                    "scene_id": rec["scene_id"],
                    "source": rec["source"],
                    "in_vista_train_domain": rec["in_vista_train_domain"],
                    "frames": frames,
                    "anchor_index": a_idx,
                    "anchor_path": os.path.join(cfg.out_root, anchor_path),
                    "valid_mask": os.path.join(cfg.out_root, rec["valid_mask"]) if rec.get("valid_mask") else None,
                    "boxes_vista": rec["anchor"]["boxes_vista"],
                    "category_ids": rec["anchor"]["category_ids"],
                }
            )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(entries, fh)
    print(f"[eval] wrote {len(entries)} VISTA anno entries -> {out_path} (anchor_index={a_idx})")
    return entries


def run_vista(cfg: Config, anno_path: str, save_dir: str, k_samples: int | None = None, gpu: int = 0) -> None:
    """Invoke the official VISTA sampler to roll out clips for ``anno_path``.

    Thin subprocess wrapper -- VISTA owns preprocessing + 3-frame latent injection. We
    register the anno as a custom dataset via env vars the sampler reads, request
    ``k_samples`` rounds (``--n_rounds``) and 3 context frames (``--n_conds``).
    Predicted frames land under ``save_dir/virtual/images`` per the repo's convention.
    """
    k = k_samples or cfg.eval.k_samples
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CODA_CLIPS_ANNO=anno_path)
    cmd = [
        os.path.join(cfg.vista_repo, ".venv", "bin", "python"),
        os.path.join(cfg.vista_repo, "sample.py"),
        "--dataset", "NUSCENES",          # NUSCENES path consumes a frames-list anno
        "--save", save_dir,
        "--n_conds", str(cfg.context_frames),
        "--n_frames", "25",
        "--n_rounds", str(k),
        "--height", str(cfg.target.height),
        "--width", str(cfg.target.width),
    ]
    print("[eval] launching VISTA:", " ".join(cmd))
    subprocess.run(cmd, cwd=cfg.vista_repo, env=env, check=True)


# ------------------------------------------------------------------- scoring core
def error_map(real: torch.Tensor, pred: torch.Tensor, kind: str = "l2", extractor=None) -> torch.Tensor:
    """Per-pixel prediction-error map ``[H, W]`` (higher = more error).

    ``kind='l2'`` is mean-squared RGB error; ``kind='perceptual'`` uses the repo's frozen
    VGG perceptual extractor (UMAD Eq. 4). Inputs are ``[3, H, W]`` in [0, 1].
    """
    if kind == "l2":
        return metrics.mse_error(real, pred)
    if kind == "perceptual":
        if extractor is None:
            extractor = metrics.VGGPerceptualExtractor(device=real.device)
        return extractor(real, pred)
    raise ValueError(f"unknown error kind {kind!r}")


@dataclass
class BoxScore:
    scene_id: str
    source: str
    in_vista_train_domain: bool
    box_scores: list[float]           # per-box aggregate error (positives)
    background: float                 # mean error over valid non-box pixels (negative ref)
    pixel_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    pixel_labels: np.ndarray = field(default_factory=lambda: np.empty(0))


def score_boxes(emap: torch.Tensor, valid_mask: np.ndarray, boxes_vista, scene_id: str,
                source: str, in_train: bool, box_agg: str = "mean") -> BoxScore:
    """Score one anchor: per-box error vs. background, with bars masked out.

    ``valid_mask`` (1 inside content, 0 in bars) zeroes the bars before any aggregation.
    Returns per-box aggregates (region-level positives), the background mean (negative
    reference), and the masked per-pixel scores/labels for pixel-level AUROC pooling.
    """
    e = emap.detach().cpu().numpy().astype(np.float64)
    H, W = e.shape
    if valid_mask is None:
        valid = np.ones((H, W), bool)
    else:
        valid = valid_mask.astype(bool)
        if valid.shape != (H, W):
            import cv2
            valid = cv2.resize(valid.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    box_label = np.zeros((H, W), bool)
    box_scores: list[float] = []
    for b in boxes_vista:
        x, y, w, h = b
        x0, y0 = int(max(0, round(x))), int(max(0, round(y)))
        x1, y1 = int(min(W, round(x + w))), int(min(H, round(y + h)))
        if x1 <= x0 or y1 <= y0:
            continue
        region = e[y0:y1, x0:x1]
        rvalid = valid[y0:y1, x0:x1]
        vals = region[rvalid]
        if vals.size == 0:
            continue
        agg = float(np.mean(vals)) if box_agg == "mean" else float(np.percentile(vals, 99))
        box_scores.append(agg)
        box_label[y0:y1, x0:x1] = True

    bg_mask = valid & ~box_label
    background = float(e[bg_mask].mean()) if bg_mask.any() else float("nan")

    pix_scores = e[valid]
    pix_labels = box_label[valid]
    return BoxScore(scene_id, source, in_train, box_scores, background,
                    pix_scores.astype(np.float64), pix_labels.astype(np.uint8))


def aggregate_k(emaps: list[torch.Tensor], mode: str = "mean") -> torch.Tensor:
    """Aggregate K rollout error maps. ``mean`` (default), ``min`` (don't penalize
    legitimate alternative futures), or ``var`` (cross-sample uncertainty signal)."""
    stack = torch.stack(emaps, 0)
    if mode == "mean":
        return stack.mean(0)
    if mode == "min":
        return stack.amin(0)
    if mode == "var":
        return stack.var(0, unbiased=False)
    raise ValueError(f"unknown K-aggregation {mode!r}")


# ----------------------------------------------------------------- aggregation
def evaluate(box_scores: list[BoxScore]) -> dict:
    """Aggregate per-source AUROC/AP at both pixel and region level.

    * pixel-level: pool masked per-pixel scores/labels within a source.
    * region-level: per-box aggregate vs. that scene's background (a balanced,
      detection-style signal that the single anti-correlated frame can't dominate).

    nuScenes is reported on its own; an ONCE+KITTI macro is the headline. No single
    number pools the in-train-domain source with the rest.
    """
    by_source: dict[str, list[BoxScore]] = {}
    for bs in box_scores:
        by_source.setdefault(bs.source, []).append(bs)

    out: dict = {"per_source": {}}
    for src, items in by_source.items():
        # pixel-level pooled
        ps = np.concatenate([b.pixel_scores for b in items]) if items else np.empty(0)
        pl = np.concatenate([b.pixel_labels for b in items]) if items else np.empty(0)
        pix = evaluation.compute_metrics(ps, pl) if ps.size else None
        # region-level: box positives vs background negatives
        r_scores, r_labels = [], []
        for b in items:
            for bsc in b.box_scores:
                r_scores.append(bsc); r_labels.append(1)
            if not np.isnan(b.background):
                r_scores.append(b.background); r_labels.append(0)
        reg = evaluation.compute_metrics(np.array(r_scores), np.array(r_labels)) if r_scores else None
        out["per_source"][src] = {
            "n_scenes": len(items),
            "pixel": _scores_dict(pix),
            "region": _scores_dict(reg),
            "in_vista_train_domain": items[0].in_vista_train_domain if items else False,
        }

    # macro over non-train-domain sources (ONCE + KITTI), region-level
    macro_items = [b for b in box_scores if not b.in_vista_train_domain]
    if macro_items:
        aurocs = [out["per_source"][s]["region"]["auroc"]
                  for s in out["per_source"]
                  if not out["per_source"][s]["in_vista_train_domain"]
                  and out["per_source"][s]["region"] is not None]
        aurocs = [a for a in aurocs if a == a]  # drop nan
        out["macro_non_train_region_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    return out


def _scores_dict(s) -> dict | None:
    if s is None:
        return None
    return {"auroc": s.auroc, "ap": s.ap, "fpr95": s.fpr95,
            "n": s.n_pixels, "n_pos": s.n_anomaly_pixels}
