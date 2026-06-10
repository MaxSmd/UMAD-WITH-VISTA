#!/usr/bin/env python
"""Stage B -- full UMAD-on-Vista anomaly pipeline (Phase 3 + 4.3 + 5).

Consumes the per-rollout predictions written by ``run_inference_anovox.py`` and
produces:

* a fused pixel anomaly map per evaluated frame (Phase 3),
* an optional mask-refined version (Phase 4.3, SAM-backed),
* AUROC / AP / FPR95 / F1 vs the AnoVox semantic-camera ground truth (Phase 5).

The evaluated frames are picked by UMAD's protocol -- every ``--eval-stride``-th
frame of the scenario that has at least one Vista prediction. For each such
frame ``t`` the most recently started rollout containing ``t`` provides the
"current" prediction; the predictions of ``t`` made by earlier rollouts feed
the temporal-difference (TD) signal (UMAD Eq. 5).

Usage:
    python scripts/run_anomaly_pipeline.py \\
        --inference-dir outputs/anovox-inference/Scenario_e2f08f8f \\
        --scenario data/anovox/Outputs/.../Scenario_e2f08f8f-... \\
        --weights mse_ssim_pd_td --gpu 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Mapping

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--inference-dir", required=False,
        default=os.path.join(REPO_ROOT, "outputs", "anovox-inference", "Scenario_e2f08f8f"),
        help="output dir from run_inference_anovox.py (contains manifest.json)",
    )
    p.add_argument(
        "--scenario", required=False, default=None,
        help="AnoVox scenario dir (for SEMANTIC-CAM ground truth); "
             "defaults to the scenario recorded in manifest.json",
    )
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "anomaly-pipeline"))
    p.add_argument("--tag", default=None, help="run tag (subdir under --out)")

    # Phase 3 fusion.
    p.add_argument("--weights", default="mse_ssim_pd_td",
                   help="fusion preset (see vista_umad.fusion.PRESETS) or comma-separated "
                        "name=weight pairs (e.g. ssim=0.4,pd=0.4,td=0.2)")

    # Phase 4.3 refinement.
    p.add_argument("--refine", default="sam", choices=["sam", "none"],
                   help="mask-refinement backend; 'none' evaluates raw pixel maps")
    p.add_argument("--sam-ckpt",
                   default=os.path.join(REPO_ROOT, "data", "sam-checkpoints",
                                        "sam_vit_b_01ec64.pth"))
    p.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--aggregate", default="p99", choices=["max", "mean", "p99", "p95"],
                   help="how to summarize pixel scores within a mask (default p99)")

    # Phase 5 eval.
    p.add_argument("--eval-stride", type=int, default=10,
                   help="evaluate every Nth frame (UMAD protocol, default 10)")
    p.add_argument("--anomaly-class-ids", default=None,
                   help="override AnoVox anomaly class ids (comma-separated)")

    # Runtime.
    p.add_argument("--gpu", type=int, default=None,
                   help="physical GPU index; sets CUDA_VISIBLE_DEVICES if not already set")
    p.add_argument("--threads", type=int, default=4, help="CPU thread cap")
    p.add_argument("--device", default="auto",
                   help="torch device (auto/cpu/cuda/cuda:0) -- evaluated AFTER CUDA masking")

    # Output controls.
    p.add_argument("--n-vis", type=int, default=8, help="save first N visualization panels")
    p.add_argument("--no-save-maps", action="store_true",
                   help="skip writing fused_*.npy / refined_*.npy (smaller output dir)")
    p.add_argument("--max-eval-frames", type=int, default=0,
                   help="0 = all; limit eval frames for a quick smoke test")
    return p.parse_args()


ARGS = parse_args()

# --- resource confinement: must precede every torch / CUDA import -------------
if ARGS.gpu is not None:
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and existing != str(ARGS.gpu):
        print(f"WARNING: --gpu={ARGS.gpu} but CUDA_VISIBLE_DEVICES={existing} is already set; "
              "respecting the environment variable.", file=sys.stderr)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(ARGS.threads))

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vista_umad import anovox, evaluation, fusion, image_io, masks, runtime  # noqa: E402
from vista_umad.pipeline import AnomalyMapPipeline  # noqa: E402


def parse_weights(spec: str) -> str | dict[str, float]:
    """Accept a preset name or a comma-separated ``name=weight`` list."""
    if "=" not in spec:
        return spec
    out: dict[str, float] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        name, val = token.split("=", 1)
        out[name.strip()] = float(val)
    return out


def index_rollouts(manifest: dict) -> dict[int, list[tuple[int, int]]]:
    """Map ``abs_frame_idx -> list of (rollout_id, position_within_rollout)``.

    Only positions ``>= n_conds`` are kept (the actual predictions).
    """
    n_conds = manifest["n_conds"]
    index: dict[int, list[tuple[int, int]]] = {}
    for r in manifest["rollouts"]:
        start = r["start"]
        for pos, abs_idx in enumerate(r["abs_indices"]):
            if pos < n_conds:
                continue
            index.setdefault(abs_idx, []).append((r["rollout"], start))
    return index


def choose_current_and_priors(
    coverage: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Pick the "current" prediction (latest rollout) and the prior ones.

    ``coverage`` is the list of ``(rollout_id, start_frame)`` rollouts that have
    a prediction for the same target frame. The latest start = freshest context;
    earlier starts = "priors" with deeper projection (UMAD's Eq. 5 setup).
    """
    sorted_cov = sorted(coverage, key=lambda rs: rs[1])  # ascending start
    current = sorted_cov[-1]
    priors = sorted_cov[:-1]
    return current, priors


def _load_pred(inference_dir: str, rollout_id: int, abs_idx: int,
               device: torch.device) -> torch.Tensor:
    path = os.path.join(inference_dir, "pred", f"rollout_{rollout_id:02d}",
                        f"{abs_idx:06d}.png")
    return image_io.load_frame(path, device=device)


def _load_real(inference_dir: str, abs_idx: int,
               device: torch.device) -> torch.Tensor:
    path = os.path.join(inference_dir, "real", f"{abs_idx:06d}.png")
    return image_io.load_frame(path, device=device)


def main() -> int:
    runtime.cap_cpu_threads(ARGS.threads)
    device = runtime.resolve_device(ARGS.device)
    print("[stage-B] runtime:")
    for line in runtime.describe_runtime(device).splitlines():
        print(f"  {line}")
    warn = runtime.multi_gpu_warning()
    if warn:
        print(warn, file=sys.stderr)

    # ---- load Stage A manifest -----------------------------------------------
    manifest_path = os.path.join(ARGS.inference_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest.json not found in {ARGS.inference_dir}", file=sys.stderr)
        return 1
    with open(manifest_path) as fh:
        manifest = json.load(fh)

    scenario_dir = ARGS.scenario or manifest["scenario"]
    if not os.path.isdir(scenario_dir):
        print(f"ERROR: scenario dir {scenario_dir} not found", file=sys.stderr)
        return 1
    scenario_tag = manifest["scenario_tag"]
    target_h, target_w = manifest["frame_size_hw"]
    n_total = manifest["n_total_frames"]

    # ---- pick the frames to evaluate -----------------------------------------
    coverage = index_rollouts(manifest)
    candidate_indices = anovox.eval_frame_indices(n_total, stride=ARGS.eval_stride)
    eval_indices = [i for i in candidate_indices if i in coverage]
    if not eval_indices:
        print("ERROR: no eval-frame index has any prediction", file=sys.stderr)
        return 1
    if ARGS.max_eval_frames > 0:
        eval_indices = eval_indices[: ARGS.max_eval_frames]

    print(f"[stage-B] scenario {scenario_tag}: {n_total} frames; "
          f"evaluating {len(eval_indices)}/{len(candidate_indices)} "
          f"(stride {ARGS.eval_stride}, predictions cover "
          f"{sum(1 for i in candidate_indices if i in coverage)} of the candidates)")

    # ---- semantic ground truth ------------------------------------------------
    semantic_frames = anovox.find_semantic_frames(scenario_dir)
    rgb_frames = anovox.find_rgb_frames(scenario_dir)
    if len(rgb_frames) != n_total:
        print(f"WARNING: scenario has {len(rgb_frames)} RGB frames but manifest expected "
              f"{n_total}", file=sys.stderr)
    # Manifest stores `anovox_frame_ids[abs_idx]`; use it so we never depend on
    # filename sort matching the rollout indexing.
    anovox_ids: list[int] = manifest["anovox_frame_ids"]
    anomaly_ids: tuple[int, ...] = (
        tuple(int(x) for x in ARGS.anomaly_class_ids.split(","))
        if ARGS.anomaly_class_ids else anovox.ANOMALY_CLASS_IDS
    )

    # ---- output dir -----------------------------------------------------------
    run_tag = ARGS.tag or f"{ARGS.weights.replace(',', '_')}__{ARGS.refine}"
    out_dir = os.path.join(ARGS.out, scenario_tag, run_tag)
    os.makedirs(out_dir, exist_ok=True)
    for sub in ("fused", "refined", "gt", "panels"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # ---- Phase 3 + 4.3 pipelines ---------------------------------------------
    weights = parse_weights(ARGS.weights)
    pipe = AnomalyMapPipeline(
        weights=weights, device=device,
        # compute every map only when we'll actually look at it later; otherwise
        # skip VGG when ssim_pd / weights without pd are used (significant VRAM win).
        compute_all=False,
    )
    print(f"[stage-B] fusion weights = {pipe.weights}")

    segmenter: masks.Segmenter | None = None
    if ARGS.refine == "sam":
        segmenter = masks.SamSegmenter(
            checkpoint=ARGS.sam_ckpt,
            model_type=ARGS.sam_model_type,
            device=device,
        )
        print(f"[stage-B] mask refinement = SAM ({ARGS.sam_model_type}) "
              f"aggregate={ARGS.aggregate}")
    else:
        print("[stage-B] mask refinement = none (raw pixel maps evaluated)")

    # ---- run -----------------------------------------------------------------
    per_frame_rows: list[dict] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_scores_refined: list[np.ndarray] = []

    t_start = time.time()
    for k, abs_idx in enumerate(eval_indices):
        cov = coverage[abs_idx]
        (cur_rid, _), prior_list = choose_current_and_priors(cov)

        real = _load_real(ARGS.inference_dir, abs_idx, device)
        current_pred = _load_pred(ARGS.inference_dir, cur_rid, abs_idx, device)
        prior_preds = [
            _load_pred(ARGS.inference_dir, pr_rid, abs_idx, device)
            for pr_rid, _ in prior_list
        ]

        fused, _maps = pipe(real, current_pred, prior_preds=prior_preds)
        fused_np = fused.detach().float().cpu().numpy().astype(np.float32)

        # Ground truth aligned to the same crop+resize as the Vista frames.
        anovox_id = anovox_ids[abs_idx]
        semantic_path = semantic_frames.get(anovox_id)
        if semantic_path is None:
            print(f"  frame {abs_idx} (AnoVox {anovox_id}): no semantic frame; skipping",
                  file=sys.stderr)
            continue
        gt = anovox.load_gt_anomaly_mask(semantic_path, (target_h, target_w),
                                         anomaly_ids=anomaly_ids)

        # Refine if requested (SAM operates on uint8 RGB).
        refined_np: np.ndarray | None = None
        n_masks = 0
        bg_score = float("nan")
        if segmenter is not None:
            rgb_uint8 = (real.detach().float().cpu().clamp(0, 1)
                         .permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            mask_list = segmenter.segment(rgb_uint8)
            n_masks = len(mask_list)
            refined_np, _ = masks.refine_with_masks(
                fused_np, mask_list, aggregate=ARGS.aggregate
            )
            bg_score = masks.background_score(fused_np, mask_list, aggregate=ARGS.aggregate)

        # Per-frame metrics (informational; the headline result is pooled).
        per_frame_pixel = evaluation.compute_metrics(fused_np, gt)
        per_frame_refined = (
            evaluation.compute_metrics(refined_np, gt) if refined_np is not None else None
        )

        all_scores.append(fused_np.reshape(-1))
        all_labels.append(gt.reshape(-1).astype(np.uint8))
        if refined_np is not None:
            all_scores_refined.append(refined_np.reshape(-1))

        per_frame_rows.append({
            "abs_idx": abs_idx,
            "anovox_frame_id": anovox_id,
            "current_rollout": cur_rid,
            "n_prior_rollouts": len(prior_preds),
            "n_anomaly_pixels": int(gt.sum()),
            "pixel_auroc": per_frame_pixel.auroc,
            "pixel_ap": per_frame_pixel.ap,
            "pixel_fpr95": per_frame_pixel.fpr95,
            "refined_auroc": per_frame_refined.auroc if per_frame_refined else None,
            "refined_ap": per_frame_refined.ap if per_frame_refined else None,
            "refined_fpr95": per_frame_refined.fpr95 if per_frame_refined else None,
            "n_sam_masks": n_masks,
            "sam_background_score": bg_score,
        })

        if not ARGS.no_save_maps:
            np.save(os.path.join(out_dir, "fused", f"{abs_idx:06d}.npy"), fused_np)
            np.save(os.path.join(out_dir, "gt", f"{abs_idx:06d}.npy"), gt)
            if refined_np is not None:
                np.save(os.path.join(out_dir, "refined", f"{abs_idx:06d}.npy"), refined_np)

        if k < ARGS.n_vis:
            real_rgb = (real.detach().float().cpu().clamp(0, 1)
                        .permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            gt_rgb = np.stack([gt.astype(np.uint8) * 255] * 3, axis=-1)
            panels = [
                ("real", real_rgb),
                ("fused", image_io.colorize_map(torch.from_numpy(fused_np))),
            ]
            if refined_np is not None:
                panels.append(("refined", image_io.colorize_map(torch.from_numpy(refined_np))))
            panels.append(("GT anomaly", gt_rgb))
            image_io.save_comparison_panel(
                panels,
                os.path.join(out_dir, "panels", f"{abs_idx:06d}.png"),
                title=(f"abs={abs_idx} AnoVox={anovox_id} "
                       f"rollout={cur_rid} priors={len(prior_preds)} "
                       f"gt_px={int(gt.sum())}"),
            )

        print(f"[{k + 1:3d}/{len(eval_indices)}] abs={abs_idx} (AnoVox {anovox_id}) "
              f"rollout={cur_rid} priors={len(prior_preds)} gt_px={int(gt.sum())} "
              f"AP={per_frame_pixel.ap:.4f} AUROC={per_frame_pixel.auroc:.4f}"
              + (f" refAP={per_frame_refined.ap:.4f}" if per_frame_refined else ""))

    if not all_scores:
        print("ERROR: no frames evaluated", file=sys.stderr)
        return 1

    # ---- pooled (UMAD-style) metrics -----------------------------------------
    pooled_scores = np.concatenate(all_scores)
    pooled_labels = np.concatenate(all_labels)
    pixel_metrics = evaluation.compute_metrics(pooled_scores, pooled_labels)

    refined_metrics: evaluation.AnomalyScores | None = None
    if all_scores_refined:
        pooled_refined = np.concatenate(all_scores_refined)
        refined_metrics = evaluation.compute_metrics(pooled_refined, pooled_labels)

    elapsed = time.time() - t_start

    # ---- write outputs --------------------------------------------------------
    csv_path = os.path.join(out_dir, "per_frame.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame_rows)

    metrics_payload = {
        "scenario_tag": scenario_tag,
        "weights": pipe.weights,
        "refine_backend": ARGS.refine,
        "aggregate": ARGS.aggregate if ARGS.refine != "none" else None,
        "eval_stride": ARGS.eval_stride,
        "n_eval_frames": len(per_frame_rows),
        "n_pixels_pooled": int(pooled_scores.size),
        "n_anomaly_pixels_pooled": int(pooled_labels.sum()),
        "pixel_metrics": pixel_metrics.to_dict(),
        "refined_metrics": refined_metrics.to_dict() if refined_metrics else None,
        "anomaly_class_ids": list(anomaly_ids),
        "total_seconds": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "global_metrics.json"), "w") as fh:
        json.dump(metrics_payload, fh, indent=2)

    config_payload = {
        "inference_dir": ARGS.inference_dir,
        "scenario_dir": scenario_dir,
        "manifest": {k: manifest[k] for k in
                     ("scenario_tag", "n_total_frames", "n_frames_per_rollout",
                      "n_conds", "rollout_stride", "frame_size_hw",
                      "n_steps", "cfg_scale")},
        "args": vars(ARGS),
    }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(config_payload, fh, indent=2, default=str)

    print()
    print(f"[stage-B] pooled pixel metrics: {pixel_metrics.as_table()}")
    if refined_metrics:
        print(f"[stage-B] pooled refined    : {refined_metrics.as_table()}")
    print(f"[stage-B] {len(per_frame_rows)} frames in {elapsed / 60:.1f} min")
    print(f"[stage-B] output in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
