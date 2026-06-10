#!/usr/bin/env python
"""Stage B driver: anomaly pipeline over all 16 benchmark scenarios + pooled metric.

Runs ``scripts/run_anomaly_pipeline.py`` once per benchmark scenario (sequentially,
on one GPU), then pools the per-pixel anomaly scores and ground-truth labels across
*all* scenarios into a single project-wide AUROC/AP/FPR95 -- the UMAD-style headline
number that averages out the single-frame dominance seen on any one scenario.

Scenario selection: every ``outputs/anovox-inference/Scenario_*`` whose manifest
records a source scenario under ``--unified-dir`` (the curated 16). The old
single-scenario test (Scenario_e2f08f8f, from a different batch) is excluded
automatically because its manifest points elsewhere.

Resumable: a scenario whose ``<run_tag>/global_metrics.json`` already exists is
skipped on re-run (use --force to override). Pooling always re-reads the saved
fused/gt/refined .npy maps, so it reflects whatever is on disk.

Output:
    outputs/anomaly-pipeline/<tag>/<run_tag>/...     per-scenario (from the pipeline)
    outputs/anomaly-pipeline/_pooled/<run_tag>.json  pooled headline + per-scenario table

Usage:
    PYTHONPATH=src external/Vista/.venv/bin/python scripts/run_stage_b_all.py \\
        --gpu 1 --weights mse_ssim_pd_td --refine sam --aggregate p99
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from vista_umad import evaluation  # noqa: E402

PIPELINE = os.path.join(REPO, "scripts", "run_anomaly_pipeline.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--inference-base",
                   default=os.path.join(REPO, "outputs", "anovox-inference"))
    p.add_argument("--unified-dir",
                   default=os.path.join(REPO, "data", "anovox", "Outputs",
                                        "Final_Output_2026_05_26-21_18"),
                   help="curated scenario dir; selects which inference outputs count "
                        "as the benchmark (the 16)")
    p.add_argument("--out", default=os.path.join(REPO, "outputs", "anomaly-pipeline"))
    p.add_argument("--pooled-out",
                   default=os.path.join(REPO, "outputs", "anomaly-pipeline", "_pooled"))

    # Pipeline settings (locked defaults from docs/evaluation-handoff.md).
    p.add_argument("--weights", default="mse_ssim_pd_td")
    p.add_argument("--refine", default="sam", choices=["sam", "none"])
    p.add_argument("--aggregate", default="p99", choices=["max", "mean", "p99", "p95"])
    p.add_argument("--eval-stride", type=int, default=10)
    p.add_argument("--tag", default=None,
                   help="run_tag (subdir under each scenario); defaults to the same "
                        "'<weights>__<refine>' the pipeline derives")

    # Runtime.
    p.add_argument("--gpu", type=int, default=1, help="physical GPU index (default 1)")
    p.add_argument("--threads", type=int, default=4)

    # Control.
    p.add_argument("--skip-run", action="store_true",
                   help="don't run the pipeline; only (re)pool existing outputs")
    p.add_argument("--force", action="store_true",
                   help="re-run the pipeline even if global_metrics.json exists")
    return p.parse_args()


def run_tag_for(args: argparse.Namespace) -> str:
    if args.tag:
        return args.tag
    return f"{args.weights.replace(',', '_')}__{args.refine}"


def select_scenarios(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return sorted (tag, inference_dir) for inference outputs in the benchmark."""
    unified = os.path.abspath(args.unified_dir)
    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(args.inference_base)):
        inf_dir = os.path.join(args.inference_base, name)
        manifest_path = os.path.join(inf_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        # Compare the *recorded* scenario path (not realpath): symlinked scenarios
        # were inferred via their unified-dir path, so their manifest still names it.
        if os.path.dirname(manifest["scenario"]) == unified:
            out.append((manifest["scenario_tag"], inf_dir))
    return out


def run_one(inf_dir: str, run_tag: str, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, PIPELINE,
        "--inference-dir", inf_dir,
        "--out", args.out,
        "--tag", run_tag,
        "--weights", args.weights,
        "--refine", args.refine,
        "--aggregate", args.aggregate,
        "--eval-stride", str(args.eval_stride),
        "--gpu", str(args.gpu),
        "--threads", str(args.threads),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, env=env).returncode


def pool_scenario(scn_run_dir: str, with_refined: bool):
    """Load and flatten this scenario's per-frame fused/gt(/refined) maps.

    Returns (fused_list, label_list, refined_list) of 1-D float/uint8 arrays, one
    entry per evaluated frame. gt files are the anchor; fused (and refined) are
    matched by identical filename.
    """
    gt_dir = os.path.join(scn_run_dir, "gt")
    fused_dir = os.path.join(scn_run_dir, "fused")
    refined_dir = os.path.join(scn_run_dir, "refined")
    fused_l, label_l, refined_l = [], [], []
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.endswith(".npy"):
            continue
        gt = np.load(os.path.join(gt_dir, fname)).reshape(-1).astype(np.uint8)
        fused = np.load(os.path.join(fused_dir, fname)).reshape(-1).astype(np.float32)
        label_l.append(gt)
        fused_l.append(fused)
        if with_refined:
            refined_l.append(
                np.load(os.path.join(refined_dir, fname)).reshape(-1).astype(np.float32)
            )
    return fused_l, label_l, refined_l


def main() -> int:
    args = parse_args()
    run_tag = run_tag_for(args)
    with_refined = args.refine == "sam"

    scenarios = select_scenarios(args)
    print(f"[stage-B all] run_tag={run_tag}  {len(scenarios)} benchmark scenarios")
    for tag, _ in scenarios:
        print(f"    {tag}")
    if len(scenarios) != 16:
        print(f"WARNING: expected 16 benchmark scenarios, found {len(scenarios)}",
              file=sys.stderr)

    # ---- run the pipeline per scenario (resumable) ---------------------------
    t_all = time.time()
    if not args.skip_run:
        for i, (tag, inf_dir) in enumerate(scenarios, 1):
            gm = os.path.join(args.out, tag, run_tag, "global_metrics.json")
            if os.path.isfile(gm) and not args.force:
                print(f"[skip] {tag}: global_metrics.json exists ({i}/{len(scenarios)})")
                continue
            print(f"[run]  {tag} ({i}/{len(scenarios)}) ...")
            t0 = time.time()
            rc = run_one(inf_dir, run_tag, args)
            print(f"[done] {tag} rc={rc} in {time.time() - t0:.0f}s")
            if rc != 0:
                print(f"ERROR: pipeline failed on {tag} (rc={rc}); aborting before pool",
                      file=sys.stderr)
                return rc

    # ---- pool across all scenarios -------------------------------------------
    print("\n[stage-B all] pooling pixel scores across scenarios ...")
    all_fused, all_labels, all_refined = [], [], []
    per_scenario = []
    for tag, _ in scenarios:
        scn_run_dir = os.path.join(args.out, tag, run_tag)
        gm_path = os.path.join(scn_run_dir, "global_metrics.json")
        if not os.path.isdir(scn_run_dir):
            print(f"WARNING: no output for {tag}; skipping in pool", file=sys.stderr)
            continue
        f_l, l_l, r_l = pool_scenario(scn_run_dir, with_refined)
        all_fused.extend(f_l)
        all_labels.extend(l_l)
        all_refined.extend(r_l)
        if os.path.isfile(gm_path):
            with open(gm_path) as fh:
                gm = json.load(fh)
            per_scenario.append({
                "scenario_tag": tag,
                "n_eval_frames": gm.get("n_eval_frames"),
                "n_anomaly_pixels_pooled": gm.get("n_anomaly_pixels_pooled"),
                "pixel_auroc": gm["pixel_metrics"]["auroc"],
                "pixel_ap": gm["pixel_metrics"]["ap"],
                "refined_auroc": (gm["refined_metrics"] or {}).get("auroc")
                if gm.get("refined_metrics") else None,
                "refined_ap": (gm["refined_metrics"] or {}).get("ap")
                if gm.get("refined_metrics") else None,
            })

    if not all_fused:
        print("ERROR: no per-frame maps found to pool", file=sys.stderr)
        return 1

    pooled_scores = np.concatenate(all_fused)
    pooled_labels = np.concatenate(all_labels)
    pixel_metrics = evaluation.compute_metrics(pooled_scores, pooled_labels)
    refined_metrics = None
    if with_refined and all_refined:
        pooled_refined = np.concatenate(all_refined)
        refined_metrics = evaluation.compute_metrics(pooled_refined, pooled_labels)

    elapsed = time.time() - t_all
    os.makedirs(args.pooled_out, exist_ok=True)
    payload = {
        "run_tag": run_tag,
        "weights": args.weights,
        "refine_backend": args.refine,
        "aggregate": args.aggregate if args.refine != "none" else None,
        "eval_stride": args.eval_stride,
        "n_scenarios": len(per_scenario),
        "n_eval_frames_total": len(all_fused),
        "n_pixels_pooled": int(pooled_scores.size),
        "n_anomaly_pixels_pooled": int(pooled_labels.sum()),
        "pooled_pixel_metrics": pixel_metrics.to_dict(),
        "pooled_refined_metrics": refined_metrics.to_dict() if refined_metrics else None,
        "per_scenario": per_scenario,
        "total_seconds": round(elapsed, 1),
    }
    pooled_path = os.path.join(args.pooled_out, f"{run_tag}.json")
    with open(pooled_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    # ---- report --------------------------------------------------------------
    print(f"\n{'scenario':<22} {'frames':>6} {'gt_px':>8} {'pix_AUROC':>10} {'ref_AUROC':>10}")
    for r in per_scenario:
        ref = f"{r['refined_auroc']:.4f}" if r["refined_auroc"] is not None else "   -   "
        pix = f"{r['pixel_auroc']:.4f}" if r["pixel_auroc"] == r["pixel_auroc"] else "  nan  "
        print(f"{r['scenario_tag']:<22} {r['n_eval_frames']:>6} "
              f"{r['n_anomaly_pixels_pooled']:>8} {pix:>10} {ref:>10}")
    print()
    print(f"[stage-B all] POOLED pixel  : {pixel_metrics.as_table()}")
    if refined_metrics:
        print(f"[stage-B all] POOLED refined: {refined_metrics.as_table()}")
    print(f"[stage-B all] {len(all_fused)} frames from {len(per_scenario)} scenarios "
          f"in {elapsed / 60:.1f} min")
    print(f"[stage-B all] pooled metrics -> {pooled_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
