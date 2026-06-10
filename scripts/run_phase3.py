#!/usr/bin/env python
"""Phase 3 demo / integration check -- run the anomaly-map pipeline on Vista output.

For one rollout it loads the ``real`` (ground-truth) and ``virtual`` (Vista-predicted)
frames produced by Vista's ``sample.py``, runs :class:`AnomalyMapPipeline` on every
frame pair, and writes:

* ``fused/``    -- raw float fused anomaly maps, one ``.npy`` per frame (Phase 4 input);
* ``panels/``   -- side-by-side visualizations (real | pred | per-metric heatmaps | overlay);
* ``scores.csv``-- per-frame mean of every difference metric and of the fused map;
* ``summary.png``-- per-frame anomaly-signal curve;
* ``metadata.json`` -- the run configuration.

GPU selection on a shared server (e.g. edward, 2x A40) -- two equivalent ways:

    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase3.py        # first A40
    python scripts/run_phase3.py --gpu 1                       # second A40

``--gpu N`` just sets ``CUDA_VISIBLE_DEVICES`` for you before CUDA initializes.
``--threads`` caps CPU threads (PyTorch otherwise grabs one per core). The process
only ever uses a single GPU. See src/vista_umad/runtime.py for details.

Caveat on the bundled outputs/ data: Phase 1-2 ran Vista in single-image -> video mode
(``--dataset IMG``), so the ``real`` frames are a static repeat of the seed image rather
than a true future. The maps this demo produces therefore measure Vista's predicted
drift from the seed, which exercises the full pipeline end-to-end but is not yet a
quantitative anomaly signal -- that needs Phase 4's multi-frame dataloader.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from vista_umad import image_io, runtime  # noqa: E402
from vista_umad.pipeline import AnomalyMapPipeline  # noqa: E402

# Heatmap panels are drawn for these metrics when present.
_PANEL_METRICS = ("abs", "mse", "ssim", "pd")

# CPU-thread environment variables to confine on a shared server.
_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_real = os.path.join(REPO_ROOT, "outputs", "vista-sample", "real", "images")
    default_virtual = os.path.join(REPO_ROOT, "outputs", "vista-sample", "virtual", "images")
    parser.add_argument("--real", default=default_real, help="directory of ground-truth frames")
    parser.add_argument("--virtual", default=default_virtual, help="directory of Vista-predicted frames")
    parser.add_argument("--rollout", type=int, default=0, help="rollout index to process")
    parser.add_argument("--prefix", default="IMG", help="dataset filename prefix (IMG / NUSCENES)")
    parser.add_argument("--weights", default="ssim_pd",
                        help="fusion preset name or comma list like 'ssim=0.5,pd=0.5'")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "phase3-demo"),
                        help="output directory")
    parser.add_argument("--tag", default=None, help="output sub-folder name (auto if omitted)")

    res = parser.add_argument_group("resource confinement (shared server)")
    res.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                     help="run on GPU or CPU (default: auto)")
    res.add_argument("--gpu", type=int, default=None, metavar="N",
                     help="physical GPU index to use; sets CUDA_VISIBLE_DEVICES=N. "
                          "Omit to honour an already-set CUDA_VISIBLE_DEVICES.")
    res.add_argument("--threads", type=int, default=4, metavar="N",
                     help="cap CPU threads (default: 4; 0 = leave PyTorch default)")
    res.add_argument("--gpu-mem-fraction", type=float, default=0.0, metavar="F",
                     help="cap this process to fraction F in (0,1) of GPU memory (0 = no cap)")

    parser.add_argument("--max-frames", type=int, default=0, help="limit frames (0 = all)")
    parser.add_argument("--panel-every", type=int, default=6, help="save a panel every N frames")
    return parser.parse_args()


def confine_resources(args: argparse.Namespace) -> None:
    """Apply process-level resource limits. Must run before the first CUDA call.

    Sets ``CUDA_VISIBLE_DEVICES`` from ``--gpu`` (PyTorch reads it when CUDA first
    initializes) and pins the CPU-thread environment variables so child libraries
    stay within the reservation too.
    """
    if args.gpu is not None:
        existing = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing not in (None, "", str(args.gpu)):
            raise SystemExit(
                f"--gpu {args.gpu} conflicts with CUDA_VISIBLE_DEVICES={existing}. "
                f"Pass only one, or set them to the same value."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.threads and args.threads > 0:
        for var in _THREAD_ENV_VARS:
            os.environ.setdefault(var, str(args.threads))


def resolve_weights(spec: str) -> str | dict[str, float]:
    """Parse the --weights argument: a preset name, or 'name=w,name=w' pairs."""
    if "=" not in spec:
        return spec
    weights: dict[str, float] = {}
    for token in spec.split(","):
        name, _, value = token.partition("=")
        weights[name.strip()] = float(value)
    return weights


def main() -> int:
    args = parse_args()

    # Resource confinement first -- before any CUDA call is made.
    confine_resources(args)
    runtime.cap_cpu_threads(args.threads)
    device = runtime.resolve_device(args.device)
    runtime.set_gpu_memory_fraction(args.gpu_mem_fraction, device)

    warning = runtime.multi_gpu_warning() if args.device != "cpu" else None
    if warning:
        print(f"[phase3] {warning}", file=sys.stderr)
    print("[phase3] runtime:")
    for line in runtime.describe_runtime(device).splitlines():
        print(f"  {line}")

    for label, path in (("real", args.real), ("virtual", args.virtual)):
        if not os.path.isdir(path):
            print(f"ERROR: {label} directory not found: {path}", file=sys.stderr)
            return 1

    tag = args.tag or f"{os.path.basename(os.path.dirname(os.path.dirname(args.real)))}_r{args.rollout:02d}"
    out_dir = os.path.join(args.out, tag)
    fused_dir = os.path.join(out_dir, "fused")
    panel_dir = os.path.join(out_dir, "panels")
    os.makedirs(fused_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)

    print(f"[phase3] weights={args.weights}  rollout={args.rollout}")
    real = image_io.load_rollout(args.real, args.rollout, prefix=args.prefix, device=device)
    virtual = image_io.load_rollout(args.virtual, args.rollout, prefix=args.prefix, device=device)
    if real.shape != virtual.shape:
        print(f"ERROR: real {tuple(real.shape)} != virtual {tuple(virtual.shape)}", file=sys.stderr)
        return 1

    n_frames = real.shape[0]
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    print(f"[phase3] loaded {n_frames} frame pairs at {tuple(real.shape[-2:])} (H x W)")

    pipeline = AnomalyMapPipeline(weights=resolve_weights(args.weights), device=device)

    rows: list[dict[str, float]] = []
    fused_means: list[float] = []
    abs_means: list[float] = []
    t_start = time.time()

    for t in range(n_frames):
        fused, maps = pipeline(real[t], virtual[t])

        fused_np = fused.detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(fused_dir, f"{args.prefix}_{args.rollout:06d}_{t:04d}.npy"), fused_np)

        row = {"frame": t, "fused_mean": float(fused_np.mean()), "fused_max": float(fused_np.max())}
        for name, amap in maps.items():
            row[f"{name}_mean"] = float(amap.mean().item())
        rows.append(row)
        fused_means.append(row["fused_mean"])
        abs_means.append(row.get("abs_mean", float("nan")))

        if args.panel_every > 0 and (t % args.panel_every == 0 or t == n_frames - 1):
            panels = [
                ("real frame t", _to_uint8(real[t])),
                ("Vista pred t-hat", _to_uint8(virtual[t])),
            ]
            for name in _PANEL_METRICS:
                if name in maps:
                    panels.append((f"D_{name.upper()}", image_io.colorize_map(maps[name])))
            panels.append(("fused overlay", image_io.overlay_map(real[t], fused, alpha=0.55)))
            image_io.save_comparison_panel(
                panels,
                os.path.join(panel_dir, f"frame_{t:04d}.png"),
                title=f"{tag}  frame {t}  fused mean={row['fused_mean']:.4f}",
            )

    elapsed = time.time() - t_start
    print(f"[phase3] processed {n_frames} frames in {elapsed:.1f}s "
          f"({elapsed / n_frames:.2f}s/frame)")

    # scores.csv
    csv_path = os.path.join(out_dir, "scores.csv")
    fieldnames = ["frame", "fused_mean", "fused_max"] + sorted(
        k for k in rows[0] if k not in ("frame", "fused_mean", "fused_max")
    )
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # summary.png -- two curves with different meanings:
    #   * raw mean |real - pred|  : absolute prediction-error magnitude (left axis).
    #     Near 0 at frame 0, rising with rollout depth -> the genuine drift signal.
    #   * mean fused score        : per-frame min-max normalized, so it is a relative
    #     within-frame statistic, NOT an absolute magnitude (right axis).
    fig, ax_left = plt.subplots(figsize=(9, 4))
    ax_left.plot(range(n_frames), abs_means, marker="o", ms=3, color="#1d3557",
                 label="raw mean |real - pred|")
    ax_left.set_xlabel("frame index t")
    ax_left.set_ylabel("raw mean absolute error", color="#1d3557")
    ax_left.tick_params(axis="y", labelcolor="#1d3557")
    ax_left.grid(alpha=0.3)
    ax_right = ax_left.twinx()
    ax_right.plot(range(n_frames), fused_means, marker="s", ms=3, color="#c1121f",
                  label="mean fused score (per-frame normalized)")
    ax_right.set_ylabel("mean fused anomaly score", color="#c1121f")
    ax_right.tick_params(axis="y", labelcolor="#c1121f")
    lines = ax_left.get_lines() + ax_right.get_lines()
    ax_left.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    ax_left.set_title(f"{tag} -- per-frame anomaly signal ({args.weights})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=120)
    plt.close(fig)

    # metadata.json
    metadata = {
        "tag": tag,
        "real_dir": args.real,
        "virtual_dir": args.virtual,
        "rollout": args.rollout,
        "prefix": args.prefix,
        "weights": pipeline.weights,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cpu_threads": torch.get_num_threads(),
        "n_frames": n_frames,
        "frame_size_hw": list(real.shape[-2:]),
        "seconds_per_frame": round(elapsed / n_frames, 4),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)

    peak = max(rows, key=lambda r: r.get("abs_mean", 0.0))
    print(f"[phase3] frame 0 raw mean |real-pred| = {abs_means[0]:.4f} "
          f"(near 0: virtual[0] is the decoded conditioning frame)")
    print(f"[phase3] peak raw mean |real-pred|    = {peak.get('abs_mean', 0.0):.4f} "
          f"at frame {peak['frame']} (prediction error grows with rollout depth)")
    print(f"[phase3] note: fused-map means are per-frame min-max normalized -- a "
          f"relative within-frame statistic, not an absolute magnitude")
    print(f"[phase3] wrote results to {out_dir}")
    return 0


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    """Convert a ``[3, H, W]`` float frame in ``[0, 1]`` to an ``[H, W, 3]`` uint8 array."""
    return (frame.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


if __name__ == "__main__":
    sys.exit(main())
