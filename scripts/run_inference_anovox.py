#!/usr/bin/env python
"""Stage A -- Vista rollout inference on an AnoVox scenario (overlapping rollouts).

Runs Vista in predict-and-compare mode over the real, consecutive CARLA frames of an
AnoVox scenario. Rollouts are *overlapping* (a fresh 25-frame window every
``--rollout-stride`` frames), so most frames are predicted by several rollouts at
different depths -- the setup UMAD's temporal-difference metric (eq. 5) needs.

Output (consumed by ``scripts/run_anomaly_pipeline.py``):
    <out>/<scenario>/
        real/<abs>.png                ground-truth frames, keyed by absolute index
        pred/rollout_<rr>/<abs>.png    Vista predictions (non-context frames only)
        manifest.json                 rollout layout + metadata

GPU selection follows the server docs -- ``--gpu N`` sets CUDA_VISIBLE_DEVICES; the
process uses one GPU. See src/vista_umad/runtime.py.

Usage:
    python scripts/run_inference_anovox.py --gpu 1 --low-vram --height 512 --width 832
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VISTA_DIR = os.path.join(REPO_ROOT, "external", "Vista")
DEFAULT_SCENARIO = os.path.join(
    REPO_ROOT, "data", "anovox", "Outputs", "Final_Output_2026_05_19-11_30",
    "Scenario_e2f08f8f-06a2-44f6-8a26-d7bdbdc44c67",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default=DEFAULT_SCENARIO, help="AnoVox scenario directory")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs", "anovox-inference"))
    p.add_argument("--gpu", type=int, default=1, metavar="N",
                   help="physical GPU index (sets CUDA_VISIBLE_DEVICES; default 1)")
    p.add_argument("--threads", type=int, default=4, help="CPU thread cap")
    p.add_argument("--height", type=int, default=576, help="Vista target height")
    p.add_argument("--width", type=int, default=1024, help="Vista target width")
    p.add_argument("--n-frames", type=int, default=25, help="frames per Vista rollout (fixed at 25)")
    p.add_argument("--n-conds", type=int, default=3, metavar="K",
                   help="real frames Vista conditions on before predicting the rest (default 3)")
    p.add_argument("--rollout-stride", type=int, default=12, metavar="S",
                   help="frames between consecutive rollout starts; < predicted span (22) "
                        "so frames are predicted by multiple rollouts -> enables TD (default 12)")
    p.add_argument("--n-steps", type=int, default=50, help="diffusion sampling steps")
    p.add_argument("--cfg-scale", type=float, default=2.5, help="classifier-free guidance scale")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--max-rollouts", type=int, default=0, help="0 = all; limit for a quick test")
    p.add_argument("--low-vram", action="store_true", help="enable Vista CPU offload")
    return p.parse_args()


ARGS = parse_args()

# --- resource confinement: must precede every torch / CUDA import -------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(ARGS.threads))

# Vista resolves config/ckpt by relative path, so run from its directory.
os.chdir(VISTA_DIR)
sys.path.insert(0, VISTA_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402

import init_proj_path  # noqa: E402,F401  (adds Vista to sys.path)
from sample import load_img  # noqa: E402
from sample_utils import (  # noqa: E402
    do_sample, init_embedder_options, init_model, init_sampling, set_lowvram_mode,
)

from vista_umad import runtime  # noqa: E402

VERSION_DICT = {"config": "configs/inference/vista.yaml", "ckpt": "ckpts/vista.safetensors"}


def find_rgb_frames(scenario_dir: str) -> list[str]:
    """Return the sorted list of RGB-CAM frame paths for an AnoVox scenario."""
    rgb_dirs = [d for d in os.listdir(scenario_dir) if d.startswith("RGB-CAM")]
    if not rgb_dirs:
        raise FileNotFoundError(f"no RGB-CAM directory in {scenario_dir}")
    rgb_dir = os.path.join(scenario_dir, rgb_dirs[0])
    frames = sorted(
        os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir) if f.endswith(".png")
    )
    if not frames:
        raise FileNotFoundError(f"no PNG frames in {rgb_dir}")
    return frames


def anovox_frame_id(path: str) -> int:
    """Extract the integer AnoVox frame id from an RGB-CAM filename."""
    return int(os.path.splitext(path)[0].rsplit("_", 1)[1])


def rollout_starts(n_total: int, n_frames: int, stride: int) -> list[int]:
    """Overlapping rollout start indices; the last window is clamped to the scenario end."""
    starts = list(range(0, n_total - n_frames + 1, stride))
    if not starts:
        raise ValueError(f"scenario has {n_total} frames, need >= {n_frames}")
    if starts[-1] + n_frames < n_total:
        starts.append(n_total - n_frames)
    return starts


def run_vista_rollout(model, sampler_args: dict, frame_paths: list[str],
                      height: int, width: int, n_conds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Vista rollout; return (predicted, real) frame stacks in [0, 1], [T,3,H,W].

    Vista is conditioned on the first ``n_conds`` real frames (their latents are clamped
    to ground truth during denoising) and predicts the rest.
    """
    img_seq = [load_img(p, height, width) for p in frame_paths]
    images = torch.stack(img_seq)

    unique_keys = set(e.input_key for e in model.conditioner.embedders)
    value_dict = init_embedder_options(unique_keys)
    cond_img = img_seq[0][None]
    value_dict["cond_frames_without_noise"] = cond_img
    value_dict["cond_aug"] = 0.0
    value_dict["cond_frames"] = cond_img + 0.0 * torch.randn_like(cond_img)

    sampler = init_sampling(guider="VanillaCFG", **sampler_args)
    uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory",
               "speed", "angle", "goal"]
    out = do_sample(
        images, model, sampler, value_dict,
        num_rounds=1, num_frames=sampler_args["num_frames"],
        force_uc_zero_embeddings=uc_keys, initial_cond_indices=list(range(n_conds)),
    )
    samples, _samples_z, inputs = out
    predicted = samples.detach().float().cpu().clamp(0, 1)            # already [0, 1]
    real = ((inputs.detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)  # [-1,1] -> [0,1]
    return predicted, real


def _save_frame(tensor: torch.Tensor, path: str) -> None:
    array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> int:
    runtime.cap_cpu_threads(ARGS.threads)
    device = runtime.resolve_device("auto")
    print("[stage-A] runtime:")
    for line in runtime.describe_runtime(device).splitlines():
        print(f"  {line}")
    if device.type != "cuda":
        print("ERROR: no CUDA device available", file=sys.stderr)
        return 1

    frame_paths = find_rgb_frames(ARGS.scenario)
    n_total = len(frame_paths)
    n_per = ARGS.n_frames
    starts = rollout_starts(n_total, n_per, ARGS.rollout_stride)
    if ARGS.max_rollouts > 0:
        starts = starts[:ARGS.max_rollouts]

    scenario_tag = os.path.basename(ARGS.scenario).split("-")[0]  # Scenario_e2f08f8f
    out_dir = os.path.join(ARGS.out, scenario_tag)
    real_dir = os.path.join(out_dir, "real")
    os.makedirs(real_dir, exist_ok=True)
    print(f"[stage-A] scenario {scenario_tag}: {n_total} frames, "
          f"{len(starts)} overlapping rollouts (stride {ARGS.rollout_stride}, k={ARGS.n_conds})")

    set_lowvram_mode(ARGS.low_vram)
    print("[stage-A] loading Vista model ...")
    t0 = time.time()
    model = init_model(VERSION_DICT)
    print(f"[stage-A] model loaded in {time.time() - t0:.0f}s")
    sampler_args = {"steps": ARGS.n_steps, "cfg_scale": ARGS.cfg_scale, "num_frames": n_per}

    rollouts_meta: list[dict] = []
    t_start = time.time()

    for r, start in enumerate(starts):
        frame_slice = frame_paths[start:start + n_per]
        abs_indices = list(range(start, start + n_per))
        ids = [anovox_frame_id(p) for p in frame_slice]
        pred_dir = os.path.join(out_dir, "pred", f"rollout_{r:02d}")
        os.makedirs(pred_dir, exist_ok=True)

        print(f"[stage-A] rollout {r}/{len(starts) - 1}: frames {start}..{start + n_per - 1} "
              f"(AnoVox {ids[0]}..{ids[-1]})")
        seed_everything(ARGS.seed + r)
        t_roll = time.time()
        try:
            predicted, real = run_vista_rollout(model, sampler_args, frame_slice,
                                                ARGS.height, ARGS.width, ARGS.n_conds)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"  OOM on rollout {r}: {exc}", file=sys.stderr)
            torch.cuda.empty_cache()
            continue
        except Exception:  # noqa: BLE001 -- surface failure, keep other rollouts
            import traceback
            print(f"  rollout {r} FAILED:\n{traceback.format_exc()}", file=sys.stderr)
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()

        for pos, abs_idx in enumerate(abs_indices):
            real_path = os.path.join(real_dir, f"{abs_idx:06d}.png")
            if not os.path.exists(real_path):  # real frame is rollout-independent
                _save_frame(real[pos], real_path)
            if pos >= ARGS.n_conds:  # context frames == real, no prediction to save
                _save_frame(predicted[pos], os.path.join(pred_dir, f"{abs_idx:06d}.png"))

        rollouts_meta.append({
            "rollout": r,
            "start": start,
            "abs_indices": abs_indices,
            "anovox_frame_ids": ids,
        })
        print(f"  done in {time.time() - t_roll:.0f}s")

    if not rollouts_meta:
        print("ERROR: no rollouts completed", file=sys.stderr)
        return 1

    elapsed = time.time() - t_start
    manifest = {
        "scenario": ARGS.scenario,
        "scenario_tag": scenario_tag,
        "n_total_frames": n_total,
        "n_frames_per_rollout": n_per,
        "n_conds": ARGS.n_conds,
        "rollout_stride": ARGS.rollout_stride,
        "frame_size_hw": [ARGS.height, ARGS.width],
        "n_steps": ARGS.n_steps,
        "cfg_scale": ARGS.cfg_scale,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "anovox_frame_ids": [anovox_frame_id(p) for p in frame_paths],
        "rollouts": rollouts_meta,
        "total_seconds": round(elapsed, 1),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[stage-A] done: {len(rollouts_meta)} rollouts in {elapsed / 60:.1f} min")
    print(f"[stage-A] output in {out_dir}  (run run_anomaly_pipeline.py next)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
