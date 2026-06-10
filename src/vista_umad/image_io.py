"""Frame loading and anomaly-map visualization helpers.

Vista's ``sample.py`` writes per-frame PNGs as ``{dataset}_{rollout:06d}_{frame:04d}.png``
under ``<save_dir>/real/images`` (the conditioning / ground-truth frames) and
``<save_dir>/virtual/images`` (Vista's predicted frames). These helpers load those PNGs
back into ``[0, 1]`` RGB tensors and render anomaly maps for qualitative inspection.
"""

from __future__ import annotations

import os
import re
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

__all__ = [
    "load_frame",
    "list_rollouts",
    "load_rollout",
    "colorize_map",
    "overlay_map",
    "save_comparison_panel",
]

_FRAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<rollout>\d{6})_(?P<frame>\d{4})\.png$")


def load_frame(
    path: str,
    size: tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load a single image as a ``[3, H, W]`` float tensor in ``[0, 1]``.

    Args:
        path: image file path.
        size: optional ``(height, width)`` to resize to (bilinear).
        device: device for the returned tensor.
    """
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if size is not None:
        # PIL.resize expects (width, height).
        image = image.resize((size[1], size[0]), resample=Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device)


def list_rollouts(images_dir: str) -> list[tuple[str, int, int]]:
    """List the rollouts found in a Vista ``images`` directory.

    Returns a sorted list of ``(prefix, rollout_index, frame_count)`` tuples.
    """
    counts: dict[tuple[str, int], int] = {}
    for name in os.listdir(images_dir):
        match = _FRAME_RE.match(name)
        if match:
            key = (match["prefix"], int(match["rollout"]))
            counts[key] = counts.get(key, 0) + 1
    return sorted((prefix, idx, n) for (prefix, idx), n in counts.items())


def load_rollout(
    images_dir: str,
    rollout_index: int,
    prefix: str = "IMG",
    n_frames: int | None = None,
    size: tuple[int, int] | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load all frames of one rollout as a ``[T, 3, H, W]`` tensor in ``[0, 1]``.

    Args:
        images_dir: a Vista ``.../images`` directory.
        rollout_index: the integer rollout id (the ``000000`` field of the filename).
        prefix: the dataset-name filename prefix (``"IMG"`` / ``"NUSCENES"``).
        n_frames: number of frames to load; inferred from disk when ``None``.
        size: optional ``(height, width)`` resize.
        device: device for the returned tensor.
    """
    if n_frames is None:
        n_frames = 0
        while os.path.exists(
            os.path.join(images_dir, f"{prefix}_{rollout_index:06d}_{n_frames:04d}.png")
        ):
            n_frames += 1
        if n_frames == 0:
            raise FileNotFoundError(
                f"no frames for rollout {rollout_index} (prefix {prefix!r}) in {images_dir}"
            )

    frames = [
        load_frame(
            os.path.join(images_dir, f"{prefix}_{rollout_index:06d}_{i:04d}.png"),
            size=size,
            device=device,
        )
        for i in range(n_frames)
    ]
    return torch.stack(frames, dim=0)


def colorize_map(
    anomaly_map: torch.Tensor,
    cmap: str = "inferno",
    vmin: float | None = 0.0,
    vmax: float | None = None,
) -> np.ndarray:
    """Render a ``[H, W]`` anomaly map to an ``[H, W, 3]`` uint8 RGB heatmap."""
    array = anomaly_map.detach().float().cpu().numpy()
    lo = float(np.min(array)) if vmin is None else vmin
    hi = float(np.max(array)) if vmax is None else vmax
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    normed = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](normed)  # [H, W, 4] in [0, 1]
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def overlay_map(
    frame: torch.Tensor,
    anomaly_map: torch.Tensor,
    alpha: float = 0.5,
    cmap: str = "inferno",
) -> np.ndarray:
    """Alpha-blend a colorized anomaly map over an RGB frame -> ``[H, W, 3]`` uint8."""
    base = (frame.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
    heat = colorize_map(anomaly_map, cmap=cmap).astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * heat
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_comparison_panel(
    panels: Sequence[tuple[str, np.ndarray]],
    path: str,
    title: str | None = None,
    dpi: int = 110,
) -> None:
    """Save a labelled row of images (each ``[H, W, 3]`` uint8) as a single figure."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
