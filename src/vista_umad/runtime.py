"""Shared-server resource confinement: GPU selection and CPU-thread limits.

The edward server has 2x A40 GPUs and many CPU cores shared between users; a
process must stay inside its reservation. This module is the single place that
decides *which* GPU and *how many* CPU threads the project uses.

GPU selection follows the server documentation -- the ``CUDA_VISIBLE_DEVICES``
environment variable, which must be set **before the process starts**::

    CUDA_VISIBLE_DEVICES=0 python scripts/run_phase3.py    # first A40
    CUDA_VISIBLE_DEVICES=1 python scripts/run_phase3.py    # second A40

``scripts/run_phase3.py`` also accepts ``--gpu N``, which sets that variable for
you before CUDA initializes. Once the variable is set, PyTorch only ever sees
one GPU and it is addressed as ``cuda:0`` regardless of which physical card it
is. This project never uses more than one GPU.

CPU threads are capped with :func:`cap_cpu_threads`; PyTorch otherwise defaults
to one intra-op thread per core (64 on edward), far above a typical reservation.
"""

from __future__ import annotations

import os

import torch

__all__ = [
    "resolve_device",
    "cap_cpu_threads",
    "set_gpu_memory_fraction",
    "multi_gpu_warning",
    "describe_runtime",
]


def resolve_device(spec: str | int | torch.device = "auto") -> torch.device:
    """Resolve a device spec into a concrete :class:`torch.device`.

    Args:
        spec: one of ``"auto"`` (CUDA if available, else CPU), ``"cpu"``,
            ``"cuda"`` / ``"gpu"`` (-> ``cuda:0``), an explicit ``"cuda:N"``
            string, an integer GPU index, or a :class:`torch.device`.

    Note:
        Indices are interpreted *after* ``CUDA_VISIBLE_DEVICES`` masking. With
        ``CUDA_VISIBLE_DEVICES=1`` only one GPU is visible and it is ``cuda:0``.

    Raises:
        RuntimeError: if a CUDA device is requested but unavailable or the
            index is out of the visible range.
    """
    if isinstance(spec, torch.device):
        device = spec
    elif isinstance(spec, int):
        device = torch.device(f"cuda:{spec}")
    else:
        text = str(spec).strip().lower()
        if text in ("auto", ""):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if text == "cpu":
            return torch.device("cpu")
        if text in ("cuda", "gpu"):
            device = torch.device("cuda:0")
        else:
            device = torch.device(text)  # e.g. "cuda:1"

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() is False"
            )
        count = torch.cuda.device_count()
        index = device.index or 0
        if index >= count:
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise RuntimeError(
                f"requested cuda:{index} but only {count} GPU(s) visible "
                f"(CUDA_VISIBLE_DEVICES={cvd}). With the variable set, use cuda:0."
            )
        device = torch.device(f"cuda:{index}")
    return device


def cap_cpu_threads(n_threads: int) -> int:
    """Cap PyTorch's intra-op CPU thread pool.

    This sizes the pool that runs the compute kernels -- the dominant CPU
    consumer. It is safe to call repeatedly. The inter-op pool is left alone on
    purpose: ``torch.set_num_interop_threads`` can only be called once and only
    before any parallel work, so the OpenMP/MKL environment variables (exported
    before the process starts -- see the module docstring and .gitlab-ci.yml)
    are the right tool for the rest.

    Args:
        n_threads: maximum threads; ``0`` or negative leaves the default
            (one per core) untouched.

    Returns:
        The intra-op thread count in effect afterwards.
    """
    if n_threads and n_threads > 0:
        torch.set_num_threads(n_threads)
    return torch.get_num_threads()


def set_gpu_memory_fraction(fraction: float, device: torch.device) -> None:
    """Cap this process's share of GPU memory (best-effort, CUDA only).

    Args:
        fraction: fraction of total device memory in ``(0, 1)``; values outside
            that range are ignored (no cap).
        device: the target CUDA device.
    """
    if device.type == "cuda" and fraction and 0.0 < fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)


def multi_gpu_warning() -> str | None:
    """Return a warning string if more than one GPU is visible, else ``None``.

    On a shared server more than one visible GPU means the process could spill
    onto a card it has not reserved -- the caller should set
    ``CUDA_VISIBLE_DEVICES``.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return (
            f"WARNING: {torch.cuda.device_count()} GPUs are visible. On a shared "
            f"server set CUDA_VISIBLE_DEVICES (or pass --gpu) to confine this "
            f"process to one reserved GPU."
        )
    return None


def describe_runtime(device: torch.device) -> str:
    """Return a human-readable summary of the resources this process will use."""
    lines = [
        f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
    ]
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        lines.append(f"visible GPUs        = {count}")
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            marker = "  <- in use" if (device.type == "cuda" and (device.index or 0) == i) else ""
            lines.append(f"  cuda:{i} = {props.name}, {props.total_memory / 1e9:.0f} GB{marker}")
    else:
        lines.append("visible GPUs        = 0 (CUDA unavailable)")
    lines.append(f"device in use       = {device}")
    lines.append(f"CPU intra-op threads= {torch.get_num_threads()}")
    return "\n".join(lines)
