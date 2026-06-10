"""Shared data model for source clip resolution (Stage B) and the 10 fps grid (Stage C).

A resolver turns one CODA scene into a :class:`SourceSequence`: the timestamp-sorted
run of native source frames around the anchor, with the anchor's position flagged.
:func:`build_clip` then samples that sequence onto VISTA's 10 fps grid, walking
backwards from the anchor until ``history_pool_frames`` are collected or the sequence
start is reached (``truncated``). Keeping this source-agnostic means ONCE/KITTI/nuScenes
only have to produce a list of frames + timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["FrameRef", "SourceSequence", "GridFrame", "Clip", "build_clip", "context_indices"]


@dataclass
class FrameRef:
    """One native source frame.

    ``t`` is a monotonic timestamp in **seconds** within the sequence (relative origin
    is arbitrary; only deltas matter). ``path`` is the on-disk pixel path, or ``None``
    when the source media has not been fetched.
    """

    key: str               # source-native frame id (ms timestamp / frame number / sample_data token)
    t: float               # seconds, monotonic within the sequence
    path: str | None       # on-disk source pixel path, or None if not fetched


@dataclass
class SourceSequence:
    scene_id: str
    frames: list[FrameRef]          # timestamp-sorted
    anchor_idx: int                 # index of the anchor frame within ``frames``
    native_fps_estimate: float
    available: bool                 # are the source pixels on disk?
    notes: list[str] = field(default_factory=list)


@dataclass
class GridFrame:
    grid_index: int                 # 0 = anchor, increasing into the past
    rel_t: float                    # seconds relative to anchor (<= 0)
    src: FrameRef


@dataclass
class Clip:
    frames: list[GridFrame]         # grid order: index 0 = anchor, then 1,2,... into the past
    truncated: bool                 # True if the sequence start was hit before filling the pool
    native_fps_estimate: float
    notes: list[str] = field(default_factory=list)

    @property
    def anchor(self) -> GridFrame:
        return self.frames[0]


def build_clip(seq: SourceSequence, fps: int, history_pool_frames: int) -> Clip:
    """Sample ``seq`` onto the 10 fps grid: anchor + ``history_pool_frames`` past frames.

    For grid tick ``i`` (i>=1) we pick the source frame nearest to ``t_anchor - i/fps``,
    searching only frames at or before the anchor (we never sample the future). A tick
    is dropped once we run out of past frames, at which point the clip is ``truncated``.
    The same native frame is never selected twice (guards against under-sampled sources).
    """
    dt = 1.0 / fps
    anchor = seq.frames[seq.anchor_idx]
    grid = [GridFrame(grid_index=0, rel_t=0.0, src=anchor)]

    past = seq.frames[: seq.anchor_idx + 1]   # anchor and everything before it, sorted
    truncated = False
    used: set[int] = {seq.anchor_idx}
    for i in range(1, history_pool_frames + 1):
        target_t = anchor.t - i * dt
        # nearest past frame to target_t that we have not already taken
        best_j, best_d = None, None
        for j in range(len(past)):
            if j in used:
                continue
            d = abs(past[j].t - target_t)
            if best_d is None or d < best_d:
                best_j, best_d = j, d
        if best_j is None:
            truncated = True
            break
        used.add(best_j)
        grid.append(GridFrame(grid_index=i, rel_t=round(past[best_j].t - anchor.t, 4), src=past[best_j]))

    return Clip(frames=grid, truncated=truncated, native_fps_estimate=seq.native_fps_estimate, notes=list(seq.notes))


def context_indices(horizon_s: float, fps: int) -> list[int]:
    """Grid indices of the 3 context frames for ``horizon_s``, ordered oldest->newest.

    Context = the 3 consecutive 10 fps frames ending ``horizon_s`` before the anchor.
    For horizon 1.0 s at 10 fps this is grid indices [12, 11, 10] (rel_t -1.2, -1.1, -1.0).
    """
    k = round(horizon_s * fps)
    return [k + 2, k + 1, k]
