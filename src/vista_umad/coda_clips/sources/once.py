"""ONCE source (PLAN.md B1).

ONCE is the one source whose corner-case (anchor) pixels ship *inside* CODA
(``images/<sequence_id>_<frame_id>.jpg``, ``frame_id`` = ms timestamp, front cam03).
So every ONCE scene has its anchor available with zero downloads and is redistributable.

The clip **history**, however, needs the surrounding cam03 frames, and ONCE's download
granularity is a per-split, per-camera tar -- there is no per-sequence fetch. CODA's
1057 ONCE scenes span **350 distinct sequences across all ONCE splits**, so pulling the
history means most of the ONCE camera archive (gated, hundreds of GB). We therefore:

* always resolve the anchor from CODA's ``images/`` (``available=True``, history depth 1);
* if extracted cam03 frames for a sequence are found under ``once.raw_cache/<sequence_id>/``,
  build the full 10 fps clip from them;
* otherwise flag ``history_unavailable`` and leave the gated tar fetch to the operator.

cam03 runs ~10 fps so frames map onto VISTA's grid directly; ``native_fps_estimate`` is
measured from the timestamp deltas when the raw frames are present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..config import Config
from .base import FrameRef, SourceSequence

__all__ = ["resolve", "fetch", "FetchReport", "sequence_dir"]


def sequence_dir(cfg: Config, sequence_id: str) -> str:
    """Where extracted cam03 frames for a sequence would live, if fetched."""
    rc = cfg.sources["once"].raw_cache
    return os.path.join(rc, sequence_id, "cam03") if rc else ""


def _coda_anchor_path(cfg: Config, scene) -> str:
    return os.path.join(cfg.coda_images_dir, scene.file_name)


def resolve(cfg: Config, scene) -> SourceSequence | None:
    seq_id = scene.source_ids["sequence_id"]
    anchor_ms = int(scene.source_ids["frame_id"])
    seq_dir = sequence_dir(cfg, seq_id)
    notes: list[str] = []

    have_history = bool(seq_dir) and os.path.isdir(seq_dir)
    if have_history:
        # cam03 frames are named <timestamp_ms>.jpg
        stamps = sorted(
            int(os.path.splitext(f)[0]) for f in os.listdir(seq_dir) if f.endswith(".jpg")
        )
        if anchor_ms not in stamps:
            notes.append("anchor_missing_from_raw_seq_falling_back_to_coda")
            have_history = False

    if not have_history:
        # Anchor-only: the CODA image is the anchor pixel; no usable history.
        notes.append("history_unavailable")
        anchor = FrameRef(key=str(anchor_ms), t=anchor_ms / 1000.0, path=_coda_anchor_path(cfg, scene))
        return SourceSequence(
            scene_id=scene.scene_id,
            frames=[anchor],
            anchor_idx=0,
            native_fps_estimate=float(cfg.target.fps),
            available=True,
            notes=notes,
        )

    frames: list[FrameRef] = []
    anchor_idx = -1
    for ms in stamps:
        if ms == anchor_ms:
            anchor_idx = len(frames)
        frames.append(FrameRef(key=str(ms), t=ms / 1000.0, path=os.path.join(seq_dir, f"{ms}.jpg")))

    deltas = sorted(stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)) if len(stamps) > 1 else []
    fps_est = float(cfg.target.fps)
    if deltas:
        med_ms = deltas[len(deltas) // 2]
        if med_ms > 0:
            fps_est = round(1000.0 / med_ms, 3)
            if abs(med_ms - 100.0) > 20.0:
                notes.append(f"native_fps={fps_est}_off_10fps_grid")

    return SourceSequence(
        scene_id=scene.scene_id,
        frames=frames,
        anchor_idx=anchor_idx,
        native_fps_estimate=fps_est,
        available=True,
        notes=notes,
    )


@dataclass
class FetchReport:
    sequences_needed: list[str] = field(default_factory=list)
    sequences_present: list[str] = field(default_factory=list)
    note: str = ""


def fetch(cfg: Config, scenes) -> FetchReport:
    """Report ONCE history fetch status; the tar pull itself is operator-gated.

    We can't surgically fetch a single ONCE sequence -- the archive is per-split tars.
    This enumerates the sequences needed and which are already extracted under
    ``once.raw_cache``, so an operator with ONCE access can drop the right ``*_cam03.tar``
    contents into ``raw_cache/<sequence_id>/cam03/`` and re-run the builder.
    """
    rep = FetchReport(
        note="ONCE history requires the gated per-split cam03 tars; place extracted frames "
        "at once.raw_cache/<sequence_id>/cam03/<timestamp_ms>.jpg. Anchors need no fetch.",
    )
    needed = {s.source_ids["sequence_id"] for s in scenes if s.source == "once"}
    for seq_id in sorted(needed):
        d = sequence_dir(cfg, seq_id)
        (rep.sequences_present if d and os.path.isdir(d) else rep.sequences_needed).append(seq_id)
    return rep
