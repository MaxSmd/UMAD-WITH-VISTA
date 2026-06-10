"""nuScenes source (PLAN.md B3) -- the expensive one, gated by default.

Metadata resolution is cheap and always implemented: ``v1.0-trainval_meta.tgz`` (no
pixels) gives ``sample.json`` (token -> record) and ``sample_data.json`` (the prev/next
linked list over CAM_FRONT *sweeps* at ~12 fps). From a CODA ``sample_token`` we find the
keyframe CAM_FRONT sample_data and walk ``prev`` to recover the ordered clip filenames +
microsecond timestamps; :func:`~vista_umad.coda_clips.sources.base.build_clip` then
resamples 12 fps -> the 10 fps grid by nearest tick.

The **pixels** only exist in the ~400 GB monolithic trainval blob tarballs (the ~60 GB
keyframe blobs give 2 Hz, below VISTA's 10 fps assumption). So ``available`` is true only
when the blob files are actually on disk, and ``nuscenes.enabled`` defaults to false --
ONCE/KITTI never block on this. Two caveats worth flagging downstream: nuScenes is
1600x900 (exactly 16:9 -> no letterbox bars), and it is *in VISTA's training set*
(``in_vista_train_domain``), so its detection numbers are optimistic and must not be
pooled into a single headline (gotcha #8).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..config import Config
from .base import FrameRef, SourceSequence

__all__ = ["resolve", "fetch", "FetchReport", "load_meta"]

_META: dict | None = None


def _meta_dir(cfg: Config) -> str:
    dr = cfg.sources["nuscenes"].dataroot or ""
    # trainval meta extracts to <dataroot>/v1.0-trainval/*.json
    return os.path.join(dr, "v1.0-trainval")


def load_meta(cfg: Config) -> dict:
    """Lazily load sample.json + sample_data.json into lookup tables."""
    global _META
    if _META is not None:
        return _META
    mdir = _meta_dir(cfg)
    sample_p = os.path.join(mdir, "sample.json")
    sd_p = os.path.join(mdir, "sample_data.json")
    if not (os.path.exists(sample_p) and os.path.exists(sd_p)):
        raise FileNotFoundError(
            f"nuScenes metadata not found under {mdir}. Download v1.0-trainval_meta.tgz "
            "and extract it to nuscenes.dataroot."
        )
    samples = {r["token"]: r for r in json.load(open(sample_p))}
    sds = {r["token"]: r for r in json.load(open(sd_p))}
    # Raw sample.json has no devkit-style ``data`` dict; build sample_token -> the
    # CAM_FRONT *keyframe* sample_data token ourselves (filename under samples/CAM_FRONT/,
    # is_key_frame). From there ``prev`` walks back over the 12 fps CAM_FRONT sweeps.
    cam_front: dict[str, str] = {}
    for r in sds.values():
        if r.get("is_key_frame") and r["filename"].startswith("samples/CAM_FRONT/"):
            cam_front[r["sample_token"]] = r["token"]
    _META = {"samples": samples, "sample_data": sds, "cam_front_keyframe": cam_front}
    return _META


def resolve(cfg: Config, scene) -> SourceSequence | None:
    meta = load_meta(cfg)
    token = scene.source_ids["sample_token"]
    if token not in meta["samples"]:
        return None
    sd_token = meta["cam_front_keyframe"].get(token)
    if sd_token is None:
        return None
    sds = meta["sample_data"]
    dataroot = cfg.sources["nuscenes"].dataroot or ""
    notes: list[str] = ["in_vista_train_domain"]

    # Walk prev to collect the anchor + history sweeps (newest first), then reverse.
    chain: list[dict] = []
    cur = sds.get(sd_token)
    steps = cfg.history_pool_frames + 8  # a little slack for the nearest-tick resampler
    while cur is not None and len(chain) <= steps:
        chain.append(cur)
        cur = sds.get(cur["prev"]) if cur["prev"] else None
    chain.reverse()  # oldest -> anchor

    frames: list[FrameRef] = []
    anchor_idx = -1
    for sd in chain:
        path = os.path.join(dataroot, sd["filename"])
        if sd["token"] == sd_token:
            anchor_idx = len(frames)
        frames.append(FrameRef(key=sd["token"], t=sd["timestamp"] / 1e6, path=path if os.path.exists(path) else None))
    if anchor_idx < 0:
        return None

    available = all(f.path is not None for f in frames)
    if not available:
        notes.append("pixels_not_fetched")

    fps_est = float(cfg.target.fps)
    if len(frames) > 1:
        deltas = sorted(frames[i + 1].t - frames[i].t for i in range(len(frames) - 1))
        med = deltas[len(deltas) // 2]
        if med > 0:
            fps_est = round(1.0 / med, 3)
            notes.append(f"native_fps={fps_est}_resampled_to_{cfg.target.fps}")

    return SourceSequence(
        scene_id=scene.scene_id,
        frames=frames,
        anchor_idx=anchor_idx,
        native_fps_estimate=fps_est,
        available=available,
        notes=notes,
    )


@dataclass
class FetchReport:
    note: str = ""
    scenes_resolvable: int = 0
    scenes_with_pixels: int = 0


def fetch(cfg: Config, scenes) -> FetchReport:
    """Report nuScenes resolvability; blob download is intentionally not automated.

    The 12 fps clips require the full trainval file blobs (~400 GB), for which there is
    no per-scene download. We resolve metadata (if present) and report how many scenes
    could be built, but never pull the blobs -- that is an explicit operator decision
    documented in the datasheet.
    """
    rep = FetchReport(
        note="nuScenes pixels live only in the ~400 GB trainval file blobs (no per-scene "
        "fetch). Extract v1.0-trainval_meta.tgz to dataroot for metadata, and the file "
        "blobs to dataroot/sweeps to enable 12->10 fps clip building.",
    )
    try:
        load_meta(cfg)
    except FileNotFoundError:
        return rep
    for s in scenes:
        if s.source != "nuscenes":
            continue
        seq = resolve(cfg, s)
        if seq is None:
            continue
        rep.scenes_resolvable += 1
        if seq.available:
            rep.scenes_with_pixels += 1
    return rep
