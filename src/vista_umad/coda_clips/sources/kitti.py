"""KITTI source (PLAN.md B2) -- the most surgical: fetch only the exact raw drives.

CODA's ``kitti_indices`` are indices into the KITTI **object-detection training** set
(7481 images). The object devkit's ``mapping/train_rand.txt`` (1-based permutation into
``mapping/train_mapping.txt``) recovers the originating raw drive + frame:

    line = train_mapping[train_rand[odet_index] - 1]   # ('2011_09_26', '..._drive_0009_sync', 384)

Empty mapping lines mean the object image has no raw correspondence -> the scene has no
clip and is dropped (gotcha #2). Across all 309 CODA-KITTI scenes there are 12 unique
drives over 4 dates, so we download only those ``<drive>_sync.zip`` (+ per-date calib),
extract just ``image_02`` (the rest of each 1.5 GB zip is lidar we don't need), and drop
the zip. Raw frames are 10 fps and 0-based, so the clip grid is already aligned.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass, field

from ..config import Config
from .base import FrameRef, SourceSequence

__all__ = ["odet_to_raw", "resolve", "fetch", "FetchReport", "drives_for_scenes"]

_RAW_BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
_DEVKIT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip"


# --- object-detection index -> raw (date, drive_sync, frame) ------------------
_RAND: list[int] | None = None
_MAPPING: list[list[str]] | None = None


def _mapping_dir(cfg: Config) -> str:
    return os.path.join(cfg.sources["kitti"].raw_cache, "mapping")


def _ensure_mapping(cfg: Config) -> None:
    """Load (downloading the devkit if needed) train_rand / train_mapping."""
    global _RAND, _MAPPING
    if _RAND is not None and _MAPPING is not None:
        return
    mdir = _mapping_dir(cfg)
    rand_p = os.path.join(mdir, "train_rand.txt")
    map_p = os.path.join(mdir, "train_mapping.txt")
    if not (os.path.exists(rand_p) and os.path.exists(map_p)):
        os.makedirs(cfg.sources["kitti"].raw_cache, exist_ok=True)
        zip_p = os.path.join(cfg.sources["kitti"].raw_cache, "devkit_object.zip")
        if not os.path.exists(zip_p):
            print(f"  [kitti] downloading object devkit -> {zip_p}")
            urllib.request.urlretrieve(_DEVKIT_URL, zip_p)
        with zipfile.ZipFile(zip_p) as z:
            for n in ("mapping/train_rand.txt", "mapping/train_mapping.txt"):
                z.extract(n, cfg.sources["kitti"].raw_cache)
    _RAND = [int(t) for t in open(rand_p).read().strip().split(",") if t.strip()]
    _MAPPING = [ln.split() for ln in open(map_p).read().splitlines()]


def odet_to_raw(cfg: Config, odet_index: int) -> tuple[str, str, int] | None:
    """Map an object-detection index (0..7480) to ``(date, drive_sync, frame)``.

    Returns ``None`` for empty mapping lines (no raw correspondence). The ``- 1`` is the
    load-bearing 1-based offset into ``train_mapping`` (gotcha #1).
    """
    _ensure_mapping(cfg)
    assert _RAND is not None and _MAPPING is not None
    if odet_index < 0 or odet_index >= len(_RAND):
        return None
    line = _MAPPING[_RAND[odet_index] - 1]
    if not line:
        return None
    return line[0], line[1], int(line[2])


def _drive_dir(cfg: Config, date: str, drive_sync: str) -> str:
    return os.path.join(cfg.sources["kitti"].raw_cache, date, drive_sync, "image_02")


def _frame_path(cfg: Config, date: str, drive_sync: str, frame: int) -> str:
    return os.path.join(_drive_dir(cfg, date, drive_sync), "data", f"{frame:010d}.png")


def _read_timestamps(image02_dir: str) -> list[float] | None:
    """Parse image_02/timestamps.txt into monotonic seconds, or None if absent."""
    p = os.path.join(image02_dir, "timestamps.txt")
    if not os.path.exists(p):
        return None
    import datetime

    out: list[float] = []
    for ln in open(p).read().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # '2011_09_26 13:02:25.013652336' -- nanosecond precision, trim to micros for fromisoformat
        date_part, time_part = ln.split(" ")
        sec, _, frac = time_part.partition(".")
        micros = (frac + "000000")[:6]
        dt = datetime.datetime.fromisoformat(f"2000-01-01 {sec}.{micros}")
        out.append(dt.timestamp())
    if out:
        t0 = out[0]
        out = [t - t0 for t in out]
    return out


def resolve(cfg: Config, scene) -> SourceSequence | None:
    """Resolve a CODA-KITTI scene to its raw drive clip (Stage B/C input)."""
    raw = odet_to_raw(cfg, int(scene.source_ids["odet_index"]))
    if raw is None:
        return None
    date, drive_sync, frame = raw
    image02 = _drive_dir(cfg, date, drive_sync)
    data_dir = os.path.join(image02, "data")
    notes: list[str] = []

    available = os.path.isdir(data_dir)
    if available:
        nums = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(data_dir) if f.endswith(".png"))
    else:
        # Not fetched yet: synthesize the frame list around the anchor at nominal 10 fps.
        nums = list(range(max(0, frame - cfg.history_pool_frames - 4), frame + 1))
        notes.append("pixels_not_fetched")

    ts = _read_timestamps(image02) if available else None
    frames: list[FrameRef] = []
    anchor_idx = -1
    for n in nums:
        if n == frame:
            anchor_idx = len(frames)
        t = ts[n] if (ts is not None and n < len(ts)) else n / cfg.target.fps
        frames.append(FrameRef(key=str(n), t=float(t), path=_frame_path(cfg, date, drive_sync, n) if available else None))
    if anchor_idx < 0:
        notes.append("anchor_frame_missing_from_drive")
        return None

    # native fps from median delta if we have real timestamps
    fps_est = float(cfg.target.fps)
    if ts is not None and len(ts) > 1:
        deltas = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
        med = deltas[len(deltas) // 2]
        if med > 0:
            fps_est = round(1.0 / med, 3)
            if abs(med - 1.0 / cfg.target.fps) > 0.02:
                notes.append(f"native_fps={fps_est}_off_10fps_grid")

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
    drives: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def drives_for_scenes(cfg: Config, scenes) -> dict[str, set[int]]:
    """``drive_sync -> {frame numbers needed}`` for the given KITTI scenes."""
    need: dict[str, set[int]] = {}
    for s in scenes:
        if s.source != "kitti":
            continue
        raw = odet_to_raw(cfg, int(s.source_ids["odet_index"]))
        if raw is None:
            continue
        _, drive_sync, frame = raw
        need.setdefault(drive_sync, set()).add(frame)
    return need


def fetch(cfg: Config, scenes, keep_zip: bool = False) -> FetchReport:
    """Download + extract image_02 for every raw drive these scenes need.

    Only ``image_02`` (data + timestamps) and the per-date calib are kept; the ~1.5 GB
    zip is deleted afterwards unless ``keep_zip``. Idempotent: drives already extracted
    are skipped.
    """
    rep = FetchReport()
    raw_cache = cfg.sources["kitti"].raw_cache
    os.makedirs(raw_cache, exist_ok=True)
    need = drives_for_scenes(cfg, scenes)
    dates = {ds.split("_drive")[0] for ds in need}

    for date in sorted(dates):
        calib_zip = os.path.join(raw_cache, f"{date}_calib.zip")
        if not os.path.exists(os.path.join(raw_cache, date, "calib_cam_to_cam.txt")):
            try:
                urllib.request.urlretrieve(f"{_RAW_BASE}/{date}_calib.zip", calib_zip)
                with zipfile.ZipFile(calib_zip) as z:
                    z.extractall(raw_cache)
                os.remove(calib_zip)
            except Exception as e:  # noqa: BLE001 -- calib is non-fatal, log and continue
                print(f"  [kitti] calib {date} failed: {e}")

    for drive_sync in sorted(need):
        date = drive_sync.split("_drive")[0]
        out_image02 = _drive_dir(cfg, date, drive_sync)
        if os.path.isdir(os.path.join(out_image02, "data")):
            rep.skipped_existing.append(drive_sync)
            continue
        drive_base = drive_sync[:-5]  # strip trailing '_sync'
        url = f"{_RAW_BASE}/{drive_base}/{drive_sync}.zip"
        zip_p = os.path.join(raw_cache, f"{drive_sync}.zip")
        try:
            print(f"  [kitti] downloading {drive_sync} ...")
            urllib.request.urlretrieve(url, zip_p)
            with zipfile.ZipFile(zip_p) as z:
                members = [m for m in z.namelist() if f"/{drive_sync}/image_02/" in m]
                z.extractall(raw_cache, members=members)
            if not keep_zip:
                os.remove(zip_p)
            rep.drives.append(drive_sync)
            print(f"  [kitti] extracted image_02 for {drive_sync}")
        except Exception as e:  # noqa: BLE001
            rep.failed[drive_sync] = str(e)
            print(f"  [kitti] FAILED {drive_sync}: {e}")
            if os.path.exists(zip_p) and not keep_zip:
                os.remove(zip_p)
    return rep
