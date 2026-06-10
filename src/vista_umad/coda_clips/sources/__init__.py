"""Stage B -- per-source clip resolution and fetch.

Each source module exposes a common pair of entry points so the clip builder and
manifest writer never special-case a source beyond what is genuinely different:

* ``resolve(cfg, scene) -> SourceSequence | None`` -- map a :class:`~vista_umad.coda_clips.index.Scene`
  onto the ordered, timestamp-sorted source frame sequence that contains the anchor,
  marking the anchor's position and whether the pixels are present on disk. Returns
  ``None`` when the scene has no recoverable clip (e.g. an empty KITTI mapping line).
* ``fetch(cfg, scenes) -> FetchReport`` -- download only the source media needed for
  ``scenes`` (per-drive for KITTI, per-split tar for ONCE, gated blobs for nuScenes).

The shared data model lives in :mod:`.base`.
"""

from . import base, kitti, nuscenes, once

__all__ = ["base", "once", "kitti", "nuscenes"]
