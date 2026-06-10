"""CODA-clips benchmark builder (see PLAN.md).

Turns CODA's single-frame corner-case benchmark into a **clip** benchmark for the
VISTA driving world model: per scene, reconstruct the short clip leading up to the
corner-case (anchor) frame, letterbox to VISTA's 1024x576 / 10 fps geometry without
cropping, transform the boxes, and emit a license-clean manifest + media store. The
downstream task is corner-case detection via VISTA prediction error.

Stages (each runnable on a handful of scenes before scaling):

* A -- :mod:`.index`      parse CODA -> normalized scene index (no downloads).
* B -- :mod:`.sources`    resolve + fetch source clips (once / kitti / nuscenes).
* C -- :mod:`.clips`      build clips on the 10 fps grid (context + history pool).
* D -- :mod:`.letterbox`  pad-not-crop to 1024x576 + matching box transform.
* E -- :mod:`.manifest`   write media, manifest.jsonl, datasheet, reconstruct.py.
* F -- :mod:`.eval_harness` prediction-error corner-case detection (wraps VISTA).
"""

from . import config, index, letterbox

__all__ = ["config", "index", "letterbox"]
