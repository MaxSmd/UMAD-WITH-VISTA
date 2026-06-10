"""Stage D -- letterbox (pad, never crop) and the matching box transform.

VISTA wants 1024x576 (16:9). All three CODA sources are wider than 16:9, so the
resize-to-fit leaves top/bottom bars. The box transform must mirror the letterbox
*exactly* (``x*scale+pad_x``, ``y*scale+pad_y``) or the boxes and the predicted
frame end up in different coordinate frames and the detection score is meaningless
(PLAN.md gotcha #4). The reference implementation in PLAN.md section 9 is the spec
followed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["LetterboxTransform", "letterbox", "box_to_vista", "boxes_to_vista"]


@dataclass
class LetterboxTransform:
    scale: float
    pad_x: int
    pad_y: int
    out_w: int
    out_h: int

    def to_json(self, pad_color: tuple[int, int, int]) -> dict:
        return {
            "scale": round(self.scale, 6),
            "pad_x": self.pad_x,
            "pad_y": self.pad_y,
            "out": [self.out_w, self.out_h],
            "pad_color": list(pad_color),
        }


def letterbox(
    img: np.ndarray,
    target_w: int = 1024,
    target_h: int = 576,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, np.ndarray, LetterboxTransform]:
    """Resize ``img`` to fit inside ``target_w x target_h`` and pad to exact size.

    Returns ``(out_img, valid_mask, transform)`` where ``valid_mask`` is 1 inside the
    real content and 0 in the bars (uint8, single channel).
    """
    h, w = img.shape[:2]
    s = min(target_w / w, target_h / h)
    nw, nh = round(w * s), round(h * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    px, py = (target_w - nw) // 2, (target_h - nh) // 2

    out = np.full((target_h, target_w, img.shape[2] if img.ndim == 3 else 1), pad_color, np.uint8)
    out[py : py + nh, px : px + nw] = resized
    mask = np.zeros((target_h, target_w), np.uint8)
    mask[py : py + nh, px : px + nw] = 1
    return out, mask, LetterboxTransform(scale=s, pad_x=px, pad_y=py, out_w=target_w, out_h=target_h)


def box_to_vista(box: list[float], t: LetterboxTransform) -> list[float]:
    """Map a native COCO xywh box into letterboxed coordinates."""
    return [box[0] * t.scale + t.pad_x, box[1] * t.scale + t.pad_y, box[2] * t.scale, box[3] * t.scale]


def boxes_to_vista(
    boxes: list[list[float]], t: LetterboxTransform, warn: bool = True
) -> list[list[float]]:
    """Transform + clip every box to the frame; warn (don't crash) on edge-grazing.

    A transformed box must lie within [0, out_w] x [0, out_h]. Boxes are clipped to the
    frame; any clip beyond a 1px tolerance is surfaced because it usually signals a
    coordinate-frame bug rather than a genuine edge object (gotcha #4).
    """
    out: list[list[float]] = []
    for b in boxes:
        x, y, w, h = box_to_vista(b, t)
        x0, y0, x1, y1 = x, y, x + w, y + h
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(float(t.out_w), x1), min(float(t.out_h), y1)
        if warn and (x0 < -1 or y0 < -1 or x1 > t.out_w + 1 or y1 > t.out_h + 1):
            print(f"  [letterbox] box {b} -> {[x, y, w, h]} grazes frame edge; clipped")
        out.append([cx0, cy0, max(0.0, cx1 - cx0), max(0.0, cy1 - cy0)])
    return out
