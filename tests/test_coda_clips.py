"""Unit tests for the CODA-clips benchmark builder.

Runnable two ways:
    python tests/test_coda_clips.py       # built-in runner, no pytest needed
    pytest tests/test_coda_clips.py

Covers the pure, error-prone bits that PLAN.md flags: the letterbox box transform,
the 10 fps clip grid + context-frame selection, the KITTI 1-based mapping, and the
Stage F masked per-box scoring (bars excluded). No GPU or network required.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vista_umad.coda_clips import eval_harness as ev  # noqa: E402
from vista_umad.coda_clips import letterbox as lb  # noqa: E402
from vista_umad.coda_clips.sources.base import FrameRef, SourceSequence, build_clip, context_indices  # noqa: E402


def test_letterbox_and_box_transform():
    img = np.zeros((1020, 1920, 3), np.uint8)
    out, mask, t = lb.letterbox(img, 1024, 576, (0, 0, 0))
    assert out.shape == (576, 1024, 3)
    # 1920x1020 -> scale 0.53333, no x-pad, symmetric y-pad of (576-544)//2 = 16
    assert abs(t.scale - 1024 / 1920) < 1e-6 and t.pad_x == 0 and t.pad_y == 16
    assert mask[t.pad_y + 5, 10] == 1 and mask[2, 10] == 0  # content vs bar
    # box transform mirrors the letterbox exactly
    b = lb.box_to_vista([100, 200, 50, 60], t)
    assert abs(b[0] - 100 * t.scale) < 1e-6
    assert abs(b[1] - (200 * t.scale + t.pad_y)) < 1e-6
    assert abs(b[2] - 50 * t.scale) < 1e-6


def test_box_clipped_to_frame():
    _, _, t = lb.letterbox(np.zeros((375, 1242, 3), np.uint8), 1024, 576)
    # a box at the far edge should be clipped, not crash, and stay within frame
    out = lb.boxes_to_vista([[1240, 370, 10, 10]], t, warn=False)[0]
    x, y, w, h = out
    assert x + w <= t.out_w + 1 and y + h <= t.out_h + 1


def test_clip_grid_and_context_indices():
    # synth 10 fps sequence of 40 frames; anchor at index 35
    frames = [FrameRef(key=str(i), t=i * 0.1, path=None) for i in range(40)]
    seq = SourceSequence("s", frames, anchor_idx=35, native_fps_estimate=10.0, available=False)
    clip = build_clip(seq, fps=10, history_pool_frames=30)
    assert clip.frames[0].grid_index == 0 and abs(clip.frames[0].rel_t) < 1e-9
    assert not clip.truncated and len(clip.frames) == 31  # anchor + 30
    # context for 1.0 s horizon = grid indices 12,11,10 -> rel_t -1.2,-1.1,-1.0
    ci = context_indices(1.0, 10)
    assert ci == [12, 11, 10]
    rel = {g.grid_index: round(g.rel_t, 1) for g in clip.frames}
    assert rel[12] == -1.2 and rel[11] == -1.1 and rel[10] == -1.0


def test_clip_truncates_at_sequence_start():
    frames = [FrameRef(key=str(i), t=i * 0.1, path=None) for i in range(8)]
    seq = SourceSequence("s", frames, anchor_idx=7, native_fps_estimate=10.0, available=False)
    clip = build_clip(seq, fps=10, history_pool_frames=30)
    assert clip.truncated and len(clip.frames) == 8  # only 7 past frames available


def test_anchor_frame_index():
    assert ev.anchor_frame_index(1.0, 10, 3) == 12
    assert ev.anchor_frame_index(2.0, 10, 3) == 22


def test_score_boxes_masks_bars_and_separates():
    # 576x1024 error map: high error inside a box, low elsewhere, bars on top/bottom
    H, W = 576, 1024
    e = torch.full((H, W), 0.1)
    e[200:260, 500:560] = 0.9            # the corner-case region
    emap = e
    valid = np.ones((H, W), np.uint8)
    valid[:133] = 0                       # letterbox bars
    valid[443:] = 0
    # put garbage in the bars to prove they are excluded
    emap = emap.clone()
    emap[:133] = 5.0
    bs = ev.score_boxes(emap, valid, [[500, 200, 60, 60]], "s", "kitti", False)
    assert bs.box_scores[0] > bs.background          # box hotter than background
    assert bs.background < 0.2                        # bars excluded -> background stays low
    # masked pixel labels only cover valid pixels
    assert bs.pixel_scores.size == int(valid.sum())
    assert bs.pixel_labels.sum() == 60 * 60


def test_evaluate_no_pooling_of_nuscenes():
    rng = np.random.default_rng(0)
    scores = []
    for src, sep in (("kitti", True), ("nuscenes", True)):
        for i in range(15):
            box = [float(rng.normal(1.0 if sep else 0.0, 0.1))]
            bg = float(rng.normal(0.0, 0.1))
            scores.append(ev.BoxScore(f"{src}_{i}", src, src == "nuscenes", box, bg,
                                      np.array([]), np.array([])))
    out = ev.evaluate(scores)
    assert "kitti" in out["per_source"] and "nuscenes" in out["per_source"]
    assert out["per_source"]["nuscenes"]["in_vista_train_domain"] is True
    # macro is computed over non-train-domain sources only
    assert "macro_non_train_region_auroc" in out


# --------------------------------------------------------------------- runner
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
