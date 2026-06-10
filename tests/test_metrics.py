"""Unit tests for the Phase 3 anomaly-map pipeline.

Runnable two ways:
    python tests/test_metrics.py          # built-in runner, no pytest needed
    pytest tests/test_metrics.py          # if pytest is installed

The VGG-perceptual tests download ImageNet weights on first run; if no network is
available they are skipped (reported, not failed).
"""

from __future__ import annotations

import os
import sys
import traceback

# Shared-server etiquette: cap threads before torch imports (see vista_umad.runtime).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vista_umad import fusion, metrics, runtime  # noqa: E402
from vista_umad.pipeline import AnomalyMapPipeline  # noqa: E402

runtime.cap_cpu_threads(4)
torch.manual_seed(0)


class SkipTest(Exception):
    """Raised to skip a test (e.g. when offline)."""


def _rand_frame(batch: int | None = None, h: int = 24, w: int = 32) -> torch.Tensor:
    shape = (3, h, w) if batch is None else (batch, 3, h, w)
    return torch.rand(shape)


# --------------------------------------------------------------------------- #
# abs / mse
# --------------------------------------------------------------------------- #
def test_abs_error_identity_is_zero():
    frame = _rand_frame()
    out = metrics.abs_error(frame, frame)
    assert out.shape == (24, 32), out.shape
    assert torch.allclose(out, torch.zeros_like(out)), out.abs().max()


def test_abs_error_known_value():
    real = torch.full((3, 4, 5), 0.8)
    pred = torch.full((3, 4, 5), 0.3)
    out = metrics.abs_error(real, pred)
    assert torch.allclose(out, torch.full((4, 5), 0.5), atol=1e-6), out.unique()


def test_mse_error_known_value():
    real = torch.full((3, 4, 5), 0.8)
    pred = torch.full((3, 4, 5), 0.3)
    out = metrics.mse_error(real, pred)
    assert torch.allclose(out, torch.full((4, 5), 0.25), atol=1e-6), out.unique()


def test_abs_mse_batched_shapes_and_consistency():
    real = _rand_frame(batch=4)
    pred = _rand_frame(batch=4)
    abs_b = metrics.abs_error(real, pred)
    mse_b = metrics.mse_error(real, pred)
    assert abs_b.shape == (4, 24, 32), abs_b.shape
    assert mse_b.shape == (4, 24, 32), mse_b.shape
    # Per-sample batched result must match the unbatched call.
    single = metrics.abs_error(real[1], pred[1])
    assert torch.allclose(abs_b[1], single, atol=1e-6)
    # For values in [0, 1], squared error <= absolute error elementwise.
    assert (mse_b <= abs_b + 1e-6).all()


def test_abs_error_shape_mismatch_raises():
    try:
        metrics.abs_error(_rand_frame(h=8), _rand_frame(h=9))
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_abs_error_non_rgb_raises():
    try:
        metrics.abs_error(torch.rand(4, 8, 8), torch.rand(4, 8, 8))
    except ValueError:
        return
    raise AssertionError("expected ValueError on non-RGB input")


# --------------------------------------------------------------------------- #
# ssim
# --------------------------------------------------------------------------- #
def test_ssim_difference_identity_is_near_zero():
    frame = _rand_frame()
    out = metrics.ssim_difference(frame, frame)
    assert out.shape == (24, 32), out.shape
    assert out.max() < 1e-3, out.max()


def test_ssim_difference_range_and_signal():
    real = _rand_frame()
    pred = _rand_frame()  # independent noise -> structurally dissimilar
    out = metrics.ssim_difference(real, pred)
    assert out.min() >= 0.0 and out.max() <= 1.0, (out.min(), out.max())
    assert out.mean() > metrics.ssim_difference(real, real).mean()


def test_ssim_difference_batched():
    real = _rand_frame(batch=3)
    pred = _rand_frame(batch=3)
    out = metrics.ssim_difference(real, pred)
    assert out.shape == (3, 24, 32), out.shape
    single = metrics.ssim_difference(real[2], pred[2])
    assert torch.allclose(out[2], single, atol=1e-5)


# --------------------------------------------------------------------------- #
# temporal difference
# --------------------------------------------------------------------------- #
def test_temporal_difference_no_priors_is_zero():
    current = _rand_frame()
    out = metrics.temporal_difference(current, [])
    assert out.shape == (24, 32), out.shape
    assert torch.count_nonzero(out) == 0


def test_temporal_difference_equal_priors_is_zero():
    current = _rand_frame()
    out = metrics.temporal_difference(current, [current.clone(), current.clone()])
    assert torch.allclose(out, torch.zeros_like(out))


def test_temporal_difference_averages_over_priors():
    current = _rand_frame()
    differing = _rand_frame()
    # One differing prior + one identical prior -> mean halves the differing term.
    out = metrics.temporal_difference(current, [differing, current.clone()])
    expected = metrics.abs_error(differing, current) / 2.0
    assert torch.allclose(out, expected, atol=1e-6), (out - expected).abs().max()


# --------------------------------------------------------------------------- #
# prediction variance
# --------------------------------------------------------------------------- #
def test_prediction_variance_identical_is_zero():
    frame = _rand_frame()
    out = metrics.prediction_variance([frame, frame.clone(), frame.clone()])
    assert out.shape == (24, 32), out.shape
    assert out.max() < 1e-8, out.max()


def test_prediction_variance_needs_two():
    try:
        metrics.prediction_variance([_rand_frame()])
    except ValueError:
        return
    raise AssertionError("expected ValueError with a single prediction")


# --------------------------------------------------------------------------- #
# perceptual (VGG) -- skipped offline
# --------------------------------------------------------------------------- #
def _vgg_extractor():
    try:
        return metrics.VGGPerceptualExtractor(device="cpu")
    except Exception as exc:  # network / download failure
        raise SkipTest(f"VGG weights unavailable: {exc}")


def test_perceptual_identity_is_zero():
    vgg = _vgg_extractor()
    frame = _rand_frame()
    out = vgg(frame, frame)
    assert out.shape == (24, 32), out.shape
    assert out.abs().max() < 1e-4, out.abs().max()


def test_perceptual_signal_and_batched():
    vgg = _vgg_extractor()
    real = _rand_frame(batch=2)
    pred = _rand_frame(batch=2)
    out = vgg(real, pred)
    assert out.shape == (2, 24, 32), out.shape
    assert (out >= 0).all()
    assert out.mean() > vgg(real, real).mean()
    single = vgg(real[0], pred[0])
    assert torch.allclose(out[0], single, atol=1e-4)


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def test_normalize_minmax_range():
    raw = torch.tensor([[1.0, 3.0], [5.0, 9.0]])
    out = fusion.normalize_map(raw, mode="minmax")
    assert abs(out.min().item()) < 1e-6 and abs(out.max().item() - 1.0) < 1e-6
    assert out.shape == raw.shape


def test_normalize_constant_map_is_finite():
    out = fusion.normalize_map(torch.full((6, 6), 0.7), mode="minmax")
    assert torch.isfinite(out).all()
    assert out.max() < 1e-3  # constant -> all ~0, no NaN/inf


def test_normalize_batched_per_sample():
    batched = torch.stack([torch.tensor([[0.0, 2.0]]), torch.tensor([[10.0, 20.0]])])
    out = fusion.normalize_map(batched, mode="minmax")
    assert out.shape == batched.shape
    for sample in out:
        assert abs(sample.min().item()) < 1e-6 and abs(sample.max().item() - 1.0) < 1e-6


def test_normalize_clip_quantiles():
    raw = torch.cat([torch.zeros(99), torch.tensor([1000.0])]).reshape(1, 100)
    clipped = fusion.normalize_map(raw, mode="minmax", clip_quantiles=(0.0, 0.95))
    # The lone outlier is clipped away, so it saturates rather than dominating.
    assert clipped.max().item() <= 1.0 + 1e-6
    assert (clipped == clipped.max()).sum() > 1


def test_normalize_none_passthrough():
    raw = torch.tensor([[2.0, 4.0]])
    assert torch.equal(fusion.normalize_map(raw, mode="none"), raw)


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #
def test_fuse_single_map_identity():
    amap = torch.rand(10, 12)
    fused = fusion.fuse({"abs": amap}, {"abs": 1.0})
    assert torch.allclose(fused, fusion.normalize_map(amap, mode="minmax"))


def test_fuse_weight_renormalization():
    amap = torch.rand(8, 8)
    # Unequal raw weights that do not sum to 1 are renormalized internally.
    a = fusion.fuse({"x": amap, "y": amap}, {"x": 2.0, "y": 2.0})
    b = fusion.fuse({"x": amap, "y": amap}, {"x": 0.5, "y": 0.5})
    assert torch.allclose(a, b, atol=1e-6)


def test_fuse_output_range():
    maps = {"a": torch.rand(16, 16), "b": torch.rand(16, 16)}
    fused = fusion.fuse(maps, {"a": 0.5, "b": 0.5})
    assert fused.min() >= -1e-6 and fused.max() <= 1.0 + 1e-6


def test_fuse_missing_weighted_map_raises():
    try:
        fusion.fuse({"a": torch.rand(4, 4)}, {"b": 1.0})
    except KeyError:
        return
    raise AssertionError("expected KeyError for missing weighted map")


def test_fuse_negative_weight_raises():
    try:
        fusion.fuse({"a": torch.rand(4, 4)}, {"a": -1.0})
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative weight")


def test_fuse_all_zero_weights_raise():
    try:
        fusion.fuse({"a": torch.rand(4, 4)}, {"a": 0.0})
    except ValueError:
        return
    raise AssertionError("expected ValueError for all-zero weights")


def test_fuse_shape_mismatch_raises():
    try:
        fusion.fuse({"a": torch.rand(4, 4), "b": torch.rand(5, 5)}, {"a": 0.5, "b": 0.5})
    except ValueError:
        return
    raise AssertionError("expected ValueError for mismatched map shapes")


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_preset_resolution():
    pipe = AnomalyMapPipeline(weights="ssim_pd", device="cpu")
    assert pipe.weights == {"ssim": 0.5, "pd": 0.5}


def test_pipeline_unknown_preset_raises():
    try:
        AnomalyMapPipeline(weights="does_not_exist", device="cpu")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown preset")


def test_pipeline_unknown_metric_name_raises():
    try:
        AnomalyMapPipeline(weights={"bogus": 1.0}, device="cpu")
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown metric name")


def test_pipeline_compute_maps_without_pd():
    # compute_all=False + no pd weight -> VGG never loaded, pd map absent.
    pipe = AnomalyMapPipeline(weights={"ssim": 1.0}, device="cpu", compute_all=False)
    maps = pipe.compute_maps(_rand_frame(), _rand_frame())
    assert set(maps) == {"ssim"}, set(maps)
    assert pipe._vgg is None


def test_pipeline_td_only_with_priors():
    pipe = AnomalyMapPipeline(weights={"abs": 1.0}, device="cpu", compute_all=True)
    real, pred = _rand_frame(), _rand_frame()
    assert "td" not in pipe.compute_maps(real, pred)
    with_priors = pipe.compute_maps(real, pred, prior_preds=[_rand_frame()])
    assert "td" in with_priors
    assert with_priors["td"].shape == (24, 32)


def test_pipeline_end_to_end_shapes():
    try:
        pipe = AnomalyMapPipeline(weights="ssim_pd", device="cpu")
        pipe.vgg  # force VGG load; may need network
    except Exception as exc:
        raise SkipTest(f"VGG weights unavailable: {exc}")
    real, pred = _rand_frame(), _rand_frame()
    fused, maps = pipe(real, pred)
    assert fused.shape == (24, 32), fused.shape
    assert {"abs", "mse", "ssim", "pd"} <= set(maps)
    assert torch.isfinite(fused).all()


def test_pipeline_fuse_drops_missing_td():
    # mse_ssim_pd_td weights td, but with no priors td is absent: fusion must still work.
    try:
        pipe = AnomalyMapPipeline(weights="mse_ssim_pd_td", device="cpu")
        pipe.vgg
    except Exception as exc:
        raise SkipTest(f"VGG weights unavailable: {exc}")
    fused, maps = pipe(_rand_frame(), _rand_frame())
    assert "td" not in maps
    assert fused.shape == (24, 32)


# --------------------------------------------------------------------------- #
# runtime / resource confinement
# --------------------------------------------------------------------------- #
def test_runtime_resolve_cpu():
    dev = runtime.resolve_device("cpu")
    assert isinstance(dev, torch.device) and dev.type == "cpu"


def test_runtime_resolve_auto():
    dev = runtime.resolve_device("auto")
    assert dev.type in ("cpu", "cuda")
    if dev.type == "cuda":
        assert dev.index == 0  # after CUDA_VISIBLE_DEVICES masking, always cuda:0


def test_runtime_resolve_out_of_range_raises():
    # cuda:99 is never valid: either CUDA is unavailable or the index is out of range.
    try:
        runtime.resolve_device("cuda:99")
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for cuda:99")


def test_runtime_cap_cpu_threads():
    previous = torch.get_num_threads()
    try:
        assert runtime.cap_cpu_threads(3) == 3
        assert torch.get_num_threads() == 3
        # 0 leaves the current setting untouched.
        assert runtime.cap_cpu_threads(0) == 3
    finally:
        torch.set_num_threads(previous)


def test_runtime_multi_gpu_warning_type():
    warning = runtime.multi_gpu_warning()
    assert warning is None or isinstance(warning, str)


def test_runtime_describe_is_informative():
    text = runtime.describe_runtime(runtime.resolve_device("cpu"))
    assert "device in use" in text and "CPU intra-op threads" in text


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _run() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = skipped = failed = 0
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP  {name}: {exc}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"ok    {name}")
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed "
          f"(of {len(tests)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
