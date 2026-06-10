"""UMAD pixel-space difference metrics, ported onto Vista predict-and-compare output.

Each visual metric is a standalone callable ``f(real, pred) -> anomaly_map`` where
``real`` (the ground-truth frame ``t``) and ``pred`` (Vista's predicted frame ``t-hat``)
are RGB tensors in ``[0, 1]`` of shape ``[3, H, W]`` or ``[B, 3, H, W]``. The returned
anomaly map is a non-negative tensor of shape ``[H, W]`` (or ``[B, H, W]``) where a
higher value means "more anomalous".

Weights are deliberately NOT baked in here -- the fusion module
(:mod:`vista_umad.fusion`) is responsible for normalization and weighted combination,
so the same metric functions can be composed into arbitrary configurations.

Equation numbers refer to:
    Bogdoll et al., "UMAD: Unsupervised Mask-Level Anomaly Detection for Autonomous
    Driving", BMVC 2024 (arXiv:2406.06370).
"""

from __future__ import annotations

from typing import Sequence

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

__all__ = [
    "abs_error",
    "mse_error",
    "ssim_difference",
    "temporal_difference",
    "prediction_variance",
    "VGGPerceptualExtractor",
]


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def _check_pair(real: torch.Tensor, pred: torch.Tensor) -> None:
    """Validate that two frame tensors are a comparable RGB pair."""
    if real.shape != pred.shape:
        raise ValueError(f"real/pred shape mismatch: {tuple(real.shape)} vs {tuple(pred.shape)}")
    if real.dim() not in (3, 4):
        raise ValueError(f"expected [3,H,W] or [B,3,H,W], got {tuple(real.shape)}")
    if real.shape[-3] != 3:
        raise ValueError(f"expected 3 RGB channels in dim -3, got {tuple(real.shape)}")


def _as_batched(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return ``(x_4d, was_batched)``; promotes a ``[C,H,W]`` tensor to ``[1,C,H,W]``."""
    if x.dim() == 3:
        return x.unsqueeze(0), False
    return x, True


# --------------------------------------------------------------------------- #
# Eq. 1 -- absolute error
# --------------------------------------------------------------------------- #
def abs_error(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """UMAD Eq. 1 -- per-pixel absolute error, averaged over the RGB channels.

    ``ABS = (|r~ - r^| + |g~ - g^| + |b~ - b^|) / 3``
    """
    _check_pair(real, pred)
    return (real - pred).abs().mean(dim=-3)


# --------------------------------------------------------------------------- #
# Eq. 2 -- squared error
# --------------------------------------------------------------------------- #
def mse_error(real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """UMAD Eq. 2 -- per-pixel squared error, averaged over the RGB channels.

    ``MSE = ((r~ - r^)^2 + (g~ - g^)^2 + (b~ - b^)^2) / 3``
    """
    _check_pair(real, pred)
    return (real - pred).pow(2).mean(dim=-3)


# --------------------------------------------------------------------------- #
# Eq. 3 -- SSIM-based difference
# --------------------------------------------------------------------------- #
def ssim_difference(
    real: torch.Tensor,
    pred: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """UMAD Eq. 3 -- structural-dissimilarity map.

    UMAD's Eq. 3 writes out the SSIM index itself (1 = identical). Used as an anomaly
    *difference* map it is inverted so that higher = more dissimilar:

        ``D_SSIM = 1 - SSIM(real, pred)``  (clamped to ``[0, 1]``, averaged over RGB)

    SSIM is computed with a sliding Gaussian window via :func:`kornia.metrics.ssim`.
    """
    _check_pair(real, pred)
    rb, batched = _as_batched(real)
    pb, _ = _as_batched(pred)
    # kornia returns a per-pixel SSIM map of shape [B, C, H, W] in [-1, 1].
    ssim_map = kornia.metrics.ssim(rb, pb, window_size=window_size, max_val=1.0)
    diff = (1.0 - ssim_map).clamp(min=0.0, max=1.0).mean(dim=1)  # [B, H, W]
    return diff if batched else diff[0]


# --------------------------------------------------------------------------- #
# Eq. 5 -- temporal difference
# --------------------------------------------------------------------------- #
def temporal_difference(
    current_pred: torch.Tensor,
    prior_preds: Sequence[torch.Tensor],
) -> torch.Tensor:
    """UMAD Eq. 5 -- temporal-difference map.

    ``D_TD = (1/n) * sum_i  ABS( z^_{t-i} , z^_t )``

    where ``current_pred`` is the prediction of frame ``t`` made from its own rollout
    and ``prior_preds`` are predictions of the *same* frame ``t`` produced by rollouts
    that started at earlier context windows (``t-1``, ``t-2``, ...). These overlapping
    predictions arise naturally from the multi-frame rollout setup of Phase 2.4.

    With an empty ``prior_preds`` (a frame no earlier rollout reached) a zero map is
    returned, so the metric degrades gracefully at the start of a scenario.
    """
    if current_pred.dim() not in (3, 4) or current_pred.shape[-3] != 3:
        raise ValueError(f"expected [3,H,W] or [B,3,H,W], got {tuple(current_pred.shape)}")
    if len(prior_preds) == 0:
        return torch.zeros(
            current_pred.shape[:-3] + current_pred.shape[-2:],
            dtype=current_pred.dtype,
            device=current_pred.device,
        )
    maps = [abs_error(prior, current_pred) for prior in prior_preds]
    return torch.stack(maps, dim=0).mean(dim=0)


def prediction_variance(predictions: Sequence[torch.Tensor]) -> torch.Tensor:
    """Pixelwise variance across multiple predictions of the same frame.

    This is the multi-sample temporal-uncertainty variant sketched in the architecture
    overview: feed the same context to Vista with different noise seeds and measure
    where the predictions disagree. It is offered alongside :func:`temporal_difference`
    (UMAD's Eq. 5) as an alternative ``td`` signal; both return a ``[H, W]`` map.
    """
    if len(predictions) < 2:
        raise ValueError("prediction_variance needs at least 2 predictions")
    stack = torch.stack(list(predictions), dim=0)  # [N, ..., 3, H, W]
    return stack.var(dim=0, unbiased=False).mean(dim=-3)


# --------------------------------------------------------------------------- #
# Eq. 4 -- perceptual difference (VGG/ImageNet feature maps)
# --------------------------------------------------------------------------- #
class VGGPerceptualExtractor(nn.Module):
    """UMAD Eq. 4 -- perceptual-difference map from VGG-16 ImageNet features.

    ``D_PD = sum_i  (1 / M_i) * || F^i(x) - F^i(x^) ||_1``

    ``F^i`` is the ``i``-th captured VGG layer. For a per-pixel anomaly map the L1
    difference of each layer is averaged over channels (this provides the ``1/M_i``
    channel normalization), bilinearly upsampled to the input resolution, and the
    layers are summed.

    An instance is itself the standalone callable ``f(real, pred) -> [H, W]`` required
    by the fusion module; the VGG weights are loaded once and frozen.

    Default layers ``(3, 8, 15, 22)`` are the post-ReLU outputs ``relu1_2, relu2_2,
    relu3_3, relu4_3`` -- the standard perceptual-loss layer set.
    """

    LAYER_NAMES = {3: "relu1_2", 8: "relu2_2", 15: "relu3_3", 22: "relu4_3"}
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        layers: Sequence[int] = (3, 8, 15, 22),
        device: str | torch.device = "cpu",
        weights: VGG16_Weights | None = VGG16_Weights.IMAGENET1K_V1,
    ) -> None:
        super().__init__()
        if len(layers) == 0:
            raise ValueError("VGGPerceptualExtractor needs at least one layer")
        self.layers = tuple(sorted(int(layer) for layer in layers))

        features = vgg16(weights=weights).features
        # Split the feature stack into consecutive slices, one per captured layer.
        self.slices = nn.ModuleList()
        prev = 0
        for layer in self.layers:
            self.slices.append(nn.Sequential(*[features[i] for i in range(prev, layer + 1)]))
            prev = layer + 1

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

        self.register_buffer("_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self._mean.device

    def _features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run ImageNet-normalized ``x`` through VGG, returning the captured feature maps."""
        x = (x - self._mean) / self._std
        feats: list[torch.Tensor] = []
        for sl in self.slices:
            x = sl(x)
            feats.append(x)
        return feats

    @torch.no_grad()
    def forward(self, real: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Compute the perceptual-difference map for an RGB ``[0, 1]`` frame pair."""
        _check_pair(real, pred)
        rb, batched = _as_batched(real)
        pb, _ = _as_batched(pred)
        height, width = rb.shape[-2:]
        rb = rb.to(self.device)
        pb = pb.to(self.device)

        feats_real = self._features(rb)
        feats_pred = self._features(pb)

        total = torch.zeros(rb.shape[0], height, width, device=self.device)
        for fr, fp in zip(feats_real, feats_pred):
            # L1 difference, averaged over channels -> [B, 1, h_i, w_i].
            layer_diff = (fr - fp).abs().mean(dim=1, keepdim=True)
            layer_diff = F.interpolate(
                layer_diff, size=(height, width), mode="bilinear", align_corners=False
            )
            total = total + layer_diff[:, 0]
        return total if batched else total[0]
