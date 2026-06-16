"""LPIPS perceptual distance (VGG-16 backbone).

LPIPS (Zhang et al., 2018) measures perceptual similarity as a weighted L2 over
deep features. This implements the standard **VGG-16** variant: tap the five
``reluX_Y`` feature maps, unit-normalize them across channels, square their
difference, weight each stage with a learned 1x1 ``lin`` head, and average.

The architecture is exact and CPU/GPU-runnable, but perceptually meaningful
scores require the **pretrained weights**, which are large and not bundled. Load
them with :meth:`LPIPS.load_weights` after converting the public ``lpips`` /
torchvision VGG-16 weights to a flat ``.safetensors``/``.npz`` (see the
"Perceptual metrics" docs). Without weights the network still runs (and
``lpips(x, x) == 0`` always holds) but is not calibrated.

For a self-contained, weight-free perceptual metric use
:func:`~mlx3d.losses.ms_ssim`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["LPIPS"]

# VGG-16 blocks; tap the last ReLU of each block (relu1_2 ... relu5_3).
_VGG_BLOCKS = [
    (3, 64, 64),
    (64, 128, 128),
    (128, 256, 256, 256),
    (256, 512, 512, 512),
    (512, 512, 512, 512),
]
_TAP_CHANNELS = [64, 128, 256, 512, 512]
# LPIPS input scaling (applied to images mapped to [-1, 1]).
_SHIFT = [-0.030, -0.088, -0.188]
_SCALE = [0.458, 0.448, 0.450]


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (VGG-16).

    Call ``lpips(pred, target)`` with images shaped ``(H, W, 3)`` or
    ``(N, H, W, 3)`` in ``[0, 1]``; returns a scalar perceptual distance
    (lower = more similar). Differentiable w.r.t. the images.
    """

    def __init__(self):
        super().__init__()
        blocks = []
        for block in _VGG_BLOCKS:
            convs = []
            for cin, cout in zip(block[:-1], block[1:]):
                convs.append(nn.Conv2d(cin, cout, kernel_size=3, padding=1))
            blocks.append(convs)
        self.blocks = blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Per-channel "lin" weights, one vector per tapped stage. Kept
        # non-negative (via abs in the forward) so the result is a valid
        # distance, matching the constrained LPIPS linear heads.
        self.lin_weights = [mx.random.uniform(shape=(c,)) * 0.1 for c in _TAP_CHANNELS]

    def _features(self, x: mx.array) -> list[mx.array]:
        feats = []
        h = x
        for bi, convs in enumerate(self.blocks):
            for conv in convs:
                h = nn.relu(conv(h))
            feats.append(h)
            if bi < len(self.blocks) - 1:
                h = self.pool(h)
        return feats

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        if pred.ndim == 3:
            pred, target = pred[None], target[None]
        shift = mx.array(_SHIFT)
        scale = mx.array(_SCALE)
        # [0, 1] -> [-1, 1] -> LPIPS scaling.
        a = (pred * 2.0 - 1.0 - shift) / scale
        b = (target * 2.0 - 1.0 - shift) / scale

        fa, fb = self._features(a), self._features(b)
        total = mx.zeros(())
        for feat_a, feat_b, w in zip(fa, fb, self.lin_weights):
            na = feat_a / mx.maximum(mx.linalg.norm(feat_a, axis=-1, keepdims=True), 1e-10)
            nb = feat_b / mx.maximum(mx.linalg.norm(feat_b, axis=-1, keepdims=True), 1e-10)
            diff = (na - nb) ** 2  # (N, H, W, C)
            total = total + mx.mean(mx.sum(diff * mx.abs(w), axis=-1))
        return total

    def load_weights_file(self, path: str) -> None:
        """Load converted VGG-16 + lin weights from a ``.safetensors``/``.npz`` file.

        The file must contain MLX arrays matching this module's parameter tree
        (see the conversion recipe in the docs).
        """
        self.load_weights(path)
