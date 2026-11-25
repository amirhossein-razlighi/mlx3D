"""Image-space resampling for NHWC feature maps, matching PyTorch semantics.

Pretrained vision models bake the exact behaviour of ``F.interpolate`` into
their weights, so ports must reproduce it, not approximate it. Both resamplers
here are separable: interpolation weights are computed once per (input, output)
size on the CPU and cached, and the GPU work is two gathers (bilinear) or two
small matmuls (bicubic).
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import numpy as np

__all__ = ["resize_bicubic", "resize_bilinear"]


@lru_cache(maxsize=64)
def _linear_taps(n_in: int, n_out: int) -> tuple[mx.array, mx.array, mx.array]:
    """Two-tap indices and weights for ``align_corners=True`` linear resampling."""
    scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
    src = np.arange(n_out, dtype=np.float32) * np.float32(scale)
    i0 = np.minimum(np.floor(src).astype(np.int32), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = (src - i0).astype(np.float32)
    return mx.array(i0), mx.array(i1), mx.array(w1)


def _lerp_axis(x: mx.array, size: int, axis: int) -> mx.array:
    if x.shape[axis] == size:
        return x
    i0, i1, w1 = _linear_taps(x.shape[axis], size)
    shape = [1] * x.ndim
    shape[axis] = size
    w1 = w1.reshape(shape).astype(x.dtype)
    a = mx.take(x, i0, axis=axis)
    b = mx.take(x, i1, axis=axis)
    return a + (b - a) * w1


def resize_bilinear(x: mx.array, size: tuple[int, int]) -> mx.array:
    """Bilinearly resize ``(B, H, W, C)`` to ``size = (H_out, W_out)``.

    Equivalent to ``F.interpolate(mode="bilinear", align_corners=True)``.
    """
    return _lerp_axis(_lerp_axis(x, size[0], axis=1), size[1], axis=2)


def _cubic_weights(t: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Keys cubic-convolution weights for the four taps around ``t`` in [0, 1)."""

    def near(d):  # |d| <= 1
        return ((a + 2) * d - (a + 3)) * d * d + 1

    def far(d):  # 1 < |d| < 2
        return ((a * d - 5 * a) * d + 8 * a) * d - 4 * a

    return np.stack([far(t + 1), near(t), near(1 - t), far(2 - t)], axis=-1)


@lru_cache(maxsize=64)
def _bicubic_matrix(n_in: int, scale_factor: float) -> np.ndarray:
    """``(n_out, n_in)`` bicubic resampling matrix for an explicit scale factor.

    Mirrors PyTorch with ``align_corners=False`` and a user-supplied
    ``scale_factor``: ``n_out = floor(n_in * scale)`` and source coordinates are
    ``(dst + 0.5) / scale - 0.5``, with border taps clamped.
    """
    n_out = int(np.floor(n_in * scale_factor))
    src = (np.arange(n_out, dtype=np.float64) + 0.5) / scale_factor - 0.5
    base = np.floor(src).astype(np.int64)
    weights = _cubic_weights(src - base)
    matrix = np.zeros((n_out, n_in), dtype=np.float64)
    for tap in range(4):
        idx = np.clip(base - 1 + tap, 0, n_in - 1)
        np.add.at(matrix, (np.arange(n_out), idx), weights[:, tap])
    return matrix


def resize_bicubic(x: mx.array, scale_factor: tuple[float, float]) -> mx.array:
    """Bicubically rescale ``(B, H, W, C)`` by ``scale_factor = (sy, sx)``.

    Equivalent to ``F.interpolate(mode="bicubic", align_corners=False,
    scale_factor=...)`` without antialiasing. Intended for small grids such as
    positional embeddings; the dense resampling matrices are ``O(H_out * H_in)``.
    """
    rows = mx.array(_bicubic_matrix(x.shape[1], float(scale_factor[0])), dtype=mx.float32)
    cols = mx.array(_bicubic_matrix(x.shape[2], float(scale_factor[1])), dtype=mx.float32)
    out = mx.einsum("oh,bhwc->bowc", rows, x.astype(mx.float32))
    out = mx.einsum("pw,bowc->bopc", cols, out)
    return out.astype(x.dtype)
