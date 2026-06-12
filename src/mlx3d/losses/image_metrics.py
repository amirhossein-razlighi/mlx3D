"""Differentiable image metrics (PSNR, SSIM) used for view-synthesis training."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["psnr", "ssim", "l1_loss"]


def l1_loss(pred: mx.array, target: mx.array) -> mx.array:
    return mx.abs(pred - target).mean()


def psnr(pred: mx.array, target: mx.array, max_val: float = 1.0) -> mx.array:
    """Peak signal-to-noise ratio in dB. Images in [0, max_val]."""
    mse = ((pred - target) ** 2).mean()
    return 10.0 * mx.log10((max_val * max_val) / mx.maximum(mse, 1e-12))


def _gaussian_window(size: int, sigma: float) -> mx.array:
    x = mx.arange(size, dtype=mx.float32) - (size - 1) / 2.0
    g = mx.exp(-(x * x) / (2.0 * sigma * sigma))
    return g / g.sum()


def ssim(
    pred: mx.array,
    target: mx.array,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> mx.array:
    """Structural similarity (mean SSIM) with a Gaussian window.

    Args:
        pred: (H, W, C) or (N, H, W, C) image in [0, max_val].
        target: image with the same shape as ``pred``.

    Returns:
        Scalar mean SSIM. Use ``1 - ssim(...)`` as a loss (the Gaussian
        filtering is fully differentiable).
    """
    if pred.ndim == 3:
        pred, target = pred[None], target[None]
    C = pred.shape[-1]

    g = _gaussian_window(window_size, sigma)
    win = (g[:, None] * g[None, :]).reshape(window_size, window_size, 1)
    # Depthwise filter: (C_out=C, kh, kw, C_in/groups=1)
    weight = mx.broadcast_to(win[None], (C, window_size, window_size, 1))
    pad = window_size // 2

    def filt(x):
        return mx.conv2d(x, weight, padding=pad, groups=C)

    mu_p = filt(pred)
    mu_t = filt(target)
    mu_pp = mu_p * mu_p
    mu_tt = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p = filt(pred * pred) - mu_pp
    sigma_t = filt(target * target) - mu_tt
    sigma_pt = filt(pred * target) - mu_pt

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2.0 * mu_pt + c1) * (2.0 * sigma_pt + c2)) / (
        (mu_pp + mu_tt + c1) * (sigma_p + sigma_t + c2)
    )
    return ssim_map.mean()
