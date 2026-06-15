"""Differentiable image metrics (PSNR, SSIM) used for view-synthesis training."""

from __future__ import annotations

import mlx.core as mx

__all__ = ["psnr", "ssim", "ms_ssim", "l1_loss"]

# Standard 5-scale MS-SSIM weights (Wang et al., 2003).
_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


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


def _ssim_maps(pred, target, max_val, window_size, sigma):
    """Return mean SSIM and mean contrast-structure (cs) for a single scale."""
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
    cs_map = (2.0 * sigma_pt + c2) / (sigma_p + sigma_t + c2)
    ssim_map = (2.0 * mu_pt + c1) / (mu_pp + mu_tt + c1) * cs_map
    return ssim_map.mean(), cs_map.mean()


def _avgpool2(x: mx.array) -> mx.array:
    """2x2 average pool (stride 2), cropping to even spatial dims."""
    n, h, w, c = x.shape
    h2, w2 = h - (h % 2), w - (w % 2)
    x = x[:, :h2, :w2, :].reshape(n, h2 // 2, 2, w2 // 2, 2, c)
    return x.mean(axis=(2, 4))


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
    return _ssim_maps(pred, target, max_val, window_size, sigma)[0]


def ms_ssim(
    pred: mx.array,
    target: mx.array,
    max_val: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    weights: tuple[float, ...] = _MSSSIM_WEIGHTS,
) -> mx.array:
    """Multi-scale SSIM (Wang et al., 2003) — a perceptual image-quality metric.

    Evaluates SSIM across ``len(weights)`` scales (2x average-pooling between
    them), combining the contrast-structure term at coarse scales with the full
    SSIM at the finest. More correlated with perceived quality than single-scale
    SSIM, and unlike LPIPS needs no pretrained network. Fully differentiable;
    use ``1 - ms_ssim(...)`` as a loss.

    Args:
        pred: (H, W, C) or (N, H, W, C) image in [0, max_val].
        target: image with the same shape as ``pred``.
        weights: per-scale weights; the image must be larger than
            ``window_size * 2**(len(weights) - 1)``.

    Returns:
        Scalar MS-SSIM in [0, 1].
    """
    if pred.ndim == 3:
        pred, target = pred[None], target[None]
    n_scales = len(weights)
    min_hw = window_size * (2 ** (n_scales - 1))
    if min(pred.shape[1], pred.shape[2]) < min_hw:
        raise ValueError(
            f"ms_ssim needs spatial dims >= {min_hw} for {n_scales} scales; "
            f"got {pred.shape[1]}x{pred.shape[2]}. Use fewer weights or a larger image."
        )

    cs_factors = []
    last_ssim = None
    for i, _ in enumerate(weights):
        ssim_mean, cs_mean = _ssim_maps(pred, target, max_val, window_size, sigma)
        if i < n_scales - 1:
            cs_factors.append(mx.maximum(cs_mean, 0.0))
            pred, target = _avgpool2(pred), _avgpool2(target)
        else:
            last_ssim = mx.maximum(ssim_mean, 0.0)

    out = last_ssim ** weights[-1]
    for cs, w in zip(cs_factors, weights[:-1]):
        out = out * cs**w
    return out
