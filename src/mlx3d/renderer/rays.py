"""Ray sampling and differentiable volume rendering (the NeRF machinery)."""

import mlx.core as mx

__all__ = [
    "sample_along_rays",
    "sample_pdf",
    "volume_render",
]


def sample_along_rays(
    origins: mx.array,
    directions: mx.array,
    near: float,
    far: float,
    num_samples: int,
    stratified: bool = True,
) -> tuple[mx.array, mx.array]:
    """Sample points along rays between ``near`` and ``far``.

    Args:
        origins: (R, 3) ray origins.
        directions: (R, 3) ray directions (need not be normalized).
        num_samples: samples per ray.
        stratified: jitter samples within their bins (use ``False`` for eval).

    Returns:
        ``(points, t_vals)`` with shapes (R, S, 3) and (R, S).
    """
    R = origins.shape[0]
    t = mx.linspace(near, far, num_samples)  # (S,)
    t = mx.broadcast_to(t[None], (R, num_samples))
    if stratified:
        mids = 0.5 * (t[:, 1:] + t[:, :-1])
        upper = mx.concatenate([mids, t[:, -1:]], axis=-1)
        lower = mx.concatenate([t[:, :1], mids], axis=-1)
        u = mx.random.uniform(shape=(R, num_samples))
        t = lower + (upper - lower) * u
    points = origins[:, None, :] + t[..., None] * directions[:, None, :]
    return points, t


def sample_pdf(
    bins: mx.array,
    weights: mx.array,
    num_samples: int,
    deterministic: bool = False,
    eps: float = 1e-5,
) -> mx.array:
    """Importance-sample new t-values from a piecewise-constant PDF.

    Used for the hierarchical (fine) sampling stage of NeRF.

    Args:
        bins: (R, S+1) bin edges (e.g. midpoints of the coarse samples).
        weights: (R, S) unnormalized bin weights.
        num_samples: new samples per ray.

    Returns:
        (R, num_samples) sampled t-values. Gradients are not propagated
        through the sampling (treated as constants, as in NeRF).
    """
    weights = mx.stop_gradient(weights) + eps
    pdf = weights / weights.sum(axis=-1, keepdims=True)
    cdf = mx.cumsum(pdf, axis=-1)
    cdf = mx.concatenate([mx.zeros_like(cdf[:, :1]), cdf], axis=-1)  # (R, S+1)

    R = cdf.shape[0]
    if deterministic:
        u = mx.linspace(0.0, 1.0 - 1e-6, num_samples)
        u = mx.broadcast_to(u[None], (R, num_samples))
    else:
        u = mx.random.uniform(shape=(R, num_samples))

    # Inverse-CDF lookup: idx[r, m] = number of cdf entries <= u (O(S*M), fine on GPU).
    idx = (cdf[:, None, :-1] <= u[..., None]).sum(axis=-1) - 1  # (R, M)
    idx = mx.clip(idx, 0, cdf.shape[-1] - 2).astype(mx.int32)

    cdf_low = mx.take_along_axis(cdf, idx, axis=-1)
    cdf_high = mx.take_along_axis(cdf, idx + 1, axis=-1)
    bins_low = mx.take_along_axis(bins, idx, axis=-1)
    bins_high = mx.take_along_axis(bins, idx + 1, axis=-1)

    denom = mx.where(cdf_high - cdf_low < eps, mx.ones_like(cdf_low), cdf_high - cdf_low)
    frac = (u - cdf_low) / denom
    return bins_low + frac * (bins_high - bins_low)


def volume_render(
    densities: mx.array,
    colors: mx.array,
    t_vals: mx.array,
    directions: mx.array | None = None,
    white_background: bool = False,
) -> dict[str, mx.array]:
    """Composite densities and colors along rays with the NeRF quadrature rule.

    Args:
        densities: (R, S) non-negative volume densities (sigma).
        colors: (R, S, 3) per-sample RGB in [0, 1].
        t_vals: (R, S) sample distances along each ray.
        directions: optional (R, 3) ray directions; if given, deltas are
            scaled by their norms so densities live in world units.
        white_background: composite onto white instead of black.

    Returns:
        dict with ``rgb`` (R, 3), ``depth`` (R,), ``acc`` (R,) opacity, and
        ``weights`` (R, S).
    """
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = mx.concatenate(
        [deltas, mx.full(deltas[:, :1].shape, 1e10)], axis=-1
    )  # (R, S)
    if directions is not None:
        deltas = deltas * mx.linalg.norm(directions, axis=-1, keepdims=True)

    alpha = 1.0 - mx.exp(-densities * deltas)  # (R, S)
    # Transmittance: T_i = prod_{j<i} (1 - alpha_j)
    one_minus = mx.clip(1.0 - alpha, 1e-10, 1.0)
    trans = mx.cumprod(one_minus, axis=-1)
    trans = mx.concatenate([mx.ones_like(trans[:, :1]), trans[:, :-1]], axis=-1)
    weights = alpha * trans  # (R, S)

    rgb = mx.sum(weights[..., None] * colors, axis=-2)
    depth = mx.sum(weights * t_vals, axis=-1)
    acc = mx.sum(weights, axis=-1)
    if white_background:
        rgb = rgb + (1.0 - acc[..., None])
    return {"rgb": rgb, "depth": depth, "acc": acc, "weights": weights}
