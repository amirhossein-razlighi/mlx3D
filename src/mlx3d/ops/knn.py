"""Brute-force k-nearest-neighbor search between point sets.

On Apple-Silicon GPUs a tiled brute-force search saturates the hardware for
the sizes typical in 3D vision (up to a few hundred thousand points), so no
spatial acceleration structure is needed.
"""

import mlx.core as mx

__all__ = ["knn_points", "knn_gather"]


def _pairwise_sqdist(a: mx.array, b: mx.array) -> mx.array:
    """Squared euclidean distances between ``a`` (..., P1, D) and ``b`` (..., P2, D)."""
    a2 = mx.sum(a * a, axis=-1, keepdims=True)            # (..., P1, 1)
    b2 = mx.sum(b * b, axis=-1, keepdims=True)            # (..., P2, 1)
    cross = a @ b.swapaxes(-1, -2)                        # (..., P1, P2)
    d = a2 - 2.0 * cross + b2.swapaxes(-1, -2)
    return mx.maximum(d, 0.0)


def knn_points(
    p1: mx.array,
    p2: mx.array,
    K: int = 1,
    chunk_size: int = 16384,
) -> tuple[mx.array, mx.array]:
    """For each point in ``p1`` find its ``K`` nearest neighbors in ``p2``.

    Args:
        p1: (N, P1, D) or (P1, D) query points.
        p2: (N, P2, D) or (P2, D) reference points.
        K: number of neighbors.
        chunk_size: queries processed per tile to bound the (P1, P2) distance
            matrix memory.

    Returns:
        ``(dists, idx)`` of shapes (..., P1, K): squared distances and indices
        into ``p2``, sorted by increasing distance.
    """
    squeeze = p1.ndim == 2
    if squeeze:
        p1, p2 = p1[None], p2[None]
    P1, P2 = p1.shape[1], p2.shape[1]
    K = min(K, P2)

    dists_out, idx_out = [], []
    for start in range(0, P1, chunk_size):
        d = _pairwise_sqdist(p1[:, start : start + chunk_size], p2)  # (N, c, P2)
        if K == 1:
            idx = mx.argmin(d, axis=-1, keepdims=True)
            dist = mx.take_along_axis(d, idx, axis=-1)
        else:
            idx_full = mx.argsort(d, axis=-1)[..., :K]
            dist = mx.take_along_axis(d, idx_full, axis=-1)
            idx = idx_full
        dists_out.append(dist)
        idx_out.append(idx)

    dists = mx.concatenate(dists_out, axis=1)
    idx = mx.concatenate(idx_out, axis=1)
    if squeeze:
        dists, idx = dists[0], idx[0]
    return dists, idx


def knn_gather(x: mx.array, idx: mx.array) -> mx.array:
    """Gather neighbor features: ``x`` (N, P2, C), ``idx`` (N, P1, K) -> (N, P1, K, C)."""
    N, P1, K = idx.shape
    C = x.shape[-1]
    flat = idx.reshape(N, P1 * K)
    gathered = mx.take_along_axis(x, flat[..., None].astype(mx.int32), axis=1)
    return gathered.reshape(N, P1, K, C)
