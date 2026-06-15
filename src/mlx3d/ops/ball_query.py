"""Radius (ball) neighbor search."""

from __future__ import annotations

import mlx.core as mx

from .knn import knn_points

__all__ = ["ball_query"]


def ball_query(
    p1: mx.array,
    p2: mx.array,
    K: int = 32,
    radius: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """For each query in ``p1`` return up to ``K`` points of ``p2`` within ``radius``.

    Like PyTorch3D's ``ball_query``: it takes the ``K`` nearest neighbors and
    then drops any that fall outside the ball, so the result is the nearest
    in-radius neighbors. Slots with no neighbor are marked with index ``-1``.

    Args:
        p1: ``(N, P1, D)`` or ``(P1, D)`` query points.
        p2: ``(N, P2, D)`` or ``(P2, D)`` reference points.
        K: maximum neighbors per query.
        radius: ball radius.

    Returns:
        ``(dists, idx)`` of shape ``(..., P1, K)``: squared distances (``inf``
        for empty slots) and indices into ``p2`` (``-1`` for empty slots),
        sorted by increasing distance.
    """
    dists, idx = knn_points(p1, p2, K=K)
    inside = dists <= radius * radius
    idx = mx.where(inside, idx, mx.full(idx.shape, -1, dtype=idx.dtype))
    dists = mx.where(inside, dists, mx.full(dists.shape, mx.inf))
    return dists, idx
