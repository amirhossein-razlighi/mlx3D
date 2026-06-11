"""Chamfer distance between point clouds."""

import mlx.core as mx

from ..ops.knn import knn_points

__all__ = ["chamfer_distance"]


def chamfer_distance(
    x: mx.array,
    y: mx.array,
    x_normals: mx.array | None = None,
    y_normals: mx.array | None = None,
    single_directional: bool = False,
) -> tuple[mx.array, mx.array | None]:
    """Bidirectional (squared) chamfer distance between batches of point clouds.

    Args:
        x: (N, P1, 3) or (P1, 3).
        y: (N, P2, 3) or (P2, 3).
        x_normals: optional (N, P1, 3) normals for ``x``.
        y_normals: optional (N, P2, 3) normals for ``y``; if both are given,
            a normal-consistency term (1 - |cos|) is also returned.
        single_directional: only use the x -> y direction.

    Returns:
        ``(loss, loss_normals)``; ``loss_normals`` is ``None`` if normals were
        not provided.
    """
    squeeze = x.ndim == 2
    if squeeze:
        x, y = x[None], y[None]
        x_normals = x_normals[None] if x_normals is not None else None
        y_normals = y_normals[None] if y_normals is not None else None

    d_xy, idx_xy = knn_points(x, y, K=1)
    cham_x = d_xy[..., 0].mean()
    if single_directional:
        loss = cham_x
    else:
        d_yx, idx_yx = knn_points(y, x, K=1)
        loss = cham_x + d_yx[..., 0].mean()

    loss_normals = None
    if x_normals is not None and y_normals is not None:
        nn_y = mx.take_along_axis(
            y_normals,
            mx.broadcast_to(idx_xy[..., 0:1].astype(mx.int32), x.shape),
            axis=1,
        )
        cos_x = mx.sum(x_normals * nn_y, axis=-1)
        loss_normals = (1.0 - mx.abs(cos_x)).mean()
        if not single_directional:
            nn_x = mx.take_along_axis(
                x_normals,
                mx.broadcast_to(idx_yx[..., 0:1].astype(mx.int32), y.shape),
                axis=1,
            )
            cos_y = mx.sum(y_normals * nn_x, axis=-1)
            loss_normals = loss_normals + (1.0 - mx.abs(cos_y)).mean()

    return loss, loss_normals
