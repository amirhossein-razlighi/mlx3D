"""Ray-triangle intersection (Möller-Trumbore) for true ray tracing."""

from __future__ import annotations

import mlx.core as mx

from ..structures import Meshes

__all__ = ["ray_mesh_intersect"]

_INF = 1e30


def ray_mesh_intersect(
    meshes: Meshes,
    origins: mx.array,
    directions: mx.array,
    face_chunk_size: int = 4096,
    eps: float = 1e-8,
) -> dict[str, mx.array]:
    """Nearest ray-triangle hit per ray (Möller-Trumbore).

    Args:
        meshes: a single-mesh :class:`~mlx3d.structures.Meshes`.
        origins: ``(R, 3)`` ray origins.
        directions: ``(R, 3)`` ray directions (need not be normalized; ``t`` is
            measured in units of ``directions``).
        face_chunk_size: faces per chunk, bounding the ``(R, F)`` memory.
        eps: parallel/`degenerate-triangle tolerance.

    Returns:
        dict with ``hit`` ``(R,)`` bool, ``t`` ``(R,)`` hit distance (``inf`` on
        miss), ``face_idx`` ``(R,)`` int32 (``-1`` on miss), ``bary`` ``(R, 3)``
        barycentric coords, and ``points`` ``(R, 3)`` world hit positions. The
        hit position is differentiable w.r.t. the mesh vertices.
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed().astype(mx.int32)
    o = origins[:, None, :]  # (R, 1, 3)
    d = directions[:, None, :]  # (R, 1, 3)

    best_t = mx.full((origins.shape[0],), _INF)
    best_f = mx.full((origins.shape[0],), -1, dtype=mx.int32)
    best_uv = mx.zeros((origins.shape[0], 2))

    for start in range(0, faces.shape[0], face_chunk_size):
        fchunk = faces[start : start + face_chunk_size]  # (C, 3)
        tri = verts[fchunk]  # (C, 3, 3)
        v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
        e1 = (v1 - v0)[None]  # (1, C, 3)
        e2 = (v2 - v0)[None]
        pvec = mx.linalg.cross(d, e2)  # (R, C, 3)
        det = mx.sum(e1 * pvec, axis=-1)  # (R, C)
        valid = mx.abs(det) > eps
        inv_det = 1.0 / mx.where(valid, det, mx.ones_like(det))

        tvec = o - v0[None]  # (R, C, 3)
        u = mx.sum(tvec * pvec, axis=-1) * inv_det
        qvec = mx.linalg.cross(tvec, e1)
        v = mx.sum(d * qvec, axis=-1) * inv_det
        t = mx.sum(e2 * qvec, axis=-1) * inv_det

        hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > eps)
        t_hit = mx.where(hit, t, mx.full(t.shape, _INF))  # (R, C)

        local = mx.argmin(t_hit, axis=-1)  # (R,)
        local_t = mx.take_along_axis(t_hit, local[:, None], axis=-1)[:, 0]
        closer = local_t < best_t
        best_t = mx.where(closer, local_t, best_t)
        best_f = mx.where(closer, (start + local).astype(mx.int32), best_f)
        cu = mx.take_along_axis(u, local[:, None], axis=-1)[:, 0]
        cv = mx.take_along_axis(v, local[:, None], axis=-1)[:, 0]
        best_uv = mx.where(closer[:, None], mx.stack([cu, cv], axis=-1), best_uv)

    hit_mask = best_f >= 0
    u, v = best_uv[:, 0], best_uv[:, 1]
    bary = mx.stack([1.0 - u - v, u, v], axis=-1) * hit_mask[:, None]
    t_out = mx.where(hit_mask, best_t, mx.full(best_t.shape, mx.inf))
    points = origins + mx.where(hit_mask, best_t, 0.0)[:, None] * directions
    return {
        "hit": hit_mask,
        "t": t_out,
        "face_idx": best_f,
        "bary": bary,
        "points": points,
    }
