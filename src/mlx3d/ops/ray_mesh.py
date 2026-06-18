"""Ray-triangle intersection (Möller-Trumbore) for true ray tracing."""

from __future__ import annotations

import mlx.core as mx

from ..structures import Meshes

__all__ = ["ray_mesh_intersect"]

_INF = 1e30


def _rays_intersect_aabb(
    origins: mx.array,
    directions: mx.array,
    bmin: mx.array,
    bmax: mx.array,
    eps: float,
) -> mx.array:
    """Return a per-ray mask for intersection with one axis-aligned box."""
    parallel = mx.abs(directions) <= eps
    inside_parallel = (origins >= (bmin - eps)) & (origins <= (bmax + eps))
    safe_d = mx.where(parallel, mx.ones_like(directions), directions)
    t0 = (bmin - origins) / safe_d
    t1 = (bmax - origins) / safe_d
    lo = mx.minimum(t0, t1)
    hi = mx.maximum(t0, t1)
    lo = mx.where(parallel, mx.where(inside_parallel, -_INF, _INF), lo)
    hi = mx.where(parallel, mx.where(inside_parallel, _INF, -_INF), hi)
    near = mx.max(lo, axis=-1)
    far = mx.min(hi, axis=-1)
    return far >= mx.maximum(near, eps)


def ray_mesh_intersect(
    meshes: Meshes,
    origins: mx.array,
    directions: mx.array,
    face_chunk_size: int = 4096,
    eps: float = 1e-8,
    aabb_cull: bool = True,
    return_stats: bool = False,
) -> dict[str, object]:
    """Nearest ray-triangle hit per ray (Möller-Trumbore).

    Args:
        meshes: a single-mesh :class:`~mlx3d.structures.Meshes`.
        origins: ``(R, 3)`` ray origins.
        directions: ``(R, 3)`` ray directions (need not be normalized; ``t`` is
            measured in units of ``directions``).
        face_chunk_size: faces per chunk, bounding the ``(R, F)`` memory.
        eps: parallel / degenerate-triangle tolerance.
        aabb_cull: if ``True``, test each face chunk's axis-aligned bounding
            box before running the expensive ray-triangle math. This preserves
            exact results and skips whole chunks for coherent rays or spatially
            sorted meshes.
        return_stats: include simple broad-phase counters useful for tests and
            profiling.

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
    chunks_total = 0
    chunks_skipped = 0
    face_tests = 0

    for start in range(0, faces.shape[0], face_chunk_size):
        fchunk = faces[start : start + face_chunk_size]  # (C, 3)
        tri = verts[fchunk]  # (C, 3, 3)
        chunks_total += 1
        if aabb_cull:
            bmin = mx.min(tri.reshape(-1, 3), axis=0)
            bmax = mx.max(tri.reshape(-1, 3), axis=0)
            chunk_hit = _rays_intersect_aabb(origins, directions, bmin, bmax, eps)
            if not bool(mx.any(chunk_hit)):
                chunks_skipped += 1
                continue
        face_tests += int(origins.shape[0]) * int(fchunk.shape[0])
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
    out = {
        "hit": hit_mask,
        "t": t_out,
        "face_idx": best_f,
        "bary": bary,
        "points": points,
    }
    if return_stats:
        out["stats"] = {
            "chunks_total": chunks_total,
            "chunks_skipped": chunks_skipped,
            "face_tests": face_tests,
        }
    return out
