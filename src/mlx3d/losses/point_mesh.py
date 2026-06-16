"""Point-to-mesh distance loss (point to nearest triangle surface)."""

from __future__ import annotations

import mlx.core as mx

from ..structures import Meshes

__all__ = ["point_mesh_face_distance", "closest_point_on_triangle"]


def closest_point_on_triangle(p: mx.array, a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    """Closest point on triangle ``(a, b, c)`` to each query ``p``.

    All inputs broadcast to a common ``(..., 3)`` shape. Uses the Voronoi-region
    method (Ericson, *Real-Time Collision Detection*), fully vectorized.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = mx.sum(ab * ap, axis=-1, keepdims=True)
    d2 = mx.sum(ac * ap, axis=-1, keepdims=True)

    bp = p - b
    d3 = mx.sum(ab * bp, axis=-1, keepdims=True)
    d4 = mx.sum(ac * bp, axis=-1, keepdims=True)

    cp = p - c
    d5 = mx.sum(ab * cp, axis=-1, keepdims=True)
    d6 = mx.sum(ac * cp, axis=-1, keepdims=True)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    eps = 1e-12
    # Edge AB: a + v*ab
    v_ab = d1 / mx.where(mx.abs(d1 - d3) < eps, mx.ones_like(d1), d1 - d3)
    pt_ab = a + mx.clip(v_ab, 0.0, 1.0) * ab
    # Edge AC: a + w*ac
    w_ac = d2 / mx.where(mx.abs(d2 - d6) < eps, mx.ones_like(d2), d2 - d6)
    pt_ac = a + mx.clip(w_ac, 0.0, 1.0) * ac
    # Edge BC: b + w*(c-b)
    denom_bc = (d4 - d3) + (d5 - d6)
    w_bc = (d4 - d3) / mx.where(mx.abs(denom_bc) < eps, mx.ones_like(denom_bc), denom_bc)
    pt_bc = b + mx.clip(w_bc, 0.0, 1.0) * (c - b)
    # Face interior (barycentric)
    denom = va + vb + vc
    denom = mx.where(mx.abs(denom) < eps, mx.ones_like(denom), denom)
    v = vb / denom
    w = vc / denom
    pt_face = a + ab * v + ac * w

    # Region selection by priority (vertices -> edges -> face).
    in_a = (d1 <= 0) & (d2 <= 0)
    in_b = (d3 >= 0) & (d4 <= d3)
    in_c = (d6 >= 0) & (d5 <= d6)
    in_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    in_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    in_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    out = pt_face
    out = mx.where(in_bc, pt_bc, out)
    out = mx.where(in_ac, pt_ac, out)
    out = mx.where(in_ab, pt_ab, out)
    out = mx.where(in_c, c, out)
    out = mx.where(in_b, b, out)
    out = mx.where(in_a, a, out)
    return out


def point_mesh_face_distance(
    meshes: Meshes,
    points: mx.array,
    face_chunk_size: int = 2048,
) -> mx.array:
    """Mean squared distance from each point to the nearest mesh face.

    Args:
        meshes: a single-mesh :class:`~mlx3d.structures.Meshes`.
        points: ``(P, 3)`` query points.
        face_chunk_size: faces processed per chunk to bound the ``(P, F)``
            memory. Lower it for very large meshes.

    Returns:
        Scalar mean over points of the squared distance to the closest triangle.
        Differentiable w.r.t. both points and mesh vertices.
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed().astype(mx.int32)
    tri = verts[faces]  # (F, 3, 3)
    p = points[:, None, :]  # (P, 1, 3)

    best = None
    for start in range(0, tri.shape[0], face_chunk_size):
        chunk = tri[start : start + face_chunk_size]  # (C, 3, 3)
        a = chunk[:, 0, :][None]  # (1, C, 3)
        b = chunk[:, 1, :][None]
        c = chunk[:, 2, :][None]
        closest = closest_point_on_triangle(p, a, b, c)  # (P, C, 3)
        d = mx.sum((p - closest) ** 2, axis=-1)  # (P, C)
        chunk_min = mx.min(d, axis=-1)  # (P,)
        best = chunk_min if best is None else mx.minimum(best, chunk_min)
    return mx.mean(best)
