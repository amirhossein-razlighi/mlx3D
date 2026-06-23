"""Signed-distance-function (SDF) building blocks: analytic primitives, CSG
combinators, and helpers to sample an SDF on a grid and turn it into a mesh.

Every primitive maps points ``(..., 3) -> (...)`` signed distances (negative
inside, positive outside) and the combinators operate on those distance arrays,
so models compose with plain function calls::

    d = sdf_smooth_union(
        sdf_box(p, half_extents=(0.6, 0.6, 0.6)),
        sdf_sphere(p, radius=0.8),
        k=0.2,
    )

All operations are pure MLX, hence differentiable w.r.t. both the query points
and the shape parameters (radii, centers, ...), which makes them usable as
losses for SDF shape fitting as well as for mesh extraction via
:func:`sample_sdf_grid` / :func:`sdf_to_mesh`.
"""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx

from ..structures import Meshes
from .marching_cubes import marching_cubes

__all__ = [
    "sdf_sphere",
    "sdf_box",
    "sdf_torus",
    "sdf_plane",
    "sdf_union",
    "sdf_intersection",
    "sdf_difference",
    "sdf_smooth_union",
    "sdf_smooth_intersection",
    "sdf_smooth_difference",
    "sample_sdf_grid",
    "sdf_to_mesh",
]


def _as_vec3(value: tuple[float, float, float] | mx.array) -> mx.array:
    return mx.array(value, dtype=mx.float32) if not isinstance(value, mx.array) else value


# --------------------------------------------------------------------------- #
# Primitives: (..., 3) points -> (...) signed distances
# --------------------------------------------------------------------------- #
def sdf_sphere(
    points: mx.array,
    radius: float = 1.0,
    center: tuple[float, float, float] | mx.array = (0.0, 0.0, 0.0),
) -> mx.array:
    """Signed distance to a sphere.

    Args:
        points: ``(..., 3)`` query positions.
        radius: sphere radius.
        center: ``(3,)`` sphere center.

    Returns:
        ``(...)`` signed distances (negative inside).
    """
    return mx.linalg.norm(points - _as_vec3(center), axis=-1) - radius


def sdf_box(
    points: mx.array,
    half_extents: tuple[float, float, float] | mx.array = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] | mx.array = (0.0, 0.0, 0.0),
) -> mx.array:
    """Signed distance to an axis-aligned box.

    Args:
        points: ``(..., 3)`` query positions.
        half_extents: ``(3,)`` half-sizes of the box along each axis.
        center: ``(3,)`` box center.

    Returns:
        ``(...)`` signed distances (negative inside).
    """
    q = mx.abs(points - _as_vec3(center)) - _as_vec3(half_extents)
    outside = mx.linalg.norm(mx.maximum(q, 0.0), axis=-1)
    inside = mx.minimum(mx.max(q, axis=-1), 0.0)
    return outside + inside


def sdf_torus(
    points: mx.array,
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: tuple[float, float, float] | mx.array = (0.0, 0.0, 0.0),
) -> mx.array:
    """Signed distance to a torus lying in the xz-plane (y is the axis).

    Args:
        points: ``(..., 3)`` query positions.
        major_radius: distance from the center to the tube center.
        minor_radius: tube radius.
        center: ``(3,)`` torus center.

    Returns:
        ``(...)`` signed distances (negative inside the tube).
    """
    p = points - _as_vec3(center)
    xz = mx.stack([p[..., 0], p[..., 2]], axis=-1)
    radial = mx.linalg.norm(xz, axis=-1) - major_radius  # sqrt(x^2 + z^2) - R
    q = mx.stack([radial, p[..., 1]], axis=-1)
    return mx.linalg.norm(q, axis=-1) - minor_radius


def sdf_plane(
    points: mx.array,
    normal: tuple[float, float, float] | mx.array = (0.0, 1.0, 0.0),
    offset: float = 0.0,
) -> mx.array:
    """Signed distance to a plane ``dot(p, n) = offset``.

    Args:
        points: ``(..., 3)`` query positions.
        normal: plane normal (need not be unit length).
        offset: signed distance of the plane from the origin along ``normal``.

    Returns:
        ``(...)`` signed distances (negative on the side opposite ``normal``).
    """
    n = _as_vec3(normal)
    n = n / mx.linalg.norm(n)
    return mx.sum(points * n, axis=-1) - offset


# --------------------------------------------------------------------------- #
# CSG combinators: operate on signed-distance arrays
# --------------------------------------------------------------------------- #
def sdf_union(*distances: mx.array) -> mx.array:
    """Boolean union of shapes: the pointwise minimum of their distances.

    Args:
        *distances: two or more broadcastable distance arrays.

    Returns:
        Combined signed distances.
    """
    if len(distances) < 2:
        raise ValueError("sdf_union needs at least two distance arrays.")
    out = distances[0]
    for d in distances[1:]:
        out = mx.minimum(out, d)
    return out


def sdf_intersection(*distances: mx.array) -> mx.array:
    """Boolean intersection of shapes: the pointwise maximum of their distances.

    Args:
        *distances: two or more broadcastable distance arrays.

    Returns:
        Combined signed distances.
    """
    if len(distances) < 2:
        raise ValueError("sdf_intersection needs at least two distance arrays.")
    out = distances[0]
    for d in distances[1:]:
        out = mx.maximum(out, d)
    return out


def sdf_difference(distance_a: mx.array, distance_b: mx.array) -> mx.array:
    """Subtract shape ``b`` from shape ``a`` (``a`` minus ``b``).

    Args:
        distance_a: distances of the shape to keep.
        distance_b: distances of the shape to carve away.

    Returns:
        Combined signed distances.
    """
    return mx.maximum(distance_a, -distance_b)


def sdf_smooth_union(distance_a: mx.array, distance_b: mx.array, k: float = 0.1) -> mx.array:
    """Smooth (blended) union, after Inigo Quilez's polynomial smin.

    Args:
        distance_a: first shape's distances.
        distance_b: second shape's distances.
        k: blend radius; larger values round the join more. As ``k -> 0`` this
            approaches :func:`sdf_union`.

    Returns:
        Combined signed distances.
    """
    h = mx.clip(0.5 + 0.5 * (distance_b - distance_a) / k, 0.0, 1.0)
    return distance_b * (1.0 - h) + distance_a * h - k * h * (1.0 - h)


def sdf_smooth_intersection(distance_a: mx.array, distance_b: mx.array, k: float = 0.1) -> mx.array:
    """Smooth (blended) intersection.

    Args:
        distance_a: first shape's distances.
        distance_b: second shape's distances.
        k: blend radius; larger values round the join more.

    Returns:
        Combined signed distances.
    """
    return -sdf_smooth_union(-distance_a, -distance_b, k)


def sdf_smooth_difference(distance_a: mx.array, distance_b: mx.array, k: float = 0.1) -> mx.array:
    """Smooth (blended) subtraction of shape ``b`` from shape ``a``.

    Args:
        distance_a: distances of the shape to keep.
        distance_b: distances of the shape to carve away.
        k: blend radius; larger values round the join more.

    Returns:
        Combined signed distances.
    """
    return sdf_smooth_intersection(distance_a, -distance_b, k)


# --------------------------------------------------------------------------- #
# Grid sampling + mesh extraction
# --------------------------------------------------------------------------- #
def _resolve_bounds(
    bounds: float | tuple[float, float] | tuple[tuple[float, float], ...],
) -> tuple[mx.array, mx.array]:
    """Return ``(lo, hi)`` as ``(3,)`` arrays from a flexible ``bounds`` spec."""
    if isinstance(bounds, (int, float)):
        b = float(bounds)
        lo = mx.array([-b, -b, -b], dtype=mx.float32)
        hi = mx.array([b, b, b], dtype=mx.float32)
        return lo, hi
    seq = list(bounds)
    if len(seq) == 2 and all(isinstance(v, (int, float)) for v in seq):
        lo = mx.full((3,), float(seq[0]), dtype=mx.float32)
        hi = mx.full((3,), float(seq[1]), dtype=mx.float32)
        return lo, hi
    if len(seq) == 3:  # per-axis (min, max) pairs
        lo = mx.array([float(a) for a, _ in seq], dtype=mx.float32)
        hi = mx.array([float(b) for _, b in seq], dtype=mx.float32)
        return lo, hi
    raise ValueError("bounds must be a scalar, a (min, max) pair, or three (min, max) pairs.")


def sample_sdf_grid(
    sdf_fn: Callable[[mx.array], mx.array],
    resolution: int = 64,
    bounds: float | tuple[float, float] | tuple[tuple[float, float], ...] = 1.5,
) -> tuple[mx.array, tuple[float, float, float], tuple[float, float, float]]:
    """Evaluate an SDF on a regular grid, ready for :func:`marching_cubes`.

    Args:
        sdf_fn: callable mapping ``(N, 3)`` points to ``(N,)`` signed distances.
        resolution: number of grid samples per axis.
        bounds: sampling extent. A scalar ``b`` gives ``[-b, b]^3``; a
            ``(min, max)`` pair applies to all axes; three ``(min, max)`` pairs
            set per-axis extents.

    Returns:
        ``(volume, spacing, origin)`` where ``volume`` is a ``(R, R, R)`` grid
        of signed distances indexed ``[x, y, z]``, and ``spacing`` / ``origin``
        are the voxel size and grid origin to pass straight to
        :func:`marching_cubes`.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")
    lo, hi = _resolve_bounds(bounds)
    xs = mx.linspace(float(lo[0]), float(hi[0]), resolution)
    ys = mx.linspace(float(lo[1]), float(hi[1]), resolution)
    zs = mx.linspace(float(lo[2]), float(hi[2]), resolution)
    gx, gy, gz = mx.meshgrid(xs, ys, zs, indexing="ij")
    pts = mx.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    volume = sdf_fn(pts).reshape(resolution, resolution, resolution)
    spacing = tuple(float((hi[i] - lo[i]) / (resolution - 1)) for i in range(3))
    origin = (float(lo[0]), float(lo[1]), float(lo[2]))
    return volume, spacing, origin


def sdf_to_mesh(
    sdf_fn: Callable[[mx.array], mx.array],
    resolution: int = 64,
    bounds: float | tuple[float, float] | tuple[tuple[float, float], ...] = 1.5,
    level: float = 0.0,
) -> Meshes:
    """Extract a triangle mesh from an SDF callable.

    Convenience wrapper that samples ``sdf_fn`` on a grid with
    :func:`sample_sdf_grid` and runs :func:`marching_cubes` at ``level``.

    Args:
        sdf_fn: callable mapping ``(N, 3)`` points to ``(N,)`` signed distances.
        resolution: grid samples per axis.
        bounds: sampling extent (see :func:`sample_sdf_grid`).
        level: isovalue to extract (``0`` is the surface).

    Returns:
        A single-mesh :class:`~mlx3d.structures.Meshes`.
    """
    volume, spacing, origin = sample_sdf_grid(sdf_fn, resolution, bounds)
    return marching_cubes(volume, level=level, spacing=spacing, origin=origin)
