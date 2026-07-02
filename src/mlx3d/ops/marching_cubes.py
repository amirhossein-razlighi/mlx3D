"""Mesh extraction from scalar grids."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..structures import Meshes

__all__ = ["marching_cubes"]

# Cube corners: (8, 3) offsets (x, y, z) in {0, 1}^3.
_CUBE_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float32,
)

# Six tetrahedra that partition each cube.
_TETS = (
    (0, 5, 1, 6),
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
)

# The 6 edges of a tetrahedron by vertex index within the tet.
_EA = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
_EB = np.array([1, 2, 3, 2, 3, 3], dtype=np.int32)


def _empty_mesh() -> Meshes:
    return Meshes(
        [mx.zeros((0, 3), dtype=mx.float32)],
        [mx.zeros((0, 3), dtype=mx.int32)],
    )


def marching_cubes(
    volume: mx.array,
    level: float = 0.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Meshes:
    """Extract an isosurface mesh from a scalar grid.

    Uses a marching-tetrahedra decomposition to avoid the ambiguous cases of
    standard marching cubes. The implementation is fully vectorized over
    voxels using NumPy — no Python loop over individual voxels.

    Args:
        volume: ``(D, H, W)`` scalar grid (MLX array or anything numpy-castable).
        level: Isovalue.
        spacing: Voxel size ``(sx, sy, sz)``.
        origin: World-space position of the ``[0,0,0]`` grid corner.

    Returns:
        A single-mesh :class:`~mlx3d.structures.Meshes` object suitable for
        downstream losses, sampling, and rendering.
    """
    grid = np.asarray(volume, dtype=np.float32)
    if grid.ndim != 3:
        raise ValueError("volume must have shape (D, H, W).")
    d, h, w = grid.shape
    if min(d, h, w) < 2:
        return _empty_mesh()

    spacing_np = np.asarray(spacing, dtype=np.float32)
    origin_np = np.asarray(origin, dtype=np.float32)

    # Corner value tensor for every voxel: (d-1, h-1, w-1, 8)
    zi0, yi0, xi0 = np.mgrid[0 : d - 1, 0 : h - 1, 0 : w - 1]
    corner_vals = np.stack(
        [grid[zi0 + int(c[2]), yi0 + int(c[1]), xi0 + int(c[0])] for c in _CUBE_CORNERS],
        axis=-1,
    )

    # Active voxels: at least one corner inside, at least one outside.
    active = (corner_vals.min(-1) <= level) & (corner_vals.max(-1) >= level)
    if not active.any():
        return _empty_mesh()

    cv = corner_vals[active]  # (M, 8)
    # Corner world positions for active voxels: (M, 8, 3) with axes (x, y, z)
    base_xyz = np.stack(
        [
            xi0[active].astype(np.float32),
            yi0[active].astype(np.float32),
            zi0[active].astype(np.float32),
        ],
        axis=-1,
    )  # (M, 3)
    corner_pos = (base_xyz[:, None, :] + _CUBE_CORNERS[None]) * spacing_np + origin_np  # (M, 8, 3)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    n_verts = 0

    for tet_vids in _TETS:
        tv = cv[:, list(tet_vids)]  # (M, 4) scalar values at tet verts
        tp = corner_pos[:, list(tet_vids), :]  # (M, 4, 3) world positions

        inside = tv < level  # (M, 4)
        n_in = inside.sum(1)  # (M,)
        act = (n_in > 0) & (n_in < 4)
        if not act.any():
            continue

        tv_a = tv[act]  # (A, 4)
        tp_a = tp[act]  # (A, 4, 3)
        ins_a = inside[act]  # (A, 4)

        # Which edges cross the iso-surface: (A, 6)
        cross = ins_a[:, _EA] != ins_a[:, _EB]

        # Interpolation parameter t for each edge: (A, 6)
        va = tv_a[:, _EA]
        vb = tv_a[:, _EB]
        t = np.clip((level - va) / (vb - va + 1e-12), 0.0, 1.0)

        # World-space crossing points: (A, 6, 3)
        pa = tp_a[:, _EA, :]
        pb = tp_a[:, _EB, :]
        edge_pts = pa + t[:, :, None] * (pb - pa)

        n_cross = cross.sum(1)  # (A,)

        # Sort edge indices so that crossing edges come first, preserving
        # their relative order (same triangle winding as the scalar loop).
        order = np.argsort(~cross, axis=1, kind="stable")  # (A, 6)

        # 3 crossing edges → 1 triangle per tet
        m3 = n_cross == 3
        if m3.any():
            M3 = int(m3.sum())
            idx3 = order[m3, :3]  # (M3, 3)
            pts3 = edge_pts[m3][np.arange(M3)[:, None], idx3, :]  # (M3, 3, 3)
            starts = np.arange(M3, dtype=np.int32) * 3 + n_verts
            all_verts.append(pts3.reshape(-1, 3))
            all_faces.append(np.stack([starts, starts + 1, starts + 2], axis=1))
            n_verts += M3 * 3

        # 4 crossing edges → 2 triangles per tet (fan)
        m4 = n_cross == 4
        if m4.any():
            M4 = int(m4.sum())
            idx4 = order[m4, :4]  # (M4, 4)
            pts4 = edge_pts[m4][np.arange(M4)[:, None], idx4, :]  # (M4, 4, 3)
            starts = np.arange(M4, dtype=np.int32) * 4 + n_verts
            f4a = np.stack([starts, starts + 1, starts + 2], axis=1)
            f4b = np.stack([starts, starts + 2, starts + 3], axis=1)
            all_verts.append(pts4.reshape(-1, 3))
            all_faces.append(np.concatenate([f4a, f4b], axis=0))
            n_verts += M4 * 4

    if not all_verts:
        return _empty_mesh()

    verts = np.concatenate(all_verts, axis=0).astype(np.float32)
    faces = np.concatenate(all_faces, axis=0).astype(np.int64)

    # Weld duplicate vertices. Each tet emits its own crossing points, but a
    # surface edge is shared by several tets and its crossing point is computed
    # identically each time, so without welding the mesh has ~2.5x more verts
    # than it needs -- inflating memory and every downstream op (normals,
    # sampling, rendering). Quantize to a small fraction of the voxel size to
    # merge coincident points robustly, then remap faces to the unique verts.
    quantum = float(spacing_np.min()) * 1e-4 + 1e-12
    keys = np.round(verts / quantum).astype(np.int64)
    _, first_idx, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded_verts = verts[first_idx]
    welded_faces = inverse[faces].astype(np.int32)
    # Drop degenerate triangles that collapsed when their corners welded together.
    nondegen = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    welded_faces = welded_faces[nondegen]

    return Meshes(
        [mx.array(welded_verts.astype(np.float32))],
        [mx.array(welded_faces.astype(np.int32))],
    )
