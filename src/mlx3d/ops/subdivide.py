"""Loop-style midpoint subdivision of triangle meshes."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..structures import Meshes

__all__ = ["subdivide_meshes"]


def subdivide_meshes(meshes: Meshes) -> Meshes:
    """Subdivide each triangle into four by inserting edge midpoints.

    Each face ``(v0, v1, v2)`` becomes four faces using the midpoints of its
    three edges; midpoints are shared between adjacent faces (so the result is
    watertight where the input was). New vertex positions are the differentiable
    averages of their two endpoints, so gradients flow back to the originals.

    Operates on a single mesh and returns a new :class:`~mlx3d.structures.Meshes`.
    """
    if len(meshes) != 1:
        raise ValueError("subdivide_meshes operates on one mesh at a time.")
    verts = meshes.verts_packed()
    faces = np.asarray(meshes.faces_packed())
    if faces.shape[0] == 0:
        return Meshes([verts], [meshes.faces_packed()])

    v = faces.shape[0]
    # Per-corner edges (canonicalized) -> unique edge ids, in face-corner order.
    corner_edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )  # (3F, 2)
    keyed = np.sort(corner_edges, axis=1)
    unique_edges, inverse = np.unique(keyed, axis=0, return_inverse=True)
    inverse = inverse.reshape(3, v).T  # (F, 3): edge id for (v0v1, v1v2, v2v0)

    num_verts = int(verts.shape[0])
    # New vertices = original verts followed by edge midpoints (differentiable).
    edges = mx.array(unique_edges.astype(np.int32))
    midpoints = 0.5 * (verts[edges[:, 0]] + verts[edges[:, 1]])
    new_verts = mx.concatenate([verts, midpoints], axis=0)

    # Midpoint vertex index for each face corner-edge.
    m = inverse + num_verts  # (F, 3)
    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
    new_faces = np.concatenate(
        [
            np.stack([v0, m01, m20], axis=1),
            np.stack([v1, m12, m01], axis=1),
            np.stack([v2, m20, m12], axis=1),
            np.stack([m01, m12, m20], axis=1),
        ],
        axis=0,
    ).astype(np.int32)

    return Meshes([new_verts], [mx.array(new_faces)])
