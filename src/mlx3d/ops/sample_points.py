"""Differentiable point sampling from mesh surfaces."""

import mlx.core as mx

from ..structures import Meshes

__all__ = ["sample_points_from_meshes"]


def sample_points_from_meshes(
    meshes: Meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
):
    """Uniformly sample points from mesh surfaces (area-weighted).

    Sampling locations are differentiable with respect to vertex positions
    (the face choice and barycentric coordinates are treated as constants,
    as in PyTorch3D).

    Args:
        meshes: a :class:`Meshes` batch of size N.
        num_samples: points per mesh.
        return_normals: also return the face normal at each sample.

    Returns:
        (N, num_samples, 3) samples, plus (N, num_samples, 3) normals if requested.
    """
    verts = meshes.verts_packed()
    faces = meshes.faces_packed()
    areas = meshes.faces_areas_packed()
    first_idx = meshes.mesh_to_faces_packed_first_idx().tolist()
    num_faces = meshes.num_faces_per_mesh.tolist()

    samples = []
    normals = []
    for i in range(len(meshes)):
        f0, nf = first_idx[i], num_faces[i]
        if nf == 0:
            samples.append(mx.zeros((num_samples, 3)))
            if return_normals:
                normals.append(mx.zeros((num_samples, 3)))
            continue
        a = areas[f0 : f0 + nf]
        logits = mx.log(mx.maximum(a, 1e-12))
        face_idx = mx.random.categorical(logits[None], num_samples=num_samples)[0]
        tri = faces[f0 + face_idx]  # (S, 3)
        v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]

        # Uniform barycentric sampling via the square-root trick.
        u = mx.random.uniform(shape=(num_samples, 1))
        v = mx.random.uniform(shape=(num_samples, 1))
        su = mx.sqrt(u)
        w0 = 1.0 - su
        w1 = su * (1.0 - v)
        w2 = su * v
        samples.append(w0 * v0 + w1 * v1 + w2 * v2)

        if return_normals:
            n = mx.linalg.cross(v1 - v0, v2 - v0)
            n = n / mx.maximum(mx.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
            normals.append(n)

    out = mx.stack(samples, axis=0)
    if return_normals:
        return out, mx.stack(normals, axis=0)
    return out
