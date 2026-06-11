"""Mesh regularization losses for shape optimization."""

import numpy as np

import mlx.core as mx

from ..structures import Meshes

__all__ = ["mesh_edge_loss", "mesh_laplacian_smoothing", "mesh_normal_consistency"]


def mesh_edge_loss(meshes: Meshes, target_length: float = 0.0) -> mx.array:
    """Mean squared deviation of edge lengths from ``target_length``."""
    verts = meshes.verts_packed()
    edges = meshes.edges_packed()
    if edges.shape[0] == 0:
        return mx.array(0.0)
    v0 = verts[edges[:, 0]]
    v1 = verts[edges[:, 1]]
    lengths = mx.linalg.norm(v0 - v1, axis=-1)
    return ((lengths - target_length) ** 2).mean()


def mesh_laplacian_smoothing(meshes: Meshes, method: str = "uniform") -> mx.array:
    """Laplacian smoothing loss: mean norm of the uniform graph Laplacian.

    For each vertex, measures the distance to the centroid of its neighbors;
    minimizing it pulls the surface toward locally smooth configurations.
    """
    if method != "uniform":
        raise NotImplementedError("Only the 'uniform' Laplacian is implemented.")
    verts = meshes.verts_packed()
    edges = meshes.edges_packed()
    if edges.shape[0] == 0:
        return mx.array(0.0)

    # Sum of neighbors and neighbor counts via scatter-add over both directions.
    neighbor_sum = mx.zeros_like(verts)
    counts = mx.zeros((verts.shape[0],))
    e0, e1 = edges[:, 0], edges[:, 1]
    neighbor_sum = neighbor_sum.at[e0].add(verts[e1])
    neighbor_sum = neighbor_sum.at[e1].add(verts[e0])
    ones = mx.ones((edges.shape[0],))
    counts = counts.at[e0].add(ones)
    counts = counts.at[e1].add(ones)
    counts = mx.maximum(counts, 1.0)[:, None]
    lap = neighbor_sum / counts - verts
    return mx.linalg.norm(lap, axis=-1).mean()


def mesh_normal_consistency(meshes: Meshes) -> mx.array:
    """Penalty on the angle between normals of faces sharing an edge.

    Returns the mean of ``1 - cos(n_a, n_b)`` over all interior edges. The
    face-adjacency structure is topology-only and computed on CPU once.
    """
    faces_np = np.array(meshes.faces_packed())
    if faces_np.shape[0] == 0:
        return mx.array(0.0)

    edges = np.concatenate(
        [faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0
    )
    edges.sort(axis=1)
    face_idx = np.tile(np.arange(faces_np.shape[0]), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[order]
    faces_sorted = face_idx[order]
    same = (edges_sorted[1:] == edges_sorted[:-1]).all(axis=1)
    pair_a = faces_sorted[:-1][same]
    pair_b = faces_sorted[1:][same]
    if pair_a.size == 0:
        return mx.array(0.0)

    normals = meshes.faces_normals_packed()
    normals = normals / mx.maximum(
        mx.linalg.norm(normals, axis=-1, keepdims=True), 1e-12
    )
    na = normals[mx.array(pair_a.astype(np.int32))]
    nb = normals[mx.array(pair_b.astype(np.int32))]
    cos = mx.sum(na * nb, axis=-1)
    return (1.0 - cos).mean()
