"""Classical geometry processing: normal estimation, ICP, mesh decimation."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..structures import Meshes
from .knn import knn_points

__all__ = ["estimate_point_normals", "icp", "decimate_mesh"]


def estimate_point_normals(
    points: mx.array,
    k: int = 16,
    orient_towards: tuple[float, float, float] | mx.array | None = None,
) -> mx.array:
    """Estimate per-point normals by PCA over each point's ``k`` nearest neighbors.

    The normal is the eigenvector of the local covariance with the smallest
    eigenvalue. Normal orientation is inherently ambiguous; pass
    ``orient_towards`` (e.g. the camera position) to flip normals consistently
    toward that point.

    Args:
        points: ``(P, 3)`` point cloud.
        k: neighborhood size.
        orient_towards: optional ``(3,)`` location to orient normals toward.

    Returns:
        ``(P, 3)`` unit normals.
    """
    pts = np.asarray(points, dtype=np.float64)
    _, idx = knn_points(mx.array(pts.astype(np.float32)), mx.array(pts.astype(np.float32)), K=k)
    nbrs = pts[np.asarray(idx)]  # (P, k, 3)
    centered = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("pki,pkj->pij", centered, centered) / k  # (P, 3, 3)
    _, vecs = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = vecs[:, :, 0]  # smallest-eigenvalue direction
    if orient_towards is not None:
        target = np.asarray(orient_towards, dtype=np.float64).reshape(3)
        flip = ((target - pts) * normals).sum(-1) < 0
        normals[flip] *= -1.0
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    return mx.array(normals.astype(np.float32))


def icp(
    source: mx.array,
    target: mx.array,
    iters: int = 20,
    tol: float = 1e-6,
) -> dict[str, mx.array]:
    """Rigidly align ``source`` to ``target`` with Iterative Closest Point.

    Each iteration matches every source point to its nearest target point and
    solves for the best rigid transform (Kabsch/SVD), accumulating the total
    ``R``/``t`` such that ``aligned = source @ R.T + t``.

    Args:
        source: ``(N, 3)`` points to move.
        target: ``(M, 3)`` reference points.
        iters: maximum iterations.
        tol: stop when the mean matched-distance improves by less than this.

    Returns:
        dict with ``R`` ``(3, 3)``, ``t`` ``(3,)``, ``aligned`` ``(N, 3)``, and
        ``rmse`` (final root-mean-square matched distance).
    """
    tgt = np.asarray(target, dtype=np.float64)
    cur = np.asarray(source, dtype=np.float64)
    R_tot = np.eye(3)
    t_tot = np.zeros(3)
    prev = np.inf
    rmse = np.inf
    tgt_mx = mx.array(tgt.astype(np.float32))
    for _ in range(iters):
        d2, idx = knn_points(mx.array(cur.astype(np.float32)), tgt_mx, K=1)
        matched = tgt[np.asarray(idx)[:, 0]]
        rmse = float(np.sqrt(np.asarray(d2)[:, 0].mean()))
        mu_s, mu_t = cur.mean(0), matched.mean(0)
        h = (cur - mu_s).T @ (matched - mu_t)
        u, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r_step = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        t_step = mu_t - r_step @ mu_s
        cur = cur @ r_step.T + t_step
        R_tot = r_step @ R_tot
        t_tot = r_step @ t_tot + t_step
        if abs(prev - rmse) < tol:
            break
        prev = rmse
    return {
        "R": mx.array(R_tot.astype(np.float32)),
        "t": mx.array(t_tot.astype(np.float32)),
        "aligned": mx.array(cur.astype(np.float32)),
        "rmse": mx.array(float(rmse)),
    }


def decimate_mesh(meshes: Meshes, voxel_size: float) -> Meshes:
    """Simplify a mesh by vertex clustering on a regular voxel grid.

    Vertices in the same ``voxel_size`` cell collapse to their centroid; faces
    are remapped and degenerate ones (with a repeated vertex) dropped. Fast and
    robust — coarser than quadric-error decimation, but topology-agnostic.

    Returns a new single-mesh :class:`~mlx3d.structures.Meshes`.
    """
    v = np.asarray(meshes.verts_packed(), dtype=np.float64)
    f = np.asarray(meshes.faces_packed())
    cells = np.floor(v / voxel_size).astype(np.int64)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.reshape(-1)

    n_new = int(inv.max()) + 1
    new_v = np.zeros((n_new, 3))
    counts = np.zeros(n_new)
    np.add.at(new_v, inv, v)
    np.add.at(counts, inv, 1)
    new_v /= counts[:, None]

    new_f = inv[f]  # (F, 3) remapped
    nondeg = (
        (new_f[:, 0] != new_f[:, 1]) & (new_f[:, 1] != new_f[:, 2]) & (new_f[:, 0] != new_f[:, 2])
    )
    new_f = new_f[nondeg]
    new_f = np.unique(np.sort(new_f, axis=1), axis=0) if new_f.shape[0] else new_f
    return Meshes([mx.array(new_v.astype(np.float32))], [mx.array(new_f.astype(np.int32))])
