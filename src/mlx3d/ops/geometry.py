"""Classical geometry processing: normal estimation, ICP, mesh decimation."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ..structures import Meshes
from .knn import knn_points
from .marching_cubes import marching_cubes

__all__ = [
    "estimate_point_normals",
    "icp",
    "decimate_mesh",
    "poisson_reconstruction",
]


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


def poisson_reconstruction(
    points: mx.array,
    normals: mx.array,
    resolution: int = 64,
    padding: float = 0.1,
) -> Meshes:
    """Reconstruct a watertight surface mesh from oriented points (Poisson).

    Solves the Poisson equation on a regular grid: the oriented points define a
    vector field whose divergence drives an indicator function ``chi`` (via an
    FFT Poisson solve), then the surface is extracted with marching cubes at the
    iso-level passing through the input points. This is the regular-grid form of
    Screened Poisson Surface Reconstruction (Kazhdan et al.) -- the same physics
    without an octree.

    Args:
        points: ``(P, 3)`` surface samples.
        normals: ``(P, 3)`` oriented normals (need not be unit; consistent
            orientation matters, so estimate/orient them first).
        resolution: grid cells per axis.
        padding: fractional padding added around the point bounding box.

    Returns:
        A single-mesh :class:`~mlx3d.structures.Meshes`.
    """
    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)

    lo = pts.min(0)
    hi = pts.max(0)
    span = (hi - lo).max()
    pad = span * padding
    lo = lo - pad
    hi = lo + (span + 2 * pad)  # cube bounds
    size = hi - lo
    res = int(resolution)

    # Grid coordinates of each point in [0, res).
    g = (pts - lo) / size * res
    i0 = np.clip(np.floor(g).astype(np.int64), 0, res - 1)
    frac = g - i0

    # Trilinearly splat normals onto a (res, res, res, 3) vector field.
    vfield = np.zeros((res, res, res, 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = np.clip(i0[:, 0] + dx, 0, res - 1)
                iy = np.clip(i0[:, 1] + dy, 0, res - 1)
                iz = np.clip(i0[:, 2] + dz, 0, res - 1)
                w = (
                    (frac[:, 0] if dx else 1 - frac[:, 0])
                    * (frac[:, 1] if dy else 1 - frac[:, 1])
                    * (frac[:, 2] if dz else 1 - frac[:, 2])
                )
                np.add.at(vfield, (ix, iy, iz), w[:, None] * nrm)

    # Divergence of the field, then solve Laplacian(chi) = div via FFT.
    div = (
        np.gradient(vfield[..., 0], axis=0)
        + np.gradient(vfield[..., 1], axis=1)
        + np.gradient(vfield[..., 2], axis=2)
    )
    k = 2.0 * np.pi * np.fft.fftfreq(res)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    denom = -(kx**2 + ky**2 + kz**2)
    denom[0, 0, 0] = 1.0  # avoid divide-by-zero for the DC term
    chi_hat = np.fft.fftn(div) / denom
    chi_hat[0, 0, 0] = 0.0
    chi = np.real(np.fft.ifftn(chi_hat)).astype(np.float32)

    # Iso-level: the indicator value the surface (the input points) sits at.
    iso = float(chi[i0[:, 0], i0[:, 1], i0[:, 2]].mean())
    spacing = tuple((size / res).astype(float))
    return marching_cubes(mx.array(chi), level=iso, spacing=spacing, origin=tuple(lo.astype(float)))
