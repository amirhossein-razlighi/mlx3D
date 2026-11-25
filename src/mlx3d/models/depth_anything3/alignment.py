"""Similarity alignment between camera sets, for chunked multi-view inference.

Each DA3 forward pass expresses its cameras in its own frame (its reference
view, at its own scale). To stitch passes together we estimate the similarity
``x_dst = s * Rw @ x_src + tw`` that maps one pass's frame onto another's from
cameras both passes saw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Similarity", "fit_similarity"]


@dataclass(frozen=True)
class Similarity:
    """``x -> scale * rotation @ x + translation`` acting on world points."""

    scale: float
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)

    def apply_to_cameras(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Re-express world-to-camera extrinsics ``(N, 3, 3)``, ``(N, 3)`` in the
        target frame (scaled so depths should be multiplied by ``scale``)."""
        centers = -np.einsum("nji,nj->ni", R, t)
        R_new = R @ self.rotation.T
        centers_new = self.scale * centers @ self.rotation.T + self.translation
        t_new = -np.einsum("nij,nj->ni", R_new, centers_new)
        return R_new, t_new


def _project_to_so3(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def fit_similarity(
    R_src: np.ndarray, t_src: np.ndarray, R_dst: np.ndarray, t_dst: np.ndarray
) -> Similarity:
    """Least-squares similarity mapping source cameras onto destination cameras.

    Rotation comes from the camera orientations (robust even when the shared
    cameras are nearly collinear), then scale and translation from the camera
    centers given that rotation.
    """
    R_src, t_src = np.asarray(R_src, np.float64), np.asarray(t_src, np.float64)
    R_dst, t_dst = np.asarray(R_dst, np.float64), np.asarray(t_dst, np.float64)
    # Camera-to-world rotations satisfy c2w_dst = Rw @ c2w_src, so Rw is the
    # rotation closest to sum_n R_dst[n].T @ R_src[n].
    rotation = _project_to_so3(np.einsum("nji,njk->ik", R_dst, R_src))

    c_src = -np.einsum("nji,nj->ni", R_src, t_src) @ rotation.T
    c_dst = -np.einsum("nji,nj->ni", R_dst, t_dst)
    src0, dst0 = c_src - c_src.mean(0), c_dst - c_dst.mean(0)
    denom = float((src0**2).sum())
    scale = float((src0 * dst0).sum() / denom) if denom > 1e-12 else 1.0
    if scale <= 0:
        scale = 1.0
    translation = c_dst.mean(0) - scale * c_src.mean(0)
    return Similarity(scale=scale, rotation=rotation, translation=translation)
