"""High-level Gaussian Splatting rendering: projection + binning + rasterization."""

import mlx.core as mx

from ..cameras import Camera
from .projection import project_gaussians
from .rasterize import rasterize, rasterize_depth
from .sh import eval_sh
from .tiles import bin_gaussians

__all__ = ["render_gaussians", "render_gaussian_depth"]


def render_gaussians(
    camera: Camera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    colors: mx.array | None = None,
    sh: mx.array | None = None,
    sh_degree: int = 3,
    background: mx.array | None = None,
    refine_tiles: bool = False,
) -> dict[str, mx.array]:
    """Render 3D Gaussians from a camera. Differentiable end to end.

    Args:
        camera: viewing camera.
        means: (N, 3) Gaussian centers.
        quats: (N, 4) rotations (w, x, y, z).
        scales: (N, 3) per-axis standard deviations (positive; apply your
            activation, e.g. ``mx.exp``, before calling).
        opacities: (N,) in [0, 1] (apply sigmoid before calling).
        colors: (N, 3) RGB in [0, 1]. Mutually exclusive with ``sh``.
        sh: (N, K, 3) spherical-harmonic coefficients (K >= (sh_degree+1)^2);
            view-dependent color is evaluated per Gaussian toward the camera.
        background: (3,) background color (default black).
        refine_tiles: experimental conservative ellipse/tile rejection after
            square radius binning. Off by default because its extra MLX work is
            not faster on all scenes.

    Returns:
        dict with ``image`` (H, W, 3), ``alpha`` (H, W), plus the projection
        outputs (``means2d``, ``depths``, ``radii``) for densification
        bookkeeping.
    """
    if (colors is None) == (sh is None):
        raise ValueError("Provide exactly one of `colors` or `sh`.")

    proj = project_gaussians(camera, means, quats, scales)

    if sh is not None:
        dirs = means - camera.camera_center
        dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
        colors = mx.maximum(eval_sh(sh_degree, sh, mx.stop_gradient(dirs)), 0.0)

    sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
        proj["means2d"],
        proj["radii"],
        proj["depths"],
        camera.width,
        camera.height,
        conics=proj["conics"] if refine_tiles else None,
    )

    out = rasterize(
        proj["means2d"],
        proj["conics"],
        colors,
        opacities,
        sorted_ids,
        tile_ranges,
        camera.width,
        camera.height,
        tiles_x,
        tiles_y,
        background=background,
    )
    out.update(
        {"means2d": proj["means2d"], "depths": proj["depths"], "radii": proj["radii"]}
    )
    return out


def render_gaussian_depth(
    camera: Camera,
    means: mx.array,
    quats: mx.array,
    scales: mx.array,
    opacities: mx.array,
    refine_tiles: bool = False,
) -> dict[str, mx.array]:
    """Render expected depth and alpha from 3D Gaussians.

    This is a forward-only diagnostic/viewer path. It uses the same projection,
    tile binning, and alpha compositing math as RGB splatting, but accumulates
    transmittance-weighted depth instead of color.
    """
    proj = project_gaussians(camera, means, quats, scales)
    sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(
        proj["means2d"],
        proj["radii"],
        proj["depths"],
        camera.width,
        camera.height,
        conics=proj["conics"] if refine_tiles else None,
    )
    out = rasterize_depth(
        proj["means2d"],
        proj["conics"],
        opacities,
        proj["depths"],
        sorted_ids,
        tile_ranges,
        camera.width,
        camera.height,
        tiles_x,
        tiles_y,
    )
    out.update(
        {"means2d": proj["means2d"], "depths": proj["depths"], "radii": proj["radii"]}
    )
    return out
