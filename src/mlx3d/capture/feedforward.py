"""Feed-forward camera poses for captures: Depth Anything 3 instead of SfM.

One network pass estimates every frame's intrinsics, pose and depth. Unlike
feature-matching SfM it does not need texture or wide overlap to register a
view, so it succeeds on sparse, low-texture or blurry captures where SfM drops
frames — at the cost of slightly less precise poses, which the trainer's joint
pose refinement then corrects. Its per-frame depth maps are kept as well, for
optional depth / normal prior losses during training.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEPTH_DIR",
    "FeedForwardConfig",
    "FeedForwardResult",
    "load_depth_prior",
    "run_feedforward_poses",
]

DEPTH_DIR = "depth"
"""Subdirectory of the capture holding one ``<image name>.npy`` depth map each."""


@dataclass
class FeedForwardConfig:
    model: str = "da3-base"
    """DA3 preset (``da3-small`` is ~2x faster; ``da3-large`` is non-commercial)."""
    resolution: int = 504
    max_views: int = 32
    """Frames per forward pass; longer captures are processed in anchored chunks."""
    num_anchors: int = 8
    conf_quantile: float = 0.5
    """Drop the least confident fraction of depth pixels from the initial points."""
    max_points: int = 100_000
    """Initial point budget, comparable to an SfM cloud; densification adds detail."""
    prior_conf_quantile: float = 0.2
    """Saved depth priors mark the least confident fraction of pixels unknown (0)."""
    seed: int = 0


@dataclass
class FeedForwardResult:
    sparse_dir: str
    num_views: int
    num_points: int
    seconds: float


def run_feedforward_poses(
    image_paths: list[str],
    out_root: str,
    config: FeedForwardConfig | None = None,
    log=print,
) -> FeedForwardResult:
    """Estimate poses + an initial point cloud for ``image_paths`` with DA3.

    Writes a COLMAP binary model to ``out_root/sparse/0`` with cameras at the
    original image resolution and a confidence-filtered, depth-fused point
    cloud, so the rest of the capture pipeline consumes it like SfM output.
    Per-frame depth maps go to ``out_root/depth/<name>.npy`` (float16 at the
    model's processing size; ``0`` = unknown), see :func:`load_depth_prior`.
    """
    import mlx.core as mx

    from ..datasets import save_colmap
    from ..models import DepthAnything3

    cfg = config or FeedForwardConfig()
    if len(image_paths) < 2:
        raise ValueError("Feed-forward poses need at least 2 images.")

    t0 = time.perf_counter()
    log(f"  Loading {cfg.model} ...")
    model = DepthAnything3.from_pretrained(cfg.model)
    log(f"  Predicting depth and cameras for {len(image_paths)} frames")
    pred = model.predict(
        image_paths,
        resolution=cfg.resolution,
        max_views=cfg.max_views,
        num_anchors=cfg.num_anchors,
    )
    points, colors = pred.point_cloud(
        conf_quantile=cfg.conf_quantile, max_points=cfg.max_points, seed=cfg.seed
    )
    # Hand the network's memory back before training starts (unified memory).
    del model
    mx.clear_cache()
    names = [os.path.basename(p) for p in image_paths]
    sparse_dir = save_colmap(out_root, pred.cameras(original_size=True), names, points, colors)
    _save_depth_priors(pred, names, os.path.join(out_root, DEPTH_DIR), cfg.prior_conf_quantile)
    seconds = time.perf_counter() - t0
    log(f"  {cfg.model}: {len(pred)} cameras, {len(points)} points in {seconds:.1f}s")
    return FeedForwardResult(
        sparse_dir=sparse_dir, num_views=len(pred), num_points=len(points), seconds=seconds
    )


def _save_depth_priors(pred, names: list[str], depth_dir: str, conf_quantile: float) -> None:
    os.makedirs(depth_dir, exist_ok=True)
    depth = np.array(pred.depth, dtype=np.float32)
    conf = np.array(pred.depth_conf)
    depth[conf < np.quantile(conf, conf_quantile)] = 0.0
    for name, d in zip(names, depth):
        np.save(os.path.join(depth_dir, f"{name}.npy"), d.astype(np.float16))


def load_depth_prior(path: str, size: tuple[int, int]):
    """Load a saved depth prior resized to ``size = (width, height)``.

    Unknown pixels (``0``) stay unknown: depth and validity are resized
    separately and pixels touched by an unknown neighbor are dropped.
    """
    import mlx.core as mx
    from PIL import Image

    depth = np.load(path).astype(np.float32)
    known = (depth > 0).astype(np.float32)

    def resize(a: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(a, mode="F").resize(size, Image.Resampling.BILINEAR))

    resized, valid = resize(depth), resize(known)
    return mx.array(np.where(valid > 0.999, resized, 0.0).astype(np.float32))
