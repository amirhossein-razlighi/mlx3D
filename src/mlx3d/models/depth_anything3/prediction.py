"""The result of a multi-view DA3 forward pass, with geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ...cameras import Camera
from ...structures import Pointclouds

__all__ = ["GeometryPrediction"]


@dataclass
class GeometryPrediction:
    """Per-view depth, confidence and cameras for ``S`` views at processing size.

    Cameras follow mlx3d's OpenCV convention (world-to-camera ``R``, ``t``); the
    world frame is the reference view's camera frame, at an arbitrary but
    consistent scale.

    Attributes:
        depth: ``(S, H, W)`` z-depth.
        depth_conf: ``(S, H, W)`` confidence (``>= 1``; higher is better).
        R, t: ``(S, 3, 3)`` / ``(S, 3)`` world-to-camera extrinsics.
        K: ``(S, 3, 3)`` intrinsics at the processing size.
        rgb: ``(S, H, W, 3)`` uint8 frames the model saw.
        original_sizes: each input's ``(width, height)``.
        ray, ray_conf: optional per-pixel ray maps, when requested.
    """

    depth: mx.array
    depth_conf: mx.array
    R: mx.array
    t: mx.array
    K: mx.array
    rgb: np.ndarray
    original_sizes: list[tuple[int, int]]
    ray: mx.array | None = None
    ray_conf: mx.array | None = None

    def __len__(self) -> int:
        return int(self.depth.shape[0])

    @property
    def image_size(self) -> tuple[int, int]:
        """Processing ``(width, height)``."""
        return int(self.depth.shape[2]), int(self.depth.shape[1])

    def camera(self, i: int, original_size: bool = False) -> Camera:
        """View ``i`` as a :class:`~mlx3d.cameras.Camera`.

        With ``original_size`` the intrinsics are rescaled to the input image
        resolution (e.g. for training on the full-resolution photos).
        """
        w, h = self.image_size
        K = np.array(self.K[i])
        sx = sy = 1.0
        if original_size:
            ow, oh = self.original_sizes[i]
            sx, sy, w, h = ow / w, oh / h, ow, oh
        return Camera(
            R=self.R[i],
            t=self.t[i],
            fx=float(K[0, 0] * sx),
            fy=float(K[1, 1] * sy),
            cx=float(K[0, 2] * sx),
            cy=float(K[1, 2] * sy),
            width=int(w),
            height=int(h),
        )

    def cameras(self, original_size: bool = False) -> list[Camera]:
        return [self.camera(i, original_size) for i in range(len(self))]

    def confidence_mask(self, quantile: float = 0.3) -> mx.array:
        """``(S, H, W)`` mask keeping pixels above the global confidence quantile."""
        conf = np.array(self.depth_conf)
        return mx.array(conf >= np.quantile(conf, quantile))

    def world_points(self, i: int) -> mx.array:
        """``(H, W, 3)`` world-space points of view ``i`` (pixel-center unprojection)."""
        w, h = self.image_size
        u = mx.arange(w, dtype=mx.float32) + 0.5
        v = mx.arange(h, dtype=mx.float32) + 0.5
        xy = mx.stack(mx.meshgrid(u, v, indexing="xy"), axis=-1)
        return self.camera(i).unproject_points(xy, self.depth[i])

    def point_cloud(
        self, conf_quantile: float = 0.3, max_points: int | None = 500_000, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fuse all views into one colored point cloud.

        Pixels below the ``conf_quantile`` confidence quantile are dropped, then
        the rest is uniformly subsampled to ``max_points``.

        Returns:
            ``(points, colors)``: ``(P, 3)`` float32 positions and ``(P, 3)``
            float32 colors in ``[0, 1]``.
        """
        mask = np.array(self.confidence_mask(conf_quantile))
        points = np.stack([np.array(self.world_points(i)) for i in range(len(self))])
        points, colors = points[mask], self.rgb[mask].astype(np.float32) / 255.0
        if max_points is not None and len(points) > max_points:
            keep = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
            points, colors = points[keep], colors[keep]
        return points.astype(np.float32), colors

    def to_pointclouds(self, **kwargs) -> Pointclouds:
        """:meth:`point_cloud` as a single-element :class:`~mlx3d.structures.Pointclouds`."""
        points, colors = self.point_cloud(**kwargs)
        return Pointclouds(points=[mx.array(points)], features=[mx.array(colors)])
