"""Loader for the Instant-NGP / nerfstudio ``transforms.json`` format.

Unlike the Blender-synthetic loader, this format keeps all frames in a single
``transforms.json`` and usually stores explicit pinhole intrinsics
(``fl_x``/``fl_y``/``cx``/``cy``/``w``/``h``), with per-frame overrides allowed.
Camera poses use the same OpenGL ``transform_matrix`` convention as Blender.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from ..cameras import Camera
from .blender import BlenderDataset, _c2w_opengl_to_opencv_extrinsics
from .images import ImageCollection

__all__ = ["load_instant_ngp"]


def _resolve_path(root: str, file_path: str) -> str:
    """Resolve a frame ``file_path`` (which may or may not carry an extension)."""
    candidate = os.path.join(root, file_path)
    if os.path.exists(candidate):
        return candidate
    for ext in (".png", ".jpg", ".jpeg", ".JPG"):
        if os.path.exists(candidate + ext):
            return candidate + ext
    return candidate  # let the downstream open() raise a clear error


def load_instant_ngp(
    root: str,
    transforms_file: str = "transforms.json",
    downscale: int = 1,
    white_background: bool = False,
    cache: str = "ram",
) -> BlenderDataset:
    """Load an Instant-NGP / nerfstudio scene.

    Args:
        root: scene directory containing ``transforms_file``.
        transforms_file: name of the transforms JSON (default ``transforms.json``).
        downscale: integer image downscaling factor (intrinsics are scaled to match).
        white_background: composite RGBA images onto white (else black).
        cache: image storage policy; see :class:`~mlx3d.datasets.images.ImageCollection`.

    Returns:
        A :class:`~mlx3d.datasets.BlenderDataset` (``cameras`` + ``images``).
    """
    with open(os.path.join(root, transforms_file)) as f:
        meta = json.load(f)

    cameras: list[Camera] = []
    images = ImageCollection(cache=cache, downscale=downscale, white_background=white_background)

    g_angle = meta.get("camera_angle_x")
    for frame in meta["frames"]:
        images.append_file(_resolve_path(root, frame["file_path"]))
        h, w = images.shape_of(len(images) - 1)

        # Per-frame intrinsics override globals when present.
        def _get(key, default=None):
            return frame.get(key, meta.get(key, default))

        orig_w = float(_get("w", w * downscale))
        orig_h = float(_get("h", h * downscale))
        sx, sy = w / orig_w, h / orig_h  # account for downscaling

        fl_x = _get("fl_x")
        if fl_x is not None:
            fx = float(fl_x) * sx
            fy = float(_get("fl_y", fl_x)) * sy
            cx = float(_get("cx", orig_w / 2.0)) * sx
            cy = float(_get("cy", orig_h / 2.0)) * sy
        else:
            angle = frame.get("camera_angle_x", g_angle)
            if angle is None:
                raise ValueError("transforms.json must provide fl_x or camera_angle_x.")
            fx = fy = 0.5 * w / math.tan(0.5 * float(angle))
            cx, cy = w / 2.0, h / 2.0

        R, t = _c2w_opengl_to_opencv_extrinsics(
            np.asarray(frame["transform_matrix"], dtype=np.float64)
        )
        cameras.append(
            Camera(R=R, t=t, fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, znear=0.01, zfar=100.0)
        )

    return BlenderDataset(cameras=cameras, images=images)
