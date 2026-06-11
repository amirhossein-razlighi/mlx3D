"""Loader for the NeRF "Blender synthetic" dataset format (transforms_*.json)."""

import json
import math
import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..cameras import Camera
from .images import ImageCollection

__all__ = ["BlenderDataset", "load_blender"]


@dataclass
class BlenderDataset:
    cameras: list[Camera]
    images: ImageCollection  # (H, W, 3) in [0, 1], alpha-composited onto white or black

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, i: int) -> tuple[Camera, mx.array]:
        return self.cameras[i], self.images[i]


def _c2w_opengl_to_opencv_extrinsics(c2w: np.ndarray) -> tuple[mx.array, mx.array]:
    """Blender/NeRF c2w matrices use OpenGL camera axes (x right, y up,
    z backward). Convert to our OpenCV world-to-camera convention."""
    R_c2w = c2w[:3, :3]
    eye = c2w[:3, 3]
    # Camera axes in world coordinates, flipped to x right / y down / z forward.
    x_axis = R_c2w[:, 0]
    y_axis = -R_c2w[:, 1]
    z_axis = -R_c2w[:, 2]
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ eye
    return mx.array(R.astype(np.float32)), mx.array(t.astype(np.float32))


def load_blender(
    root: str,
    split: str = "train",
    downscale: int = 1,
    white_background: bool = True,
    cache: str = "ram",
) -> BlenderDataset:
    """Load a Blender-synthetic scene (lego, chair, ...).

    Args:
        root: scene directory containing ``transforms_<split>.json``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        downscale: integer image downscaling factor.
        white_background: composite the RGBA renders onto white (else black).
        cache: image storage policy (``"ram"``, ``"uint8"`` or ``"disk"``);
            see :class:`~mlx3d.datasets.images.ImageCollection`.
    """
    with open(os.path.join(root, f"transforms_{split}.json")) as f:
        meta = json.load(f)

    cameras: list[Camera] = []
    images = ImageCollection(
        cache=cache, downscale=downscale, white_background=white_background
    )
    for frame in meta["frames"]:
        img_path = os.path.join(root, frame["file_path"] + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(root, frame["file_path"])
        images.append_file(img_path)
        H, W = images.shape_of(len(images) - 1)

        focal = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])
        R, t = _c2w_opengl_to_opencv_extrinsics(
            np.asarray(frame["transform_matrix"], dtype=np.float64)
        )
        cameras.append(
            Camera(R=R, t=t, fx=focal, fy=focal, cx=W / 2.0, cy=H / 2.0,
                   width=W, height=H, znear=0.01, zfar=100.0)
        )

    return BlenderDataset(cameras=cameras, images=images)
