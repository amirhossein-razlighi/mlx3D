"""Loader for COLMAP sparse reconstructions (the standard input for 3DGS).

Reads ``cameras.bin`` / ``images.bin`` / ``points3D.bin`` (binary format) and
the corresponding images directory.
"""

import os
import struct
from dataclasses import dataclass

import numpy as np

import mlx.core as mx

from ..cameras import Camera

__all__ = ["ColmapDataset", "load_colmap"]


@dataclass
class ColmapDataset:
    cameras: list[Camera]
    images: list[mx.array]
    image_names: list[str]
    points: mx.array        # (P, 3) SfM points
    point_colors: mx.array  # (P, 3) in [0, 1]

    def __len__(self) -> int:
        return len(self.cameras)

    def __getitem__(self, i: int) -> tuple[Camera, mx.array]:
        return self.cameras[i], self.images[i]

    @property
    def scene_extent(self) -> float:
        """Radius of the camera-centers bounding sphere (used to scale
        densification thresholds, as in 3DGS)."""
        centers = np.stack([np.array(c.camera_center) for c in self.cameras])
        center = centers.mean(axis=0)
        return float(np.linalg.norm(centers - center, axis=1).max()) * 1.1


def _read_cameras_bin(path: str) -> dict[int, dict]:
    cameras = {}
    # COLMAP camera models: id -> (name, num_params)
    models = {
        0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
        4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    }
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, n_params = models.get(model_id, (None, None))
            if name is None:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}.")
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {"model": name, "width": w, "height": h, "params": params}
    return cameras


def _read_images_bin(path: str) -> dict[int, dict]:
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            image_id = struct.unpack("<i", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points)  # skip 2D points
            images[image_id] = {
                "qvec": np.array(qvec), "tvec": np.array(tvec),
                "camera_id": camera_id, "name": name.decode("utf-8"),
            }
    return images


def _read_points3d_bin(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        xyz = np.empty((num, 3), dtype=np.float64)
        rgb = np.empty((num, 3), dtype=np.uint8)
        for i in range(num):
            f.read(8)  # point id
            xyz[i] = struct.unpack("<3d", f.read(24))
            rgb[i] = struct.unpack("<3B", f.read(3))
            f.read(8)  # reprojection error
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_len)
    return xyz, rgb


def _qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_colmap(
    root: str,
    images_dir: str = "images",
    downscale: int = 1,
    load_images: bool = True,
) -> ColmapDataset:
    """Load a COLMAP scene laid out as ``root/sparse/0`` + ``root/<images_dir>``.

    COLMAP already uses the OpenCV camera convention, so extrinsics map
    directly onto :class:`~mlx3d.cameras.Camera`.
    """
    from PIL import Image

    sparse = os.path.join(root, "sparse", "0")
    if not os.path.isdir(sparse):
        sparse = os.path.join(root, "sparse")
    cams_meta = _read_cameras_bin(os.path.join(sparse, "cameras.bin"))
    imgs_meta = _read_images_bin(os.path.join(sparse, "images.bin"))
    xyz, rgb = _read_points3d_bin(os.path.join(sparse, "points3D.bin"))

    cameras: list[Camera] = []
    images: list[mx.array] = []
    names: list[str] = []
    for _, meta in sorted(imgs_meta.items(), key=lambda kv: kv[1]["name"]):
        cam = cams_meta[meta["camera_id"]]
        params = cam["params"]
        if cam["model"] == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam["model"] in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[:4]
        elif cam["model"] in ("SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:  # pragma: no cover
            raise ValueError(f"Unsupported camera model {cam['model']}")

        W, H = cam["width"], cam["height"]
        s = 1.0 / downscale
        R = _qvec_to_rotmat(meta["qvec"])
        t = meta["tvec"]

        img_arr = None
        if load_images:
            img = Image.open(os.path.join(root, images_dir, meta["name"])).convert("RGB")
            if downscale > 1:
                img = img.resize((img.width // downscale, img.height // downscale),
                                 Image.LANCZOS)
            img_arr = mx.array(np.asarray(img, dtype=np.float32) / 255.0)
            H, W = img_arr.shape[:2]
            s = W / cam["width"]

        cameras.append(
            Camera(
                R=mx.array(R.astype(np.float32)),
                t=mx.array(t.astype(np.float32)),
                fx=fx * s, fy=fy * s, cx=cx * s, cy=cy * s,
                width=int(W), height=int(H),
            )
        )
        if img_arr is not None:
            images.append(img_arr)
        names.append(meta["name"])

    return ColmapDataset(
        cameras=cameras,
        images=images,
        image_names=names,
        points=mx.array(xyz.astype(np.float32)),
        point_colors=mx.array(rgb.astype(np.float32) / 255.0),
    )
