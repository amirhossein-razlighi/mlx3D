"""Loader/writer for COLMAP sparse reconstructions (the standard input for 3DGS).

Reads and writes ``cameras.bin`` / ``images.bin`` / ``points3D.bin`` (binary
format) alongside the corresponding images directory.
"""

import os
import struct
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..cameras import Camera
from .images import ImageCollection

__all__ = ["ColmapDataset", "load_colmap", "save_colmap"]


@dataclass
class ColmapDataset:
    cameras: list[Camera]
    images: ImageCollection
    image_names: list[str]
    points: mx.array  # (P, 3) SfM points
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
        0: ("SIMPLE_PINHOLE", 3),
        1: ("PINHOLE", 4),
        2: ("SIMPLE_RADIAL", 4),
        3: ("RADIAL", 5),
        4: ("OPENCV", 8),
        5: ("OPENCV_FISHEYE", 8),
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
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
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


def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to COLMAP quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array(
            [0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 1e-12))
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def _camera_bin_entry(cam: Camera) -> tuple[int, int, int, tuple[float, ...]]:
    """Map a :class:`Camera` to a COLMAP (model_id, width, height, params) tuple."""
    fx, fy, cx, cy = float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)
    if cam.distortion is None:
        return 1, cam.width, cam.height, (fx, fy, cx, cy)  # PINHOLE
    d = tuple(float(v) for v in cam.distortion) + (0.0,) * 4
    if cam.fisheye:
        return 5, cam.width, cam.height, (fx, fy, cx, cy, *d[:4])  # OPENCV_FISHEYE
    return 4, cam.width, cam.height, (fx, fy, cx, cy, *d[:4])  # OPENCV


def save_colmap(
    root: str,
    cameras: list[Camera],
    image_names: list[str],
    points,
    point_colors,
    point_errors=None,
) -> str:
    """Write a COLMAP binary sparse model to ``root/sparse/0``.

    The inverse of :func:`load_colmap` (2D observations/tracks are not stored).
    Identical intrinsics are deduplicated into a shared COLMAP camera. Colors
    are float in [0, 1]; points/colors accept MLX or NumPy arrays.

    Returns the sparse model directory.
    """
    if len(cameras) != len(image_names):
        raise ValueError("save_colmap needs one image name per camera.")
    sparse = os.path.join(root, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    # cameras.bin (deduplicate identical intrinsics)
    entries: list[tuple] = []
    cam_ids: list[int] = []
    for cam in cameras:
        entry = _camera_bin_entry(cam)
        if entry not in entries:
            entries.append(entry)
        cam_ids.append(entries.index(entry) + 1)
    with open(os.path.join(sparse, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(entries)))
        for cam_id, (model_id, w, h, params) in enumerate(entries, start=1):
            f.write(struct.pack("<iiQQ", cam_id, model_id, w, h))
            f.write(struct.pack(f"<{len(params)}d", *params))

    # images.bin
    with open(os.path.join(sparse, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for i, (cam, name) in enumerate(zip(cameras, image_names)):
            q = _rotmat_to_qvec(np.array(cam.R, dtype=np.float64))
            t = np.array(cam.t, dtype=np.float64)
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<4d", *q))
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<i", cam_ids[i]))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D observations

    # points3D.bin
    xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rgb = np.asarray(point_colors, dtype=np.float64).reshape(-1, 3)
    if rgb.shape[0] != xyz.shape[0]:
        raise ValueError("points and point_colors must have the same length.")
    rgb8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    err = (
        np.full((xyz.shape[0],), -1.0)
        if point_errors is None
        else np.asarray(point_errors, dtype=np.float64).reshape(-1)
    )
    with open(os.path.join(sparse, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", xyz.shape[0]))
        for i in range(xyz.shape[0]):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(struct.pack("<3B", *rgb8[i]))
            f.write(struct.pack("<d", float(err[i])))
            f.write(struct.pack("<Q", 0))  # empty track
    return sparse


def _qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _image_size(path: str) -> tuple[int, int] | None:
    if not os.path.exists(path):
        return None
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def load_colmap(
    root: str,
    images_dir: str = "images",
    downscale: int = 1,
    load_images: bool = True,
    cache: str = "ram",
) -> ColmapDataset:
    """Load a COLMAP scene laid out as ``root/sparse/0`` + ``root/<images_dir>``.

    COLMAP already uses the OpenCV camera convention, so extrinsics map
    directly onto :class:`~mlx3d.cameras.Camera`.

    Args:
        cache: image storage policy — ``"ram"`` (float32, fastest),
            ``"uint8"`` (4x less memory, negligible per-access cost), or
            ``"disk"`` (paths only, decode on access; near-zero resident
            memory). See :class:`~mlx3d.datasets.images.ImageCollection`.
    """
    sparse = os.path.join(root, "sparse", "0")
    if not os.path.isdir(sparse):
        sparse = os.path.join(root, "sparse")
    cams_meta = _read_cameras_bin(os.path.join(sparse, "cameras.bin"))
    imgs_meta = _read_images_bin(os.path.join(sparse, "images.bin"))
    xyz, rgb = _read_points3d_bin(os.path.join(sparse, "points3D.bin"))

    cameras: list[Camera] = []
    images = ImageCollection(cache=cache, downscale=downscale, white_background=None)
    names: list[str] = []
    for _, meta in sorted(imgs_meta.items(), key=lambda kv: kv[1]["name"]):
        cam = cams_meta[meta["camera_id"]]
        params = cam["params"]
        distortion = None
        fisheye = False
        model = cam["model"]
        if model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            distortion = (params[3], 0.0, 0.0, 0.0)
        elif model == "RADIAL":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
            distortion = (params[3], params[4], 0.0, 0.0)
        elif model == "OPENCV":
            fx, fy, cx, cy = params[:4]
            distortion = tuple(params[4:8])  # k1, k2, p1, p2
        elif model == "OPENCV_FISHEYE":
            fx, fy, cx, cy = params[:4]
            distortion = tuple(params[4:8])  # k1, k2, k3, k4
            fisheye = True
        else:  # pragma: no cover
            raise ValueError(f"Unsupported camera model {model}")

        image_path = os.path.join(root, images_dir, meta["name"])
        file_size = _image_size(image_path)
        base_w, base_h = file_size or (cam["width"], cam["height"])
        # Use the actual image files as the training-resolution source of
        # truth. Some public COLMAP scenes ship pre-resized images while
        # retaining full-resolution camera metadata.
        W = max(1, base_w // downscale)
        H = max(1, base_h // downscale)
        sx = W / cam["width"]
        sy = H / cam["height"]
        R = _qvec_to_rotmat(meta["qvec"])
        t = meta["tvec"]

        cameras.append(
            Camera(
                R=mx.array(R.astype(np.float32)),
                t=mx.array(t.astype(np.float32)),
                fx=fx * sx,
                fy=fy * sy,
                cx=cx * sx,
                cy=cy * sy,
                width=int(W),
                height=int(H),
                distortion=distortion,
                fisheye=fisheye,
            )
        )
        if load_images:
            images.append_file(image_path)
        names.append(meta["name"])

    return ColmapDataset(
        cameras=cameras,
        images=images,
        image_names=names,
        points=mx.array(xyz.astype(np.float32)),
        point_colors=mx.array(rgb.astype(np.float32) / 255.0),
    )
