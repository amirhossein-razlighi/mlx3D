import json
import os
import struct

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx3d.datasets import load_blender, load_colmap, load_instant_ngp


def _write_ngp_scene(root, n_frames=3, w=60, h=40):
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    frames = []
    for i in range(n_frames):
        c2w = np.eye(4)
        c2w[2, 3] = 4.0  # OpenGL c2w camera at +z
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(
            os.path.join(root, "images", f"r{i}.png")
        )
        frames.append({"file_path": f"images/r{i}.png", "transform_matrix": c2w.tolist()})
    meta = {"w": w, "h": h, "fl_x": 50.0, "fl_y": 50.0, "cx": w / 2, "cy": h / 2, "frames": frames}
    with open(os.path.join(root, "transforms.json"), "w") as f:
        json.dump(meta, f)


def test_load_instant_ngp(tmp_path):
    root = str(tmp_path)
    _write_ngp_scene(root)
    ds = load_instant_ngp(root)
    assert len(ds) == 3
    cam, img = ds[0]
    assert img.shape == (40, 60, 3)
    assert cam.fx == 50.0 and cam.cx == 30.0
    assert (cam.width, cam.height) == (60, 40)
    np.testing.assert_allclose(np.array(cam.camera_center), [0, 0, 4.0], atol=1e-5)


def test_load_instant_ngp_downscale_scales_intrinsics(tmp_path):
    root = str(tmp_path)
    _write_ngp_scene(root)
    ds = load_instant_ngp(root, downscale=2)
    cam, img = ds[0]
    assert (cam.width, cam.height) == (30, 20)
    assert cam.fx == 25.0 and cam.cx == 15.0  # intrinsics halve with the image


def _write_blender_scene(root, n_frames=3, size=32):
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    frames = []
    for i in range(n_frames):
        # Camera on the -z axis looking at origin, OpenGL c2w (identity-ish).
        c2w = np.eye(4)
        c2w[2, 3] = 4.0  # OpenGL camera at +z looks down -z toward origin
        frames.append({"file_path": f"train/r_{i}", "transform_matrix": c2w.tolist()})
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[8:24, 8:24] = [255, 0, 0, 255]  # red square, transparent elsewhere
        Image.fromarray(rgba).save(os.path.join(root, "train", f"r_{i}.png"))
    meta = {"camera_angle_x": 0.8, "frames": frames}
    with open(os.path.join(root, "transforms_train.json"), "w") as f:
        json.dump(meta, f)


def test_load_blender(tmp_path):
    root = str(tmp_path)
    _write_blender_scene(root)
    ds = load_blender(root, "train", white_background=True)
    assert len(ds) == 3
    cam, img = ds[0]
    assert img.shape == (32, 32, 3)
    # Transparent corner composited to white.
    np.testing.assert_allclose(np.array(img[0, 0]), [1.0, 1.0, 1.0], atol=1e-5)
    # Red square present.
    np.testing.assert_allclose(np.array(img[16, 16]), [1.0, 0.0, 0.0], atol=1e-5)
    # Camera at (0, 0, 4) in world space (OpenGL c2w translation).
    np.testing.assert_allclose(np.array(cam.camera_center), [0, 0, 4.0], atol=1e-5)
    # Looking toward the origin: a world point at the origin projects to center.
    xy, z = cam.project_points(mx.zeros((1, 3)))
    np.testing.assert_allclose(np.array(xy[0]), [16.0, 16.0], atol=1e-4)
    assert float(z[0]) > 0


def _write_colmap_scene(root, size=16):
    sparse = os.path.join(root, "sparse", "0")
    images_dir = os.path.join(root, "images")
    os.makedirs(sparse, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # cameras.bin: one PINHOLE camera.
    with open(os.path.join(sparse, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, size, size))
        f.write(struct.pack("<4d", 20.0, 20.0, size / 2, size / 2))

    # images.bin: two images with identity rotation.
    with open(os.path.join(sparse, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", 2))
        for i in range(2):
            f.write(struct.pack("<i", i + 1))
            f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))  # qvec (w,x,y,z)
            f.write(struct.pack("<3d", 0.0, 0.0, float(2 + i)))  # tvec
            f.write(struct.pack("<i", 1))
            f.write(f"img_{i}.png".encode() + b"\x00")
            f.write(struct.pack("<Q", 0))  # no 2D points

    # points3D.bin: 5 points.
    with open(os.path.join(sparse, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", 5))
        for i in range(5):
            f.write(struct.pack("<Q", i))
            f.write(struct.pack("<3d", float(i), 0.0, 1.0))
            f.write(struct.pack("<3B", 255, 128, 0))
            f.write(struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", 0))  # empty track

    for i in range(2):
        Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8)).save(
            os.path.join(images_dir, f"img_{i}.png")
        )


def test_load_colmap(tmp_path):
    root = str(tmp_path)
    _write_colmap_scene(root)
    ds = load_colmap(root)
    assert len(ds) == 2
    assert ds.points.shape == (5, 3)
    np.testing.assert_allclose(np.array(ds.point_colors[0]), [1.0, 128 / 255, 0.0], atol=1e-5)
    cam, img = ds[0]
    assert img.shape == (16, 16, 3)
    assert cam.fx == 20.0
    # tvec=(0,0,2), R=I -> camera center at (0,0,-2).
    np.testing.assert_allclose(np.array(cam.camera_center), [0, 0, -2.0], atol=1e-5)
    assert ds.scene_extent > 0


@pytest.mark.parametrize("cache", ["ram", "uint8", "disk"])
def test_image_cache_modes_identical(tmp_path, cache):
    root = str(tmp_path)
    _write_blender_scene(root)
    ds = load_blender(root, "train", cache=cache)
    img = ds.images[0]
    assert img.shape == (32, 32, 3)
    assert img.dtype == mx.float32
    np.testing.assert_allclose(np.array(img[16, 16]), [1.0, 0.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(np.array(img[0, 0]), [1.0, 1.0, 1.0], atol=1e-2)
    # Resident memory ordering: disk < uint8 < ram.
    if cache == "disk":
        assert ds.images.nbytes_resident == 0
    elif cache == "uint8":
        assert ds.images.nbytes_resident == 3 * 32 * 32 * 3
    else:
        assert ds.images.nbytes_resident == 3 * 32 * 32 * 3 * 4


@pytest.mark.parametrize("cache", ["uint8", "disk"])
def test_colmap_cache_modes(tmp_path, cache):
    root = str(tmp_path)
    _write_colmap_scene(root)
    ds = load_colmap(root, cache=cache)
    cam, img = ds[0]
    assert img.shape == (16, 16, 3)
    assert (cam.width, cam.height) == (16, 16)
    # Iteration works for training loops.
    assert sum(1 for _ in ds.images) == 2
