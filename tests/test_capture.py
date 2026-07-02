"""Tests for the capture pipeline (photos/video -> poses -> 3DGS)."""

import os

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx3d.cameras import Camera
from mlx3d.datasets import load_colmap, save_colmap


def _random_pose(rng):
    # Random rotation via QR, positive determinant.
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q, rng.normal(size=(3,))


def _make_cameras(n=4, w=32, h=24, distortion=None, fisheye=False, seed=0):
    rng = np.random.default_rng(seed)
    cams = []
    for _ in range(n):
        R, t = _random_pose(rng)
        cams.append(
            Camera(
                R=mx.array(R.astype(np.float32)),
                t=mx.array(t.astype(np.float32)),
                fx=40.0,
                fy=42.0,
                cx=w / 2,
                cy=h / 2,
                width=w,
                height=h,
                distortion=distortion,
                fisheye=fisheye,
            )
        )
    return cams


def test_save_colmap_roundtrip(tmp_path):
    root = str(tmp_path)
    w, h = 32, 24
    cams = _make_cameras(4, w, h)
    names = [f"img_{i:03d}.png" for i in range(len(cams))]
    os.makedirs(os.path.join(root, "images"))
    for name in names:
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(
            os.path.join(root, "images", name)
        )
    rng = np.random.default_rng(1)
    points = rng.normal(size=(50, 3)).astype(np.float32)
    colors = rng.uniform(size=(50, 3)).astype(np.float32)

    save_colmap(root, cams, names, points, colors)
    ds = load_colmap(root)

    assert len(ds) == len(cams)
    assert ds.image_names == names
    np.testing.assert_allclose(np.array(ds.points), points, atol=1e-5)
    np.testing.assert_allclose(np.array(ds.point_colors), colors, atol=2 / 255)
    for cam, ref in zip(ds.cameras, cams):
        np.testing.assert_allclose(np.array(cam.R), np.array(ref.R), atol=1e-6)
        np.testing.assert_allclose(np.array(cam.t), np.array(ref.t), atol=1e-6)
        assert (cam.fx, cam.fy, cam.cx, cam.cy) == pytest.approx((ref.fx, ref.fy, ref.cx, ref.cy))
        assert (cam.width, cam.height) == (ref.width, ref.height)
        assert cam.distortion is None


@pytest.mark.parametrize(
    "distortion,fisheye",
    [((0.1, -0.05, 0.001, 0.002), False), ((0.1, -0.05, 0.01, 0.002), True)],
)
def test_save_colmap_roundtrip_distortion(tmp_path, distortion, fisheye):
    root = str(tmp_path)
    cams = _make_cameras(2, distortion=distortion, fisheye=fisheye)
    names = [f"v{i}.png" for i in range(len(cams))]
    save_colmap(root, cams, names, np.zeros((1, 3)), np.zeros((1, 3)))
    ds = load_colmap(root, load_images=False)
    for cam in ds.cameras:
        assert cam.fisheye == fisheye
        np.testing.assert_allclose(cam.distortion, distortion, atol=1e-9)


def test_save_colmap_dedupes_shared_intrinsics(tmp_path):
    import struct

    root = str(tmp_path)
    cams = _make_cameras(5)
    names = [f"v{i}.png" for i in range(len(cams))]
    save_colmap(root, cams, names, np.zeros((1, 3)), np.zeros((1, 3)))
    with open(os.path.join(root, "sparse", "0", "cameras.bin"), "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
    assert num == 1  # identical intrinsics collapse to one COLMAP camera


# --------------------------------------------------------------------- frames
def _save_noise_image(path, size=64, blur=0, seed=0):
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    img.save(path)


def test_sharpness_score_ranks_blur():
    from mlx3d.capture import sharpness_score

    rng = np.random.default_rng(0)
    sharp = rng.uniform(0, 255, size=(64, 64)).astype(np.float32)
    blurred = (sharp[::2, ::2] + sharp[1::2, ::2] + sharp[::2, 1::2] + sharp[1::2, 1::2]) / 4
    assert sharpness_score(sharp) > sharpness_score(blurred)


def test_select_sharpest_keeps_temporal_coverage(tmp_path):
    from mlx3d.capture import select_sharpest

    paths = []
    for i in range(12):
        p = str(tmp_path / f"f_{i:03d}.png")
        # One sharp frame per bucket of 3; the rest heavily blurred.
        _save_noise_image(p, blur=0 if i % 3 == 1 else 4, seed=i)
        paths.append(p)
    keep = select_sharpest(paths, 4)
    assert len(keep) == 4
    assert keep == sorted(keep)  # temporal order preserved
    assert [os.path.basename(p) for p in keep] == [
        "f_001.png",
        "f_004.png",
        "f_007.png",
        "f_010.png",
    ]
    # Fewer inputs than target: everything is kept.
    assert select_sharpest(paths[:3], 8) == paths[:3]


def test_estimate_focal_px(tmp_path):
    from PIL import Image

    from mlx3d.capture import estimate_focal_px

    p = str(tmp_path / "img.jpg")
    Image.new("RGB", (200, 100)).save(p)
    f, source = estimate_focal_px(p)
    assert source == "prior" and f == pytest.approx(1.2 * 200)

    # With a 35mm-equivalent EXIF focal length, use it.
    import PIL.Image

    img = PIL.Image.new("RGB", (200, 100))
    exif = img.getexif()
    from PIL import ExifTags

    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.FocalLengthIn35mmFilm] = 28
    p2 = str(tmp_path / "img_exif.jpg")
    img.save(p2, exif=exif)
    f2, source2 = estimate_focal_px(p2)
    assert source2 == "exif" and f2 == pytest.approx(28 / 36 * 200)


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="ffmpeg not installed")
def test_extract_video_frames(tmp_path):
    import subprocess

    from mlx3d.capture import extract_video_frames

    video = str(tmp_path / "clip.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=160x120:rate=24",
            video,
        ],
        check=True,
    )
    frames = extract_video_frames(
        video, str(tmp_path / "frames"), num_frames=8, log=lambda *_: None
    )
    assert len(frames) == 8
    assert all(os.path.exists(p) for p in frames)
    # Rejected frames were deleted from the output directory.
    from mlx3d.capture import list_images

    assert list_images(str(tmp_path / "frames")) == frames
