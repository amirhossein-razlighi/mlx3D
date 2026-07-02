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


# ------------------------------------------------------------------ built-in SfM
def _render_corner_scene(img_dir, views=12, size=320, seed=0):
    """Render a textured 'room corner' with known poses; returns {name: Camera}."""
    from mlx3d.splatting import GaussianModel

    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    planes, colors = [], []
    for k in range(3):
        uv = rng.uniform(size=(8000, 2))
        col = np.zeros((8000, 3))
        for octave in (2, 5, 11, 23, 47):
            for c in range(3):
                col[:, c] += octave**-0.5 * np.sin(
                    2 * np.pi * octave * (uv @ rng.uniform(0.5, 1.5, 2))
                    + rng.uniform(0, 2 * np.pi)
                )
        col = (col - col.min()) / (np.ptp(col) + 1e-9)
        a, b = uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1
        if k == 0:
            xyz = np.stack([a, np.ones_like(a), b], axis=1)  # floor
        elif k == 1:
            xyz = np.stack([a, b, -np.ones_like(a)], axis=1)  # back wall
        else:
            xyz = np.stack([-np.ones_like(a), a, b], axis=1)  # side wall
        planes.append(xyz)
        colors.append(col)
    model = GaussianModel.from_points(
        mx.array(np.concatenate(planes).astype(np.float32)),
        mx.array(np.concatenate(colors).astype(np.float32)),
        sh_degree=0,
    )
    model.params["scales"] = mx.full(model.params["scales"].shape, np.log(0.015))
    model.params["opacities"] = mx.full(model.params["opacities"].shape, 6.0)

    from PIL import Image

    os.makedirs(img_dir, exist_ok=True)
    cams = {}
    for k in range(views):
        az = 45.0 + 90.0 * k / (views - 1)
        elev = 15.0 + 8.0 * np.sin(3.0 * np.pi * k / views)
        eye = (
            3.2 * np.cos(np.radians(elev)) * np.cos(np.radians(az)),
            -3.2 * np.sin(np.radians(elev)),
            3.2 * np.cos(np.radians(elev)) * np.sin(np.radians(az)),
        )
        cam = Camera.look_at(eye=eye, at=(-0.2, 0.2, -0.2), fov=55.0, width=size, height=size)
        name = f"v{k:03d}.png"
        arr = (np.clip(np.array(model.render(cam)["image"]), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(img_dir, name))
        cams[name] = cam
    return cams


def _similarity_align(est: np.ndarray, gt: np.ndarray):
    """Procrustes: returns (s, R, t) with gt ~ s * est @ R + t (row vectors)."""
    mu_e, mu_g = est.mean(0), gt.mean(0)
    E, G = est - mu_e, gt - mu_g
    U, S, Vt = np.linalg.svd(E.T @ G)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (E**2).sum()
    return s, R, mu_g - s * mu_e @ R


def test_builtin_sfm_recovers_synthetic_poses(tmp_path):
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from mlx3d.capture import SfmConfig, list_images, run_sfm

    img_dir = str(tmp_path / "input")
    gt = _render_corner_scene(img_dir, views=12, size=320)
    result = run_sfm(
        list_images(img_dir), str(tmp_path), config=SfmConfig(seed=0), log=lambda *_: None
    )
    assert len(result.registered) >= 10  # nearly all views register
    assert result.num_points > 200
    assert result.mean_reproj_px < 2.0

    ds = load_colmap(str(tmp_path), images_dir=img_dir, load_images=False)
    est = np.stack([np.array(c.camera_center) for c in ds.cameras])
    ref = np.stack([np.array(gt[n].camera_center) for n in ds.image_names])
    s, R_align, t = _similarity_align(est, ref)
    center_err = np.linalg.norm(s * est @ R_align + t - ref, axis=1)
    assert center_err.mean() < 0.05 * 3.2  # < 5% of the camera-orbit radius

    rot_errs = []
    for cam, name in zip(ds.cameras, ds.image_names):
        R_delta = np.array(gt[name].R) @ (np.array(cam.R) @ R_align).T
        rot_errs.append(
            np.degrees(np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1)))
        )
    assert np.mean(rot_errs) < 3.0  # degrees

    # Focal refinement pulls the 1.2*dim prior toward the true focal.
    true_f = gt[ds.image_names[0]].fx
    assert abs(ds.cameras[0].fx - true_f) / true_f < 0.05
