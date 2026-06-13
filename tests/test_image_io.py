"""Unit tests for image loading and saving."""

import mlx.core as mx
import numpy as np

from mlx3d.io import load_image, save_image


def test_save_load_roundtrip_rgb(tmp_path):
    img = mx.random.uniform(shape=(12, 16, 3))
    path = tmp_path / "nested" / "img.png"  # parent dirs auto-created
    save_image(str(path), img)
    assert path.exists()

    back = load_image(str(path))
    assert back.shape == (12, 16, 3)
    assert back.dtype == mx.float32
    # PNG is lossless; values match to within 8-bit quantization.
    np.testing.assert_allclose(np.array(back), np.array(img), atol=1.0 / 255 + 1e-4)


def test_save_clips_out_of_range(tmp_path):
    img = mx.array([[[-1.0, 0.5, 2.0]]])  # below 0 and above 1
    path = tmp_path / "clip.png"
    save_image(str(path), img)
    back = load_image(str(path))
    np.testing.assert_allclose(np.array(back[0, 0]), [0.0, 0.5, 1.0], atol=1.0 / 255 + 1e-4)


def test_load_without_normalize_returns_uint8(tmp_path):
    img = mx.zeros((4, 4, 3)) + 0.5
    path = tmp_path / "u8.png"
    save_image(str(path), img)
    raw = load_image(str(path), normalize=False)
    assert raw.dtype == mx.uint8
    assert int(raw.max()) <= 255 and int(raw.min()) >= 0


def test_save_grayscale(tmp_path):
    img = mx.broadcast_to(mx.linspace(0.0, 1.0, 8)[None, :], (8, 8))
    path = tmp_path / "gray.png"
    save_image(str(path), img)
    back = load_image(str(path))
    # Single-channel images round-trip as (H, W) per load_image's contract.
    assert back.shape == (8, 8)
    np.testing.assert_allclose(np.array(back), np.array(img), atol=1.0 / 255 + 1e-4)
