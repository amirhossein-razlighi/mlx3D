"""Tests for the Depth Anything 3 port (mlx3d.models.depth_anything3).

Architecture and plumbing are tested on a tiny randomly initialized network so
the suite needs no downloads. One test exercises the real pretrained weights
and runs only when they are already in the local Hugging Face cache.
"""

import os

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx3d.models._resample import resize_bicubic, resize_bilinear
from mlx3d.models.depth_anything3 import (
    DA3Config,
    DepthAnything3,
    GeometryPrediction,
    convert_checkpoint,
    processing_size,
    select_reference_view,
)
from mlx3d.models.depth_anything3.alignment import Similarity, fit_similarity
from mlx3d.models.depth_anything3.backbone import (
    _apply_rope,
    _permute_views,
    _reference_first_order,
    _rope_tables,
)
from mlx3d.models.depth_anything3.camera import decode_pose_encoding

TINY = DA3Config(
    repo_id="",
    license="",
    embed_dim=32,
    depth=4,
    num_heads=2,
    out_layers=(0, 1, 2, 3),
    alt_start=2,
    qknorm_start=2,
    rope_start=2,
    head_features=16,
    head_out_channels=(8, 8, 16, 16),
    pretrain_grid=4,
)


def _random_rotations(n, rng):
    out = []
    for _ in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        out.append(q * np.sign(np.linalg.det(q)))
    return np.stack(out)


# --------------------------------------------------------------- resampling


def test_bilinear_matches_align_corners_semantics():
    ramp = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)  # values 0..3 along W
    out = resize_bilinear(mx.broadcast_to(ramp, (1, 2, 4, 1)), (3, 7))
    # align_corners keeps the endpoints and samples a linear ramp exactly.
    np.testing.assert_allclose(np.array(out[0, 0, :, 0]), np.linspace(0, 3, 7), atol=1e-6)
    assert out.shape == (1, 3, 7, 1)


def test_bicubic_identity_scale_and_constant_preservation():
    x = mx.random.normal((1, 5, 6, 3))
    np.testing.assert_allclose(np.array(resize_bicubic(x, (1.0, 1.0))), np.array(x), atol=1e-5)
    const = mx.full((1, 5, 5, 2), 3.0)
    out = resize_bicubic(const, ((7 + 0.1) / 5, (4 + 0.1) / 5))
    assert out.shape == (1, 7, 4, 2)
    np.testing.assert_allclose(np.array(out), 3.0, atol=1e-5)


# ---------------------------------------------------------------- backbone


def test_rope_is_a_relative_rotation():
    """q·k after RoPE depends only on the positional offset between tokens."""
    rng = np.random.default_rng(0)
    q = mx.array(rng.normal(size=(1, 1, 1, 16)).astype(np.float32))
    k = mx.array(rng.normal(size=(1, 1, 1, 16)).astype(np.float32))

    def score(pq, pk):
        rope = _rope_tables((pq, pk), 16, 100.0)
        rq = _apply_rope(q, rope.cos[:1], rope.sin[:1])
        rk = _apply_rope(k, rope.cos[1:], rope.sin[1:])
        return float((rq * rk).sum())

    assert score((2, 3), (5, 1)) == pytest.approx(score((4, 6), (7, 4)), rel=1e-4)
    rope = _rope_tables(((3, 7),), 16, 100.0)
    rotated = _apply_rope(q, rope.cos, rope.sin)
    assert float(mx.linalg.norm(rotated)) == pytest.approx(float(mx.linalg.norm(q)), rel=1e-5)


def test_reference_first_order_and_restore():
    x = mx.arange(5)[None, :, None].astype(mx.float32)  # (1, S=5, 1)
    order = _reference_first_order(mx.array([3]), 5)
    assert np.array(order).tolist() == [[3, 0, 1, 2, 4]]
    restored = _permute_views(_permute_views(x, order), mx.argsort(order, axis=1))
    np.testing.assert_array_equal(np.array(restored), np.array(x))


def test_select_reference_view_strategies():
    tokens = mx.random.normal((2, 5, 3, 8))
    assert np.array(select_reference_view(tokens, "first")).tolist() == [0, 0]
    assert np.array(select_reference_view(tokens, "middle")).tolist() == [2, 2]
    for strategy in ("saddle_balanced", "saddle_sim_range"):
        idx = np.array(select_reference_view(tokens, strategy))
        assert idx.shape == (2,) and ((0 <= idx) & (idx < 5)).all()
    with pytest.raises(ValueError):
        select_reference_view(tokens, "nope")


# ------------------------------------------------------------------- model


@pytest.fixture(scope="module")
def tiny_model():
    mx.random.seed(0)
    model = DepthAnything3(TINY)
    model.eval()
    return model


def test_forward_shapes(tiny_model):
    pixels = mx.random.normal((1, 3, 28, 42, 3))
    out = tiny_model(pixels, with_rays=True)
    assert out["depth"].shape == (1, 3, 28, 42)
    assert out["depth_conf"].shape == (1, 3, 28, 42)
    assert out["ray"].shape == (1, 3, 16, 24, 6)
    assert out["R"].shape == (1, 3, 3, 3) and out["t"].shape == (1, 3, 3)
    assert out["K"].shape == (1, 3, 3, 3)
    assert float(out["depth"].min()) > 0 and float(out["depth_conf"].min()) >= 1.0


def test_rejects_sizes_not_multiple_of_patch(tiny_model):
    with pytest.raises(ValueError, match="multiple of 14"):
        tiny_model(mx.random.normal((1, 2, 30, 42, 3)))


def test_head_chunking_matches_single_pass(tiny_model):
    pixels = mx.random.normal((1, 5, 28, 28, 3))
    full = tiny_model._forward_views(pixels, "first", False, head_chunk=8)
    chunked = tiny_model._forward_views(pixels, "first", False, head_chunk=2)
    for key in ("depth", "depth_conf", "R", "t"):
        np.testing.assert_allclose(np.array(full[key]), np.array(chunked[key]), atol=1e-5)


def test_predict_single_and_chunked(tiny_model, tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(7):
        path = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (30, 45, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))

    joint = tiny_model.predict(paths, resolution=42)
    chunked = tiny_model.predict(paths, resolution=42, max_views=4, num_anchors=2)
    for pred in (joint, chunked):
        assert len(pred) == 7 and pred.image_size == (42, 28)
        assert pred.original_sizes == [(45, 30)] * 7
        rot = np.array(pred.R)
        np.testing.assert_allclose(
            rot @ rot.transpose(0, 2, 1), np.tile(np.eye(3), (7, 1, 1)), atol=1e-4
        )
    with pytest.raises(ValueError, match="single pass"):
        tiny_model.predict(paths, resolution=42, max_views=4, with_rays=True)


# --------------------------------------------------------------- cameras


def test_decode_pose_encoding_inverts_camera_to_world():
    # 90 degree rotation about z (xyzw), camera center (1, 2, 3), 60/90 degree fov.
    s = np.sqrt(0.5)
    enc = mx.array([[1.0, 2.0, 3.0, 0.0, 0.0, s, s, np.pi / 3, np.pi / 2]])
    R, t, K = decode_pose_encoding(enc, (100, 200))
    center = -(np.array(R[0]).T @ np.array(t[0]))
    np.testing.assert_allclose(center, [1, 2, 3], atol=1e-5)
    np.testing.assert_allclose(np.array(R[0]) @ [1, 0, 0], [0, -1, 0], atol=1e-5)
    np.testing.assert_allclose(float(K[0, 0, 0]), 100.0, rtol=1e-5)  # (W/2) / tan(45°)
    np.testing.assert_allclose(float(K[0, 1, 1]), 50.0 / np.tan(np.pi / 6), rtol=1e-5)
    assert float(K[0, 0, 2]) == 100.0 and float(K[0, 1, 2]) == 50.0


def test_fit_similarity_recovers_known_transform():
    rng = np.random.default_rng(1)
    R_src = _random_rotations(6, rng)
    t_src = rng.normal(size=(6, 3))
    truth = Similarity(2.5, _random_rotations(1, rng)[0], rng.normal(size=3))
    R_dst, t_dst = truth.apply_to_cameras(R_src, t_src)
    fit = fit_similarity(R_src, t_src, R_dst, t_dst)
    assert fit.scale == pytest.approx(2.5, rel=1e-6)
    np.testing.assert_allclose(fit.rotation, truth.rotation, atol=1e-6)
    np.testing.assert_allclose(fit.translation, truth.translation, atol=1e-6)


def _flat_prediction(depth=2.0, size=(6, 4)):
    w, h = size
    return GeometryPrediction(
        depth=mx.full((2, h, w), depth),
        depth_conf=mx.array(np.stack([np.full((h, w), 1.0), np.full((h, w), 5.0)])),
        R=mx.broadcast_to(mx.eye(3), (2, 3, 3)),
        t=mx.zeros((2, 3)),
        K=mx.broadcast_to(mx.array([[3.0, 0, w / 2], [0, 3.0, h / 2], [0, 0, 1]]), (2, 3, 3)),
        rgb=np.full((2, h, w, 3), 255, dtype=np.uint8),
        original_sizes=[(60, 40), (60, 40)],
    )


def test_prediction_cameras_and_points():
    pred = _flat_prediction()
    cam = pred.camera(0, original_size=True)
    assert (cam.width, cam.height) == (60, 40)
    assert cam.fx == pytest.approx(30.0) and cam.cx == pytest.approx(30.0)
    pts = np.array(pred.world_points(0))
    assert pts.shape == (4, 6, 3)
    np.testing.assert_allclose(pts[..., 2], 2.0)
    np.testing.assert_allclose(pts[0, 0, :2], [(0.5 - 3) * 2 / 3, (0.5 - 2) * 2 / 3], atol=1e-6)

    points, colors = pred.point_cloud(conf_quantile=0.5)
    assert len(points) == 24  # only the confident view survives
    np.testing.assert_allclose(colors, 1.0)
    points, _ = pred.point_cloud(conf_quantile=0.0, max_points=10)
    assert len(points) == 10
    assert pred.to_pointclouds(conf_quantile=0.0).num_points_per_cloud.tolist() == [48]


def test_processing_size_rounds_to_patch_multiples():
    assert processing_size(1024, 576, 504) == (504, 280)
    assert processing_size(1297, 840, 504) == (504, 322)
    w, h = processing_size(480, 640, 504)
    assert max(w, h) == 504 and w % 14 == 0 and h % 14 == 0


# ------------------------------------------------------------- conversion


def test_convert_checkpoint_renames_and_transposes():
    weights = {
        "model.backbone.pretrained.blocks.0.attn.qkv.weight": mx.zeros((6, 2)),
        "model.backbone.pretrained.patch_embed.proj.weight": mx.zeros((4, 3, 14, 14)),
        "model.head.resize_layers.0.weight": mx.zeros((5, 7, 4, 4)),  # (in, out, kh, kw)
        "model.head.scratch.layer3_rn.weight": mx.zeros((8, 2, 3, 3)),
        "model.head.scratch.refinenet2_aux.resConfUnit1.conv1.bias": mx.zeros((8,)),
        "model.head.scratch.output_conv2_aux.0.2.weight": mx.ones((32,)),
        "model.head.scratch.output_conv2_aux.1.0.weight": mx.zeros((1, 1, 1, 1)),
        "model.cam_dec.backbone.2.bias": mx.zeros((3,)),
        "model.cam_enc.token_norm.weight": mx.zeros((3,)),
    }
    out = convert_checkpoint(weights)
    assert out["backbone.blocks.0.attn.qkv.weight"].shape == (6, 2)
    assert out["backbone.patch_embed.proj.weight"].shape == (4, 14, 14, 3)
    assert out["head.upsample4.weight"].shape == (7, 4, 4, 5)
    assert out["head.layer_rn.2.weight"].shape == (8, 3, 3, 2)
    assert "head.ray_fusion.refine.1.res_unit1.conv1.bias" in out
    assert "head.ray_norm.weight" in out  # shared aux LayerNorm, stored under level 0
    assert "cam_dec.fc2.bias" in out
    assert len(out) == 7  # cam_enc and the unused aux level were dropped


# ---------------------------------------------------- pretrained (optional)


def _cached_weights(repo_id):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(repo_id, "model.safetensors")
    return path if isinstance(path, str) and os.path.exists(path) else None


@pytest.mark.skipif(
    _cached_weights("depth-anything/DA3-SMALL") is None,
    reason="DA3-SMALL weights not in the local Hugging Face cache",
)
def test_pretrained_small_is_multiview_consistent():
    """Real weights on a synthetic two-plane scene: the predicted depth must be
    consistent across views through the predicted cameras."""
    model = DepthAnything3.from_pretrained(
        "da3-small", weights=_cached_weights("depth-anything/DA3-SMALL")
    )
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (300, 600, 3), dtype=np.uint8)
    texture = np.array(Image.fromarray(texture).resize((1200, 600), Image.Resampling.NEAREST))
    views = [texture[:, x : x + 504] for x in (0, 60, 120, 180)]
    pred = model.predict([v[:280] for v in views])

    assert len(pred) == 4 and pred.image_size == (504, 280)
    depth = np.array(pred.depth)
    assert np.isfinite(depth).all() and depth.min() > 0
    xy, z = pred.camera(1).project_points(pred.world_points(0).reshape(-1, 3))
    assert np.isfinite(np.array(xy)).all() and np.isfinite(np.array(z)).all()
