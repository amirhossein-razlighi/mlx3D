"""Depth Anything 3: feed-forward multi-view depth and camera estimation."""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .._hub import resolve_weights
from .alignment import Similarity, fit_similarity
from .backbone import DinoBackbone, LayerOutput, RefViewStrategy
from .camera import CameraDecoder, decode_pose_encoding
from .config import PRESETS, DA3Config
from .convert import convert_checkpoint
from .head import DualDPT
from .prediction import GeometryPrediction
from .preprocess import ImageInput, load_views

__all__ = ["DepthAnything3"]


class DepthAnything3(nn.Module):
    """DA3 any-view geometry model (MLX port of ByteDance-Seed/Depth-Anything-3).

    From one or more images of a static scene it predicts, in a single forward
    pass, a depth map and confidence per view plus every view's intrinsics and
    pose — the job usually done by structure-from-motion. Load pretrained
    weights with :meth:`from_pretrained` and run :meth:`predict`.

    The backbone may run in reduced precision (``dtype``); the dense head and
    camera decoder always run in float32, as in the reference implementation.
    """

    def __init__(self, config: DA3Config, dtype: mx.Dtype = mx.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.backbone = DinoBackbone(config)
        self.head = DualDPT(
            dim_in=2 * config.embed_dim,
            features=config.head_features,
            out_channels=config.head_out_channels,
            patch_size=config.patch_size,
        )
        self.cam_dec = CameraDecoder(2 * config.embed_dim)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        name: str = "da3-base",
        weights: str | None = None,
        dtype: mx.Dtype = mx.float32,
    ) -> DepthAnything3:
        """Build a preset and load its pretrained weights.

        Args:
            name: a preset in :data:`~mlx3d.models.depth_anything3.PRESETS`
                (``"da3-small"``, ``"da3-base"``: Apache-2.0; ``"da3-large"``:
                CC BY-NC 4.0, non-commercial).
            weights: local ``model.safetensors`` (or its directory). Defaults
                to downloading the preset's Hub repository once.
            dtype: backbone compute dtype. ``mx.float16`` tracks float32
                closely (~0.03% median depth difference); ``mx.bfloat16``'s
                shorter mantissa costs visibly more accuracy.
        """
        if name not in PRESETS:
            raise ValueError(f"Unknown DA3 preset {name!r}; choose from {sorted(PRESETS)}.")
        config = PRESETS[name]
        model = cls(config, dtype=dtype)
        path = resolve_weights(weights or config.repo_id, repo_id=config.repo_id)
        model.load_weights(list(convert_checkpoint(mx.load(str(path))).items()), strict=True)
        if dtype != mx.float32:
            model.backbone.set_dtype(dtype)
        model.eval()
        mx.eval(model.parameters())
        return model

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def encode(
        self, pixels: mx.array, ref_view_strategy: RefViewStrategy = "saddle_balanced"
    ) -> list[LayerOutput]:
        """Backbone taps for normalized ``(B, S, H, W, 3)`` pixels, in float32."""
        taps = self.backbone(pixels.astype(self.dtype), ref_view_strategy)
        return [
            LayerOutput(t.patch_tokens.astype(mx.float32), t.camera_token.astype(mx.float32))
            for t in taps
        ]

    def __call__(
        self,
        pixels: mx.array,
        ref_view_strategy: RefViewStrategy = "saddle_balanced",
        with_rays: bool = False,
    ) -> dict[str, mx.array]:
        """Full forward pass on normalized ``(B, S, H, W, 3)`` pixels.

        Returns ``(B, S, ...)`` arrays: ``depth``, ``depth_conf``, ``R``, ``t``,
        ``K`` (and ``ray``, ``ray_conf`` with ``with_rays``).
        """
        image_hw = (pixels.shape[2], pixels.shape[3])
        taps = self.encode(pixels, ref_view_strategy)
        out = self.head.decode(taps, image_hw, with_rays)
        out["R"], out["t"], out["K"] = decode_pose_encoding(
            self.cam_dec(taps[-1].camera_token), image_hw
        )
        return out

    def predict(
        self,
        images: Sequence[ImageInput],
        resolution: int = 504,
        ref_view_strategy: RefViewStrategy = "saddle_balanced",
        with_rays: bool = False,
        max_views: int = 32,
        num_anchors: int = 8,
        head_chunk: int = 8,
    ) -> GeometryPrediction:
        """Predict depth and cameras for a set of views of one scene.

        Up to ``max_views`` views are processed jointly in one forward pass.
        Longer sequences run in chunks that all share ``num_anchors`` evenly
        spaced anchor views; each chunk is mapped into the first chunk's frame
        by the similarity that best aligns its anchor cameras, so cost grows
        linearly instead of quadratically and nothing drifts along the sequence.

        Args:
            images: file paths, PIL images, or ``(H, W, 3)`` arrays — all with
                the same aspect ratio.
            resolution: longest processing side (the model was trained at 504).
            ref_view_strategy: how the view defining the world frame is chosen.
            with_rays: also return DA3's per-pixel ray maps (joint pass only).
            max_views: most views per forward pass.
            num_anchors: views shared by every chunk of a long sequence.
            head_chunk: views decoded per dense-head pass; lower it to reduce
                peak memory.
        """
        pixels, rgb, sizes = load_views(images, resolution, self.config.patch_size)
        n_views = pixels.shape[1]
        if n_views <= max_views:
            out = self._forward_views(pixels, ref_view_strategy, with_rays, head_chunk)
        elif with_rays:
            raise ValueError(f"with_rays needs a single pass: pass max_views >= {n_views}.")
        else:
            out = self._forward_chunked(
                pixels, ref_view_strategy, max_views, num_anchors, head_chunk
            )
        return GeometryPrediction(rgb=rgb, original_sizes=sizes, **out)

    def _forward_views(
        self,
        pixels: mx.array,
        ref_view_strategy: RefViewStrategy,
        with_rays: bool,
        head_chunk: int,
    ) -> dict[str, mx.array]:
        """One joint pass over ``(1, S, H, W, 3)`` pixels, decoding the dense head
        a few views at a time to bound memory. Returns ``(S, ...)`` arrays.

        The attention and decoder buffers differ in size between passes, so
        MLX's buffer cache is released after each stage; otherwise it keeps
        growing across calls and, on unified memory, slows the whole system.
        """
        image_hw = (pixels.shape[2], pixels.shape[3])
        taps = self.encode(pixels, ref_view_strategy)
        R, t, K = decode_pose_encoding(self.cam_dec(taps[-1].camera_token), image_hw)
        mx.eval([tap.arrays() for tap in taps], R, t, K)
        mx.clear_cache()

        maps: dict[str, list[mx.array]] = {}
        for start in range(0, pixels.shape[1], head_chunk):
            chunk = [
                LayerOutput(tap.patch_tokens[:, start : start + head_chunk], tap.camera_token)
                for tap in taps
            ]
            out = self.head.decode(chunk, image_hw, with_rays)
            mx.eval(out)
            for key, value in out.items():
                maps.setdefault(key, []).append(value[0])
        mx.clear_cache()
        result = {key: mx.concatenate(parts, axis=0) for key, parts in maps.items()}
        result.update(R=R[0], t=t[0], K=K[0])
        return result

    def _forward_chunked(
        self,
        pixels: mx.array,
        ref_view_strategy: RefViewStrategy,
        max_views: int,
        num_anchors: int,
        head_chunk: int,
    ) -> dict[str, mx.array]:
        n_views = pixels.shape[1]
        if not 0 < num_anchors < max_views:
            raise ValueError("Need 0 < num_anchors < max_views.")
        anchors = np.unique(np.linspace(0, n_views - 1, num_anchors).round().astype(int))
        others = np.setdiff1d(np.arange(n_views), anchors)
        step = max_views - len(anchors)

        depth = [None] * n_views
        conf = [None] * n_views
        R = np.zeros((n_views, 3, 3))
        t = np.zeros((n_views, 3))
        K = np.zeros((n_views, 3, 3))
        reference = None  # anchor cameras of the first chunk define the world frame
        for start in range(0, len(others), step):
            views = np.concatenate([anchors, others[start : start + step]])
            out = self._forward_views(
                pixels[:, mx.array(views)], ref_view_strategy, False, head_chunk
            )
            Rc, tc = np.array(out["R"], np.float64), np.array(out["t"], np.float64)
            n_a = len(anchors)
            if reference is None:
                reference = (Rc[:n_a], tc[:n_a])
                sim = Similarity(1.0, np.eye(3), np.zeros(3))
            else:
                sim = fit_similarity(Rc[:n_a], tc[:n_a], *reference)
            Rc, tc = sim.apply_to_cameras(Rc, tc)
            # Anchors keep the first chunk's estimate; each other view is seen once.
            keep = slice(0, None) if start == 0 else slice(n_a, None)
            for local, view in zip(range(len(views))[keep], views[keep]):
                depth[view] = out["depth"][local] * sim.scale
                conf[view] = out["depth_conf"][local]
                R[view], t[view], K[view] = Rc[local], tc[local], np.array(out["K"][local])
        return {
            "depth": mx.stack(depth),
            "depth_conf": mx.stack(conf),
            "R": mx.array(R, dtype=mx.float32),
            "t": mx.array(t, dtype=mx.float32),
            "K": mx.array(K, dtype=mx.float32),
        }
