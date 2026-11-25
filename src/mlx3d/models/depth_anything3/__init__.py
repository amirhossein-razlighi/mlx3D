"""Depth Anything 3: feed-forward multi-view depth and camera estimation.

An MLX port of `Depth Anything 3 <https://github.com/ByteDance-Seed/Depth-Anything-3>`_
(Apache-2.0). From any number of images of a static scene, one forward pass
predicts per-view depth, confidence, intrinsics and poses.
"""

from .backbone import RefViewStrategy, select_reference_view
from .camera import decode_pose_encoding
from .config import PRESETS, DA3Config
from .convert import convert_checkpoint
from .model import DepthAnything3
from .prediction import GeometryPrediction
from .preprocess import load_views, processing_size

__all__ = [
    "DA3Config",
    "DepthAnything3",
    "GeometryPrediction",
    "PRESETS",
    "RefViewStrategy",
    "convert_checkpoint",
    "decode_pose_encoding",
    "load_views",
    "processing_size",
    "select_reference_view",
]
