"""Spaghetti Splatting: radiance fields of Gaussian-blurred curve segments.

An MLX / Metal implementation of *Spaghetti Splatting: Differentiating
Curvilinear Dirac Masses via Convolution* (Shramko, Govindarajan, Lam, Xie,
Rebain, Lindell, Tagliasacchi, 2026). Each primitive is a line segment
convolved with a 3D Gaussian kernel; chains of segments represent curves —
hair, wires, grass — far more economically than strings of blobs, and a
zero-length segment is exactly a 3D Gaussian, so the representation is a
strict superset of 3D Gaussian Splatting.
"""

from .model import SpaghettiModel
from .projection import SegmentProjection, project_segments, segment_optical_depth
from .rasterize import pack_segments, rasterize_segments
from .render import render_spaghetti, render_spaghetti_reference, segment_colors
from .trainer import SpaghettiTrainer, SpaghettiTrainerConfig, default_config_for

__all__ = [
    "SegmentProjection",
    "SpaghettiModel",
    "SpaghettiTrainer",
    "SpaghettiTrainerConfig",
    "default_config_for",
    "pack_segments",
    "project_segments",
    "rasterize_segments",
    "render_spaghetti",
    "render_spaghetti_reference",
    "segment_colors",
    "segment_optical_depth",
]
