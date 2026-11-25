"""Camera decoding: per-view camera token -> pinhole intrinsics + world-to-camera pose."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["CameraDecoder", "decode_pose_encoding"]


class CameraDecoder(nn.Module):
    """MLP regressing a 9-D pose encoding from each view's camera token.

    The encoding is ``[t (3), q (4, xyzw), fov_y, fov_x]`` describing the
    *camera-to-world* transform of the view in the reference view's frame.
    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in)
        self.fc2 = nn.Linear(dim_in, dim_in)
        self.fc_t = nn.Linear(dim_in, 3)
        self.fc_qvec = nn.Linear(dim_in, 4)
        self.fc_fov = nn.Linear(dim_in, 2)

    def __call__(self, camera_token: mx.array) -> mx.array:
        h = nn.relu(self.fc2(nn.relu(self.fc1(camera_token.astype(mx.float32)))))
        return mx.concatenate([self.fc_t(h), self.fc_qvec(h), nn.relu(self.fc_fov(h))], axis=-1)


def _quat_xyzw_to_matrix(q: mx.array) -> mx.array:
    """Rotation matrices from (not necessarily unit) xyzw quaternions."""
    x, y, z, w = (q[..., i] for i in range(4))
    s = 2.0 / (q * q).sum(axis=-1)
    rows = [
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ]
    return mx.stack([mx.stack(r, axis=-1) for r in rows], axis=-2)


def decode_pose_encoding(
    encoding: mx.array, image_hw: tuple[int, int]
) -> tuple[mx.array, mx.array, mx.array]:
    """Turn ``(..., 9)`` pose encodings into OpenCV-convention cameras.

    Returns:
        ``(R, t, K)``: world-to-camera rotation ``(..., 3, 3)`` and translation
        ``(..., 3)``, and intrinsics ``(..., 3, 3)`` with the principal point at
        the image center.
    """
    H, W = image_hw
    c2w_R = _quat_xyzw_to_matrix(encoding[..., 3:7])
    c2w_t = encoding[..., :3]
    R = mx.swapaxes(c2w_R, -1, -2)
    t = -(R @ c2w_t[..., None])[..., 0]

    fy = (H / 2.0) / mx.maximum(mx.tan(encoding[..., 7] / 2.0), 1e-6)
    fx = (W / 2.0) / mx.maximum(mx.tan(encoding[..., 8] / 2.0), 1e-6)
    zeros, ones = mx.zeros_like(fx), mx.ones_like(fx)
    K = mx.stack(
        [
            mx.stack([fx, zeros, zeros + W / 2.0], axis=-1),
            mx.stack([zeros, fy, zeros + H / 2.0], axis=-1),
            mx.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )
    return R, t, K
