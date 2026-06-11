"""Pinhole cameras.

MLX3D uses the OpenCV / COLMAP camera convention:

- Camera frame: ``+x`` right, ``+y`` down, ``+z`` forward (into the screen).
- Extrinsics map world to camera: ``X_cam = R @ X_world + t``.
- Intrinsics are in pixels: ``u = fx * x/z + cx``, ``v = fy * y/z + cy``.

This matches the convention used by COLMAP, most NeRF codebases and the
Gaussian Splatting reference implementation, so real-world datasets load
without axis gymnastics.
"""

import math
from dataclasses import dataclass

import mlx.core as mx

__all__ = ["Camera", "look_at", "look_at_view_transform", "fov_to_focal", "focal_to_fov"]


def fov_to_focal(fov: float, pixels: int) -> float:
    """Focal length in pixels from a field of view in radians."""
    return pixels / (2.0 * math.tan(fov / 2.0))


def focal_to_fov(focal: float, pixels: int) -> float:
    """Field of view in radians from a focal length in pixels."""
    return 2.0 * math.atan(pixels / (2.0 * focal))


def look_at(eye: mx.array, at: mx.array, up: mx.array) -> tuple[mx.array, mx.array]:
    """Build OpenCV-convention extrinsics ``(R, t)`` for a camera at ``eye`` looking at ``at``.

    Args:
        eye: (3,) camera position in world coordinates.
        at: (3,) target point in world coordinates.
        up: (3,) approximate world up vector.

    Returns:
        ``R`` (3, 3) and ``t`` (3,) such that ``X_cam = R @ X_world + t``.
    """
    eye, at, up = mx.array(eye), mx.array(at), mx.array(up)
    z = at - eye
    z = z / mx.linalg.norm(z)
    x = mx.linalg.cross(z, up)
    x = x / mx.linalg.norm(x)
    y = mx.linalg.cross(z, x)
    R = mx.stack([x, y, z], axis=0)
    t = -(R @ eye)
    return R, t


def look_at_view_transform(
    dist: float = 1.0,
    elev: float = 0.0,
    azim: float = 0.0,
    at: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    degrees: bool = True,
) -> tuple[mx.array, mx.array]:
    """Extrinsics for a camera on a sphere around ``at``.

    ``elev`` is the angle above the xz-plane, ``azim`` the angle around ``+y``
    measured from ``+z``. Returns ``(R, t)`` in the OpenCV convention.
    """
    if degrees:
        elev = math.radians(elev)
        azim = math.radians(azim)
    x = dist * math.cos(elev) * math.sin(azim)
    y = dist * math.sin(elev)
    z = dist * math.cos(elev) * math.cos(azim)
    eye = mx.array([at[0] + x, at[1] + y, at[2] + z])
    return look_at(eye, mx.array(at), mx.array(up))


@dataclass
class Camera:
    """A single pinhole camera.

    Attributes:
        R: (3, 3) world-to-camera rotation.
        t: (3,) world-to-camera translation.
        fx, fy: focal lengths in pixels.
        cx, cy: principal point in pixels.
        width, height: image size in pixels.
        znear, zfar: clipping range used by renderers.
    """

    R: mx.array
    t: mx.array
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    znear: float = 0.01
    zfar: float = 100.0

    @classmethod
    def from_fov(
        cls,
        fov: float,
        width: int,
        height: int,
        R: mx.array | None = None,
        t: mx.array | None = None,
        degrees: bool = True,
        **kwargs,
    ) -> "Camera":
        """Create a camera from a vertical field of view (the horizontal FoV
        follows from the aspect ratio)."""
        if degrees:
            fov = math.radians(fov)
        f = fov_to_focal(fov, height)
        if R is None:
            R = mx.eye(3)
        if t is None:
            t = mx.zeros((3,))
        return cls(
            R=R, t=t, fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
            width=width, height=height, **kwargs,
        )

    @classmethod
    def look_at(
        cls,
        eye,
        at=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov: float = 60.0,
        width: int = 512,
        height: int = 512,
        degrees: bool = True,
        **kwargs,
    ) -> "Camera":
        """Create a camera at ``eye`` looking at ``at``."""
        R, t = look_at(mx.array(eye), mx.array(at), mx.array(up))
        return cls.from_fov(fov, width, height, R=R, t=t, degrees=degrees, **kwargs)

    @property
    def K(self) -> mx.array:
        """(3, 3) intrinsic matrix."""
        return mx.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def fov_x(self) -> float:
        return focal_to_fov(self.fx, self.width)

    @property
    def fov_y(self) -> float:
        return focal_to_fov(self.fy, self.height)

    @property
    def camera_center(self) -> mx.array:
        """(3,) camera position in world coordinates."""
        return -(self.R.T @ self.t)

    @property
    def world_to_camera_matrix(self) -> mx.array:
        """(4, 4) homogeneous world-to-camera matrix."""
        top = mx.concatenate([self.R, self.t[:, None]], axis=1)
        bottom = mx.array([[0.0, 0.0, 0.0, 1.0]])
        return mx.concatenate([top, bottom], axis=0)

    def world_to_camera(self, points: mx.array) -> mx.array:
        """Transform world points ``(..., 3)`` into the camera frame."""
        return points @ self.R.T + self.t

    def camera_to_world(self, points: mx.array) -> mx.array:
        """Transform camera-frame points ``(..., 3)`` back to world coordinates."""
        return (points - self.t) @ self.R

    def project_points(self, points: mx.array, eps: float = 1e-8) -> tuple[mx.array, mx.array]:
        """Project world points ``(..., 3)`` to pixel coordinates.

        Returns:
            ``(xy, depth)`` where ``xy`` is ``(..., 2)`` pixel coordinates and
            ``depth`` is ``(...,)`` z-depth in the camera frame. Points behind
            the camera have negative depth; callers should mask on it.
        """
        pc = self.world_to_camera(points)
        z = pc[..., 2]
        inv_z = 1.0 / mx.where(mx.abs(z) < eps, mx.full(z.shape, eps), z)
        u = self.fx * pc[..., 0] * inv_z + self.cx
        v = self.fy * pc[..., 1] * inv_z + self.cy
        return mx.stack([u, v], axis=-1), z

    def unproject_points(self, xy: mx.array, depth: mx.array) -> mx.array:
        """Lift pixel coordinates ``(..., 2)`` with z-depths ``(...,)`` back to world points."""
        x = (xy[..., 0] - self.cx) / self.fx * depth
        y = (xy[..., 1] - self.cy) / self.fy * depth
        return self.camera_to_world(mx.stack([x, y, depth], axis=-1))

    def generate_rays(self) -> tuple[mx.array, mx.array]:
        """Generate one ray per pixel (at pixel centers).

        Returns:
            ``(origins, directions)``, both ``(height, width, 3)`` in world
            coordinates. Directions are normalized.
        """
        u = mx.arange(self.width, dtype=mx.float32) + 0.5
        v = mx.arange(self.height, dtype=mx.float32) + 0.5
        uu = mx.broadcast_to(u[None, :], (self.height, self.width))
        vv = mx.broadcast_to(v[:, None], (self.height, self.width))
        dirs_cam = mx.stack(
            [(uu - self.cx) / self.fx, (vv - self.cy) / self.fy, mx.ones_like(uu)],
            axis=-1,
        )
        dirs_world = dirs_cam @ self.R  # == dirs_cam @ R^-T == R^T applied per-vector
        dirs_world = dirs_world / mx.linalg.norm(dirs_world, axis=-1, keepdims=True)
        origins = mx.broadcast_to(self.camera_center, dirs_world.shape)
        return origins, dirs_world
