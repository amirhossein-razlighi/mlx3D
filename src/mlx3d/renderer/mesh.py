"""Differentiable soft mesh rasterization and textured triangle rendering."""

from __future__ import annotations

import mlx.core as mx

from ..cameras import Camera
from ..structures import Meshes

__all__ = ["render_mesh_soft", "sample_texture"]


def _as_verts_faces(mesh_or_verts, faces: mx.array | None) -> tuple[mx.array, mx.array]:
    if isinstance(mesh_or_verts, Meshes):
        if len(mesh_or_verts) != 1:
            raise ValueError("render_mesh_soft currently renders one mesh at a time.")
        return mesh_or_verts.verts_packed(), mesh_or_verts.faces_packed()
    if faces is None:
        raise ValueError("faces must be provided when verts are passed directly.")
    return mesh_or_verts, faces


def sample_texture(texture: mx.array, uv: mx.array) -> mx.array:
    """Bilinearly sample an image texture at UV coordinates.

    Args:
        texture: ``(H, W, 3)`` RGB texture in ``[0, 1]``.
        uv: ``(..., 2)`` UV coordinates. OBJ convention is used: ``v=0`` is
            the bottom of the image.
    """
    tex = texture.astype(mx.float32)
    h, w = tex.shape[:2]
    u = mx.clip(uv[..., 0], 0.0, 1.0) * (w - 1)
    v = (1.0 - mx.clip(uv[..., 1], 0.0, 1.0)) * (h - 1)

    x0 = mx.floor(u).astype(mx.int32)
    y0 = mx.floor(v).astype(mx.int32)
    x1 = mx.minimum(x0 + 1, w - 1)
    y1 = mx.minimum(y0 + 1, h - 1)
    wx = (u - x0.astype(mx.float32))[..., None]
    wy = (v - y0.astype(mx.float32))[..., None]

    flat = tex.reshape(h * w, 3)
    c00 = flat[(y0 * w + x0).reshape(-1)].reshape(*uv.shape[:-1], 3)
    c10 = flat[(y0 * w + x1).reshape(-1)].reshape(*uv.shape[:-1], 3)
    c01 = flat[(y1 * w + x0).reshape(-1)].reshape(*uv.shape[:-1], 3)
    c11 = flat[(y1 * w + x1).reshape(-1)].reshape(*uv.shape[:-1], 3)
    return (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11


def render_mesh_soft(
    camera: Camera,
    mesh_or_verts: Meshes | mx.array,
    faces: mx.array | None = None,
    verts_colors: mx.array | None = None,
    face_colors: mx.array | None = None,
    texcoords: mx.array | None = None,
    faces_texcoords_idx: mx.array | None = None,
    texture: mx.array | None = None,
    sigma: float = 1e-2,
    depth_temperature: float = 25.0,
    background: mx.array | tuple[float, float, float] | float = 0.0,
    eps: float = 1e-8,
    face_chunk_size: int | None = 256,
) -> dict[str, mx.array]:
    """Render a triangle mesh with a SoftRas-style differentiable rasterizer.

    Fully MLX-differentiable w.r.t. vertices, vertex colors, face colors, and
    texture values. Topology and UV indices are treated as discrete.

    The rasterizer processes faces in batches of ``face_chunk_size`` to keep
    memory use bounded: each chunk creates ``(chunk, H, W)`` intermediates so
    even large meshes can be rendered on 8-16 GB machines.  Set
    ``face_chunk_size=None`` to disable chunking (faster for small meshes that
    fit comfortably in memory).

    Args:
        camera: Pinhole camera.
        mesh_or_verts: A :class:`~mlx3d.structures.Meshes` (single mesh) or
            ``(V, 3)`` vertex array.
        faces: ``(F, 3)`` index array, required when ``mesh_or_verts`` is an
            array.
        verts_colors: ``(V, 3)`` per-vertex colors, interpolated in barycentric
            space.
        face_colors: ``(F, 3)`` constant color per face.
        texcoords: ``(VT, 2)`` UV coordinates for textured rendering.
        faces_texcoords_idx: ``(F, 3)`` per-corner indices into ``texcoords``.
        texture: ``(H, W, 3)`` diffuse texture image.
        sigma: Soft boundary width (larger → smoother, less sharp edges).
        depth_temperature: Controls depth-based face ordering; higher values
            make nearer faces dominate more sharply.
        background: Background color — scalar, ``(3,)`` array, or tuple.
        eps: Numerical epsilon for safe divisions.
        face_chunk_size: Process this many faces per chunk; ``None`` = no
            chunking.

    Returns:
        A dict with keys ``"image"`` ``(H, W, 3)``, ``"alpha"`` ``(H, W)``,
        and ``"depth"`` ``(H, W)``.
    """
    verts, faces_idx = _as_verts_faces(mesh_or_verts, faces)
    faces_idx = faces_idx.astype(mx.int32)
    h, w = camera.height, camera.width
    f = faces_idx.shape[0]

    xy, z = camera.project_points(verts)
    tri_xy = xy[faces_idx]  # (F, 3, 2)
    tri_z = z[faces_idx]  # (F, 3)

    # Global max inverse depth from vertex depths — O(F) space, used for
    # numerical stability of the exp-based depth weighting across chunks.
    valid_verts = tri_z > camera.znear
    if bool(valid_verts.any()):
        min_z = mx.min(mx.where(valid_verts, tri_z, mx.full(tri_z.shape, 1e9)))
        max_inv_depth = 1.0 / mx.maximum(min_z, camera.znear)
    else:
        max_inv_depth = mx.array(1.0 / float(camera.znear))

    # Pixel-centre grids: (1, H, W)
    px = mx.arange(w, dtype=mx.float32) + 0.5
    py = mx.arange(h, dtype=mx.float32) + 0.5
    gx = mx.broadcast_to(px[None, :], (h, w))[None]
    gy = mx.broadcast_to(py[:, None], (h, w))[None]

    # Optional per-face texture UV coords — indexed once for the whole batch.
    uv_tri_full: mx.array | None = None
    vc_full: mx.array | None = None
    fc_full: mx.array | None = None
    if texture is not None:
        if texcoords is None or faces_texcoords_idx is None:
            raise ValueError("texcoords and faces_texcoords_idx are required with texture.")
        uv_tri_full = texcoords[faces_texcoords_idx.astype(mx.int32)]  # (F, 3, 2)
    elif verts_colors is not None:
        vc_full = verts_colors[faces_idx]  # (F, 3, C)
    else:
        fc_full = face_colors if face_colors is not None else mx.ones((f, 3), dtype=mx.float32)

    # Accumulators for incremental sum over chunks.
    sum_w: mx.array = mx.zeros((h, w), dtype=mx.float32)
    sum_wc: mx.array = mx.zeros((h, w, 3), dtype=mx.float32)
    sum_wz: mx.array = mx.zeros((h, w), dtype=mx.float32)

    chunk = f if face_chunk_size is None else max(1, int(face_chunk_size))
    for start in range(0, f, chunk):
        end = min(start + chunk, f)
        sl = slice(start, end)

        txy = tri_xy[sl]  # (C, 3, 2)
        tz = tri_z[sl]  # (C, 3)
        C = end - start

        x0c = txy[:, 0, 0][:, None, None]
        y0c = txy[:, 0, 1][:, None, None]
        x1c = txy[:, 1, 0][:, None, None]
        y1c = txy[:, 1, 1][:, None, None]
        x2c = txy[:, 2, 0][:, None, None]
        y2c = txy[:, 2, 1][:, None, None]

        denom = (y1c - y2c) * (x0c - x2c) + (x2c - x1c) * (y0c - y2c)
        valid_area = mx.abs(denom) > eps
        denom = mx.where(valid_area, denom, mx.ones_like(denom))

        l0 = ((y1c - y2c) * (gx - x2c) + (x2c - x1c) * (gy - y2c)) / denom
        l1 = ((y2c - y0c) * (gx - x2c) + (x0c - x2c) * (gy - y2c)) / denom
        l2 = 1.0 - l0 - l1

        signed_inside = mx.minimum(mx.minimum(l0, l1), l2)
        coverage = mx.sigmoid(signed_inside / max(float(sigma), 1e-6))

        z_face = (
            l0 * tz[:, 0][:, None, None]
            + l1 * tz[:, 1][:, None, None]
            + l2 * tz[:, 2][:, None, None]
        )
        in_front = mx.all(tz > camera.znear, axis=-1)[:, None, None]
        coverage = coverage * valid_area * in_front

        # Barycentric coords go outside [0, 1] for pixels off the triangle, so
        # ``z_face`` there can fall below ``znear`` and make ``inv_depth`` blow
        # up. Clamp the exponent at 0 (the globally nearest face gets weight 1):
        # this both matches the soft z-buffer semantics and avoids the
        # ``coverage(=0) * exp(=inf) = NaN`` that would otherwise appear where a
        # distant face projects far from the pixel.
        inv_depth = 1.0 / mx.maximum(z_face, camera.znear)
        depth_w = mx.exp(mx.minimum(depth_temperature * (inv_depth - max_inv_depth), 0.0))
        weights = coverage * depth_w  # (C, H, W)

        if texture is not None:
            uv_tri_c = uv_tri_full[sl]  # type: ignore[index]
            uv = (
                l0[..., None] * uv_tri_c[:, 0, :][:, None, None, :]
                + l1[..., None] * uv_tri_c[:, 1, :][:, None, None, :]
                + l2[..., None] * uv_tri_c[:, 2, :][:, None, None, :]
            )
            colors = sample_texture(texture, uv)
        elif vc_full is not None:
            vc = vc_full[sl]
            colors = (
                l0[..., None] * vc[:, 0, :][:, None, None, :]
                + l1[..., None] * vc[:, 1, :][:, None, None, :]
                + l2[..., None] * vc[:, 2, :][:, None, None, :]
            )
        else:
            colors = mx.broadcast_to(fc_full[sl][:, None, None, :], (C, h, w, 3))  # type: ignore[index]

        sum_w = sum_w + mx.sum(weights, axis=0)
        sum_wc = sum_wc + mx.sum(weights[..., None] * colors, axis=0)
        sum_wz = sum_wz + mx.sum(weights * z_face, axis=0)

    image = sum_wc / mx.maximum(sum_w[..., None], eps)
    alpha = 1.0 - mx.exp(-sum_w)
    bg = mx.array(background, dtype=mx.float32)
    if bg.ndim == 0:
        bg = mx.broadcast_to(bg, (3,))
    image = image * alpha[..., None] + bg * (1.0 - alpha[..., None])
    depth = sum_wz / mx.maximum(sum_w, eps)

    return {"image": image, "alpha": alpha, "depth": depth}
