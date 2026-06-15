"""Hard z-buffer mesh rasterizer.

A Metal kernel resolves visibility (the nearest triangle per pixel), which is an
inherently discrete operation; all attribute interpolation and shading then
happen in differentiable MLX from the recomputed barycentric coordinates. So
there is no backward kernel to maintain, yet gradients still flow to vertex
positions (through the barycentrics), colors, lights and textures.

The kernel is memory-bounded — O(H*W), not O(F*H*W) like the soft rasterizer —
so it stays cheap at high resolution. For silhouette gradients (where coverage
itself must be differentiable), use :func:`mlx3d.renderer.render_mesh_soft`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ..cameras import Camera
from ..splatting.tiles import bin_gaussians
from ..structures import Meshes

__all__ = ["Fragments", "rasterize_meshes", "interpolate_face_attributes"]

_EPS = 1e-12

# Tile-based: each pixel only scans the faces binned to its 16x16 tile (faces
# sorted by tile then depth), so cost is ~O(faces_in_tile) instead of O(F).
# params = [width, height, tiles_x].
_RASTER_SRC = """
    constexpr int TILE = 16;
    const int W = params[0];
    const int H = params[1];
    const int tiles_x = params[2];

    const uint2 tile = threadgroup_position_in_grid.xy;
    const uint2 lid = thread_position_in_threadgroup.xy;
    const int px = tile.x * TILE + lid.x;
    const int py = tile.y * TILE + lid.y;
    const bool inside = (px < W) && (py < H);

    const int tile_id = tile.y * tiles_x + tile.x;
    const int range_start = tile_ranges[2 * tile_id];
    const int range_end = tile_ranges[2 * tile_id + 1];

    const float znear = znear_arr[0];
    const float cx = (float)px + 0.5f;
    const float cy = (float)py + 0.5f;

    int best_f = -1;
    float best_z = 1e30f;

    for (int k = range_start; k < range_end; ++k) {
        const int f = sorted_ids[k];
        const float ax = tri_xy[f * 6 + 0];
        const float ay = tri_xy[f * 6 + 1];
        const float bx = tri_xy[f * 6 + 2];
        const float by = tri_xy[f * 6 + 3];
        const float ccx = tri_xy[f * 6 + 4];
        const float ccy = tri_xy[f * 6 + 5];

        const float denom = (by - ccy) * (ax - ccx) + (ccx - bx) * (ay - ccy);
        if (metal::fabs(denom) < 1e-12f) continue;

        const float l0 = ((by - ccy) * (cx - ccx) + (ccx - bx) * (cy - ccy)) / denom;
        const float l1 = ((ccy - ay) * (cx - ccx) + (ax - ccx) * (cy - ccy)) / denom;
        const float l2 = 1.0f - l0 - l1;
        if (l0 < -1e-5f || l1 < -1e-5f || l2 < -1e-5f) continue;

        const float z = l0 * tri_z[f * 3 + 0] + l1 * tri_z[f * 3 + 1] + l2 * tri_z[f * 3 + 2];
        if (z <= znear) continue;
        if (z < best_z) { best_z = z; best_f = f; }
    }

    if (inside) {
        const int idx = py * W + px;
        pix_to_face[idx] = best_f;
        zbuf[idx] = (best_f >= 0) ? best_z : 0.0f;
    }
"""

_raster_kernel = mx.fast.metal_kernel(
    name="mesh_rasterize_hard_tiled",
    input_names=["tri_xy", "tri_z", "sorted_ids", "tile_ranges", "params", "znear_arr"],
    output_names=["pix_to_face", "zbuf"],
    source=_RASTER_SRC,
)


@dataclass
class Fragments:
    """Per-pixel rasterizer output.

    Attributes:
        pix_to_face: ``(H, W)`` int32 index of the nearest face, ``-1`` if empty.
        zbuf: ``(H, W)`` interpolated camera-space depth (0 where empty).
        bary: ``(H, W, 3)`` barycentric coordinates of the hit (differentiable
            w.r.t. vertex positions).
        valid: ``(H, W)`` bool mask of covered pixels.
        vert_ids: ``(H, W, 3)`` vertex indices of the hit face.
    """

    pix_to_face: mx.array
    zbuf: mx.array
    bary: mx.array
    valid: mx.array
    vert_ids: mx.array


def _as_verts_faces(mesh_or_verts, faces):
    if isinstance(mesh_or_verts, Meshes):
        if len(mesh_or_verts) != 1:
            raise ValueError("rasterize_meshes renders one mesh at a time.")
        return mesh_or_verts.verts_packed(), mesh_or_verts.faces_packed()
    if faces is None:
        raise ValueError("faces must be provided when verts are passed directly.")
    return mesh_or_verts, faces


def rasterize_meshes(
    camera: Camera, mesh_or_verts: Meshes | mx.array, faces: mx.array | None = None
) -> Fragments:
    """Rasterize a triangle mesh to per-pixel :class:`Fragments`.

    Args:
        camera: the viewing camera.
        mesh_or_verts: a single-mesh :class:`~mlx3d.structures.Meshes` or a
            ``(V, 3)`` vertex array.
        faces: ``(F, 3)`` indices, required when passing a raw vertex array.
    """
    verts, faces_idx = _as_verts_faces(mesh_or_verts, faces)
    faces_idx = faces_idx.astype(mx.int32)
    h, w = int(camera.height), int(camera.width)

    xy, z = camera.project_points(verts)  # (V, 2), (V,)
    tri_xy = xy[faces_idx]  # (F, 3, 2)
    tri_z = z[faces_idx]  # (F, 3)

    # Bin faces into screen tiles via a per-face bounding circle (reusing the
    # Gaussian-Splatting tiler), so each pixel only scans faces in its tile.
    sxy = mx.stop_gradient(tri_xy)
    centroid = mx.mean(sxy, axis=1)  # (F, 2)
    radii = mx.sqrt(mx.max(mx.sum((sxy - centroid[:, None, :]) ** 2, axis=-1), axis=1))
    depths = mx.mean(mx.stop_gradient(tri_z), axis=1)  # (F,) for tile sort order
    sorted_ids, tile_ranges, tiles_x, tiles_y = bin_gaussians(centroid, radii, depths, w, h)

    params = mx.array([w, h, tiles_x], dtype=mx.int32)
    znear_arr = mx.array([float(camera.znear)], dtype=mx.float32)

    # Visibility is discrete: detach the kernel inputs so autodiff never tries
    # to differentiate the custom kernel. Gradients reach the vertices through
    # the MLX barycentric recompute below instead.
    pix_to_face, zbuf = _raster_kernel(
        inputs=[
            mx.contiguous(sxy.reshape(-1).astype(mx.float32)),
            mx.contiguous(mx.stop_gradient(tri_z).reshape(-1).astype(mx.float32)),
            sorted_ids.astype(mx.int32),
            tile_ranges.astype(mx.int32),
            params,
            znear_arr,
        ],
        output_shapes=[(h * w,), (h * w,)],
        output_dtypes=[mx.int32, mx.float32],
        grid=(tiles_x * 16, tiles_y * 16, 1),
        threadgroup=(16, 16, 1),
    )
    pix_to_face = pix_to_face.reshape(h, w)
    valid = pix_to_face >= 0
    fidx = mx.where(valid, pix_to_face, 0)  # safe gather index

    vert_ids = faces_idx[fidx]  # (H, W, 3)
    # Recompute barycentrics in MLX so they are differentiable w.r.t. vertices.
    tri = xy[vert_ids]  # (H, W, 3, 2)
    ax, ay = tri[..., 0, 0], tri[..., 0, 1]
    bx, by = tri[..., 1, 0], tri[..., 1, 1]
    cxv, cyv = tri[..., 2, 0], tri[..., 2, 1]
    px = (mx.arange(w, dtype=mx.float32) + 0.5)[None, :]
    py = (mx.arange(h, dtype=mx.float32) + 0.5)[:, None]
    denom = (by - cyv) * (ax - cxv) + (cxv - bx) * (ay - cyv)
    denom = mx.where(mx.abs(denom) < _EPS, mx.ones_like(denom), denom)
    l0 = ((by - cyv) * (px - cxv) + (cxv - bx) * (py - cyv)) / denom
    l1 = ((cyv - ay) * (px - cxv) + (ax - cxv) * (py - cyv)) / denom
    l2 = 1.0 - l0 - l1
    bary = mx.stack([l0, l1, l2], axis=-1)  # (H, W, 3)
    bary = bary * valid[..., None]

    return Fragments(
        pix_to_face=pix_to_face,
        zbuf=zbuf.reshape(h, w),
        bary=bary,
        valid=valid,
        vert_ids=vert_ids,
    )


def interpolate_face_attributes(frag: Fragments, attrs: mx.array) -> mx.array:
    """Interpolate a per-vertex attribute over the rasterized fragments.

    Args:
        frag: fragments from :func:`rasterize_meshes`.
        attrs: ``(V, C)`` per-vertex values.

    Returns:
        ``(H, W, C)`` interpolated values; ``0`` on empty pixels. Differentiable
        w.r.t. both ``attrs`` and the vertex positions (through the barycentrics).
    """
    tri_attr = attrs[frag.vert_ids]  # (H, W, 3, C)
    out = mx.sum(frag.bary[..., None] * tri_attr, axis=-2)  # (H, W, C)
    return out * frag.valid[..., None]
