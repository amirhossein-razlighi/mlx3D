"""Mesh primitives: icosphere, cube, torus."""

import math

import mlx.core as mx
import numpy as np

from ..structures import Meshes

__all__ = ["ico_sphere", "cube", "torus"]


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return verts, faces


def ico_sphere(level: int = 2, radius: float = 1.0) -> Meshes:
    """Unit icosphere obtained by subdividing an icosahedron ``level`` times."""
    verts, faces = _icosahedron()
    for _ in range(level):
        edge_mid: dict[tuple[int, int], int] = {}
        new_faces = []
        verts_list = verts.tolist()

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in edge_mid:
                m = (np.array(verts_list[a]) + np.array(verts_list[b])) / 2.0
                m /= np.linalg.norm(m)
                edge_mid[key] = len(verts_list)
                verts_list.append(m.tolist())
            return edge_mid[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(verts_list)
        faces = np.array(new_faces, dtype=np.int64)

    return Meshes(
        [mx.array((verts * radius).astype(np.float32))],
        [mx.array(faces.astype(np.int32))],
    )


def cube(size: float = 2.0) -> Meshes:
    """Axis-aligned cube centered at the origin with edge length ``size``."""
    s = size / 2.0
    verts = mx.array(
        [
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
        ]
    )
    faces = mx.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
        ],
        dtype=mx.int32,
    )
    return Meshes([verts], [faces])


def torus(r: float = 0.5, R: float = 1.0, sides: int = 16, rings: int = 32) -> Meshes:
    """Torus with tube radius ``r`` and center radius ``R``."""
    verts = []
    for i in range(rings):
        phi = 2.0 * math.pi * i / rings
        for j in range(sides):
            theta = 2.0 * math.pi * j / sides
            x = (R + r * math.cos(theta)) * math.cos(phi)
            y = r * math.sin(theta)
            z = (R + r * math.cos(theta)) * math.sin(phi)
            verts.append([x, y, z])
    faces = []
    for i in range(rings):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            c = ((i + 1) % rings) * sides + j
            d = ((i + 1) % rings) * sides + (j + 1) % sides
            faces += [[a, c, b], [b, c, d]]
    return Meshes(
        [mx.array(np.asarray(verts, dtype=np.float32))],
        [mx.array(np.asarray(faces, dtype=np.int32))],
    )
