"""A batched triangle-mesh container, in the spirit of PyTorch3D's ``Meshes``.

A batch holds ``N`` meshes with possibly different numbers of vertices and
faces. Three views of the data are available:

- **list**: ``verts_list()[i]`` is ``(V_i, 3)``, ``faces_list()[i]`` is ``(F_i, 3)``.
- **packed**: all meshes concatenated, ``verts_packed()`` is ``(sum(V_i), 3)``
  and ``faces_packed()`` is ``(sum(F_i), 3)`` with vertex indices offset so
  they index directly into the packed vertices.
- **padded**: ``verts_padded()`` is ``(N, max(V_i), 3)`` zero-padded and
  ``faces_padded()`` is ``(N, max(F_i), 3)`` padded with ``-1``.

Construction and the packed/padded conversions are differentiable with
respect to vertex positions, so a ``Meshes`` can be rebuilt every iteration
of an optimization loop from a vertex array with gradients flowing through.
"""

import numpy as np

import mlx.core as mx

__all__ = ["Meshes", "join_meshes_as_batch"]


def _as_float_array(a) -> mx.array:
    a = mx.array(a) if not isinstance(a, mx.array) else a
    return a.astype(mx.float32) if mx.issubdtype(a.dtype, mx.integer) else a


def _as_int_array(a) -> mx.array:
    a = mx.array(a) if not isinstance(a, mx.array) else a
    return a.astype(mx.int32)


class Meshes:
    """Batch of triangle meshes.

    Args:
        verts (list[mx.array] | mx.array): list of ``(V_i, 3)`` float arrays,
            or a padded ``(N, V, 3)`` array.
        faces (list[mx.array] | mx.array): list of ``(F_i, 3)`` integer
            arrays, or a padded ``(N, F, 3)`` array where unused rows are
            filled with ``-1``.
    """

    def __init__(self, verts, faces) -> None:
        if isinstance(verts, (list, tuple)) and isinstance(faces, (list, tuple)):
            if len(verts) != len(faces):
                raise ValueError("verts and faces must contain the same number of meshes.")
            self._verts_list = [_as_float_array(v) for v in verts]
            self._faces_list = [_as_int_array(f) for f in faces]
            for v, f in zip(self._verts_list, self._faces_list):
                if v.ndim != 2 or v.shape[-1] != 3:
                    raise ValueError("Each verts entry must have shape (V, 3).")
                if f.ndim != 2 or f.shape[-1] != 3:
                    raise ValueError("Each faces entry must have shape (F, 3).")
        elif isinstance(verts, mx.array) and isinstance(faces, mx.array):
            if verts.ndim != 3 or verts.shape[-1] != 3:
                raise ValueError("Padded verts must have shape (N, V, 3).")
            if faces.ndim != 3 or faces.shape[-1] != 3:
                raise ValueError("Padded faces must have shape (N, F, 3).")
            if verts.shape[0] != faces.shape[0]:
                raise ValueError("verts and faces must have the same batch dimension.")
            faces = _as_int_array(faces)
            verts = _as_float_array(verts)
            # Number of valid (non -1-padded) faces per mesh.
            valid = (faces > -1).all(axis=-1)
            num_faces = valid.sum(axis=-1)
            self._verts_list = [verts[i] for i in range(verts.shape[0])]
            nf = num_faces.tolist()
            self._faces_list = [faces[i, : nf[i]] for i in range(faces.shape[0])]
        else:
            raise ValueError(
                "verts and faces must both be lists of arrays or both padded mx.arrays."
            )

        self._N = len(self._verts_list)
        # Caches.
        self._verts_packed = None
        self._faces_packed = None
        self._verts_padded = None
        self._faces_padded = None
        self._verts_normals_packed = None
        self._faces_normals_packed = None
        self._faces_areas_packed = None
        self._edges_packed = None

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return self._N

    def __getitem__(self, index) -> "Meshes":
        if isinstance(index, int):
            index = [index]
        if isinstance(index, slice):
            index = list(range(self._N))[index]
        return Meshes(
            [self._verts_list[i] for i in index],
            [self._faces_list[i] for i in index],
        )

    def isempty(self) -> bool:
        return self._N == 0 or all(v.shape[0] == 0 for v in self._verts_list)

    @property
    def num_verts_per_mesh(self) -> mx.array:
        return mx.array([v.shape[0] for v in self._verts_list], dtype=mx.int32)

    @property
    def num_faces_per_mesh(self) -> mx.array:
        return mx.array([f.shape[0] for f in self._faces_list], dtype=mx.int32)

    # -------------------------------------------------------------- list views
    def verts_list(self) -> list[mx.array]:
        return self._verts_list

    def faces_list(self) -> list[mx.array]:
        return self._faces_list

    # ------------------------------------------------------------ packed views
    def verts_packed(self) -> mx.array:
        if self._verts_packed is None:
            self._verts_packed = (
                mx.concatenate(self._verts_list, axis=0)
                if self._N > 0
                else mx.zeros((0, 3))
            )
        return self._verts_packed

    def faces_packed(self) -> mx.array:
        """(sum(F_i), 3) faces with vertex indices into ``verts_packed``."""
        if self._faces_packed is None:
            faces = []
            offset = 0
            for v, f in zip(self._verts_list, self._faces_list):
                faces.append(f + offset)
                offset += v.shape[0]
            self._faces_packed = (
                mx.concatenate(faces, axis=0)
                if faces
                else mx.zeros((0, 3), dtype=mx.int32)
            )
        return self._faces_packed

    def mesh_to_verts_packed_first_idx(self) -> mx.array:
        counts = self.num_verts_per_mesh
        return mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)[:-1]])

    def mesh_to_faces_packed_first_idx(self) -> mx.array:
        counts = self.num_faces_per_mesh
        return mx.concatenate([mx.zeros((1,), dtype=mx.int32), mx.cumsum(counts)[:-1]])

    def verts_packed_to_mesh_idx(self) -> mx.array:
        return mx.concatenate(
            [
                mx.full((v.shape[0],), i, dtype=mx.int32)
                for i, v in enumerate(self._verts_list)
            ]
            or [mx.zeros((0,), dtype=mx.int32)]
        )

    def faces_packed_to_mesh_idx(self) -> mx.array:
        return mx.concatenate(
            [
                mx.full((f.shape[0],), i, dtype=mx.int32)
                for i, f in enumerate(self._faces_list)
            ]
            or [mx.zeros((0,), dtype=mx.int32)]
        )

    # ------------------------------------------------------------ padded views
    def verts_padded(self) -> mx.array:
        if self._verts_padded is None:
            V = max((v.shape[0] for v in self._verts_list), default=0)
            rows = []
            for v in self._verts_list:
                pad = V - v.shape[0]
                rows.append(mx.pad(v, ((0, pad), (0, 0))) if pad > 0 else v)
            self._verts_padded = mx.stack(rows, axis=0) if rows else mx.zeros((0, 0, 3))
        return self._verts_padded

    def faces_padded(self) -> mx.array:
        if self._faces_padded is None:
            F = max((f.shape[0] for f in self._faces_list), default=0)
            rows = []
            for f in self._faces_list:
                pad = F - f.shape[0]
                rows.append(
                    mx.pad(f, ((0, pad), (0, 0)), constant_values=-1) if pad > 0 else f
                )
            self._faces_padded = (
                mx.stack(rows, axis=0) if rows else mx.zeros((0, 0, 3), dtype=mx.int32)
            )
        return self._faces_padded

    # -------------------------------------------------------- derived geometry
    def faces_normals_packed(self) -> mx.array:
        """(sum(F_i), 3) unnormalized face normals (length = 2 * face area)."""
        if self._faces_normals_packed is None:
            verts = self.verts_packed()
            faces = self.faces_packed()
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            self._faces_normals_packed = mx.linalg.cross(v1 - v0, v2 - v0)
        return self._faces_normals_packed

    def faces_areas_packed(self) -> mx.array:
        """(sum(F_i),) face areas."""
        if self._faces_areas_packed is None:
            n = self.faces_normals_packed()
            self._faces_areas_packed = 0.5 * mx.linalg.norm(n, axis=-1)
        return self._faces_areas_packed

    def verts_normals_packed(self, eps: float = 1e-12) -> mx.array:
        """(sum(V_i), 3) area-weighted vertex normals."""
        if self._verts_normals_packed is None:
            verts = self.verts_packed()
            faces = self.faces_packed()
            fn = self.faces_normals_packed()
            normals = mx.zeros_like(verts)
            for k in range(3):
                normals = normals.at[faces[:, k]].add(fn)
            self._verts_normals_packed = normals / mx.maximum(
                mx.linalg.norm(normals, axis=-1, keepdims=True), eps
            )
        return self._verts_normals_packed

    def verts_normals_list(self) -> list[mx.array]:
        normals = self.verts_normals_packed()
        out = []
        offset = 0
        for v in self._verts_list:
            out.append(normals[offset : offset + v.shape[0]])
            offset += v.shape[0]
        return out

    def edges_packed(self) -> mx.array:
        """(E, 2) unique undirected edges with indices into ``verts_packed``.

        Edge connectivity carries no gradients, so this is computed once on
        CPU with NumPy and cached.
        """
        if self._edges_packed is None:
            faces = np.array(self.faces_packed())
            if faces.shape[0] == 0:
                self._edges_packed = mx.zeros((0, 2), dtype=mx.int32)
            else:
                e = np.concatenate(
                    [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
                )
                e.sort(axis=1)
                e = np.unique(e, axis=0)
                self._edges_packed = mx.array(e.astype(np.int32))
        return self._edges_packed

    def bounding_boxes(self) -> mx.array:
        """(N, 3, 2) per-mesh min/max corners."""
        boxes = []
        for v in self._verts_list:
            boxes.append(mx.stack([v.min(axis=0), v.max(axis=0)], axis=-1))
        return mx.stack(boxes, axis=0)

    # ------------------------------------------------------------ modification
    def offset_verts(self, offsets: mx.array) -> "Meshes":
        """Return a new ``Meshes`` with packed-vertex ``offsets`` added.

        ``offsets`` has shape ``(sum(V_i), 3)`` (or broadcastable ``(3,)``).
        """
        new_verts = self.verts_packed() + offsets
        out = []
        offset = 0
        for v in self._verts_list:
            out.append(new_verts[offset : offset + v.shape[0]])
            offset += v.shape[0]
        return Meshes(out, self._faces_list)

    def update_padded(self, new_verts_padded: mx.array) -> "Meshes":
        """Return a new ``Meshes`` with the same topology and new padded vertices."""
        new_list = [
            new_verts_padded[i, : v.shape[0]] for i, v in enumerate(self._verts_list)
        ]
        return Meshes(new_list, self._faces_list)

    def scale_verts(self, scale) -> "Meshes":
        return Meshes([v * scale for v in self._verts_list], self._faces_list)


def join_meshes_as_batch(meshes: list[Meshes]) -> Meshes:
    """Concatenate several ``Meshes`` batches into one."""
    verts = [v for m in meshes for v in m.verts_list()]
    faces = [f for m in meshes for f in m.faces_list()]
    return Meshes(verts, faces)
