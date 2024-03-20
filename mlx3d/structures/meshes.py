import mlx.core as mx
from ..utils import *


class Meshes:
    def __init__(self, verts, faces, textures=None, *, verts_normals=None) -> None:
        self.device = mx.Device(mx.gpu)
        self.textures = textures  # TODO: Check valid texture
        self.equisized = False
        self.is_valid = None

        self._N = 0
        self._V = 0
        self._F = 0

        self._verts_list = None
        self._faces_list = None

        self._verts_padded = None
        self._faces_padded = None

        self._num_verts_per_mesh = None
        self._num_faces_per_mesh = None

    def _check_for_verts_and_faces(self, verts, faces):
        if isinstance(verts, list) and isinstance(faces, list):
            self._verts_list = verts
            self._faces_list = [
                face[mx.greater(face, -1).all(1)].astype(mx.int64)
                if len(face) > 0
                else face
                for face in faces
            ]
            self._N = len(self._verts_list)
            self.is_valid = mx.zeros((self._N,), dtype=mx.bool_, device=self.device)

            if self._N > 0:
                self._num_verts_per_mesh = mx.array(
                    [len(verts) for verts in self._verts_list], device=self.device
                )
                self._V = self._num_verts_per_mesh.max()
                self._num_faces_per_mesh = mx.array(
                    [len(faces) for faces in self._faces_list], device=self.device
                )
                self._F = self._num_faces_per_mesh.max()
                self.is_valid = mx.array(
                    [
                        len(verts) > 0 and len(faces) > 0
                        for verts, faces in zip(self._verts_list, self._faces_list)
                    ],
                    dtype=mx.bool_,
                    device=self.device,
                )
                if (unique_num_items(self._num_verts_per_mesh) == 1) and (
                    unique_num_items(self._num_faces_per_mesh) == 1
                ):
                    self.equisized = True

        elif isinstance(verts, mx.array) and isinstance(faces, mx.array):
            if verts.ndim != 3 or faces.ndim != 3:
                raise ValueError(
                    "verts and faces must be of shape (N, V, 3) and (N, F, 3)"
                )
            if verts.shape[2] != 3 or faces.shape[2] != 3:
                raise ValueError(
                    "verts and faces must be of shape (N, V, 3) and (N, F, 3)"
                )
            self._N = verts.shape[0]
            self._V = verts.shape[1]
            self._verts_padded = verts
            self._faces_padded = faces.astype(mx.int64)

            self.is_valid = mx.zeros((self._N,), dtype=mx.bool_, device=self.device)

            if self._N > 0:
                faces_not_padded = mx.all(mx.greater(faces, -1), axis=2)
                self._num_faces_per_mesh = mx.sum(faces_not_padded, axis=1)
                if mx.any(faces_not_padded[:, :, :-1] < faces_not_padded[:, :, 1:]):
                    raise ValueError("Padding of faces must be at the end of the array")

                self.is_valid = self._num_faces_per_mesh > 0
                self._F = self._num_faces_per_mesh.max()
                
                if unique_num_items(self._num_faces_per_mesh) == 1:
                    self.equisized = True

                self._num_verts_per_mesh = mx.full(
                    shape=(self._N,),
                    fill_value=self._V,
                    dtype=mx.int64,
                    device=self.device,
                )
        else:
            raise ValueError("verts and faces must be of type list or mx.array")