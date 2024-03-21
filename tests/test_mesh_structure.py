import unittest
import mlx.core as mx
from mlx3d.structures.meshes import Meshes
import numpy as np

class TestMeshStructure(unittest.TestCase):
    def test_with_array_simple(self):
        verts = mx.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        faces = mx.array([[[0, 1, 2]]])
        meshes = Meshes(verts, faces)
        self.assertEqual(meshes._N, 1)
        self.assertEqual(meshes._V, 3)
        self.assertEqual(meshes._F, 1)
        self.assertEqual(meshes.equisized, True)
        self.assertEqual(meshes.is_valid, mx.array([True], dtype=mx.bool_))
        self.assertEqual(meshes._num_verts_per_mesh, mx.array([3]))
        self.assertEqual(meshes._num_faces_per_mesh, mx.array([1]))
        self.assertEqual(meshes._verts_list, None) # because we passed mx.array and not a list
        self.assertEqual(meshes._faces_list, None)
  
    def test_with_list_simple(self):
          verts = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
          faces = [[[0, 1, 2]]]

          meshes = Meshes(verts, faces)
          self.assertEqual(meshes._N, 1)
          self.assertEqual(meshes._V, 3)
          self.assertEqual(meshes._F, 1)
          self.assertEqual(meshes.equisized, True)
          self.assertEqual(meshes.is_valid, mx.array([True], dtype=mx.bool_))
          self.assertEqual(meshes._num_verts_per_mesh, mx.array([3]))
          self.assertEqual(meshes._num_faces_per_mesh, mx.array([1]))
          self.assert_((np.array(meshes._verts_list) == np.array(verts)).all())
          self.assert_((np.array(meshes._faces_list) == np.array(faces)).all())
