
import mlx.core as mx
import numpy as np

from mlx3d.losses import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    psnr,
    ssim,
)
from mlx3d.ops import knn_gather, knn_points, sample_points_from_meshes
from mlx3d.utils import cube, ico_sphere, torus


def assert_close(a, b, atol=1e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol)


def test_knn_matches_numpy():
    p1 = mx.random.normal((2, 100, 3))
    p2 = mx.random.normal((2, 80, 3))
    d, idx = knn_points(p1, p2, K=3)
    a, b = np.array(p1), np.array(p2)
    full = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)
    ref_idx = np.argsort(full, axis=-1)[..., :3]
    ref_d = np.take_along_axis(full, ref_idx, axis=-1)
    np.testing.assert_allclose(np.array(d), ref_d, atol=1e-4)
    assert (np.array(idx) == ref_idx).mean() > 0.99  # ties may differ


def test_knn_unbatched_small_k_matches_numpy():
    p1 = mx.random.normal((128, 3))
    p2 = mx.random.normal((96, 3))
    d, idx = knn_points(p1, p2, K=4)
    a, b = np.array(p1), np.array(p2)
    full = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    ref_idx = np.argsort(full, axis=-1)[:, :4]
    ref_d = np.take_along_axis(full, ref_idx, axis=-1)
    np.testing.assert_allclose(np.array(d), ref_d, atol=1e-4)
    assert (np.array(idx) == ref_idx).mean() > 0.99


def test_knn_chunked_consistent():
    p1 = mx.random.normal((1, 500, 3))
    p2 = mx.random.normal((1, 200, 3))
    d1, i1 = knn_points(p1, p2, K=2, chunk_size=64)
    d2, i2 = knn_points(p1, p2, K=2, chunk_size=100000)
    assert_close(d1, d2)
    assert i1.tolist() == i2.tolist()


def test_knn_gather():
    x = mx.random.normal((2, 50, 4))
    idx = mx.random.randint(0, 50, (2, 10, 3))
    g = knn_gather(x, idx)
    assert g.shape == (2, 10, 3, 4)
    assert_close(g[0, 0, 0], x[0, int(idx[0, 0, 0])])


def test_chamfer_zero_for_identical():
    x = mx.random.normal((1, 100, 3))
    loss, _ = chamfer_distance(x, x)
    assert float(loss) < 1e-6


def test_chamfer_gradient():
    y = mx.random.normal((1, 50, 3))

    def f(x):
        loss, _ = chamfer_distance(x, y)
        return loss

    x = mx.random.normal((1, 50, 3))
    g = mx.grad(f)(x)
    assert g.shape == x.shape
    assert not bool(mx.isnan(g).any())


def test_chamfer_with_normals():
    x = mx.random.normal((1, 30, 3))
    n = mx.random.normal((1, 30, 3))
    n = n / mx.linalg.norm(n, axis=-1, keepdims=True)
    loss, loss_n = chamfer_distance(x, x, x_normals=n, y_normals=n)
    assert float(loss_n) < 1e-6


def test_sample_points_on_sphere():
    sphere = ico_sphere(level=2)
    pts, normals = sample_points_from_meshes(sphere, 2000, return_normals=True)
    assert pts.shape == (1, 2000, 3)
    radii = mx.linalg.norm(pts[0], axis=-1)
    # Icosphere level 2 approximates the unit sphere well.
    assert float(radii.min()) > 0.9
    assert float(radii.max()) < 1.01
    norm_len = mx.linalg.norm(normals[0], axis=-1)
    assert_close(norm_len, mx.ones((2000,)), atol=1e-4)


def test_mesh_losses_on_primitives():
    sphere = ico_sphere(level=1)
    e = mesh_edge_loss(sphere)
    assert float(e) > 0
    lap = mesh_laplacian_smoothing(sphere)
    assert float(lap) >= 0
    nc = mesh_normal_consistency(sphere)
    # Sphere normals vary smoothly; consistency loss small but positive.
    assert 0 < float(nc) < 0.2
    # Cube has 90-degree dihedral angles -> larger penalty than sphere.
    assert float(mesh_normal_consistency(cube())) > float(nc)


def test_mesh_loss_gradients():
    sphere = ico_sphere(level=1)
    faces = sphere.faces_list()

    def f(verts):
        from mlx3d.structures import Meshes

        m = Meshes([verts], faces)
        return (
            mesh_edge_loss(m)
            + mesh_laplacian_smoothing(m)
            + mesh_normal_consistency(m)
        )

    g = mx.grad(f)(sphere.verts_packed())
    assert not bool(mx.isnan(g).any())
    assert float(mx.abs(g).sum()) > 0


def test_torus_topology():
    t = torus(sides=8, rings=12)
    assert t.verts_packed().shape == (96, 3)
    assert t.faces_packed().shape == (192, 3)
    # Closed manifold: E = 3F/2, V - E + F = 0 for genus 1.
    E = t.edges_packed().shape[0]
    assert E == 192 * 3 // 2
    assert 96 - E + 192 == 0


def test_psnr_ssim():
    img = mx.random.uniform(shape=(32, 32, 3))
    assert float(psnr(img, img)) > 100
    assert abs(float(ssim(img, img)) - 1.0) < 1e-5
    noisy = mx.clip(img + mx.random.normal(img.shape) * 0.2, 0.0, 1.0)
    assert float(ssim(img, noisy)) < 0.9
    assert float(psnr(img, noisy)) < 25


def test_ssim_gradient():
    target = mx.random.uniform(shape=(16, 16, 3))

    def f(x):
        return 1.0 - ssim(x, target)

    g = mx.grad(f)(mx.random.uniform(shape=(16, 16, 3)))
    assert not bool(mx.isnan(g).any())
