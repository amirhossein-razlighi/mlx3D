import mlx.core as mx
import numpy as np

from mlx3d.losses import (
    LPIPS,
    chamfer_distance,
    closest_point_on_triangle,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    ms_ssim,
    point_mesh_face_distance,
    psnr,
    ssim,
)
from mlx3d.ops import (
    ball_query,
    decimate_mesh,
    estimate_point_normals,
    icp,
    knn_gather,
    knn_points,
    marching_cubes,
    poisson_reconstruction,
    ray_mesh_intersect,
    sample_points_from_meshes,
    spatially_sort_faces,
    subdivide_meshes,
)
from mlx3d.structures import Meshes
from mlx3d.transforms import so3_exp_map
from mlx3d.utils import cube, ico_sphere, torus


def test_poisson_reconstruction_recovers_sphere():
    mx.random.seed(0)
    sph = ico_sphere(level=4, radius=1.0)
    pts = sample_points_from_meshes(sph, 3000)[0]
    # Outward normals (orient toward origin gives inward, so negate).
    nrm = -estimate_point_normals(pts, k=16, orient_towards=(0.0, 0.0, 0.0))
    mesh = poisson_reconstruction(pts, nrm, resolution=48)
    v = mesh.verts_packed()
    assert v.shape[0] > 0 and mesh.faces_packed().shape[0] > 0
    r = mx.linalg.norm(v, axis=-1)
    # Reconstructed surface sits on the unit sphere.
    assert abs(float(r.mean()) - 1.0) < 0.05
    assert float(r.std()) < 0.1


def test_estimate_point_normals_on_sphere():
    mx.random.seed(0)
    sph = ico_sphere(level=4, radius=1.0)
    pts = sample_points_from_meshes(sph, 1500)[0]
    n = estimate_point_normals(pts, k=16, orient_towards=(0.0, 0.0, 0.0))
    radial = pts / mx.linalg.norm(pts, axis=-1, keepdims=True)
    # On a sphere the normal is (anti)parallel to the radial direction.
    assert float(mx.abs(mx.sum(n * radial, axis=-1)).mean()) > 0.98
    # Oriented toward the origin -> points inward (n . radial < 0).
    assert float((mx.sum(n * radial, axis=-1) < 0).mean()) > 0.95


def test_icp_recovers_rigid_transform():
    mx.random.seed(0)
    src = mx.random.normal((400, 3))
    r_true = so3_exp_map(mx.array([0.2, -0.3, 0.1]))
    t_true = mx.array([0.5, -0.2, 0.3])
    tgt = src @ r_true.T + t_true
    res = icp(src, tgt, iters=30)
    assert float(res["rmse"]) < 1e-3
    np.testing.assert_allclose(np.array(res["aligned"]), np.array(tgt), atol=1e-3)


def test_decimate_mesh_reduces_and_stays_valid():
    big = ico_sphere(level=4, radius=1.0)
    dec = decimate_mesh(big, voxel_size=0.3)
    assert dec.verts_packed().shape[0] < big.verts_packed().shape[0]
    assert dec.faces_packed().shape[0] < big.faces_packed().shape[0]
    f = np.asarray(dec.faces_packed())
    assert f.max() < dec.verts_packed().shape[0]
    # No degenerate faces.
    assert ((f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])).all()


def test_ray_mesh_intersect_hit_and_miss():
    verts = mx.array([[-1.0, -1, 2], [1.0, -1, 2], [0.0, 1, 2]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    m = Meshes([verts], [faces])
    o = mx.array([[0.0, 0, 0], [5.0, 5, 0]])  # first hits center, second misses
    d = mx.array([[0.0, 0, 1.0], [0.0, 0, 1.0]])
    out = ray_mesh_intersect(m, o, d)
    assert bool(out["hit"][0]) and not bool(out["hit"][1])
    assert abs(float(out["t"][0]) - 2.0) < 1e-4
    np.testing.assert_allclose(np.array(out["points"][0]), [0.0, 0.0, 2.0], atol=1e-4)
    assert int(out["face_idx"][1]) == -1


def test_ray_mesh_intersect_nearest_of_two():
    # Two parallel triangles at z=2 and z=4; ray should hit the nearer (z=2).
    verts = mx.array(
        [
            [-1.0, -1, 2],
            [1.0, -1, 2],
            [0.0, 1, 2],
            [-1.0, -1, 4],
            [1.0, -1, 4],
            [0.0, 1, 4],
        ]
    )
    faces = mx.array([[3, 4, 5], [0, 1, 2]], dtype=mx.int32)  # far listed first
    m = Meshes([verts], [faces])
    out = ray_mesh_intersect(m, mx.array([[0.0, 0, 0]]), mx.array([[0.0, 0, 1.0]]))
    assert abs(float(out["t"][0]) - 2.0) < 1e-4
    assert int(out["face_idx"][0]) == 1  # the nearer triangle


def test_ray_mesh_aabb_cull_matches_unculled_and_skips_chunks():
    verts = mx.array(
        [
            [-1.0, -1.0, 2.0],
            [1.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            [100.0, 100.0, 2.0],
            [102.0, 100.0, 2.0],
            [101.0, 102.0, 2.0],
        ]
    )
    faces = mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32)
    mesh = Meshes([verts], [faces])
    origins = mx.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.0]])
    directions = mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])

    culled = ray_mesh_intersect(
        mesh, origins, directions, face_chunk_size=1, aabb_cull=True, return_stats=True
    )
    plain = ray_mesh_intersect(
        mesh, origins, directions, face_chunk_size=1, aabb_cull=False, return_stats=True
    )

    np.testing.assert_array_equal(np.array(culled["hit"]), np.array(plain["hit"]))
    np.testing.assert_allclose(np.array(culled["t"]), np.array(plain["t"]), atol=1e-6)
    np.testing.assert_array_equal(np.array(culled["face_idx"]), np.array(plain["face_idx"]))
    assert culled["stats"]["chunks_total"] == 2
    assert culled["stats"]["chunks_skipped"] == 1
    assert culled["stats"]["face_tests"] < plain["stats"]["face_tests"]


def test_ray_mesh_aabb_cull_handles_parallel_rays_inside_slab():
    verts = mx.array([[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]])
    faces = mx.array([[0, 1, 2]], dtype=mx.int32)
    mesh = Meshes([verts], [faces])
    out = ray_mesh_intersect(
        mesh,
        mx.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        mx.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        face_chunk_size=1,
        aabb_cull=True,
        return_stats=True,
    )

    assert bool(out["hit"][0])
    assert not bool(out["hit"][1])
    assert out["stats"]["chunks_skipped"] == 0


def test_spatially_sort_faces_preserves_topology_with_face_remap():
    verts = mx.array(
        [
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = mx.array([[0, 1, 2], [3, 4, 5]], dtype=mx.int32)
    sorted_mesh, remap = spatially_sort_faces(Meshes([verts], [faces]))

    assert sorted(remap.tolist()) == [0, 1]
    sorted_faces = np.array(sorted_mesh.faces_packed())
    original = np.array(faces)
    for new_i, old_i in enumerate(remap.tolist()):
        np.testing.assert_array_equal(sorted_faces[new_i], original[old_i])


def test_ray_mesh_spatial_sort_improves_chunk_culling_and_preserves_face_ids():
    near = np.array(
        [
            [[-0.6, -0.6, 2.0], [0.6, -0.6, 2.0], [0.0, 0.6, 2.0]],
            [[-0.5, -0.5, 2.5], [0.7, -0.5, 2.5], [0.1, 0.7, 2.5]],
        ],
        dtype=np.float32,
    )
    far = near + np.array([100.0, 100.0, 100.0], dtype=np.float32)
    tris = np.stack([near[0], far[0], near[1], far[1]], axis=0)
    verts = mx.array(tris.reshape(-1, 3))
    faces = mx.arange(12, dtype=mx.int32).reshape(4, 3)
    mesh = Meshes([verts], [faces])
    origins = mx.array([[0.0, 0.0, 0.0]])
    directions = mx.array([[0.0, 0.0, 1.0]])

    unsorted = ray_mesh_intersect(
        mesh,
        origins,
        directions,
        face_chunk_size=2,
        aabb_cull=True,
        spatial_sort=False,
        return_stats=True,
    )
    sorted_out = ray_mesh_intersect(
        mesh,
        origins,
        directions,
        face_chunk_size=2,
        aabb_cull=True,
        spatial_sort=True,
        return_stats=True,
    )

    assert bool(sorted_out["hit"][0])
    assert int(sorted_out["face_idx"][0]) == int(unsorted["face_idx"][0]) == 0
    np.testing.assert_allclose(np.array(sorted_out["t"]), np.array(unsorted["t"]), atol=1e-6)
    assert sorted_out["stats"]["chunks_skipped"] > unsorted["stats"]["chunks_skipped"]
    assert sorted_out["stats"]["face_tests"] < unsorted["stats"]["face_tests"]
    assert sorted_out["stats"]["spatial_sort"] is True


def test_subdivide_meshes_counts_and_validity():
    m = ico_sphere(level=2, radius=1.0)
    v, f, e = (
        m.verts_packed().shape[0],
        m.faces_packed().shape[0],
        m.edges_packed().shape[0],
    )
    sub = subdivide_meshes(m)
    # Each face -> 4; one new vertex per unique edge.
    assert sub.faces_packed().shape[0] == 4 * f
    assert sub.verts_packed().shape[0] == v + e
    # No degenerate faces.
    f2 = sub.faces_packed()
    degenerate = mx.sum((f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2]))
    assert int(degenerate) == 0
    # Original vertices are preserved as the first V rows.
    np.testing.assert_allclose(
        np.array(sub.verts_packed()[:v]), np.array(m.verts_packed()), atol=1e-6
    )


def test_closest_point_on_triangle_regions():
    a = mx.array([0.0, 0.0, 0.0])
    b = mx.array([1.0, 0.0, 0.0])
    c = mx.array([0.0, 1.0, 0.0])
    # Point above the interior projects to its in-plane location.
    p = mx.array([0.25, 0.25, 1.0])
    cp = closest_point_on_triangle(p, a, b, c)
    np.testing.assert_allclose(np.array(cp), [0.25, 0.25, 0.0], atol=1e-5)
    # Point past vertex a clamps to a.
    cp2 = closest_point_on_triangle(mx.array([-1.0, -1.0, 0.0]), a, b, c)
    np.testing.assert_allclose(np.array(cp2), [0.0, 0.0, 0.0], atol=1e-5)
    # Point beyond edge AB midpoint clamps onto the edge.
    cp3 = closest_point_on_triangle(mx.array([0.5, -1.0, 0.0]), a, b, c)
    np.testing.assert_allclose(np.array(cp3), [0.5, 0.0, 0.0], atol=1e-5)


def test_point_mesh_face_distance_zero_on_surface():
    mesh = ico_sphere(level=3, radius=1.0)
    surf = sample_points_from_meshes(mesh, 400)[0]
    d_on = float(point_mesh_face_distance(mesh, surf))
    d_far = float(point_mesh_face_distance(mesh, surf * 1.5))
    assert d_on < 1e-3  # sampled surface points sit on faces
    assert d_far > 0.1  # scaled-out points are clearly off-surface


def test_ball_query_keeps_in_radius_neighbors():
    p2 = mx.array([[0.0, 0, 0], [0.1, 0, 0], [0.5, 0, 0], [2.0, 0, 0], [0.2, 0.1, 0]])
    p1 = mx.array([[0.0, 0, 0]])
    # radius 0.3: points 0 (0.0), 1 (0.1), 4 (~0.224) are in; 2 (0.5) and 3 (2.0) are out.
    dists, idx = ball_query(p1, p2, K=4, radius=0.3)
    ids = [int(i) for i in idx[0]]
    assert ids[:3] == [0, 1, 4]
    assert ids[3] == -1  # only three neighbors within radius
    assert not np.isfinite(float(dists[0, 3]))  # empty slot distance is inf
    # Distances are sorted ascending for the filled slots.
    filled = [float(dists[0, k]) for k in range(3)]
    assert filled == sorted(filled)


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


def test_marching_cubes_extracts_sphere_mesh():
    xs = np.linspace(-1.0, 1.0, 12)
    zz, yy, xx = np.meshgrid(xs, xs, xs, indexing="ij")
    sdf = xx * xx + yy * yy + zz * zz - 0.45
    mesh = marching_cubes(
        mx.array(sdf),
        level=0.0,
        spacing=(2 / 11, 2 / 11, 2 / 11),
        origin=(-1, -1, -1),
    )
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    r = mx.linalg.norm(verts, axis=-1)
    assert 0.45 < float(r.mean()) < 0.85


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
        m = Meshes([verts], faces)
        return mesh_edge_loss(m) + mesh_laplacian_smoothing(m) + mesh_normal_consistency(m)

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


def test_ms_ssim_monotonic_and_differentiable():
    mx.random.seed(0)
    img = mx.random.uniform(shape=(200, 200, 3))
    assert abs(float(ms_ssim(img, img)) - 1.0) < 1e-4  # identical -> 1
    # More degradation -> lower MS-SSIM.
    vals = [
        float(ms_ssim(img, mx.clip(img + mx.random.normal(img.shape) * s, 0.0, 1.0)))
        for s in (0.05, 0.15, 0.4)
    ]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    assert vals[-1] < 0.95

    def loss(x):
        return 1.0 - ms_ssim(x, img)

    g = mx.grad(loss)(mx.clip(img + 0.1, 0, 1))
    assert not bool(mx.isnan(g).any())


def test_lpips_structural_properties():
    mx.random.seed(0)
    lp = LPIPS()
    x = mx.random.uniform(shape=(48, 48, 3))
    # Identical images -> exactly zero (holds for any weights).
    assert abs(float(lp(x, x))) < 1e-6
    # Non-negative and monotonic with corruption (lin weights kept >= 0).
    small = float(lp(x, mx.clip(x + mx.random.normal(x.shape) * 0.05, 0, 1)))
    large = float(lp(x, mx.clip(x + mx.random.normal(x.shape) * 0.6, 0, 1)))
    assert 0.0 <= small < large
    # Differentiable w.r.t. the image.
    g = mx.grad(lambda z: lp(z, x))(mx.clip(x + 0.1, 0, 1))
    assert not bool(mx.isnan(g).any()) and float(mx.abs(g).sum()) > 0


def test_ms_ssim_small_image_raises():
    small = mx.zeros((32, 32, 3))
    try:
        ms_ssim(small, small)
        raise AssertionError("expected ValueError for too-small image")
    except ValueError:
        pass
