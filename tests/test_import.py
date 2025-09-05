def test_import():
    import mlx3d
    assert hasattr(mlx3d, "__version__")
