def test_import():
    import mlx3d

    assert hasattr(mlx3d, "__version__")


def test_version_matches_package_metadata():
    from importlib.metadata import version

    import mlx3d

    assert mlx3d.__version__ == version("mlx3d")
