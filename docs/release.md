# Release

MLX3D publishes to PyPI from GitHub releases whose tag is contained in the
`main` branch. Publishing uses PyPI Trusted Publishing, so the workflow does not
need a long-lived PyPI token.

## PyPI Setup

Configure a Trusted Publisher on the `mlx3d` PyPI project:

- Owner: `amirhossein-razlighi`
- Repository name: `mlx3D`
- Workflow name: `publish.yml`
- Environment name: `pypi`

The GitHub repository should also define an environment named `pypi`. Use
environment protection rules if releases should require manual approval before
uploading to PyPI.

## Release Steps

1. Update `version` in `pyproject.toml`.
2. Merge the release commit to `main`.
3. Create and publish a GitHub release from a tag on `main`.
4. The `publish` workflow verifies the tag is reachable from `main`, builds the
   source distribution and wheel, then publishes them to PyPI.
