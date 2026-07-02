# Security Policy

## Supported Versions

Only the latest released version of `mlx3d` receives security fixes.

## Reporting a Vulnerability

Please do **not** open a public issue for security problems. Instead, report
them privately:

- Use GitHub's [private vulnerability reporting](https://github.com/amirhossein-razlighi/mlx3D/security/advisories/new), or
- Email <arazlighi@gmail.com> with a description, reproduction steps, and the
  affected version.

You can expect an acknowledgment within a few days. Once a fix is available, the
vulnerability will be disclosed in the release notes with credit to the
reporter (unless you prefer to stay anonymous).

## Scope notes

`mlx3d` executes no network services by default; the interactive viewer
(`mlx3d-view`, live training previews) binds to `127.0.0.1` and is intended for
local use only. Loading untrusted PLY/OBJ/COLMAP files is parsed in pure
Python/NumPy — malformed-file crashes are ordinary bugs, but anything that
leads to memory unsafety or code execution is in scope for this policy.
