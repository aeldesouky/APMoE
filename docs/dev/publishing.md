# Publishing to PyPI

APMoE publishes two distributions through GitHub Actions and PyPI Trusted
Publishing:

- `apmoe`: framework code and runtime dependencies, without model artifacts
- `apmoe-models`: demo model artifact package used by `apmoe[models]`

The workflow file is:

```text
.github/workflows/publish.yml
```

When configuring each PyPI project publisher, use:

```text
Owner: aeldesouky
Repository name: APMoE
Workflow filename: publish.yml
Environment name: pypi
```

Create this trusted publisher for both PyPI projects: `apmoe` and
`apmoe-models`.

The workflow runs when a GitHub release is published and can also be started
manually from the Actions tab. It builds source distributions and wheels for
both packages, runs `twine check`, smoke-installs the wheels, and publishes
with OIDC. The workflow publishes `apmoe-models` first, then `apmoe`, so the
framework package is not released if the model artifact package cannot be
published. No PyPI API token secret is required.

Before publishing a release:

1. Update `pyproject.toml`, `src/apmoe/__init__.py`,
   `packages/apmoe-models/pyproject.toml`, and
   `packages/apmoe-models/src/apmoe_models/__init__.py` to the same version.
2. Run the tests and package checks.
3. Commit and push to `main`.
4. Create the release tag from that pushed commit, not from an older commit.
   The tag must point at the commit that contains the matching package
   metadata.
5. Create a GitHub release whose tag matches the package version, such as
   `v0.1.0` for version `0.1.0`.

If the workflow reports a mismatch such as `Release tag 'vX.Y.Z' does not
match package version 'X.Y.(Z-1)'`, the tag is pointing at an older commit.
Delete and recreate the GitHub release/tag from the current `main` commit, or
publish a new patch version with a new tag.

The workflow fails early if the release tag does not match the package version.
PyPI does not allow overwriting an existing version, so every release must use a
new version number.

Recommended local checks:

```bash
uv run pytest -q
uv build --wheel --sdist --out-dir .pytest-tmp-pypi-publish
python -m twine check .pytest-tmp-pypi-publish/*
```

The `apmoe` wheel excludes model artifacts. The `apmoe-models` wheel contains
the packaged demo artifacts. Users acquire files with `pip install
"apmoe[models]"`, `apmoe download-models`, or their own configured artifact
source.
