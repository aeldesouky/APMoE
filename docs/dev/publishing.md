# Publishing to PyPI

APMoE publishes through GitHub Actions and PyPI Trusted Publishing. The workflow
file is:

```text
.github/workflows/publish.yml
```

When configuring the PyPI project publisher, use:

```text
Owner: aeldesouky
Repository name: APMoE
Workflow filename: publish.yml
Environment name: pypi
```

The workflow runs when a GitHub release is published and can also be started
manually from the Actions tab. It builds the source distribution and wheel,
runs `twine check`, smoke-installs the wheel, and publishes with OIDC. No PyPI
API token secret is required.

Before publishing a release:

1. Update `pyproject.toml` and `src/apmoe/__init__.py` to the same version.
2. Run the tests and package checks.
3. Commit and push to `main`.
4. Create a GitHub release whose tag matches the package version, such as
   `v0.1.0` for version `0.1.0`.

The workflow fails early if the release tag does not match the package version.
PyPI does not allow overwriting an existing version, so every release must use a
new version number.

Recommended local checks:

```bash
uv run pytest -q
uv build --wheel --sdist --out-dir .pytest-tmp-pypi-publish
python -m twine check .pytest-tmp-pypi-publish/*
```

The wheel excludes model artifacts. Users acquire demo model files explicitly
with `apmoe download-models` or provide their own artifacts.
