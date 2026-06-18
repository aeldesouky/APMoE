"""Packaging metadata tests for the installable APMoE distribution."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _pyproject() -> dict[str, object]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_default_dependencies_are_lightweight() -> None:
    """The base install should not pull heavyweight runtime backends."""
    project = _pyproject()["project"]  # type: ignore[index]
    dependencies = set(project["dependencies"])  # type: ignore[index]

    assert dependencies == {"click>=8.1", "numpy>=1.26", "pydantic>=2.5"}


def test_expected_optional_extras_exist() -> None:
    """Runtime backends and development tools are grouped behind extras."""
    project = _pyproject()["project"]  # type: ignore[index]
    extras = project["optional-dependencies"]  # type: ignore[index]

    assert {
        "serve",
        "image",
        "onnx",
        "tensorflow",
        "torch",
        "remote",
        "security",
        "redis",
        "models",
        "dev",
    }.issubset(extras)


def test_model_artifacts_are_excluded_from_builds() -> None:
    """PyPI artifacts should not include demo model binaries."""
    hatch = _pyproject()["tool"]["hatch"]["build"]  # type: ignore[index]
    excluded = set(hatch["exclude"])  # type: ignore[index]

    assert "/src/apmoe/weights/**" in excluded
    assert "/weights/**" in excluded


def test_extension_entry_point_groups_are_declared() -> None:
    """Installed plugin packages can target stable APMoE entry-point groups."""
    project = _pyproject()["project"]  # type: ignore[index]
    entry_points = project["entry-points"]  # type: ignore[index]

    assert "apmoe.experts" in entry_points
    assert "apmoe.aggregators" in entry_points
    assert "apmoe.modality_processors" in entry_points


def test_project_urls_are_publish_ready() -> None:
    """PyPI metadata should point at the real public repository."""
    project = _pyproject()["project"]  # type: ignore[index]
    urls = project["urls"]  # type: ignore[index]

    assert all("your-org" not in url for url in urls.values())
    assert urls["Repository"] == "https://github.com/aeldesouky/APMoE"
    assert "your-org" not in Path("README.md").read_text(encoding="utf-8")
    assert "your-org" not in Path("src/apmoe/cli/main.py").read_text(encoding="utf-8")


def test_package_versions_are_in_sync() -> None:
    """Release automation expects package metadata and public version to match."""
    project = _pyproject()["project"]  # type: ignore[index]
    init_text = Path("src/apmoe/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', init_text, re.MULTILINE)

    assert match is not None
    assert project["version"] == match.group(1)  # type: ignore[index]


def test_pypi_trusted_publishing_workflow_exists() -> None:
    """PyPI Trusted Publishing references this workflow filename."""
    workflow = Path(".github/workflows/publish.yml")

    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "id-token: write" in text
    assert "pypa/gh-action-pypi-publish" in text
    assert "group: pypi-publish" in text
    assert "github.event.release.tag_name" in text
    assert "Version mismatch:" in text
