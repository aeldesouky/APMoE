"""Packaging metadata tests for the installable APMoE distribution."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _pyproject() -> dict[str, object]:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))


def test_default_dependencies_include_runtime_integrations() -> None:
    """The base install should include framework, serving, security, and ML runtimes."""
    project = _pyproject()["project"]  # type: ignore[index]
    dependencies = set(project["dependencies"])  # type: ignore[index]

    assert {
        "click>=8.1",
        "cryptography>=42.0",
        "fastapi>=0.110",
        "httpx>=0.27",
        "numpy>=1.26",
        "onnxruntime>=1.17",
        "pillow>=10.0",
        "pydantic>=2.5",
        "PyJWT[crypto]>=2.8",
        "python-multipart>=0.0.22",
        "redis>=5.0",
        "torch>=2.2",
        "uvicorn[standard]>=0.29",
    }.issubset(dependencies)
    assert any(dep.startswith("tensorflow>=2.15") for dep in dependencies)


def test_expected_optional_extras_exist() -> None:
    """Runtime extra names remain as compatibility aliases; dev installs tools."""
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
    for alias in (
        "serve",
        "image",
        "onnx",
        "tensorflow",
        "torch",
        "remote",
        "security",
        "redis",
    ):
        assert extras[alias] == []
    assert extras["models"] == [f"apmoe-models=={project['version']}"]
    assert "pytest>=8.0" in extras["dev"]


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
    model_project = tomllib.loads(
        Path("packages/apmoe-models/pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    model_init_text = Path("packages/apmoe-models/src/apmoe_models/__init__.py").read_text(
        encoding="utf-8"
    )
    model_match = re.search(r'^__version__ = "([^"]+)"$', model_init_text, re.MULTILINE)

    assert model_match is not None
    assert model_project["version"] == project["version"]  # type: ignore[index]
    assert model_match.group(1) == project["version"]  # type: ignore[index]


def test_model_artifact_package_contains_expected_files() -> None:
    """The optional PyPI model package should contain the demo artifacts."""
    weights_dir = Path("packages/apmoe-models/src/apmoe_models/weights")

    assert (weights_dir / "face_age_expert.keras").is_file()
    assert (weights_dir / "keystroke_age_expert.onnx").is_file()
    assert (weights_dir / "keystroke_constants.json").is_file()


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
