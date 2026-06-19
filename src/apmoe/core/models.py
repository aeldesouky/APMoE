"""Model artifact acquisition helpers for the APMoE CLI."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.resources
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal

from apmoe.core.exceptions import ConfigurationError

ModelSelection = Literal["all", "face", "keystroke"]
MODEL_PACKAGE_DISTRIBUTION = "apmoe-models"
MODEL_PACKAGE_MODULE = "apmoe_models"


@dataclass(frozen=True)
class ModelArtifact:
    """Download/copy metadata for one demo model artifact."""

    key: str
    model: Literal["face", "keystroke"]
    filename: str
    sha256: str
    size_bytes: int
    source_url: str | None
    license_note: str


MODEL_ARTIFACTS: tuple[ModelArtifact, ...] = (
    ModelArtifact(
        key="face",
        model="face",
        filename="face_age_expert.keras",
        sha256="c7e10f984023a85e4cf8e5e2a16d379766e036207667977d7cc81247e639442a",
        size_bytes=13_574_208,
        source_url=None,
        license_note=(
            "Derived demo face-age model. Confirm dataset/model redistribution "
            "rights before production use."
        ),
    ),
    ModelArtifact(
        key="keystroke_onnx",
        model="keystroke",
        filename="keystroke_age_expert.onnx",
        sha256="351a1998ffbe699c9bdd1efe97f56aa041e2d57ba3fbd388d51e65d315f8ea55",
        size_bytes=368_712,
        source_url=None,
        license_note=(
            "Derived demo keystroke model. Confirm dataset/model redistribution "
            "rights before production use."
        ),
    ),
    ModelArtifact(
        key="keystroke_constants",
        model="keystroke",
        filename="keystroke_constants.json",
        sha256="2aef91e5ee8c597740af3327dcbda2c797194a5b0688642b289646822124fe33",
        size_bytes=34_479,
        source_url=None,
        license_note=(
            "Operational constants for the demo keystroke model. Treat with the "
            "same provenance as the ONNX artifact."
        ),
    ),
)


def selected_artifacts(selection: ModelSelection) -> list[ModelArtifact]:
    """Return the artifacts included in *selection*."""
    if selection == "all":
        return list(MODEL_ARTIFACTS)
    return [artifact for artifact in MODEL_ARTIFACTS if artifact.model == selection]


def download_model_artifacts(
    dest: str | Path,
    *,
    model: ModelSelection = "all",
    force: bool = False,
    skip_existing: bool = True,
    install_model_package: bool = False,
) -> list[Path]:
    """Copy or download selected model artifacts into *dest*.

    Local source checkouts can provide model files from ``src/apmoe/weights``.
    PyPI users can install the ``apmoe-models`` artifact package through
    ``pip install "apmoe[models]"``. CLI callers may set
    ``install_model_package=True`` to install that package automatically when
    no local source is available. Packaged ``apmoe`` wheels intentionally omit
    model binaries.

    Args:
        dest: Destination directory.
        model: Artifact group to acquire.
        force: Overwrite destination files when they already exist.
        skip_existing: Leave existing destination files untouched unless
            ``force`` is true.
        install_model_package: Install ``apmoe[models]`` from PyPI when the
            selected artifact source is unavailable locally.

    Returns:
        Paths that were copied or downloaded.

    Raises:
        ConfigurationError: If a source is unavailable or checksum validation
            fails.
    """
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for artifact in selected_artifacts(model):
        target = dest_path / artifact.filename
        if target.exists() and skip_existing and not force:
            continue
        if target.exists() and not force and not skip_existing:
            raise ConfigurationError(
                f"Model artifact already exists: {target}. Use --force to overwrite.",
                context={"path": str(target), "artifact": artifact.key},
            )

        source = _resolve_artifact_source(artifact)
        if source is None and install_model_package:
            _install_model_package_from_pypi()
            source = _resolve_artifact_source(artifact)
        if source is None:
            raise ConfigurationError(
                "No source is configured for model artifact "
                f"'{artifact.filename}'. Wheels do not bundle demo model files; "
                f"install them with `pip install \"apmoe[models]\"`, set "
                "APMOE_MODEL_SOURCE_DIR to a directory containing the files, "
                f"or set APMOE_MODEL_SOURCE_{artifact.key.upper()} to a file path or URL.",
                context={"artifact": artifact.key, "filename": artifact.filename},
            )

        _copy_or_download(source, target)
        _verify_sha256(target, artifact.sha256)
        written.append(target)

    return written


def _resolve_artifact_source(
    artifact: ModelArtifact,
) -> str | Path | Traversable | None:
    """Find an available source for *artifact*."""
    per_artifact = os.environ.get(f"APMOE_MODEL_SOURCE_{artifact.key.upper()}")
    if per_artifact:
        return per_artifact

    source_dir = os.environ.get("APMOE_MODEL_SOURCE_DIR")
    if source_dir:
        candidate = Path(source_dir) / artifact.filename
        if candidate.exists():
            return candidate

    package_candidate = Path(__file__).resolve().parents[1] / "weights" / artifact.filename
    if package_candidate.exists():
        return package_candidate

    model_package_source = _resolve_model_package_artifact(artifact)
    if model_package_source is not None:
        return model_package_source

    return artifact.source_url


def _resolve_model_package_artifact(
    artifact: ModelArtifact,
) -> Traversable | None:
    """Return an artifact resource from ``apmoe-models`` if installed."""
    try:
        candidate = importlib.resources.files(MODEL_PACKAGE_MODULE).joinpath(
            "weights",
            artifact.filename,
        )
    except ModuleNotFoundError:
        return None
    if candidate.is_file():
        return candidate
    return None


def _copy_or_download(
    source: str | Path | Traversable,
    target: Path,
) -> None:
    """Copy a local artifact or download a URL to *target*."""
    if isinstance(source, Path):
        shutil.copy2(source, target)
        return
    if isinstance(source, Traversable):
        with source.open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return

    parsed = urllib.parse.urlparse(source)
    if parsed.scheme in {"http", "https", "file"}:
        try:
            with urllib.request.urlopen(source) as response, target.open("wb") as fh:
                shutil.copyfileobj(response, fh)
        except OSError as exc:
            raise ConfigurationError(
                f"Failed to download model artifact from {source}: {exc}",
                context={"source": source, "target": str(target)},
            ) from exc
        return

    source_path = Path(source)
    if source_path.exists():
        shutil.copy2(source_path, target)
        return

    raise ConfigurationError(
        f"Model artifact source does not exist and is not a supported URL: {source}",
        context={"source": source, "target": str(target)},
    )


def model_package_requirement() -> str:
    """Return the PyPI requirement used to install packaged model artifacts."""
    try:
        version = importlib.metadata.version("apmoe")
    except importlib.metadata.PackageNotFoundError:
        from apmoe import __version__ as version
    return f"{MODEL_PACKAGE_DISTRIBUTION}=={version}"


def _install_model_package_from_pypi() -> None:
    """Install the version-matched ``apmoe-models`` package from PyPI."""
    if _model_package_is_available():
        return

    requirement = model_package_requirement()
    cmd = [sys.executable, "-m", "pip", "install", requirement]
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    importlib.invalidate_caches()
    if result.returncode == 0 and _model_package_is_available():
        return

    details = (result.stderr or result.stdout).strip()
    raise ConfigurationError(
        f"Could not install model artifact package from PyPI: {requirement}.",
        context={"requirement": requirement, "pip_output": details},
    )


def _model_package_is_available() -> bool:
    """Return whether the model artifact package can be imported."""
    try:
        importlib.resources.files(MODEL_PACKAGE_MODULE)
    except ModuleNotFoundError:
        return False
    return True


def _verify_sha256(path: Path, expected: str) -> None:
    """Validate the SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        with suppress(OSError):
            path.unlink()
        raise ConfigurationError(
            f"Checksum mismatch for model artifact '{path.name}'.",
            context={"path": str(path), "expected": expected, "actual": actual},
        )
