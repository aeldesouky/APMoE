"""Tests for ``apmoe init`` scaffolding and project-local extension of all component types.

Verifies:

* Which files ``apmoe init`` creates (inventory + README keywords).
* That a generated project can reference ``proj_ext.*`` classes for every
  configurable component and pass ``apmoe validate``.
* That :class:`~apmoe.core.app.APMoEApp.from_config` requires the project
  directory on ``sys.path`` unless the CLI has already inserted it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest
from click.testing import CliRunner

from apmoe.cli.main import cli
from apmoe.core.app import APMoEApp
from apmoe.core.exceptions import ModalityError

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURE_EXT = Path(__file__).resolve().parent.parent / "fixtures" / "init_project_ext.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_init(runner: CliRunner, name: str = "init_proj") -> None:
    result = runner.invoke(cli, ["init", name])
    assert result.exit_code == 0, result.output


def _install_proj_ext(project_dir: Path) -> None:
    shutil.copyfile(_FIXTURE_EXT, project_dir / "proj_ext.py")


def _modality(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for m in cfg["apmoe"]["modalities"]:
        if m["name"] == name:
            return m
    raise KeyError(f"modality {name!r} not in config")


def _load_cfg(project_dir: Path) -> dict[str, Any]:
    return json.loads((project_dir / "config.json").read_text(encoding="utf-8"))


def _write_cfg(project_dir: Path, cfg: dict[str, Any]) -> Path:
    path = project_dir / "config.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def _strip_stale_project_dirs_from_syspath() -> None:
    """Remove temp ``.../proj`` entries left by earlier ``apmoe validate`` runs.

    ``validate`` inserts the config parent into ``sys.path`` and never removes
    it; string equality misses Windows path variants, so drop any temp-dir
    path that contains a ``proj_ext.py`` (our scaffolded extension module).
    """
    tmp_root = Path(tempfile.gettempdir()).resolve()

    def keep(entry: str) -> bool:
        if not entry:
            return True
        resolved = Path(entry).resolve()
        if not resolved.is_relative_to(tmp_root):
            return True
        return not (resolved / "proj_ext.py").is_file()

    sys.path[:] = [p for p in sys.path if keep(p)]


def _rewrite_weights_to_absolute(cfg: dict[str, Any], project_dir: Path) -> None:
    """Make expert weight paths absolute so bootstrap works when CWD is not the project."""
    for ex in cfg["apmoe"]["experts"]:
        w = ex["weights"]
        if isinstance(w, str) and w.startswith("./"):
            ex["weights"] = str((project_dir / w[2:]).resolve())


def _patch_face_expert(c: dict[str, Any]) -> None:
    for ex in c["apmoe"]["experts"]:
        if ex["name"] == "face_age_expert":
            ex["class"] = "proj_ext.InitFaceAgeExpert"
            return
    raise KeyError("face_age_expert")


def _patch_keystroke_expert(c: dict[str, Any]) -> None:
    for ex in c["apmoe"]["experts"]:
        if ex["name"] == "keystroke_age_expert":
            ex["class"] = "proj_ext.InitKeystrokeAgeExpert"
            return
    raise KeyError("keystroke_age_expert")


def _patch_aggregator(c: dict[str, Any]) -> None:
    c["apmoe"]["aggregation"]["strategy"] = "proj_ext.InitWeightedAverageAggregator"


def _patch_image_processor(c: dict[str, Any]) -> None:
    _modality(c, "image")["processor"] = "proj_ext.InitImageProcessor"


def _patch_keystroke_processor(c: dict[str, Any]) -> None:
    _modality(c, "keystroke")["processor"] = "proj_ext.InitKeystrokeProcessor"


def _patch_image_cleaner(c: dict[str, Any]) -> None:
    _modality(c, "image")["pipeline"]["cleaner"] = "proj_ext.InitImageCleaner"


def _patch_image_anonymizer(c: dict[str, Any]) -> None:
    _modality(c, "image")["pipeline"]["anonymizer"] = "proj_ext.InitImageAnonymizer"


def _patch_keystroke_cleaner(c: dict[str, Any]) -> None:
    _modality(c, "keystroke")["pipeline"]["cleaner"] = "proj_ext.InitKeystrokeCleaner"


def _patch_keystroke_anonymizer(c: dict[str, Any]) -> None:
    _modality(c, "keystroke")["pipeline"]["anonymizer"] = "proj_ext.InitKeystrokeAnonymizer"


def _patch_image_embedder(c: dict[str, Any]) -> None:
    _modality(c, "image")["pipeline"]["embedder"] = "proj_ext.InitImageEmbedder"


#: Single-field overrides: mutates a deep-copied baseline config.
_OVERRIDE_PATCHERS: dict[str, Callable[[dict[str, Any]], None]] = {
    "image_processor": _patch_image_processor,
    "keystroke_processor": _patch_keystroke_processor,
    "image_cleaner": _patch_image_cleaner,
    "image_anonymizer": _patch_image_anonymizer,
    "keystroke_cleaner": _patch_keystroke_cleaner,
    "keystroke_anonymizer": _patch_keystroke_anonymizer,
    "image_embedder": _patch_image_embedder,
    "face_expert": _patch_face_expert,
    "keystroke_expert": _patch_keystroke_expert,
    "aggregator": _patch_aggregator,
}


def _apply_all_proj_ext_classes(cfg: dict[str, Any]) -> None:
    for key in _OVERRIDE_PATCHERS:
        _OVERRIDE_PATCHERS[key](cfg)


def _validate_cli(project_dir: Path, runner: CliRunner) -> None:
    """Run validate with cwd at the project root so ``./weights/`` paths resolve."""
    prev = os.getcwd()
    os.chdir(project_dir.resolve())
    try:
        result = runner.invoke(cli, ["validate", "--config", "config.json"])
    finally:
        os.chdir(prev)
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Scaffold inventory
# ---------------------------------------------------------------------------


class TestInitScaffoldInventory:
    """What ``apmoe init`` creates on disk and what the README advertises."""

    def test_expected_top_level_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            _run_init(runner, "proj")
            root = Path("proj")
            names = {p.name for p in root.iterdir()}
            assert names >= {"config.json", "custom_expert.py", "README.md", "weights"}
            py_files = {p.name for p in root.glob("*.py")}
            assert py_files == {"custom_expert.py"}

    def test_readme_mentions_extension_points(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            _run_init(runner, "proj")
            text = (Path("proj") / "README.md").read_text(encoding="utf-8").lower()
            for needle in ("processor", "cleaner", "anonymizer", "expert", "config.json"):
                assert needle in text, f"README should mention {needle!r}"


# ---------------------------------------------------------------------------
# Project-local dotted paths + validate
# ---------------------------------------------------------------------------


class TestInitProjectExtensionValidate:
    """``proj_ext`` subclasses wired via ``config.json`` and ``apmoe validate``."""

    @pytest.mark.parametrize("field", sorted(_OVERRIDE_PATCHERS.keys()))
    def test_single_component_override_validates(
        self, tmp_path: Path, field: str
    ) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            _run_init(runner, "proj")
            project_dir = Path("proj")
            _install_proj_ext(project_dir)
            cfg = _load_cfg(project_dir)
            cfg = json.loads(json.dumps(cfg))
            _OVERRIDE_PATCHERS[field](cfg)
            _write_cfg(project_dir, cfg)
            _validate_cli(project_dir, runner)

    def test_all_components_override_validates(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            _run_init(runner, "proj")
            project_dir = Path("proj")
            _install_proj_ext(project_dir)
            cfg = _load_cfg(project_dir)
            cfg = json.loads(json.dumps(cfg))
            _apply_all_proj_ext_classes(cfg)
            _write_cfg(project_dir, cfg)
            _validate_cli(project_dir, runner)


# ---------------------------------------------------------------------------
# from_config vs CLI sys.path
# ---------------------------------------------------------------------------


class TestFromConfigSysPath:
    """``APMoEApp.from_config`` does not add the config directory to ``sys.path``."""

    def test_from_config_requires_sys_path_for_proj_ext(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            _run_init(runner, "proj")
            project_dir = Path("proj").resolve()
            isolated_root = Path.cwd()
            _install_proj_ext(project_dir)
            cfg = _load_cfg(project_dir)
            cfg = json.loads(json.dumps(cfg))
            _modality(cfg, "image").update({"processor": "proj_ext.InitImageProcessor"})
            _rewrite_weights_to_absolute(cfg, project_dir)
            _write_cfg(project_dir, cfg)
            cfg_path = (project_dir / "config.json").resolve()

            # Earlier tests invoke ``apmoe validate``, which inserts the project
            # directory into ``sys.path`` and never removes it — clear that so
            # this test actually exercises import failure without an explicit
            # path entry.
            resolved_pd = project_dir.resolve()
            _strip_stale_project_dirs_from_syspath()
            sys.path[:] = [p for p in sys.path if Path(p).resolve() != resolved_pd]
            sys.modules.pop("proj_ext", None)

            # CWD above the project: ``proj_ext`` is not importable without the
            # project dir on ``sys.path``. Weight paths are absolute so bootstrap
            # does not depend on CWD.
            os.chdir(isolated_root)
            with pytest.raises(ModalityError):
                APMoEApp.from_config(cfg_path)

            sys.path.insert(0, str(project_dir))
            try:
                app = APMoEApp.from_config(cfg_path)
            finally:
                sys.path.remove(str(project_dir))

            assert app is not None

