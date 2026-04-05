"""Unit tests for the APMoE CLI (Phase 5).

Covers:

* ``apmoe init`` — directory scaffolding, file creation, duplicate guard.
* ``apmoe serve`` — config loading, CLI flag → env-var overrides, error paths.
* ``apmoe predict`` — directory scan, JSON manifest, output file, error propagation.
* ``apmoe validate`` — success path, bootstrap error, validation error, health display.
* ``-h`` / ``--help`` output for every command.
* ``--version`` flag on the CLI group.
* Invalid config paths and malformed JSON handled gracefully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from apmoe.cli.main import cli
from apmoe.core.exceptions import APMoEError, ConfigurationError, PipelineError
from apmoe.core.types import ExpertOutput, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: Path, data: dict[str, Any]) -> Path:
    """Write *data* as JSON to *path* and return *path*."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal_config_data() -> dict[str, Any]:
    """Return a minimal but schema-valid raw config dict."""
    return {
        "apmoe": {
            "modalities": [
                {
                    "name": "visual",
                    "processor": "myproject.processors.VisualProcessor",
                    "pipeline": {
                        "cleaner": "myproject.cleaners.ImageCleaner",
                        "anonymizer": "myproject.anonymizers.FaceAnonymizer",
                    },
                }
            ],
            "experts": [
                {
                    "name": "face_expert",
                    "class": "myproject.experts.FaceExpert",
                    "weights": "./weights/face.pt",
                    "modalities": ["visual"],
                }
            ],
            "aggregation": {
                "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
            },
        }
    }


def _make_prediction(age: float = 30.0, confidence: float = 0.9) -> Prediction:
    """Build a minimal :class:`~apmoe.core.types.Prediction` for testing."""
    return Prediction(
        predicted_age=age,
        confidence=confidence,
        confidence_interval=(age - 5.0, age + 5.0),
        per_expert_outputs=[
            ExpertOutput(
                expert_name="face_expert",
                consumed_modalities=["visual"],
                predicted_age=age,
                confidence=confidence,
                metadata={},
            )
        ],
        skipped_experts=[],
        metadata={},
    )


def _make_mock_app(prediction: Prediction | None = None) -> MagicMock:
    """Return a MagicMock standing in for :class:`~apmoe.core.app.APMoEApp`."""
    mock = MagicMock()
    mock.predict.return_value = prediction if prediction is not None else _make_prediction()
    mock.validate.return_value = {
        "valid": True,
        "expert_health": {"face_expert": True},
        "issues": [],
    }
    mock.config.apmoe.serving.host = "0.0.0.0"
    mock.config.apmoe.serving.port = 8000
    mock.config.apmoe.serving.workers = 4
    mock.config.apmoe.serving.log_level = "info"
    return mock


# ---------------------------------------------------------------------------
# apmoe init
# ---------------------------------------------------------------------------


class TestInitCommand:
    """Tests for the ``apmoe init`` command."""

    def test_creates_project_directory(self, tmp_path: Path) -> None:
        """Running ``init myproject`` creates the project directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "myproject"])
            assert result.exit_code == 0, result.output
            assert Path("myproject").is_dir()

    def test_creates_config_json(self, tmp_path: Path) -> None:
        """``config.json`` is written inside the project directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "myproject"])
            assert (Path("myproject") / "config.json").is_file()

    def test_config_json_is_valid_json(self, tmp_path: Path) -> None:
        """The generated ``config.json`` is valid JSON with an ``apmoe`` root key."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "myproject"])
            content = (Path("myproject") / "config.json").read_text(encoding="utf-8")
            parsed = json.loads(content)
            assert "apmoe" in parsed

    def test_creates_custom_expert_py(self, tmp_path: Path) -> None:
        """``custom_expert.py`` is written inside the project directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "myproject"])
            assert (Path("myproject") / "custom_expert.py").is_file()

    def test_creates_extension_stub_files(self, tmp_path: Path) -> None:
        """The scaffold includes starter files for all major extension points."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "myproject"])
            project_dir = Path("myproject")
            expected_files = {
                "custom_processor.py",
                "custom_cleaner.py",
                "custom_anonymizer.py",
                "custom_embedder.py",
                "custom_expert.py",
                "custom_aggregator.py",
            }
            actual_files = {path.name for path in project_dir.iterdir() if path.is_file()}
            assert expected_files.issubset(actual_files)

    def test_creates_weights_directory(self, tmp_path: Path) -> None:
        """A ``weights/`` subdirectory is created for pretrained files."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "myproject"])
            assert (Path("myproject") / "weights").is_dir()

    def test_creates_readme(self, tmp_path: Path) -> None:
        """A ``README.md`` file is written inside the project directory."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "myproject"])
            assert (Path("myproject") / "README.md").is_file()

    def test_default_project_name(self, tmp_path: Path) -> None:
        """Omitting PROJECT_NAME uses ``my_apmoe_project`` as the default."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path("my_apmoe_project").is_dir()

    def test_existing_directory_exits_nonzero(self, tmp_path: Path) -> None:
        """Running ``init`` when the target directory exists exits non-zero."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("myproject").mkdir()
            result = runner.invoke(cli, ["init", "myproject"])
            assert result.exit_code != 0

    def test_config_references_face_model(self, tmp_path: Path) -> None:
        """The generated ``config.json`` points at the bundled face model file."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "cool_project"])
            content = (Path("cool_project") / "config.json").read_text(encoding="utf-8")
            assert "face_age_expert.keras" in content

    def test_config_references_keystroke_model(self, tmp_path: Path) -> None:
        """The generated ``config.json`` points at the bundled keystroke ONNX model."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init", "my-project"])
            content = (Path("my-project") / "config.json").read_text(encoding="utf-8")
            assert "keystroke_age_expert.onnx" in content

    def test_output_mentions_created_files(self, tmp_path: Path) -> None:
        """The command output lists the scaffolded files."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "proj"])
            assert "config.json" in result.output
            assert "custom_processor.py" in result.output
            assert "custom_cleaner.py" in result.output
            assert "custom_anonymizer.py" in result.output
            assert "custom_embedder.py" in result.output
            assert "custom_expert.py" in result.output
            assert "custom_aggregator.py" in result.output
            assert "weights" in result.output

    def test_init_short_help(self) -> None:
        """``-h`` shows the same help as ``--help`` for ``init``."""
        runner = CliRunner()
        long_h = runner.invoke(cli, ["init", "--help"])
        short_h = runner.invoke(cli, ["init", "-h"])
        assert long_h.exit_code == 0
        assert short_h.exit_code == 0
        assert long_h.output == short_h.output

    def test_output_shows_next_steps(self, tmp_path: Path) -> None:
        """The command output includes a 'Next steps' hint."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init", "proj"])
            assert "Next steps" in result.output


# ---------------------------------------------------------------------------
# apmoe serve
# ---------------------------------------------------------------------------


class TestServeCommand:
    """Tests for the ``apmoe serve`` command."""

    def test_help_output(self) -> None:
        """``--help`` exits 0 and lists the key options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--host" in result.output
        assert "--port" in result.output

    def test_missing_config_flag_exits_nonzero(self) -> None:
        """Invoking ``serve`` without ``--config`` exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve"])
        assert result.exit_code != 0

    def test_config_not_found_exits_nonzero(self, tmp_path: Path) -> None:
        """A non-existent config path causes Click to exit non-zero."""
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--config", str(tmp_path / "missing.json")])
        assert result.exit_code != 0

    def test_malformed_json_exits_nonzero(self, tmp_path: Path) -> None:
        """A config file containing invalid JSON causes a non-zero exit."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--config", str(bad)])
        assert result.exit_code != 0

    @patch("apmoe.core.app.APMoEApp")
    def test_calls_app_serve(self, mock_cls: MagicMock, tmp_path: Path) -> None:
        """``serve`` bootstraps the app and delegates to ``app.serve()``."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_instance = _make_mock_app()
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        runner.invoke(cli, ["serve", "--config", str(cfg_path)])

        mock_instance.serve.assert_called_once()

    @patch("apmoe.core.app.APMoEApp")
    def test_host_override_sets_env_var(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """``--host`` sets the ``APMOE_SERVING_HOST`` env var before bootstrap."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        captured_env: dict[str, str] = {}

        def capture(path: str) -> MagicMock:
            """Capture env at bootstrap time."""
            captured_env.update(os.environ)
            return _make_mock_app()

        mock_cls.from_config.side_effect = capture
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--config", str(cfg_path), "--host", "127.0.0.1"])
        assert captured_env.get("APMOE_SERVING_HOST") == "127.0.0.1"

    @patch("apmoe.core.app.APMoEApp")
    def test_port_override_sets_env_var(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """``--port`` sets the ``APMOE_SERVING_PORT`` env var before bootstrap."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        captured_env: dict[str, str] = {}

        def capture(path: str) -> MagicMock:
            """Capture env at bootstrap time."""
            captured_env.update(os.environ)
            return _make_mock_app()

        mock_cls.from_config.side_effect = capture
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--config", str(cfg_path), "--port", "9999"])
        assert captured_env.get("APMOE_SERVING_PORT") == "9999"

    @patch("apmoe.core.app.APMoEApp")
    def test_workers_override_sets_env_var(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """``--workers`` sets the ``APMOE_SERVING_WORKERS`` env var before bootstrap."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        captured_env: dict[str, str] = {}

        def capture(path: str) -> MagicMock:
            """Capture env at bootstrap time."""
            captured_env.update(os.environ)
            return _make_mock_app()

        mock_cls.from_config.side_effect = capture
        runner = CliRunner()
        runner.invoke(cli, ["serve", "--config", str(cfg_path), "--workers", "8"])
        assert captured_env.get("APMOE_SERVING_WORKERS") == "8"

    @patch("apmoe.core.app.APMoEApp")
    def test_apmoe_error_during_bootstrap_exits_nonzero(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """An :class:`APMoEError` raised during bootstrap exits non-zero."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_cls.from_config.side_effect = ConfigurationError("Cannot resolve class.")

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--config", str(cfg_path)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# apmoe predict
# ---------------------------------------------------------------------------


class TestPredictCommand:
    """Tests for the ``apmoe predict`` command."""

    def test_help_output(self) -> None:
        """``--help`` exits 0 and lists the key options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--input" in result.output
        assert "--output" in result.output

    def test_config_not_found_exits_nonzero(self, tmp_path: Path) -> None:
        """A non-existent ``--config`` path causes Click to exit non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["predict", "--config", str(tmp_path / "no.json"), "--input", str(tmp_path)],
        )
        assert result.exit_code != 0

    def test_no_matching_files_exits_nonzero(self, tmp_path: Path) -> None:
        """No files matching configured modalities in the input dir exits non-zero."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "unknown.txt").write_bytes(b"data")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--config", str(cfg_path), "--input", str(input_dir)]
        )
        assert result.exit_code != 0

    @patch("apmoe.core.app.APMoEApp")
    def test_matching_directory_file_forwarded(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A file named after a configured modality is forwarded to ``predict``."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "visual.jpg").write_bytes(b"\xff\xd8\xff")

        mock_instance = _make_mock_app()
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--config", str(cfg_path), "--input", str(input_dir)]
        )

        assert result.exit_code == 0
        mock_instance.predict.assert_called_once()
        call_inputs: dict[str, Any] = mock_instance.predict.call_args[0][0]
        assert "visual" in call_inputs
        assert call_inputs["visual"] == b"\xff\xd8\xff"

    @patch("apmoe.core.app.APMoEApp")
    def test_prediction_json_printed_to_stdout(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Without ``--output``, the prediction JSON is printed to stdout."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "visual.jpg").write_bytes(b"\xff\xd8")

        mock_instance = _make_mock_app(_make_prediction(age=27.5))
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--config", str(cfg_path), "--input", str(input_dir)]
        )

        assert result.exit_code == 0
        # The diagnostic line goes to stderr (mixed into output); extract the
        # JSON by slicing from the first '{' character.
        json_start = result.output.index("{")
        parsed = json.loads(result.output[json_start:])
        assert parsed["predicted_age"] == pytest.approx(27.5)

    @patch("apmoe.core.app.APMoEApp")
    def test_output_json_written_to_file(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """With ``--output``, the prediction JSON is written to the given file."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "visual.jpg").write_bytes(b"\xff\xd8")
        output_file = tmp_path / "result.json"

        mock_instance = _make_mock_app(_make_prediction(age=25.0))
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "predict",
                "--config",
                str(cfg_path),
                "--input",
                str(input_dir),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.is_file()
        parsed = json.loads(output_file.read_text(encoding="utf-8"))
        assert parsed["predicted_age"] == pytest.approx(25.0)

    @patch("apmoe.core.app.APMoEApp")
    def test_json_manifest_forwarded(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A ``.json`` manifest file is accepted and modality bytes forwarded."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        visual_file = tmp_path / "face.jpg"
        visual_file.write_bytes(b"\xff\xd8")
        manifest = {"visual": str(visual_file)}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        mock_instance = _make_mock_app()
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["predict", "--config", str(cfg_path), "--input", str(manifest_path)],
        )

        assert result.exit_code == 0
        call_inputs = mock_instance.predict.call_args[0][0]
        assert "visual" in call_inputs
        assert call_inputs["visual"] == b"\xff\xd8"

    @patch("apmoe.core.app.APMoEApp")
    def test_manifest_with_unknown_modality_warns_and_skips(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Manifest entries for unconfigured modalities are skipped with a warning."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        visual_file = tmp_path / "face.jpg"
        visual_file.write_bytes(b"\xff\xd8")
        manifest = {"visual": str(visual_file), "eeg": str(tmp_path / "eeg.bin")}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        mock_instance = _make_mock_app()
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["predict", "--config", str(cfg_path), "--input", str(manifest_path)],
        )

        assert result.exit_code == 0
        call_inputs = mock_instance.predict.call_args[0][0]
        assert "visual" in call_inputs
        assert "eeg" not in call_inputs

    @patch("apmoe.core.app.APMoEApp")
    def test_pipeline_error_exits_nonzero(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A :class:`PipelineError` during ``predict`` causes non-zero exit."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "visual.jpg").write_bytes(b"\xff\xd8")

        mock_instance = _make_mock_app()
        mock_instance.predict.side_effect = PipelineError("All experts skipped.")
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--config", str(cfg_path), "--input", str(input_dir)]
        )
        assert result.exit_code != 0

    @patch("apmoe.core.app.APMoEApp")
    def test_bootstrap_error_exits_nonzero(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """An :class:`APMoEError` during bootstrap exits non-zero."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "visual.jpg").write_bytes(b"\xff\xd8")

        mock_cls.from_config.side_effect = ConfigurationError("Cannot resolve expert.")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--config", str(cfg_path), "--input", str(input_dir)]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# apmoe validate
# ---------------------------------------------------------------------------


class TestValidateCommand:
    """Tests for the ``apmoe validate`` command."""

    def test_help_output(self) -> None:
        """``--help`` exits 0 and mentions ``--config``."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_missing_config_flag_exits_nonzero(self) -> None:
        """Invoking ``validate`` without ``--config`` exits non-zero."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate"])
        assert result.exit_code != 0

    def test_config_not_found_exits_nonzero(self, tmp_path: Path) -> None:
        """A non-existent config path causes Click to exit non-zero."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["validate", "--config", str(tmp_path / "nope.json")]
        )
        assert result.exit_code != 0

    def test_malformed_json_exits_nonzero(self, tmp_path: Path) -> None:
        """A config file with invalid JSON causes a non-zero exit."""
        bad = tmp_path / "bad.json"
        bad.write_text("{oops}", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(bad)])
        assert result.exit_code != 0

    @patch("apmoe.core.app.APMoEApp")
    def test_valid_config_exits_zero(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A valid config with all experts loaded exits 0 and prints a success message."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_cls.from_config.return_value = _make_mock_app()

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(cfg_path)])

        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    @patch("apmoe.core.app.APMoEApp")
    def test_bootstrap_failure_exits_nonzero(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """An error during bootstrap exits non-zero."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_cls.from_config.side_effect = ConfigurationError("Cannot resolve class.")

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(cfg_path)])
        assert result.exit_code != 0

    @patch("apmoe.core.app.APMoEApp")
    def test_validation_failure_exits_nonzero(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A failed ``app.validate()`` call exits non-zero."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_instance = _make_mock_app()
        mock_instance.validate.side_effect = ConfigurationError(
            "Weight file missing.",
            context={"issues": ["Weight file missing for expert 'face_expert'."]},
        )
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(cfg_path)])
        assert result.exit_code != 0

    @patch("apmoe.core.app.APMoEApp")
    def test_expert_health_displayed(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Expert names and their loaded status appear in the output."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_instance = _make_mock_app()
        mock_instance.validate.return_value = {
            "valid": True,
            "expert_health": {"face_expert": True, "audio_expert": False},
            "issues": [],
        }
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(cfg_path)])
        assert "face_expert" in result.output
        assert "audio_expert" in result.output

    @patch("apmoe.core.app.APMoEApp")
    def test_no_experts_shows_placeholder(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """An empty expert registry causes a placeholder message to appear."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_instance = _make_mock_app()
        mock_instance.validate.return_value = {
            "valid": True,
            "expert_health": {},
            "issues": [],
        }
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(cfg_path)])
        assert "no experts" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI group — --help and --version
# ---------------------------------------------------------------------------


class TestCLIGroup:
    """Tests for the top-level ``apmoe`` CLI group."""

    def test_main_help(self) -> None:
        """``apmoe --help`` exits 0 and lists all sub-commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "serve" in result.output
        assert "predict" in result.output
        assert "validate" in result.output

    def test_main_short_help(self) -> None:
        """``apmoe -h`` matches ``apmoe --help``."""
        runner = CliRunner()
        long_h = runner.invoke(cli, ["--help"])
        short_h = runner.invoke(cli, ["-h"])
        assert long_h.exit_code == 0
        assert short_h.exit_code == 0
        assert long_h.output == short_h.output

    def test_version_flag(self) -> None:
        """``apmoe --version`` exits 0 and prints the version string."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_no_args_shows_usage(self) -> None:
        """Invoking the CLI with no arguments shows the usage / help text."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # Click may exit with 0 or 2 depending on version; either is acceptable.
        assert "Usage" in result.output or "Commands" in result.output

    def test_unknown_command_exits_nonzero(self) -> None:
        """An unknown sub-command name causes a non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge-case tests."""

    @patch("apmoe.core.app.APMoEApp")
    def test_predict_multiple_modality_files_all_forwarded(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """All matching modality files in a directory are forwarded."""
        cfg_data = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": "p.VP",
                        "pipeline": {"cleaner": "p.C", "anonymizer": "p.A"},
                    },
                    {
                        "name": "audio",
                        "processor": "p.AP",
                        "pipeline": {"cleaner": "p.C", "anonymizer": "p.A"},
                    },
                ],
                "experts": [
                    {
                        "name": "e",
                        "class": "p.E",
                        "weights": "./w.pt",
                        "modalities": ["visual", "audio"],
                    }
                ],
                "aggregation": {"strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"},
            }
        }
        cfg_path = _write_config(tmp_path / "config.json", cfg_data)
        input_dir = tmp_path / "data"
        input_dir.mkdir()
        (input_dir / "visual.jpg").write_bytes(b"\xff\xd8")
        (input_dir / "audio.wav").write_bytes(b"RIFF")
        (input_dir / "ignored.txt").write_bytes(b"ignored")

        mock_instance = _make_mock_app()
        mock_cls.from_config.return_value = mock_instance

        runner = CliRunner()
        result = runner.invoke(
            cli, ["predict", "--config", str(cfg_path), "--input", str(input_dir)]
        )
        assert result.exit_code == 0
        call_inputs = mock_instance.predict.call_args[0][0]
        assert "visual" in call_inputs
        assert "audio" in call_inputs
        assert "ignored" not in call_inputs

    @patch("apmoe.core.app.APMoEApp")
    def test_apmoe_error_subclass_caught_in_serve(
        self, mock_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Any :class:`APMoEError` subclass raised during bootstrap exits non-zero."""
        cfg_path = _write_config(tmp_path / "config.json", _minimal_config_data())
        mock_cls.from_config.side_effect = APMoEError("Generic framework error.")

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--config", str(cfg_path)])
        assert result.exit_code != 0
