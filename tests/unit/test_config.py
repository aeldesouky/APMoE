"""Unit tests for apmoe.core.config (config loading and validation)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from apmoe.core.config import (
    APMoEConfig,
    ExpertConfig,
    FrameworkConfig,
    ModalityConfig,
    ServingConfig,
    load_config,
)
from apmoe.core.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_config(path: Path, data: dict[str, Any]) -> Path:
    """Write *data* as JSON to *path* and return *path*."""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _base_config() -> dict[str, Any]:
    return {
        "apmoe": {
            "modalities": [
                {
                    "name": "visual",
                    "processor": "myproject.VisualProcessor",
                    "pipeline": {
                        "cleaner": "myproject.ImageCleaner",
                        "anonymizer": "myproject.FaceAnonymizer",
                    },
                }
            ],
            "experts": [
                {
                    "name": "face_expert",
                    "class": "myproject.FaceExpert",
                    "weights": "./weights/face.pt",
                    "modalities": ["visual"],
                }
            ],
            "aggregation": {"strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"},
        }
    }


# ---------------------------------------------------------------------------
# load_config: file I/O
# ---------------------------------------------------------------------------


class TestLoadConfigFileIO:
    def test_valid_file_loads_successfully(self, minimal_config_file: Path) -> None:
        cfg = load_config(minimal_config_file)
        assert isinstance(cfg, FrameworkConfig)

    def test_missing_file_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "nonexistent.json")

    def test_malformed_json_raises_configuration_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Malformed JSON"):
            load_config(bad)

    def test_accepts_string_path(self, minimal_config_file: Path) -> None:
        cfg = load_config(str(minimal_config_file))
        assert isinstance(cfg, FrameworkConfig)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_missing_modalities_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        del data["apmoe"]["modalities"]
        with pytest.raises(ConfigurationError):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_missing_experts_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        del data["apmoe"]["experts"]
        with pytest.raises(ConfigurationError):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_missing_aggregation_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        del data["apmoe"]["aggregation"]
        with pytest.raises(ConfigurationError):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_expert_references_unknown_modality(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["experts"][0]["modalities"] = ["visual", "audio"]
        with pytest.raises(ConfigurationError, match="undeclared modalities"):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_duplicate_modality_names_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["modalities"].append(data["apmoe"]["modalities"][0].copy())
        with pytest.raises(ConfigurationError, match="Duplicate modality name"):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_duplicate_expert_names_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["experts"].append(data["apmoe"]["experts"][0].copy())
        with pytest.raises(ConfigurationError, match="Duplicate expert name"):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_empty_modalities_list_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["modalities"] = []
        with pytest.raises(ConfigurationError):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_expert_with_no_modalities_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["experts"][0]["modalities"] = []
        with pytest.raises(ConfigurationError, match="at least one modality"):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_invalid_port_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["serving"] = {"port": 99999}
        with pytest.raises(ConfigurationError, match="65535"):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_invalid_workers_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["serving"] = {"workers": 0}
        with pytest.raises(ConfigurationError):
            load_config(_write_config(tmp_path / "c.json", data))

    def test_modality_name_empty_string_raises(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["modalities"][0]["name"] = "   "
        with pytest.raises(ConfigurationError):
            load_config(_write_config(tmp_path / "c.json", data))


# ---------------------------------------------------------------------------
# Parsed values
# ---------------------------------------------------------------------------


class TestParsedValues:
    def test_modality_fields(self, minimal_config_file: Path) -> None:
        cfg = load_config(minimal_config_file)
        modality = cfg.apmoe.modalities[0]
        assert modality.name == "visual"
        assert modality.pipeline.embedder is None  # not specified in minimal config

    def test_expert_fields(self, minimal_config_file: Path) -> None:
        cfg = load_config(minimal_config_file)
        expert = cfg.apmoe.experts[0]
        assert expert.name == "face_expert"
        assert expert.class_path == "myproject.experts.FaceExpert"
        assert expert.modalities == ["visual"]

    def test_serving_defaults(self, minimal_config_file: Path) -> None:
        cfg = load_config(minimal_config_file)
        serving = cfg.apmoe.serving
        assert serving.host == "0.0.0.0"
        assert serving.port == 8000
        assert serving.workers == 4

    def test_full_config_multi_modality(self, full_config_file: Path) -> None:
        cfg = load_config(full_config_file)
        assert len(cfg.apmoe.modalities) == 2
        assert len(cfg.apmoe.experts) == 3
        assert cfg.apmoe.serving.port == 9000

    def test_embedder_optional_present(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["modalities"][0]["pipeline"]["embedder"] = "myproject.Embedder"
        cfg = load_config(_write_config(tmp_path / "c.json", data))
        assert cfg.apmoe.modalities[0].pipeline.embedder == "myproject.Embedder"

    def test_aggregation_weights(self, tmp_path: Path) -> None:
        data = _base_config()
        data["apmoe"]["aggregation"]["weights"] = {"face_expert": 1.0}
        cfg = load_config(_write_config(tmp_path / "c.json", data))
        assert cfg.apmoe.aggregation.weights == {"face_expert": 1.0}


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    def test_port_override(
        self, minimal_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APMOE_SERVING_PORT", "9999")
        cfg = load_config(minimal_config_file)
        assert cfg.apmoe.serving.port == 9999

    def test_host_override(
        self, minimal_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APMOE_SERVING_HOST", "127.0.0.1")
        cfg = load_config(minimal_config_file)
        assert cfg.apmoe.serving.host == "127.0.0.1"

    def test_workers_override(
        self, minimal_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APMOE_SERVING_WORKERS", "8")
        cfg = load_config(minimal_config_file)
        assert cfg.apmoe.serving.workers == 8

    def test_cors_origins_override(
        self, minimal_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APMOE_SERVING_CORS_ORIGINS", "https://a.com,https://b.com")
        cfg = load_config(minimal_config_file)
        assert cfg.apmoe.serving.cors_origins == ["https://a.com", "https://b.com"]

    def test_invalid_port_env_var_raises(
        self, minimal_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APMOE_SERVING_PORT", "not_a_number")
        with pytest.raises(ConfigurationError):
            load_config(minimal_config_file)

    def test_env_var_port_validates_range(
        self, minimal_config_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("APMOE_SERVING_PORT", "99999")
        with pytest.raises(ConfigurationError):
            load_config(minimal_config_file)
