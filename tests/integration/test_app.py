"""Integration tests for APMoEApp and the full end-to-end pipeline.

These tests exercise the complete bootstrap flow — from a JSON config file
through component resolution, weight loading, and inference — using
lightweight mock components registered in the global registries.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.app import APMoEApp
from apmoe.core.exceptions import ConfigurationError, ExpertError, PipelineError
from apmoe.core.types import (
    EmbeddingResult,
    ExpertOutput,
    ModalityData,
    Prediction,
    ProcessedInput,
)
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import modality_registry
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    EmbedderStrategy,
    anonymizer_registry,
    cleaner_registry,
    embedder_registry,
)


# ---------------------------------------------------------------------------
# Minimal concrete components for integration
# ---------------------------------------------------------------------------


class _IntegrationVisualProcessor(ModalityProcessor):
    @property
    def modality_name(self) -> str:
        return "visual"

    def validate(self, data: object) -> bool:
        return data is not None

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality="visual", data=data, metadata={"source": "integration"})


class _IntegrationAudioProcessor(ModalityProcessor):
    @property
    def modality_name(self) -> str:
        return "audio"

    def validate(self, data: object) -> bool:
        return data is not None

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality="audio", data=data)


class _IntegrationCleaner(CleanerStrategy):
    def clean(self, data: ModalityData) -> ModalityData:
        return data


class _IntegrationAnonymizer(AnonymizerStrategy):
    def anonymize(self, data: ModalityData) -> ModalityData:
        return data


class _IntegrationEmbedder(EmbedderStrategy):
    def embed(self, data: ModalityData) -> EmbeddingResult:
        return EmbeddingResult(
            modality=data.modality,
            embedding=np.array([0.1, 0.2, 0.3]),
        )


class _IntegrationVisualExpert(ExpertPlugin):
    def __init__(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return "visual_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        # Accept any path (dummy weights in tests)
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["visual"],
            predicted_age=32.0,
            confidence=0.9,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _IntegrationAudioExpert(ExpertPlugin):
    def __init__(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return "audio_expert"

    def declared_modalities(self) -> list[str]:
        return ["audio"]

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["audio"],
            predicted_age=28.0,
            confidence=0.7,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _IntegrationAggregator(AggregatorStrategy):
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        age = sum(o.predicted_age for o in outputs) / len(outputs)
        conf = sum(o.confidence for o in outputs) / len(outputs)
        return Prediction(
            predicted_age=age,
            confidence=min(conf, 1.0),
            per_expert_outputs=list(outputs),
        )


# ---------------------------------------------------------------------------
# Registry setup fixtures
# ---------------------------------------------------------------------------

# Short names used inside JSON configs
_VISUAL_PROC = "test.integration.VisualProcessor"
_AUDIO_PROC = "test.integration.AudioProcessor"
_CLEANER = "test.integration.Cleaner"
_ANONYMIZER = "test.integration.Anonymizer"
_EMBEDDER = "test.integration.Embedder"
_VISUAL_EXPERT = "test.integration.VisualExpert"
_AUDIO_EXPERT = "test.integration.AudioExpert"
_AGGREGATOR = "test.integration.Aggregator"


@pytest.fixture(autouse=True)
def _register_integration_components() -> None:
    """Register all test doubles in their respective global registries."""
    # Use overwrite=True so multiple tests can re-register without conflicts.
    modality_registry.register_class(_VISUAL_PROC, _IntegrationVisualProcessor, overwrite=True)
    modality_registry.register_class(_AUDIO_PROC, _IntegrationAudioProcessor, overwrite=True)
    cleaner_registry.register_class(_CLEANER, _IntegrationCleaner, overwrite=True)
    anonymizer_registry.register_class(_ANONYMIZER, _IntegrationAnonymizer, overwrite=True)
    embedder_registry.register_class(_EMBEDDER, _IntegrationEmbedder, overwrite=True)
    expert_registry.register_class(_VISUAL_EXPERT, _IntegrationVisualExpert, overwrite=True)
    expert_registry.register_class(_AUDIO_EXPERT, _IntegrationAudioExpert, overwrite=True)
    aggregator_registry.register_class(_AGGREGATOR, _IntegrationAggregator, overwrite=True)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _minimal_app_config(
    tmp_path: Path,
    *,
    include_audio: bool = False,
    include_embedder: bool = False,
    weights_dir: Path | None = None,
) -> Path:
    """Write a minimal valid config JSON and return its path."""
    weights_dir = weights_dir or tmp_path
    # Create dummy weight files
    visual_weights = weights_dir / "visual.pt"
    visual_weights.write_bytes(b"dummy")

    modalities: list[dict[str, Any]] = [
        {
            "name": "visual",
            "processor": _VISUAL_PROC,
            "pipeline": {
                "cleaner": _CLEANER,
                "anonymizer": _ANONYMIZER,
                **({"embedder": _EMBEDDER} if include_embedder else {}),
            },
        }
    ]
    experts: list[dict[str, Any]] = [
        {
            "name": "visual_expert",
            "class": _VISUAL_EXPERT,
            "weights": str(visual_weights),
            "modalities": ["visual"],
        }
    ]

    if include_audio:
        audio_weights = weights_dir / "audio.pt"
        audio_weights.write_bytes(b"dummy")
        modalities.append(
            {
                "name": "audio",
                "processor": _AUDIO_PROC,
                "pipeline": {"cleaner": _CLEANER, "anonymizer": _ANONYMIZER},
            }
        )
        experts.append(
            {
                "name": "audio_expert",
                "class": _AUDIO_EXPERT,
                "weights": str(audio_weights),
                "modalities": ["audio"],
            }
        )

    config: dict[str, Any] = {
        "apmoe": {
            "modalities": modalities,
            "experts": experts,
            "aggregation": {"strategy": _AGGREGATOR},
            "serving": {"host": "127.0.0.1", "port": 9000, "workers": 1},
        }
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    return cfg_path


# ---------------------------------------------------------------------------
# APMoEApp.from_config() — bootstrap tests
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_from_config_returns_app_instance(self, tmp_path: Path) -> None:
        cfg_path = _minimal_app_config(tmp_path)
        app = APMoEApp.from_config(cfg_path)
        assert isinstance(app, APMoEApp)

    def test_from_config_string_path(self, tmp_path: Path) -> None:
        cfg_path = _minimal_app_config(tmp_path)
        app = APMoEApp.from_config(str(cfg_path))
        assert isinstance(app, APMoEApp)

    def test_from_config_experts_loaded(self, tmp_path: Path) -> None:
        cfg_path = _minimal_app_config(tmp_path)
        app = APMoEApp.from_config(cfg_path)
        assert app.expert_registry.all_healthy() is True

    def test_from_config_multi_modality(self, tmp_path: Path) -> None:
        cfg_path = _minimal_app_config(tmp_path, include_audio=True)
        app = APMoEApp.from_config(cfg_path)
        assert len(app.expert_registry) == 2

    def test_from_config_with_embedder(self, tmp_path: Path) -> None:
        cfg_path = _minimal_app_config(tmp_path, include_embedder=True)
        app = APMoEApp.from_config(cfg_path)
        assert isinstance(app, APMoEApp)

    def test_from_config_missing_file_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            APMoEApp.from_config("/nonexistent/path/config.json")

    def test_from_config_unknown_cleaner_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"dummy")
        config: dict[str, Any] = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": _VISUAL_PROC,
                        "pipeline": {
                            "cleaner": "no.such.Cleaner",
                            "anonymizer": _ANONYMIZER,
                        },
                    }
                ],
                "experts": [
                    {
                        "name": "visual_expert",
                        "class": _VISUAL_EXPERT,
                        "weights": str(weights),
                        "modalities": ["visual"],
                    }
                ],
                "aggregation": {"strategy": _AGGREGATOR},
            }
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))
        with pytest.raises(ConfigurationError, match="cleaner"):
            APMoEApp.from_config(cfg_path)

    def test_from_config_unknown_anonymizer_raises(self, tmp_path: Path) -> None:
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"dummy")
        config: dict[str, Any] = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": _VISUAL_PROC,
                        "pipeline": {
                            "cleaner": _CLEANER,
                            "anonymizer": "no.such.Anonymizer",
                        },
                    }
                ],
                "experts": [
                    {
                        "name": "ve",
                        "class": _VISUAL_EXPERT,
                        "weights": str(weights),
                        "modalities": ["visual"],
                    }
                ],
                "aggregation": {"strategy": _AGGREGATOR},
            }
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))
        with pytest.raises(ConfigurationError, match="anonymizer"):
            APMoEApp.from_config(cfg_path)

    def test_from_config_unknown_embedder_raises(self, tmp_path: Path) -> None:
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"dummy")
        config: dict[str, Any] = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": _VISUAL_PROC,
                        "pipeline": {
                            "cleaner": _CLEANER,
                            "anonymizer": _ANONYMIZER,
                            "embedder": "no.such.Embedder",
                        },
                    }
                ],
                "experts": [
                    {
                        "name": "ve",
                        "class": _VISUAL_EXPERT,
                        "weights": str(weights),
                        "modalities": ["visual"],
                    }
                ],
                "aggregation": {"strategy": _AGGREGATOR},
            }
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))
        with pytest.raises(ConfigurationError, match="embedder"):
            APMoEApp.from_config(cfg_path)

    def test_from_config_unknown_aggregator_raises(self, tmp_path: Path) -> None:
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"dummy")
        config: dict[str, Any] = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": _VISUAL_PROC,
                        "pipeline": {"cleaner": _CLEANER, "anonymizer": _ANONYMIZER},
                    }
                ],
                "experts": [
                    {
                        "name": "ve",
                        "class": _VISUAL_EXPERT,
                        "weights": str(weights),
                        "modalities": ["visual"],
                    }
                ],
                "aggregation": {"strategy": "no.such.Aggregator"},
            }
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))
        with pytest.raises(ConfigurationError, match="aggregator"):
            APMoEApp.from_config(cfg_path)

    def test_from_config_unknown_expert_class_raises(self, tmp_path: Path) -> None:
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"dummy")
        config: dict[str, Any] = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": _VISUAL_PROC,
                        "pipeline": {"cleaner": _CLEANER, "anonymizer": _ANONYMIZER},
                    }
                ],
                "experts": [
                    {
                        "name": "ve",
                        "class": "no.such.Expert",
                        "weights": str(weights),
                        "modalities": ["visual"],
                    }
                ],
                "aggregation": {"strategy": _AGGREGATOR},
            }
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))
        with pytest.raises(ExpertError, match="Cannot resolve expert class"):
            APMoEApp.from_config(cfg_path)


# ---------------------------------------------------------------------------
# app.predict() — end-to-end inference
# ---------------------------------------------------------------------------


class TestAppPredict:
    def test_predict_single_modality_returns_prediction(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        prediction = app.predict({"visual": b"fake_image"})
        assert isinstance(prediction, Prediction)

    def test_predict_age_matches_expert_output(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        prediction = app.predict({"visual": b"fake_image"})
        # _IntegrationVisualExpert always returns 32.0
        assert prediction.predicted_age == pytest.approx(32.0)

    def test_predict_with_embedder_produces_embedding_result_in_pipeline(
        self, tmp_path: Path
    ) -> None:
        """When an embedder is configured, the expert receives EmbeddingResult."""
        received_inputs: dict[str, ProcessedInput] = {}

        class _SpyExpert(ExpertPlugin):
            def __init__(self) -> None:
                self._loaded = False

            @property
            def name(self) -> str:
                return "visual_expert"

            def declared_modalities(self) -> list[str]:
                return ["visual"]

            def load_weights(self, path: str) -> None:
                self._loaded = True

            def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
                received_inputs.update(inputs)
                return ExpertOutput("visual_expert", ["visual"], 25.0, 0.8)

            @property
            def is_loaded(self) -> bool:
                return self._loaded

        expert_registry.register_class("_spy_expert", _SpyExpert, overwrite=True)

        weights = tmp_path / "w.pt"
        weights.write_bytes(b"dummy")
        config: dict[str, Any] = {
            "apmoe": {
                "modalities": [
                    {
                        "name": "visual",
                        "processor": _VISUAL_PROC,
                        "pipeline": {
                            "cleaner": _CLEANER,
                            "anonymizer": _ANONYMIZER,
                            "embedder": _EMBEDDER,
                        },
                    }
                ],
                "experts": [
                    {
                        "name": "visual_expert",
                        "class": "_spy_expert",
                        "weights": str(weights),
                        "modalities": ["visual"],
                    }
                ],
                "aggregation": {"strategy": _AGGREGATOR},
            }
        }
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps(config))
        app = APMoEApp.from_config(cfg_path)
        app.predict({"visual": b"img"})

        assert "visual" in received_inputs
        assert isinstance(received_inputs["visual"], EmbeddingResult)

    def test_predict_multi_modality_averages_experts(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path, include_audio=True))
        prediction = app.predict({"visual": b"img", "audio": b"wav"})
        # visual=32.0, audio=28.0 → average=30.0
        assert prediction.predicted_age == pytest.approx(30.0)

    def test_predict_missing_modality_skips_expert(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path, include_audio=True))
        # Only provide visual input; audio expert should be skipped
        prediction = app.predict({"visual": b"img"})
        assert "audio_expert" in prediction.skipped_experts
        assert prediction.predicted_age == pytest.approx(32.0)

    def test_predict_no_valid_modality_raises_pipeline_error(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        with pytest.raises(PipelineError):
            app.predict({})

    def test_predict_metadata_contains_latency(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        prediction = app.predict({"visual": b"img"})
        assert "pipeline_latency_s" in prediction.metadata
        assert prediction.metadata["pipeline_latency_s"] >= 0

    def test_predict_per_expert_outputs_populated(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        prediction = app.predict({"visual": b"img"})
        assert len(prediction.per_expert_outputs) == 1
        assert prediction.per_expert_outputs[0].expert_name == "visual_expert"


# ---------------------------------------------------------------------------
# app.predict_async()
# ---------------------------------------------------------------------------


class TestAppPredictAsync:
    def test_predict_async_returns_prediction(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        prediction = asyncio.run(app.predict_async({"visual": b"img"}))
        assert isinstance(prediction, Prediction)
        assert prediction.predicted_age == pytest.approx(32.0)

    def test_predict_async_multi_modality(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path, include_audio=True))
        prediction = asyncio.run(app.predict_async({"visual": b"img", "audio": b"wav"}))
        assert prediction.predicted_age == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# app.validate()
# ---------------------------------------------------------------------------


class TestAppValidate:
    def test_validate_passes_when_all_healthy(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        report = app.validate()
        assert report["valid"] is True
        assert report["issues"] == []

    def test_validate_fails_when_weight_file_deleted(self, tmp_path: Path) -> None:
        cfg_path = _minimal_app_config(tmp_path)
        app = APMoEApp.from_config(cfg_path)

        # Delete a weight file after loading
        for expert_cfg in app.config.apmoe.experts:
            Path(expert_cfg.weights).unlink(missing_ok=True)

        with pytest.raises(ConfigurationError, match="missing"):
            app.validate()

    def test_validate_expert_health_in_report(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        report = app.validate()
        assert "expert_health" in report
        assert report["expert_health"]["visual_expert"] is True


# ---------------------------------------------------------------------------
# app.get_info()
# ---------------------------------------------------------------------------


class TestAppGetInfo:
    def test_get_info_contains_version(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        info = app.get_info()
        assert "version" in info
        assert isinstance(info["version"], str)

    def test_get_info_contains_modalities(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        info = app.get_info()
        assert "visual" in info["modalities"]

    def test_get_info_contains_expert_list(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        info = app.get_info()
        expert_names = [e["name"] for e in info["experts"]]
        assert "visual_expert" in expert_names

    def test_get_info_contains_aggregator(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        info = app.get_info()
        assert "aggregator" in info
        assert "aggregator_class" in info["aggregator"]

    def test_get_info_contains_serving(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        info = app.get_info()
        assert "serving" in info
        assert info["serving"]["port"] == 9000


# ---------------------------------------------------------------------------
# app repr
# ---------------------------------------------------------------------------


class TestAppRepr:
    def test_repr_contains_key_info(self, tmp_path: Path) -> None:
        app = APMoEApp.from_config(_minimal_app_config(tmp_path))
        r = repr(app)
        assert "APMoEApp" in r
        assert "visual" in r
        assert "visual_expert" in r
