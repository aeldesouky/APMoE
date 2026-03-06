"""Unit tests for apmoe.experts.base and apmoe.experts.registry."""

from __future__ import annotations

import pytest

from apmoe.core.config import ExpertConfig, ModalityConfig, PipelineConfig
from apmoe.core.exceptions import ExpertError
from apmoe.core.types import EmbeddingResult, ExpertOutput, ModalityData, ProcessedInput
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import ExpertRegistry, expert_registry


# ---------------------------------------------------------------------------
# Concrete helper experts
# ---------------------------------------------------------------------------


class _VisualExpert(ExpertPlugin):
    """Single-modality expert that consumes 'visual' data."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return "visual_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        # Accept any path for testing; just mark as loaded
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["visual"],
            predicted_age=30.0,
            confidence=0.9,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _AudioExpert(ExpertPlugin):
    """Single-modality expert that consumes 'audio' data."""

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


class _MultiModalExpert(ExpertPlugin):
    """Multi-modal expert consuming 'visual' and 'audio'."""

    def __init__(self) -> None:
        self._loaded = False

    @property
    def name(self) -> str:
        return "multimodal_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual", "audio"]

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=list(inputs.keys()),
            predicted_age=29.0,
            confidence=0.85,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# ExpertPlugin ABC
# ---------------------------------------------------------------------------


class TestExpertPluginABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ExpertPlugin()  # type: ignore[abstract]

    def test_missing_name_raises(self) -> None:
        class _Bad(ExpertPlugin):
            def declared_modalities(self) -> list[str]:
                return ["visual"]

            def load_weights(self, path: str) -> None:
                pass

            def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
                return ExpertOutput("x", ["visual"], 0.0, 0.5)

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_missing_declared_modalities_raises(self) -> None:
        class _Bad(ExpertPlugin):
            @property
            def name(self) -> str:
                return "x"

            def load_weights(self, path: str) -> None:
                pass

            def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
                return ExpertOutput("x", [], 0.0, 0.5)

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_missing_load_weights_raises(self) -> None:
        class _Bad(ExpertPlugin):
            @property
            def name(self) -> str:
                return "x"

            def declared_modalities(self) -> list[str]:
                return ["visual"]

            def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
                return ExpertOutput("x", ["visual"], 0.0, 0.5)

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_missing_predict_raises(self) -> None:
        class _Bad(ExpertPlugin):
            @property
            def name(self) -> str:
                return "x"

            def declared_modalities(self) -> list[str]:
                return ["visual"]

            def load_weights(self, path: str) -> None:
                pass

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        expert = _VisualExpert()
        assert expert.name == "visual_expert"

    def test_declared_modalities_single(self) -> None:
        expert = _VisualExpert()
        assert expert.declared_modalities() == ["visual"]

    def test_declared_modalities_multi(self) -> None:
        expert = _MultiModalExpert()
        assert expert.declared_modalities() == ["visual", "audio"]

    def test_load_weights_sets_loaded(self) -> None:
        expert = _VisualExpert()
        assert expert.is_loaded is False
        expert.load_weights("any/path.pt")
        assert expert.is_loaded is True

    def test_predict_returns_expert_output(self) -> None:
        expert = _VisualExpert()
        result = expert.predict({"visual": ModalityData(modality="visual", data=b"x")})
        assert isinstance(result, ExpertOutput)
        assert result.expert_name == "visual_expert"
        assert result.predicted_age == 30.0
        assert result.confidence == 0.9

    def test_get_info_default_contains_name_and_modalities(self) -> None:
        expert = _VisualExpert()
        info = expert.get_info()
        assert info["name"] == "visual_expert"
        assert info["modalities"] == ["visual"]
        assert "_VisualExpert" in str(info["expert_class"])

    def test_is_loaded_default_true(self) -> None:
        """The default is_loaded returns True; concrete override may differ."""

        class _Simple(ExpertPlugin):
            @property
            def name(self) -> str:
                return "simple"

            def declared_modalities(self) -> list[str]:
                return ["visual"]

            def load_weights(self, path: str) -> None:
                pass

            def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
                return ExpertOutput("simple", ["visual"], 25.0, 0.5)

        assert _Simple().is_loaded is True


# ---------------------------------------------------------------------------
# ExpertRegistry — registration & lookup
# ---------------------------------------------------------------------------


class TestExpertRegistryRegistration:
    def setup_method(self) -> None:
        self.registry = ExpertRegistry()
        self.visual = _VisualExpert()
        self.visual.load_weights("fake.pt")

    def test_register_and_get(self) -> None:
        self.registry.register_instance(self.visual)
        assert self.registry.get("visual_expert") is self.visual

    def test_duplicate_registration_raises(self) -> None:
        self.registry.register_instance(self.visual)
        with pytest.raises(ExpertError, match="already registered"):
            self.registry.register_instance(self.visual)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(ExpertError, match="No expert named"):
            self.registry.get("nonexistent")

    def test_list_experts_empty(self) -> None:
        assert self.registry.list_experts() == []

    def test_list_experts_sorted(self) -> None:
        audio = _AudioExpert()
        audio.load_weights("fake.pt")
        self.registry.register_instance(audio)
        self.registry.register_instance(self.visual)
        assert self.registry.list_experts() == ["audio_expert", "visual_expert"]

    def test_all_instances_order(self) -> None:
        self.registry.register_instance(self.visual)
        instances = self.registry.all_instances()
        assert len(instances) == 1
        assert instances[0] is self.visual

    def test_len(self) -> None:
        assert len(self.registry) == 0
        self.registry.register_instance(self.visual)
        assert len(self.registry) == 1

    def test_contains(self) -> None:
        self.registry.register_instance(self.visual)
        assert "visual_expert" in self.registry
        assert "audio_expert" not in self.registry

    def test_repr(self) -> None:
        self.registry.register_instance(self.visual)
        r = repr(self.registry)
        assert "visual_expert" in r


# ---------------------------------------------------------------------------
# ExpertRegistry — modality dispatch
# ---------------------------------------------------------------------------


class TestExpertRegistryDispatch:
    def setup_method(self) -> None:
        self.registry = ExpertRegistry()
        self.visual = _VisualExpert()
        self.visual.load_weights("fake.pt")
        self.audio = _AudioExpert()
        self.audio.load_weights("fake.pt")
        self.multi = _MultiModalExpert()
        self.multi.load_weights("fake.pt")

        self.registry.register_instance(self.visual)
        self.registry.register_instance(self.audio)
        self.registry.register_instance(self.multi)

    def test_all_modalities_available_returns_all(self) -> None:
        runnable = self.registry.get_runnable_experts({"visual", "audio"})
        names = {e.name for e in runnable}
        assert names == {"visual_expert", "audio_expert", "multimodal_expert"}

    def test_only_visual_available_skips_audio_and_multi(self) -> None:
        runnable = self.registry.get_runnable_experts({"visual"})
        names = {e.name for e in runnable}
        assert names == {"visual_expert"}

    def test_only_audio_available(self) -> None:
        runnable = self.registry.get_runnable_experts({"audio"})
        names = {e.name for e in runnable}
        assert names == {"audio_expert"}

    def test_no_modalities_runs_nothing(self) -> None:
        runnable = self.registry.get_runnable_experts(set())
        assert runnable == []

    def test_skipped_experts_with_only_visual(self) -> None:
        skipped = self.registry.get_skipped_experts({"visual"})
        assert "audio_expert" in skipped
        assert "multimodal_expert" in skipped
        assert "visual_expert" not in skipped

    def test_skipped_experts_none_when_all_available(self) -> None:
        skipped = self.registry.get_skipped_experts({"visual", "audio"})
        assert skipped == []


# ---------------------------------------------------------------------------
# ExpertRegistry — health check
# ---------------------------------------------------------------------------


class TestExpertRegistryHealthCheck:
    def setup_method(self) -> None:
        self.registry = ExpertRegistry()

    def test_health_check_all_loaded(self) -> None:
        v = _VisualExpert()
        v.load_weights("fake.pt")
        self.registry.register_instance(v)
        health = self.registry.health_check()
        assert health["visual_expert"] is True
        assert self.registry.all_healthy() is True

    def test_health_check_unloaded_expert(self) -> None:
        v = _VisualExpert()  # NOT calling load_weights
        self.registry.register_instance(v)
        assert self.registry.health_check()["visual_expert"] is False
        assert self.registry.all_healthy() is False

    def test_all_healthy_empty_registry(self) -> None:
        """Empty registry is vacuously healthy."""
        assert self.registry.all_healthy() is True


# ---------------------------------------------------------------------------
# ExpertRegistry — validate_expert_modalities
# ---------------------------------------------------------------------------


def _make_expert_config(name: str, class_path: str, modalities: list[str]) -> ExpertConfig:
    return ExpertConfig(name=name, **{"class": class_path}, weights="w.pt", modalities=modalities)


def _make_modality_config(name: str) -> ModalityConfig:
    return ModalityConfig(
        name=name,
        processor="dummy.Processor",
        pipeline=PipelineConfig(cleaner="dummy.Cleaner", anonymizer="dummy.Anonymizer"),
    )


class TestExpertRegistryValidation:
    def test_valid_modalities_pass(self) -> None:
        experts = [_make_expert_config("e1", "dummy.Class", ["visual", "audio"])]
        modalities = [_make_modality_config("visual"), _make_modality_config("audio")]
        ExpertRegistry.validate_expert_modalities(experts, modalities)  # no exception

    def test_undeclared_modality_raises(self) -> None:
        experts = [_make_expert_config("e1", "dummy.Class", ["eeg"])]
        modalities = [_make_modality_config("visual")]
        with pytest.raises(ExpertError, match="undeclared modalities"):
            ExpertRegistry.validate_expert_modalities(experts, modalities)

    def test_empty_experts_list_passes(self) -> None:
        ExpertRegistry.validate_expert_modalities([], [_make_modality_config("visual")])

    def test_empty_modalities_with_expert_raises(self) -> None:
        experts = [_make_expert_config("e1", "dummy.Class", ["visual"])]
        with pytest.raises(ExpertError):
            ExpertRegistry.validate_expert_modalities(experts, [])


# ---------------------------------------------------------------------------
# ExpertRegistry — from_configs (bootstrap factory)
# ---------------------------------------------------------------------------


class TestExpertRegistryFromConfigs:
    def setup_method(self) -> None:
        expert_registry.register_class(
            "_test_visual_expert", _VisualExpert, overwrite=True
        )
        expert_registry.register_class(
            "_test_audio_expert", _AudioExpert, overwrite=True
        )

    def test_from_configs_creates_loaded_instances(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # Create dummy weight files so load_weights() doesn't fail
        # (our stub accepts any path)
        expert_cfgs = [
            _make_expert_config("visual_expert", "_test_visual_expert", ["visual"]),
            _make_expert_config("audio_expert", "_test_audio_expert", ["audio"]),
        ]
        modality_cfgs = [
            _make_modality_config("visual"),
            _make_modality_config("audio"),
        ]
        registry = ExpertRegistry.from_configs(expert_cfgs, modality_cfgs)
        assert len(registry) == 2
        assert registry.all_healthy() is True

    def test_from_configs_unknown_class_raises(self) -> None:
        expert_cfgs = [
            _make_expert_config("e1", "nonexistent.Module.Expert", ["visual"])
        ]
        modality_cfgs = [_make_modality_config("visual")]
        with pytest.raises(ExpertError, match="Cannot resolve expert class"):
            ExpertRegistry.from_configs(expert_cfgs, modality_cfgs)

    def test_from_configs_undeclared_modality_raises(self) -> None:
        expert_cfgs = [
            _make_expert_config("visual_expert", "_test_visual_expert", ["eeg"])
        ]
        modality_cfgs = [_make_modality_config("visual")]
        with pytest.raises(ExpertError, match="undeclared modalities"):
            ExpertRegistry.from_configs(expert_cfgs, modality_cfgs)
