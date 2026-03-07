"""Integration tests for cross-module boundaries in the APMoE framework.

Each test class targets a specific interface between two adjacent modules,
verifying that the real wiring (registry → factory → pipeline, etc.) works
correctly *without* going through the full APMoEApp bootstrap.

Boundaries covered
------------------
1. modality_registry → ModalityProcessorFactory  (config-driven resolution)
2. cleaner/anonymizer/embedder registries → ModalityChain  (strategy wiring)
3. Processing chain data contract  (ModalityData flowing through Cleaner → Anonymizer → Embedder)
4. expert_registry → ExpertRegistry.from_configs  (class-to-instance lifecycle)
5. ExpertRegistry → InferencePipeline  (dispatch + predict wiring)
6. aggregator_registry → AggregatorStrategy → Prediction  (aggregation wiring)
7. Config → ModalityProcessorFactory + strategy registries → InferencePipeline
   (the full wiring logic that APMoEApp.from_config performs, tested directly
   without the app layer)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.config import (
    AggregationConfig,
    ExpertConfig,
    ModalityConfig,
    PipelineConfig,
    load_config,
)
from apmoe.core.exceptions import ConfigurationError, ModalityError
from apmoe.core.pipeline import InferencePipeline, ModalityChain
from apmoe.core.types import (
    EmbeddingResult,
    ExpertOutput,
    ModalityData,
    Prediction,
    ProcessedInput,
)
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import ExpertRegistry, expert_registry
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import ModalityProcessorFactory, modality_registry
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    EmbedderStrategy,
    anonymizer_registry,
    cleaner_registry,
    embedder_registry,
)


# ---------------------------------------------------------------------------
# Shared minimal concrete implementations
# ---------------------------------------------------------------------------


class _BVisualProcessor(ModalityProcessor):
    """Real processor: wraps bytes as ModalityData with shape metadata."""

    @property
    def modality_name(self) -> str:
        return "visual"

    def validate(self, data: object) -> bool:
        return isinstance(data, (bytes, np.ndarray))

    def preprocess(self, data: object) -> ModalityData:
        arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, bytes) else np.array(data)
        return ModalityData(
            modality="visual",
            data=arr,
            metadata={"shape": arr.shape, "dtype": str(arr.dtype)},
        )


class _BAudioProcessor(ModalityProcessor):
    @property
    def modality_name(self) -> str:
        return "audio"

    def validate(self, data: object) -> bool:
        return isinstance(data, (bytes, np.ndarray))

    def preprocess(self, data: object) -> ModalityData:
        arr = np.frombuffer(data, dtype=np.float32) if isinstance(data, bytes) else np.array(data)
        return ModalityData(modality="audio", data=arr, metadata={"samples": arr.size})


class _NormCleaner(CleanerStrategy):
    """Cleaner: clips array values to [0, 1]."""

    def clean(self, data: ModalityData) -> ModalityData:
        arr = np.clip(np.array(data.data, dtype=float), 0.0, 1.0)
        md = {**data.metadata, "normalized": True}
        return ModalityData(modality=data.modality, data=arr, metadata=md)


class _TagAnonymizer(AnonymizerStrategy):
    """Anonymizer: adds an 'anonymized' tag to metadata without touching data."""

    def anonymize(self, data: ModalityData) -> ModalityData:
        md = {**data.metadata, "anonymized": True}
        return ModalityData(modality=data.modality, data=data.data, metadata=md)


class _FlatEmbedder(EmbedderStrategy):
    """Embedder: flattens the array and takes first 4 values as embedding."""

    def embed(self, data: ModalityData) -> EmbeddingResult:
        arr = np.array(data.data, dtype=float).flatten()
        emb = arr[:4] if len(arr) >= 4 else np.pad(arr, (0, 4 - len(arr)))
        return EmbeddingResult(
            modality=data.modality,
            embedding=emb,
            metadata={**data.metadata, "embedder": "flat"},
        )


class _VisualExpert(ExpertPlugin):
    def __init__(self) -> None:
        self._loaded = False
        self._received_input: ProcessedInput | None = None

    @property
    def name(self) -> str:
        return "visual_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        self._received_input = inputs.get("visual")
        return ExpertOutput("visual_expert", ["visual"], 30.0, 0.85)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _AudioExpert(ExpertPlugin):
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
        return ExpertOutput("audio_expert", ["audio"], 25.0, 0.70)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _MultiExpert(ExpertPlugin):
    def __init__(self) -> None:
        self._loaded = False
        self._received: dict[str, ProcessedInput] = {}

    @property
    def name(self) -> str:
        return "multi_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual", "audio"]

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        self._received = dict(inputs)
        return ExpertOutput("multi_expert", list(inputs.keys()), 28.0, 0.90)

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _MeanAggregator(AggregatorStrategy):
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        age = sum(o.predicted_age for o in outputs) / len(outputs)
        conf = sum(o.confidence for o in outputs) / len(outputs)
        return Prediction(
            predicted_age=age,
            confidence=min(conf, 1.0),
            per_expert_outputs=list(outputs),
        )


# Short keys used to register components in global registries
_VPROC = "_bnd.VisualProcessor"
_APROC = "_bnd.AudioProcessor"
_NCLEANER = "_bnd.NormCleaner"
_TANON = "_bnd.TagAnonymizer"
_FEMB = "_bnd.FlatEmbedder"
_VEXP = "_bnd.VisualExpert"
_AEXP = "_bnd.AudioExpert"
_MEXP = "_bnd.MultiExpert"
_MAGG = "_bnd.MeanAggregator"


@pytest.fixture(autouse=True)
def _register_boundary_components() -> None:
    """Register all boundary test components in their global registries."""
    modality_registry.register_class(_VPROC, _BVisualProcessor, overwrite=True)
    modality_registry.register_class(_APROC, _BAudioProcessor, overwrite=True)
    cleaner_registry.register_class(_NCLEANER, _NormCleaner, overwrite=True)
    anonymizer_registry.register_class(_TANON, _TagAnonymizer, overwrite=True)
    embedder_registry.register_class(_FEMB, _FlatEmbedder, overwrite=True)
    expert_registry.register_class(_VEXP, _VisualExpert, overwrite=True)
    expert_registry.register_class(_AEXP, _AudioExpert, overwrite=True)
    expert_registry.register_class(_MEXP, _MultiExpert, overwrite=True)
    aggregator_registry.register_class(_MAGG, _MeanAggregator, overwrite=True)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_modality_cfg(name: str, proc: str, embedder: str | None = None) -> ModalityConfig:
    return ModalityConfig(
        name=name,
        processor=proc,
        pipeline=PipelineConfig(cleaner=_NCLEANER, anonymizer=_TANON, embedder=embedder),
    )


def _make_expert_cfg(name: str, cls: str, modalities: list[str], weights: str) -> ExpertConfig:
    return ExpertConfig(name=name, **{"class": cls}, weights=weights, modalities=modalities)


# ---------------------------------------------------------------------------
# Boundary 1: modality_registry → ModalityProcessorFactory
# ---------------------------------------------------------------------------


class TestModalityRegistryToFactory:
    """modality_registry.resolve() feeds into ModalityProcessorFactory."""

    def test_factory_resolves_registered_name(self) -> None:
        cfgs = [_make_modality_cfg("visual", _VPROC)]
        processors = ModalityProcessorFactory.from_configs(cfgs)
        assert "visual" in processors
        assert isinstance(processors["visual"], _BVisualProcessor)

    def test_factory_resolves_two_modalities(self) -> None:
        cfgs = [_make_modality_cfg("visual", _VPROC), _make_modality_cfg("audio", _APROC)]
        processors = ModalityProcessorFactory.from_configs(cfgs)
        assert set(processors.keys()) == {"visual", "audio"}
        assert isinstance(processors["visual"], _BVisualProcessor)
        assert isinstance(processors["audio"], _BAudioProcessor)

    def test_factory_modality_name_mismatch_raises_modality_error(self) -> None:
        """Processor whose modality_name differs from config name must raise."""
        # _BVisualProcessor declares 'visual' but config says 'audio'
        cfgs = [_make_modality_cfg("audio", _VPROC)]
        with pytest.raises(ModalityError, match="modality_name"):
            ModalityProcessorFactory.from_configs(cfgs)

    def test_factory_unknown_processor_raises_modality_error(self) -> None:
        cfgs = [_make_modality_cfg("visual", "no.such.Processor")]
        with pytest.raises(ModalityError):
            ModalityProcessorFactory.from_configs(cfgs)

    def test_factory_produces_independent_instances(self) -> None:
        """Each call to from_configs should produce a fresh instance."""
        cfgs = [_make_modality_cfg("visual", _VPROC)]
        p1 = ModalityProcessorFactory.from_configs(cfgs)["visual"]
        p2 = ModalityProcessorFactory.from_configs(cfgs)["visual"]
        assert p1 is not p2


# ---------------------------------------------------------------------------
# Boundary 2: Strategy registries → ModalityChain
# ---------------------------------------------------------------------------


class TestStrategyRegistriesToChain:
    """cleaner/anonymizer/embedder registries resolve into ModalityChain."""

    def _resolve_chain(self, with_embedder: bool = False) -> ModalityChain:
        cleaner_cls = cleaner_registry.resolve(_NCLEANER)
        anonymizer_cls = anonymizer_registry.resolve(_TANON)
        embedder: EmbedderStrategy | None = None
        if with_embedder:
            embedder_cls = embedder_registry.resolve(_FEMB)
            embedder = embedder_cls()
        return ModalityChain(
            processor=_BVisualProcessor(),
            cleaner=cleaner_cls(),
            anonymizer=anonymizer_cls(),
            embedder=embedder,
        )

    def test_chain_without_embedder_has_none_embedder(self) -> None:
        chain = self._resolve_chain(with_embedder=False)
        assert chain.embedder is None

    def test_chain_with_embedder_has_embedder_instance(self) -> None:
        chain = self._resolve_chain(with_embedder=True)
        assert isinstance(chain.embedder, _FlatEmbedder)

    def test_registry_resolve_returns_class_not_instance(self) -> None:
        cls = cleaner_registry.resolve(_NCLEANER)
        assert callable(cls)
        assert not isinstance(cls, CleanerStrategy)

    def test_unknown_cleaner_raises_registry_error(self) -> None:
        from apmoe.core.exceptions import RegistryError
        with pytest.raises(RegistryError):
            cleaner_registry.resolve("no.such.Cleaner")

    def test_unknown_anonymizer_raises_registry_error(self) -> None:
        from apmoe.core.exceptions import RegistryError
        with pytest.raises(RegistryError):
            anonymizer_registry.resolve("no.such.Anonymizer")


# ---------------------------------------------------------------------------
# Boundary 3: Processing chain data contract (Cleaner → Anonymizer → Embedder)
# ---------------------------------------------------------------------------


class TestProcessingChainDataContract:
    """ModalityData flows correctly through each strategy stage."""

    def test_cleaner_output_feeds_anonymizer(self) -> None:
        """The ModalityData produced by cleaner is a valid input for anonymizer."""
        raw = ModalityData(modality="visual", data=np.array([0.5, 1.5, -0.5]))
        cleaner = _NormCleaner()
        anonymizer = _TagAnonymizer()

        cleaned = cleaner.clean(raw)
        assert isinstance(cleaned, ModalityData)

        anonymized = anonymizer.anonymize(cleaned)
        assert isinstance(anonymized, ModalityData)
        assert anonymized.metadata["normalized"] is True
        assert anonymized.metadata["anonymized"] is True
        assert anonymized.modality == "visual"

    def test_clamp_values_preserved_through_chain(self) -> None:
        """NormCleaner clamps, TagAnonymizer preserves the clamped data."""
        raw = ModalityData(modality="visual", data=np.array([-1.0, 0.5, 2.0]))
        cleaned = _NormCleaner().clean(raw)
        assert np.all(np.array(cleaned.data) >= 0.0)
        assert np.all(np.array(cleaned.data) <= 1.0)

        anon = _TagAnonymizer().anonymize(cleaned)
        assert np.all(np.array(anon.data) >= 0.0)  # clamping survived anonymizer

    def test_anonymizer_output_feeds_embedder(self) -> None:
        """The ModalityData produced by anonymizer is a valid input for embedder."""
        raw = ModalityData(modality="visual", data=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        anon = _TagAnonymizer().anonymize(raw)
        result = _FlatEmbedder().embed(anon)

        assert isinstance(result, EmbeddingResult)
        assert result.modality == "visual"
        assert result.embedding.shape == (4,)
        # Metadata from the anonymizer step should survive into the embedding
        assert result.metadata.get("anonymized") is True

    def test_full_chain_cleaner_anonymizer_embedder(self) -> None:
        """Full three-step chain produces EmbeddingResult with correct shape."""
        raw = ModalityData(modality="visual", data=np.array([0.1, 0.9, 1.5, 0.4]))
        cleaned = _NormCleaner().clean(raw)
        anon = _TagAnonymizer().anonymize(cleaned)
        result = _FlatEmbedder().embed(anon)

        assert isinstance(result, EmbeddingResult)
        assert result.embedding.shape == (4,)
        assert all(0.0 <= v <= 1.0 for v in result.embedding)  # clamped values

    def test_full_chain_without_embedder_returns_modality_data(self) -> None:
        """Cleaner → Anonymizer without embedder still produces ModalityData."""
        raw = ModalityData(modality="visual", data=np.array([0.5, 0.5]))
        cleaned = _NormCleaner().clean(raw)
        anon = _TagAnonymizer().anonymize(cleaned)
        assert isinstance(anon, ModalityData)
        assert anon.metadata["normalized"] is True
        assert anon.metadata["anonymized"] is True

    def test_modality_name_is_preserved_through_chain(self) -> None:
        """modality field on ModalityData must survive all three stages."""
        raw = ModalityData(modality="audio", data=np.array([0.0]))
        cleaned = _NormCleaner().clean(raw)
        anon = _TagAnonymizer().anonymize(cleaned)
        result = _FlatEmbedder().embed(anon)
        assert cleaned.modality == "audio"
        assert anon.modality == "audio"
        assert result.modality == "audio"


# ---------------------------------------------------------------------------
# Boundary 4: expert_registry → ExpertRegistry.from_configs
# ---------------------------------------------------------------------------


class TestExpertRegistryFromConfigsBoundary:
    """expert_registry class resolution feeds ExpertRegistry.from_configs."""

    def test_from_configs_resolves_and_instantiates(self, tmp_path: Path) -> None:
        weights = tmp_path / "v.pt"
        weights.write_bytes(b"dummy")
        mod_cfgs = [_make_modality_cfg("visual", _VPROC)]
        exp_cfgs = [_make_expert_cfg("visual_expert", _VEXP, ["visual"], str(weights))]

        registry = ExpertRegistry.from_configs(exp_cfgs, mod_cfgs)
        assert "visual_expert" in registry
        assert registry.all_healthy() is True

    def test_from_configs_calls_load_weights(self, tmp_path: Path) -> None:
        weights = tmp_path / "v.pt"
        weights.write_bytes(b"dummy")
        mod_cfgs = [_make_modality_cfg("visual", _VPROC)]
        exp_cfgs = [_make_expert_cfg("visual_expert", _VEXP, ["visual"], str(weights))]

        registry = ExpertRegistry.from_configs(exp_cfgs, mod_cfgs)
        expert = registry.get("visual_expert")
        assert expert.is_loaded is True

    def test_from_configs_two_experts(self, tmp_path: Path) -> None:
        vw = tmp_path / "v.pt"
        aw = tmp_path / "a.pt"
        vw.write_bytes(b"dummy")
        aw.write_bytes(b"dummy")
        mod_cfgs = [_make_modality_cfg("visual", _VPROC), _make_modality_cfg("audio", _APROC)]
        exp_cfgs = [
            _make_expert_cfg("visual_expert", _VEXP, ["visual"], str(vw)),
            _make_expert_cfg("audio_expert", _AEXP, ["audio"], str(aw)),
        ]

        registry = ExpertRegistry.from_configs(exp_cfgs, mod_cfgs)
        assert len(registry) == 2
        assert registry.all_healthy() is True

    def test_from_configs_produces_runnable_expert_for_available_modality(
        self, tmp_path: Path
    ) -> None:
        weights = tmp_path / "v.pt"
        weights.write_bytes(b"dummy")
        mod_cfgs = [_make_modality_cfg("visual", _VPROC)]
        exp_cfgs = [_make_expert_cfg("visual_expert", _VEXP, ["visual"], str(weights))]

        registry = ExpertRegistry.from_configs(exp_cfgs, mod_cfgs)
        runnable = registry.get_runnable_experts({"visual"})
        assert len(runnable) == 1
        assert runnable[0].name == "visual_expert"

    def test_from_configs_skips_expert_with_missing_modality(self, tmp_path: Path) -> None:
        weights = tmp_path / "v.pt"
        weights.write_bytes(b"dummy")
        mod_cfgs = [_make_modality_cfg("visual", _VPROC)]
        exp_cfgs = [_make_expert_cfg("visual_expert", _VEXP, ["visual"], str(weights))]

        registry = ExpertRegistry.from_configs(exp_cfgs, mod_cfgs)
        runnable = registry.get_runnable_experts(set())
        assert runnable == []
        assert "visual_expert" in registry.get_skipped_experts(set())


# ---------------------------------------------------------------------------
# Boundary 5: ExpertRegistry → InferencePipeline (dispatch + predict)
# ---------------------------------------------------------------------------


class TestExpertRegistryToPipeline:
    """ExpertRegistry.get_runnable_experts feeds InferencePipeline Phase B."""

    def _make_registry(self, experts: list[ExpertPlugin]) -> ExpertRegistry:
        reg = ExpertRegistry()
        for e in experts:
            reg.register_instance(e)
        return reg

    def _make_pipeline(
        self,
        experts: list[ExpertPlugin],
        modalities: list[str] | None = None,
        with_embedder: bool = False,
    ) -> InferencePipeline:
        modalities = modalities or ["visual"]
        chains = {}
        for mod in modalities:
            proc = _BVisualProcessor() if mod == "visual" else _BAudioProcessor()
            chains[mod] = ModalityChain(
                processor=proc,
                cleaner=_NormCleaner(),
                anonymizer=_TagAnonymizer(),
                embedder=_FlatEmbedder() if with_embedder else None,
            )
        return InferencePipeline(
            chains=chains,
            expert_registry=self._make_registry(experts),
            aggregator=_MeanAggregator(),
        )

    def test_single_expert_receives_correct_modality_key(self) -> None:
        expert = _VisualExpert()
        expert.load_weights("")
        pipeline = self._make_pipeline([expert])

        pipeline.run({"visual": b"\x00\x01\x02\x03"})

        assert expert._received_input is not None
        assert isinstance(expert._received_input, ModalityData)

    def test_single_expert_receives_embedding_when_embedder_configured(self) -> None:
        expert = _VisualExpert()
        expert.load_weights("")
        pipeline = self._make_pipeline([expert], with_embedder=True)

        pipeline.run({"visual": b"\x00\x01\x02\x03"})

        assert isinstance(expert._received_input, EmbeddingResult)

    def test_multi_expert_receives_all_declared_modalities(self) -> None:
        expert = _MultiExpert()
        expert.load_weights("")
        pipeline = self._make_pipeline([expert], modalities=["visual", "audio"])

        pipeline.run({"visual": b"\x00\x01\x02\x03", "audio": b"\x00\x00\x80\x3f"})

        assert "visual" in expert._received
        assert "audio" in expert._received

    def test_expert_skipped_when_modality_absent(self) -> None:
        visual_expert = _VisualExpert()
        visual_expert.load_weights("")
        audio_expert = _AudioExpert()
        audio_expert.load_weights("")
        pipeline = self._make_pipeline(
            [visual_expert, audio_expert], modalities=["visual", "audio"]
        )

        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03"})

        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert "visual_expert" in names
        assert "audio_expert" not in names
        assert "audio_expert" in prediction.skipped_experts

    def test_expert_output_flows_correctly_to_aggregator(self) -> None:
        visual_expert = _VisualExpert()
        visual_expert.load_weights("")
        audio_expert = _AudioExpert()
        audio_expert.load_weights("")
        pipeline = self._make_pipeline(
            [visual_expert, audio_expert], modalities=["visual", "audio"]
        )

        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03", "audio": b"\x00\x00\x80\x3f"})

        # _MeanAggregator averages: (30.0 + 25.0) / 2 = 27.5
        assert prediction.predicted_age == pytest.approx(27.5)
        assert len(prediction.per_expert_outputs) == 2


# ---------------------------------------------------------------------------
# Boundary 6: aggregator_registry → AggregatorStrategy → Prediction
# ---------------------------------------------------------------------------


class TestAggregatorRegistryToPipeline:
    """aggregator_registry.resolve() feeds the pipeline's aggregation step."""

    def test_resolved_aggregator_produces_valid_prediction(self) -> None:
        agg_cls = aggregator_registry.resolve(_MAGG)
        agg: AggregatorStrategy = agg_cls()

        outputs = [
            ExpertOutput("e1", ["visual"], 30.0, 0.9),
            ExpertOutput("e2", ["audio"], 40.0, 0.8),
        ]
        prediction = agg.aggregate(outputs)

        assert isinstance(prediction, Prediction)
        assert prediction.predicted_age == pytest.approx(35.0)
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.per_expert_outputs) == 2

    def test_aggregator_registered_by_dotted_path_resolves(self) -> None:
        """Registry should also resolve via dotted import path."""
        cls = aggregator_registry.resolve(_MAGG)
        instance = cls()
        assert isinstance(instance, AggregatorStrategy)

    def test_aggregator_in_pipeline_produces_correct_result(self) -> None:
        agg_cls = aggregator_registry.resolve(_MAGG)
        expert = _VisualExpert()
        expert.load_weights("")
        registry = ExpertRegistry()
        registry.register_instance(expert)

        pipeline = InferencePipeline(
            chains={
                "visual": ModalityChain(
                    processor=_BVisualProcessor(),
                    cleaner=_NormCleaner(),
                    anonymizer=_TagAnonymizer(),
                )
            },
            expert_registry=registry,
            aggregator=agg_cls(),
        )
        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03"})
        assert prediction.predicted_age == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Boundary 7: Config → component resolution → InferencePipeline
#              (the wiring APMoEApp.from_config does, tested without the app)
# ---------------------------------------------------------------------------


class TestConfigToManualPipelineWiring:
    """Exercise the exact wiring logic that APMoEApp.from_config() performs,
    but assembled manually here to test each step in isolation."""

    def _write_config(
        self, tmp_path: Path, *, with_audio: bool = False, with_embedder: bool = False
    ) -> Path:
        vw = tmp_path / "v.pt"
        vw.write_bytes(b"dummy")
        modalities: list[dict[str, Any]] = [
            {
                "name": "visual",
                "processor": _VPROC,
                "pipeline": {
                    "cleaner": _NCLEANER,
                    "anonymizer": _TANON,
                    **({"embedder": _FEMB} if with_embedder else {}),
                },
            }
        ]
        experts: list[dict[str, Any]] = [
            {"name": "visual_expert", "class": _VEXP, "weights": str(vw), "modalities": ["visual"]}
        ]
        if with_audio:
            aw = tmp_path / "a.pt"
            aw.write_bytes(b"dummy")
            modalities.append(
                {
                    "name": "audio",
                    "processor": _APROC,
                    "pipeline": {"cleaner": _NCLEANER, "anonymizer": _TANON},
                }
            )
            experts.append(
                {
                    "name": "audio_expert",
                    "class": _AEXP,
                    "weights": str(aw),
                    "modalities": ["audio"],
                }
            )
        cfg = {
            "apmoe": {
                "modalities": modalities,
                "experts": experts,
                "aggregation": {"strategy": _MAGG},
            }
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(cfg))
        return p

    def _wire_pipeline_from_config(self, cfg_path: Path) -> InferencePipeline:
        """Replicate APMoEApp.from_config wiring without using the app class."""
        cfg = load_config(cfg_path)
        apmoe_cfg = cfg.apmoe

        # Processors
        processors = ModalityProcessorFactory.from_configs(apmoe_cfg.modalities)

        # Chains
        chains: dict[str, ModalityChain] = {}
        for mod_cfg in apmoe_cfg.modalities:
            name = mod_cfg.name
            pipeline_cfg = mod_cfg.pipeline
            cleaner = cleaner_registry.resolve(pipeline_cfg.cleaner)()
            anonymizer = anonymizer_registry.resolve(pipeline_cfg.anonymizer)()
            embedder = None
            if pipeline_cfg.embedder:
                embedder = embedder_registry.resolve(pipeline_cfg.embedder)()
            chains[name] = ModalityChain(
                processor=processors[name],
                cleaner=cleaner,
                anonymizer=anonymizer,
                embedder=embedder,
            )

        # Expert registry
        expert_reg = ExpertRegistry.from_configs(apmoe_cfg.experts, apmoe_cfg.modalities)

        # Aggregator
        agg = aggregator_registry.resolve(apmoe_cfg.aggregation.strategy)()

        return InferencePipeline(chains=chains, expert_registry=expert_reg, aggregator=agg)

    def test_wired_pipeline_runs_single_modality(self, tmp_path: Path) -> None:
        pipeline = self._wire_pipeline_from_config(self._write_config(tmp_path))
        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03"})
        assert isinstance(prediction, Prediction)
        assert prediction.predicted_age == pytest.approx(30.0)

    def test_wired_pipeline_runs_multi_modality(self, tmp_path: Path) -> None:
        pipeline = self._wire_pipeline_from_config(
            self._write_config(tmp_path, with_audio=True)
        )
        # Bytes are dtype-realistic: visual→uint8 array([0,1,2,3]), audio→float32 array([1.0]) (0x3F800000 in IEEE 754)
        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03", "audio": b"\x00\x00\x80\x3f"})
        # visual=30.0, audio=25.0 → average=27.5
        assert prediction.predicted_age == pytest.approx(27.5)
        assert len(prediction.per_expert_outputs) == 2

    def test_wired_pipeline_with_embedder_passes_embedding_result(self, tmp_path: Path) -> None:
        expert = _VisualExpert()
        expert.load_weights("")
        # Manually build pipeline with embedder
        registry = ExpertRegistry()
        registry.register_instance(expert)
        chain = ModalityChain(
            processor=_BVisualProcessor(),
            cleaner=_NormCleaner(),
            anonymizer=_TagAnonymizer(),
            embedder=_FlatEmbedder(),
        )
        pipeline = InferencePipeline(
            chains={"visual": chain},
            expert_registry=registry,
            aggregator=_MeanAggregator(),
        )
        pipeline.run({"visual": b"\x00\x01\x02\x03"})
        assert isinstance(expert._received_input, EmbeddingResult)

    def test_missing_modality_skips_dependent_experts(self, tmp_path: Path) -> None:
        pipeline = self._wire_pipeline_from_config(
            self._write_config(tmp_path, with_audio=True)
        )
        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03"})
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"visual_expert"}
        assert "audio_expert" in prediction.skipped_experts

    def test_config_loaded_modality_names_match_processor_names(self, tmp_path: Path) -> None:
        """Ensure config name and processor.modality_name are in sync end-to-end."""
        cfg = load_config(self._write_config(tmp_path))
        processors = ModalityProcessorFactory.from_configs(cfg.apmoe.modalities)
        for mod_cfg in cfg.apmoe.modalities:
            assert processors[mod_cfg.name].modality_name == mod_cfg.name

    def test_pipeline_latency_metadata_present_after_run(self, tmp_path: Path) -> None:
        pipeline = self._wire_pipeline_from_config(self._write_config(tmp_path))
        prediction = pipeline.run({"visual": b"\x00\x01\x02\x03"})
        assert "pipeline_latency_s" in prediction.metadata
        assert prediction.metadata["pipeline_latency_s"] >= 0.0
