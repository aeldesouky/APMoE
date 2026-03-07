"""Unit tests for apmoe.core.pipeline (InferencePipeline + ModalityChain)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from apmoe.aggregation.base import AggregatorStrategy
from apmoe.core.exceptions import ExpertError, ModalityError, PipelineError
from apmoe.core.pipeline import InferencePipeline, ModalityChain
from apmoe.core.types import (
    EmbeddingResult,
    ExpertOutput,
    ModalityData,
    Prediction,
    ProcessedInput,
)
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import ExpertRegistry
from apmoe.modality.base import ModalityProcessor
from apmoe.processing.base import AnonymizerStrategy, CleanerStrategy, EmbedderStrategy


# ---------------------------------------------------------------------------
# Test doubles (concrete implementations of all ABCs)
# ---------------------------------------------------------------------------


class _PassProcessor(ModalityProcessor):
    """Minimal processor that wraps raw bytes as ModalityData."""

    def __init__(self, modality: str = "visual") -> None:
        self._modality = modality

    @property
    def modality_name(self) -> str:
        return self._modality

    def validate(self, data: object) -> bool:
        return data is not None

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality=self._modality, data=data)


class _RejectingProcessor(ModalityProcessor):
    """Processor whose validate() always returns False."""

    @property
    def modality_name(self) -> str:
        return "visual"

    def validate(self, data: object) -> bool:
        return False

    def preprocess(self, data: object) -> ModalityData:  # pragma: no cover
        return ModalityData(modality="visual", data=data)


class _ExplodingProcessor(ModalityProcessor):
    """Processor whose validate() raises an exception."""

    @property
    def modality_name(self) -> str:
        return "visual"

    def validate(self, data: object) -> bool:
        raise RuntimeError("validator exploded")

    def preprocess(self, data: object) -> ModalityData:  # pragma: no cover
        return ModalityData(modality="visual", data=data)


class _PassCleaner(CleanerStrategy):
    def clean(self, data: ModalityData) -> ModalityData:
        return data


class _TagCleaner(CleanerStrategy):
    """Adds a 'cleaned' metadata flag."""

    def clean(self, data: ModalityData) -> ModalityData:
        md = dict(data.metadata)
        md["cleaned"] = True
        return ModalityData(modality=data.modality, data=data.data, metadata=md)


class _ExplodingCleaner(CleanerStrategy):
    def clean(self, data: ModalityData) -> ModalityData:
        raise RuntimeError("cleaner exploded")


class _PassAnonymizer(AnonymizerStrategy):
    def anonymize(self, data: ModalityData) -> ModalityData:
        return data


class _TagAnonymizer(AnonymizerStrategy):
    """Adds an 'anonymized' metadata flag."""

    def anonymize(self, data: ModalityData) -> ModalityData:
        md = dict(data.metadata)
        md["anonymized"] = True
        return ModalityData(modality=data.modality, data=data.data, metadata=md)


class _ExplodingAnonymizer(AnonymizerStrategy):
    def anonymize(self, data: ModalityData) -> ModalityData:
        raise RuntimeError("anonymizer exploded")


class _FlatEmbedder(EmbedderStrategy):
    """Wraps data as a 1-D float array embedding."""

    def embed(self, data: ModalityData) -> EmbeddingResult:
        arr = np.array([1.0, 2.0, 3.0])
        return EmbeddingResult(modality=data.modality, embedding=arr)


class _ExplodingEmbedder(EmbedderStrategy):
    def embed(self, data: ModalityData) -> EmbeddingResult:
        raise RuntimeError("embedder exploded")


class _ConstantExpert(ExpertPlugin):
    """Returns a fixed age/confidence regardless of input."""

    def __init__(self, expert_name: str, modalities: list[str], age: float = 30.0) -> None:
        self._name = expert_name
        self._modalities = modalities
        self._age = age
        self._loaded = True

    @property
    def name(self) -> str:
        return self._name

    def declared_modalities(self) -> list[str]:
        return list(self._modalities)

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self._name,
            consumed_modalities=list(inputs.keys()),
            predicted_age=self._age,
            confidence=0.8,
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _ExplodingExpert(ExpertPlugin):
    """Expert whose predict() raises a plain RuntimeError."""

    @property
    def name(self) -> str:
        return "exploding_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        pass

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        raise RuntimeError("expert exploded")


class _SumAggregator(AggregatorStrategy):
    """Aggregates by averaging age and confidence."""

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        age = sum(o.predicted_age for o in outputs) / len(outputs)
        conf = sum(o.confidence for o in outputs) / len(outputs)
        return Prediction(
            predicted_age=age,
            confidence=min(conf, 1.0),
            per_expert_outputs=list(outputs),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chain(
    modality: str = "visual",
    *,
    with_embedder: bool = False,
    cleaner: CleanerStrategy | None = None,
    anonymizer: AnonymizerStrategy | None = None,
    processor: ModalityProcessor | None = None,
    embedder: EmbedderStrategy | None = None,
) -> ModalityChain:
    """Return a :class:`ModalityChain` wired with pass-through doubles."""
    return ModalityChain(
        processor=processor or _PassProcessor(modality),
        cleaner=cleaner or _PassCleaner(),
        anonymizer=anonymizer or _PassAnonymizer(),
        embedder=embedder if embedder is not None else (_FlatEmbedder() if with_embedder else None),
    )


def _make_pipeline(
    chains: dict[str, ModalityChain],
    experts: list[ExpertPlugin],
    aggregator: AggregatorStrategy | None = None,
) -> InferencePipeline:
    """Wire an :class:`InferencePipeline` from simple components."""
    registry = ExpertRegistry()
    for expert in experts:
        registry.register_instance(expert)
    return InferencePipeline(
        chains=chains,
        expert_registry=registry,
        aggregator=aggregator or _SumAggregator(),
    )


# ---------------------------------------------------------------------------
# ModalityChain construction
# ---------------------------------------------------------------------------


class TestModalityChain:
    def test_fields_stored_correctly(self) -> None:
        proc = _PassProcessor("visual")
        clean = _PassCleaner()
        anon = _PassAnonymizer()
        emb = _FlatEmbedder()
        chain = ModalityChain(processor=proc, cleaner=clean, anonymizer=anon, embedder=emb)

        assert chain.processor is proc
        assert chain.cleaner is clean
        assert chain.anonymizer is anon
        assert chain.embedder is emb

    def test_embedder_defaults_to_none(self) -> None:
        chain = ModalityChain(
            processor=_PassProcessor(),
            cleaner=_PassCleaner(),
            anonymizer=_PassAnonymizer(),
        )
        assert chain.embedder is None


# ---------------------------------------------------------------------------
# _process_one_modality — happy path
# ---------------------------------------------------------------------------


class TestProcessOneModality:
    def setup_method(self) -> None:
        self.pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[_ConstantExpert("e1", ["visual"])],
        )

    def test_returns_modality_data_without_embedder(self) -> None:
        chain = _make_chain("visual", with_embedder=False)
        result = self.pipeline._process_one_modality("visual", b"raw", chain)
        assert isinstance(result, ModalityData)
        assert result.modality == "visual"

    def test_returns_embedding_result_with_embedder(self) -> None:
        chain = _make_chain("visual", with_embedder=True)
        result = self.pipeline._process_one_modality("visual", b"raw", chain)
        assert isinstance(result, EmbeddingResult)
        assert result.modality == "visual"
        assert result.embedding.shape == (3,)

    def test_cleaner_and_anonymizer_are_called(self) -> None:
        chain = _make_chain(
            "visual",
            cleaner=_TagCleaner(),
            anonymizer=_TagAnonymizer(),
        )
        result = self.pipeline._process_one_modality("visual", b"raw", chain)
        assert isinstance(result, ModalityData)
        assert result.metadata["cleaned"] is True
        assert result.metadata["anonymized"] is True

    def test_validation_failure_raises_modality_error(self) -> None:
        chain = _make_chain("visual", processor=_RejectingProcessor())
        with pytest.raises(ModalityError, match="validation failed"):
            self.pipeline._process_one_modality("visual", b"data", chain)

    def test_validator_exception_wrapped_in_modality_error(self) -> None:
        chain = _make_chain("visual", processor=_ExplodingProcessor())
        with pytest.raises(ModalityError, match="Validation raised"):
            self.pipeline._process_one_modality("visual", b"data", chain)

    def test_cleaner_exception_wrapped(self) -> None:
        chain = _make_chain("visual", cleaner=_ExplodingCleaner())
        with pytest.raises(ModalityError, match="Cleaning failed"):
            self.pipeline._process_one_modality("visual", b"data", chain)

    def test_anonymizer_exception_wrapped(self) -> None:
        chain = _make_chain("visual", anonymizer=_ExplodingAnonymizer())
        with pytest.raises(ModalityError, match="Anonymization failed"):
            self.pipeline._process_one_modality("visual", b"data", chain)

    def test_embedder_exception_wrapped(self) -> None:
        chain = _make_chain("visual", embedder=_ExplodingEmbedder())
        with pytest.raises(ModalityError, match="Embedding failed"):
            self.pipeline._process_one_modality("visual", b"data", chain)


# ---------------------------------------------------------------------------
# run() — Phase A + B, single modality
# ---------------------------------------------------------------------------


class TestRunSingleModality:
    def setup_method(self) -> None:
        self.expert = _ConstantExpert("visual_expert", ["visual"], age=35.0)
        self.pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[self.expert],
        )

    def test_returns_prediction(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        assert isinstance(prediction, Prediction)

    def test_predicted_age_matches_expert(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        assert prediction.predicted_age == 35.0

    def test_per_expert_outputs_populated(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        assert len(prediction.per_expert_outputs) == 1
        assert prediction.per_expert_outputs[0].expert_name == "visual_expert"

    def test_skipped_experts_empty_when_all_present(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        assert prediction.skipped_experts == []

    def test_pipeline_latency_recorded(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        assert "pipeline_latency_s" in prediction.metadata
        assert prediction.metadata["pipeline_latency_s"] >= 0

    def test_available_modalities_in_metadata(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        assert prediction.metadata["available_modalities"] == ["visual"]

    def test_extra_input_modality_ignored(self) -> None:
        """Input keys for unconfigured modalities are silently ignored."""
        prediction = self.pipeline.run({"visual": b"img", "eeg": b"signal"})
        assert isinstance(prediction, Prediction)
        assert "eeg" not in prediction.metadata["available_modalities"]


# ---------------------------------------------------------------------------
# run() — multi-modality + graceful degradation
# ---------------------------------------------------------------------------


class TestRunMultiModality:
    def setup_method(self) -> None:
        self.visual_expert = _ConstantExpert("visual_expert", ["visual"], age=30.0)
        self.audio_expert = _ConstantExpert("audio_expert", ["audio"], age=28.0)
        self.multi_expert = _ConstantExpert("multi_expert", ["visual", "audio"], age=29.0)

        self.pipeline = _make_pipeline(
            chains={
                "visual": _make_chain("visual"),
                "audio": _make_chain("audio", processor=_PassProcessor("audio")),
            },
            experts=[self.visual_expert, self.audio_expert, self.multi_expert],
        )

    def test_all_modalities_all_experts_run(self) -> None:
        prediction = self.pipeline.run({"visual": b"img", "audio": b"wav"})
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"visual_expert", "audio_expert", "multi_expert"}
        assert prediction.skipped_experts == []

    def test_missing_audio_skips_audio_and_multi_experts(self) -> None:
        prediction = self.pipeline.run({"visual": b"img"})
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"visual_expert"}
        assert "audio_expert" in prediction.skipped_experts
        assert "multi_expert" in prediction.skipped_experts

    def test_missing_visual_skips_visual_and_multi_experts(self) -> None:
        prediction = self.pipeline.run({"audio": b"wav"})
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"audio_expert"}
        assert "visual_expert" in prediction.skipped_experts
        assert "multi_expert" in prediction.skipped_experts

    def test_modality_processing_failure_treated_as_missing(self) -> None:
        """A ModalityError in processing should degrade gracefully."""
        failing_chain = _make_chain("visual", processor=_RejectingProcessor())
        pipeline = _make_pipeline(
            chains={
                "visual": failing_chain,
                "audio": _make_chain("audio", processor=_PassProcessor("audio")),
            },
            experts=[self.visual_expert, self.audio_expert, self.multi_expert],
        )
        prediction = pipeline.run({"visual": b"img", "audio": b"wav"})
        # Only audio expert can run; visual and multi experts are skipped
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"audio_expert"}
        assert "visual" in prediction.metadata["failed_modalities"]

    def test_no_runnable_experts_raises_pipeline_error(self) -> None:
        """If every expert is skipped the pipeline should raise PipelineError."""
        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            # Only provide audio input — visual chain exists but no audio input
            experts=[_ConstantExpert("audio_only", ["audio"])],
        )
        with pytest.raises(PipelineError, match="No experts produced output"):
            pipeline.run({"visual": b"img"})

    def test_empty_inputs_raises_pipeline_error(self) -> None:
        with pytest.raises(PipelineError, match="No experts produced output"):
            self.pipeline.run({})


# ---------------------------------------------------------------------------
# run() — expert failure
# ---------------------------------------------------------------------------


class TestRunExpertFailure:
    def test_expert_plain_exception_wrapped_in_expert_error(self) -> None:
        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[_ExplodingExpert()],
        )
        with pytest.raises(ExpertError, match="exploded"):
            pipeline.run({"visual": b"img"})

    def test_expert_error_propagates_unchanged(self) -> None:
        class _ExpertErrorExpert(ExpertPlugin):
            @property
            def name(self) -> str:
                return "err_expert"

            def declared_modalities(self) -> list[str]:
                return ["visual"]

            def load_weights(self, path: str) -> None:
                pass

            def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
                raise ExpertError("direct expert error")

        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[_ExpertErrorExpert()],
        )
        with pytest.raises(ExpertError, match="direct expert error"):
            pipeline.run({"visual": b"img"})


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class TestHooks:
    def setup_method(self) -> None:
        self.pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[_ConstantExpert("e1", ["visual"])],
        )

    def test_on_before_process_fired(self) -> None:
        calls: list[tuple[str, Any]] = []
        self.pipeline.on_before_process.append(lambda name, data: calls.append((name, data)))

        self.pipeline.run({"visual": b"img"})

        assert len(calls) == 1
        assert calls[0][0] == "visual"
        assert calls[0][1] == b"img"

    def test_on_after_embed_fired(self) -> None:
        calls: list[tuple[str, ProcessedInput]] = []
        self.pipeline.on_after_embed.append(lambda name, result: calls.append((name, result)))

        self.pipeline.run({"visual": b"img"})

        assert len(calls) == 1
        assert calls[0][0] == "visual"
        assert isinstance(calls[0][1], ModalityData)

    def test_on_after_expert_fired(self) -> None:
        calls: list[ExpertOutput] = []
        self.pipeline.on_after_expert.append(lambda output: calls.append(output))

        self.pipeline.run({"visual": b"img"})

        assert len(calls) == 1
        assert calls[0].expert_name == "e1"

    def test_on_after_aggregate_fired(self) -> None:
        calls: list[Prediction] = []
        self.pipeline.on_after_aggregate.append(lambda pred: calls.append(pred))

        self.pipeline.run({"visual": b"img"})

        assert len(calls) == 1
        assert isinstance(calls[0], Prediction)

    def test_multiple_hooks_all_fired_in_order(self) -> None:
        order: list[str] = []
        self.pipeline.on_before_process.append(lambda n, d: order.append("hook1"))
        self.pipeline.on_before_process.append(lambda n, d: order.append("hook2"))

        self.pipeline.run({"visual": b"img"})

        assert order == ["hook1", "hook2"]

    def test_hooks_fired_per_modality(self) -> None:
        """on_before_process must fire once per modality, not once total."""
        pipeline = _make_pipeline(
            chains={
                "visual": _make_chain("visual"),
                "audio": _make_chain("audio", processor=_PassProcessor("audio")),
            },
            experts=[
                _ConstantExpert("v", ["visual"]),
                _ConstantExpert("a", ["audio"]),
            ],
        )
        fired: list[str] = []
        pipeline.on_before_process.append(lambda name, data: fired.append(name))

        pipeline.run({"visual": b"img", "audio": b"wav"})

        assert set(fired) == {"visual", "audio"}
        assert len(fired) == 2

    def test_on_after_embed_fired_with_embedding_result_when_embedder_configured(self) -> None:
        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual", with_embedder=True)},
            experts=[_ConstantExpert("e1", ["visual"])],
        )
        embed_results: list[ProcessedInput] = []
        pipeline.on_after_embed.append(lambda name, r: embed_results.append(r))

        pipeline.run({"visual": b"img"})

        assert len(embed_results) == 1
        assert isinstance(embed_results[0], EmbeddingResult)


# ---------------------------------------------------------------------------
# run_async()
# ---------------------------------------------------------------------------


class TestRunAsync:
    def test_async_run_returns_prediction(self) -> None:
        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[_ConstantExpert("e1", ["visual"])],
        )
        prediction = asyncio.run(pipeline.run_async({"visual": b"img"}))
        assert isinstance(prediction, Prediction)
        assert prediction.predicted_age == 30.0

    def test_async_run_multi_modality(self) -> None:
        pipeline = _make_pipeline(
            chains={
                "visual": _make_chain("visual"),
                "audio": _make_chain("audio", processor=_PassProcessor("audio")),
            },
            experts=[
                _ConstantExpert("v", ["visual"], age=30.0),
                _ConstantExpert("a", ["audio"], age=40.0),
            ],
        )
        prediction = asyncio.run(pipeline.run_async({"visual": b"img", "audio": b"wav"}))
        assert isinstance(prediction, Prediction)
        assert prediction.predicted_age == pytest.approx(35.0)

    def test_async_graceful_degradation_missing_modality(self) -> None:
        pipeline = _make_pipeline(
            chains={
                "visual": _make_chain("visual"),
                "audio": _make_chain("audio", processor=_PassProcessor("audio")),
            },
            experts=[
                _ConstantExpert("v", ["visual"]),
                _ConstantExpert("a", ["audio"]),
            ],
        )
        prediction = asyncio.run(pipeline.run_async({"visual": b"img"}))
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"v"}
        assert "a" in prediction.skipped_experts

    def test_async_processing_error_treated_as_missing(self) -> None:
        pipeline = _make_pipeline(
            chains={
                "visual": _make_chain("visual", processor=_RejectingProcessor()),
                "audio": _make_chain("audio", processor=_PassProcessor("audio")),
            },
            experts=[
                _ConstantExpert("v", ["visual"]),
                _ConstantExpert("a", ["audio"]),
            ],
        )
        prediction = asyncio.run(pipeline.run_async({"visual": b"img", "audio": b"wav"}))
        names = {o.expert_name for o in prediction.per_expert_outputs}
        assert names == {"a"}
        assert "visual" in prediction.metadata["failed_modalities"]

    def test_async_no_runnable_experts_raises_pipeline_error(self) -> None:
        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[_ConstantExpert("audio_only", ["audio"])],
        )
        with pytest.raises(PipelineError):
            asyncio.run(pipeline.run_async({"visual": b"img"}))


# ---------------------------------------------------------------------------
# Aggregator integration
# ---------------------------------------------------------------------------


class TestAggregatorIntegration:
    def test_average_of_two_experts(self) -> None:
        pipeline = _make_pipeline(
            chains={"visual": _make_chain("visual")},
            experts=[
                _ConstantExpert("e1", ["visual"], age=20.0),
                _ConstantExpert("e2", ["visual"], age=40.0),
            ],
        )
        prediction = pipeline.run({"visual": b"img"})
        assert prediction.predicted_age == pytest.approx(30.0)

    def test_mock_aggregator_called_with_all_outputs(self) -> None:
        mock_agg = MagicMock(spec=AggregatorStrategy)
        mock_agg.aggregate.return_value = Prediction(
            predicted_age=99.0, confidence=0.5, per_expert_outputs=[]
        )

        registry = ExpertRegistry()
        expert = _ConstantExpert("e1", ["visual"])
        registry.register_instance(expert)

        pipeline = InferencePipeline(
            chains={"visual": _make_chain("visual")},
            expert_registry=registry,
            aggregator=mock_agg,
        )
        pipeline.run({"visual": b"img"})

        mock_agg.aggregate.assert_called_once()
        args = mock_agg.aggregate.call_args[0][0]
        assert len(args) == 1
        assert args[0].expert_name == "e1"
