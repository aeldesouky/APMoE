"""Unit tests for apmoe.core.types (ModalityData, EmbeddingResult, ExpertOutput, Prediction)."""

from __future__ import annotations

import numpy as np
import pytest

from apmoe.core.types import (
    EmbeddingResult,
    ExpertOutput,
    ModalityData,
    Prediction,
    ProcessedInput,
)


# ---------------------------------------------------------------------------
# ModalityData
# ---------------------------------------------------------------------------


class TestModalityData:
    def test_basic_construction(self) -> None:
        md = ModalityData(modality="visual", data=b"some_image_bytes")
        assert md.modality == "visual"
        assert md.data == b"some_image_bytes"
        assert md.metadata == {}
        assert md.timestamp is None
        assert md.source is None

    def test_full_construction(self) -> None:
        md = ModalityData(
            modality="audio",
            data=np.zeros(100),
            metadata={"sample_rate": 16000},
            timestamp=1_700_000_000.0,
            source="microphone:0",
        )
        assert md.metadata["sample_rate"] == 16000
        assert md.timestamp == 1_700_000_000.0
        assert md.source == "microphone:0"

    def test_with_data_returns_new_instance(self) -> None:
        original = ModalityData(
            modality="visual",
            data=np.zeros(10),
            metadata={"key": "value"},
            timestamp=1.0,
            source="cam",
        )
        new_payload = np.ones(10)
        replaced = original.with_data(new_payload)

        assert replaced is not original
        assert replaced.modality == original.modality
        assert replaced.metadata == original.metadata
        assert replaced.timestamp == original.timestamp
        assert replaced.source == original.source
        np.testing.assert_array_equal(replaced.data, new_payload)

    def test_with_data_does_not_mutate_original(self) -> None:
        original = ModalityData(modality="visual", data=np.zeros(5))
        _ = original.with_data(np.ones(5))
        np.testing.assert_array_equal(original.data, np.zeros(5))

    def test_metadata_isolation_in_with_data(self) -> None:
        """Mutating the copy's metadata must not affect the original."""
        original = ModalityData(modality="visual", data=None, metadata={"k": 1})
        copy = original.with_data("new")
        copy.metadata["k"] = 999
        assert original.metadata["k"] == 1


# ---------------------------------------------------------------------------
# EmbeddingResult
# ---------------------------------------------------------------------------


class TestEmbeddingResult:
    def test_embedding_dim_inferred(self) -> None:
        arr = np.zeros(512)
        er = EmbeddingResult(modality="visual", embedding=arr)
        assert er.embedding_dim == 512

    def test_embedding_dim_inferred_2d(self) -> None:
        arr = np.zeros((4, 256))
        er = EmbeddingResult(modality="audio", embedding=arr)
        assert er.embedding_dim == 256

    def test_embedding_dim_explicit(self) -> None:
        arr = np.zeros(128)
        er = EmbeddingResult(modality="eeg", embedding=arr, embedding_dim=128)
        assert er.embedding_dim == 128

    def test_metadata_defaults_empty(self) -> None:
        er = EmbeddingResult(modality="visual", embedding=np.zeros(64))
        assert er.metadata == {}

    def test_modality_stored(self) -> None:
        er = EmbeddingResult(modality="audio", embedding=np.zeros(32))
        assert er.modality == "audio"


# ---------------------------------------------------------------------------
# ProcessedInput union
# ---------------------------------------------------------------------------


class TestProcessedInputUnion:
    """Verify that ProcessedInput correctly accepts both types."""

    def _accept(self, inp: ProcessedInput) -> str:
        """Helper that asserts the type is one of the union members."""
        assert isinstance(inp, (ModalityData, EmbeddingResult))
        return type(inp).__name__

    def test_accepts_modality_data(self) -> None:
        md = ModalityData(modality="visual", data=None)
        assert self._accept(md) == "ModalityData"

    def test_accepts_embedding_result(self) -> None:
        er = EmbeddingResult(modality="visual", embedding=np.zeros(64))
        assert self._accept(er) == "EmbeddingResult"


# ---------------------------------------------------------------------------
# ExpertOutput
# ---------------------------------------------------------------------------


class TestExpertOutput:
    def test_basic_construction(self) -> None:
        out = ExpertOutput(
            expert_name="face_expert",
            consumed_modalities=["visual"],
            predicted_age=32.5,
            confidence=0.9,
        )
        assert out.expert_name == "face_expert"
        assert out.predicted_age == 32.5
        assert out.confidence == 0.9
        assert out.consumed_modalities == ["visual"]
        assert out.metadata == {}

    def test_confidence_zero_valid(self) -> None:
        out = ExpertOutput("e", ["visual"], 25.0, 0.0)
        assert out.confidence == 0.0

    def test_confidence_one_valid(self) -> None:
        out = ExpertOutput("e", ["visual"], 25.0, 1.0)
        assert out.confidence == 1.0

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            ExpertOutput("e", ["visual"], 25.0, 1.01)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            ExpertOutput("e", ["visual"], 25.0, -0.01)

    def test_confidence_sentinel_negative_one_valid(self) -> None:
        """-1.0 means the expert does not report a confidence score."""
        out = ExpertOutput("e", ["visual"], 25.0, -1.0)
        assert out.confidence == -1.0

    def test_multi_modal_expert(self) -> None:
        out = ExpertOutput(
            expert_name="multi",
            consumed_modalities=["visual", "audio"],
            predicted_age=40.0,
            confidence=0.75,
            metadata={"latency_ms": 12},
        )
        assert "audio" in out.consumed_modalities
        assert out.metadata["latency_ms"] == 12


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    def _make_expert_output(self, age: float = 30.0, conf: float = 0.8) -> ExpertOutput:
        return ExpertOutput("e", ["visual"], age, conf)

    def test_basic_construction(self) -> None:
        pred = Prediction(predicted_age=35.0, confidence=0.85)
        assert pred.predicted_age == 35.0
        assert pred.confidence == 0.85
        assert pred.confidence_interval is None
        assert pred.per_expert_outputs == []
        assert pred.skipped_experts == []
        assert pred.metadata == {}

    def test_with_confidence_interval(self) -> None:
        pred = Prediction(
            predicted_age=35.0, confidence=0.85, confidence_interval=(30.0, 40.0)
        )
        assert pred.confidence_interval == (30.0, 40.0)

    def test_confidence_interval_inverted_raises(self) -> None:
        with pytest.raises(ValueError, match="lower bound"):
            Prediction(predicted_age=35.0, confidence=0.85, confidence_interval=(50.0, 30.0))

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Prediction(predicted_age=35.0, confidence=1.1)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Prediction(predicted_age=35.0, confidence=-0.1)

    def test_per_expert_outputs(self) -> None:
        expert_out = self._make_expert_output()
        pred = Prediction(
            predicted_age=30.0,
            confidence=0.8,
            per_expert_outputs=[expert_out],
        )
        assert len(pred.per_expert_outputs) == 1
        assert pred.per_expert_outputs[0] is expert_out

    def test_skipped_experts(self) -> None:
        pred = Prediction(
            predicted_age=30.0, confidence=0.8, skipped_experts=["audio_expert"]
        )
        assert "audio_expert" in pred.skipped_experts

    def test_equal_interval_bounds_valid(self) -> None:
        """A degenerate interval where lower == upper should be accepted."""
        pred = Prediction(
            predicted_age=30.0, confidence=0.8, confidence_interval=(30.0, 30.0)
        )
        assert pred.confidence_interval == (30.0, 30.0)
