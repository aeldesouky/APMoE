"""Unit tests for apmoe.aggregation.base (AggregatorStrategy ABC)."""

from __future__ import annotations

import pytest

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.types import ExpertOutput, Prediction


# ---------------------------------------------------------------------------
# Concrete helpers
# ---------------------------------------------------------------------------


class _MeanAggregator(AggregatorStrategy):
    """Simple unweighted mean aggregator."""

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        n = len(outputs)
        age = sum(o.predicted_age for o in outputs) / n
        conf = sum(o.confidence for o in outputs) / n
        return Prediction(
            predicted_age=age,
            confidence=min(conf, 1.0),
            per_expert_outputs=list(outputs),
        )


class _FixedAggregator(AggregatorStrategy):
    """Returns a hard-coded prediction (for testing load_weights override)."""

    def __init__(self) -> None:
        self._weights_loaded = False

    def load_weights(self, path: str) -> None:
        self._weights_loaded = True

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        return Prediction(
            predicted_age=42.0,
            confidence=1.0,
            per_expert_outputs=list(outputs),
        )


# ---------------------------------------------------------------------------
# AggregatorStrategy ABC
# ---------------------------------------------------------------------------


class TestAggregatorStrategyABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AggregatorStrategy()  # type: ignore[abstract]

    def test_missing_aggregate_raises(self) -> None:
        class _Bad(AggregatorStrategy):
            pass

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        agg = _MeanAggregator()
        assert isinstance(agg, AggregatorStrategy)

    def test_aggregate_single_expert(self) -> None:
        agg = _MeanAggregator()
        outputs = [
            ExpertOutput(
                expert_name="face_expert",
                consumed_modalities=["visual"],
                predicted_age=35.0,
                confidence=0.8,
            )
        ]
        result = agg.aggregate(outputs)
        assert isinstance(result, Prediction)
        assert result.predicted_age == pytest.approx(35.0)
        assert result.confidence == pytest.approx(0.8)
        assert len(result.per_expert_outputs) == 1

    def test_aggregate_multiple_experts(self) -> None:
        agg = _MeanAggregator()
        outputs = [
            ExpertOutput("exp1", ["visual"], 30.0, 0.9),
            ExpertOutput("exp2", ["audio"], 34.0, 0.6),
        ]
        result = agg.aggregate(outputs)
        assert result.predicted_age == pytest.approx(32.0)
        assert result.confidence == pytest.approx(0.75)
        assert len(result.per_expert_outputs) == 2

    def test_per_expert_outputs_are_preserved(self) -> None:
        agg = _MeanAggregator()
        o1 = ExpertOutput("e1", ["visual"], 20.0, 0.5)
        o2 = ExpertOutput("e2", ["audio"], 40.0, 0.5)
        result = agg.aggregate([o1, o2])
        assert o1 in result.per_expert_outputs
        assert o2 in result.per_expert_outputs

    def test_load_weights_default_noop(self) -> None:
        """Default load_weights should not raise and do nothing."""
        agg = _MeanAggregator()
        agg.load_weights("any/path.pt")  # should not raise

    def test_load_weights_override(self) -> None:
        agg = _FixedAggregator()
        assert agg._weights_loaded is False
        agg.load_weights("combiner.pt")
        assert agg._weights_loaded is True

    def test_fixed_aggregator_returns_constant(self) -> None:
        agg = _FixedAggregator()
        outputs = [ExpertOutput("e1", ["visual"], 10.0, 0.3)]
        result = agg.aggregate(outputs)
        assert result.predicted_age == 42.0
        assert result.confidence == 1.0

    def test_get_info_default(self) -> None:
        agg = _MeanAggregator()
        info = agg.get_info()
        assert "_MeanAggregator" in str(info["aggregator_class"])

    def test_aggregator_registry_instance(self) -> None:
        assert aggregator_registry.name == "aggregators"

    def test_register_and_resolve_aggregator(self) -> None:
        aggregator_registry.register_class("_mean_agg", _MeanAggregator, overwrite=True)
        cls = aggregator_registry.resolve("_mean_agg")
        assert cls is _MeanAggregator

    def test_result_confidence_within_bounds(self) -> None:
        """Aggregated confidence must never exceed 1.0."""
        agg = _MeanAggregator()
        outputs = [
            ExpertOutput("e1", ["visual"], 25.0, 1.0),
            ExpertOutput("e2", ["audio"], 30.0, 1.0),
        ]
        result = agg.aggregate(outputs)
        assert 0.0 <= result.confidence <= 1.0
