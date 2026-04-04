"""Tests for built-in aggregators with ``ExpertOutput.confidence == -1.0``."""

from __future__ import annotations

import pytest

from apmoe.aggregation.builtin import (
    ConfidenceWeightedAggregator,
    MedianAggregator,
    WeightedAverageAggregator,
)
from apmoe.core.types import ExpertOutput


class TestBuiltinAggregatorsSentinelConfidence:
    def test_weighted_average_maps_negative_one_to_zero_for_confidence(self) -> None:
        agg = WeightedAverageAggregator()
        outputs = [
            ExpertOutput("face_age_expert", ["image"], 30.0, -1.0),
            ExpertOutput("keystroke_age_expert", ["keystroke"], 50.0, 0.8),
        ]
        pred = agg.aggregate(outputs)
        assert pred.predicted_age == pytest.approx(40.0)
        assert pred.confidence == pytest.approx(0.4)

    def test_confidence_weighted_falls_back_when_all_negative_one(self) -> None:
        agg = ConfidenceWeightedAggregator()
        outputs = [
            ExpertOutput("a", ["x"], 30.0, -1.0),
            ExpertOutput("b", ["y"], 40.0, -1.0),
        ]
        pred = agg.aggregate(outputs)
        assert pred.predicted_age == pytest.approx(35.0)
        assert pred.confidence == 0.0

    def test_median_mean_confidence_ignores_negative_one(self) -> None:
        agg = MedianAggregator()
        outputs = [
            ExpertOutput("a", ["x"], 10.0, -1.0),
            ExpertOutput("b", ["y"], 30.0, 1.0),
        ]
        pred = agg.aggregate(outputs)
        assert pred.predicted_age == 20.0
        assert pred.confidence == pytest.approx(1.0)
