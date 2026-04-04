"""Built-in aggregation strategies for the APMoE framework.

After all experts have produced their individual
:class:`~apmoe.core.types.ExpertOutput` objects, an
:class:`~apmoe.aggregation.base.AggregatorStrategy` combines them into a
single final :class:`~apmoe.core.types.Prediction`.

Currently provided:

* :class:`WeightedAverageAggregator` — combines predictions using
  per-expert weights from config (equal weights when none are given).
* :class:`ConfidenceWeightedAggregator` — weights each expert by its
  own reported confidence score (no config weights needed).
* :class:`MedianAggregator` — takes the median predicted age; robust to
  outlier experts.

All strategies are registered with
:data:`~apmoe.aggregation.base.aggregator_registry`.
"""

from __future__ import annotations

import statistics

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.types import ExpertOutput, Prediction


def _scalar_in_unit_interval(confidence: float) -> float:
    """Return *confidence* if it is a real score in ``[0, 1]``; else ``0.0``.

    :class:`~apmoe.core.types.ExpertOutput` uses ``-1.0`` to mean “confidence
    not reported”; that value must not pull numeric aggregates negative.
    """
    if 0.0 <= confidence <= 1.0:
        return confidence
    return 0.0


def _mean_reported_confidence(outputs: list[ExpertOutput]) -> float:
    """Mean of per-expert confidences in ``[0, 1]``; ignores ``-1`` (unknown)."""
    valid = [o.confidence for o in outputs if 0.0 <= o.confidence <= 1.0]
    if not valid:
        return 0.0
    return min(sum(valid) / len(valid), 1.0)


# ---------------------------------------------------------------------------
# WeightedAverageAggregator
# ---------------------------------------------------------------------------


@aggregator_registry.register("weighted_average")
class WeightedAverageAggregator(AggregatorStrategy):
    """Combine expert predictions using configurable per-expert weights.

    Weights are supplied via the ``aggregation.weights`` config key as a
    mapping of ``expert_name → float``.  Experts not present in the mapping
    receive weight ``1.0`` (uniform fallback).  Weights are normalised to
    sum to 1.0 before use.

    Config example
    --------------
    .. code-block:: json

        "aggregation": {
          "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator",
          "weights": {
            "keystroke_age_expert": 1.0
          }
        }
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialise with optional pre-configured weights.

        Args:
            weights: Mapping of expert name → unnormalised weight.  Passed
                programmatically; in normal operation the framework sets this
                via :meth:`set_weights` after reading the config.
        """
        self._weights: dict[str, float] = weights or {}

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update the per-expert weight mapping at runtime.

        Args:
            weights: New expert-name → weight mapping.
        """
        self._weights = weights

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        """Produce a weighted-average prediction from *outputs*.

        Args:
            outputs: Non-empty list of expert outputs.

        Returns:
            A :class:`~apmoe.core.types.Prediction` with the weighted-average
            age and confidence, plus all expert outputs and a ``"weights_used"``
            metadata entry.
        """
        raw_weights = [
            self._weights.get(o.expert_name, 1.0) for o in outputs
        ]
        total = sum(raw_weights) or 1.0
        norm_weights = [w / total for w in raw_weights]

        predicted_age = sum(
            o.predicted_age * w for o, w in zip(outputs, norm_weights)
        )
        confidence = min(
            sum(
                _scalar_in_unit_interval(o.confidence) * w
                for o, w in zip(outputs, norm_weights)
            ),
            1.0,
        )

        return Prediction(
            predicted_age=predicted_age,
            confidence=confidence,
            per_expert_outputs=list(outputs),
            metadata={
                "aggregator": "WeightedAverageAggregator",
                "weights_used": {
                    o.expert_name: round(w, 4)
                    for o, w in zip(outputs, norm_weights)
                },
            },
        )

    def get_info(self) -> dict[str, object]:
        """Return aggregator metadata."""
        return {
            "aggregator_class": type(self).__qualname__,
            "configured_weights": self._weights,
        }


# ---------------------------------------------------------------------------
# ConfidenceWeightedAggregator
# ---------------------------------------------------------------------------


@aggregator_registry.register("confidence_weighted")
class ConfidenceWeightedAggregator(AggregatorStrategy):
    """Weight each expert by its own reported confidence score.

    No configuration weights are needed.  Experts that are more confident
    in their predictions automatically contribute more to the final estimate.
    Experts with ``confidence == -1.0`` (not reported) contribute weight
    ``0``; if all experts are ``-1`` or zero, the aggregator falls back to a
    uniform average for the age blend. Final ``Prediction.confidence`` is the
    mean of scores in ``[0, 1]`` only, or ``0.0`` if none.
    """

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        """Produce a confidence-weighted prediction from *outputs*.

        Args:
            outputs: Non-empty list of expert outputs.

        Returns:
            A :class:`~apmoe.core.types.Prediction` with confidence-weighted
            age and mean confidence.
        """
        raw_weights = [_scalar_in_unit_interval(o.confidence) for o in outputs]
        total_conf = sum(raw_weights)
        if total_conf <= 0.0:
            # No reported confidences (e.g. all -1): uniform blend, same as all-zero.
            norm_weights = [1.0 / len(outputs)] * len(outputs)
        else:
            norm_weights = [w / total_conf for w in raw_weights]

        predicted_age = sum(
            o.predicted_age * w for o, w in zip(outputs, norm_weights)
        )
        confidence = _mean_reported_confidence(outputs)

        return Prediction(
            predicted_age=predicted_age,
            confidence=confidence,
            per_expert_outputs=list(outputs),
            metadata={"aggregator": "ConfidenceWeightedAggregator"},
        )


# ---------------------------------------------------------------------------
# MedianAggregator
# ---------------------------------------------------------------------------


@aggregator_registry.register("median")
class MedianAggregator(AggregatorStrategy):
    """Use the median predicted age across all experts.

    Robust to outlier experts — a single expert with a wildly incorrect
    prediction has little influence on the final result.
    """

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        """Return the median predicted age and mean confidence.

        Args:
            outputs: Non-empty list of expert outputs.

        Returns:
            A :class:`~apmoe.core.types.Prediction` with median age and
            mean confidence.
        """
        ages = [o.predicted_age for o in outputs]
        predicted_age = statistics.median(ages)
        confidence = _mean_reported_confidence(outputs)

        return Prediction(
            predicted_age=predicted_age,
            confidence=confidence,
            per_expert_outputs=list(outputs),
            metadata={"aggregator": "MedianAggregator"},
        )
