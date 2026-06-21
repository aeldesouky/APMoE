# Implementing `AggregatorStrategy`

Aggregators are the final step in the pipeline. They receive the `ExpertOutput` records produced for one request and combine them into one `Prediction`.

```text
ABC:        apmoe.aggregation.base.AggregatorStrategy
Registry:   apmoe.aggregation.base.aggregator_registry
Config key: aggregation.strategy
```

## Confidence Values

Per-expert `confidence` may be in `[0.0, 1.0]` or exactly `-1.0`, where `-1.0` means the expert does not report calibrated confidence. `FaceAgeExpert` uses `-1.0` because its Keras regressor returns only an age value.

Built-in aggregators treat `-1.0` as "not reported" rather than as a negative score.

## Interface

```python
from abc import ABC, abstractmethod
from apmoe.core.types import ExpertOutput, Prediction

class AggregatorStrategy(ABC):
    @abstractmethod
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        """Combine non-empty expert outputs into a final prediction."""
```

The framework raises `PipelineError` before calling the aggregator if no expert ran.

## Simple Custom Aggregator

```python
from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.types import ExpertOutput, Prediction

@aggregator_registry.register("uniform_average")
class UniformAverageAggregator(AggregatorStrategy):
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        ages = [o.predicted_age for o in outputs]
        confidences = [o.confidence for o in outputs if 0.0 <= o.confidence <= 1.0]
        return Prediction(
            predicted_age=sum(ages) / len(ages),
            confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
            per_expert_outputs=outputs,
            metadata={"aggregator": "UniformAverageAggregator"},
        )
```

## Config-Driven Weights

Custom aggregators can read weights through constructor arguments or a setter, matching the built-in weighted average pattern.

```json
{
  "aggregation": {
    "strategy": "myproject.aggregators.WeightedAverageAggregator",
    "weights": {
      "face_age_expert": 0.6,
      "keystroke_age_expert": 0.4
    }
  }
}
```

Weight keys must match `experts[].name` values.

## Learned or Model-Based Aggregators

The current package does not ship a built-in learned combiner. Applications can still implement one as a custom `AggregatorStrategy` and expose it by dotted path or entry point. If the combiner needs weights, load them in your custom class from a path passed through an extra config field.

## Confidence Intervals

Aggregators may optionally populate `confidence_interval`:

```python
return Prediction(
    predicted_age=mean_age,
    confidence=0.85,
    confidence_interval=(lower_age, upper_age),
    per_expert_outputs=outputs,
)
```

## Contract Rules

1. `outputs` is non-empty.
2. `Prediction.confidence` must be in `[0.0, 1.0]`.
3. `confidence_interval` must have `lower <= upper`.
4. Always preserve `per_expert_outputs` for explainability and debugging.
5. Put strategy-specific diagnostics in `Prediction.metadata`.

## Built-In Aggregators

| Registered name | Class path | Description |
|---|---|---|
| `weighted_average` | `apmoe.aggregation.builtin.WeightedAverageAggregator` | Weighted average of predicted ages; falls back to uniform weights. |
| `confidence_weighted` | `apmoe.aggregation.builtin.ConfidenceWeightedAggregator` | Weights predictions by reported expert confidence. |
| `median` | `apmoe.aggregation.builtin.MedianAggregator` | Uses the median predicted age and ignores outlier ages. |
