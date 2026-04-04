# Implementing `AggregatorStrategy`

The aggregator is the final step of the pipeline. It receives all
`ExpertOutput` objects produced in a single request and combines them into
one `Prediction`. This is where the Mixture-of-Experts "gating" logic lives.

```
ABC:         apmoe.aggregation.base.AggregatorStrategy
Registry:    apmoe.aggregation.base.aggregator_registry
Config key:  aggregation.strategy
```

---

## `ExpertOutput.confidence` and `-1.0`

Per-expert `confidence` may be in `[0.0, 1.0]` or exactly **`-1.0`**, meaning the expert does not report a score (e.g. `FaceAgeExpert`, a pure Keras regressor). Built-in aggregators in `apmoe.aggregation.builtin` treat `-1` as **no numeric confidence**: it does not contribute to confidence-weighted blends as a negative value, and means are taken only over experts with real scores in `[0, 1]`. The returned `Prediction.confidence` is always in `[0.0, 1.0]`.

---

## Abstract interface

```python
from abc import ABC, abstractmethod
from apmoe.core.types import ExpertOutput, Prediction

class AggregatorStrategy(ABC):

    @abstractmethod
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        """Combine expert predictions into a single final answer.

        Called once per request after all experts have run. The list
        contains only the outputs of experts that actually ran (experts
        that were skipped due to missing modalities are absent here —
        their names appear in Prediction.skipped_experts separately).

        Args:
            outputs: Non-empty list of ExpertOutput instances. The framework
                     guarantees at least one output; if all experts were
                     skipped it raises PipelineError before calling this.

        Returns:
            A Prediction with predicted_age and confidence in [0.0, 1.0].

        Raises:
            PipelineError: If combination fails unrecoverably.
        """
```

---

## Simple implementations

### Uniform average

```python
from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.types import ExpertOutput, Prediction


@aggregator_registry.register("uniform_average")
class UniformAverageAggregator(AggregatorStrategy):
    """Equal-weight average of all expert age predictions."""

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        ages = [o.predicted_age for o in outputs]
        confs = [o.confidence for o in outputs]
        avg_age = sum(ages) / len(ages)
        avg_conf = sum(confs) / len(confs)
        return Prediction(
            predicted_age=avg_age,
            confidence=avg_conf,
            per_expert_outputs=outputs,
        )
```

### Confidence-weighted average

```python
@aggregator_registry.register("confidence_weighted")
class ConfidenceWeightedAggregator(AggregatorStrategy):
    """Weight each expert's prediction by its self-reported confidence."""

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        total_conf = sum(o.confidence for o in outputs)
        if total_conf == 0:
            # Fall back to uniform if all confidences are zero
            weights = [1.0 / len(outputs)] * len(outputs)
        else:
            weights = [o.confidence / total_conf for o in outputs]

        age = sum(w * o.predicted_age for w, o in zip(weights, outputs))
        conf = sum(w * o.confidence for w, o in zip(weights, outputs))

        return Prediction(
            predicted_age=age,
            confidence=min(conf, 1.0),
            per_expert_outputs=outputs,
        )
```

### Config-driven weighted average

```python
class WeightedAverageAggregator(AggregatorStrategy):
    """Use per-expert weights from the config's aggregation.weights map."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        # weights is populated from AggregationConfig.weights at bootstrap
        self.weights = weights or {}

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        total_w = 0.0
        weighted_age = 0.0
        weighted_conf = 0.0

        for out in outputs:
            w = self.weights.get(out.expert_name, 1.0)
            weighted_age  += w * out.predicted_age
            weighted_conf += w * out.confidence
            total_w       += w

        age  = weighted_age  / total_w
        conf = min(weighted_conf / total_w, 1.0)

        return Prediction(
            predicted_age=age,
            confidence=conf,
            per_expert_outputs=outputs,
        )
```

---

## Learned combiner (small model)

An aggregator can itself hold a pretrained model:

```python
import torch
import torch.nn as nn

class LearnedCombiner(AggregatorStrategy):
    """Small MLP that takes stacked (age, confidence) pairs and outputs age."""

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    def load_weights(self, path: str) -> None:
        """Called by the framework if weights_path is set in config."""
        self.model = torch.load(path, map_location="cpu")
        self.model.eval()

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        assert self.model is not None, "load_weights() not called"
        # Stack into (N, 2) tensor: [predicted_age, confidence]
        features = torch.tensor(
            [[o.predicted_age, o.confidence] for o in outputs],
            dtype=torch.float32,
        ).unsqueeze(0)   # (1, N, 2)

        with torch.inference_mode():
            result = self.model(features).squeeze()

        age  = float(result[0])
        conf = float(result[1].sigmoid())

        return Prediction(
            predicted_age=age,
            confidence=conf,
            per_expert_outputs=outputs,
        )
```

```json
{
  "aggregation": {
    "strategy":     "myproject.aggregators.LearnedCombiner",
    "weights_path": "./weights/combiner.pt"
  }
}
```

---

## Producing a confidence interval

Aggregators may optionally compute a confidence interval:

```python
import numpy as np

def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
    ages = np.array([o.predicted_age for o in outputs])
    mean_age = float(ages.mean())
    std_age  = float(ages.std())

    return Prediction(
        predicted_age=mean_age,
        confidence=0.85,
        confidence_interval=(mean_age - 2 * std_age, mean_age + 2 * std_age),
        per_expert_outputs=outputs,
    )
```

---

## Contract rules

1. `outputs` is guaranteed non-empty — the framework raises `PipelineError`
   before calling `aggregate()` if no experts ran.
2. `predicted_age` has no enforced range, but should be a reasonable age in
   years.
3. `confidence` must be in `[0.0, 1.0]`; `Prediction.__post_init__` validates
   this and raises `ValueError`.
4. `confidence_interval` lower must be ≤ upper.
5. Always populate `per_expert_outputs` — it powers the `/info` breakdown
   and explainability features.

---

## Config wiring

```json
{
  "aggregation": {
    "strategy": "myproject.aggregators.WeightedAverageAggregator",
    "weights": {
      "face_expert":  0.5,
      "audio_expert": 0.3,
      "eeg_expert":   0.2
    }
  }
}
```

Weights do not need to sum to 1 — normalisation is the aggregator's
responsibility. Keys must match the `name` fields in `experts[]`.

---

## Built-in aggregators (Phase 6)

| Class | Path | Description |
|---|---|---|
| `WeightedAverageAggregator` | `apmoe.aggregation.builtin.WeightedAverageAggregator` | Config-driven weights; falls back to uniform. |
| `MedianAggregator` | `apmoe.aggregation.builtin.MedianAggregator` | Median of predicted ages; robust to outlier experts. |
| `ConfidenceWeightedAggregator` | `apmoe.aggregation.builtin.ConfidenceWeightedAggregator` | Weight = expert's self-reported confidence. |
| `LearnedCombiner` | `apmoe.aggregation.builtin.LearnedCombiner` | Pretrained MLP combiner; requires `weights_path`. |
