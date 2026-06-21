# Pipeline Types (`apmoe.core.types`)

These dataclasses are the shared contracts between processors, cleaners, anonymizers, embedders, experts, aggregators, the CLI, and the HTTP layer.

```text
Import path: apmoe.core.types
Re-exported: apmoe
Source: src/apmoe/core/types.py
```

## Type Flow Summary

```text
raw bytes / JSON value
  -> ModalityProcessor
  -> ModalityData
  -> CleanerStrategy
  -> AnonymizerStrategy
  -> optional EmbedderStrategy
  -> ModalityData or EmbeddingResult
  -> ExpertPlugin.predict()
  -> ExpertOutput
  -> AggregatorStrategy.aggregate()
  -> Prediction
```

## `ModalityData`

```python
@dataclass
class ModalityData:
    modality: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    source: str | None = None
```

`ModalityData` wraps one modality payload before, or instead of, embedding. The concrete `data` type depends on the processor.

| Modality | Typical built-in payload |
|---|---|
| `image` | Pillow/array/tensor-ready image data after validation and preprocessing. |
| `keystroke` | Parsed timing features or normalized keystroke session data. |
| custom modalities | Any type accepted by the matching expert. |

### `with_data(new_data) -> ModalityData`

Returns a copy with `data` replaced and metadata preserved.

```python
def clean(self, data: ModalityData) -> ModalityData:
    return data.with_data(my_cleaning_fn(data.data))
```

## `EmbeddingResult`

```python
@dataclass
class EmbeddingResult:
    modality: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_dim: int = 0
```

Produced by `EmbedderStrategy.embed()`. `embedding_dim` is inferred from `embedding.shape[-1]` when not supplied.

## `ProcessedInput`

```python
ProcessedInput = ModalityData | EmbeddingResult
```

Experts receive a mapping of modality name to processed input:

```python
def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
    image = inputs["image"]
    # Branch on ModalityData vs EmbeddingResult when your expert supports both.
```

## `ExpertOutput`

```python
@dataclass
class ExpertOutput:
    expert_name: str
    consumed_modalities: list[str]
    predicted_age: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)
```

| Field | Description |
|---|---|
| `expert_name` | Should match the configured expert instance name. |
| `consumed_modalities` | Modalities used by the expert for this prediction. |
| `predicted_age` | Age estimate in years. |
| `confidence` | A score in `[0.0, 1.0]`, or `-1.0` when the expert does not report calibrated confidence. |
| `metadata` | Optional observability details such as latency, model version, logits, or feature coverage. |

`FaceAgeExpert` reports `-1.0` because its Keras regressor does not emit calibrated confidence. Aggregators treat `-1.0` as not reported.

## `Prediction`

```python
@dataclass
class Prediction:
    predicted_age: float
    confidence: float
    confidence_interval: tuple[float, float] | None = None
    per_expert_outputs: list[ExpertOutput] = field(default_factory=list)
    skipped_experts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
```

The final output returned by `APMoEApp.predict()` and serialized by `POST /v1/predict`.

| Field | Description |
|---|---|
| `predicted_age` | Aggregated age estimate in years. |
| `confidence` | Aggregated confidence in `[0.0, 1.0]`. |
| `confidence_interval` | Optional `(lower, upper)` bounds. |
| `per_expert_outputs` | Individual expert results for debugging and explainability. |
| `skipped_experts` | Experts not run because required modalities were unavailable. |
| `metadata` | Pipeline latency, available modalities, failed modalities, failed experts, fallback metadata, and other framework details. |

## Validation

- `ExpertOutput.confidence` must be in `[0.0, 1.0]` or exactly `-1.0`.
- `Prediction.confidence` must remain in `[0.0, 1.0]`.
- `confidence_interval` must have `lower <= upper`.

## Importing

```python
from apmoe import ModalityData, EmbeddingResult, ProcessedInput, ExpertOutput, Prediction
from apmoe.core.types import ModalityData, EmbeddingResult, ProcessedInput, ExpertOutput, Prediction
```
