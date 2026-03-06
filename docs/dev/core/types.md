# Pipeline Types (`apmoe.core.types`)

These dataclasses are the currency of the inference pipeline. Every component
— processors, cleaners, anonymisers, embedders, experts, and aggregators —
communicates by producing and consuming these types.

```
Import path:  apmoe.core.types
Re-exported:  apmoe  (top-level package)
Source:       src/apmoe/core/types.py
```

---

## Type flow summary

```
raw bytes / file
      │
      ▼ ModalityProcessor
ModalityData           ← produced here, passed through Cleaner + Anonymizer
      │
      ▼ EmbedderStrategy  (optional)
EmbeddingResult        ← produced when embedder is configured
      │
      │  either of the above ──► ProcessedInput (type alias)
      │
      ▼ ExpertPlugin.predict()
ExpertOutput           ← one per expert
      │
      ▼ AggregatorStrategy.aggregate()
Prediction             ← final result returned to the caller
```

---

## `ModalityData`

```python
@dataclass
class ModalityData:
    modality:  str
    data:      Any
    metadata:  dict[str, Any]   = field(default_factory=dict)
    timestamp: float | None     = None
    source:    str | None       = None
```

Wraps a single modality's payload at any point **before** (or without) the
embedding step. The `data` field is opaque — its concrete type depends on
the modality:

| Modality | Typical `data` type |
|---|---|
| `"visual"` | `torch.Tensor` (C×H×W) or `numpy.ndarray` |
| `"audio"` | `numpy.ndarray` (samples,) or `torch.Tensor` |
| `"eeg"` | `numpy.ndarray` (channels × time) |

### `with_data(new_data) → ModalityData`

Returns a **copy** of the instance with `data` replaced. All metadata fields
(`modality`, `metadata`, `timestamp`, `source`) are preserved. Use this in
Cleaner and Anonymizer implementations to avoid mutating the input:

```python
def clean(self, data: ModalityData) -> ModalityData:
    cleaned_tensor = my_cleaning_fn(data.data)
    return data.with_data(cleaned_tensor)   # ✅ immutable style
```

---

## `EmbeddingResult`

```python
@dataclass
class EmbeddingResult:
    modality:      str
    embedding:     np.ndarray        # shape: (D,) or (N, D)
    metadata:      dict[str, Any]    = field(default_factory=dict)
    embedding_dim: int               = 0   # auto-inferred from embedding.shape[-1]
```

Produced by `EmbedderStrategy.embed()`. Carries a dense numeric feature
vector. `embedding_dim` is inferred automatically from `embedding.shape[-1]`
when not supplied explicitly.

> **Note:** Concrete embedders may store a `torch.Tensor` in `embedding`
> instead of a `numpy.ndarray` — as long as the consuming expert handles it.
> The type annotation is `np.ndarray` by convention; no runtime enforcement.

---

## `ProcessedInput`

```python
ProcessedInput = ModalityData | EmbeddingResult
```

This is a **type alias**, not a class. It represents whatever exits a
modality's processing chain:

- `EmbeddingResult` when `pipeline.embedder` is configured for that modality.
- `ModalityData` when no embedder is configured.

Every expert's `predict()` method receives
`dict[str, ProcessedInput]` — a mapping of modality name to its
corresponding processed output.

Use `isinstance` to branch on the concrete type inside an expert:

```python
def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
    inp = inputs["visual"]
    if isinstance(inp, EmbeddingResult):
        features = inp.embedding          # numpy array, skip CNN
    else:
        features = self.cnn(inp.data)     # run CNN on raw tensor
    age = self.head(features).item()
    return ExpertOutput(self.name, ["visual"], age, confidence=0.88)
```

---

## `ExpertOutput`

```python
@dataclass
class ExpertOutput:
    expert_name:          str
    consumed_modalities:  list[str]
    predicted_age:        float
    confidence:           float           # must be in [0.0, 1.0]
    metadata:             dict[str, Any]  = field(default_factory=dict)
```

Returned by `ExpertPlugin.predict()`. The `confidence` field is validated on
construction — values outside `[0.0, 1.0]` raise `ValueError`.

| Field | Description |
|---|---|
| `expert_name` | Should match the `name` field in config; used by aggregators and in the `Prediction` breakdown. |
| `consumed_modalities` | Which modalities this expert actually used (may be a subset of its declared modalities if some were optional and missing). |
| `predicted_age` | Age estimate in years. No upper or lower bound is enforced. |
| `confidence` | Self-reported certainty in `[0.0, 1.0]`. Used by `ConfidenceWeightedAggregator` and `LearnedCombiner`. |
| `metadata` | Optional key/value pairs for observability (inference latency, internal logits, model version, etc.). |

---

## `Prediction`

```python
@dataclass
class Prediction:
    predicted_age:        float
    confidence:           float                         # [0.0, 1.0]
    confidence_interval:  tuple[float, float] | None   = None
    per_expert_outputs:   list[ExpertOutput]            = field(default_factory=list)
    skipped_experts:      list[str]                     = field(default_factory=list)
    metadata:             dict[str, Any]                = field(default_factory=dict)
```

The final output of the inference pipeline, returned by
`APMoEApp.predict()` and serialised to JSON by the `/predict` endpoint.

| Field | Description |
|---|---|
| `predicted_age` | The framework's best age estimate (years). |
| `confidence` | Aggregated confidence in `[0.0, 1.0]`. |
| `confidence_interval` | Optional `(lower, upper)` bounds; lower must be ≤ upper. |
| `per_expert_outputs` | Individual expert predictions — useful for debugging and explainability. |
| `skipped_experts` | Names of experts that were not run (e.g. required modality missing from input). |
| `metadata` | Framework-level metadata (pipeline version, total wall-clock latency, etc.). |

### Validation

- `confidence` outside `[0.0, 1.0]` raises `ValueError`.
- `confidence_interval` with `lower > upper` raises `ValueError`.
- Equal bounds (`lower == upper`) are accepted (degenerate interval).

---

## Importing

```python
# From the top-level package (recommended)
from apmoe import ModalityData, EmbeddingResult, ProcessedInput, ExpertOutput, Prediction

# Or directly from the module
from apmoe.core.types import ModalityData, EmbeddingResult, ProcessedInput, ExpertOutput, Prediction
```
