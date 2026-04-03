# APMoE Developer Documentation

APMoE is an **inference-only, Inversion-of-Control framework** for age prediction
using a Mixture of Experts (MoE) architecture.

> **Hollywood Principle:** *Don't call us, we'll call you.*
> You extend abstract base classes and register your components.
> The framework owns the execution lifecycle — it loads your weights,
> wires the pipeline, and orchestrates every prediction.

---

## How it works

```
                         ┌─────────────────────────────────┐
                         │  Your code (extension points)   │
                         │                                 │
                         │  MyVisualProcessor              │
                         │  MyImageCleaner                 │
                         │  MyAgeExpert                    │
                         │  MyAggregator                   │
                         └────────────┬────────────────────┘
                                      │ register / subclass
                         ┌────────────▼────────────────────┐
                         │     APMoE Framework             │
                         │                                 │
                         │  loads config.json              │
                         │  bootstraps pipeline            │
                         │  runs inference                 │
                         │  serves HTTP API                │
                         └─────────────────────────────────┘
```

The framework reads your `config.json`, imports and instantiates every
registered component, loads pretrained weights, and exposes a `/predict`
endpoint — all without you writing any glue code.

---

## Pipeline data flow

```
Raw input (bytes / file)
        │
        ▼  ModalityProcessor.preprocess()
  ModalityData
        │
        ▼  CleanerStrategy.clean()
  ModalityData  (cleaned)
        │
        ▼  AnonymizerStrategy.anonymize()
  ModalityData  (anonymised)
        │
        ▼  EmbedderStrategy.embed()   ← optional; omit to skip
  EmbeddingResult  OR  ModalityData
        │
        │  ProcessedInput = EmbeddingResult | ModalityData
        │
        ├──► Expert A  ──► ExpertOutput (predicted_age, confidence)
        ├──► Expert B  ──► ExpertOutput
        └──► Expert C  ──► ExpertOutput
                  │
                  ▼  AggregatorStrategy.aggregate()
              Prediction  (final answer)
```

---

## Documentation map

| Document | What it covers |
|---|---|
| [configuration.md](configuration.md) | Full JSON config reference and environment variable overrides |
| [core/types.md](core/types.md) | Pipeline data types: `ModalityData`, `EmbeddingResult`, `ExpertOutput`, `Prediction` |
| [core/exceptions.md](core/exceptions.md) | Exception hierarchy — when each error is raised and how to handle it |
| [core/registry.md](core/registry.md) | `Registry[T]` — registering and resolving components |
| [core/pipeline.md](core/pipeline.md) | `InferencePipeline` + `ModalityChain` — the two-phase execution loop *(Phase 3)* |
| [core/app.md](core/app.md) | `APMoEApp` — IoC container, bootstrap lifecycle, inference API *(Phase 3)* |
| [serving.md](serving.md) | FastAPI serving layer — routes, middleware order, auth and rate-limit behavior *(Phase 4)* |
| [cli.md](cli.md) | CLI reference — `init`, `serve`, `predict`, `validate`, exit/error behavior *(Phase 5)* |
| [new-code-analysis.md](new-code-analysis.md) | High-level analysis of newly added app, serving, and CLI code paths |
| [extension-points/index.md](extension-points/index.md) | Overview of every extension point and the IoC contract |
| [extension-points/modality-processor.md](extension-points/modality-processor.md) | How to implement `ModalityProcessor` |
| [extension-points/processing-strategies.md](extension-points/processing-strategies.md) | How to implement `CleanerStrategy`, `AnonymizerStrategy`, `EmbedderStrategy` |
| [extension-points/expert-plugin.md](extension-points/expert-plugin.md) | How to implement `ExpertPlugin` |
| [extension-points/aggregator.md](extension-points/aggregator.md) | How to implement `AggregatorStrategy` |
| [testing.md](testing.md) | Testing strategy — unit, boundary, and integration test layers |

---

## Quickstart

**1. Install**

```bash
uv add apmoe
# or
pip install apmoe
```

**2. Write a config file**

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "visual",
        "processor": "myproject.processors.VisualProcessor",
        "pipeline": {
          "cleaner":    "myproject.cleaners.ImageCleaner",
          "anonymizer": "myproject.anonymizers.FaceAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "face_expert",
        "class":      "myproject.experts.FaceExpert",
        "weights":    "./weights/face.pt",
        "modalities": ["visual"]
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
    }
  }
}
```

**3. Implement the extension points**

```python
# myproject/processors.py
from apmoe.modality.base import ModalityProcessor
from apmoe.core.types import ModalityData

class VisualProcessor(ModalityProcessor):
    def validate(self, data: bytes) -> bool:
        return len(data) > 0

    def preprocess(self, data: bytes) -> ModalityData:
        # decode, resize, normalise ...
        return ModalityData(modality="visual", data=tensor)
```

```python
# myproject/experts.py
from apmoe.experts.base import ExpertPlugin
from apmoe.core.types import ProcessedInput, ExpertOutput

class FaceExpert(ExpertPlugin):
    @classmethod
    def declared_modalities(cls) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        self.model = torch.load(path)

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        age = float(self.model(inputs["visual"].data))
        return ExpertOutput("face_expert", ["visual"], age, confidence=0.9)
```

**4. Bootstrap and predict**

```python
from apmoe import APMoEApp

# One call wires the entire pipeline and loads all weights
app = APMoEApp.from_config("config.json")

# Run inference
prediction = app.predict({"visual": image_bytes})
print(prediction.predicted_age)    # e.g. 34.2
print(prediction.confidence)       # e.g. 0.87

# Async variant (inside FastAPI / asyncio)
prediction = await app.predict_async({"visual": image_bytes})

# Health check (weight files, expert liveness)
report = app.validate()

# Inspect what was loaded
info = app.get_info()
print(info["modalities"])  # ["visual"]
print(info["experts"])     # ["face_expert"]
```

**5. Serve over HTTP**

```bash
apmoe serve --config config.json
# → http://localhost:8000/predict
# → http://localhost:8000/health
# → http://localhost:8000/info
```

---

## Key design constraints

1. **No pre-prediction fusion.** There is no layer that merges all modality
   embeddings into one representation before experts see them. Each expert
   receives only the modalities it declares and predicts independently.

2. **Experts are not restricted to a single modality.** An expert may declare
   `["visual"]`, `["audio"]`, or `["visual", "audio"]`. Multi-modal experts
   handle their own internal combination.

3. **Embedding is optional per modality.** Omit `pipeline.embedder` in config
   and your expert receives a `ModalityData` (preprocessed tensor). Include it
   and your expert receives an `EmbeddingResult` (feature vector). This lets
   experts that do their own feature extraction skip the embedding step.

4. **Pretrained models only.** The framework loads and runs weights; it does
   not train them. All `load_weights()` calls are one-time operations at
   bootstrap.
