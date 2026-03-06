# Extension Points

APMoE is a framework, not a library. You extend it by subclassing abstract
base classes (ABCs) and declaring them in config — the framework calls your
code; you never call the framework's orchestration logic directly.

---

## The IoC contract

```
You provide:                        Framework calls:
────────────────────────────────    ──────────────────────────────────
class MyProcessor(ModalityProcessor)  → processor.validate(raw)
                                      → processor.preprocess(raw)

class MyCleaner(CleanerStrategy)      → cleaner.clean(modality_data)

class MyAnonymizer(AnonymizerStrategy)→ anonymizer.anonymize(modality_data)

class MyEmbedder(EmbedderStrategy)    → embedder.embed(modality_data)
  (optional)

class MyExpert(ExpertPlugin)          → expert.load_weights(path)   ← once
                                      → expert.predict(inputs)      ← per request

class MyAggregator(AggregatorStrategy)→ aggregator.aggregate(outputs)
```

You never instantiate these yourself. Register them in config; the framework
does the rest.

---

## Extension points at a glance

| ABC | Module | Config location | Docs |
|---|---|---|---|
| `ModalityProcessor` | `apmoe.modality.base` | `modalities[].processor` | [modality-processor.md](modality-processor.md) |
| `CleanerStrategy` | `apmoe.processing.base` | `modalities[].pipeline.cleaner` | [processing-strategies.md](processing-strategies.md) |
| `AnonymizerStrategy` | `apmoe.processing.base` | `modalities[].pipeline.anonymizer` | [processing-strategies.md](processing-strategies.md) |
| `EmbedderStrategy` | `apmoe.processing.base` | `modalities[].pipeline.embedder` | [processing-strategies.md](processing-strategies.md) |
| `ExpertPlugin` | `apmoe.experts.base` | `experts[].class` | [expert-plugin.md](expert-plugin.md) |
| `AggregatorStrategy` | `apmoe.aggregation.base` | `aggregation.strategy` | [aggregator.md](aggregator.md) |

---

## Registration

Every component must be **importable** from the dotted path you put in config.
There are two ways to make this work:

### Option A — Dotted path (no explicit registration needed)

```json
{ "processor": "myproject.processors.MyVisualProcessor" }
```

The framework calls `importlib.import_module("myproject.processors")` and
retrieves `MyVisualProcessor`. Your package just needs to be on `sys.path`
(i.e. installed or in the working directory).

### Option B — Registered short name

```python
# myproject/processors.py
from apmoe.modality.base import ModalityProcessor, modality_registry

@modality_registry.register("my_visual")
class MyVisualProcessor(ModalityProcessor):
    ...
```

```json
{ "processor": "my_visual" }
```

Use Option B when you want short, readable config keys and are sure the module
is imported before the framework bootstraps (e.g. via an entry point or an
explicit `import myproject.processors` at startup).

---

## Processing chain order

For each modality, the framework calls the following in sequence:

```
ModalityProcessor.preprocess(raw_input)
        ↓
CleanerStrategy.clean(ModalityData)
        ↓
AnonymizerStrategy.anonymize(ModalityData)
        ↓
EmbedderStrategy.embed(ModalityData)    ← only if pipeline.embedder is set
        ↓
ProcessedInput  →  dispatched to each expert that declared this modality
```

Each step produces a new object; the framework does **not** mutate inputs.
Use `ModalityData.with_data()` inside your Cleaner and Anonymizer to follow
the same convention.

---

## Built-in implementations

Phase 6 ships reference implementations for every extension point:

| Extension point | Built-in classes |
|---|---|
| `ModalityProcessor` | `VisualProcessor`, `AudioProcessor`, `EEGProcessor` |
| `CleanerStrategy` | `ImageCleaner`, `AudioCleaner`, `EEGCleaner` |
| `AnonymizerStrategy` | `FaceAnonymizer`, `VoiceAnonymizer`, `EEGAnonymizer` |
| `EmbedderStrategy` | `MobileNetEmbedder`, `MelSpectrogramEmbedder`, `EEGEmbedder` |
| `ExpertPlugin` | `CNNAgeExpert`, `MLPAgeExpert`, `EEGAgeExpert` |
| `AggregatorStrategy` | `WeightedAverageAggregator`, `MedianAggregator`, `ConfidenceWeightedAggregator`, `LearnedCombiner` |
