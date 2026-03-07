# Inference Pipeline (`apmoe.core.pipeline`)

The `InferencePipeline` is the heart of the APMoE runtime. It owns the complete
two-phase execution loop — from raw multi-modal input to a final
`Prediction` — and is wired automatically by `APMoEApp.from_config()`.

You will not instantiate `InferencePipeline` directly in most projects. However,
understanding it is essential for:

- Writing custom hooks for observability or logging.
- Testing components in isolation without the full app bootstrap.
- Implementing advanced serving patterns (e.g. batching, streaming).

---

## Key classes

| Class | Role |
|---|---|
| `ModalityChain` | Bundles all processing components for a single modality |
| `InferencePipeline` | Executes the two-phase inference loop |

---

## `ModalityChain`

A `ModalityChain` is a plain dataclass that groups the four components of one
modality's processing chain:

```python
from apmoe.core.pipeline import ModalityChain

chain = ModalityChain(
    processor=MyVisualProcessor(),   # ModalityProcessor
    cleaner=MyImageCleaner(),        # CleanerStrategy
    anonymizer=MyFaceAnonymizer(),   # AnonymizerStrategy
    embedder=MyMobileNetEmbedder(),  # EmbedderStrategy | None
)
```

| Field | Type | Required | Description |
|---|---|---|---|
| `processor` | `ModalityProcessor` | ✅ | Validates and preprocesses raw input into `ModalityData`. |
| `cleaner` | `CleanerStrategy` | ✅ | Cleans the `ModalityData` (denoising, normalisation, etc.). |
| `anonymizer` | `AnonymizerStrategy` | ✅ | Removes PII from the cleaned `ModalityData`. |
| `embedder` | `EmbedderStrategy \| None` | ❌ | Produces an `EmbeddingResult`. When `None`, the pipeline output for this modality is the preprocessed `ModalityData`. |

`APMoEApp.from_config()` constructs one `ModalityChain` per modality entry in
config and passes the full dict to `InferencePipeline`.

---

## `InferencePipeline`

```python
from apmoe.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    chains={"visual": visual_chain, "audio": audio_chain},
    expert_registry=expert_reg,
    aggregator=my_aggregator,
)
prediction = pipeline.run({"visual": b"...", "audio": b"..."})
```

### Constructor

| Parameter | Type | Description |
|---|---|---|
| `chains` | `dict[str, ModalityChain]` | Mapping of modality name → chain. |
| `expert_registry` | `ExpertRegistry` | Live registry of expert instances with weights loaded. |
| `aggregator` | `AggregatorStrategy` | Combines all expert outputs into a single `Prediction`. |
| `on_before_process` | `list[callable]` | Hook list (see [Hooks](#hooks)). |
| `on_after_embed` | `list[callable]` | Hook list. |
| `on_after_expert` | `list[callable]` | Hook list. |
| `on_after_aggregate` | `list[callable]` | Hook list. |

---

## Execution flow

```
raw_inputs: dict[str, Any]
        │
        │  Phase A — Modality Processing (one branch per modality, parallel in async)
        │
        ├── "visual" branch ──► processor.validate()
        │                   ──► processor.preprocess()   → ModalityData
        │                   ──► cleaner.clean()          → ModalityData
        │                   ──► anonymizer.anonymize()   → ModalityData
        │                   ──► embedder.embed()         → EmbeddingResult  (if configured)
        │                                                  ModalityData      (if no embedder)
        │
        ├── "audio" branch  ──► (same steps)
        │
        └── "eeg" branch    ──► (same steps)
                │
                │  Phase B — Expert Inference + Aggregation
                │
                │  processed: dict[str, ProcessedInput]
                │
                ├── Expert A (declares ["visual"])
                │   receives {"visual": ProcessedInput}
                │   returns  ExpertOutput(predicted_age, confidence)
                │
                ├── Expert B (declares ["audio"])
                │   receives {"audio": ProcessedInput}
                │   returns  ExpertOutput
                │
                ├── Expert C (declares ["visual", "audio"])
                │   receives {"visual": ProcessedInput, "audio": ProcessedInput}
                │   returns  ExpertOutput
                │
                ▼
         aggregator.aggregate([ExpertOutput, ExpertOutput, ExpertOutput])
                │
                ▼
           Prediction  (predicted_age, confidence, per_expert_outputs, skipped_experts, metadata)
```

---

## Phase A — Modality processing

For each key in `raw_inputs` that has a corresponding entry in `chains`, the
pipeline runs the full processing chain:

1. `processor.validate(raw_data)` — returns `bool`. If `False` or if an exception
   is raised, the modality is recorded in `Prediction.metadata["failed_modalities"]`
   and excluded from the processed dict (graceful degradation, see below).
2. `processor.preprocess(raw_data)` → `ModalityData`
3. `cleaner.clean(ModalityData)` → `ModalityData`
4. `anonymizer.anonymize(ModalityData)` → `ModalityData`
5. *(optional)* `embedder.embed(ModalityData)` → `EmbeddingResult`

Input keys for modalities **not** in `chains` are silently ignored. This allows
callers to pass a superset of inputs without error.

---

## Phase B — Expert inference

After Phase A, the pipeline holds `dict[str, ProcessedInput]` — a mapping of
modality name to whatever the chain produced (an `EmbeddingResult` if an embedder
was configured, or a `ModalityData` otherwise).

For each expert in the registry:

1. `expert_registry.get_runnable_experts(available_modalities)` — returns only the
   experts whose **entire** `declared_modalities()` list is present in the processed
   dict.
2. For each runnable expert: `expert.predict(inputs_subset)` where `inputs_subset`
   is a dict containing only the modalities that expert declared.
3. Each `ExpertOutput` is passed to the configured hooks.

Experts whose required modalities are all absent or failed are listed in
`Prediction.skipped_experts`.

If **no expert** produces an output (all skipped), the pipeline raises
`PipelineError` rather than calling the aggregator with an empty list.

---

## Graceful degradation

The pipeline is designed to produce **partial results** rather than fail completely
when some modalities are unavailable:

```
Request contains: {"visual": ..., "audio": ...}  ← audio processing fails

Processed dict:   {"visual": <ProcessedInput>}    ← only visual succeeded

Runnable experts:  [VisualExpert]                  ← AudioExpert, MultiModalExpert skipped
Skipped experts:   ["audio_expert", "multi_expert"]
```

The final `Prediction` includes:

```python
prediction.skipped_experts           # ["audio_expert", "multi_expert"]
prediction.metadata["failed_modalities"]  # {"audio": "Validation failed..."}
prediction.metadata["available_modalities"]  # ["visual"]
```

**Exceptions that cause a modality to be skipped:**

- `processor.validate()` returns `False`.
- `processor.validate()` raises any exception.
- `cleaner.clean()`, `anonymizer.anonymize()`, or `embedder.embed()` raise any
  exception (wrapped in `ModalityError` if not already one).

**Expert exceptions are NOT silently degraded.** If `expert.predict()` raises, the
pipeline re-raises (wrapped in `ExpertError`) and the request fails entirely. This
is intentional: a broken expert is a code bug; a missing modality is a runtime
condition.

---

## Hooks

The pipeline exposes four hook lists. Add any number of callables; they fire in
order for every call to `run()` or `run_async()`:

```python
# Log every modality as it starts processing
pipeline.on_before_process.append(
    lambda name, raw_data: logger.info("Processing %s", name)
)

# Record embedding dimensions in metrics
pipeline.on_after_embed.append(
    lambda name, result: metrics.gauge("embed_dim", result.embedding_dim
                                       if hasattr(result, "embedding_dim") else 0)
)

# Trace each expert prediction
pipeline.on_after_expert.append(
    lambda output: logger.info("Expert %s: age=%.1f conf=%.2f",
                               output.expert_name, output.predicted_age,
                               output.confidence)
)

# Record final latency
pipeline.on_after_aggregate.append(
    lambda pred: metrics.histogram("pipeline_latency",
                                   pred.metadata["pipeline_latency_s"])
)
```

Hook signatures:

| Hook list | Signature | Fires when |
|---|---|---|
| `on_before_process` | `(modality_name: str, raw_data: Any) -> None` | Before each modality's chain starts |
| `on_after_embed` | `(modality_name: str, result: ProcessedInput) -> None` | After each modality's chain finishes |
| `on_after_expert` | `(output: ExpertOutput) -> None` | After each expert's `predict()` returns |
| `on_after_aggregate` | `(prediction: Prediction) -> None` | After the aggregator returns |

Hooks fire even during graceful degradation for the modalities that **did**
succeed. They do not fire for failed/missing modalities.

---

## Synchronous execution (`run`)

```python
prediction = pipeline.run({"visual": image_bytes, "audio": audio_bytes})
```

Phase A processes modalities **sequentially** in iteration order of `raw_inputs`.
Use this mode for single-threaded services or when individual processing steps
already use multi-threading internally (e.g. PyTorch DataLoader workers).

---

## Asynchronous execution (`run_async`)

```python
prediction = await pipeline.run_async({"visual": image_bytes, "audio": audio_bytes})
```

Phase A runs each modality's chain concurrently using `asyncio.gather` and a
thread-pool executor. Since processing chains are CPU-bound (not natively async),
each chain runs in a separate thread. Use this in async web frameworks (FastAPI)
to make full use of parallelism across modalities.

Phase B (expert inference) always runs sequentially after all Phase A tasks
complete.

---

## `Prediction` metadata

Both `run()` and `run_async()` enrich the `Prediction.metadata` dict with
pipeline-level information:

| Key | Type | Description |
|---|---|---|
| `pipeline_latency_s` | `float` | Total wall-clock time from `run()` entry to `aggregate()` return, in seconds. |
| `available_modalities` | `list[str]` | Sorted list of modalities that were successfully processed. |
| `failed_modalities` | `dict[str, str]` | Modalities that failed during Phase A, mapped to their error message. |

---

## Direct usage (advanced)

If you need to use the pipeline without `APMoEApp` (e.g. in a custom serving
layer or test harness), wire it manually:

```python
from apmoe.core.pipeline import InferencePipeline, ModalityChain
from apmoe.experts.registry import ExpertRegistry
from apmoe.aggregation.base import aggregator_registry

# Build chains
chain = ModalityChain(
    processor=MyProcessor(),
    cleaner=MyCleaner(),
    anonymizer=MyAnonymizer(),
    embedder=MyEmbedder(),   # or None
)

# Build expert registry
registry = ExpertRegistry()
expert = MyExpert()
expert.load_weights("weights/my_expert.pt")
registry.register_instance(expert)

# Resolve aggregator
agg = aggregator_registry.resolve("apmoe.aggregation.builtin.WeightedAverageAggregator")()

# Wire pipeline
pipeline = InferencePipeline(
    chains={"visual": chain},
    expert_registry=registry,
    aggregator=agg,
)

# Run
prediction = pipeline.run({"visual": image_bytes})
```

---

## See also

- [app.md](app.md) — `APMoEApp` wires the pipeline automatically from config
- [../extension-points/modality-processor.md](../extension-points/modality-processor.md) — implementing a processor
- [../extension-points/processing-strategies.md](../extension-points/processing-strategies.md) — implementing cleaner / anonymizer / embedder
- [../extension-points/expert-plugin.md](../extension-points/expert-plugin.md) — implementing an expert
- [../extension-points/aggregator.md](../extension-points/aggregator.md) — implementing an aggregator
- [../testing.md](../testing.md) — how the pipeline is tested
