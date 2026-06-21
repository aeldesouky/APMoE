# Inference Pipeline (`apmoe.core.pipeline`)

`InferencePipeline` owns the runtime path from raw multimodal input to the final `Prediction`. Most projects use it through `APMoEApp.from_config()`, but understanding the pipeline helps when writing extension points, debugging config, or testing components in isolation.

## Key Classes

| Class | Role |
|---|---|
| `ModalityChain` | Bundles one modality processor plus cleaner, anonymizer, and optional embedder. |
| `InferencePipeline` | Executes modality processing, expert dispatch, aggregation, and hooks. |

## ModalityChain

```python
from apmoe.core.pipeline import ModalityChain

chain = ModalityChain(
    processor=MyImageProcessor(),
    cleaner=MyImageCleaner(),
    anonymizer=MyImageAnonymizer(),
    embedder=None,
)
```

| Field | Type | Required | Description |
|---|---|---|---|
| `processor` | `ModalityProcessor` | yes | Validates and preprocesses raw input into `ModalityData`. |
| `cleaner` | `CleanerStrategy` | yes | Cleans the `ModalityData`. |
| `anonymizer` | `AnonymizerStrategy` | yes | Removes or masks sensitive information. |
| `embedder` | `EmbedderStrategy | None` | no | Produces an `EmbeddingResult`; when absent, the modality output remains `ModalityData`. |

## Construction

```python
from apmoe.core.pipeline import InferencePipeline

pipeline = InferencePipeline(
    chains={"image": image_chain, "keystroke": keystroke_chain},
    expert_registry=expert_reg,
    aggregator=my_aggregator,
)

prediction = pipeline.run({"image": image_bytes, "keystroke": keystroke_payload})
```

## Execution Flow

```text
raw_inputs
  -> for each configured modality present in the request:
       processor.validate()
       processor.preprocess()
       cleaner.clean()
       anonymizer.anonymize()
       optional embedder.embed()
  -> processed inputs keyed by modality name
  -> runnable experts whose declared modalities are available
  -> ExpertOutput records
  -> aggregator.aggregate()
  -> Prediction
```

## Modality Processing

For each request key that matches a configured chain, the pipeline runs:

1. `processor.validate(raw_data)`
2. `processor.preprocess(raw_data)`
3. `cleaner.clean(ModalityData)`
4. `anonymizer.anonymize(ModalityData)`
5. `embedder.embed(ModalityData)` when configured

Input keys for modalities not in `chains` are ignored so clients can pass a superset of values safely.

## Expert Inference

After processing, the pipeline asks the expert registry for experts whose required modalities are available. Each runnable expert receives only its declared subset of inputs.

If no expert can run, the pipeline raises `PipelineError`. If a runnable expert raises, behavior depends on `expert_failure_policy`: `fail_fast` aborts the request, while `skip_failed` aggregates successful expert outputs and records failed experts in metadata.

## Graceful Degradation

The pipeline can still return a prediction when one modality is missing or fails, as long as at least one expert can run.

```python
prediction = pipeline.run({"keystroke": keystroke_payload})
print(prediction.skipped_experts)                  # e.g. ["face_age_expert"]
print(prediction.metadata["available_modalities"]) # ["keystroke"]
```

Modality failures are recorded in `Prediction.metadata["failed_modalities"]`.

## Hooks

The pipeline exposes four hook lists for logging or metrics:

| Hook list | Signature | Fires when |
|---|---|---|
| `on_before_process` | `(modality_name, raw_data)` | Before each modality chain starts. |
| `on_after_embed` | `(modality_name, result)` | After each modality chain finishes. |
| `on_after_expert` | `(output)` | After an expert returns. |
| `on_after_aggregate` | `(prediction)` | After aggregation returns. |

```python
pipeline.on_after_expert.append(
    lambda output: logger.info("%s age=%.1f", output.expert_name, output.predicted_age)
)
```

## Sync and Async Execution

```python
prediction = pipeline.run({"image": image_bytes})
prediction = await pipeline.run_async({"image": image_bytes, "keystroke": keystroke_payload})
```

`run_async()` processes modality chains concurrently through `asyncio.gather` and a thread-pool executor. Expert inference runs after modality processing completes.

## Prediction Metadata

The pipeline adds framework metadata to the final prediction:

| Key | Type | Description |
|---|---|---|
| `pipeline_latency_s` | `float` | Wall-clock pipeline latency. |
| `available_modalities` | `list[str]` | Modalities that processed successfully. |
| `failed_modalities` | `dict[str, str]` | Modality processing failures. |
| `failed_experts` | `dict[str, str]` | Runnable expert failures when `expert_failure_policy="skip_failed"`. |

## Direct Usage

```python
from apmoe.aggregation.base import aggregator_registry
from apmoe.core.pipeline import InferencePipeline, ModalityChain
from apmoe.experts.registry import ExpertRegistry

chain = ModalityChain(
    processor=MyImageProcessor(),
    cleaner=MyCleaner(),
    anonymizer=MyAnonymizer(),
    embedder=None,
)

registry = ExpertRegistry()
expert = MyExpert()
expert.load_weights("weights/my_expert.pt")
registry.register_instance(expert)

agg = aggregator_registry.resolve("weighted_average")()

pipeline = InferencePipeline(
    chains={"image": chain},
    expert_registry=registry,
    aggregator=agg,
)

prediction = pipeline.run({"image": image_bytes})
```

## See Also

- [app.md](app.md)
- [modality processor extension point](../extension-points/modality-processor.md)
- [processing strategies](../extension-points/processing-strategies.md)
- [expert plugins](../extension-points/expert-plugin.md)
- [aggregators](../extension-points/aggregator.md)
- [testing strategy](../testing.md)
