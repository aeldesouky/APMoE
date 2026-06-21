# APMoEApp (`apmoe.core.app`)

`APMoEApp` is the IoC container and lifecycle manager for the framework. It reads configuration, resolves component classes, loads pretrained weights, wires the inference pipeline, and exposes prediction, validation, metadata, and serving APIs.

```python
from apmoe import APMoEApp

app = APMoEApp.from_config("configs/multimodal.json")
prediction = app.predict({"image": image_bytes, "keystroke": keystroke_payload})
```

## Responsibilities

| Responsibility | Description |
|---|---|
| Config loading | Calls `load_config()` and validates JSON structure and cross-field rules. |
| Component resolution | Resolves processors, cleaners, anonymizers, embedders, experts, and aggregators through registries or dotted imports. |
| Weight loading | Calls `expert.load_weights(path)` once per local expert during bootstrap. |
| Pipeline wiring | Builds `ModalityChain` objects and passes them to `InferencePipeline`. |
| Inference | Delegates `predict()` and `predict_async()` to the pipeline. |
| Validation | Checks configured weight files and expert health. |
| Serving | Starts the FastAPI/uvicorn HTTP server. |
| Metadata | Returns the running app summary through `get_info()`. |

## Bootstrap Lifecycle

`APMoEApp.from_config(path)` performs these steps in order:

1. Load and validate the JSON config.
2. Resolve and instantiate modality processors.
3. Resolve and instantiate cleaner, anonymizer, and optional embedder strategies.
4. Resolve and instantiate experts.
5. Load local expert weights or configure remote endpoints.
6. Resolve and instantiate the aggregation strategy.
7. Assemble `InferencePipeline`.
8. Store config and return a ready app instance.

A failure at any step raises an `APMoEError` subclass and startup stops before the app can serve partial state.

## Prediction API

```python
prediction = app.predict(raw_inputs)
prediction = await app.predict_async(raw_inputs)
```

`raw_inputs` is a `dict[str, Any]` keyed by modality name. For the current built-ins, common keys are `image` and `keystroke`.

```python
prediction = app.predict({
    "image": image_bytes,
    "keystroke": [[65, 0, 120], [65, 83, 80]],
})

keystroke_only = app.predict({"keystroke": [[65, 0, 120]]})
print(keystroke_only.skipped_experts)  # experts that required missing image data
```

Extra unknown input keys are ignored. Missing modalities cause incompatible experts to be skipped when another expert can still run.

## Prediction Result

```python
prediction.predicted_age
prediction.confidence
prediction.per_expert_outputs
prediction.skipped_experts
prediction.metadata
```

See [types.md](types.md) for the full `Prediction` type reference.

## Validation

```python
report = app.validate()
```

`validate()` runs a lightweight health check without executing prediction. It is useful for CI, pre-start checks, and readiness probes.

Example shape:

```python
{
    "status": "ok",
    "modalities": ["image", "keystroke"],
    "experts": {
        "face_age_expert": {"status": "ok", "weights_exist": True},
        "keystroke_age_expert": {"status": "ok", "weights_exist": True},
    },
    "aggregator": "WeightedAverageAggregator",
}
```

## Metadata

```python
info = app.get_info()
```

Example shape:

```python
{
    "framework_version": "0.1.8",
    "modalities": ["image", "keystroke"],
    "experts": ["face_age_expert", "keystroke_age_expert"],
    "aggregator": "WeightedAverageAggregator",
    "serving": {"host": "0.0.0.0", "port": 8000, "workers": 1},
}
```

## Serving

```python
app.serve()
```

The HTTP app exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/predict` | POST | Submit raw inputs and receive a `Prediction`. |
| `/v1/health` | GET | Expert readiness/liveness status. |
| `/v1/info` | GET | Runtime metadata from `app.get_info()`. |

Legacy `/predict`, `/health`, and `/info` aliases remain mounted with deprecation headers for compatibility.

## Error Handling

| Situation | Exception |
|---|---|
| Invalid JSON or schema violation | `ConfigurationError` |
| Unknown processor, strategy, or expert path | `RegistryError` |
| Local weight loading failure | `ExpertError` |
| Weight file missing during validation | `ConfigurationError` |
| Expert class does not subclass `ExpertPlugin` | `RegistryError` |

All framework exceptions inherit from `APMoEError`:

```python
from apmoe import APMoEApp, APMoEError

try:
    app = APMoEApp.from_config("configs/prod.json")
except APMoEError as exc:
    logger.critical("APMoE failed to start: %s", exc)
    raise SystemExit(1)
```

## Public Imports

The main runtime symbols are re-exported from the top-level package:

```python
from apmoe import APMoEApp, InferencePipeline, ModalityChain
```

## See Also

- [pipeline.md](pipeline.md)
- [types.md](types.md)
- [exceptions.md](exceptions.md)
- [configuration.md](../configuration.md)
- [testing.md](../testing.md)
