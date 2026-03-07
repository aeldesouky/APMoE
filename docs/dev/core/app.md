# APMoEApp (`apmoe.core.app`)

`APMoEApp` is the **IoC container and lifecycle manager** for the entire framework.
It reads your configuration file, resolves all component classes, loads pretrained
weights, and exposes a clean API for inference and serving — without any glue code
on your part.

```python
from apmoe import APMoEApp

app = APMoEApp.from_config("configs/my_project.json")
prediction = app.predict({"visual": image_bytes, "audio": audio_bytes})
```

---

## Responsibilities

| Responsibility | Description |
|---|---|
| **Config loading** | Calls `load_config()`, validates JSON structure and cross-field constraints. |
| **Component resolution** | Resolves every `processor`, `cleaner`, `anonymizer`, `embedder`, `expert.class`, and `aggregation.strategy` string to a Python class via registry lookup or dotted-path import. |
| **Weight loading** | Calls `expert.load_weights(path)` once per expert at bootstrap time. |
| **Pipeline wiring** | Assembles `ModalityChain` objects and passes them to `InferencePipeline`. |
| **Inference** | Delegates `predict()` / `predict_async()` directly to `InferencePipeline.run()` / `run_async()`. |
| **Validation** | Checks that all configured weight files still exist and that each expert's `health_check()` passes. |
| **Serving** | Starts a FastAPI/uvicorn HTTP server *(Phase 4)*. |
| **Metadata** | Returns a structured summary of the running app via `get_info()`. |

---

## Bootstrap lifecycle (`from_config`)

```python
app = APMoEApp.from_config("path/to/config.json")
# or
app = APMoEApp.from_config(Path("path/to/config.json"))
```

Bootstrap executes these steps **in order**. A failure at any step raises an
appropriate exception and halts startup — the app never reaches a partially-valid
state.

```
1. load_config(path)
   └─ Reads JSON, applies APMOE_* env var overrides, validates with Pydantic.
      Raises ConfigurationError on any schema or cross-field violation.

2. Resolve modality processors
   └─ For each modalities[] entry, call modality_registry.resolve(processor_str).
      Raises RegistryError if the class cannot be found.

3. Instantiate processors
   └─ processor_cls() — default constructor. No weights loading here.

4. Build ModalityChains
   └─ For each modality, resolve and instantiate:
        cleaner_registry.resolve(pipeline.cleaner)()
        anonymizer_registry.resolve(pipeline.anonymizer)()
        embedder_registry.resolve(pipeline.embedder)()  ← skipped if None
      Raises RegistryError on unknown class; raises TypeError on bad constructor.

5. Build ExpertRegistry
   └─ For each experts[] entry:
        a. expert_registry.resolve(expert.class)  → ExpertPlugin subclass
        b. Instantiate: expert_cls(expert_config)
        c. expert_instance.load_weights(expert.weights)
      Raises ExpertError if load_weights() raises.
      Raises RegistryError if the class cannot be found.

6. Resolve aggregator
   └─ aggregator_registry.resolve(aggregation.strategy)()
      Raises RegistryError if unknown.

7. Assemble InferencePipeline
   └─ InferencePipeline(chains=..., expert_registry=..., aggregator=...)

8. Store config and return APMoEApp instance
```

### Example — inspecting a freshly bootstrapped app

```python
app = APMoEApp.from_config("configs/prod.json")

info = app.get_info()
print(info["modalities"])        # ["visual", "audio", "eeg"]
print(info["experts"])           # ["face_expert", "audio_expert", "eeg_expert"]
print(info["aggregator"])        # "WeightedAverageAggregator"
print(info["framework_version"]) # "1.0.0"
```

---

## `predict` / `predict_async`

```python
# Synchronous
prediction = app.predict(raw_inputs)

# Asynchronous (use inside an async function / FastAPI route)
prediction = await app.predict_async(raw_inputs)
```

### `raw_inputs`

A `dict[str, Any]` mapping modality name → raw data. Raw data is whatever your
`ModalityProcessor.preprocess()` implementation accepts (typically `bytes`).

You may pass **more** modalities than the app has registered — extra keys are
silently ignored. You may pass **fewer** — missing modalities cause the affected
experts to be skipped (graceful degradation; see
[pipeline.md — Graceful degradation](pipeline.md#graceful-degradation)).

```python
# All three modalities
prediction = app.predict({
    "visual": image_bytes,
    "audio":  audio_bytes,
    "eeg":    eeg_bytes,
})

# Audio and EEG unavailable — visual-only experts still run
prediction = app.predict({"visual": image_bytes})
print(prediction.skipped_experts)   # experts that need audio or eeg
```

### `Prediction` result

```python
prediction.predicted_age       # float — final age estimate
prediction.confidence          # float — aggregator-provided confidence
prediction.per_expert_outputs  # list[ExpertOutput] — individual expert results
prediction.skipped_experts     # list[str] — experts skipped due to missing modalities
prediction.metadata            # dict — pipeline latency, failed/available modalities
```

See [core/types.md](types.md) for the full `Prediction` type reference.

---

## `validate`

```python
report = app.validate()
```

Runs a **lightweight health check** without running inference. Returns a
structured report dict. Raises `ConfigurationError` if the check fails hard
(e.g. a required weight file no longer exists).

```python
{
    "status": "ok",           # "ok" | "degraded" | "error"
    "modalities": ["visual", "audio"],
    "experts": {
        "face_expert":  {"status": "ok",  "weights_exist": True},
        "audio_expert": {"status": "ok",  "weights_exist": True},
    },
    "aggregator": "WeightedAverageAggregator",
}
```

Typical use cases:
- Kubernetes liveness / readiness probes.
- CI smoke tests that verify a config file + weights bundle is internally consistent.
- Admin tooling that surfaces misconfiguration before an inference request fails.

---

## `get_info`

```python
info = app.get_info()
```

Returns a read-only metadata dict describing the running application. No
I/O or computation is performed — all values are derived from the already-loaded
config and instantiated components.

```python
{
    "framework_version": "1.0.0",
    "modalities":  ["visual", "audio", "eeg"],
    "experts":     ["face_expert", "audio_expert", "eeg_expert"],
    "aggregator":  "WeightedAverageAggregator",
    "serving":     {"host": "0.0.0.0", "port": 8000, "workers": 4},
}
```

Use this in the HTTP `/info` endpoint or to log a startup summary.

---

## `serve`

```python
app.serve()
```

Starts the FastAPI/uvicorn HTTP server using the `serving` block from config.
Exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Submit raw inputs; returns a `Prediction` as JSON. |
| `/health` | GET | Calls `app.validate()` and returns status code 200/503. |
| `/info` | GET | Returns `app.get_info()` as JSON. |

> **Phase 4:** The HTTP serving layer is implemented in Phase 4. Calling `serve()`
> in Phase 3 raises `ServingError("HTTP serving not yet implemented — Phase 4")`.

---

## Error handling at bootstrap

| Situation | Exception | Message pattern |
|---|---|---|
| Invalid JSON / schema violation | `ConfigurationError` | `"Config validation failed: ..."` |
| Unknown processor / strategy string | `RegistryError` | `"Cannot resolve 'my.Cls': not registered and not importable"` |
| `load_weights()` raises | `ExpertError` | `"Expert 'face_expert' failed to load weights: ..."` |
| Weight file path does not exist | `ExpertError` at bootstrap, `ConfigurationError` at `validate()` | — |
| Expert class does not subclass `ExpertPlugin` | `RegistryError` | `"Resolved class 'Foo' is not a subclass of ExpertPlugin"` |

All exceptions inherit from `APMoEError` so you can catch them with a single
handler:

```python
from apmoe import APMoEApp, APMoEError

try:
    app = APMoEApp.from_config("configs/prod.json")
except APMoEError as exc:
    logger.critical("APMoE failed to start: %s", exc)
    raise SystemExit(1)
```

---

## Using `APMoEApp` from the public API

All Phase 3 symbols are re-exported from the top-level `apmoe` package:

```python
from apmoe import APMoEApp, InferencePipeline, ModalityChain
```

You should **not** import directly from `apmoe.core.app` unless you are
intentionally accessing internals.

---

## `__repr__`

```python
str(app)
# → "APMoEApp(modalities=['visual', 'audio'], experts=['face_expert', 'audio_expert'])"
```

---

## See also

- [pipeline.md](pipeline.md) — `InferencePipeline` and `ModalityChain` — what the app wires together
- [../configuration.md](../configuration.md) — full config file reference
- [../testing.md](../testing.md) — how `APMoEApp` is tested (app integration tests)
- [types.md](types.md) — `Prediction` and related types
- [exceptions.md](exceptions.md) — full exception hierarchy
