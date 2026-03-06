# Exception Hierarchy (`apmoe.core.exceptions`)

All APMoE errors inherit from a single base class so callers can choose how
broadly or narrowly to catch them.

```
Import path:  apmoe.core.exceptions
Re-exported:  apmoe  (top-level package)
Source:       src/apmoe/core/exceptions.py
```

---

## Class hierarchy

```
Exception
└── APMoEError
    ├── ConfigurationError
    ├── RegistryError
    ├── PipelineError
    ├── ModalityError
    ├── ExpertError
    └── ServingError
```

---

## `APMoEError` (base)

```python
class APMoEError(Exception):
    message: str
    context: dict[str, object]
```

Base class for every framework exception. All subclasses accept:

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | Human-readable description of the error. |
| `context` | `dict[str, object] \| None` | Optional key/value pairs with diagnostic details (file paths, registry names, raw values, etc.). Printed in `__str__` as `[key='value', ...]`. |

```python
from apmoe import APMoEError

try:
    cfg = load_config("config.json")
except APMoEError as exc:
    print(exc.message)   # clean message
    print(exc.context)   # {'path': 'config.json', ...}
```

---

## `ConfigurationError`

Raised by `load_config()` whenever the framework cannot produce a valid
`FrameworkConfig`.

**When it is raised:**

| Situation | Example |
|---|---|
| Config file not found | Path typo in `--config` flag |
| File cannot be read | Permission denied |
| Malformed JSON | Missing closing brace, trailing comma |
| Schema validation failure | Missing required field, wrong type |
| Duplicate modality or expert name | Two modalities with `"name": "visual"` |
| Expert references undeclared modality | Expert declares `"eeg"` but no EEG modality configured |
| Environment variable type coercion failure | `APMOE_SERVING_PORT=abc` |

**Handling:**

```python
from apmoe import load_config
from apmoe import ConfigurationError

try:
    cfg = load_config("config.json")
except ConfigurationError as exc:
    # exc.message includes the file path and Pydantic's validation detail
    logging.error("Bad config: %s", exc)
    sys.exit(1)
```

---

## `RegistryError`

Raised by `Registry` methods when component resolution fails.

**When it is raised:**

| Situation | Method |
|---|---|
| Registering a name that already exists (and `overwrite=False`) | `register()`, `register_class()` |
| Looking up a name that is not registered | `get()` |
| `resolve()` called with an unregistered name that has no `.` | `resolve()` |
| Dotted-path import fails (module not found, attribute missing) | `resolve()` |

**Handling:**

```python
from apmoe import RegistryError
from apmoe.core.registry import Registry

try:
    cls = my_registry.get("unknown_expert")
except RegistryError as exc:
    # exc.context contains 'registry' and 'key'
    print(f"Available: {my_registry.list_registered()}")
```

---

## `PipelineError`

Raised by the inference pipeline orchestrator (`InferencePipeline`, Phase 3)
when execution cannot complete.

**When it is raised:**

- No experts remain after filtering by available modalities (all required
  modalities absent from the input).
- The aggregator receives an empty `ExpertOutput` list.
- An unrecoverable error occurs in a processing step that is not wrapped by
  `ModalityError` or `ExpertError`.

---

## `ModalityError`

Raised within a modality's processing branch.

**When it is raised:**

- `ModalityProcessor.validate()` returns `False` for the raw input.
- A `CleanerStrategy`, `AnonymizerStrategy`, or `EmbedderStrategy` raises an
  unhandled exception — the pipeline wraps it in `ModalityError` with the
  modality name in `context`.

**Context keys:** `modality`, `stage` (`"cleaner"` / `"anonymizer"` / `"embedder"`).

```python
from apmoe import ModalityError

try:
    result = pipeline.run(inputs)
except ModalityError as exc:
    # gracefully degrade: skip this modality, re-run without it
    print(f"Modality '{exc.context['modality']}' failed at {exc.context['stage']}")
```

---

## `ExpertError`

Raised within an expert plugin.

**When it is raised:**

- `load_weights()` cannot open or parse the weight file.
- `predict()` raises an unhandled exception — wrapped with the expert name in
  `context`.
- Required modality data is absent from the dispatch dict passed to `predict()`.

**Context keys:** `expert_name`, optionally `weights_path`.

---

## `ServingError`

Raised in the HTTP serving layer (`serving/`, Phase 4).

**When it is raised:**

- FastAPI cannot start (port already in use, SSL cert missing).
- A middleware is misconfigured.
- Request parsing fails in the `/predict` handler and no other exception type
  fits.

---

## Broad vs. targeted catching

```python
from apmoe import APMoEError, ConfigurationError, ExpertError

# Broad — catch any framework error
try:
    prediction = app.predict(inputs)
except APMoEError as exc:
    log.error(exc)

# Targeted — only handle expert failures, let others propagate
try:
    prediction = app.predict(inputs)
except ExpertError as exc:
    log.warning("Expert %s failed, using fallback", exc.context.get("expert_name"))
    prediction = fallback_prediction()
```
