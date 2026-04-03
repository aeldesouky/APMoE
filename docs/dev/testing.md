# Testing Strategy

APMoE uses a **three-layer testing approach** that mirrors the architecture of the
framework itself: isolated unit tests per module, cross-module boundary tests, and
end-to-end bootstrap tests through the application container.

---

## Test layout

```
tests/
├── conftest.py                          # Shared pytest fixtures (config builders, env cleanup)
├── unit/
│   ├── test_types.py                    # ModalityData, EmbeddingResult, ExpertOutput, Prediction
│   ├── test_exceptions.py               # Exception hierarchy and context formatting
│   ├── test_registry.py                 # Registry[T]: register, get, resolve, dotted-path import
│   ├── test_config.py                   # JSON loading, env-var overrides, Pydantic validation
│   ├── test_modality.py                 # ModalityProcessor ABC + ModalityProcessorFactory
│   ├── test_processing.py               # CleanerStrategy, AnonymizerStrategy, EmbedderStrategy ABCs
│   ├── test_experts.py                  # ExpertPlugin ABC + ExpertRegistry lifecycle
│   ├── test_aggregation.py              # AggregatorStrategy ABC + aggregator_registry
│   ├── test_pipeline.py                 # InferencePipeline: both phases, hooks, async, degradation
│   ├── test_serving.py                  # Serving layer: routes, middleware, auth, CORS, rate limiting
│   └── test_cli.py                      # CLI commands: init, serve, predict, validate, help/version
└── integration/
    ├── test_app.py                      # APMoEApp.from_config() end-to-end bootstrap + inference
    └── test_module_boundaries.py        # Cross-module interface contracts (7 boundaries)
```

---

## Layer 1 — Unit tests (`tests/unit/`)

**Principle:** Each test file covers exactly one module. All dependencies are replaced
with minimal test doubles defined in the same file. No global registry entries are
written; no files are read.

### What is tested

| Test file | Module under test | What the doubles replace |
|---|---|---|
| `test_types.py` | `core/types.py` | — (no dependencies) |
| `test_exceptions.py` | `core/exceptions.py` | — |
| `test_registry.py` | `core/registry.py` | — |
| `test_config.py` | `core/config.py` | Uses `tmp_path` for JSON files |
| `test_modality.py` | `modality/base.py` + `modality/factory.py` | `modality_registry` entries (local to file) |
| `test_processing.py` | `processing/base.py` | — |
| `test_experts.py` | `experts/base.py` + `experts/registry.py` | `expert_registry` entries (local to file) |
| `test_aggregation.py` | `aggregation/base.py` | — |
| `test_pipeline.py` | `core/pipeline.py` | All six surrounding modules (processor, cleaner, anonymizer, embedder, expert, aggregator) |

### Test doubles pattern

Each unit test file defines private helper classes (prefixed with `_`) that are the
minimal concrete implementations needed to exercise the module under test:

```python
class _PassProcessor(ModalityProcessor):
    """Wraps any input as ModalityData without modification."""
    @property
    def modality_name(self) -> str: return "visual"
    def validate(self, data: object) -> bool: return data is not None
    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality="visual", data=data)
```

Doubles are **never shared across test files**. Each file is self-contained.

### What unit tests do NOT cover

- Whether a registry-resolved class produces an instance that satisfies the next
  stage's interface (→ covered by boundary tests).
- Whether `ModalityData` produced by a real `CleanerStrategy` flows correctly into
  a real `AnonymizerStrategy` (→ covered by boundary tests).
- Whether `APMoEApp.from_config()` correctly wires all components together
  (→ covered by the app integration tests).

---

## Layer 2 — Module boundary tests (`tests/integration/test_module_boundaries.py`)

**Principle:** Test the exact interface contract between two adjacent real modules,
without going through the `APMoEApp` wrapper. Stub components are registered in the
real global registries (with `overwrite=True`) by an `autouse` fixture that runs
before every test.

### The seven boundaries

```
modality_registry ──► ModalityProcessorFactory        Boundary 1
        │
strategy registries ──► ModalityChain construction     Boundary 2
        │
CleanerStrategy ──► AnonymizerStrategy ──► EmbedderStrategy
                                           (data contract)  Boundary 3
        │
expert_registry ──► ExpertRegistry.from_configs()     Boundary 4
        │
ExpertRegistry ──► InferencePipeline (dispatch)        Boundary 5
        │
aggregator_registry ──► AggregatorStrategy ──► Prediction  Boundary 6
        │
Config ──► manual pipeline wiring (without APMoEApp)   Boundary 7
```

### Boundary 3 — the data contract (most important)

This boundary verifies that the output of one processing stage is a valid input
for the next — a contract that the ABCs enforce by type but do not enforce on
field content:

```python
def test_cleaner_output_feeds_anonymizer(self) -> None:
    raw = ModalityData(modality="visual", data=np.array([0.5, 1.5, -0.5]))
    cleaned = _NormCleaner().clean(raw)          # produces ModalityData
    anonymized = _TagAnonymizer().anonymize(cleaned)  # consumes ModalityData
    assert anonymized.metadata["normalized"] is True  # metadata survived
    assert anonymized.metadata["anonymized"] is True
```

Specifically it checks that:
- Metadata added in the cleaner step (`"normalized": True`) is still present in
  the anonymizer output.
- Clamped data values survive through all three stages unchanged.
- The `modality` field is preserved through every stage.
- An `EmbeddingResult` produced by the embedder carries forward metadata set in
  the anonymizer.

### Boundary 7 — wiring without the app

Replicates the exact steps that `APMoEApp.from_config()` performs internally, but
assembled manually so that a failure pinpoints the wiring logic rather than the app
wrapper:

```python
def _wire_pipeline_from_config(self, cfg_path: Path) -> InferencePipeline:
    cfg = load_config(cfg_path)
    processors = ModalityProcessorFactory.from_configs(cfg.apmoe.modalities)
    chains = {}
    for mod_cfg in cfg.apmoe.modalities:
        name = mod_cfg.name
        pipeline_cfg = mod_cfg.pipeline
        cleaner    = cleaner_registry.resolve(pipeline_cfg.cleaner)()
        anonymizer = anonymizer_registry.resolve(pipeline_cfg.anonymizer)()
        embedder   = embedder_registry.resolve(pipeline_cfg.embedder)() \
                     if pipeline_cfg.embedder else None
        chains[name] = ModalityChain(processor=processors[name],
                                     cleaner=cleaner, anonymizer=anonymizer,
                                     embedder=embedder)
    expert_reg = ExpertRegistry.from_configs(cfg.apmoe.experts, cfg.apmoe.modalities)
    agg = aggregator_registry.resolve(cfg.apmoe.aggregation.strategy)()
    return InferencePipeline(chains=chains, expert_registry=expert_reg, aggregator=agg)
```

---

## Layer 3 — App integration tests (`tests/integration/test_app.py`)

**Principle:** Test the full `APMoEApp` API (`from_config`, `predict`,
`predict_async`, `validate`, `get_info`) from the perspective of a framework user,
with stub components registered in the real global registries.

### What is tested

| Test class | Coverage |
|---|---|
| `TestFromConfig` | Bootstrap lifecycle: valid config, string paths, multi-modality, embedder, all error paths (unknown cleaner, anonymizer, embedder, aggregator, expert class) |
| `TestAppPredict` | Single modality, multi-modality, missing modality → graceful skip, no valid modality → `PipelineError`, metadata, per-expert breakdown |
| `TestAppPredictAsync` | Same cases as `TestAppPredict` for the async path |
| `TestAppValidate` | Healthy app passes, deleted weight file → `ConfigurationError`, expert health in report |
| `TestAppGetInfo` | All info fields present (version, experts, modalities, aggregator, serving) |
| `TestAppRepr` | `__repr__` contains framework name, modalities, experts |

### Stub component strategy

Stubs are registered once per test via an `autouse` fixture using unique dotted-path
keys (prefixed with `test.integration.`) to avoid polluting the real built-in
namespace:

```python
@pytest.fixture(autouse=True)
def _register_integration_components() -> None:
    modality_registry.register_class(
        "test.integration.VisualProcessor", _IntegrationVisualProcessor, overwrite=True
    )
    cleaner_registry.register_class(
        "test.integration.Cleaner", _IntegrationCleaner, overwrite=True
    )
    # ...
```

The `overwrite=True` flag ensures re-registration never raises across repeated test
runs (e.g. when using `pytest-xdist` parallel mode).

---

## Shared fixtures (`tests/conftest.py`)

| Fixture | Scope | What it provides |
|---|---|---|
| `minimal_config_dict` | function | A raw `dict` for the smallest valid APMoE config (one modality, one expert) |
| `minimal_config_file` | function | Writes `minimal_config_dict` as JSON to `tmp_path/config.json` and returns the `Path` |
| `full_config_file` | function | Multi-modality (visual + audio), three experts, weighted aggregation |
| `clean_env` (autouse) | function | Removes all `APMOE_*` environment variables before each test |

---

## Running the tests

```bash
# Full suite
uv run pytest

# With coverage report
uv run pytest --cov --cov-report=term-missing

# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# Specific boundary
uv run pytest tests/integration/test_module_boundaries.py -v

# Single test class
uv run pytest tests/unit/test_pipeline.py::TestRunMultiModality -v

# Stop on first failure
uv run pytest -x
```

---

## Guidelines for contributors

### Adding a new extension point (e.g. a new phase)

1. **Unit tests** — create `tests/unit/test_<module>.py`. Cover:
   - ABC cannot be instantiated directly.
   - All abstract methods must be implemented (missing one raises `TypeError`).
   - Each method's happy path with a concrete subclass.
   - All error paths documented in the method's docstring.

2. **Boundary tests** — add a `TestXxxToYyy` class to
   `tests/integration/test_module_boundaries.py`. Cover:
   - The registry can resolve the new class by registered name and by dotted path.
   - The output type of the new module is a valid input for the consuming module.
   - Relevant metadata fields survive the boundary.

3. **App integration tests** — add cases to `tests/integration/test_app.py`
   exercising the new extension point through the full `APMoEApp.from_config()` path.

### Test double naming conventions

| Prefix | Meaning | Example |
|---|---|---|
| `_Pass*` | No-op; passes input through unchanged | `_PassCleaner` |
| `_Tag*` | Adds a metadata flag without modifying data | `_TagAnonymizer` |
| `_Reject*` | Always returns `False` / invalid | `_RejectingProcessor` |
| `_Exploding*` | Raises a `RuntimeError` to test error wrapping | `_ExplodingEmbedder` |
| `_Constant*` | Returns a fixed, predictable value | `_ConstantExpert` |
| `_Spy*` | Captures received inputs for post-run assertion | `_SpyExpert` |
| `_Integration*` | Slightly richer doubles used in integration tests | `_IntegrationVisualProcessor` |

### Testing async paths

The pipeline's `run_async()` method must be tested with `asyncio.run()` (not
`async def test_*`), because the `asyncio_mode = "auto"` setting in
`pyproject.toml` is reserved for future async API tests (Phase 4). Keep pipeline
async tests synchronous at the test level:

```python
def test_async_run_returns_prediction(self) -> None:
    prediction = asyncio.run(pipeline.run_async({"visual": b"img"}))
    assert isinstance(prediction, Prediction)
```

### What NOT to test in unit tests

- Configuration file parsing (belongs in `test_config.py`).
- Registry resolution by dotted path (belongs in boundary tests).
- `APMoEApp` behaviour (belongs in `test_app.py`).
- Real neural network forward passes (Phase 6 built-ins are integration-tested
  with mock weights only; actual model correctness is out of scope).

---

## Coverage target

The project enforces a minimum of **80% line coverage** via `pytest-cov`:

```toml
[tool.coverage.report]
fail_under = 80
```

Run with coverage to see a per-file breakdown:

```bash
uv run pytest --cov --cov-report=term-missing
```

The combination of unit tests (granular), boundary tests (interface contracts),
and app integration tests (bootstrap) is designed to keep coverage well above
this threshold without relying on brittle end-to-end tests that require real
pretrained weights.
