# Core Module (`apmoe.core`)

The `core` package contains the framework internals. You will interact with
these as a consumer (catching exceptions, reading `Prediction` objects,
using the registry) but you generally do not subclass or modify them — that
is what the [extension points](../extension-points/index.md) are for.

---

## Modules

| Module | Import path | Purpose |
|---|---|---|
| Types | `apmoe.core.types` | Pipeline data types shared by all components |
| Exceptions | `apmoe.core.exceptions` | Framework exception hierarchy |
| Registry | `apmoe.core.registry` | Generic component registry |
| Config | `apmoe.core.config` | JSON config loader and Pydantic schema |
| Pipeline | `apmoe.core.pipeline` | Inference pipeline orchestrator *(Phase 3)* |
| App | `apmoe.core.app` | `APMoEApp` IoC container and lifecycle *(Phase 3)* |

---

## Public re-exports

All core symbols are re-exported from the top-level `apmoe` package for
convenience:

```python
from apmoe import (
    # Types
    ModalityData, EmbeddingResult, ProcessedInput,
    ExpertOutput, Prediction,
    # Exceptions
    APMoEError, ConfigurationError, RegistryError,
    PipelineError, ModalityError, ExpertError, ServingError,
    # Registry
    Registry,
    # Config
    load_config, FrameworkConfig,
    # Pipeline (Phase 3)
    InferencePipeline, ModalityChain,
    # App (Phase 3)
    APMoEApp,
)
```

---

## See also

- [types.md](types.md) — Full type reference
- [exceptions.md](exceptions.md) — Exception hierarchy and handling guide
- [registry.md](registry.md) — Registry usage reference
- [pipeline.md](pipeline.md) — `InferencePipeline` and `ModalityChain` execution model
- [app.md](app.md) — `APMoEApp` bootstrap lifecycle and inference API
- [../configuration.md](../configuration.md) — Config file format reference
- [../testing.md](../testing.md) — Testing strategy for all core modules
