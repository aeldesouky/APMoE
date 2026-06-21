# Core Module (`apmoe.core`)

The core package contains the framework primitives used by the CLI, serving layer, extension points, and application container.

## Modules

| Module | Purpose |
|---|---|
| `apmoe.core.types` | Shared dataclasses and type aliases used across the pipeline. |
| `apmoe.core.exceptions` | Framework exception hierarchy. |
| `apmoe.core.registry` | Generic registry and dotted-path resolution utilities. |
| `apmoe.core.pipeline` | `InferencePipeline` and `ModalityChain`. |
| `apmoe.core.app` | `APMoEApp` bootstrap lifecycle, prediction API, validation, and serving. |
| `apmoe.core.config` | JSON configuration models and environment overrides. |
| `apmoe.core.security` | Security helpers, audit events, redaction, endpoint policy, and integrity checks. |

## Public Re-Exports

Most application code should import public runtime types from `apmoe`:

```python
from apmoe import (
    APMoEApp,
    APMoEError,
    ConfigurationError,
    EmbeddingResult,
    ExpertOutput,
    InferencePipeline,
    ModalityChain,
    ModalityData,
    Prediction,
    ProcessedInput,
)
```

## See Also

- [types.md](types.md)
- [exceptions.md](exceptions.md)
- [registry.md](registry.md)
- [pipeline.md](pipeline.md)
- [app.md](app.md)
- [configuration.md](../configuration.md)
- [testing.md](../testing.md)
