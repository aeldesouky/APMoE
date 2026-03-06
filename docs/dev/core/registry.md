# Registry (`apmoe.core.registry`)

`Registry[T]` is a generic, type-safe name-to-class store used by every
extension point in the framework. Developers register custom components once;
the framework resolves and instantiates them at bootstrap.

```
Import path:  apmoe.core.registry
Re-exported:  apmoe  (top-level package)
Source:       src/apmoe/core/registry.py
```

---

## Class signature

```python
class Registry(Generic[T]):
    name: str                          # label used in error messages

    def register(self, key: str) -> Callable[[type[T]], type[T]]: ...
    def register_class(self, key: str, cls: type[T], *, overwrite: bool = False) -> None: ...
    def get(self, key: str) -> type[T]: ...
    def resolve(self, name_or_path: str) -> type[T]: ...
    def list_registered(self) -> list[str]: ...
    def __contains__(self, key: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[str]: ...
```

---

## Creating a registry

Each extension point owns its own `Registry` instance. The framework creates
one per subsystem at startup:

```python
from apmoe.core.registry import Registry
from apmoe.experts.base import ExpertPlugin

expert_registry: Registry[ExpertPlugin] = Registry("experts")
```

The `name` argument is a human-readable label used in error messages only —
it has no runtime effect on behaviour.

---

## Registering components

### Via decorator (recommended)

```python
from apmoe.experts.base import ExpertPlugin
from apmoe.core.types import ProcessedInput, ExpertOutput

@expert_registry.register("cnn_age_expert")
class CNNAgeExpert(ExpertPlugin):
    ...
```

The decorator returns the class unchanged, so it can still be imported and
used normally. Registration is a side effect that fires when the module is
imported.

### Via `register_class()`

Use this when you don't control the class definition (e.g. registering a
third-party class):

```python
from thirdparty.models import TheirExpert

expert_registry.register_class("their_expert", TheirExpert)
```

### Overwriting an existing registration

By default, re-registering the same key raises `RegistryError`. Pass
`overwrite=True` to replace silently:

```python
expert_registry.register_class("cnn_age_expert", ImprovedExpert, overwrite=True)
```

---

## Looking up components

### `get(key)` — by registered name

```python
cls = expert_registry.get("cnn_age_expert")
instance = cls()
```

Raises `RegistryError` if the key is not found. The error message lists all
available keys.

### `resolve(name_or_path)` — by name or dotted import path

```python
# Short registered name
cls = expert_registry.resolve("cnn_age_expert")

# Fully-qualified dotted path (imported on demand, no prior registration needed)
cls = expert_registry.resolve("myproject.experts.CustomExpert")
```

Resolution order:

1. Check the in-memory store (registered names take priority).
2. If the string contains `.`, import the module and retrieve the attribute.
3. If neither matches, raise `RegistryError`.

This is how config file class paths like
`"apmoe.experts.builtin.CNNAgeExpert"` are resolved at bootstrap — the
framework calls `registry.resolve(config_value)` for every component in the
config.

---

## Introspection

```python
# All registered names, sorted alphabetically
expert_registry.list_registered()
# → ['audio_expert', 'cnn_age_expert', 'eeg_expert']

# Membership test
"cnn_age_expert" in expert_registry   # → True

# Count
len(expert_registry)                  # → 3

# Iterate (insertion order)
for name in expert_registry:
    print(name)
```

---

## Error handling

```python
from apmoe import RegistryError

try:
    cls = expert_registry.get("unknown")
except RegistryError as exc:
    print(exc.message)          # "No component named 'unknown' ..."
    print(exc.context["key"])   # "unknown"
    print(exc.context["registry"])  # "experts"
```

---

## Framework registries (reference)

The framework maintains one `Registry` instance per extension point. These
are created in the respective base modules and should not be imported directly
by application code — use the decorator on the base class instead (see the
extension-point docs):

| Registry | Extension point | Base class |
|---|---|---|
| `modality_registry` | `apmoe.modality` | `ModalityProcessor` |
| `cleaner_registry` | `apmoe.processing` | `CleanerStrategy` |
| `anonymizer_registry` | `apmoe.processing` | `AnonymizerStrategy` |
| `embedder_registry` | `apmoe.processing` | `EmbedderStrategy` |
| `expert_registry` | `apmoe.experts` | `ExpertPlugin` |
| `aggregator_registry` | `apmoe.aggregation` | `AggregatorStrategy` |

> These registries are populated by Phase 2 (extension-point abstractions) and
> Phase 6 (built-in implementations).
