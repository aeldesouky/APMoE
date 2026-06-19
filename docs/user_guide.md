# APMoE User Guide

This guide walks through the normal application-owner workflow: install APMoE,
create a project, add or install extensions, configure local and remote
experts, enable fallback, plug in Redis-backed stores, and run the service.

APMoE is an inference framework. It wires your processors, cleaning steps,
experts, aggregators, security settings, and serving layer from one
`config.json`. The default PyPI wheel includes the demo model artifact files;
`apmoe download-models` copies those files into each project so local paths stay
explicit and easy to replace.

---

## 1. Install the Package

Install APMoE:

```bash
pip install apmoe
```

The default install includes the framework core, CLI, serving stack, local and
remote expert runtimes, security features, Redis client integration, image/ONNX
support, TensorFlow, Torch, and built-in demo model artifact files.

Install commands:

| Goal | Install command |
|---|---|
| Runtime use, including serving, remote experts, security, Redis, ML backends, and demo artifacts | `pip install apmoe` |
| Contributor environment | `pip install -e ".[dev]"` |

The old runtime extra names such as `apmoe[security]` and `apmoe[redis]` remain
as compatibility aliases, but users no longer need them. The `models` extra also
remains as a compatibility alias; demo artifacts are bundled in `apmoe`.

---

## 2. Start a Project

Create a scaffold:

```bash
apmoe init my_app
cd my_app
```

In an interactive terminal, `apmoe init` asks whether to acquire demo model
artifacts. For scripts, choose explicitly:

```bash
apmoe init my_app --download-models
apmoe init my_app --no-download-models
```

The generated project includes:

```text
my_app/
  config.json
  custom_processor.py
  custom_cleaner.py
  custom_anonymizer.py
  custom_embedder.py
  custom_expert.py
  custom_aggregator.py
  custom_security.py
  weights/
  README.md
```

If you skip models during init, acquire them later:

```bash
apmoe download-models --dest weights --model all
```

`apmoe download-models` uses local configured sources first. If no local source
is available, it copies the demo artifacts bundled in the installed `apmoe`
package. If those resources are unavailable, it falls back to release-hosted
artifact URLs and still verifies SHA-256 checksums after download.

To copy model artifacts from a local source directory instead of the installed
package, set:

```bash
export APMOE_MODEL_SOURCE_DIR=/path/to/apmoe-model-files
apmoe download-models --dest weights
```

On PowerShell:

```powershell
$env:APMOE_MODEL_SOURCE_DIR = "D:\models\apmoe"
apmoe download-models --dest weights
```

To download artifacts from a different Git ref, set `APMOE_MODEL_SOURCE_REF`
before running the command.

---

## 3. Validate and Run

Validate config, imports, weights, and expert health:

```bash
apmoe validate --config config.json
```

Run one prediction from files:

```bash
apmoe predict --config config.json --input data/
```

Serve the API:

```bash
apmoe serve --config config.json --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

The CLI prints whether each expert is `[local]`, `[remote]`, or
`[local fallback]`, plus fallback warnings and health status.

---

## 4. Add Local Application Code

Use dotted paths when the code lives in your project:

```json
{
  "processor": "custom_processor.MyImageProcessor"
}
```

```json
{
  "class": "custom_expert.MyAgeExpert"
}
```

The framework imports those modules during `APMoEApp.from_config()`. Your
classes should subclass the relevant APMoE base classes:

| Extension | Base class |
|---|---|
| Modality processor | `apmoe.modality.base.ModalityProcessor` |
| Cleaner | `apmoe.processing.base.CleanerStrategy` |
| Anonymizer | `apmoe.processing.base.AnonymizerStrategy` |
| Embedder | `apmoe.processing.base.EmbedderStrategy` |
| Expert | `apmoe.experts.base.ExpertPlugin` |
| Aggregator | `apmoe.aggregation.base.AggregatorStrategy` |

Example expert:

```python
from apmoe.core.types import ExpertOutput, ProcessedInput
from apmoe.experts.base import ExpertPlugin


class MyAgeExpert(ExpertPlugin):
    @property
    def name(self) -> str:
        return "my_age_expert"

    def declared_modalities(self) -> list[str]:
        return ["image"]

    def load_weights(self, path: str) -> None:
        self.weights_path = path

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["image"],
            predicted_age=30.0,
            confidence=0.75,
        )
```

---

## 5. Install Extension Packages

Installed extension packages can expose components through Python entry points.
APMoE discovers them automatically during `APMoEApp.from_config()`.

In the extension package:

```toml
[project.entry-points."apmoe.experts"]
my_expert = "my_package.experts:MyExpert"

[project.entry-points."apmoe.aggregators"]
my_aggregator = "my_package.aggregation:MyAggregator"
```

Then install the extension next to APMoE:

```bash
pip install apmoe-my-extension
```

Use the short entry-point name in config:

```json
{
  "class": "my_expert"
}
```

Supported entry-point groups:

| Group | Config field |
|---|---|
| `apmoe.modality_processors` | `modalities[].processor` |
| `apmoe.cleaners` | `modalities[].pipeline.cleaner` |
| `apmoe.anonymizers` | `modalities[].pipeline.anonymizer` |
| `apmoe.embedders` | `modalities[].pipeline.embedder` |
| `apmoe.experts` | `experts[].class` |
| `apmoe.aggregators` | `aggregation.strategy` |

---

## 6. Configure Local Experts

Local experts load a model artifact from `weights`:

```json
{
  "name": "face_age_expert",
  "class": "apmoe.experts.builtin.FaceAgeExpert",
  "weights": "./weights/face_age_expert.keras",
  "modalities": ["image"]
}
```

Pin a local artifact hash when model files are managed outside the package:

```json
{
  "name": "face_age_expert",
  "class": "apmoe.experts.builtin.FaceAgeExpert",
  "weights": "./weights/face_age_expert.keras",
  "modalities": ["image"],
  "integrity": {
    "sha256": "64-character-hex-digest"
  }
}
```

The default package includes the backends used by built-in experts. ONNX,
TensorFlow/Keras, PyTorch, Pillow, remote HTTP, security, and Redis client
dependencies and demo artifacts are installed by `pip install apmoe`. Run
`apmoe download-models` to copy packaged demo model artifacts into a project.

---

## 7. Configure Remote Experts

Remote experts call an HTTP model endpoint instead of loading local weights.
Remote support is included in the default install.

Config:

```json
{
  "name": "remote_face_expert",
  "class": "apmoe.experts.remote.RemoteExpert",
  "endpoint": "$REMOTE_FACE_ENDPOINT",
  "endpoint_headers": {
    "Authorization": "Bearer $REMOTE_FACE_TOKEN"
  },
  "endpoint_timeout": 20.0,
  "modalities": ["image"],
  "request_template": {
    "image": "{{modalities.image}}"
  },
  "response_mapping": {
    "predicted_age": "result.age",
    "confidence": "result.confidence"
  }
}
```

Set secrets outside config:

```bash
export REMOTE_FACE_ENDPOINT="https://models.example.com/predict"
export REMOTE_FACE_TOKEN="replace-me"
```

In production, configure a remote endpoint allowlist:

```json
{
  "environment": "production",
  "security": {
    "remote_endpoint_allowlist": ["models.example.com"],
    "remote_enforce_https": true,
    "remote_allow_private_networks": false
  }
}
```

---

## 8. Add Local Fallback for a Remote Expert

Use a paired local fallback when degraded local inference is better than
failing during a remote outage.

```json
{
  "apmoe": {
    "remote_fallback_policy": "transient_only",
    "experts": [
      {
        "name": "remote_face_expert",
        "class": "apmoe.experts.remote.RemoteExpert",
        "endpoint": "$REMOTE_FACE_ENDPOINT",
        "modalities": ["image"],
        "fallback_expert": "local_face_standby"
      },
      {
        "name": "local_face_standby",
        "class": "apmoe.experts.builtin.FaceAgeExpert",
        "weights": "./weights/face_age_expert.keras",
        "modalities": ["image"],
        "fallback_only": true
      }
    ]
  }
}
```

Fallback policies:

| Policy | Behavior |
|---|---|
| `transient_only` | Fall back on timeout, network error, transient HTTP status, or open circuit |
| `any_remote_error` | Fall back on any remote expert error |
| `disabled` | Do not use paired local fallback |

Retries and circuit breakers are configured at the top level:

```json
{
  "remote_retry": {
    "max_attempts": 3,
    "initial_delay_s": 0.25,
    "max_delay_s": 2.0,
    "backoff_multiplier": 2.0,
    "jitter": true
  },
  "remote_circuit_breaker": {
    "enabled": true,
    "failure_threshold": 5,
    "recovery_timeout_s": 30.0
  }
}
```

---

## 9. Plug in Redis

Redis client support is included in the default install.

Use Redis for shared rate limiting:

```json
{
  "serving": {
    "rate_limit": 120,
    "rate_limit_store": "redis",
    "rate_limit_redis_url": "$APMOE_REDIS_URL",
    "rate_limit_key_prefix": "apmoe:rate:"
  }
}
```

Use Redis for JWT invalidation:

```json
{
  "serving": {
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "$APMOE_REDIS_URL",
    "token_invalidation_key_prefix": "apmoe:jwt:invalid:"
  }
}
```

Set the URL:

```bash
export APMOE_REDIS_URL="redis://localhost:6379/0"
```

Equivalent environment overrides are also available:

```bash
export APMOE_SERVING_RATE_LIMIT_STORE=redis
export APMOE_SERVING_RATE_LIMIT_REDIS_URL="redis://localhost:6379/0"
export APMOE_SERVING_TOKEN_INVALIDATION_STORE=redis
export APMOE_SERVING_TOKEN_INVALIDATION_REDIS_URL="redis://localhost:6379/0"
```

Redis operation failures fall back to process-local memory and emit audit
events. This keeps the API available, but fallback state is not shared across
workers or nodes.

---

## 10. Secure and Serve

For a local demo, generated config can disable authentication. For production,
use the included security and Redis integrations and pass a stateless provider
when creating the API.

Recommended production settings:

```json
{
  "environment": "production",
  "serving": {
    "authentication_enabled": true,
    "authorization_enabled": true,
    "rate_limit": 120,
    "rate_limit_store": "redis",
    "rate_limit_redis_url": "$APMOE_REDIS_URL",
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "$APMOE_REDIS_URL",
    "cors_origins": ["https://app.example.com"]
  },
  "security": {
    "remote_endpoint_allowlist": ["models.example.com"],
    "remote_enforce_https": true,
    "remote_allow_private_networks": false,
    "audit_enabled": true,
    "audit_success_events": true
  }
}
```

Start serving:

```bash
apmoe validate --config config.json
apmoe serve --config config.json --workers 2
```

For the complete reference, see:

- [CLI reference](dev/cli.md)
- [Configuration reference](dev/configuration.md)
- [Extension points](dev/extension-points/index.md)
- [Remote expert endpoints](remote_expert_endpoints.md)
- [Security reference](dev/security.md)
- [Licensing and model artifact guidance](licensing.md)
