# APMoE Configuration Reference

The framework is configured via a single **JSON file** passed to `load_config()` or the
`--config` CLI flag. Environment variables prefixed with `APMOE_` can override
individual fields at runtime without editing the file.

```python
from apmoe.core.config import load_config

cfg = load_config("configs/my_project.json")
```

---

## Document structure

The JSON file must have a single top-level key `"apmoe"` containing the core
sections `modalities`, `experts`, and `aggregation`, plus optional operational
sections such as `serving`, `environment`, `security`, and resilience settings:

```json
{
  "apmoe": {
    "modalities": [ ... ],
    "experts":    [ ... ],
    "aggregation": { ... },
    "serving":    { ... }
  }
}
```

`modalities`, `experts`, and `aggregation` are **required**.  
`serving`, `environment`, `security`, `expert_failure_policy`, `remote_retry`,
and `remote_circuit_breaker` are **optional** â€” all their fields have defaults.

---

## Common Configuration Recipes

Use these recipes as starting points, then consult the field reference below.

### Local-Only Built-In Experts

Install:

```bash
pip install apmoe
apmoe init my_app --download-models
cd my_app
apmoe validate --config config.json
```

Config shape:

```json
{
  "name": "face_age_expert",
  "class": "apmoe.experts.builtin.FaceAgeExpert",
  "weights": "./weights/face_age_expert.keras",
  "modalities": ["image"]
}
```

Local experts use `weights`. The default install includes ONNX, TensorFlow,
Torch, Pillow, remote HTTP, security, and Redis client dependencies. Demo model
artifact files are bundled in `apmoe`; use `apmoe download-models` to copy them
into a project, or set `APMOE_MODEL_SOURCE_DIR` to provide your own files.

### Application-Local Extensions

For code inside a scaffolded project, use dotted paths:

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

The class must be importable from the working directory or installed package
environment.

### Installed Extension Packages

For reusable packages, expose entry points:

```toml
[project.entry-points."apmoe.experts"]
my_expert = "my_package.experts:MyExpert"
```

After `pip install my-package`, use the short name:

```json
{
  "class": "my_expert"
}
```

APMoE discovers entry points during `APMoEApp.from_config()`.

### Remote Primary With Local Fallback

Install:

```bash
pip install apmoe
```

Config shape:

```json
{
  "remote_fallback_policy": "transient_only",
  "experts": [
    {
      "name": "remote_face",
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
```

Use `transient_only` for outages and rate-limit style failures,
`any_remote_error` for aggressive degradation, and `disabled` to turn paired
fallback off.

### Redis-Backed Serving

Install:

```bash
pip install apmoe
```

Config shape:

```json
{
  "serving": {
    "rate_limit": 120,
    "rate_limit_store": "redis",
    "rate_limit_redis_url": "$APMOE_REDIS_URL",
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "$APMOE_REDIS_URL"
  }
}
```

Redis shares rate-limit and JWT invalidation state across workers and nodes.
If Redis operations fail after startup, APMoE falls back to process-local
memory and emits audit events.

---

## `modalities` â€” array, required

Each entry defines one input modality and its three-step processing chain.
Modality names must be **unique** across the list.

```json
{
  "name":      "image",
  "processor": "apmoe.modality.builtin.image.ImageProcessor",
  "pipeline": {
    "cleaner":    "apmoe.processing.builtin.image_cleaners.ImageCleaner",
    "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | âœ… | Canonical key for this modality (e.g. `"image"`, `"keystroke"`). Referenced by `experts[].modalities`. |
| `processor` | string | âœ… | Dotted import path **or** registered name of a `ModalityProcessor` subclass. Responsible for validating and preprocessing raw input into a `ModalityData` object. |
| `pipeline.cleaner` | string | âœ… | Dotted import path or registered name of a `CleanerStrategy` subclass. Runs first on the `ModalityData`. |
| `pipeline.anonymizer` | string | âœ… | Dotted import path or registered name of an `AnonymizerStrategy` subclass. Runs after the cleaner. |
| `pipeline.embedder` | string | âŒ | Dotted import path or registered name of an `EmbedderStrategy` subclass. **When omitted**, experts receive the preprocessed `ModalityData` directly (useful for experts that do their own feature extraction). **When present**, experts receive an `EmbeddingResult` (a dense feature vector). |

### Resolving `processor` / `pipeline.*` values

All four string fields accept either form:

- **Registered name** â€” a short key previously passed to `@registry.register("name")`.
- **Dotted import path** â€” a fully-qualified Python class path such as
  `"myproject.processors.MyVisualProcessor"`. The framework imports the module
  at startup and retrieves the attribute.

### Modality name constraints

- Must be a non-empty string after stripping whitespace.
- Must be unique across all modality entries.
- Every modality name used in an `experts[].modalities` list must appear here.

---

## `experts` â€” array, required

Each entry declares one expert plugin: which modalities it consumes, which
pretrained weights to load, and which class implements it.
Expert names must be **unique** across the list.

```json
{
  "name":       "face_age_expert",
  "class":      "apmoe.experts.builtin.FaceAgeExpert",
  "weights":    "./weights/face_age_expert.keras",
  "modalities": ["image"]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | âœ… | Unique identifier for this expert instance. Used as the key in aggregation `weights` maps and in per-expert breakdown output. |
| `class` | string | âœ… | Dotted import path or registered name of an `ExpertPlugin` subclass. |
| `weights` | string | local experts | Filesystem path to the pretrained weight file (`.keras`, `.onnx`, `.pt`, etc.). Resolved relative to the current working directory. |
| `endpoint` | string | remote experts | HTTP endpoint used by `RemoteExpert`/provider experts instead of a local weight file. |
| `modalities` | array of strings | âœ… | One or more modality names this expert consumes. Every name must appear in `modalities[].name`. An expert may consume a single modality **or** multiple (multi-modal expert). Must not be empty. |
| *(any extra key)* | any | âŒ | Additional expert-specific parameters (e.g. `"threshold"`, `"temperature"`) are collected into an `extra` dict and passed to the expert at bootstrap. |

Local experts use `weights`; remote experts use `endpoint`. The two fields are
mutually exclusive. Remote experts may set `fallback_expert` to name a standby
local expert that has `fallback_only=true`. Fallback-only experts load at
startup and appear in CLI health output, but they are excluded from normal
inference unless their paired remote expert fails.

### Model artifact integrity

Local experts may pin a SHA-256 digest for the configured `weights` file:

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

Remote experts should not rely on a hash returned by the remote model service.
Instead, configure an RSA-PSS-SHA256 signed manifest fetched from an allowed
endpoint and verified with a pinned public key:

```json
{
  "name": "remote_age_expert",
  "class": "apmoe.experts.remote.RemoteExpert",
  "endpoint": "https://models.example.com/predict",
  "modalities": ["keystroke"],
  "integrity": {
    "manifest_url": "https://models.example.com/.well-known/apmoe-manifest.json",
    "manifest_public_key": "$APMOE_REMOTE_MANIFEST_PUBLIC_KEY",
    "manifest_required": true,
    "signature_algorithm": "RSA-PSS-SHA256"
  }
}
```

The manifest signature covers canonical JSON excluding the `signature` field
and must include `expert_name`, `model_id`, `model_version`, `endpoint_origin`,
`model_digest` or `artifact_digest`, `issued_at`, `expires_at`, and `signature`.
Keep the private signing key outside the remote model runtime, ideally in
release CI or a KMS-backed signing process.

### Single-modality expert

```json
{
  "name":       "keystroke_age_expert",
  "class":      "apmoe.experts.builtin.KeystrokeAgeExpert",
  "weights":    "./weights/keystroke_age_expert.onnx",
  "modalities": ["keystroke"]
}
```

The expert's `predict()` receives `{"keystroke": <ProcessedInput>}`.

### Multi-modal expert

```json
{
  "name":       "multimodal_expert",
  "class":      "myproject.experts.MultiModalExpert",
  "weights":    "./weights/multimodal_expert.pt",
  "modalities": ["image", "keystroke"]
}
```

The expert's `predict()` receives `{"image": <ProcessedInput>, "keystroke": <ProcessedInput>}`.
The expert is responsible for combining them internally.

### Expert with extra parameters

```json
{
  "name":       "face_age_expert",
  "class":      "myproject.experts.CalibratedCNNExpert",
  "weights":    "./weights/face.pt",
  "modalities": ["image"],
  "temperature": 1.5,
  "threshold":   0.6
}
```

`temperature` and `threshold` land in `expert_config.extra` and are available
to the constructor of `CalibratedCNNExpert`.

### Remote primary with local fallback

Pair a remote expert with a standby local expert when a degraded local
prediction is preferable to failing during a remote outage:

```json
{
  "experts": [
    {
      "name": "remote_face",
      "class": "apmoe.experts.remote.RemoteExpert",
      "endpoint": "https://models.example.com/predict",
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
```

The fallback expert receives the same processed modality inputs as the remote
primary, so both experts must be compatible with the configured modality
pipeline.

---

## `aggregation` â€” object, required

Defines how individual expert predictions are combined into a single final answer.

```json
{
  "strategy":     "apmoe.aggregation.builtin.WeightedAverageAggregator",
  "weights": {
    "face_age_expert":      0.6,
    "keystroke_age_expert": 0.4
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `strategy` | string | âœ… | Dotted import path or registered name of an `AggregatorStrategy` subclass. |
| `weights` | object | âŒ | Expert-name â†’ numeric weight map. Used by `WeightedAverageAggregator`. Weights do **not** need to sum to 1 â€” they are normalised internally. |
| *(any extra key)* | any | âŒ | Additional strategy-specific parameters collected into `extra`. |

### Built-in strategies

| Class path | Description |
|---|---|
| `apmoe.aggregation.builtin.WeightedAverageAggregator` | Weighted average of predicted ages; weights from `aggregation.weights` (falls back to uniform if omitted). |
| `apmoe.aggregation.builtin.MedianAggregator` | Median of predicted ages; ignores confidence. |
| `apmoe.aggregation.builtin.ConfidenceWeightedAggregator` | Weighted average where each expert's weight equals its self-reported confidence. |

---

## Resilience settings -- optional

These top-level fields control how the prediction path behaves when remote
experts or individual runnable experts fail:

```json
{
  "apmoe": {
    "expert_failure_policy": "fail_fast",
    "remote_fallback_policy": "transient_only",
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
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `expert_failure_policy` | string | `"fail_fast"` | `"fail_fast"` preserves historical behavior: any runnable expert failure aborts prediction. `"skip_failed"` records failed runnable experts in `Prediction.metadata["failed_experts"]` and aggregates the remaining successful outputs. If every runnable expert fails, the pipeline raises `PipelineError`. |
| `remote_fallback_policy` | string | `"transient_only"` | Paired remote-to-local fallback behavior. `"transient_only"` falls back on timeout, network error, transient HTTP `429/502/503/504`, and open circuit. `"any_remote_error"` falls back on any remote `ExpertError`. `"disabled"` never uses paired fallback. |
| `remote_retry.max_attempts` | integer | `3` | Total attempts for each remote inference call, including the first try. Must be at least 1. |
| `remote_retry.initial_delay_s` | number | `0.25` | First retry delay in seconds. |
| `remote_retry.max_delay_s` | number | `2.0` | Upper bound for exponential backoff delay. Must be greater than or equal to `initial_delay_s`. |
| `remote_retry.backoff_multiplier` | number | `2.0` | Multiplier applied after each failed attempt. |
| `remote_retry.jitter` | boolean | `true` | Randomizes retry sleep between 0 and the calculated delay to avoid synchronized retry bursts. |
| `remote_circuit_breaker.enabled` | boolean | `true` | Enables the per-`RemoteExpert` in-memory circuit breaker. |
| `remote_circuit_breaker.failure_threshold` | integer | `5` | Consecutive remote call failures required before the circuit opens. |
| `remote_circuit_breaker.recovery_timeout_s` | number | `30.0` | Time before an open circuit moves to half-open and allows one trial call. |

Remote retry applies only to outbound `RemoteExpert` inference calls. It retries
timeouts, network errors, and HTTP `429`, `502`, `503`, and `504`. It does not
retry request/template errors, non-JSON responses, response-size violations,
response mapping errors, or non-transient HTTP statuses such as `400`, `401`,
`403`, and `404`.

Circuit breakers are process-local and per remote expert instance:

- `closed`: requests run normally.
- `open`: calls fail immediately with `ExpertError` and circuit metadata.
- `half-open`: after cooldown, one trial request is allowed; success closes the
  circuit and failure reopens it.

---

## `serving` â€” object, optional

Controls the FastAPI/uvicorn HTTP serving layer. The entire block may be
omitted; all fields have defaults.

```json
{
  "host":         "0.0.0.0",
  "port":         8000,
  "workers":      4,
  "cors_origins": ["*"],
  "rate_limit":   null,
  "rate_limit_store": "memory",
  "rate_limit_redis_url": null,
  "rate_limit_key_prefix": "apmoe:rate:",
  "log_level":    "info",
  "authentication_enabled": true,
  "authorization_enabled": true,
  "token_invalidation_store": "memory",
  "token_invalidation_redis_url": null,
  "token_invalidation_key_prefix": "apmoe:jwt:invalid:"
}
```

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `host` | string | `"0.0.0.0"` | â€” | Network interface for uvicorn to bind to. Use `"127.0.0.1"` to restrict to localhost. |
| `port` | integer | `8000` | 1 â€“ 65535 | TCP port number. |
| `workers` | integer | `4` | â‰¥ 1 | Number of uvicorn worker processes. |
| `cors_origins` | array of strings | `["*"]` | â€” | Allowed CORS origin patterns. Use `["*"]` to permit all origins, or list explicit origins like `["https://myapp.com"]`. |
| `rate_limit` | integer \| null | `null` | â‰¥ 1 | Maximum requests per minute per client IP. `null` disables rate limiting entirely. |
| `log_level` | string | `"info"` | `"debug"` \| `"info"` \| `"warning"` \| `"error"` \| `"critical"` | Uvicorn log verbosity. |
| `authentication_enabled` | boolean | `true` | - | Enables stateless authentication middleware. When true, `create_api(...)` requires a `StatelessAuthProvider` or fails closed. |
| `authorization_enabled` | boolean | `true` | - | Enables scope authorization middleware. Requires authentication to be enabled too. |
| `rate_limit_store` | string | `"memory"` | `"memory"` or `"redis"` | Backend for rate limiting. Redis shares limits across workers/nodes. |
| `rate_limit_redis_url` | string \| null | `null` | required when Redis is selected | Redis URL for shared rate limiting. |
| `rate_limit_key_prefix` | string | `"apmoe:rate:"` | - | Redis key prefix for rate-limit entries. |
| `token_invalidation_store` | string | `"memory"` | `"memory"` or `"redis"` | Backend for JWT `jti` invalidation. Redis shares invalidation across workers/nodes. |
| `token_invalidation_redis_url` | string \| null | `null` | required when Redis is selected | Redis URL for shared token invalidation. |
| `token_invalidation_key_prefix` | string | `"apmoe:jwt:invalid:"` | - | Redis key prefix for invalidated JWT ids. |

Redis-backed rate limiting and token invalidation fall back to process-local
memory if Redis operations fail after startup. APMoE emits
`redis_rate_limit_fallback` or `redis_token_invalidation_fallback` audit events
when this happens. The fallback keeps the API available, but fallback state is
local to each worker and is not synchronized back into Redis.

---

## `environment` and `security`

`apmoe.environment` defaults to `"development"` and may be one of
`"development"`, `"test"`, `"staging"`, or `"production"`. `APMOE_ENV`
overrides it at load time.

For the full operational security reference, including authn/authz, shared
stores, remote endpoint policy, signed manifests, audit events, redaction, and
the production checklist, see [security.md](security.md).

`apmoe.security` controls framework-level hardening:

```json
{
  "environment": "production",
  "security": {
    "remote_endpoint_allowlist": ["models.example.com", "*.trusted.ai"],
    "remote_enforce_https": true,
    "remote_allow_private_networks": false,
    "remote_response_max_bytes": 1048576,
    "audit_enabled": true,
    "audit_success_events": true
  }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `remote_endpoint_allowlist` | array of strings \| null | `null` | Hostname allowlist for remote expert endpoints and manifest URLs. Exact hosts and wildcard suffixes like `"*.example.com"` are supported. Non-production treats missing allowlist as `["*"]`; production remote experts require an explicit non-wildcard allowlist. |
| `remote_enforce_https` | boolean | `true` | Reject HTTP remote endpoints unless private networks are explicitly allowed. |
| `remote_allow_private_networks` | boolean | `false` | Reject localhost, loopback, private, link-local, reserved, multicast, and metadata IP hosts unless set to `true`. |
| `remote_response_max_bytes` | integer | `1048576` | Maximum remote response or manifest bytes accepted before JSON parsing. |
| `audit_enabled` | boolean | `true` | Enables structured security audit events. |
| `audit_success_events` | boolean | `true` | Emits successful authn/authz events as well as denials/blocks. |

Production fail-closed rule: if any expert has an `endpoint`, production config
must provide a concrete `remote_endpoint_allowlist`; missing, empty, or `["*"]`
is rejected at config load.

Remote response protection rejects responses above the configured cap and
obvious non-JSON content types before parsing. Per-expert
`endpoint_response_max_bytes` overrides the global cap.

---

## Environment variable overrides

These variables override the corresponding `serving` fields **after** the JSON
file is loaded. They take precedence over anything in the file.

| Variable | Overrides | Type | Example |
|---|---|---|---|
| `APMOE_SERVING_HOST` | `serving.host` | string | `APMOE_SERVING_HOST=127.0.0.1` |
| `APMOE_SERVING_PORT` | `serving.port` | integer | `APMOE_SERVING_PORT=9000` |
| `APMOE_SERVING_WORKERS` | `serving.workers` | integer | `APMOE_SERVING_WORKERS=8` |
| `APMOE_SERVING_LOG_LEVEL` | `serving.log_level` | string | `APMOE_SERVING_LOG_LEVEL=debug` |
| `APMOE_SERVING_RATE_LIMIT` | `serving.rate_limit` | integer | `APMOE_SERVING_RATE_LIMIT=60` |
| `APMOE_SERVING_RATE_LIMIT_STORE` | `serving.rate_limit_store` | string | `APMOE_SERVING_RATE_LIMIT_STORE=redis` |
| `APMOE_SERVING_RATE_LIMIT_REDIS_URL` | `serving.rate_limit_redis_url` | string | `APMOE_SERVING_RATE_LIMIT_REDIS_URL=redis://localhost:6379/0` |
| `APMOE_SERVING_RATE_LIMIT_KEY_PREFIX` | `serving.rate_limit_key_prefix` | string | `APMOE_SERVING_RATE_LIMIT_KEY_PREFIX=apmoe:rate:` |
| `APMOE_SERVING_CORS_ORIGINS` | `serving.cors_origins` | comma-separated strings | `APMOE_SERVING_CORS_ORIGINS=https://a.com,https://b.com` |
| `APMOE_SERVING_AUTHENTICATION_ENABLED` | `serving.authentication_enabled` | boolean | `APMOE_SERVING_AUTHENTICATION_ENABLED=false` |
| `APMOE_SERVING_AUTHORIZATION_ENABLED` | `serving.authorization_enabled` | boolean | `APMOE_SERVING_AUTHORIZATION_ENABLED=false` |
| `APMOE_SERVING_TOKEN_INVALIDATION_STORE` | `serving.token_invalidation_store` | string | `APMOE_SERVING_TOKEN_INVALIDATION_STORE=redis` |
| `APMOE_SERVING_TOKEN_INVALIDATION_REDIS_URL` | `serving.token_invalidation_redis_url` | string | `APMOE_SERVING_TOKEN_INVALIDATION_REDIS_URL=redis://localhost:6379/0` |
| `APMOE_SERVING_TOKEN_INVALIDATION_KEY_PREFIX` | `serving.token_invalidation_key_prefix` | string | `APMOE_SERVING_TOKEN_INVALIDATION_KEY_PREFIX=apmoe:jwt:invalid:` |
| `APMOE_ENV` | `environment` | string | `APMOE_ENV=production` |

If a variable is set but cannot be cast to the target type (e.g. `APMOE_SERVING_PORT=abc`),
`load_config()` raises a `ConfigurationError` immediately.

Boolean env vars accept `true/false`, `1/0`, `yes/no`, and `on/off`.

---

## Validation rules

The following cross-field rules are enforced at load time and produce a
`ConfigurationError` with a clear message if violated:

1. **Modality names unique** â€” no two entries in `modalities` may share the same `name`.
2. **Expert names unique** â€” no two entries in `experts` may share the same `name`.
3. **Expert modalities declared** â€” every string in `experts[].modalities` must match
   a `name` in the `modalities` array.
4. **Expert modalities non-empty** â€” `experts[].modalities` must contain at least one entry.
5. **Port in range** â€” `serving.port` must be between 1 and 65535.
6. **Workers â‰¥ 1** â€” `serving.workers` must be at least 1.
7. **Modality name non-empty** â€” `modalities[].name` must not be blank after stripping whitespace.

---

## Minimal configuration example

The smallest valid config has one modality and one expert (no `serving` block needed):

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "image",
        "processor": "myproject.processors.ImageProcessor",
        "pipeline": {
          "cleaner":    "myproject.cleaners.ImageCleaner",
          "anonymizer": "myproject.anonymizers.ImageAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "face_expert",
        "class":      "myproject.experts.FaceExpert",
        "weights":    "./weights/face.pt",
        "modalities": ["image"]
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
    }
  }
}
```

---

## Full configuration example

```json
{
  "apmoe": {
    "modalities": [
      {
        "name":      "image",
        "processor": "apmoe.modality.builtin.image.ImageProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.image_cleaners.ImageCleaner",
          "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
        }
      },
      {
        "name":      "keystroke",
        "processor": "apmoe.modality.builtin.keystroke.KeystrokeProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.cleaners.KeystrokeCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.KeystrokeAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "face_age_expert",
        "class":      "apmoe.experts.builtin.FaceAgeExpert",
        "weights":    "./weights/face_age_expert.keras",
        "modalities": ["image"]
      },
      {
        "name":       "keystroke_age_expert",
        "class":      "apmoe.experts.builtin.KeystrokeAgeExpert",
        "weights":    "./weights/keystroke_age_expert.onnx",
        "modalities": ["keystroke"]
      },
      {
        "name":       "multimodal_expert",
        "class":      "myproject.experts.MultiModalExpert",
        "weights":    "./weights/multimodal.pt",
        "modalities": ["image", "keystroke"],
        "threshold":  0.7
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator",
      "weights": {
        "face_age_expert":   0.35,
        "keystroke_age_expert":  0.40,
        "multimodal_expert": 0.25
      }
    },
    "serving": {
      "host":         "0.0.0.0",
      "port":         8000,
      "workers":      4,
      "cors_origins": ["https://myapp.com"],
      "rate_limit":   120,
      "log_level":    "info"
    }
  }
}
```

