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

The JSON file must have a single top-level key `"apmoe"` containing four
sections — `modalities`, `experts`, `aggregation`, and `serving`:

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
`serving`, `environment`, and `security` are **optional** — all their fields
have defaults.

---

## `modalities` — array, required

Each entry defines one input modality and its three-step processing chain.
Modality names must be **unique** across the list.

```json
{
  "name":      "visual",
  "processor": "apmoe.modality.builtin.visual.VisualProcessor",
  "pipeline": {
    "cleaner":    "apmoe.processing.builtin.cleaners.ImageCleaner",
    "anonymizer": "apmoe.processing.builtin.anonymizers.FaceAnonymizer",
    "embedder":   "apmoe.processing.builtin.embedders.MobileNetEmbedder"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | Canonical key for this modality (e.g. `"visual"`, `"audio"`, `"eeg"`). Referenced by `experts[].modalities`. |
| `processor` | string | ✅ | Dotted import path **or** registered name of a `ModalityProcessor` subclass. Responsible for validating and preprocessing raw input into a `ModalityData` object. |
| `pipeline.cleaner` | string | ✅ | Dotted import path or registered name of a `CleanerStrategy` subclass. Runs first on the `ModalityData`. |
| `pipeline.anonymizer` | string | ✅ | Dotted import path or registered name of an `AnonymizerStrategy` subclass. Runs after the cleaner. |
| `pipeline.embedder` | string | ❌ | Dotted import path or registered name of an `EmbedderStrategy` subclass. **When omitted**, experts receive the preprocessed `ModalityData` directly (useful for experts that do their own feature extraction). **When present**, experts receive an `EmbeddingResult` (a dense feature vector). |

### Resolving `processor` / `pipeline.*` values

All four string fields accept either form:

- **Registered name** — a short key previously passed to `@registry.register("name")`.
- **Dotted import path** — a fully-qualified Python class path such as
  `"myproject.processors.MyVisualProcessor"`. The framework imports the module
  at startup and retrieves the attribute.

### Modality name constraints

- Must be a non-empty string after stripping whitespace.
- Must be unique across all modality entries.
- Every modality name used in an `experts[].modalities` list must appear here.

---

## `experts` — array, required

Each entry declares one expert plugin: which modalities it consumes, which
pretrained weights to load, and which class implements it.
Expert names must be **unique** across the list.

```json
{
  "name":       "face_age_expert",
  "class":      "apmoe.experts.builtin.CNNAgeExpert",
  "weights":    "./weights/visual_age_expert.pt",
  "modalities": ["visual"]
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | ✅ | Unique identifier for this expert instance. Used as the key in aggregation `weights` maps and in per-expert breakdown output. |
| `class` | string | ✅ | Dotted import path or registered name of an `ExpertPlugin` subclass. |
| `weights` | string | ✅ | Filesystem path to the pretrained weight file (`.pt`, `.onnx`, etc.). Resolved relative to the current working directory. |
| `modalities` | array of strings | ✅ | One or more modality names this expert consumes. Every name must appear in `modalities[].name`. An expert may consume a single modality **or** multiple (multi-modal expert). Must not be empty. |
| *(any extra key)* | any | ❌ | Additional expert-specific parameters (e.g. `"threshold"`, `"temperature"`) are collected into an `extra` dict and passed to the expert at bootstrap. |

### Model artifact integrity

Local experts may pin a SHA-256 digest for the configured `weights` file:

```json
{
  "name": "face_age_expert",
  "class": "apmoe.experts.builtin.CNNAgeExpert",
  "weights": "./weights/visual_age_expert.pt",
  "modalities": ["visual"],
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
  "name":       "audio_age_expert",
  "class":      "apmoe.experts.builtin.MLPAgeExpert",
  "weights":    "./weights/audio_age_expert.pt",
  "modalities": ["audio"]
}
```

The expert's `predict()` receives `{"audio": <ProcessedInput>}`.

### Multi-modal expert

```json
{
  "name":       "multimodal_expert",
  "class":      "myproject.experts.MultiModalExpert",
  "weights":    "./weights/multimodal_expert.pt",
  "modalities": ["visual", "audio"]
}
```

The expert's `predict()` receives `{"visual": <ProcessedInput>, "audio": <ProcessedInput>}`.
The expert is responsible for combining them internally.

### Expert with extra parameters

```json
{
  "name":       "face_age_expert",
  "class":      "myproject.experts.CalibratedCNNExpert",
  "weights":    "./weights/face.pt",
  "modalities": ["visual"],
  "temperature": 1.5,
  "threshold":   0.6
}
```

`temperature` and `threshold` land in `expert_config.extra` and are available
to the constructor of `CalibratedCNNExpert`.

---

## `aggregation` — object, required

Defines how individual expert predictions are combined into a single final answer.

```json
{
  "strategy":     "apmoe.aggregation.builtin.WeightedAverageAggregator",
  "weights": {
    "face_age_expert":  0.5,
    "audio_age_expert": 0.3,
    "eeg_age_expert":   0.2
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `strategy` | string | ✅ | Dotted import path or registered name of an `AggregatorStrategy` subclass. |
| `weights` | object | ❌ | Expert-name → numeric weight map. Used by `WeightedAverageAggregator`. Weights do **not** need to sum to 1 — they are normalised internally. |
| `weights_path` | string | ❌ | Path to a pretrained combiner model file. Used by `LearnedCombiner`. |
| *(any extra key)* | any | ❌ | Additional strategy-specific parameters collected into `extra`. |

### Built-in strategies (Phase 6)

| Class path | Description |
|---|---|
| `apmoe.aggregation.builtin.WeightedAverageAggregator` | Weighted average of predicted ages; weights from `aggregation.weights` (falls back to uniform if omitted). |
| `apmoe.aggregation.builtin.MedianAggregator` | Median of predicted ages; ignores confidence. |
| `apmoe.aggregation.builtin.ConfidenceWeightedAggregator` | Weighted average where each expert's weight equals its self-reported confidence. |
| `apmoe.aggregation.builtin.LearnedCombiner` | Small pretrained model that takes expert predictions + confidences as input. Requires `weights_path`. |

---

## `serving` — object, optional

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
| `host` | string | `"0.0.0.0"` | — | Network interface for uvicorn to bind to. Use `"127.0.0.1"` to restrict to localhost. |
| `port` | integer | `8000` | 1 – 65535 | TCP port number. |
| `workers` | integer | `4` | ≥ 1 | Number of uvicorn worker processes. |
| `cors_origins` | array of strings | `["*"]` | — | Allowed CORS origin patterns. Use `["*"]` to permit all origins, or list explicit origins like `["https://myapp.com"]`. |
| `rate_limit` | integer \| null | `null` | ≥ 1 | Maximum requests per minute per client IP. `null` disables rate limiting entirely. |
| `log_level` | string | `"info"` | `"debug"` \| `"info"` \| `"warning"` \| `"error"` \| `"critical"` | Uvicorn log verbosity. |
| `authentication_enabled` | boolean | `true` | - | Enables stateless authentication middleware. When true, `create_api(...)` requires a `StatelessAuthProvider` or fails closed. |
| `authorization_enabled` | boolean | `true` | - | Enables scope authorization middleware. Requires authentication to be enabled too. |
| `rate_limit_store` | string | `"memory"` | `"memory"` or `"redis"` | Backend for rate limiting. Redis shares limits across workers/nodes. |
| `rate_limit_redis_url` | string \| null | `null` | required when Redis is selected | Redis URL for shared rate limiting. |
| `rate_limit_key_prefix` | string | `"apmoe:rate:"` | - | Redis key prefix for rate-limit entries. |
| `token_invalidation_store` | string | `"memory"` | `"memory"` or `"redis"` | Backend for JWT `jti` invalidation. Redis shares invalidation across workers/nodes. |
| `token_invalidation_redis_url` | string \| null | `null` | required when Redis is selected | Redis URL for shared token invalidation. |
| `token_invalidation_key_prefix` | string | `"apmoe:jwt:invalid:"` | - | Redis key prefix for invalidated JWT ids. |

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

1. **Modality names unique** — no two entries in `modalities` may share the same `name`.
2. **Expert names unique** — no two entries in `experts` may share the same `name`.
3. **Expert modalities declared** — every string in `experts[].modalities` must match
   a `name` in the `modalities` array.
4. **Expert modalities non-empty** — `experts[].modalities` must contain at least one entry.
5. **Port in range** — `serving.port` must be between 1 and 65535.
6. **Workers ≥ 1** — `serving.workers` must be at least 1.
7. **Modality name non-empty** — `modalities[].name` must not be blank after stripping whitespace.

---

## Minimal configuration example

The smallest valid config has one modality and one expert (no `serving` block needed):

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "visual",
        "processor": "myproject.processors.VisualProcessor",
        "pipeline": {
          "cleaner":    "myproject.cleaners.ImageCleaner",
          "anonymizer": "myproject.anonymizers.FaceAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "face_expert",
        "class":      "myproject.experts.FaceExpert",
        "weights":    "./weights/face.pt",
        "modalities": ["visual"]
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
        "name":      "visual",
        "processor": "apmoe.modality.builtin.visual.VisualProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.cleaners.ImageCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.FaceAnonymizer",
          "embedder":   "apmoe.processing.builtin.embedders.MobileNetEmbedder"
        }
      },
      {
        "name":      "audio",
        "processor": "apmoe.modality.builtin.audio.AudioProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.cleaners.AudioCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.VoiceAnonymizer"
        }
      },
      {
        "name":      "eeg",
        "processor": "apmoe.modality.builtin.eeg.EEGProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.cleaners.EEGCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.EEGAnonymizer",
          "embedder":   "apmoe.processing.builtin.embedders.EEGEmbedder"
        }
      }
    ],
    "experts": [
      {
        "name":       "face_age_expert",
        "class":      "apmoe.experts.builtin.CNNAgeExpert",
        "weights":    "./weights/visual_age_expert.pt",
        "modalities": ["visual"]
      },
      {
        "name":       "audio_age_expert",
        "class":      "apmoe.experts.builtin.MLPAgeExpert",
        "weights":    "./weights/audio_age_expert.pt",
        "modalities": ["audio"]
      },
      {
        "name":       "eeg_age_expert",
        "class":      "apmoe.experts.builtin.EEGAgeExpert",
        "weights":    "./weights/eeg_age_expert.pt",
        "modalities": ["eeg"]
      },
      {
        "name":       "multimodal_expert",
        "class":      "myproject.experts.MultiModalExpert",
        "weights":    "./weights/multimodal.pt",
        "modalities": ["visual", "audio"],
        "threshold":  0.7
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator",
      "weights": {
        "face_age_expert":   0.35,
        "audio_age_expert":  0.25,
        "eeg_age_expert":    0.15,
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
