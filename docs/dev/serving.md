# Serving Layer (`apmoe.serving`)

The serving layer adapts a bootstrapped `APMoEApp` to a FastAPI application. It
exposes the versioned HTTP API, OpenAPI docs, request logging, CORS, rate
limiting, optional stateless authentication/authorization, and the legacy
authentication hook.

`APMoEApp.serve()` uses this layer internally. Applications that embed APMoE in
another ASGI app can call `apmoe.serving.app_factory.create_api(...)` directly.

---

## API Surface

Current endpoints are mounted under `/v1`:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/predict` | Run age prediction from a multimodal JSON object. |
| `GET` | `/v1/health` | Readiness/liveness status from expert load state. |
| `GET` | `/v1/info` | Runtime metadata, loaded components, and redacted config. |
| `GET` | `/docs` | Swagger UI generated from the current OpenAPI schema. |
| `GET` | `/redoc` | ReDoc generated from the current OpenAPI schema. |
| `GET` | `/openapi.json` | Raw OpenAPI schema. |

Legacy unversioned routes remain mounted for compatibility:

| Current | Legacy |
|---|---|
| `/v1/predict` | `/predict` |
| `/v1/health` | `/health` |
| `/v1/info` | `/info` |

All API responses include `X-Correlation-ID`. Versioned `/v1/*` route
responses include `X-API-Version: 1`. Legacy route responses also include
`Deprecation`, `Sunset`, and `Link` headers; new clients should use `/v1/*`.

---

## `POST /v1/predict`

Runs the configured APMoE inference pipeline.

Request:

- Header: `Content-Type: application/json`
- Body: a JSON object mapping modality names to modality payloads.
- The root body must be an object, not an array.
- Keys should match configured modality names such as `image` or `keystroke`.
- String values are forwarded as UTF-8 bytes.
- Non-string JSON values are serialized to UTF-8 JSON bytes.
- Missing modalities are allowed; dependent experts are listed in
  `skipped_experts`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"keystroke\": [[8, 0, 95.0], [13, 0, 100.0]]}"
```

Successful response (`200`):

```json
{
  "predicted_age": 32.5,
  "confidence": 0.82,
  "confidence_interval": null,
  "per_expert_outputs": [
    {
      "expert_name": "keystroke_age_expert",
      "consumed_modalities": ["keystroke"],
      "predicted_age": 32.5,
      "confidence": 0.82,
      "metadata": {
        "predicted_group": "26-35"
      }
    }
  ],
  "skipped_experts": ["face_age_expert"],
  "metadata": {
    "pipeline_latency_s": 0.006,
    "available_modalities": ["keystroke"],
    "failed_modalities": {}
  }
}
```

Response fields:

| Field | Type | Description |
|---|---|---|
| `predicted_age` | number | Aggregated age estimate in years. |
| `confidence` | number | Aggregated confidence in `[0.0, 1.0]`. |
| `confidence_interval` | array or null | Optional `[lower, upper]` age bounds. |
| `per_expert_outputs` | array | One row per expert that produced an output. |
| `skipped_experts` | array | Experts skipped because required modalities were missing. |
| `metadata` | object | Pipeline latency, available modalities, failed modalities, confidence threshold data, and recommendations when available. |

Per-expert output fields:

| Field | Type | Description |
|---|---|---|
| `expert_name` | string | Expert instance name. |
| `consumed_modalities` | array | Modalities consumed by that expert. |
| `predicted_age` | number | Expert-level age estimate in years. |
| `confidence` | number | Expert score in `[0.0, 1.0]`, or `-1.0` when not reported. |
| `metadata` | object | Expert-specific details such as predicted age group or feature coverage. |

Error responses:

| Status | Body | Cause |
|---|---|---|
| `401` | `{"detail": "Unauthorized."}` | Stateless authentication is enabled and credentials are missing or invalid. |
| `403` | `{"detail": "Forbidden."}` | Authorization is enabled and the token lacks the required scope. |
| `422` | FastAPI validation detail | Body is malformed JSON or the root value is not an object. |
| `429` | `{"detail": "Rate limit exceeded: ... "}` | Request count exceeded `serving.rate_limit`. |
| `503` | `{"detail": "..."}` | The pipeline could not run any expert. |
| `500` | `{"detail": "..."}` | A framework error escaped prediction handling. |

---

## `GET /v1/health`

Returns expert readiness from `expert_registry.health_check()`.

Example:

```bash
curl http://127.0.0.1:8000/v1/health
```

Healthy response (`200`):

```json
{
  "status": "healthy",
  "experts": {
    "face_age_expert": true,
    "keystroke_age_expert": true
  }
}
```

Degraded response (`503`):

```json
{
  "status": "degraded",
  "experts": {
    "face_age_expert": true,
    "keystroke_age_expert": false
  }
}
```

An empty expert registry is considered healthy.

---

## `GET /v1/info`

Returns `APMoEApp.get_info()` for diagnostics and operational inventory.

Example:

```bash
curl http://127.0.0.1:8000/v1/info
```

Current response keys include:

| Field | Type | Description |
|---|---|---|
| `version` | string | Installed `apmoe` package version. |
| `experts` | array | `get_info()` output for loaded expert instances. |
| `modalities` | array | Configured modality names. |
| `aggregator` | object | Active aggregator metadata. |
| `serving` | object | Redacted serving configuration. |
| `environment` | string | Runtime environment, such as `development` or `production`. |
| `security` | object | Redacted security configuration. |
| `confidence_threshold` | number or null | Confidence gate used for recommendations. |
| `expert_failure_policy` | string | Expert failure behavior. |
| `remote_fallback_policy` | string | Remote-to-local fallback behavior. |
| `remote_retry` | object | Remote expert retry configuration. |
| `remote_circuit_breaker` | object | Remote expert circuit-breaker configuration. |

Secrets and sensitive values are redacted before they are returned.

---

## OpenAPI Docs

FastAPI serves generated documentation automatically:

| Path | Description |
|---|---|
| `/docs` | Swagger UI with request examples for keystroke triples, IKDD text, precomputed keystroke features, and image plus keystroke payloads. |
| `/redoc` | ReDoc view of the same OpenAPI schema. |
| `/openapi.json` | Machine-readable OpenAPI schema. |

The OpenAPI metadata lives in `apmoe.serving.openapi_schemas`. Update that
module when route shapes or examples change.

---

## Headers

Every response:

| Header | Description |
|---|---|
| `X-Correlation-ID` | Inbound safe `X-Correlation-ID` value or a generated UUID4. Use it to correlate logs and client errors. |

Versioned route responses:

| Header | Description |
|---|---|
| `X-API-Version: 1` | Current HTTP API version. |

Legacy route responses:

| Header | Description |
|---|---|
| `X-API-Version: 1` | Current HTTP API version. |
| `Deprecation` | Date indicating the route is deprecated. |
| `Sunset` | Planned end of the legacy route migration window. |
| `Link` | Link relation pointing clients to documentation. |

Rate-limit failures:

| Header | Description |
|---|---|
| `Retry-After: 60` | Clients should wait before retrying. |

Authentication failures:

| Header | Description |
|---|---|
| `WWW-Authenticate: Bearer` | Returned by stateless authentication on missing or invalid credentials. |

---

## Middleware

`create_api(...)` configures:

- CORS from `serving.cors_origins`.
- Structured request logging and correlation IDs.
- Rate limiting when `serving.rate_limit` is set.
- Either legacy `AuthMiddleware` or stateless authentication/authorization.

The legacy `auth_plugin` path is mutually exclusive with stateless
`security_provider` and `authorization_policy`.

---

## Request Logging

`RequestLoggingMiddleware`:

- accepts a safe inbound `X-Correlation-ID` value or generates a UUID4 value;
- stores it as `request.state.correlation_id`;
- returns it in the `X-Correlation-ID` response header;
- logs method, path, redacted query string, status, duration, and client host.

Sensitive query parameters such as `token`, `api_key`, and `secret` are
redacted before logging.

---

## Rate Limiting

`RateLimitMiddleware` implements a per-client-IP sliding window:

- window size: 60 seconds;
- limit: `serving.rate_limit` requests per minute;
- overflow response: `429` with `Retry-After: 60`.

Stores:

| Store | Behavior |
|---|---|
| `memory` | Process-local. With multiple workers, each worker has its own window. |
| `redis` | Shared across workers and nodes. If Redis fails after startup, APMoE audits the fallback and uses process-local memory for that worker. |

Config:

```json
{
  "serving": {
    "rate_limit": 120,
    "rate_limit_store": "redis",
    "rate_limit_redis_url": "redis://localhost:6379/0"
  }
}
```

---

## Stateless Authentication And Authorization

Stateless security is controlled by serving config:

```json
{
  "serving": {
    "authentication_enabled": true,
    "authorization_enabled": true
  }
}
```

When authentication is enabled, `create_api(...)` fails closed unless a
`StatelessAuthProvider` is supplied. For local demos, disable both flags.

Default scopes:

| Route | Required scope |
|---|---|
| `POST /v1/predict`, `POST /predict` | `predict` |
| `GET /v1/info`, `GET /info` | `info:read` |
| `GET /v1/health`, `GET /health` | public |

JWT bearer setup:

```python
from apmoe.core.app import APMoEApp
from apmoe.serving.app_factory import create_api
from apmoe.serving.middleware import JWTBearerAuthProvider

apmoe_app = APMoEApp.from_config("config.json")
provider = JWTBearerAuthProvider(
    signing_key="replace-with-secret-or-public-key",
    algorithms=["HS256"],
    issuer="https://issuer.example.com",
    audience="apmoe-api",
)
api = create_api(apmoe_app, security_provider=provider)
```

JWTs must include:

- `sub`: subject/principal id;
- `jti`: stable token id used for invalidation;
- `exp`: expiry timestamp;
- `scope` or `scopes`: permission strings.

Token invalidation uses `serving.token_invalidation_store`. Use Redis for
shared invalidation across workers:

```json
{
  "serving": {
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "redis://localhost:6379/0"
  }
}
```

---

## Legacy Authentication Plugin

Legacy binary auth is still supported through `create_api(auth_plugin=...)`.
It is opt-in and mutually exclusive with the stateless provider/policy.

Default excluded paths:

- `/health`
- `/info`
- `/v1/health`
- `/v1/info`

Example:

```python
from starlette.requests import Request
from apmoe.serving.middleware import AuthPlugin


class ApiKeyAuth(AuthPlugin):
    def __init__(self, valid_key: str) -> None:
        self._valid_key = valid_key

    def authenticate(self, request: Request) -> bool:
        return request.headers.get("X-API-Key") == self._valid_key
```

---

## Embedding In Another ASGI App

```python
from apmoe.core.app import APMoEApp
from apmoe.serving.app_factory import create_api

apmoe_app = APMoEApp.from_config("config.json")
api = create_api(apmoe_app)
```

Multi-worker `APMoEApp.serve()` uses the factory
`apmoe.serving.app_factory:create_worker_app`. Each worker reads
`APMOE_CONFIG_PATH`, bootstraps its own `APMoEApp`, and returns a FastAPI app.

---

## Config Used By Serving

Serving reads these fields:

- `host`
- `port`
- `workers`
- `cors_origins`
- `rate_limit`
- `log_level`
- `rate_limit_store`
- `rate_limit_redis_url`
- `rate_limit_key_prefix`
- `authentication_enabled`
- `authorization_enabled`
- `token_invalidation_store`
- `token_invalidation_redis_url`
- `token_invalidation_key_prefix`

Environment overrides are documented in [configuration.md](configuration.md).
