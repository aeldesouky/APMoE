# Serving Layer (`apmoe.serving`)

The serving layer adapts a bootstrapped `APMoEApp` to a FastAPI application.
It provides HTTP endpoints plus middleware for logging, CORS, rate limiting,
and optional authentication.

---

## Architecture

- `apmoe.serving.app_factory.create_api(app, ...)` builds the FastAPI app.
- `apmoe.serving.routes.create_router(app)` defines route handlers.
- `apmoe.serving.middleware` contains reusable middleware and auth hooks.

`APMoEApp.serve()` uses this layer internally, but you can also embed it in an
existing ASGI app by calling `create_api(...)` directly.

---

## Endpoints

All endpoints are versioned under `/v1`. Legacy unversioned paths remain
for backward compatibility and return `Deprecation` + `Sunset` headers to
communicate migration timelines. Responses also include `X-API-Version: 1`.

### `POST /v1/predict`

Request:
- `application/json` body: an object whose keys are modality names and whose values are JSON-serialisable payloads for that modality (values are normalised to bytes before the pipeline).

Response (`200`):
- `predicted_age`
- `confidence`
- `confidence_interval` (`null` if unavailable)
- `per_expert_outputs`
- `skipped_experts`
- `metadata`

Error mapping:
- `PipelineError` -> `503`
- other `APMoEError` -> `500`
- body not valid JSON or not a JSON object -> `422`

### `GET /v1/health`

Returns expert load/readiness status from `expert_registry.health_check()`.

- all experts loaded (or no experts) -> `200`, `{"status": "healthy", ...}`
- one or more unloaded -> `503`, `{"status": "degraded", ...}`

### `GET /v1/info`

Returns `APMoEApp.get_info()` output:
- framework `version`
- loaded `experts`
- active `modalities`
- `aggregator` metadata
- `serving` configuration

---

## Middleware order

`create_api(...)` applies middleware in this order:

1. `CORSMiddleware`
2. `RequestLoggingMiddleware`
3. `RateLimitMiddleware` (only when `serving.rate_limit` is set)
4. `AuthenticationMiddleware` (when `serving.authentication_enabled` is true)
5. `AuthorizationMiddleware` (when `serving.authorization_enabled` is true)

The legacy `AuthMiddleware` still runs when `create_api(auth_plugin=...)` is
used, but it is mutually exclusive with the newer stateless authn/authz
middlewares.

This order is important so preflight/CORS handling occurs before auth/rate
checks.

---

## Request logging

For the full security reference and production checklist, see
[security.md](security.md).

`RequestLoggingMiddleware`:
- accepts a safe inbound `X-Correlation-ID` value or generates a UUID4 value
- stores it as `request.state.correlation_id`
- returns it in response header `X-Correlation-ID`
- emits structured JSON logs with method, path, query, status, duration, client

Use this header for cross-service request tracing.

Sensitive query parameters such as `token`, `api_key`, and `secret` are
redacted before logging.

---

## Security audit hooks

Security-sensitive events are emitted as `SecurityAuditEvent` objects with the
current correlation ID. The default sink writes structured JSON to
`apmoe.security.audit`; applications can also attach hooks:

```python
from apmoe.core.security import SecurityAuditEvent

def send_to_siem(event: SecurityAuditEvent) -> None:
    ...

apmoe_app.security_audit_hooks.append(send_to_siem)
api = create_api(apmoe_app, security_provider=provider)
```

Events include authentication success/failure, authorization allow/deny,
rate-limit blocks, token invalidation, Redis fallback activation, remote
endpoint allow/block, remote call success/failure, remote circuit-breaker
blocks, response-limit blocks, and model integrity pass/fail. Set
`apmoe.security.audit_enabled=false` to suppress serving audit events or
`audit_success_events=false` to log only denials/blocks for authn/authz.

---

## Rate limiting

`RateLimitMiddleware` implements a per-client-IP sliding window:
- window size: 60 seconds
- limit: `serving.rate_limit` requests/minute
- on overflow: `429` with `Retry-After: 60`

Important deployment note:
- `serving.rate_limit_store="memory"` is process-local. With multiple workers,
  effective total limit is multiplied by worker count.
- `serving.rate_limit_store="redis"` uses Redis as the shared sliding-window
  store across workers and nodes. Install it with `pip install apmoe[redis]`.
- If a Redis rate-limit operation fails after startup, APMoE emits
  `redis_rate_limit_fallback` and uses an in-memory process-local sliding
  window for that worker. This preserves API availability but the limit is no
  longer globally coordinated until Redis recovers.
- For strict global limits across heterogeneous services, Redis or an API
  gateway in front of APMoE is recommended.

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

## Stateless authentication and authorization

The new serving security layer is enabled by default:

```json
{
  "serving": {
    "authentication_enabled": true,
    "authorization_enabled": true
  }
}
```

When authentication is enabled, `create_api(...)` fails closed unless a
`StatelessAuthProvider` is supplied. For local demos only, disable both layers:

```json
{
  "serving": {
    "authentication_enabled": false,
    "authorization_enabled": false
  }
}
```

Default route scopes:

| Route | Required scope |
|---|---|
| `POST /v1/predict`, `POST /predict` | `predict` |
| `GET /v1/info`, `GET /info` | `info:read` |
| `GET /v1/health`, `GET /health` | public |

JWT Bearer setup:

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
- `sub`: subject/principal id
- `jti`: stable token id used for invalidation
- `exp`: expiry timestamp
- `scope` or `scopes`: permission strings

Invalidation uses `TokenInvalidationStore`. The default
`InMemoryTokenInvalidationStore` is process-local and appropriate only for
single-process/local use. In horizontally scaled deployments, select Redis in
serving config:

```json
{
  "serving": {
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "redis://localhost:6379/0"
  }
}
```

Install Redis support with `pip install apmoe[redis]`. You can also inject any
external implementation through `create_api(..., invalidation_store=...)`:

```python
class RedisTokenInvalidationStore(TokenInvalidationStore):
    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def is_invalid(self, token_id: str) -> bool:
        return self._redis.exists(f"apmoe:revoked:{token_id}") == 1

    def invalidate(self, token_id: str, expires_at: datetime) -> None:
        ttl = max(0, int((expires_at - datetime.now(UTC)).total_seconds()))
        self._redis.setex(f"apmoe:revoked:{token_id}", ttl, "1")
```

Pass the same store to `JWTBearerAuthProvider` when constructing the provider.
Redis is optional and not imported unless a Redis store is selected. If a Redis
token invalidation operation fails after startup, APMoE emits
`redis_token_invalidation_fallback` and uses an in-memory process-local
fallback store. Invalidations made while Redis is unavailable are local to that
worker and are not replayed into Redis.

---

## Remote expert security

Remote expert endpoints are governed by `apmoe.security`:
- non-production defaults missing `remote_endpoint_allowlist` to `["*"]`
- production remote experts require an explicit allowlist without `"*"`
- endpoint and manifest hosts are matched case-insensitively against exact
  hosts or wildcard suffixes such as `"*.example.com"`
- HTTPS is enforced by default
- localhost, loopback, private, link-local, reserved, multicast, and metadata
  IP hosts are blocked unless `remote_allow_private_networks=true`

Remote responses are capped by `remote_response_max_bytes` before JSON parsing.
Set `experts[].endpoint_response_max_bytes` to override the global cap for a
specific expert. Obvious non-JSON content types are rejected.

Remote inference calls also support transient retries and per-expert circuit
breakers. Configure `apmoe.remote_retry`, `apmoe.remote_circuit_breaker`, and
`apmoe.expert_failure_policy` in the top-level config; see
[configuration.md](configuration.md) for the exact fields and defaults.

Local model artifacts can be pinned with `experts[].integrity.sha256`. Remote
model integrity uses an RSA-PSS-SHA256 signed manifest verified with a pinned
public key; a plain hash served by the remote model runtime is intentionally not
treated as sufficient. See `docs/dev/configuration.md` for config examples.

---

## Legacy authentication plugin

Authentication is opt-in via `create_api(auth_plugin=...)`.

Implement `AuthPlugin`:
- `authenticate(request) -> bool`
- optionally override `unauthenticated_response()`

Default excluded paths (no auth required):
- `/health`
- `/info`

You can override excluded paths with `auth_exclude_paths`.

Minimal API-key example:

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

## Config knobs used by serving

From `apmoe.serving` and `apmoe.core.config.ServingConfig`:

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

Environment overrides are documented in `docs/dev/configuration.md`.
