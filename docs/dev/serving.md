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

### `POST /predict`

Request:
- `multipart/form-data`
- field name = modality name (`visual`, `audio`, etc.)
- file field values are read as bytes
- text fields are forwarded as raw strings

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
- malformed multipart parse -> `422`

### `GET /health`

Returns expert load/readiness status from `expert_registry.health_check()`.

- all experts loaded (or no experts) -> `200`, `{"status": "healthy", ...}`
- one or more unloaded -> `503`, `{"status": "degraded", ...}`

### `GET /info`

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
4. `AuthMiddleware` (only when `auth_plugin` is provided)

This order is important so preflight/CORS handling occurs before auth/rate
checks.

---

## Request logging

`RequestLoggingMiddleware`:
- generates a UUID4 correlation ID per request
- stores it as `request.state.correlation_id`
- returns it in response header `X-Correlation-ID`
- emits structured JSON logs with method, path, query, status, duration, client

Use this header for cross-service request tracing.

---

## Rate limiting

`RateLimitMiddleware` implements a per-client-IP sliding window:
- window size: 60 seconds
- limit: `serving.rate_limit` requests/minute
- on overflow: `429` with `Retry-After: 60`

Important deployment note:
- This limiter is in-memory and process-local.
- With multiple workers, effective total limit is multiplied by worker count.
- For strict global limits, place a shared limiter (for example Redis) in front.

---

## Authentication plugin

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

Environment overrides are documented in `docs/dev/configuration.md`.
