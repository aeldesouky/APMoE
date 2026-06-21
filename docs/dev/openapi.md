# OpenAPI Reference

APMoE serves its OpenAPI schema from the running FastAPI application. The schema
is generated from the route handlers and the Pydantic models in
`apmoe.serving.openapi_schemas`.

## Live Documentation URLs

Start the server:

```bash
apmoe serve --config config.json --host 127.0.0.1 --port 8000
```

Then open:

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8000/docs` | Swagger UI with interactive request examples. |
| `http://127.0.0.1:8000/redoc` | ReDoc view of the same schema. |
| `http://127.0.0.1:8000/openapi.json` | Raw machine-readable OpenAPI JSON. |

## Documented API Surface

| Method | Path | Summary |
|---|---|---|
| `POST` | `/v1/predict` | Run age prediction from a multimodal JSON body. |
| `GET` | `/v1/health` | Return expert readiness/liveness state. |
| `GET` | `/v1/info` | Return runtime metadata and redacted configuration. |

Legacy `/predict`, `/health`, and `/info` routes are still mounted for
compatibility and are marked deprecated in the generated schema.

## Request Examples

Swagger UI includes examples for:

- keystroke triples: `[[key1, key2, timing_ms], ...]`
- keystroke IKDD text
- precomputed keystroke feature dictionaries
- image plus keystroke multimodal payloads

For JSON clients, image values should be base64 image strings or local file
paths that are available to the server. Keystroke values can be JSON lists,
JSON dictionaries, or IKDD text strings.

## Response Models

`POST /v1/predict` returns:

- `predicted_age`
- `confidence`
- `confidence_interval`
- `per_expert_outputs`
- `skipped_experts`
- `metadata`

`GET /v1/health` returns `status` and per-expert load state.

`GET /v1/info` returns package version, experts, modalities, aggregator,
serving settings, environment, redacted security settings, confidence threshold,
and remote resilience settings.

## Headers

The generated OpenAPI schema documents response headers:

| Header | Applies to | Meaning |
|---|---|---|
| `X-Correlation-ID` | All API responses | Request trace ID for logs and support. |
| `X-API-Version` | Versioned and legacy API routes | Current API version, currently `1`. |
| `Deprecation` | Legacy routes | Date the unversioned route was deprecated. |
| `Sunset` | Legacy routes | Planned end of the compatibility window. |
| `Link` | Legacy routes | Documentation link for migration. |

## Maintenance Rule

When a route shape, request example, response model, or documented header
changes, update both:

- `src/apmoe/serving/openapi_schemas.py`
- [Serving layer](serving.md)

