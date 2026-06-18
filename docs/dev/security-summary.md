# Security Summary

This document summarizes the security controls currently implemented in APMoE
and links each area to the code that enforces it. For the operational runbook
and production checklist, see [security.md](security.md).

---

## Code Map

| Area | Primary implementation |
|---|---|
| Security config models | [`src/apmoe/core/config.py`](../../src/apmoe/core/config.py) |
| Shared security helpers, redaction, audit events, URL policy, integrity checks | [`src/apmoe/core/security.py`](../../src/apmoe/core/security.py) |
| App bootstrap and config wiring | [`src/apmoe/core/app.py`](../../src/apmoe/core/app.py) |
| Inference fallback policy | [`src/apmoe/core/pipeline.py`](../../src/apmoe/core/pipeline.py) |
| Expert registry, local hash checks, remote manifest verification | [`src/apmoe/experts/registry.py`](../../src/apmoe/experts/registry.py) |
| Remote expert HTTP safety, retries, response limits, circuit breaker | [`src/apmoe/experts/remote.py`](../../src/apmoe/experts/remote.py) |
| Serving app assembly and middleware order | [`src/apmoe/serving/app_factory.py`](../../src/apmoe/serving/app_factory.py) |
| Authn/authz, rate limiting, Redis stores, request logging | [`src/apmoe/serving/middleware.py`](../../src/apmoe/serving/middleware.py) |
| HTTP routes and error mapping | [`src/apmoe/serving/routes.py`](../../src/apmoe/serving/routes.py) |
| Security unit tests | [`tests/unit/test_security.py`](../../tests/unit/test_security.py), [`tests/unit/test_serving.py`](../../tests/unit/test_serving.py), [`tests/unit/test_remote_expert.py`](../../tests/unit/test_remote_expert.py) |
| Resilience e2e smoke script | [`scripts/e2e_resilience.py`](../../scripts/e2e_resilience.py) |

---

## Configuration Guardrails

APMoE validates security-sensitive settings at config load time in
[`src/apmoe/core/config.py`](../../src/apmoe/core/config.py):

- `SecurityConfig` controls remote endpoint allowlists, HTTPS enforcement,
  private-network blocking, remote response size limits, and audit switches.
- Production configs with remote experts must provide a concrete
  `remote_endpoint_allowlist`; missing, empty, or wildcard-only allowlists are
  rejected.
- Redis-backed serving stores require Redis URLs when selected.
- Remote retry and circuit-breaker settings validate positive numeric fields.
- `expert_failure_policy` is restricted to `"fail_fast"` or `"skip_failed"`.

Environment variable overrides are limited to known serving fields and are cast
with explicit type handling in
[`src/apmoe/core/config.py`](../../src/apmoe/core/config.py).

---

## Authentication and Authorization

The serving layer implements stateless authentication and route-scope
authorization in [`src/apmoe/serving/middleware.py`](../../src/apmoe/serving/middleware.py):

- `JWTBearerAuthProvider` validates Bearer tokens, signatures, `exp`, optional
  issuer/audience, required subject, required token id, scopes, and token
  invalidation.
- `AuthenticationMiddleware` fails requests closed with `401` when credentials
  are missing or invalid.
- `AuthorizationMiddleware` enforces route scopes and returns `403` on
  insufficient permissions.
- `ScopeAuthorizationPolicy` maps prediction and info routes to required
  scopes while keeping health checks public for orchestrators.
- `create_api()` in [`src/apmoe/serving/app_factory.py`](../../src/apmoe/serving/app_factory.py)
  fails closed if authentication is enabled but no security provider is passed.

Legacy `AuthPlugin` support remains available, but `create_api()` prevents
mixing legacy auth with the stateless authn/authz path.

---

## Token Invalidation

Token invalidation is implemented in
[`src/apmoe/serving/middleware.py`](../../src/apmoe/serving/middleware.py):

- `InMemoryTokenInvalidationStore` stores invalidated JWT ids with TTL pruning
  for local or single-worker deployments.
- `RedisTokenInvalidationStore` shares invalidation state across workers and
  nodes.
- If Redis operations fail after startup, the Redis store emits
  `redis_token_invalidation_fallback` and falls back to a process-local
  in-memory store so authentication remains available.

Fallback invalidations are local to the current worker and are not replayed into
Redis. Horizontal deployments that require strict revocation still need healthy
Redis or another shared store.

---

## Rate Limiting

Request throttling is implemented in
[`src/apmoe/serving/middleware.py`](../../src/apmoe/serving/middleware.py):

- `RateLimitMiddleware` enforces a per-client-IP sliding window before
  authentication and authorization.
- `InMemoryRateLimitStore` is process-local.
- `RedisRateLimitStore` uses Redis sorted sets for shared limits across
  workers/nodes.
- If Redis rate-limit operations fail after startup, the Redis store emits
  `redis_rate_limit_fallback` and uses a process-local in-memory sliding
  window.

During Redis fallback, availability is preserved but limits are no longer
globally coordinated.

---

## Request Logging and Correlation IDs

Request tracing is implemented in
[`src/apmoe/serving/middleware.py`](../../src/apmoe/serving/middleware.py) and
[`src/apmoe/core/security.py`](../../src/apmoe/core/security.py):

- `RequestLoggingMiddleware` accepts a safe inbound `X-Correlation-ID` or
  generates a UUID4 value.
- Correlation IDs are stored in a `contextvars` context and returned in the
  response header.
- Request logs include method, path, redacted query string, status code,
  duration, and client host.
- Direct app predictions also call `ensure_correlation_id()` in
  [`src/apmoe/core/app.py`](../../src/apmoe/core/app.py).

---

## Audit Events

Structured audit logging is implemented in
[`src/apmoe/core/security.py`](../../src/apmoe/core/security.py) and emitted
from serving, remote expert, and registry paths:

- `SecurityAuditEvent` stores event type, outcome, correlation id, subject,
  request path, client IP, expert name, endpoint host, reason, and metadata.
- `LoggingSecurityAuditSink` writes redacted JSON to `apmoe.security.audit`.
- `APMoEApp.security_audit_hooks` can attach custom sinks through
  [`src/apmoe/core/app.py`](../../src/apmoe/core/app.py) and
  [`src/apmoe/serving/app_factory.py`](../../src/apmoe/serving/app_factory.py).

Implemented event types include authentication, authorization, rate limiting,
token invalidation, Redis fallback, remote endpoint policy, remote calls, remote
circuit breaker blocks, remote response limits, and local/remote model
integrity checks.

---

## Redaction

Secret redaction is centralized in
[`src/apmoe/core/security.py`](../../src/apmoe/core/security.py):

- URL credentials are removed.
- Sensitive query parameters such as `token`, `api_key`, `secret`, and `key`
  are replaced with `***`.
- Sensitive headers such as authorization, API keys, cookies, tokens, and
  secrets are redacted.
- Nested dict/list values are recursively redacted before audit emission and
  `/info` output.

Remote expert errors and diagnostics call `redact_url()` and `redact_value()` in
[`src/apmoe/experts/remote.py`](../../src/apmoe/experts/remote.py). App info
output redacts serving and security config via
[`src/apmoe/core/app.py`](../../src/apmoe/core/app.py).

---

## Remote Endpoint Policy

Outbound remote expert endpoints are guarded by
[`src/apmoe/core/security.py`](../../src/apmoe/core/security.py),
[`src/apmoe/experts/remote.py`](../../src/apmoe/experts/remote.py), and
[`src/apmoe/experts/registry.py`](../../src/apmoe/experts/registry.py):

- Hostnames must match exact or wildcard allowlist entries.
- HTTPS is enforced by default.
- Localhost, loopback, private, link-local, reserved, multicast, and metadata
  IP hosts are blocked unless private networks are explicitly allowed.
- Production remote experts require an explicit non-wildcard allowlist.
- Manifest URLs are checked with the same remote URL policy.

Allow/block decisions emit `remote_endpoint_policy` audit events.

---

## Remote Response Safety

Remote response controls live in
[`src/apmoe/experts/remote.py`](../../src/apmoe/experts/remote.py):

- Responses are capped by `apmoe.security.remote_response_max_bytes` before JSON
  parsing.
- `experts[].endpoint_response_max_bytes` can set a stricter per-expert cap.
- Obvious non-JSON content types are rejected.
- Oversized responses emit `remote_response_limit` audit events.
- Response snippets included in errors are redacted.

Invalid JSON, oversized responses, non-JSON responses, template errors, and
response mapping errors are treated as non-transient failures and are not
retried.

---

## Remote Retries and Circuit Breakers

Remote inference resilience is implemented in
[`src/apmoe/experts/remote.py`](../../src/apmoe/experts/remote.py), configured
in [`src/apmoe/core/config.py`](../../src/apmoe/core/config.py), and wired by
[`src/apmoe/core/app.py`](../../src/apmoe/core/app.py) through
[`src/apmoe/experts/registry.py`](../../src/apmoe/experts/registry.py):

- Remote calls retry transient timeouts, network errors, and HTTP `429`, `502`,
  `503`, and `504`.
- Retry timing uses exponential backoff with optional jitter.
- Each `RemoteExpert` instance owns an in-memory circuit breaker.
- Closed circuits run normally, open circuits fail immediately, and half-open
  circuits allow one trial request after cooldown.
- Open-circuit short-circuits emit `remote_circuit_breaker` audit events.

Retries apply only to remote inference calls, not local ONNX/Keras prediction.

---

## Expert Failure Policy

Pipeline-level expert fallback is implemented in
[`src/apmoe/core/pipeline.py`](../../src/apmoe/core/pipeline.py):

- Default `expert_failure_policy="fail_fast"` preserves historical behavior.
- `expert_failure_policy="skip_failed"` records failed runnable experts in
  `Prediction.metadata["failed_experts"]` and aggregates successful expert
  outputs.
- `skipped_experts` remains reserved for experts skipped because their required
  modalities were unavailable.
- If every runnable expert fails, the pipeline raises `PipelineError`.

This gives operators an explicit choice between strict fail-fast behavior and
degraded predictions.

---

## Model Artifact Integrity

Local and remote model integrity controls are implemented in
[`src/apmoe/core/security.py`](../../src/apmoe/core/security.py) and wired by
[`src/apmoe/experts/registry.py`](../../src/apmoe/experts/registry.py):

- Local expert weights can pin a SHA-256 digest through
  `experts[].integrity.sha256`.
- Local artifacts are stream-hashed before `load_weights()`.
- Mismatches raise `ExpertError` and emit `local_artifact_integrity` events.
- Remote experts can require signed manifests using
  `experts[].integrity.manifest_url`, `manifest_public_key`, and
  `manifest_required`.
- Remote manifests are verified with RSA-PSS-SHA256 over canonical JSON.
- Required manifest fields include expert name, model id/version, endpoint
  origin, digest, issue/expiry times, and signature.
- Manifest failures block startup when required or in production.

The remote model runtime is not trusted to self-report integrity; trust is
anchored to a pinned public key.

---

## Serving Surface Controls

The FastAPI app is assembled in
[`src/apmoe/serving/app_factory.py`](../../src/apmoe/serving/app_factory.py) and
routes are defined in [`src/apmoe/serving/routes.py`](../../src/apmoe/serving/routes.py):

- Middleware order puts CORS first, then request logging, rate limiting,
  authentication, and authorization.
- `create_api()` rejects inconsistent security configuration, such as
  authorization enabled while authentication is disabled.
- Prediction route errors map `PipelineError` to `503`, other APMoE errors to
  `500`, and invalid request bodies to `422`.
- Health remains public by default.
- Legacy unversioned routes return deprecation headers while versioned `/v1`
  routes are the primary API surface.

---

## Test Coverage and E2E Checks

Security and resilience behavior is covered by:

- Security helper tests in [`tests/unit/test_security.py`](../../tests/unit/test_security.py).
- Serving auth, rate-limit, Redis fallback, and JWT tests in
  [`tests/unit/test_serving.py`](../../tests/unit/test_serving.py).
- Remote expert security, retry, response limit, and circuit breaker tests in
  [`tests/unit/test_remote_expert.py`](../../tests/unit/test_remote_expert.py).
- Config validation tests in [`tests/unit/test_config.py`](../../tests/unit/test_config.py).
- Pipeline fallback tests in [`tests/unit/test_pipeline.py`](../../tests/unit/test_pipeline.py).
- App bootstrap wiring tests in [`tests/integration/test_app.py`](../../tests/integration/test_app.py).
- Standalone resilience smoke coverage in
  [`scripts/e2e_resilience.py`](../../scripts/e2e_resilience.py).

---

## Application-Owned Responsibilities

APMoE implements framework-level controls, but production deployments still own:

- JWT issuance, key rotation, and identity lifecycle.
- Redis availability, authentication, clustering, backups, and network ACLs.
- TLS termination, ingress policy, API gateway/WAF rules, and infrastructure
  logging.
- Remote model signing-key custody and release approval.
- Centralized audit retention and alerting.
- Privacy policy, user consent, and regulatory compliance decisions.
