# Security Reference

This page documents the framework-level security controls implemented by APMoE
and the deployment responsibilities that remain with an application owner. APMoE
is a framework, so these controls are intentionally generic: they protect the
serving layer, remote expert calls, model artifact integrity, and auditability
without assuming a specific identity provider, API gateway, Redis topology, or
model release process.

---

## Threat Model

APMoE focuses on these framework-level risks:

- unauthenticated or unauthorized access to expensive inference endpoints
- replay or continued use of JWTs after logout/revocation
- inconsistent auth, revocation, or rate-limit behavior in multi-worker systems
- outbound request tampering or SSRF-like routing to unexpected remote experts
- remote responses that are too large or not JSON
- local model artifact replacement on disk
- remote model compromise where a model service could also lie about its hash
- missing request traceability during incidents
- accidental logging of credentials, tokens, API keys, cookies, or secret URLs

APMoE does not replace infrastructure controls such as TLS termination, WAF/API
gateway policy, cloud IAM, host hardening, secret managers, CI signing systems,
or centralized log retention. It provides application/framework hooks that work
well with those controls.

---

## Configuration Summary

Security-related config is split across `apmoe.environment`, `apmoe.security`,
`apmoe.serving`, and `experts[].integrity`:

```json
{
  "apmoe": {
    "environment": "production",
    "security": {
      "remote_endpoint_allowlist": ["models.example.com", "*.trusted.ai"],
      "remote_enforce_https": true,
      "remote_allow_private_networks": false,
      "remote_response_max_bytes": 1048576,
      "audit_enabled": true,
      "audit_success_events": true
    },
    "serving": {
      "authentication_enabled": true,
      "authorization_enabled": true,
      "rate_limit": 120,
      "rate_limit_store": "redis",
      "rate_limit_redis_url": "redis://localhost:6379/0",
      "token_invalidation_store": "redis",
      "token_invalidation_redis_url": "redis://localhost:6379/0"
    }
  }
}
```

`APMOE_ENV` overrides `apmoe.environment`. Serving env overrides are documented
in [configuration.md](configuration.md).

---

## Authentication

APMoE uses two consecutive request middlewares when stateless security is
enabled:

1. `AuthenticationMiddleware`
2. `AuthorizationMiddleware`

Both are enabled by default through config:

```json
{
  "serving": {
    "authentication_enabled": true,
    "authorization_enabled": true
  }
}
```

When authentication is enabled and no `security_provider` is passed to
`create_api(...)`, the app fails closed at creation time. Local demos can
explicitly disable both middlewares:

```json
{
  "serving": {
    "authentication_enabled": false,
    "authorization_enabled": false
  }
}
```

Environment overrides:

- `APMOE_SERVING_AUTHENTICATION_ENABLED`
- `APMOE_SERVING_AUTHORIZATION_ENABLED`

Boolean values accept `true/false`, `1/0`, `yes/no`, and `on/off`. Invalid
boolean env values raise `ConfigurationError`.

---

## JWT Bearer Provider

`JWTBearerAuthProvider` is the built-in stateless provider. It is imported only
when used and requires `pip install apmoe[security]`.

It validates:

- `Authorization: Bearer <token>`
- JWT signature
- `exp`
- optional `issuer`
- optional `audience`
- required subject claim, default `sub`
- required stable token id claim, default `jti`
- scopes from `scope` or `scopes` by default
- token invalidation by `jti`

Malformed, expired, unsigned, wrong-issuer, wrong-audience, missing-`jti`, and
invalidated tokens are rejected. Authentication failures return `401` with
`WWW-Authenticate: Bearer`.

Example:

```python
from apmoe.core.app import APMoEApp
from apmoe.serving.app_factory import create_api
from apmoe.serving.middleware import JWTBearerAuthProvider

apmoe_app = APMoEApp.from_config("config.json")
provider = JWTBearerAuthProvider(
    signing_key="replace-with-secret-or-public-key",
    algorithms=["RS256"],
    issuer="https://issuer.example.com",
    audience="apmoe-api",
)
api = create_api(apmoe_app, security_provider=provider)
```

Prefer asymmetric JWT algorithms such as `RS256` or `ES256` when the issuer is a
separate service. `HS256` is supported but requires careful symmetric secret
distribution.

---

## Authorization

`AuthorizationMiddleware` reads `request.state.auth_context` and enforces an
`AuthorizationPolicy`. The default `ScopeAuthorizationPolicy` uses these route
scopes:

| Route | Required scope |
|---|---|
| `POST /v1/predict`, `POST /predict` | `predict` |
| `GET /v1/info`, `GET /info` | `info:read` |
| `GET /v1/health`, `GET /health` | public |

Authorization failures return `403`. Health remains public by default so
orchestrator probes do not need user credentials.

Custom policies can be injected through:

```python
api = create_api(
    apmoe_app,
    security_provider=provider,
    authorization_policy=my_policy,
)
```

Legacy `AuthPlugin` remains supported for backwards compatibility but is
mutually exclusive with the stateless security provider/policy path.

---

## Token Invalidation

JWT invalidation is based on the stable token id claim, default `jti`.

Available stores:

- `InMemoryTokenInvalidationStore`: default, process-local, TTL-pruned
- `RedisTokenInvalidationStore`: shared across workers/nodes
- custom `TokenInvalidationStore`: inject through `create_api(...)`

Config-driven Redis selection:

```json
{
  "serving": {
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "redis://localhost:6379/0",
    "token_invalidation_key_prefix": "apmoe:jwt:invalid:"
  }
}
```

Install Redis support with `pip install apmoe[redis]`.

Important scaling rule: the in-memory store does not share revocation state
across processes or machines. Horizontal deployments that require logout,
emergency revocation, or strict global invalidation must use Redis or another
shared store.

Redis operation failures fall back to the same process-local invalidation store
used for local deployments. APMoE emits a `redis_token_invalidation_fallback`
audit event when this happens. This keeps authentication available, but
invalidations written during the outage are local to that worker and are not
replayed into Redis.

---

## Rate Limiting

`RateLimitMiddleware` implements a per-client-IP sliding window when
`serving.rate_limit` is configured. It runs before authentication and
authorization so abusive unauthenticated requests are throttled early.

Available stores:

- `InMemoryRateLimitStore`: default, process-local
- `RedisRateLimitStore`: shared across workers/nodes
- custom `RateLimitStore`: inject through `create_api(...)`

Config-driven Redis selection:

```json
{
  "serving": {
    "rate_limit": 120,
    "rate_limit_store": "redis",
    "rate_limit_redis_url": "redis://localhost:6379/0",
    "rate_limit_key_prefix": "apmoe:rate:"
  }
}
```

Important scaling rule: process-local rate limiting multiplies the effective
limit by the number of workers. Use Redis or an API gateway for strict global
limits.

Redis operation failures fall back to process-local rate limiting for the
current worker and emit a `redis_rate_limit_fallback` audit event. During the
fallback window, rate limits are no longer globally coordinated across workers
or nodes.

---

## Remote Endpoint Policy

Remote experts are outbound HTTP calls, so APMoE applies a remote endpoint
policy when a `security_config` is present during bootstrap.

Controls:

- `remote_endpoint_allowlist`: hostname allowlist for remote expert endpoints
  and signed manifest URLs
- exact host matching, case-insensitive
- wildcard suffix matching such as `*.example.com`
- `remote_enforce_https=true` by default
- `remote_allow_private_networks=false` by default
- localhost, loopback, private, link-local, reserved, multicast, and metadata IP
  hosts are denied unless private networks are explicitly allowed

Production behavior:

- non-production missing allowlist behaves as `["*"]`
- production with any remote expert requires an explicit allowlist
- production missing, empty, or wildcard-only allowlists raise
  `ConfigurationError` during config load

Example:

```json
{
  "environment": "production",
  "security": {
    "remote_endpoint_allowlist": ["models.example.com", "*.trusted.ai"],
    "remote_enforce_https": true,
    "remote_allow_private_networks": false
  }
}
```

The allowlist applies to outbound remote expert endpoints, not browser CORS.
CORS remains configured separately with `serving.cors_origins`.

---

## TLS Guidance

TLS is still required. APMoE's application-level controls are not a replacement
for HTTPS. TLS protects transport confidentiality and integrity between network
peers; APMoE adds framework checks for route authorization, token invalidation,
remote endpoint selection, response limits, model artifact integrity, and audit
tracing.

Use both in production:

- terminate TLS at a trusted load balancer, ingress, or ASGI server
- keep `remote_enforce_https=true` for public remote experts
- use private network exceptions only for deliberate local/offline deployments
- pin remote model trust to signed manifests, not to a hash served by the model
  runtime itself

---

## Remote Response Limits

Remote responses are bounded before JSON parsing:

- global cap: `apmoe.security.remote_response_max_bytes`, default `1048576`
- per-expert override: `experts[].endpoint_response_max_bytes`
- obvious non-JSON `Content-Type` values are rejected when present
- response snippets in errors/logs are redacted

Example:

```json
{
  "security": {
    "remote_response_max_bytes": 1048576
  },
  "experts": [
    {
      "name": "remote_age",
      "class": "apmoe.experts.remote.RemoteExpert",
      "endpoint": "https://models.example.com/predict",
      "endpoint_response_max_bytes": 262144,
      "modalities": ["keystroke"]
    }
  ]
}
```

---

## Remote Expert Resilience

Remote expert inference calls can be retried, protected by a per-expert circuit
breaker, and combined with configurable pipeline fallback:

- `apmoe.remote_retry` retries transient timeouts, network errors, and HTTP
  `429`, `502`, `503`, and `504` with exponential backoff and optional jitter.
- `apmoe.remote_circuit_breaker` opens after consecutive remote failures,
  short-circuits calls while open, and allows a half-open trial after cooldown.
- `apmoe.expert_failure_policy="skip_failed"` lets prediction aggregate the
  remaining successful runnable experts and records failed runnable experts in
  `Prediction.metadata["failed_experts"]`. The default is `"fail_fast"` for
  backwards compatibility.

Retries do not apply to local ONNX/Keras experts, response mapping errors,
template errors, response-size violations, invalid JSON, or non-transient HTTP
statuses such as `400`, `401`, `403`, and `404`.

---

## Local Model Integrity

Local model artifacts can be hash-pinned with SHA-256:

```json
{
  "name": "face_age_expert",
  "class": "apmoe.experts.builtin.CNNAgeExpert",
  "weights": "./weights/face_age_expert.keras",
  "modalities": ["visual"],
  "integrity": {
    "sha256": "64-character-hex-digest"
  }
}
```

The framework stream-hashes the configured `weights` file before
`load_weights()`. A mismatch raises `ExpertError` and emits a
`local_artifact_integrity` audit event.

If `integrity.sha256` is omitted, local model behavior is unchanged.

---

## Remote Model Integrity

A plain hash returned by a remote model service is not a strong integrity
guarantee because a compromised model service can also return a compromised
hash. APMoE therefore supports RSA-PSS-SHA256 signed manifests verified with a
pinned public key.

Remote integrity config:

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

Manifest fields:

- `expert_name`
- `model_id`
- `model_version`
- `endpoint_origin`
- `model_digest` or `artifact_digest`
- `issued_at`
- `expires_at`
- `signature`

Verification requires:

- manifest URL passes the same allowlist/HTTPS/private-network policy
- response size is under `remote_response_max_bytes`
- RSA-PSS SHA-256 signature verifies under the pinned public key
- signature covers canonical JSON excluding the `signature` field
- manifest is not expired
- `expert_name` matches config
- `endpoint_origin` matches the configured remote endpoint origin

Failure behavior:

- `manifest_required=true`: failure blocks startup
- production: any configured manifest failure blocks startup
- non-production with `manifest_required=false`: failure is audited but does not
  block startup

Keep the private signing key outside the model serving runtime. Good places for
the private key are release CI, a KMS-backed signing job, or another dedicated
artifact signing process. The remote model runtime should only have the signed
manifest and artifact metadata, not the signing key.

---

## Correlation IDs

APMoE uses a global `contextvars` correlation context.

HTTP behavior:

- accepts inbound `X-Correlation-ID` only if it contains safe characters and is
  at most 128 characters
- otherwise generates a UUID4
- stores it at `request.state.correlation_id`
- returns it in `X-Correlation-ID`
- includes it in request logs and security audit events

Direct/CLI predictions generate a correlation ID when none exists.

Safe correlation IDs may contain letters, digits, `_`, `.`, `:`, and `-`.

---

## Audit Events

APMoE emits structured security audit events through:

- `SecurityAuditEvent`
- `SecurityAuditSink`
- `LoggingSecurityAuditSink`, logger name `apmoe.security.audit`
- `APMoEApp.security_audit_hooks`

Custom hook example:

```python
from apmoe.core.security import SecurityAuditEvent

def send_to_siem(event: SecurityAuditEvent) -> None:
    payload = event.to_dict()
    ...

apmoe_app.security_audit_hooks.append(send_to_siem)
```

Implemented event types include:

| Event type | Typical outcomes |
|---|---|
| `authentication` | `success`, `failure` |
| `authorization` | `success`, `failure` |
| `rate_limit` | `failure` |
| `remote_endpoint_policy` | `allow`, `block` |
| `remote_call` | `success`, `failure` |
| `remote_response_limit` | `failure` |
| `remote_circuit_breaker` | `failure` |
| `local_artifact_integrity` | `success`, `failure` |
| `remote_manifest_integrity` | `success`, `failure` |
| `token_invalidation` | `success` |
| `redis_rate_limit_fallback` | `failure` |
| `redis_token_invalidation_fallback` | `failure` |

Config:

```json
{
  "security": {
    "audit_enabled": true,
    "audit_success_events": true
  }
}
```

`audit_enabled=false` suppresses serving audit events. `audit_success_events`
controls successful authentication and authorization events; denials and blocks
are still emitted while audit is enabled.

---

## Redaction

APMoE redacts common secret-bearing values in `/info`, security audit event
payloads, remote error contexts, and request logging.

Redacted data includes:

- URL credentials
- sensitive query params such as `token`, `api_key`, `secret`, and `key`
- headers such as `authorization`, `api-key`, `cookie`, `token`, and `secret`
- Redis URLs or other secret-looking config values when passed through
  structured redaction

Redaction is a defense-in-depth control. Avoid putting secrets in URLs when
headers or secret managers are available.

---

## Production Checklist

Before serving production traffic:

- set `APMOE_ENV=production` or `apmoe.environment="production"`
- configure explicit `remote_endpoint_allowlist` for every remote expert host
- keep `remote_enforce_https=true`
- keep `remote_allow_private_networks=false` unless a private deployment needs it
- configure `authentication_enabled=true` and pass a `StatelessAuthProvider`
- configure `authorization_enabled=true`
- issue JWTs with `sub`, `jti`, `exp`, and scopes
- use Redis or another shared store for token invalidation in multi-worker or
  multi-node deployments
- use Redis or an API gateway for strict global rate limiting
- monitor `redis_rate_limit_fallback` and `redis_token_invalidation_fallback`
  audit events because fallback state is local to each worker
- tune `remote_retry`, `remote_circuit_breaker`, and `expert_failure_policy`
  deliberately for remote experts that can tolerate degraded predictions
- pin local model artifacts with `integrity.sha256` where artifacts are managed
  outside the package
- require signed remote manifests for production remote experts
- keep remote manifest private signing keys outside the model runtime
- set a bounded `remote_response_max_bytes`
- ship audit logs to centralized storage/SIEM with retention appropriate for
  your environment
- terminate TLS and forward only trusted traffic to APMoE
- configure CORS to explicit browser origins instead of `["*"]`

---

## What Remains Application-Owned

APMoE intentionally leaves these decisions to the application or platform:

- user identity lifecycle and account management
- JWT issuing, rotation, and key management
- Redis deployment, clustering, authentication, and network ACLs
- TLS certificates and ingress/load-balancer configuration
- API gateway/WAF policy
- SIEM/log retention policy
- model release approval and private signing-key custody
- privacy policy, consent flows, data retention, and regulatory compliance
- custom authorization policies beyond the default route scopes
