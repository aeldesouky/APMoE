# Deployment, SLA, and Fallback Guidance

This document describes how to deploy APMoE on serverless or dedicated
infrastructure, how fallback works at the framework and infrastructure levels,
and how vendors can use measured load-test results to define practical service
level agreements.

## Architecture Baseline

APMoE is an inference-only Python framework with a FastAPI serving layer. The
runtime is built around a stateless request path:

1. `APMoEApp.from_config()` loads the JSON config and model artifacts at
   startup.
2. `POST /v1/predict` receives JSON modality payloads.
3. The pipeline processes available modalities, runs matching experts, and
   aggregates successful outputs into a `Prediction`.
4. The response includes the final age estimate, confidence, per-expert output,
   skipped experts, failed modalities, failed experts, and latency metadata.

The serving workers are stateless for request processing. They keep read-only
model weights in memory after bootstrap, but they do not store user sessions or
prediction history between requests.

Stateful infrastructure still exists around the stateless workers:

- Redis for shared rate limiting and JWT token invalidation when enabled;
- model artifact storage or container images that carry weight files;
- JWT issuer and key management;
- remote expert providers;
- centralized logs, audit retention, and metrics;
- load balancers, ingress, and DNS.

## APMoE Deployment Profiles

The deployment decision should start from the actual APMoE configuration being
served. The framework can look like a tiny CPU service for keystroke-only
inference, or like a remote-provider orchestration service when using
`configs/llm_remote.json`.

| Profile | Reference config | Active modalities | Active experts | Best infrastructure fit | Notes |
|---|---|---|---|---|---|
| Keystroke-only CPU API | `configs/keystroke.json` | `keystroke` | `KeystrokeAgeExpert` using `weights/keystroke_age_expert.onnx` | Lambda-style or small dedicated CPU service | Lowest latency and smallest payload. Good first SLA baseline because it has measured local load-test results. |
| Local multimodal API | `configs/multimodal.json` | `keystroke`, `image` | ONNX keystroke expert plus Keras face expert | Dedicated containers or VMs | Better prediction coverage, but image decoding and Keras inference dominate CPU/memory. Validate on production-like hardware before committing latency. |
| Remote vision LLM API | `configs/llm_remote.json` | `image` | `LMStudioExpert` over HTTP | Dedicated orchestrator or serverless with provisioned concurrency | Local APMoE work is mostly preprocessing and outbound HTTP. SLA is bounded by the remote model server. |
| Mixed local + remote vendor API | custom config combining local experts and `RemoteExpert` | usually `keystroke` plus `image` | local ONNX/Keras plus hosted model API | Dedicated service with Redis and provider-aware autoscaling | Use `expert_failure_policy="skip_failed"` so local experts can return degraded predictions during vendor outages. |

For this repository, the most defensible measured SLA baseline is the
keystroke-only path. The face and multimodal paths are implemented, but their
production SLA should wait for load tests that include real image payloads and
the exact hardware or GPU profile.

## Serverless Lambda-Style Deployment

APMoE can be deployed behind a serverless HTTP entry point, but the repository
does not currently ship a first-class Lambda handler. A production Lambda-style
deployment should wrap the FastAPI app with an ASGI adapter such as Mangum,
package dependencies and weights in a container image or layer, and call
`APMoEApp.from_config()` during cold start so each execution environment loads
models once.

Recommended shape:

```python
from mangum import Mangum

from apmoe.core.app import APMoEApp
from apmoe.serving.app_factory import create_api

apmoe_app = APMoEApp.from_config("/var/task/config.json")
api = create_api(apmoe_app, security_provider=...)
handler = Mangum(api)
```

Use serverless when:

- traffic is bursty or idle for long periods;
- the active model set is small enough for package and memory limits;
- remote experts do most heavy inference work;
- cold starts can be hidden with provisioned concurrency;
- simple regional availability is enough.

Advantages:

- scales down to zero for low-traffic services;
- minimal server operations;
- per-request cost model is attractive for sparse workloads;
- easy isolation between versions through aliases and weighted routing.

Disadvantages:

- cold starts include Python import time and model loading unless provisioned
  concurrency is used;
- large TensorFlow/Keras or GPU workloads may exceed size, memory, startup, or
  execution-time limits;
- each warm execution environment has its own in-memory circuit breakers and
  fallback state;
- Redis or another external store is still needed for shared JWT invalidation
  and global rate limits;
- local file writes are limited to ephemeral storage.

Serverless production checklist:

- package the config and local weights immutably with the function version;
- set provisioned concurrency for latency-sensitive endpoints;
- use API Gateway or an equivalent layer for TLS, WAF, request-size limits, and
  coarse rate limits;
- configure Redis or another shared store for strict invalidation and
  cross-instance rate limiting;
- set conservative `endpoint_timeout` values for remote experts so Lambda
  invocations do not wait until platform timeout;
- emit logs and audit events to a centralized sink.

APMoE-specific serverless examples:

- `configs/keystroke.json`: viable for Lambda-style deployment because the
  ONNX model is small and the request payload can be under 1 KB for short
  sessions. Use provisioned concurrency if p95 latency must exclude cold
  starts.
- `configs/multimodal.json`: usually not the first serverless choice because
  TensorFlow/Keras import and model loading can make cold starts expensive.
  Prefer dedicated infrastructure unless traffic is sparse and latency is not
  strict.
- `configs/llm_remote.json`: viable when the remote LLM does the heavy work,
  but set the function timeout above `endpoint_timeout` plus retry backoff. With
  the sample `endpoint_timeout=60.0`, a production Lambda timeout below 70 to
  90 seconds is risky if retries remain enabled.

## Dedicated Infrastructure Deployment

Dedicated infrastructure means VMs, containers, ECS, Kubernetes, or another
long-running compute platform running `apmoe serve`.

Basic command:

```bash
apmoe serve --config config.json --workers 4
```

Use dedicated infrastructure when:

- latency targets are strict and cold starts are unacceptable;
- the deployment uses local TensorFlow/Keras, ONNX, or GPU inference;
- traffic is steady enough to justify always-on capacity;
- the service needs custom networking, private model endpoints, or GPUs;
- autoscaling and rollout control must be explicit.

Advantages:

- predictable warm latency because models load once per worker at startup;
- better support for GPU nodes and larger model artifacts;
- simple health-check integration with load balancers through `/v1/health`;
- easier sidecar or daemon integration for logging, metrics, and secrets;
- mature blue/green and canary deployment patterns.

Disadvantages:

- baseline cost continues during idle periods;
- operators own patching, image builds, node health, and autoscaling policy;
- each uvicorn worker loads its own model copy, increasing memory usage;
- horizontal scaling requires shared Redis for strict rate limits and token
  invalidation;
- multi-region failover must be designed outside the framework.

Dedicated production checklist:

- build immutable images that include source, config, and pinned model weights;
- run `apmoe validate --config config.json` in CI and as a pre-start check;
- use `/v1/health` as a readiness probe and remove degraded replicas from
  traffic;
- terminate TLS at ingress or load balancer;
- configure Redis-backed rate limiting and token invalidation for multi-worker
  or multi-node deployments;
- centralize request logs, security audit logs, and error alerts;
- autoscale on CPU/GPU utilization, request concurrency, p95 latency, and error
  rate rather than CPU alone.

Recommended production serving block for a horizontally scaled APMoE service:

```json
{
  "serving": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "cors_origins": ["https://your-product.example"],
    "rate_limit": 600,
    "rate_limit_store": "redis",
    "rate_limit_redis_url": "redis://apmoe-redis:6379/0",
    "token_invalidation_store": "redis",
    "token_invalidation_redis_url": "redis://apmoe-redis:6379/0",
    "authentication_enabled": true,
    "authorization_enabled": true,
    "log_level": "info"
  }
}
```

For the actual repository configs, apply this serving block to
`configs/keystroke.json` or `configs/multimodal.json` before production use.
The sample configs intentionally keep deployment simple for local demos.

## Fallback Behavior

Fallback happens at multiple layers. These layers should be understood
separately because they protect different failure modes.

### Modality Fallback

If a request omits a configured modality, or a modality processing chain raises
`ModalityError`, the pipeline excludes that modality from the processed input
map. Experts that require missing or failed modalities are skipped and reported
in `Prediction.skipped_experts`.

This lets a multimodal service continue with the remaining available modalities
when the configured experts allow it.

APMoE example using `configs/multimodal.json`:

- Request includes only `keystroke`.
- `keystroke_age_expert` can run.
- `face_age_expert` is skipped because `image` is missing.
- The response still returns a prediction, and `skipped_experts` contains
  `["face_age_expert"]`.

This is the preferred fallback for clients that can collect keystrokes even
when image capture is blocked by camera permissions, browser policy, or user
choice.

### Expert Fallback

`expert_failure_policy` controls runnable expert failures:

```json
{
  "apmoe": {
    "expert_failure_policy": "skip_failed"
  }
}
```

- `fail_fast` is the default. Any runnable expert failure aborts prediction.
- `skip_failed` records failed runnable experts in
  `Prediction.metadata["failed_experts"]` and aggregates the successful
  experts.
- If every runnable expert fails, the pipeline raises `PipelineError` and the
  HTTP layer returns `503`.

Use `skip_failed` when partial predictions are acceptable and the response
metadata is consumed by the client or monitoring system. Use `fail_fast` for
strict validation, regulated workflows, or cases where every configured expert
is required.

Recommended multimodal fallback config:

```json
{
  "apmoe": {
    "expert_failure_policy": "skip_failed",
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator",
      "weights": {
        "keystroke_age_expert": 1.0,
        "face_age_expert": 1.0
      }
    }
  }
}
```

If `FaceAgeExpert` fails on a bad image but the keystroke expert succeeds, the
prediction will include only the keystroke output and will record
`face_age_expert` in `metadata.failed_experts`. Operators should alert on an
increase in `failed_experts.face_age_expert` because it may indicate a camera
payload regression, bad image preprocessing, or a broken Keras artifact.

### Remote Expert Fallback

Remote experts include:

- `endpoint_timeout` per expert;
- retries for transient timeouts, network errors, and HTTP `429`, `502`, `503`,
  and `504`;
- exponential backoff with optional jitter;
- a process-local circuit breaker per remote expert;
- response byte limits and JSON content-type checks.

Example:

```json
{
  "apmoe": {
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

Retries and circuit breakers reduce the blast radius of temporary vendor
outages. To convert a remote expert outage into a degraded prediction, prefer
a paired local fallback: configure the remote expert with `fallback_expert` and
mark the local standby expert with `fallback_only=true`.

APMoE remote fallback pattern:

```json
{
  "apmoe": {
    "expert_failure_policy": "skip_failed",
    "remote_fallback_policy": "transient_only",
    "remote_retry": {
      "max_attempts": 2,
      "initial_delay_s": 0.2,
      "max_delay_s": 1.0,
      "backoff_multiplier": 2.0,
      "jitter": true
    },
    "remote_circuit_breaker": {
      "enabled": true,
      "failure_threshold": 3,
      "recovery_timeout_s": 20.0
    }
  }
}
```

Use `remote_fallback_policy="transient_only"` to fallback on timeouts, network
errors, transient HTTP `429/502/503/504`, and open circuits. Use
`"any_remote_error"` only when local fallback is acceptable for malformed remote
responses or other provider-side errors. Use `"disabled"` when remote output is
required.

Fallback-only local experts are loaded and shown by `apmoe validate`, but they
do not run during normal inference. When fallback succeeds, aggregation treats
the output as the remote expert's slot so existing `aggregation.weights` remain
stable, and the response records the actual local fallback in
`metadata.fallback_experts`.

### Redis Fallback

Redis-backed rate limiting and JWT invalidation preserve availability if Redis
fails after startup by falling back to process-local in-memory stores.

This is an availability fallback, not a consistency guarantee:

- rate limits are no longer globally coordinated while Redis is unavailable;
- token invalidations created during fallback are local to the current worker;
- fallback state is not replayed into Redis after recovery.

For strict global enforcement, run Redis in a highly available managed
configuration and alert on `redis_rate_limit_fallback` and
`redis_token_invalidation_fallback` audit events.

### Infrastructure Fallback

At the infrastructure level:

- use `/v1/health` to remove replicas with unloaded experts from service;
- run at least two replicas or two provisioned serverless environments for any
  SLA above a single-node best effort;
- set ingress request-size limits to protect workers from large payloads;
- use load-balancer timeouts that are longer than the configured remote expert
  timeout but shorter than the client timeout;
- avoid automatic HTTP retries for non-idempotent client workflows unless the
  client can tolerate duplicate predictions. APMoE prediction is logically
  read-only, but upstream systems may still count or bill requests.

## Stateful And Stateless System Parts

Treat the following as stateless and replaceable:

- APMoE FastAPI replicas;
- uvicorn workers;
- local in-memory remote circuit breaker state;
- local in-memory rate-limit fallback state;
- local in-memory JWT invalidation fallback state;
- per-request modality, expert, and aggregation objects.

Treat the following as stateful or externally consistent:

- Redis keys for shared rate limiting and token invalidation;
- JWT signing keys, issuer configuration, and revocation policy;
- model artifact versions and SHA-256 or signed manifest metadata;
- remote expert endpoint versions and vendor-side state;
- centralized logs, audit records, and monitoring time series;
- DNS, load balancer routing, and deployment version aliases.

The key operational rule is simple: APMoE workers can be killed and replaced at
any time after a new worker has passed bootstrap and `/v1/health`; shared
security and observability state must live outside those workers.

Failure-state matrix:

| Component | Stateless or stateful | What breaks if it disappears | APMoE fallback | Infrastructure action |
|---|---|---|---|---|
| One APMoE replica | Stateless | In-flight requests to that replica fail | Other replicas continue | Load balancer removes failed target; autoscaler replaces it. |
| One uvicorn worker | Stateless | Requests handled by that process fail | Parent process or orchestrator may keep other workers alive | Restart process or replace container. |
| Local model file at runtime | Immutable artifact | New pods fail bootstrap or validation | Existing loaded workers keep serving until restarted | Fail deployment before traffic shift using `apmoe validate`. |
| Remote expert endpoint | External stateful dependency | Remote expert raises `ExpertError` | Retries, circuit breaker, `skip_failed` if other experts exist | Route to alternate provider or rollback config. |
| Redis | Stateful shared store | Global rate limits and JWT revocation lose coordination | Process-local fallback | Restore Redis and alert; consider fail-closed at gateway for strict security. |
| JWT issuer/key store | Stateful security dependency | New tokens cannot be trusted or issued | None inside APMoE | Rotate keys and coordinate issuer outage response. |

## Load-Test Results And SLA Guidance

The current measured results are local benchmark results, not a universal SLA.
They are useful as a baseline for the built-in lightweight keystroke path.

Measured on May 3, 2026, single worker, localhost:

| Endpoint | Concurrency | Duration | Throughput | Error rate | Latency |
|---|---:|---:|---:|---:|---|
| `GET /v1/health` | 20 users | 15 s | 2,208 RPS | 0% | avg 9.0 ms, p95 10.7 ms, p99 12.1 ms |
| `POST /v1/predict` keystroke | 10 users | 15 s | 1,620 RPS | 0% | avg 6 ms, p95 8 ms, p99 13 ms |

Additional remote LLM testing recorded approximately 5.9 to 6.0 seconds of
pipeline latency for a local LM Studio vision-language model path. That path is
model- and hardware-dependent and should not be mixed with the ONNX keystroke
baseline.

Recommended initial SLA targets:

| Tier | Suggested use | Availability target | Latency target | Notes |
|---|---|---:|---|---|
| Internal demo | research, classroom, prototype | best effort or 99.0% | p95 under 1 s for local lightweight experts | Single region and manual recovery are acceptable. |
| Standard vendor | production integration with local ONNX/Keras experts | 99.5% to 99.9% | p95 under 300 ms for keystroke-only after environment-specific testing | Requires at least two replicas and shared Redis. |
| Premium vendor | customer-facing production with remote dependency fallback | 99.9% to 99.95% | p95 target per modality bundle, including vendor remote p95 | Requires multi-AZ, alerting, canary rollout, and documented degradation behavior. |

Do not promise the measured 1,620 RPS as a vendor SLA without reproducing it on
the vendor's own hardware, payloads, model mix, auth settings, network path,
and logging configuration.

### Example SLA: Keystroke-Only Vendor API

This is the most realistic first SLA for the current repository because it maps
directly to `configs/keystroke.json` and the measured `scripts/load_test_predict.py`
benchmark.

Recommended published statement after reproducing the benchmark on vendor
infrastructure:

```text
Service: APMoE keystroke-only age prediction
Framework version: apmoe 0.1.0
Config: configs/keystroke.json with production serving/security overrides
Model: keystroke_age_expert.onnx + keystroke_constants.json
Availability target: 99.5% monthly
Latency objective: p95 <= 100 ms for requests up to 200 keystroke triples
Throughput objective: 600 RPS sustained per region with 0.1% or lower 5xx rate
Payload limit: 32 KB JSON body
Fallback: no alternate expert; if the keystroke expert is unavailable, return 503
Measurement window: 5-minute rolling p95, excluding planned maintenance
```

Why not set the target at the measured 1,620 RPS? Because the benchmark was
single-worker, localhost, and unauthenticated. A vendor SLA should leave room
for TLS, authn/authz, real gateway behavior, logging, noisy neighbors, and
deployment variance.

### Example SLA: Multimodal Local API

This maps to `configs/multimodal.json`.

```text
Service: APMoE multimodal age prediction
Modalities: keystroke + image
Experts: KeystrokeAgeExpert + FaceAgeExpert
Availability target: 99.0% until production image load tests are complete
Latency objective: p95 defined only after measuring real image payloads
Payload limit: 1 MB JSON body, 500 KB recommended client image cap before base64
Fallback: if image is missing or FaceAgeExpert fails, return degraded keystroke-only prediction when expert_failure_policy="skip_failed"
Degraded response requirement: clients must inspect skipped_experts and metadata.failed_experts
```

This SLA should not borrow keystroke-only latency numbers. The face path uses
image decoding and Keras inference, and the bottleneck is different.

### Example SLA: Remote LLM API

This maps to `configs/llm_remote.json`.

```text
Service: APMoE remote vision LLM age prediction
Expert: LMStudioExpert or compatible RemoteExpert subclass
Availability target: vendor provider SLA minus APMoE orchestration error budget
Latency objective: p95 = provider p95 + APMoE preprocessing p95 + network p95
Timeout: endpoint_timeout must be below client timeout and platform timeout
Fallback: if no local alternate expert exists, remote outage returns 503
Provider requirement: publish remote endpoint quota, timeout, and support window
```

For this profile, the APMoE SLA cannot be stronger than the remote model
provider unless the deployment includes an alternate expert that can serve a
degraded prediction.

## Bandwidth Guidance

APMoE bandwidth planning must account for both inbound client traffic and
outbound remote expert traffic. The endpoint itself accepts JSON bodies, and
the route serializes each modality value to bytes before the modality processor
runs. That means base64 images in JSON incur the usual base64 expansion before
APMoE even decodes them.

Estimate average application bandwidth from measured payloads:

```text
inbound_mbps =
  rps * (average_client_request_bytes + average_apmoe_response_bytes) * 8 / 1_000_000

outbound_mbps =
  remote_rps * (average_remote_request_bytes + average_remote_response_bytes) * 8 / 1_000_000

recommended_network_mbps = (inbound_mbps + outbound_mbps) * headroom_multiplier
```

Use `headroom_multiplier=2` for stable internal traffic and `3` for bursty
public traffic, mobile clients, large images, or remote providers with retries.

### Project Payload Sizes

Concrete sizes from this repository:

| Payload | Source | JSON body bytes | Notes |
|---|---:|---:|---|
| Current load-test keystroke request | `scripts/load_test_predict.py` | 248 bytes | 20 triples with the wrapper key `{"keystroke": ...}`. |
| Generated 50-triple keystroke session | same JSON shape | about 595 bytes | Representative short browser session. |
| Generated 100-triple keystroke session | same JSON shape | about 1,175 bytes | Good target for normal keystroke-only testing. |
| Generated 200-triple keystroke session | same JSON shape | about 2,355 bytes | Longer session; still small compared with images. |
| 50 KB image as base64 JSON | `{"image": "<base64>"}` | about 66,680 bytes | Base64 expands binary by about 33%. |
| 250 KB image as base64 JSON | `{"image": "<base64>"}` | about 333,348 bytes | Common for compressed mobile camera uploads. |
| 1 MB image as base64 JSON | `{"image": "<base64>"}` | about 1,333,348 bytes | Should be guarded by ingress limits. |
| LLM remote image after `Base64ImageCleaner` | `configs/llm_remote.json` outbound body | about 1.3 KB to 2.5 KB for image string, plus prompt/model JSON | This is outbound from APMoE to LM Studio or a remote provider after compression. |

For responses, plan with actual observed payloads from your deployment. A
keystroke-only response with one expert is commonly around 1 KB to 2 KB because
it includes `per_expert_outputs`, metadata, skipped experts, and
recommendations. A multimodal response with two experts is commonly around
2 KB to 5 KB depending on expert metadata.

### Worked Example: Measured Keystroke Baseline

Assumptions:

- endpoint: `POST /v1/predict`;
- config: `configs/keystroke.json`;
- measured throughput: 1,620 RPS;
- request size: 248 bytes from `scripts/load_test_predict.py`;
- assumed average response size: 1,500 bytes;
- no remote expert outbound traffic.

```text
inbound_mbps = 1,620 * (248 + 1,500) * 8 / 1,000,000
             = 22.65 Mbps

recommended_network_mbps = 22.65 * 2
                         = 45.3 Mbps
```

For keystroke-only traffic, reserving hundreds of Mbps is usually too
conservative for the actual project payload. CPU and worker count will usually
be the constraint before network bandwidth.

### Worked Example: Production Keystroke Session

Assumptions:

- 600 RPS sustained production target;
- 100-triple request size: 1,175 bytes;
- response size: 1,800 bytes;
- 3x public-traffic headroom.

```text
inbound_mbps = 600 * (1,175 + 1,800) * 8 / 1,000,000
             = 14.28 Mbps

recommended_network_mbps = 14.28 * 3
                         = 42.84 Mbps
```

Recommended vendor statement: "For keystroke-only APMoE traffic at 600 RPS
with 100-triple sessions, reserve at least 50 Mbps of application bandwidth per
region, then validate with production TLS, gateway, and logging enabled."

### Worked Example: Multimodal Local Inference

Assumptions:

- config: `configs/multimodal.json`;
- 120 RPS sustained;
- 100-triple keystroke body: 1,175 bytes;
- client sends a 250 KB JPEG as base64 JSON: 333,348 bytes;
- response size: 3,500 bytes;
- no remote expert outbound traffic;
- 3x headroom because image sizes vary heavily.

```text
inbound_mbps = 120 * (1,175 + 333,348 + 3,500) * 8 / 1,000,000
             = 324.50 Mbps

recommended_network_mbps = 324.50 * 3
                         = 973.50 Mbps
```

For image-heavy APMoE deployments, bandwidth becomes a real capacity concern.
Use client-side image resizing, API gateway request-size limits, and a documented
maximum image size. A practical first cap is 250 KB to 500 KB compressed image
bytes before base64.

### Worked Example: Remote LM Studio Path

Assumptions:

- config: `configs/llm_remote.json`;
- 20 RPS sustained;
- client sends a 250 KB image as base64 JSON: 333,348 bytes inbound;
- APMoE compresses it with `Base64ImageCleaner` to a 2,500-byte base64 string;
- remote request body, including `model`, `system_prompt`, and `input`: 3,500
  bytes;
- remote response body: 2,000 bytes;
- APMoE response to client: 2,500 bytes;
- 3x headroom because remote retries can duplicate outbound calls.

```text
inbound_mbps = 20 * (333,348 + 2,500) * 8 / 1,000,000
             = 53.74 Mbps

outbound_mbps = 20 * (3,500 + 2,000) * 8 / 1,000,000
              = 0.88 Mbps

recommended_network_mbps = (53.74 + 0.88) * 3
                         = 163.86 Mbps
```

In this path, inbound image upload bandwidth dominates. The remote LLM call is
small because APMoE compresses the image before calling the provider.

### Bandwidth Controls To Put In Front Of APMoE

- Limit JSON body size at the ingress layer. Suggested defaults: 32 KB for
  keystroke-only, 1 MB for local multimodal, and 1 MB for LLM image workflows.
- Publish a client image guidance: resize longest side before upload and prefer
  JPEG/WebP over PNG screenshots.
- Track `Content-Length`, response size, and `pipeline_latency_s` together.
  Large payloads can look like model latency unless upload time is separated
  from inference time.
- For remote experts, track outbound provider request size separately from
  client request size.

## Vendor SLA Template

Vendors building on APMoE should publish an SLA per deployed model bundle, not
only per framework version. A useful vendor SLA should include:

- APMoE version and config hash;
- expert names, classes, model artifact versions, and integrity policy;
- local hardware or remote provider details;
- supported modalities and maximum payload size per modality;
- authentication and rate-limit assumptions;
- measured p50, p95, p99, max latency, RPS, concurrency, and error rate;
- availability target and monthly error budget;
- fallback behavior for missing modalities, failed local experts, failed remote
  experts, Redis outages, and vendor dependency outages;
- maintenance window and deprecation policy;
- rollback and incident response time objectives.

Minimum validation before publishing an SLA:

```bash
apmoe validate --config config.json
python scripts/load_test.py --url http://127.0.0.1:8000/v1/health --users 20 --duration 15
python scripts/load_test_predict.py http://127.0.0.1:8000/v1/predict 10 15
python scripts/e2e_resilience.py
```

Vendors should run longer soak tests for production commitments, usually 30 to
60 minutes at expected steady load plus a separate burst test.

## Hot Swapping, Rollout, Rollback, And Autoscaling

APMoE does not hot-reload local weights inside a running process. Treat each
config and model bundle as an immutable version and replace stateless servers
instead.

Recommended versioning:

- build image `apmoe:<app-version>` with pinned dependencies;
- include `config.json` or mount an immutable config version;
- version model artifacts as `expert-name:model-version`;
- pin local weights with `experts[].integrity.sha256`;
- pin remote models with signed manifests where possible.

Blue/green rollout:

1. Deploy green replicas with the new config and model artifacts.
2. Run `apmoe validate` and wait for `/v1/health` to return healthy.
3. Send synthetic prediction checks for every supported modality bundle.
4. Shift a small percentage of traffic to green.
5. Compare latency, error rate, confidence distribution, skipped experts, and
   age output distribution against blue.
6. Move all traffic to green if checks pass.
7. Keep blue warm until the rollback window closes.

Rollback:

1. Shift load-balancer traffic back to the previous stateless server pool.
2. Stop or scale down the bad version.
3. Preserve logs and audit records for the failed version.
4. Re-run validation with the previous config if artifacts or external
   endpoints changed.

Canary rollout:

- route 1% to 5% of traffic to the new version first;
- use the same request and response schema across versions;
- alert on increased `PipelineError`, remote circuit-breaker opens, p95/p99
  latency, confidence drift, or sudden changes in `skipped_experts`;
- avoid canarying a stateful Redis key-prefix change unless both versions can
  operate against the same key scheme.

Expert-level hot swap without replacing servers is limited. You can run two
versions as two configured experts and use aggregator weights to compare or
blend them, but changing those weights still requires a new config and process
restart. For production, prefer stateless server replacement over runtime
mutation.

### Concrete Hot-Swap Scenario: Keystroke Model v1 to v2

Goal: replace `weights/keystroke_age_expert.onnx` with a new model while keeping
the API available.

Version `apmoe-keystroke-v1`:

```json
{
  "name": "keystroke_age_expert",
  "class": "apmoe.experts.builtin.KeystrokeAgeExpert",
  "weights": "./weights/keystroke_age_expert_v1.onnx",
  "modalities": ["keystroke"],
  "integrity": {
    "sha256": "<v1-sha256>"
  }
}
```

Version `apmoe-keystroke-v2`:

```json
{
  "name": "keystroke_age_expert",
  "class": "apmoe.experts.builtin.KeystrokeAgeExpert",
  "weights": "./weights/keystroke_age_expert_v2.onnx",
  "modalities": ["keystroke"],
  "integrity": {
    "sha256": "<v2-sha256>"
  }
}
```

Rollout:

1. Build a new image containing `keystroke_age_expert_v2.onnx`,
   matching `keystroke_constants.json`, and the v2 config.
2. Run `apmoe validate --config config.json` in CI. This catches missing
   files, invalid SHA-256, bad config, and unloaded experts.
3. Start v2 replicas without sending production traffic.
4. Wait for `/v1/health` to return `healthy`.
5. Send the same synthetic keystroke request to v1 and v2. Compare response
   shape, status code, confidence, and prediction distribution. Exact age may
   differ; schema and operational health should not.
6. Shift 5% of traffic to v2 for 15 to 30 minutes.
7. Promote to 50%, then 100%, if p95 latency, 5xx rate, and confidence
   distribution stay within rollout thresholds.

Rollback:

1. Shift traffic back to v1 at the load balancer.
2. Keep v2 logs and audit events for analysis.
3. Disable or delete v2 replicas only after confirming v1 is healthy.
4. Open an incident note if v2 failed because of model behavior, constants
   mismatch, or config/integrity failure.

This pattern is preferred over replacing a weight file on disk under a running
worker. Existing workers have already loaded model state in memory, and runtime
mutation would make replicas disagree about which model they are serving.

### Concrete Fallback Scenario During Rollout

For `configs/multimodal.json`, a rollout can temporarily run:

- v1 pool: stable keystroke + stable face;
- v2 pool: stable keystroke + new face model.

Set `expert_failure_policy="skip_failed"` in v2 during canary. If the new face
model fails on some images, v2 can still return keystroke-only degraded
predictions. Rollback immediately if either condition occurs:

- `metadata.failed_experts.face_age_expert` increases above the agreed
  threshold;
- `skipped_experts` increases for requests that include valid image payloads.

Do not hide this degradation from clients. The response metadata is part of the
contract and must be included in SLA reporting.

Autoscaling guidance:

- scale vertically first with `serving.workers` until memory or CPU/GPU
  contention appears;
- scale horizontally with more replicas behind a load balancer;
- use Redis-backed stores when scaling beyond one process;
- scale on p95 latency, request concurrency, CPU/GPU utilization, and error
  rate;
- keep minimum warm capacity for local model deployments to avoid repeated
  bootstrap latency;
- for remote-heavy deployments, scale based on outbound provider latency and
  provider rate limits as well as local CPU.

## Out Of Scope Features

The following are intentionally outside the current framework scope or not yet
implemented as first-class features:

- model training and fine-tuning APIs;
- dataset hosting or redistribution;
- legal compliance decisions for age verification use cases;
- first-class Lambda/Mangum handler;
- Dockerfile, Compose, Helm chart, or Kubernetes manifests;
- multi-region failover controller;
- in-process model hot reload;
- built-in autoscaler or queue manager;
- Prometheus `/metrics` endpoint and Grafana dashboards;
- distributed tracing backend;
- persistent user profiles, sessions, or prediction history;
- GPU scheduling and model placement;
- request deduplication, idempotency keys, or billing counters;
- contractual SLA guarantees for third-party remote providers.

## Open Decisions

These items need project-owner or vendor input before final production
commitments:

- target public SLA tier and error budget;
- preferred cloud provider and deployment shape;
- whether Lambda support is required as a shipped adapter or only documented as
  a pattern;
- expected production payload mix and average request size;
- measured face-model and multimodal load-test results on production-like
  hardware;
- vendor remote expert availability, timeout, quota, and support commitments;
- exact copyright holder for the MIT license notice.

