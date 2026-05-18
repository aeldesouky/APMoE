# Developer Experience (DX) and Extensibility Guide

APMoE is designed from the ground up as a generic, Inversion of Control (IoC) framework. This document details how the developer experience (DX), configuration management, and error handling have been optimized to make extending, deploying, and debugging the framework as seamless as possible.

## 1. Concrete DX Highlights

These are the specific, implemented DX features that differentiate APMoE's integration and development lifecycle:

- **One-command, runnable project scaffold**: `apmoe init <project> --builtin` creates a ready-to-serve project with a valid `config.json`, stub extension files (`custom_expert.py`, `custom_cleaner.py`, etc.), and a `README.md`. The `--builtin` flag bundles working ONNX/Keras weights natively, making `apmoe serve -c config.json` immediately functional out-of-the-box.
- **Below-Threshold Confidence & Active Recommendations**: The framework actively monitors inference confidence against a user-defined `confidence_threshold` in the configuration. If the score falls below this threshold, the pipeline automatically populates `Prediction.metadata["recommendations"]` with an ordered, actionable list of improvement hints (e.g., warning about low keystroke coverage, uncalibrated regression models, or suggesting additional modalities). 
- **Pre-flight validation before runtime**: `apmoe validate --config config.json` acts as a dry-run health check. It executes Pydantic schema validation, dotted-path resolution, weight-file existence checks, and expert health checks. Configuration mistakes fail fast with highly actionable error messages before the server even binds to a port.
- **Config overrides without editing JSON**: Serving parameters can be overridden via `APMOE_SERVING_*` environment variables (host, port, workers, CORS, rate limit), ensuring deploy-time infrastructure tweaks do not require mutating the core configuration file.
- **API docs with concrete examples**: Swagger UI (`/docs`) and ReDoc (`/redoc`) ship automatically, pre-filled with OpenAPI schema examples for all supported modalities (keystroke triples, IKDD strings, image payloads), allowing instant inference testing without writing a client.
- **Versioned API with safe migration paths**: Endpoints are properly versioned under `/v1/*` (`/v1/predict`, `/v1/health`, `/v1/info`). Legacy unversioned paths remain for backward compatibility but automatically return standard HTTP `Deprecation` and `Sunset` headers alongside `X-API-Version: 1` to explicitly guide client migration.
- **Correlation IDs and structured logs**: Every HTTP request is automatically assigned an `X-Correlation-ID`. Structured JSON logs are emitted containing the method, path, status, and latency, making debugging and distributed tracing instantly accessible.
- **Zero-boilerplate plugin registration**: Decorators like `@expert_registry.register("name")` allow new experts, processors, and strategies to be registered without manual wiring. IoC binding occurs cleanly via dotted paths in the `config.json`.
- **Local inference without a client**: `apmoe predict --input <dir>` reads files named after modality keys directly from disk and prints formatted JSON output, enabling offline smoke tests and batch debugging.
- **Graceful degradation built-in**: The inference pipeline is inherently fault-tolerant. If a modality is missing, malformed, or an expert fails, the pipeline safely skips it and reports the details in `skipped_experts` and `failed_modalities` within the response payload rather than crashing the system.
- **Lifecycle observability hooks**: The `InferencePipeline` exposes hooks (`on_before_process`, `on_after_embed`, `on_after_expert`, `on_after_aggregate`) allowing integration teams to attach custom logging, metrics, or auditing without monkey-patching core code.
- **Schema-level guardrails**: The Pydantic configuration layer enforces strict structural rules (e.g., ensuring all expert modalities are declared, unique names, valid bounds for thresholds), rejecting invalid states natively at boot.
- **Multi-worker serving that just works**: `apmoe serve --workers N` correctly utilizes an ASGI factory app pattern, completely avoiding the broken state-sharing issues common in naive Python multi-process deployments.
- **Remote expert endpoints**: Any expert can delegate inference to an external HTTP model server (HuggingFace Inference API, OpenAI-compatible endpoints, other APMoE instances, local LLMs, custom REST APIs) via a single `"endpoint"` config field — no weight file required. A fully customisable `request_template` (with `{{modalities.<name>}}` predict-time placeholder substitution) and `response_mapping` (dot-path extraction) allow adaptation to any remote API schema. `$ENV_VAR` substitution applies at bootstrap to the endpoint URL, all header values, and all literal string leaves in the request template — keeping every deployment-specific value (server URL, model name, system prompts, API keys) out of the config file entirely. A dedicated `apmoe.processing.llm` module provides `Base64ImageCleaner` and `PassthroughImageAnonymizer` for routing the image modality to vision-LLM endpoints without affecting the standard local-inference pipeline. See [`docs/remote_expert_endpoints.md`](./remote_expert_endpoints.md) for the full reference.

## 2. CLI Toolchain and Fast Feedback

The APMoE CLI provides an immediate, guided path from blank slate to production serving.

### Scaffolding a Project
Running `apmoe init <project-name> --builtin` gives developers a fully working project hierarchy:
- A `config.json` file pre-wired with valid processing strategies and models.
- Placeholder stub files (`custom_expert.py`, `custom_cleaner.py`, etc.) demonstrating exactly how to subclass the framework's Abstract Base Classes (ABCs).
- A `weights/` directory populated with working ONNX and Keras models (via the `--builtin` flag).
- A custom `README.md` containing exact next steps for validation and serving.

### Discoverability (`--help`)
Every command in the APMoE CLI exposes concise, actionable help text via `--help` or `-h`. 
- `apmoe --help` lists top-level commands.
- `apmoe init --help` shows scaffolding options.

### Pre-flight Configuration Validation
To prevent deep runtime failures in production, `apmoe validate --config config.json` acts as a comprehensive pre-flight health check, verifying:
1. Strict Pydantic schema compliance for the JSON structure.
2. Successful Python `importlib` resolution of all classes specified via dotted paths.
3. Existence and readability of all required weight files on disk.
4. Execution of generic health checks (`is_loaded()`) against each initialized `ExpertPlugin`.

## 3. Error Handling Hierarchy

APMoE utilizes a strongly typed, hierarchical exception system defined in `src/apmoe/core/exceptions.py`. This ensures high visibility and predictability.

All errors inherit from `APMoEError`, allowing broad catch-all handling, subdivided into granular domains:

### Configuration and Initialization
- `ConfigurationError`: Raised when the `config.json` structure is malformed, required fields are missing, or Pydantic cross-validation fails.
- `RegistryError`: Raised when attempting to register duplicate plugin names or failing to load an unregistered class via dotted path.

### Inference Pipeline
- `PipelineError`: Raised when the orchestrator fails structurally (e.g., post-filtering leaves absolutely zero valid experts to execute).
- `ModalityError`: Raised specifically for data-level issues during the preparation phase (Cleaner, Anonymizer, Embedder). Also handles raw data format validation failures.

### Plugin Execution
- `ExpertError`: Thrown when an expert plugin fails to load properly (missing weights), encounters an invalid tensor shape during `predict()`, or returns a malformed `ExpertOutput`.

### HTTP Serving
- `ServingError`: Handles ASGI-level failures, framework bootstrapping errors within the FastAPI startup sequence, or missing middleware requirements.

## 4. Below-Threshold Confidence & Recommendations

The pipeline does not just execute inference; it actively helps integrators improve their input data quality.

If the final `Prediction.confidence` drops below the `confidence_threshold` defined in the active configuration, the pipeline's recommendation engine evaluates the context and appends human-readable guidance to `Prediction.metadata["recommendations"]`.

**Example Scenarios Handled:**
- **Zero experts executed**: Recommends which valid modality keys should be provided.
- **Sparse keystroke sessions**: If the `KeystrokeAgeExpert` detects a low feature coverage fraction (relying heavily on median imputation), the API explicitly requests longer sessions (e.g., ">50 keystrokes").
- **Uncalibrated modalities**: Warns when only a regression model (like Face Age) executes, explaining that regression yields no probabilistic confidence and suggesting the addition of keystroke data.

This ensures that "bad" predictions are accompanied by exact, programmatic instructions on how to achieve "good" predictions.

## 5. Complete Extensibility Strategy (Inversion of Control)

APMoE's true power lies in its dependency injection architecture. Instead of hard-coding how specific modalities are processed, APMoE uses the JSON configuration to bind developer implementations to framework interfaces:

1. **ModalityProcessors**: Convert raw payload bytes into strictly typed `ModalityData` (e.g., parsing IKDD strings or decoding base64 JPEGs).
2. **Pipelines**: Each modality defines a chained processing sequence:
   * **Cleaners**: Ensure data validity (resizing images, standardizing keystroke timings).
   * **Anonymizers**: Mutate the payload to preserve privacy *before* inference (e.g., blurring faces).
   * **Embedders (Optional)**: Pre-compute vector embeddings for downstream models.
3. **Experts (MoE)**: Isolated inference modules (`ExpertPlugin`). The developer specifies required modalities (e.g., `["keystroke"]`), and the framework automatically isolates the data and invokes each expert concurrently. Two execution modes are available:
   * **Local experts** — load a pretrained weight file (`weights` field); inference runs in-process.
   * **Remote experts** (`RemoteExpert`) — no weight file; inference is delegated to an external HTTP endpoint (`endpoint` field) with a fully customisable request/response schema. A two-pass substitution system separates deployment-time configuration (`$VAR` → expanded once at bootstrap from env) from per-request data injection (`{{modalities.<name>}}` → substituted on every predict call). The `apmoe.processing.llm` module provides LLM-compatible image processors that keep the standard local-inference pipeline untouched. See [`docs/remote_expert_endpoints.md`](./remote_expert_endpoints.md).
4. **Aggregators**: Combine multiple `ExpertOutput` objects into a single final `Prediction`. The built-in `WeightedAverageAggregator` natively handles confidence/value arithmetic, but custom heuristic combiners can be easily plugged in.

## 6. Serving Middleware and Security Out-of-the-Box

The `apmoe serve` command generates a robust FastAPI ASGI application wrapped in multiple layers of production-ready middleware.

**Currently Implemented:**
- **CORS Middleware**: Native, configurable Cross-Origin settings to allow web dashboard integrations.
- **Request Logging**: Built-in correlation ID tracking (`X-Correlation-ID`) tied to structured logs for every inbound request.
- **Stateless Authn/Authz**: JWT Bearer authentication, route-scope authorization, and fail-closed serving defaults.
- **Token Invalidation**: In-memory or Redis-backed JWT `jti` invalidation, with an abstract store interface for external backends.
- **Rate Limiting**: In-memory or Redis-backed sliding-window rate limits for protecting heavy AI inference endpoints.
- **Remote Expert Hardening**: Production-gated endpoint allowlists, HTTPS/private-network safeguards, response byte caps, and JSON content-type checks.
- **Model Integrity**: Local SHA-256 artifact checks and RSA-PSS-SHA256 signed remote manifests verified with pinned public keys.
- **Auditability**: Structured security audit events, custom audit hooks, global correlation IDs, and secret redaction.
- **Legacy AuthPlugin Hook**: Backwards-compatible binary authentication for existing integrations.

See [`docs/dev/security.md`](./dev/security.md) for the full security reference and production checklist.

## 7. Observability

APMoE treats observability as a first-class citizen so that integrators have full visibility into the framework's operational health.

**✅ Currently Implemented:**
- **Structured Telemetry**: Every HTTP request emits a JSON-structured log with latency, HTTP status, and an auto-generated `X-Correlation-ID`.
- **Pipeline Metrics**: The final prediction object tracks `pipeline_latency_s`, `skipped_experts`, and `failed_modalities`.
- **Health & Readiness**: `GET /v1/health` dynamically queries `ExpertRegistry.health_check()`, returning `{"status": "healthy"}` only if all models are loaded in memory. If any fail, it gracefully returns a 503 degraded status.

**🔄 In Progress:**
- **Full User-Facing Observability**: We are extending the framework to provide complete, zero-configuration observability suites natively for users. This will allow integrators to monitor system health, individual expert performance, and traffic analysis seamlessly.
- **Prometheus Metrics Endpoint (`/metrics`)**: Currently, logs are localized to the uvicorn worker process. Once implemented, APMoE will expose a `/metrics` route allowing Grafana/Prometheus to natively scrape real-time error rates, RPS (requests per second), and latency percentiles.

## 8. User Experience (UX)

While APMoE is an API framework, the "user" is the client application integrator. The UX is designed to be highly deterministic and actionable.

**✅ Currently Implemented:**
- **Actionable Fallbacks**: Instead of failing blindly, missing data results in partial predictions, returning exactly what succeeded and what failed.
- **Proactive Guidance**: The Below-Threshold recommendation engine transforms low-confidence failures into clear, human-readable instructions.
- **Consistent Response Shapes**: Errors always return a predictable `{"detail": "..."}` shape with appropriate HTTP status codes (422 for malformed payloads, 503 for unavailable experts).

## 9. Frictionless Setup & Documentation

**✅ Currently Implemented:**
- **Explicit Setup Documentation**: The root `README.md` provides explicit, copy-pasteable instructions for isolating Python 3.11+ environments, installing dependencies from source, and running the test suite.
- **Zero-to-Serve Guide**: The documentation walks the developer linearly through scaffolding (`apmoe init --builtin`), validation (`apmoe validate`), and serving (`apmoe serve`).
- **Localized Context**: In addition to the root documentation, `apmoe init` natively scaffolds a localized `README.md` tailored specifically to the newly generated project structure.

**🔄 In Progress:**
- **PyPI Package Release**: We are preparing the package for public registry distribution. Soon, setup will be as completely frictionless as running `pip install apmoe` directly from PyPI, entirely bypassing the need to clone the repository manually.
- **Docker Support**: Complete environment isolation is pending. We are actively finalizing a `Dockerfile` and `docker-compose.yml`. Once implemented, *any* user will be able to bypass Python setup entirely by simply running `docker compose up`.

## 10. Fine-Tuning (Integrator-Supplied Data)

**🔄 In Progress**

The architecture for integrator-led fine-tuning is fully designed, but the implementation is pending. 

Currently, the `ExpertPlugin` Abstract Base Class provides a `load_weights()` hook for initialization. In the future state:
1. `ExpertPlugin` will expose a `finetune()` ABC hook.
2. Developers will be able to trigger this dynamically via an `apmoe finetune` CLI command or a `POST /finetune` HTTP endpoint.
3. This will allow integrators to adapt the bundled ONNX or Keras experts to their specific domain data without altering the framework's core code.

---

## 11. LLM Integration

APMoE supports routing the image modality to a local or hosted vision-language
model (LLM) as a remote expert, enabling age estimation without requiring a
locally trained face model.

### Two-pass substitution model

Remote expert configs use two independent substitution syntaxes that operate at
different lifecycle points:

| Syntax | When | Scope | Purpose |
|---|---|---|---|
| `$VARNAME` | Bootstrap (startup) | `endpoint`, header values, non-placeholder template leaves | Deployment config — server URL, model ID, system prompt, API keys |
| `{{modalities.<name>}}` | Per-request (predict) | `request_template` leaf strings | Live inference data for that call |

This cleanly separates **what changes per deployment** from **what changes per
request**, and keeps every sensitive or environment-specific value out of the
config file.

### `apmoe.processing.llm` module

`src/apmoe/processing/llm/__init__.py` provides two processors exclusively for
remote LLM dispatch. They are **not** part of the core builtin pipeline and
must never be used with local experts like `FaceAgeExpert`.

| Class | Purpose |
|---|---|
| `Base64ImageCleaner` | Resizes image to 160 px max side, JPEG-compresses at quality 35, base64-encodes — produces ~1.3–2.5 KB / ~1900 tokens, fitting within a 4096-token context window |
| `PassthroughImageAnonymizer` | No-op; satisfies the mandatory `Cleaner → Anonymizer` pipeline contract |

Both are registered in their respective registries and can be referenced by
dotted path in any config:

```json
"pipeline": {
  "cleaner":    "apmoe.processing.llm.Base64ImageCleaner",
  "anonymizer": "apmoe.processing.llm.PassthroughImageAnonymizer"
}
```

### Reference config and test script

`configs/llm_remote.json` is a fully portable reference config with zero
hardcoded values — every deployment-specific field is a `$VAR`:

```json
"endpoint":   "$LLM_ENDPOINT",
"request_template": {
  "model":         "$LLM_MODEL",
  "system_prompt": "$LLM_SYSTEM_PROMPT",
  "input":         "{{modalities.image}}"
}
```

`scripts/test_llm_remote.py` is a self-contained end-to-end script that
demonstrates the full pipeline (image file → `Base64ImageCleaner` →
`RemoteExpert` → local LLM → age prediction) and includes a `LMStudioExpert`
subclass showing how to parse non-standard LLM response schemas.

**Verified working with:** LM Studio + `google/gemma-4-e4b`, ~6 s pipeline
latency on a MacBook (no GPU), predicted age within expected range.

### DX properties of the LLM integration

- **Zero hardcoding**: all server addresses, model names, and prompts live in
  environment variables — the same config file works across dev / staging / prod.
- **Pipeline isolation**: the LLM image processors live in a separate namespace
  (`apmoe.processing.llm`) so the standard `ImageCleaner` / `FaceAgeExpert`
  chain is completely unaffected.
- **Fail-fast secrets**: missing `$VAR` references raise `ExpertError` at
  startup, not mid-request — the server never partially starts with a broken
  remote configuration.
- **Subclass-friendly**: `_build_request_body` and `_parse_response` are clean
  extension points; `$VAR` expansion is inherited automatically by subclasses
  that call `super().load_weights()`.
