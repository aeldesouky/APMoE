# APMoE — Final Software Maturity Checklist

> **This is a hard tracking document. Every item must be answered honestly.**
> Status legend:
> - ✅ **DONE** — fully implemented, evidence exists
> - 🔄 **IN PROGRESS** — partially implemented, specific gap noted
> - ❌ **NOT YET** — not implemented, must be resolved before demo/submission

---

## 1. Final Gate *(Must pass ALL before demo)*

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 1.1 | Can your system be deployed and accessed live right now? | ✅ | `pip install -e . && apmoe init my_project --builtin && apmoe serve -c config.json` starts FastAPI at `http://0.0.0.0:8000`. `GET /docs` is immediately accessible. |
| 1.2 | Can it handle multiple concurrent users without crashing? | 🔄 | FastAPI is async; `run_async()` uses `asyncio.gather` for parallel modality processing. Multiple uvicorn workers are configurable. **BUT:** no load test has been run. No RPS, p95 latency, or concurrency numbers exist. This is a gap. |
| 1.3 | Can a new user set it up from scratch in a few minutes? | ✅ | `apmoe init live_demo --builtin` + `apmoe serve -c config.json`. CLI scaffolds a working project with bundled weights in one command. README documents the flow. |
| 1.4 | Can you explain every major design and implementation decision? | ✅ | IoC / GoF patterns, two-phase pipeline, registry-based plugin resolution, Pydantic v2 config, graceful degradation — all documented in the progress report with file/line references. |

**Gate status: 🔄 BLOCKED on 1.2 — load testing must be run before this gate passes.**

---

## 2. Architecture & Design

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 2.1 | What architecture did you choose and why? | ✅ | **Monolith with plugin IoC** — a single deployable Python process with a FastAPI HTTP layer. Chosen because: (a) inference is CPU/GPU-bound single-node; (b) microservices would add network latency with no scaling benefit at current scale; (c) IoC + registry gives extensibility without the operational overhead of distributed services. |
| 2.2 | What trade-offs did you consider? | ✅ | Monolith: simpler ops, no service discovery, but no horizontal expert isolation. In-process rate limiter: simple, but not shared across uvicorn workers (documented in `middleware.py` L. 149–155). Single-node inference: no GPU parallelism between experts. |
| 2.3 | What are the main components and the responsibility of each? | ✅ | (1) `ModalityProcessor` — raw input → `ModalityData`. (2) `CleanerStrategy` — normalize. (3) `AnonymizerStrategy` — strip PII. (4) `EmbedderStrategy` — optional dense encoding. (5) `ExpertPlugin` — ML inference. (6) `AggregatorStrategy` — combine outputs. (7) `InferencePipeline` — orchestrate. (8) `APMoEApp` — IoC container. (9) FastAPI layer — HTTP serving. (10) CLI — DX toolchain. |
| 2.4 | Where can your system fail and how did you handle that? | ✅ | Six-layer exception hierarchy (`APMoEError → ConfigurationError, RegistryError, PipelineError, ModalityError, ExpertError, ServingError`). Graceful degradation: failed modalities are excluded and reported in `Prediction.skipped_experts`; experts whose modalities are missing are skipped, not crashed. HTTP layer maps exceptions to 422/503/500. |

---

## 3. Functional Completeness

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 3.1 | Are all core features implemented and working end-to-end? | ✅ | CLI (`init`, `validate`, `serve`, `predict`), HTTP API (`/predict`, `/health`, `/info`, `/docs`), two expert models (Keras face, ONNX keystroke), three aggregation strategies, full preprocessing pipelines for both modalities. |
| 3.2 | What are the main user flows and are they fully tested? | ✅ | Flow 1: `apmoe init --builtin` → `apmoe validate` → `apmoe serve` → `POST /predict` → JSON response. Flow 2: `apmoe predict --input data/` → CLI JSON output. Both covered by `test_cli.py`, `test_face_e2e.py`, `test_keystroke_e2e.py`, `test_app.py`. |
| 3.3 | How do you handle invalid input and edge cases? | ✅ | Pydantic v2 rejects malformed config at load time with field-level errors. `ImageCleaner` rejects images smaller than 4px. `KeystrokeAgeExpert` fills missing features with training-set medians. HTTP `/predict` returns 422 on non-JSON body, 503 if no experts run. Age output is clamped to `[1, 120]`. |
| 3.4 | Fine-tuning (integrator-supplied data) | ❌ | **NOT YET.** `ExpertPlugin` ABC has no `finetune()` hook. `apmoe finetune` CLI command does not exist. `POST /finetune` endpoint does not exist. Architecture is designed; implementation is pending. |
| 3.5 | Below-threshold confidence + recommendations | ❌ | **NOT YET.** `confidence_threshold` config key does not exist. `Prediction.metadata["recommendations"]` is not populated. Per-expert recommendation rules are specified but not coded. |

---

## 4. Testing

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 4.1 | What types of tests did you implement? | ✅ | **Unit** (13 files, `tests/unit/`): isolated module tests. **Integration** (5 files, `tests/integration/`): cross-module boundary tests and full E2E pipeline tests (with mock models). **Manual**: `apmoe validate` + Swagger UI verified. |
| 4.2 | What critical scenarios are covered? | ✅ | Config schema violations, dotted-path resolution failures, registry duplicate registration, graceful degradation (missing modality), age clamping, ONNX off-by-one feature count, HTTP 422/503/500 error paths, CLI scaffolding completeness. |
| 4.3 | What happens when something fails — are failure cases tested? | ✅ | Yes. `test_pipeline.py` tests degraded-modality paths. `test_experts.py` tests `ExpertError` on unloaded model. `test_serving.py` tests 503 health degraded and 422 bad request. `test_app.py` tests all bootstrap error paths. |
| 4.4 | Coverage enforcement | ✅ | `pyproject.toml` `fail_under = 80`. `pytest-cov` enforced in CI. |

---

## 4b. Performance & Load Testing

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 4b.1 | How many users can your system handle (RPS / concurrency)? | ❌ | **NOT YET.** No load test has been run. No RPS or concurrency number exists. Must run `locust` or `wrk` and document results before demo. |
| 4b.2 | What are your average and p95 latency? | ❌ | **NOT YET.** `Prediction.metadata["pipeline_latency_s"]` captures per-request pipeline time, but no aggregate statistics exist across many requests. |
| 4b.3 | Where is the performance bottleneck, and why? | 🔄 | **Known but unmeasured.** The bottleneck is model inference: `tf.keras.model.predict()` (FaceAgeExpert) and `onnxruntime.InferenceSession.run()` (KeystrokeAgeExpert) are CPU-bound with no GPU at current deployment. No profiling has been done to quantify this. |
| 4b.4 | Did you optimize anything? Before vs after? | 🔄 | `run_async()` uses `asyncio.gather` to process modality chains concurrently (vs sequential in `run()`). No before/after benchmarks have been recorded. |

**This section is the most critical gap before demo.**

---

## 5. Observability

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 5.1 | How do you know your system is healthy? | ✅ | `GET /health` queries `ExpertRegistry.health_check()` — each expert reports `is_loaded`. Returns `{"status": "healthy"/"degraded", "experts": {...}}`. HTTP 503 on degraded. |
| 5.2 | What metrics do you collect? | 🔄 | **Partial.** Per-request latency: `Prediction.metadata["pipeline_latency_s"]`. Correlation ID: every request tagged, returned in `X-Correlation-ID` header. Structured JSON logs per request via `RequestLoggingMiddleware` (method, path, status, duration_ms, client_host). **Missing:** no Prometheus/Grafana, no error-rate counter, no throughput metric. |
| 5.3 | How do you debug a failure? | 🔄 | **Partial.** Correlation ID (`X-Correlation-ID` response header) ties logs to requests. Structured log record includes `status_code` and `duration_ms`. `ExpertError` / `PipelineError` include `context` dicts in their `str()` output. **Missing:** no distributed tracing, no log aggregation system. |
| 5.4 | Do you have logs that help trace a request? | ✅ | Yes. `RequestLoggingMiddleware` emits a structured JSON log per request. Error/warning paths log with correlation ID prefix `[{correlation_id}]`. `apmoe.pipeline` logger logs skipped experts and failed modalities. |

---

## 6. Deployment & DevOps

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 6.1 | How do you deploy your system? (Docker / Compose / K8s) | ❌ | **NOT YET.** No `Dockerfile`, no `docker-compose.yml`, no K8s manifests exist. Current deployment is `pip install -e . && apmoe serve`. |
| 6.2 | Can your system run with one command? | 🔄 | **Partial.** After install: `apmoe init my_project --builtin && cd my_project && apmoe serve -c config.json` is three commands. No `docker run ...` or `make run` single-command option exists. |
| 6.3 | How do you manage configuration across environments? | ✅ | Environment variable overrides: `APMOE_SERVING_HOST`, `APMOE_SERVING_PORT`, `APMOE_SERVING_WORKERS`, `APMOE_SERVING_LOG_LEVEL`, `APMOE_SERVING_RATE_LIMIT` override JSON config at startup. Four ready-made config files (`default.json`, `image.json`, `keystroke.json`, `multimodal.json`). |
| 6.4 | If deployment fails, how do you rollback? | ❌ | **NOT YET.** No versioning strategy, no rollback procedure, no blue/green or canary setup. `apmoe validate` can detect a broken config before serving starts, which is a partial mitigation. |

---

## 7. Security

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 7.1 | How do you authenticate users? | 🔄 | **Hook exists, not wired.** `AuthPlugin` ABC and `AuthMiddleware` are implemented in `middleware.py` (L. 214–324). An `ApiKeyAuth` example is in the docstring. **No concrete `AuthPlugin` is instantiated by default** — the framework ships open (no auth enforced). |
| 7.2 | How do you authorize access (who can do what)? | ❌ | **NOT YET.** No RBAC or scope-based authorization. `AuthPlugin.authenticate()` is binary (allow/deny). No per-endpoint permission model exists. |
| 7.3 | How do you prevent invalid or malicious input? | ✅ | Pydantic v2 rejects invalid config. `ImageProcessor` rejects non-image bytes. `KeystrokeProcessor` rejects malformed session data. HTTP 422 on non-JSON or non-object body. Age output clamped to `[1, 120]`. Rate limiting (HTTP 429) prevents flood attacks. |
| 7.4 | Where are your secrets stored and how are they protected? | ❌ | **NOT YET.** No secrets management. Model weight paths are in plaintext JSON config. No `.env` file handling, no Vault, no secret injection. If an API key auth is added, it would currently be hardcoded or in plaintext env vars. |

---

## 8. API Design

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 8.1 | Do you have documented APIs (Swagger / OpenAPI)? | ✅ | Yes. FastAPI auto-generates OpenAPI schema from Python type hints. Swagger UI is live at `GET /docs` when serving. ReDoc at `GET /redoc`. |
| 8.2 | Are your APIs consistent in structure and error handling? | ✅ | All three endpoints return plain dicts (JSON serialized). Errors always use `{"detail": "..."}` via `HTTPException`. Status codes are consistent: 200 OK, 422 Unprocessable Entity, 503 Service Unavailable, 500 Internal Server Error. |
| 8.3 | How would you handle API changes (versioning)? | ❌ | **NOT YET.** No API versioning strategy. No `/v1/` prefix. No `Deprecation` header mechanism. A breaking change to the `/predict` request body would break existing clients immediately. |

---

## 9. Data & Persistence

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 9.1 | How is your data structured? Why this design? | ✅ | Stateless inference pipeline. Data flows as typed `@dataclass` objects: `ModalityData → ProcessedInput → ExpertOutput → Prediction`. No persistent state is held between requests. Dataclasses chosen for immutability, type safety, and zero-overhead serialization. |
| 9.2 | How do you ensure data consistency and correctness? | ✅ | `ModalityData.with_data()` returns a new instance (immutable transform). `ExpertOutput.__post_init__` validates confidence in `[0,1]` or `-1.0`. `Prediction.__post_init__` validates confidence and confidence interval. Pydantic validates all config at load time. |
| 9.3 | What happens if a transaction fails midway? | ✅ | Inference is stateless — there is no transaction. Failed modality processing: the modality is excluded; dependent experts are skipped and listed in `Prediction.skipped_experts`. The pipeline returns a partial `Prediction` rather than crashing, unless zero experts can run (then `PipelineError` is raised). |

---

## 10. Scalability & Reliability

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 10.1 | Can your system scale? How? | 🔄 | **Vertical only today.** Uvicorn supports multiple worker processes (`serving.workers` config). `asyncio.gather` processes modality chains concurrently per request. **Horizontal scaling** (multiple machines behind a load balancer) is architecturally possible (the pipeline is stateless) but not deployed or tested. |
| 10.2 | Is your system stateless where needed? | ✅ | Yes. The inference pipeline holds no request state. Expert models are loaded once at bootstrap (read-only weights). Each `predict()` call is fully independent. |
| 10.3 | What happens under high load or partial failure? | 🔄 | Rate limiting returns HTTP 429 under flood. Graceful degradation handles missing modalities. **BUT:** no circuit breaker, no backpressure mechanism, no request queue. Under sustained overload the server will queue requests until workers exhaust memory. |
| 10.4 | Do you implement retries, timeouts, or fallback mechanisms? | ❌ | **NOT YET.** No request timeout on `/predict`. No retry logic for failed experts. No fallback to a simpler model when the primary expert fails. `ExpertError` is raised immediately and propagated as HTTP 500. |

---

## 11. Engineering Understanding

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 11.1 | Can you explain how each part works internally? | ✅ | All six extension-point ABCs, the two-phase pipeline, IoC bootstrap, Pydantic v2 schema, registry dotted-path resolution, `asyncio.gather` parallel processing, and middleware chain are all documented with file/line references in the progress report. |
| 11.2 | Did you use any code you don't fully understand? | ✅ | No black-box code. Third-party libraries used are: Pydantic v2 (schema validation), FastAPI/Starlette (ASGI routing and middleware), uvicorn (ASGI server), ONNX Runtime (model inference), TensorFlow/Keras (model loading and inference), Pillow (image decoding). All are production libraries with documented APIs. |
| 11.3 | If something breaks, can you debug it yourself? | ✅ | Yes. Correlation ID traces each request through logs. Exception hierarchy provides structured `context` dicts. `apmoe validate` isolates bootstrap failures before serving starts. |

---

## 12. Reproducibility & Documentation

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 12.1 | Does your README include clear setup and run steps? | 🔄 | `README.md` exists. `apmoe init`-generated projects get a `README.md` with quick-start steps. **The root repository README** needs to be verified for completeness (install from source, dev setup, run tests). |
| 12.2 | Can someone run your system without asking you questions? | 🔄 | `apmoe init --builtin && apmoe serve` is self-contained **if the user already has Python 3.11+ and the package installed**. There is no Docker image, so environment setup (Python version, dependency install) requires reading the README. |
| 12.3 | Did you document architecture and APIs? | ✅ | Architecture documented in the progress report (Implementation & Testing Report 1) with component responsibilities, design patterns, and file/line references. APIs documented via Swagger at `/docs`. |

---

## 13. Team Contribution

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 13.1 | What did each team member contribute? | ✅ | Documented in progress report §6: Ahmed Mostafa (keystroke model), Seif Eldin Khashaba (face model), Ahmed Mohammed (framework architecture + CLI), Mohamed Abdelaziz (serving layer + unit tests). |
| 13.2 | Does your Git history clearly show individual work? | ❌ | **UNKNOWN — must verify.** Run `git log --oneline --author="..."` for each member to confirm commits are attributable. If history is squashed or all committed by one person, this is a risk. |
| 13.3 | How did you coordinate and review work? | ❌ | **NOT documented.** No pull request review process, branching strategy, or coordination log is referenced in the report. Must be prepared to answer this verbally or add a short note. |

---

## 14. User Experience

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 14.1 | Are your user flows clear and logical? | ✅ | CLI flow is sequential and guided: `init` → `validate` → `serve` → `predict`. Each command prints coloured next-step hints. Swagger UI provides interactive API exploration. |
| 14.2 | Are there any broken or confusing interactions? | 🔄 | **Not fully verified.** CLI has been manually tested. The apmoe-website (separate repo) has been built and tested. **No formal usability evaluation has been conducted.** |
| 14.3 | Does your system solve a real user problem? | ✅ | Yes. Multimodal age prediction from face images and keystroke dynamics, served as an accessible REST API with a no-code setup path for integrators. |

---

## 15. Cost Awareness

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 15.1 | If deployed in cloud, what would it cost to run? | ❌ | **NOT YET calculated.** No cloud cost estimate exists. This must be prepared before demo: estimate for a small VM (e.g., AWS `t3.medium` for CPU inference, or `g4dn.xlarge` for GPU inference) + storage for model weights (~13.5 MB total). |
| 15.2 | What components are the most expensive? | 🔄 | **Known but unquantified.** GPU inference (TensorFlow/Keras for FaceAgeExpert) is the most expensive component if GPU is required. Without GPU, CPU inference on `t3.medium` is likely $30–60/month for low traffic. No actual measurement. |

---

## 16. SLA & Production Metrics

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 16.1 | What is your target availability? | ❌ | **NOT DEFINED.** No SLA has been stated. Must define (e.g., 99.5% uptime = <4h downtime/month). |
| 16.2 | What is acceptable latency? | ❌ | **NOT DEFINED.** No p95 latency target exists. Must define (e.g., "95% of `/predict` requests complete within 2 seconds on CPU"). |
| 16.3 | How do you handle downtime or degradation? | 🔄 | **Partial.** `GET /health` returns HTTP 503 when an expert is not loaded — allowing a load balancer to remove the instance. `Prediction.skipped_experts` enables partial-modality graceful degradation. **No auto-restart, no health-check-driven restart, no alerting.** |

---

## 17. Demo Readiness

| # | Question | Status | Evidence / Gap |
| :--- | :--- | :---: | :--- |
| 17.1 | Can you demonstrate a live system? | ✅ | Yes. `apmoe init live_demo --builtin && cd live_demo && apmoe serve -c config.json` starts the server. `http://127.0.0.1:8000/docs` opens Swagger UI immediately. |
| 17.2 | Can you demonstrate the full user flow? | ✅ | Yes. `apmoe init` → `apmoe validate` → `apmoe serve` → Swagger UI `POST /predict` with keystroke JSON → prediction response with per-expert breakdown. |
| 17.3 | Can you demonstrate load testing results? | ❌ | **NOT YET.** No load test has been run. Must run `locust` or `wrk` before demo and capture RPS, p95 latency, and error rate under concurrent load. |
| 17.4 | Can you demonstrate monitoring/logs? | 🔄 | **Partial.** Structured JSON logs are emitted to stdout (visible in terminal during `apmoe serve`). `X-Correlation-ID` in response headers. **No dashboard, no log viewer, no Grafana.** Can show terminal logs during demo. |
| 17.5 | Can you simulate a failure and explain it? | ✅ | Yes. Remove a weight file → `apmoe validate` shows `NOT LOADED`. Start server with broken config → HTTP 503 on `GET /health`. Send request with wrong body → HTTP 422. All can be live-demonstrated. |

---

## Final Question: Production Failure Playbook

| Scenario | Answer | Status |
| :--- | :--- | :---: |
| **How do you detect a failure?** | `GET /health` returns `{"status": "degraded"}` with HTTP 503. Request logging emits error-level JSON records with the correlation ID. | ✅ |
| **How do you debug it?** | (1) Check the `X-Correlation-ID` from the failing request. (2) Grep logs for that ID — the structured log includes method, path, status, duration, and client IP. (3) The `ExpertError` / `PipelineError` `context` dict in the log body identifies the exact failing component and its weight path. (4) `apmoe validate --config config.json` re-runs the health check in isolation to pinpoint the issue. | ✅ |
| **How do you fix it?** | For a missing/corrupt weight file: replace the file and restart (`apmoe serve`). For a bad config: fix the JSON and `apmoe validate` to confirm. For an expert crash: check the expert's `ExpertError` context for the failing input shape, then fix the preprocessing upstream. **No hot-reload exists — a restart is required.** | 🔄 |

---

## Summary: What Must Be Done Before Demo

> These are the items that will fail under scrutiny. Do not go to the demo without addressing them.

| Priority | Item | Owner | Effort |
| :--- | :--- | :--- | :--- |
| 🔴 **CRITICAL** | Run load test — capture RPS, p95 latency, concurrency limit | Software Team | 2–3 hours |
| 🔴 **CRITICAL** | Define SLA targets (availability %, latency target) | Both Teams | 30 min |
| 🔴 **CRITICAL** | Verify Git history shows individual contributions per member | All Members | 1 hour |
| 🟠 **HIGH** | Write a `Dockerfile` and confirm `docker run ...` works | Software Team | 2–3 hours |
| 🟠 **HIGH** | Calculate cloud cost estimate (AWS/GCP VM + storage) | Software Team | 1 hour |
| 🟠 **HIGH** | Instantiate a concrete `AuthPlugin` (API key auth) by default | Software Team | 2 hours |
| 🟠 **HIGH** | Add API versioning (`/v1/` prefix or `Accept-Version` header) | Software Team | 2 hours |
| 🟡 **MEDIUM** | Implement `confidence_threshold` config key + `recommendations` field | Software Team | 4–6 hours |
| 🟡 **MEDIUM** | Implement `apmoe finetune` CLI command + `ExpertPlugin.finetune()` hook | Software Team | 1–2 days |
| 🟡 **MEDIUM** | Add request timeout on `/predict` endpoint | Software Team | 1 hour |
| 🟡 **MEDIUM** | Document team coordination (branching strategy, PR process) | All Members | 30 min |
| 🟢 **LOW** | Add retries/fallback for failed expert inference | Software Team | 3–4 hours |
| 🟢 **LOW** | Add Prometheus metrics endpoint (`/metrics`) | Software Team | 2–3 hours |
| 🟢 **LOW** | Secrets management (`.env` + `python-dotenv` at minimum) | Software Team | 1 hour |
