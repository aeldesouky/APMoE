# Implementation & Testing Progress Report 2 (Week 12\)

This template must be followed strictly. All sections require evidence-based reporting.

* **Course:** CSAI 499 – Senior Project   
* **Deliverable:** Implementation & Testing Progress Report 2   
* **Team Number:**  21  
* **Team Members (Name – Program):** Ahmed Mostafa 202201114 \- DSAI,  Ahmed Mohammed 202201983 \- HCI, Mohamed Abdelaziz 202201794 \- HCI, Seif Eldin 202200973 \- DSAI  
* **Supervisor:** Dr. Doaa Shawky, Dr. Khaled Mostafa  
* **Date:** 02 May

# **1\. Project Status & Progress**

Provide:  
\- Brief summary of progress since MVP (5–6 lines max)  
\- Estimated % completion: 80%  
\- Key milestones achieved:

* **Complete CLI toolset** (`apmoe init`, `apmoe serve`, `apmoe predict`, `apmoe validate`) fully implemented and operational.  
  * **Two trained expert models** integrated and shipping with the package:  
    * `FaceAgeExpert` — Keras deep-learning regression model (MAE ≈ 11.8 years) on a 200×200 RGB face image, saved as `face_age_expert.keras` (\~13 MB).  
    * `KeystrokeAgeExpert` — ONNX logistic-regression classifier using 201 keystroke timing features, saved as `keystroke_age_expert.onnx` (\~360 KB), with companion `keystroke_constants.json`.  
  * **Full multimodal preprocessing pipelines** implemented as built-in strategy classes for both image and keystroke modalities, including cleaners and anonymizers.  
  * **Complete HTTP serving layer** via FastAPI: `POST /predict`, `GET /health`, `GET /info`, Swagger UI at `/docs`, with CORS, rate-limiting, correlation-ID middleware.  
  * **Three built-in aggregation strategies**: `WeightedAverageAggregator`, `ConfidenceWeightedAggregator`, `MedianAggregator`.  
  * **Full IoC framework** with Pydantic v2 config validation, generic type-safe registries, and eight-step bootstrap.  
  * **Comprehensive test suite**: 13 unit test files \+ 5 integration test files, covering all modules, interface boundaries, and full end-to-end pipelines with ≥80% line coverage enforced.  
  * **`apmoe init --builtin`**: scaffolds a new project and copies bundled pretrained weights in a single command.  
* Current items in progress:

  * ECG modality experts are still in the research/data-collection phase (not yet integrated).  
  * Voice, fingerprint, and gait modalities remain at the research/experimentation stage.  
  * Distributed / multi-GPU inference is not yet supported (single-node only).  
  * **Fine-tuning support** — the framework architecture (ABCs, registries, config schema) is ready; an `apmoe finetune` CLI command and `POST /finetune` endpoint are being designed so integrators can supply labeled data and trigger retraining of the built-in experts without modifying framework code.  
  * **Structured Errors** — enhancing the `APMoEError` base class to explicitly mandate `cause` and `recovery_suggestion` fields.
  * **Mocking Mode** — `--mock` dry-run mode for CI/CD environments.
  * **DX Enhancements & Usability Testing** — conducting formal Developer Experience (DX) usability testing sessions with external engineers to evaluate framework intuitiveness. To achieve a "perfect", zero-friction onboarding journey, we are developing interactive Jupyter notebook tutorials, auto-generated API client SDKs (Python/Node), and a visual web-based sandbox playground.

**Evidence (REQUIRED):**  
\- Codebase architecture and feature implementations are available in the repository (`src/apmoe`).
\- `pytest --cov` execution logs confirming >80% test coverage.
\- *(Please insert screenshots of the CLI running and the FastAPI Swagger UI here)*

**Risks / Delays:**  
\- **Hardware Limitations:** Lack of dedicated GPUs is currently preventing the training of deeper neural networks for new modalities (like ECG), causing us to pivot to traditional ML baselines until computational resources are secured.

# **2\. Feature Completion**

Fill the table with ONLY completed features. Evidence is mandatory.

| Feature | Description | Owner (Name \+ Program) | Status | Evidence |
| :---- | :---- | :---- | :---- | :---- |
| CLI Toolset Engine | Implemented a robust `click`-based CLI scaffolding engine providing `init`, `serve`, `predict`, and `validate` commands. Includes automatic generation of template configs and stubs. | Mohamed Abdelaziz (HCI) | Done | `src/apmoe/cli/main.py:L20-L150` |
| HTTP Serving Layer | Built a production-ready ASGI application using `FastAPI`. Includes a `POST /predict` endpoint, `GET /health`, correlation-ID middleware, and rate-limiting. | Mohamed Abdelaziz (HCI) | Done | `src/apmoe/serving/routes.py:L30-L120` |
| IoC Inference Pipeline | Designed the core framework architecture using Gang of Four patterns (Strategy, Factory). Includes a two-phase async orchestrator capable of concurrent modality processing. | Ahmed Mohammed (HCI) | Done | `src/apmoe/core/pipeline.py:L40-L200` |
| Config & Registry | Implemented strict Pydantic v2 validation for `config.json` with cross-field schema checks. Created type-safe decorator-based registries for dynamic component resolution. | Ahmed Mohammed (HCI) | Done | `src/apmoe/core/config.py:L1-L150` |
| FaceAge CNN Expert | Integrated a 13MB Keras deep learning regression model. Validates 200x200 RGB image inputs and predicts continuous age with an MAE of ~11.8 years. | Seif Eldin Khashaba (DSAI) | Done | `src/apmoe/experts/builtin.py:L110-L180` |
| Keystroke ONNX Expert | Integrated a lightweight 360KB ONNX logistic-regression model. Processes 201 keystroke temporal features to predict age groups with ~70% classification accuracy. | Ahmed Mostafa (DSAI) | Done | `src/apmoe/experts/builtin.py:L400-L460` |
| Multimodal Preprocessing | Developed ModalityProcessor implementations. Built custom CleanerStrategy and AnonymizerStrategy classes to strip PII and normalize raw signals before inference. | Ahmed Mostafa & Seif Eldin | Done | `src/apmoe/processing/builtin/` |
| Result Aggregation | Implemented multiple fusion strategies (`WeightedAverageAggregator`, `ConfidenceWeightedAggregator`, `MedianAggregator`) to combine outputs into a single prediction. | Ahmed Mohammed & Seif Eldin | Done | `src/apmoe/aggregation/builtin.py:L20-L90` |
| Below-Threshold Guidance | Engine that evaluates sub-threshold confidence scores and injects human-readable quality improvement recommendations directly into the `Prediction` metadata. | Ahmed Mohammed (HCI) | Done | `src/apmoe/core/pipeline.py:L179-L260` |
| Advanced Debug Mode | Implemented `--log-level debug` CLI switch, detailed pipeline state tracking via `logging`, and exposed `skipped_experts` in prediction metadata for deep inspection. | Mohamed Abdelaziz (HCI) | Done | `src/apmoe/cli/main.py`, `pipeline.py` |

# **3\. Remaining Tasks & Planning**

All tasks must have owner, priority, and deadline.

| Task | Description | Owner | Priority | Deadline | Status |
| :---- | :---- | :---- | :---- | :---- | :---- |
| ECG Modality Expert | Collect data and train ECG model | Seif Eldin Khashaba | High | Week 13 | In Progress |
| Production SLAs & Economics | Define SLA targets, estimate AWS/GCP cloud hosting costs, and expose Prometheus `/metrics`. | Both Teams | Critical | Week 15 | Planned |
| Enterprise Security & Reliability | Enforce default `AuthPlugin` (API Key), add `/predict` request timeouts, and implement fallback/retries. | Software Team | High | Week 14 | Planned |
| Continuous Model Adaptation | Implement the `apmoe finetune` CLI command and the `ExpertPlugin.finetune()` hook. | Software Team | Medium | Week 14 | Planned |

# **4\. Testing & Quality Evidence**

## **4.1 Test Coverage**

| Component | Test Type | Coverage % | Notes |
| :---- | :---- | :---- | :---- |
| Core Framework | Unit | >85% | Covers Pipeline, Config, Registry (`tests/core/`) |
| Serving Layer | Unit/Integration | >80% | Covers API endpoints, ASGI factory (`tests/serving/`) |
| CLI | Unit/Integration | >80% | Validates scaffolding and commands (`tests/cli/`) |
| Experts | Unit | >80% | Mocks ONNX/Keras weight loading (`tests/experts/`) |
| Processing | Unit | >85% | Covers cleaners and anonymizers (`tests/processing/`) |

## **4.2 Testing Depth**

Our testing strategy evaluates the framework across deep integration paths. Specifically:  
\- **Edge cases tested**: 
  - Empty or malformed JSON payloads sent to the `POST /predict` endpoint.
  - Configuration files (`config.json`) with missing required fields or conflicting settings (e.g., duplicate plugin names).
  - Component registry lookup failures when an unregistered class is referenced via a dotted path.
  - Modality payloads that omit required fields specific to a branch (e.g., missing base64 string for the image processor).
\- **Failure scenarios**: 
  - Complete failure of an expert model during the `predict()` execution phase due to tensor shape mismatches.
  - Missing or corrupted `.onnx` and `.keras` weight files during the application bootstrap sequence.
  - Uvicorn server startup failures when the assigned port (e.g., 8000) is already bound by another process.
  - Validation errors causing all input modalities to be dropped, triggering the Graceful Degradation fallback.
\- **Example inputs/outputs**: 
  - *Input*: A `POST /predict` request containing a valid `image` payload (base64 string) and a `keystroke` payload (array of 201 floats).
  - *Output*: A strongly-typed `PredictionResponse` JSON object containing the aggregated `age` prediction, individual `ExpertOutput` objects, confidence scores, a list of any `skipped_experts`, and embedded performance telemetry via `metadata.pipeline_latency_s`.

## **4.3 Bugs & Issues**

| Bug ID | Description | Severity | Status | Fix |
| :---- | :---- | :---- | :---- | :---- |
| DS-01 | Lack of dedicated GPU instances halted the training of deeper, more complex neural networks for biometric modalities. | High | Ongoing | Transitioned to utilizing baseline ML architectures (e.g., ONNX Logistic Regression) to ensure framework integration could proceed. |
| SW-01 | A structural flaw in the early orchestrator caused the entire pipeline to crash and return a 500 error if a single modality failed preprocessing. | High | Fixed | Refactored the `InferencePipeline` to implement a Graceful Degradation pattern, allowing healthy modalities to continue. |
| CLI-01 | The `apmoe init` command generated boilerplate configuration files that failed the framework's own Pydantic strict schema validation. | High | Fixed | Updated the CLI scaffolding logic to serialize directly from the Pydantic schema models. |

## **4.4 Fix Validation**

All framework-level fixes undergo a strict Test-Driven Development (TDD) validation cycle. For instance, when resolving the Graceful Degradation bug (SW-01), the Software Team wrote an isolated unit test where a `MockModalityProcessor` was injected and configured to intentionally raise a `ModalityError`. The orchestrator's output was then asserted to ensure that:
1. The server did not crash.
2. The remaining healthy modalities were dispatched successfully.
3. The crashed modality was correctly appended to the `failed_modalities` metadata array.

Only after passing these local unit tests was the fix merged into the main branch, ensuring the before-behavior (server crash) was permanently resolved into the after-behavior (graceful continuation).

## **4.5 Performance & Load Testing**

To ensure the system meets production requirements, we conducted local load testing against the FastAPI `POST /predict` endpoint using the custom asynchronous `scripts/load_test_predict.py` testing suite. The test simulated multiple concurrent users sending simultaneous, valid keystroke inference requests.

**Test Environment:**
- Local execution on a single node.
- Concurrent virtual users: 50
- Duration: 15 seconds

**Results:**
- **Peak Throughput:** ~2,094 Requests Per Second (RPS)
- **Average Latency:** ~24 ms
- **P95 Latency:** ~27 ms
- **P99 Latency:** ~30 ms
- **Error Rate:** 0.0% (31,853 successful requests)

**Observation:** The high throughput (>2k RPS) and low latency (<30ms P99) demonstrate that the `KeystrokeAgeExpert` (ONNX) inference path is extremely lightweight and that the async IoC orchestrator adds virtually zero overhead. 

Evidence for this testing is located in the `scripts/` directory, which contains the `load_test_predict.py` testing script.

# **5\. Individual Technical Contribution (MANDATORY)**

| Student | Program | Contribution | Evidence | % Contribution |
| :---- | :---- | :---- | :---- | :---- |
| Ahmed Mostafa | DSAI | **Data Pipeline & ML:** Led data extraction for keystrokes. Trained the ONNX Logistic Regression model and integrated it by authoring the `KeystrokeAgeExpert` plugin and `KeystrokeCleaner` strategy. | `src/apmoe/experts/builtin.py:L400-L460`, `processing/builtin/keystroke_cleaners.py` | 25% |
| Seif Eldin Khashaba | DSAI | **Vision & Signals ML:** Developed facial image pipeline and trained the CNN model in Keras. Integrated the `FaceAgeExpert` plugin and `ImageCleaner` strategy. Leading R&D for ECG modality. | `src/apmoe/experts/builtin.py:L110-L180`, `processing/builtin/image_cleaners.py` | 25% |
| Ahmed Mohammed | HCI/SWD | **Architecture & Orchestration:** Designed the core Inversion of Control (IoC) framework. Authored the Pydantic v2 configuration system, component `Registry`, and async `InferencePipeline` orchestrator. | `src/apmoe/core/pipeline.py:L40-L200`, `src/apmoe/core/registry.py:L1-L100` | 25% |
| Mohamed Abdelaziz | HCI/SWD | **API, CLI & DX:** Developed the Developer Experience layer. Built the HTTP `FastAPI` serving architecture with OpenAPI docs. Created the `Click` CLI toolset (`init`, `serve`) and observability metrics. | `src/apmoe/serving/routes.py:L30-L120`, `src/apmoe/cli/main.py:L20-L150` | 25% |

Each student must provide verifiable technical evidence aligned with their program:  
\- SWD: code, architecture, APIs  
\- IT: deployment, infrastructure, security  
\- DSAI: data pipeline, model, evaluation

# **6\. Report Quality**

\- Clear structure  
\- Professional formatting  
\- Logical flow  
\- No grammatical errors