# Implementation & Testing Progress Report 1

Age Prediction Using Mixture of Experts (APMoE) — Team 21

| | Student Name | ID | Program |
| :---- | :---- | :---- | :---- |
| 1 | Ahmed Mostafa | 202201114 | DSAI |
| 2 | Seif Eldin Hisham Khashaba | 202200973 | DSAI |
| 3 | Ahmed Mohammed | 202201983 | HCI/SWD |
| 4 | Mohamed Abdelaziz | 202201794 | HCI/SWD |

Supervisors:

* Prof. Khaled Mostafa
* Prof. Doaa Shawky

Submission Date: May 2, 2026

---

# **1. Project Status Overview**

### **1.1 Overall Completion Status**

* **~95% completion toward MVP** — all planned MVP features are fully implemented and tested.
* Major milestones achieved:
  * **Complete CLI toolset** (`apmoe init`, `apmoe serve`, `apmoe predict`, `apmoe validate`) fully implemented and operational.
  * **Two trained expert models** integrated and shipping with the package:
    * `FaceAgeExpert` — Keras deep-learning regression model (MAE ≈ 11.8 years) on a 200×200 RGB face image, saved as `face_age_expert.keras` (~13 MB).
    * `KeystrokeAgeExpert` — ONNX logistic-regression classifier using 201 keystroke timing features, saved as `keystroke_age_expert.onnx` (~360 KB), with companion `keystroke_constants.json`.
  * **Full multimodal preprocessing pipelines** implemented as built-in strategy classes for both image and keystroke modalities, including cleaners and anonymizers.
  * **Complete HTTP serving layer** via FastAPI: `POST /predict`, `GET /health`, `GET /info`, Swagger UI at `/docs`, with CORS, rate-limiting, correlation-ID middleware.
  * **Three built-in aggregation strategies**: `WeightedAverageAggregator`, `ConfidenceWeightedAggregator`, `MedianAggregator`.
  * **Full IoC framework** with Pydantic v2 config validation, generic type-safe registries, and eight-step bootstrap.
  * **Comprehensive test suite**: 13 unit test files + 5 integration test files, covering all modules, interface boundaries, and full end-to-end pipelines with ≥80% line coverage enforced.
  * **`apmoe init --builtin`**: scaffolds a new project and copies bundled pretrained weights in a single command.

* Current items in progress:
  * ECG/EEG modality experts are still in the research/data-collection phase (not yet integrated).
  * Voice, fingerprint, and gait modalities remain at the research/experimentation stage.
  * Distributed / multi-GPU inference is not yet supported (single-node only).
  * **Fine-tuning support** — the framework architecture (ABCs, registries, config schema) is ready; an `apmoe finetune` CLI command and `POST /finetune` endpoint are being designed so integrators can supply labeled data and trigger retraining of the built-in experts without modifying framework code.
  * **Below-threshold confidence handling** — `Prediction.confidence` and `ExpertOutput.confidence` are already tracked per request; a `confidence_threshold` config key and a new `POST /predict` response field `recommendations` (actionable improvement hints when confidence falls below the threshold) are being designed.

### **1.2 Alignment with Final Design**

* **Is the architecture unchanged?**

  The overall system architecture underwent meaningful refinements over the original design plan:

  * The architecture follows a **modular pipeline with modality-specific preprocessing and expert models combined through a Mixture-of-Experts routing mechanism**, exactly as designed.
  * The framework is now a **fully installable Python package** (`pip install apmoe`) with a registered `apmoe` CLI entry point.

* **Changes from the original design — why:**

  Minor AI adjustments:

  * ECG model training was paused due to GPU availability; the framework architecture is ready to integrate it once weights are available.
  * Both face and keystroke models now produce per-expert metadata and gracefully degrade when a modality is absent.

  Architecture refinements (SWD):

  * **Patternification of the Pipeline**: Strategy Pattern for cleaners, anonymizers, and embedders; Factory Method for all object construction from config. The serving layer is now fully operational (not a stub).
  * **Decentralized Data Governance**: Anonymizer strategies are modality-specific (e.g., `ImageAnonymizer` is a pass-through as no PII leaks from pixel arrays; `KeystrokeAnonymizer` strips identifying metadata).
  * **"Dumb Pipes, Smart Endpoints"**: Expert plugins subscribe to the modalities they need. The pipeline gracefully skips experts whose declared modalities are not available and reports them in `Prediction.skipped_experts`.
  * **Three aggregation strategies** instead of one, selectable via config.

---

# **2. Implementation Progress**

Implementation work is organized around the main architectural components from the system design.

## **2.1 Component A — Data Processing & Feature Extraction Pipelines**

### **Purpose**

Transform raw modality inputs into normalized, model-ready representations through a strategy-based pipeline: `ModalityProcessor → CleanerStrategy → AnonymizerStrategy → (optional) EmbedderStrategy`.

### **Implemented Features**

**Image modality pipeline** (`src/apmoe/modality/builtin/image.py`, `src/apmoe/processing/builtin/image_cleaners.py`, `src/apmoe/processing/builtin/image_anonymizers.py`):

* `ImageProcessor` — accepts raw image bytes (PNG, JPEG, etc.) via Pillow; wraps them in a `ModalityData`.
* `ImageCleaner` — full preprocessing chain:
  1. Grayscale → RGB (stack channel 3×).
  2. RGBA → RGB (drop alpha channel).
  3. Resize to 200×200 with LANCZOS filter.
  4. Normalize: `/255.0` → `float32` array in `[0, 1]`, shape `(200, 200, 3)`.
* `ImageAnonymizer` — pass-through (pixel arrays carry no PII).

**Keystroke modality pipeline** (`src/apmoe/modality/builtin/keystroke.py`, `src/apmoe/processing/builtin/cleaners.py`, `src/apmoe/processing/builtin/anonymizers.py`):

* `KeystrokeProcessor` — accepts three input shapes:
  * List of triples `[[key1, key2, ms], ...]` (raw keystroke session).
  * Dict of lists `{"dur_8": [95, 102], ...}` (pre-computed features).
  * Raw IKDD text string.
* `KeystrokeCleaner` — filters outlier timings, computes per-feature mean timing vectors.
* `KeystrokeAnonymizer` — strips user/session identifiers from metadata.

**Technology Stack:**

* Python 3.11+, Pillow (image I/O and resizing), NumPy (array operations), ONNX Runtime (keystroke model), TensorFlow/Keras (face model).

**Config example** (`configs/image.json`, `configs/keystroke.json`, `configs/multimodal.json`):

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "image",
        "processor": "apmoe.modality.builtin.image.ImageProcessor",
        "pipeline": {
          "cleaner": "apmoe.processing.builtin.image_cleaners.ImageCleaner",
          "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
        }
      },
      {
        "name": "keystroke",
        "processor": "apmoe.modality.builtin.keystroke.KeystrokeProcessor",
        "pipeline": {
          "cleaner": "apmoe.processing.builtin.cleaners.KeystrokeCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.KeystrokeAnonymizer"
        }
      }
    ]
  }
}
```

> 📸 **Screenshot evidence for §2.1** — see [Appendix §7, Evidence E-01 – E-04](#7-screenshot-evidence-appendix).

---

## **2.2 Component B — Expert Model Implementations**

### **Purpose**

Specialized ML models ("experts") — each responsible for predicting age from a specific modality. Outputs are aggregated by the configured `AggregatorStrategy`.

### **Implemented Experts**

**`KeystrokeAgeExpert`** (`src/apmoe/experts/builtin.py`):

* ONNX logistic-regression model on **201 keystroke features** (hold-time `dur_*` and digraph flight-time `dig_*_*`).
* Predicts one of four **age groups**: `18-25`, `26-35`, `36-45`, `46+`.
* Continuous `predicted_age` is the **probability-weighted midpoint** of the predicted group.
* `confidence` = maximum class probability.
* Metadata includes `predicted_group`, `age_group_probs` (per-class), `features_observed_fraction`.
* Gracefully handles missing features by substituting training-set medians from `keystroke_constants.json`.
* Weight file: `weights/keystroke_age_expert.onnx` (~360 KB) + `weights/keystroke_constants.json`.
* Current performance: **~70% accuracy for age-group classification**.

**`FaceAgeExpert`** (`src/apmoe/experts/builtin.py`):

* Keras deep-learning regression model on **200×200 RGB float32 images**.
* Predicts a **continuous age** (years) — output is clamped to `[1, 120]`.
* `confidence = -1.0` (regressor has no calibrated class probabilities; aggregators handle this gracefully).
* Metadata includes `raw_output` and `rounded_age`.
* Weight file: `weights/face_age_expert.keras` (~13 MB).
* Current performance: **MAE ≈ 11.8 years**.

**Expert registry** (`src/apmoe/experts/registry.py`): Both experts are auto-registered via `@expert_registry.register("...")` decorators. Dotted-path resolution (`"apmoe.experts.builtin.FaceAgeExpert"`) allows users to reference custom experts in config without modifying framework code.

### **Technology Stack**

* TensorFlow 2.15+ / Keras — face age regression model.
* ONNX Runtime 1.17+ — keystroke age classification model.
* NumPy — feature vector construction and array operations.

> 📸 **Screenshot evidence for §2.2** — see [Appendix §7, Evidence E-05 – E-08](#7-screenshot-evidence-appendix).

---

## **2.3 Component C — Multimodal Framework (IoC Container & Pipeline)**

### **Purpose**

The full Mixture-of-Experts inference framework — routes multimodal inputs through independent expert models and combines their predictions into a unified age estimate. Follows **Inversion of Control (IoC)**: everything is wired from config; the framework owns the execution lifecycle entirely.

### **Implemented Features**

**Core Data Types & Exception Hierarchy** (`src/apmoe/core/types.py`, `src/apmoe/core/exceptions.py`):

* `ModalityData` — typed container for any modality's raw or preprocessed payload.
* `EmbeddingResult` — dense feature vector produced by an optional embedder step.
* `ExpertOutput` — single expert's age prediction, confidence score, consumed modalities, and metadata.
* `Prediction` — final aggregated result with per-expert breakdown, skipped experts, confidence interval, and pipeline latency metadata.
* Full exception hierarchy: `APMoEError → ConfigurationError, RegistryError, PipelineError, ModalityError, ExpertError, ServingError`.

**Configuration System** (`src/apmoe/core/config.py`):

* JSON-based configuration loader with **Pydantic v2 schema validation**.
* Cross-field validation (unique modality/expert names, expert modalities declared, port range, etc.).
* Environment variable overrides (`APMOE_SERVING_HOST`, `APMOE_SERVING_PORT`, `APMOE_SERVING_WORKERS`, `APMOE_SERVING_LOG_LEVEL`).
* Four ready-to-use config files: `configs/default.json`, `configs/image.json`, `configs/keystroke.json`, `configs/multimodal.json`.

**Generic Component Registry** (`src/apmoe/core/registry.py`):

* Type-safe `Registry[T]` supporting decorator-based registration (`@registry.register("name")`) and dotted-path importlib resolution (`"myproject.experts.MyExpert"`) for all six extension points.
* Per-extension-point global registries: `modality_registry`, `cleaner_registry`, `anonymizer_registry`, `embedder_registry`, `expert_registry`, `aggregator_registry`.

**Abstract Extension Points / ABCs** (`src/apmoe/modality/base.py`, `src/apmoe/processing/base.py`, `src/apmoe/experts/base.py`, `src/apmoe/aggregation/base.py`):

* `ModalityProcessor` — validates and preprocesses raw input into `ModalityData`.
* `CleanerStrategy` — removes noise and normalizes a `ModalityData`.
* `AnonymizerStrategy` — strips PII from a `ModalityData`.
* `EmbedderStrategy` (optional) — maps `ModalityData` to a dense `EmbeddingResult`.
* `ExpertPlugin` — declares required modalities, loads pretrained weights once at bootstrap, and runs inference per request.
* `AggregatorStrategy` — combines all `ExpertOutput` objects into a `Prediction`.

**Two-Phase Inference Pipeline** (`src/apmoe/core/pipeline.py`):

* `ModalityChain` dataclass — bundles `Processor + Cleaner + Anonymizer + optional Embedder` for a single modality.
* `InferencePipeline` — orchestrates the full execution loop:
  * **Phase A**: processes each configured modality independently through its chain (sequential in `run()`, concurrent via `asyncio.gather` in `run_async()`).
  * **Phase B**: dispatches processed outputs to matching experts, aggregates results, enriches `Prediction` with pipeline latency metadata.
* Graceful degradation: failed or missing modalities are excluded; only experts whose entire declared-modality set is available are run; skipped experts are reported in `Prediction.skipped_experts`.
* Observer hooks at four pipeline stages for logging, metrics, and tracing.

**APMoEApp IoC Container** (`src/apmoe/core/app.py`):

* `APMoEApp.from_config(path)` — eight-step bootstrap: loads config, resolves all component classes, instantiates processors/strategies, loads pretrained expert weights, wires the pipeline.
* `app.predict(inputs)` / `app.predict_async(inputs)` — single-call inference API.
* `app.validate()` — health check (weight file existence, expert liveness).
* `app.get_info()` — structured metadata for the `/info` endpoint.
* `app.serve()` — launches FastAPI + uvicorn server.

**Three Built-in Aggregation Strategies** (`src/apmoe/aggregation/builtin.py`):

* `WeightedAverageAggregator` — configurable per-expert weights from config; uniform fallback.
* `ConfidenceWeightedAggregator` — weights each expert by its own reported confidence.
* `MedianAggregator` — robust to outlier experts.

**Technology Stack:**

* Python 3.11+
* Pydantic v2 — configuration schema and validation.
* NumPy — modality data and embedding arrays.
* ONNX Runtime — keystroke model inference.
* TensorFlow/Keras — face model inference.
* FastAPI + uvicorn — HTTP serving layer.
* Click — CLI interface.
* pytest + pytest-asyncio + pytest-cov — test suite.
* uv — dependency management and test runner.
* ruff + mypy — linting and static type checking.
* hatchling — build backend; package ships pretrained weights as package data.

> 📸 **Screenshot evidence for §2.3** — see [Appendix §7, Evidence E-09 – E-13](#7-screenshot-evidence-appendix).

---

## **2.4 Component D — HTTP Serving Layer (FastAPI)**

### **Purpose**

Expose the APMoE inference pipeline as a production-ready REST API with middleware, error handling, and automatic OpenAPI documentation.

### **Implemented Features**

**Three endpoints** (`src/apmoe/serving/routes.py`):

* `POST /predict` — accepts a JSON body mapping modality names to raw session data; returns the aggregated `Prediction` with per-expert breakdown, skipped experts, confidence interval, and metadata.
* `GET /health` — readiness/liveness probe; returns `{"status": "healthy"|"degraded", "experts": {...}}`. Returns HTTP 503 if any expert is not loaded.
* `GET /info` — framework metadata: version, expert list, active modalities, aggregator info, serving config.
* `GET /docs` — Swagger UI (auto-generated by FastAPI).

**Middleware** (`src/apmoe/serving/middleware.py`):

* CORS — configurable allowed origins, headers, and methods.
* Rate limiting — per-client request throttling.
* Correlation ID — every request tagged with a UUID propagated in logs and response headers.
* Authentication hooks — placeholder for API key / JWT injection.

**App Factory** (`src/apmoe/serving/app_factory.py`): `create_api(apmoe_app)` wires the FastAPI application, includes all middleware, and attaches the router — keeping test and production setup identical.

> 📸 **Screenshot evidence for §2.4** — see [Appendix §7, Evidence E-14 – E-16](#7-screenshot-evidence-appendix).

---

## **2.5 Component E — CLI**

### **Purpose**

Developer experience toolchain: project scaffolding, config validation, inference, and serving — all from one `apmoe` command.

### **Implemented Commands** (`src/apmoe/cli/main.py`):

| Command | Description |
| :--- | :--- |
| `apmoe init [PROJECT_NAME]` | Scaffold a new project directory with `config.json`, custom stubs, and (with `--builtin`) bundled pretrained weights. |
| `apmoe validate --config <path>` | Validate JSON/Pydantic schema, resolve all component classes, check weight files, verify expert health. |
| `apmoe serve --config <path>` | Bootstrap `APMoEApp` and start FastAPI/uvicorn server. Supports `--host`, `--port`, `--workers`, `--log-level` overrides. |
| `apmoe predict --config <path> --input <path>` | Run inference on a directory (files named after modalities) or a JSON manifest file. Prints or writes the result JSON. |

> 📸 **Screenshot evidence for §2.5** — see [Appendix §7, Evidence E-17 – E-19](#7-screenshot-evidence-appendix).

---

## **2.6 Component F — Fine-Tuning Support** *(In Progress)*

### **Purpose**

Allow integrators who possess labeled domain data to fine-tune the built-in expert models — without modifying any framework source code — by supplying a dataset path and invoking a single CLI command or API call.

### **Current State**

The framework architecture already provides all the structural prerequisites:

* **`ExpertPlugin` ABC** exposes a `load_weights(path)` hook that is called at bootstrap. Adding a symmetric `save_weights(path)` hook and a `finetune(data_path, epochs, lr)` hook to the ABC gives every custom and built-in expert a standardized fine-tuning contract.
* **IoC config schema** can be extended with a `training` block per expert:

  ```json
  "experts": [
    {
      "name": "face_age_expert",
      "class": "apmoe.experts.builtin.FaceAgeExpert",
      "weights": "./weights/face_age_expert.keras",
      "training": {
        "data": "./data/faces/",
        "epochs": 10,
        "learning_rate": 1e-4
      }
    }
  ]
  ```

* **Registry pattern** means the fine-tuning workflow resolves the correct expert class dynamically — integrators swap in custom experts by changing a string in config, not framework code.

### **Planned DX surfaces**

| Surface | Command / Endpoint | Description |
| :--- | :--- | :--- |
| CLI | `apmoe finetune --config <path> --expert <name>` | Loads the named expert, reads its `training.data` config block, runs `expert.finetune()`, and saves updated weights to `training.output` path. |
| HTTP API | `POST /finetune` | Accepts a multipart body with `expert_name`, labeled data files, and optional hyperparameter overrides; returns job status and updated weight path. |
| Validation | `apmoe validate` extended | Will check that `training.data` paths exist when a `training` block is declared. |

### **Status**

`ExpertPlugin` ABC extension and config schema changes are designed; CLI and API implementation pending. Both `FaceAgeExpert` (Keras `model.fit()`) and `KeystrokeAgeExpert` (retrain-from-numpy → ONNX re-export) have identified training entry points.

> 📸 **Screenshot evidence for §2.6** — see [Appendix §7, Evidence E-20](#7-screenshot-evidence-appendix).

---

## **2.7 Component G — Below-Threshold Confidence Handling & Recommendations** *(In Progress)*

### **Purpose**

When the pipeline's aggregated confidence falls below a configurable threshold, automatically attach actionable improvement suggestions to the response — rather than silently returning a low-confidence prediction.

### **Current State**

The confidence data model is already in place:

* `ExpertOutput.confidence` — each expert reports its own confidence (or `-1.0` if not applicable, e.g. `FaceAgeExpert`).
* `Prediction.confidence` — the aggregator computes an aggregated confidence from all expert outputs.
* The `POST /predict` response currently returns `confidence` as a float.

### **Planned Behaviour**

A `confidence_threshold` key will be added to the `apmoe` config block:

```json
"apmoe": {
  "confidence_threshold": 0.60,
  ...
}
```

When `Prediction.confidence < confidence_threshold`, the pipeline populates a `recommendations` list in `Prediction.metadata` (and in the HTTP response body). Recommendations are rule-based and expert-specific:

| Trigger | Recommendation surfaced |
| :--- | :--- |
| `KeystrokeAgeExpert.features_observed_fraction < 0.5` | "Keystroke session too short — collect at least 50 keystrokes for reliable age inference." |
| `FaceAgeExpert` returns `confidence == -1.0` and only one expert ran | "Image-only prediction has no calibrated confidence. Add keystroke data for a multi-modal estimate." |
| All experts skipped (no modalities provided) | "No modalities were recognized. Provide at least one of: image (PNG/JPEG), keystroke (session JSON)." |
| Aggregated confidence below threshold | "Confidence below threshold ({value:.0%}). Suggestions: (1) supply additional modalities, (2) ensure image is well-lit and face is clearly visible, (3) extend keystroke session length, (4) consider fine-tuning expert weights on domain-specific data (`apmoe finetune`)." |

### **DX surfaces for recommendations**

| Surface | Detail |
| :--- | :--- |
| `POST /predict` response | `recommendations` field: list of strings present when `confidence < threshold`; empty list otherwise — zero breaking change. |
| `apmoe predict` CLI | Prints recommendations below the JSON result block when present, coloured yellow via Click. |
| `GET /info` | Exposes the configured `confidence_threshold` value so API consumers know the threshold in effect. |

### **Status**

Config schema extension designed. Per-expert recommendation rules specified. Pipeline integration (`InferencePipeline._maybe_add_recommendations()`) and CLI output formatting pending.

> 📸 **Screenshot evidence for §2.7** — see [Appendix §7, Evidence E-21](#7-screenshot-evidence-appendix).

---

# **3. Program-Specific Technical Evidence**

## **DSAI**

The DSAI sub-team focused on **data acquisition, model training, and integration of pretrained weights** into the framework.

**Trained Expert Models (integrated and shipped):**

| Modality | Model Type | Performance | Weight File |
| :--- | :--- | :--- | :--- |
| Face | Keras CNN Regression | MAE ≈ 11.8 years | `face_age_expert.keras` (~13 MB) |
| Keystroke | ONNX Logistic Regression | ~70% age-group accuracy | `keystroke_age_expert.onnx` (~360 KB) |

**Status of Additional Modalities:**

| Modality | Current Status | Notes |
| :--- | :--- | :--- |
| Face | ✅ Integrated & Shipped | Keras model integrated with full preprocessing pipeline |
| Keystroke | ✅ Integrated & Shipped | ONNX model with 201 features and companion constants file |
| ECG | Architecture prepared | Training paused — GPU availability constraints |
| EEG | Preprocessing completed | Model experimentation planned |
| Voice | Research stage | Dataset and architecture explored |
| Fingerprint | Research stage | Feasibility analysis complete |
| Gait/Mobility | Research stage | Dataset exploration |

**Integration contracts:**

Both expert implementations follow the `ExpertPlugin` ABC exactly. DSAI preprocessing outputs are now formally aligned with the framework's `ModalityData` and `ProcessedInput` types (previously SW-01 blocker — now resolved).

## **SWD**

**Implemented architectural layers:**

* ✅ **Core data/type layer** — `ModalityData`, `EmbeddingResult`, `ExpertOutput`, `Prediction`, full exception hierarchy.
* ✅ **Configuration layer** — Pydantic v2 schema, JSON loader, env-var overrides.
* ✅ **Registry layer** — type-safe generic `Registry[T]` with six per-extension-point singletons.
* ✅ **Abstraction layer (ABCs)** — all 6 ABCs: `ModalityProcessor`, `CleanerStrategy`, `AnonymizerStrategy`, `EmbedderStrategy`, `ExpertPlugin`, `AggregatorStrategy`.
* ✅ **Orchestration layer** — `ModalityChain`, `InferencePipeline` (sync + async), graceful degradation, observer hooks.
* ✅ **IoC Container** — `APMoEApp.from_config()`, eight-step bootstrap, `predict` / `predict_async` / `validate` / `get_info` / `serve`.
* ✅ **Built-in implementations layer** — `ImageProcessor`, `KeystrokeProcessor`, `ImageCleaner`, `KeystrokeCleaner`, `ImageAnonymizer`, `KeystrokeAnonymizer`, `FaceAgeExpert`, `KeystrokeAgeExpert`, three aggregators.
* ✅ **Serving layer** — FastAPI routes, middleware (CORS, rate limiting, correlation ID), app factory, uvicorn launch.
* ✅ **CLI layer** — `init`, `serve`, `predict`, `validate` fully wired.
* 🔄 **Fine-tuning layer** *(in progress)* — `ExpertPlugin` ABC extension, `apmoe finetune` CLI command, `POST /finetune` endpoint.
* 🔄 **Confidence-threshold & recommendations layer** *(in progress)* — config key, `Prediction.metadata["recommendations"]`, CLI output formatting.

**Design patterns in use:**

* Inversion of Control — governs the entire framework.
* Strategy Pattern — all four swappable algorithm slots (cleaner, anonymizer, embedder, aggregator).
* Factory Method — all object construction from config.
* Chain of Responsibility — per-modality processing steps.
* Registry / Service Locator — component resolution by name.
* Observer — hook system at four pipeline stages.

### **DX Differentiators — Concrete Mechanisms**

The following are the specific, verifiable developer-experience features that distinguish APMoE from generic inference frameworks. Each point names the exact mechanism and where it lives in the codebase.

| # | DX Feature | Concrete Mechanism |
| :--- | :--- | :--- |
| 1 | **Single-command project bootstrap with real working models** | `apmoe init my_project --builtin` creates a directory, writes a valid `config.json`, copies the pretrained `face_age_expert.keras` (~13 MB) and `keystroke_age_expert.onnx` + `keystroke_constants.json` from `src/apmoe/weights/` (shipped as package data via `pyproject.toml` `artifacts`), and writes six typed stub files. The project runs `apmoe serve` immediately — no additional download, no dataset, no training. |
| 2 | **Zero-code expert injection via dotted class paths** | The `"class"` field in `config.json` is any importable Python class string, e.g. `"myproject.experts.MyExpert"`. `APMoEApp.from_config()` resolves it via `importlib.import_module` + `getattr` in the IoC bootstrap step — the integrator adds a Python file and a JSON string; no framework source is modified and no plugin registration API is called at import time. |
| 3 | **Pydantic v2 config validation with field-level error messages** | `load_config(path)` runs a Pydantic v2 `model_validate` on the parsed JSON. Cross-field validators enforce: unique modality names, unique expert names, every expert's `modalities` list references a declared modality, and port in `[1, 65535]`. Errors name the exact JSON path that is wrong (e.g. `apmoe.experts[1].modalities[0]: 'eeg' is not declared`), so integrators fix config without reading source code. |
| 4 | **`apmoe validate` — three-pass pre-flight check** | Pass 1: Pydantic schema (as above). Pass 2: all `"class"` strings are importlib-resolved and imported — catches typos before serving starts. Pass 3: `app.validate()` calls `ExpertRegistry.health_check()` — each expert reports `is_loaded` and weight file existence is verified. Exit code 1 with coloured error messages if any check fails; exit 0 with green ✓ expert health table on success. |
| 5 | **Auto-generated REST API with zero configuration** | `apmoe serve --config config.json` calls `create_api(apmoe_app)` which mounts a FastAPI `APIRouter` with `POST /predict`, `GET /health`, `GET /info`, and FastAPI's built-in Swagger UI at `GET /docs`. OpenAPI schema is auto-generated from Python type hints — integrators get interactive API documentation without writing a single OpenAPI YAML line. |
| 6 | **Graceful multi-modality degradation surfaced in the API response** | When a modality is absent from a `POST /predict` body, `InferencePipeline` marks its chain as skipped. Experts whose declared modality set is not fully satisfied are listed in `Prediction.skipped_experts` and returned in the JSON response. API consumers know exactly which experts ran and which were bypassed — no silent failures, no 500 errors. |
| 7 | **Async-first inference with transparent sync fallback** | `app.predict_async(inputs)` uses `asyncio.gather` to process all modality chains concurrently. `app.predict(inputs)` calls the same path via `asyncio.run()` so CLI consumers and unit tests get identical behaviour with no async boilerplate. The concurrency model is declared in one method; switching between sync and async requires no architecture change. |
| 8 | **Config-selectable aggregation strategy** | Three aggregators (`WeightedAverageAggregator`, `ConfidenceWeightedAggregator`, `MedianAggregator`) are already registered. Switching strategy requires changing one JSON string: `"strategy": "apmoe.aggregation.builtin.MedianAggregator"`. Per-expert weights are an optional second JSON key. No code change, no recompilation. |
| 9 | **Observer hooks for zero-coupling observability** | `InferencePipeline` fires `on_before_process`, `on_after_embed`, `on_after_expert`, `on_after_aggregate` callbacks at four stages. Integrators attach loggers, metrics exporters, or tracing spans by registering a callback — they never subclass the pipeline or modify inference logic. |
| 10 | **`apmoe predict` — file-system inference without a running server** | `apmoe predict --config config.json --input data/` discovers files whose stem matches a configured modality name (e.g. `image.jpg` → `image` modality) and runs the full pipeline locally. A JSON manifest mode (`--input manifest.json`) maps arbitrary file paths to modality names. Result JSON is printed to stdout or written to `--output path`. Developers validate models and data pipelines on a laptop without standing up HTTP infrastructure. |
| 11 | **Fine-tuning via config block** *(in progress)* | A `training` block in `config.json` per expert (`data`, `epochs`, `learning_rate`, `output`) will allow `apmoe finetune --expert face_age_expert` to re-train and re-save the model with integrator-supplied data. No Python code required; the ABC's `finetune()` hook does the work. |
| 12 | **Below-threshold confidence recommendations in API response** *(in progress)* | A `confidence_threshold` config key will trigger automatic population of `Prediction.metadata["recommendations"]` with expert-specific, actionable strings (e.g. "Extend keystroke session", "Supply both image and keystroke modalities") when confidence drops below threshold — surfaced in `POST /predict` JSON and `apmoe predict` CLI output. |

---

# **4. Testing Summary**

## **4.1 Testing Methods Conducted**

**Unit Tests** (13 files, `tests/unit/`):

* `test_types.py` — all core data types and exception hierarchy.
* `test_config.py` — Pydantic schema, cross-field rules, env-var overrides.
* `test_registry.py` — Registry generic, decorator and dotted-path registration.
* `test_modality.py` — `ModalityProcessor` ABC enforcement and built-in implementations.
* `test_processing.py` — `CleanerStrategy`, `AnonymizerStrategy`, built-in strategies.
* `test_experts.py` — `ExpertPlugin` ABC, `KeystrokeAgeExpert`, `FaceAgeExpert` (with mock models/ONNX sessions).
* `test_aggregation.py` — all three aggregator strategies edge-case coverage.
* `test_aggregation_builtin.py` — confidence handling, median computation.
* `test_pipeline.py` — `ModalityChain`, `InferencePipeline` sync/async, graceful degradation, observer hooks.
* `test_image_modality.py` — `ImageProcessor`, `ImageCleaner`, `ImageAnonymizer` full processing chain.
* `test_serving.py` — FastAPI routes (predict, health, info), middleware, error responses.
* `test_cli.py` — all four CLI commands, scaffolding output, config validation.
* Coverage enforced: **≥80% line coverage per CI run** via `pytest-cov`.

**Integration Tests** (5 files, `tests/integration/`):

* `test_module_boundaries.py` — 7 cross-module interface contracts (registry → instantiation, processing chain, metadata preservation).
* `test_app.py` — full `APMoEApp.from_config()` bootstrap: single-modality, multi-modality, optional embedder, graceful degradation, all error paths.
* `test_face_e2e.py` — full image pipeline end-to-end: `bytes → ImageProcessor → ImageCleaner → ImageAnonymizer → FaceAgeExpert (mock Keras) → Prediction`. Tests RGB, grayscale, RGBA, large images, age clamping.
* `test_keystroke_e2e.py` — full keystroke pipeline end-to-end: raw session → `KeystrokeProcessor → KeystrokeCleaner → KeystrokeAnonymizer → KeystrokeAgeExpert (mock ONNX) → Prediction`. Tests feature coverage, median fill-in, group predictions.
* `test_init_project_extensions.py` — `apmoe init` scaffolding completeness, `--builtin` weight copying, custom stub correctness.

**Manual Tests:**

* Manual inspection of `apmoe init --builtin` + `apmoe validate` + `apmoe serve` flow.
* Swagger UI `/docs` verified with live multimodal inference requests.
* Observer hook traces confirmed data flows correctly between all pipeline stages.

## **4.2 Test Evidence Summary**

| Test Area | Files | Key Verifications |
| :--- | :--- | :--- |
| Core types | `test_types.py` | Dataclass fields, exception hierarchy, ABC enforcement |
| Config | `test_config.py` | Schema validation, cross-field rules, env-var overrides |
| Registry | `test_registry.py` | Decorator & dotted-path registration, type safety |
| Pipeline | `test_pipeline.py` | Sync/async execution, graceful degradation, observer hooks |
| Serving | `test_serving.py` | `/predict`, `/health`, `/info` routes, middleware, error codes |
| CLI | `test_cli.py` | `init`, `serve`, `predict`, `validate` command correctness |
| Face E2E | `test_face_e2e.py` | RGB/RGBA/Grayscale images, age clamping, batch dim |
| Keystroke E2E | `test_keystroke_e2e.py` | Feature vectors, median fill, group prediction |
| App Bootstrap | `test_app.py` | Multi-modality, degradation, all error paths |
| Module Boundaries | `test_module_boundaries.py` | 7 interface contracts across all module layers |

> 📸 **Screenshot evidence for §4** — see [Appendix §7, Evidence E-22 – E-26](#7-screenshot-evidence-appendix).

**Model performance:**

| Model | Metric | Result |
| :--- | :--- | :--- |
| `FaceAgeExpert` | Age regression MAE | ≈ 11.8 years |
| `KeystrokeAgeExpert` | Age-group classification accuracy | ~70% |

## **4.3 Issues and Bugs Identified**

| Issue ID | Description | Severity | Status | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| DS-01 | Limited GPU resources halted ECG/EEG model training | High | Ongoing | Continue with available resources; framework is ready to integrate once weights exist |
| DS-02 | Dataset imbalance in some modalities | Low | Under review | Data augmentation and balancing techniques being explored |
| DS-03 | Multi-modal fusion strategies require tuning | Medium | Mitigated | Three aggregation strategies available; WeightedAverage default works well |
| SW-01 | API contract between DSAI preprocessing and framework types not formally aligned | High | **Resolved** | Both expert implementations now fully conform to `ExpertPlugin` ABC and exchange `ModalityData` / `ProcessedInput` types |
| SW-02 | ONNX export off-by-one feature count | Low | **Resolved** | `KeystrokeAgeExpert.load_weights()` silently trims the extra column when `len(feature_cols) == expected + 1` |
| SW-03 | Fine-tuning not yet supported — integrators with labeled data cannot improve models without modifying source | Medium | **In Progress** | `ExpertPlugin` ABC `finetune()` hook and `apmoe finetune` CLI command designed; implementation pending |
| SW-04 | No below-threshold confidence handling — low-confidence predictions are returned without guidance | Medium | **In Progress** | `confidence_threshold` config key and `Prediction.metadata["recommendations"]` designed; pipeline integration pending |

---

# **5. MVP Delivery Status**

All core MVP deliverables are complete. Two additional features (fine-tuning and confidence recommendations) are in progress for the extended MVP:

| Feature | Status | Evidence |
| :--- | :--- | :--- |
| **At least 2–3 trained expert models** | ✅ Complete | `FaceAgeExpert` (Keras, MAE≈11.8yr) + `KeystrokeAgeExpert` (ONNX, 70% accuracy) |
| **Working preprocessing pipelines** | ✅ Complete | Image and keystroke full pipelines (processor → cleaner → anonymizer) |
| **Model aggregation mechanism** | ✅ Complete | 3 strategies: `WeightedAverage`, `ConfidenceWeighted`, `Median` |
| **Functional CLI** | ✅ Complete | `apmoe init/validate/serve/predict` — all commands operational |
| **Functional HTTP API** | ✅ Complete | `POST /predict`, `GET /health`, `GET /info`, Swagger UI at `/docs` |
| **Project scaffolding** | ✅ Complete | `apmoe init --builtin` copies bundled weights in one command |
| **Middleware & DX** | ✅ Complete | CORS, rate limiting, correlation ID, auth hooks, OpenAPI docs, 12-point DX differentiators |
| **Fine-tuning support** | 🔄 In Progress | ABC hook designed, CLI command designed; implementation pending (Component F) |
| **Below-threshold confidence & recommendations** | 🔄 In Progress | Config key, response field, and per-expert rules designed; pipeline integration pending (Component G) |

**Live Demo Readiness:**

```bash
pip install -e ".[dev]"
apmoe init live_demo --builtin
cd live_demo
apmoe validate --config config.json
apmoe serve --config config.json
# → http://127.0.0.1:8000/docs
```

**Next steps:**

| Task | Owner | Risk | Notes |
| :--- | :--- | :--- | :--- |
| Implement `ExpertPlugin.finetune()` ABC hook + `save_weights()` | Software Team | Low | Architecture fully designed; see Component F |
| Implement `apmoe finetune` CLI command | Software Team | Low | Extends existing Click group pattern |
| Implement `POST /finetune` HTTP endpoint | Software Team | Low | Extends existing FastAPI router pattern |
| Implement `confidence_threshold` config key + `recommendations` field | Software Team | Low | Architecture fully designed; see Component G |
| Resume ECG baseline training | DSAI Team | GPU availability | Framework ready to integrate once weights exist |
| Add EEG expert plugin | DSAI Team | Medium | Preprocessing pipeline complete |
| Explore additional modalities (voice, gait, fingerprint) | DSAI Team | Low | Research stage; no blockers on SWD side |
| Distributed inference support | Software Team | Low | Single-node inference is the current limitation |
| Add JWT/API key authentication middleware | Software Team | Low | Auth hook placeholder exists in serving layer |

---

# **6. Individual Contribution Summary**

| Member | Implemented Tasks | % Overall Contribution |
| :--- | :--- | :--- |
| Ahmed Mostafa | Keystroke dataset preparation, `KeystrokeAgeExpert` ONNX model training and integration, `keystroke_constants.json` export pipeline | 25% |
| Seif Eldin Khashaba | Face model training (Keras CNN regression, MAE≈11.8yr), `FaceAgeExpert` integration, image preprocessing pipeline | 25% |
| Ahmed Mohammed | Framework architecture design (`APMoEApp`, `InferencePipeline`, ABCs, registries), CLI (`init`, `serve`, `predict`, `validate`), integration testing | 25% |
| Mohamed Abdelaziz | System integration design, FastAPI serving layer (routes, middleware, app factory), unit testing (all 13 files), Pydantic config schema | 25% |

---

# **7. Screenshot Evidence Appendix**

Every claim in this report is backed by a specific file and line range that should be opened and screenshotted. All paths are relative to the repository root (`APMoE/`).

## **§2.1 — Data Processing & Feature Extraction Pipelines**

### E-01 · `ImageProcessor` — class declaration and `preprocess()` method

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/modality/builtin/image.py` |
| **Lines** | 41 – 119 |
| **Shows** | `@modality_registry.register("image")` decorator, `validate()` and `preprocess()` methods, `ModalityData` construction with `width`/`height`/`mode` metadata |

### E-02 · `ImageCleaner` — four-step preprocessing pipeline

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/processing/builtin/image_cleaners.py` |
| **Lines** | 31 – 129 |
| **Shows** | `@cleaner_registry.register("image_cleaner")`, grayscale→RGB (L. 101-103), RGBA→RGB (L. 106-108), 200×200 LANCZOS resize (L. 111-113), `/255.0 float32` normalization (L. 116) |

### E-03 · `KeystrokeProcessor` — multi-shape input acceptance

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/modality/builtin/keystroke.py` |
| **Lines** | 1 – 80 |
| **Shows** | `@modality_registry.register("keystroke")`, `preprocess()` accepting triples list, feature dict, and raw IKDD text |

### E-04 · Config files showing pipeline wiring

| Field | Value |
| :--- | :--- |
| **File** | `configs/multimodal.json` |
| **Lines** | 1 – end |
| **Shows** | JSON config wiring `ImageProcessor`, `ImageCleaner`, `ImageAnonymizer`, `KeystrokeProcessor`, `KeystrokeCleaner`, `KeystrokeAnonymizer` |

---

## **§2.2 — Expert Model Implementations**

### E-05 · `KeystrokeAgeExpert` — class declaration, `load_weights()`, feature vector construction

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/experts/builtin.py` |
| **Lines** | 57 – 200 |
| **Shows** | `@expert_registry.register("keystroke_age_expert")`, ONNX session loading (L. 143-144), constants file loading (L. 165-176), `_build_feature_vector()` (L. 308-332) |

### E-06 · `KeystrokeAgeExpert` — `predict()` and age-group-to-continuous mapping

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/experts/builtin.py` |
| **Lines** | 200 – 355 |
| **Shows** | ONNX `session.run()` (L. 238-246), probability-weighted midpoint age formula (L. 256-258), `ExpertOutput` construction with group probs (L. 264-277) |

### E-07 · `FaceAgeExpert` — class declaration, `load_weights()`, Keras inference

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/experts/builtin.py` |
| **Lines** | 362 – 527 |
| **Shows** | `@expert_registry.register("face_age_expert")`, `tf.keras.models.load_model()` (L. 449), `np.expand_dims` batch dim (L. 501), age clamp `[1, 120]` (L. 514) |

### E-08 · Shipped weight files

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/weights/` (directory listing) |
| **Shows** | `face_age_expert.keras` (~13 MB), `keystroke_age_expert.onnx` (~360 KB), `keystroke_constants.json` — the three bundled model files |

---

## **§2.3 — Multimodal Framework (IoC Container & Pipeline)**

### E-09 · Core data types — `ModalityData`, `EmbeddingResult`, `ExpertOutput`, `Prediction`

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/core/types.py` |
| **Lines** | 26 – 196 |
| **Shows** | All four `@dataclass` definitions; `ExpertOutput.confidence` sentinel `-1.0` validation (L. 146-155); `Prediction.skipped_experts` field (L. 181) |

### E-10 · Exception hierarchy — `APMoEError` and all five sub-types

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/core/exceptions.py` |
| **Lines** | 1 – 90 |
| **Shows** | `APMoEError` base (L. 11-28), `ConfigurationError` (L. 31), `RegistryError` (L. 42), `PipelineError` (L. 52), `ModalityError` (L. 62), `ExpertError` (L. 72), `ServingError` (L. 82) |

### E-11 · Pydantic v2 config schema — cross-field validators

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/core/config.py` |
| **Lines** | 184 – 345 |
| **Shows** | `APMoEConfig` with `@model_validator` for undeclared expert modalities (L. 199-210), unique modality names (L. 212-222), unique expert names (L. 224-234); `load_config()` with env-var override logic (L. 295-344) |

### E-12 · Generic `Registry[T]` — decorator registration and dotted-path importlib resolution

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/core/registry.py` |
| **Lines** | 69 – 220 |
| **Shows** | `class Registry(Generic[T])` (L. 69), `register()` decorator (L. 99-123), `resolve()` dotted-path via `importlib.import_module` + `getattr` (L. 172-220) |

### E-13 · `InferencePipeline` — two-phase loop, observer hooks, graceful degradation

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/core/pipeline.py` |
| **Lines** | 127 – 501 |
| **Shows** | `ModalityChain` dataclass (L. 72-97), hook type aliases (L. 104-119), `_process_one_modality()` 7-step chain (L. 178-275), `_phase_a_sync()` graceful degradation (L. 277-313), `_phase_b()` expert dispatch + aggregation (L. 315-405), `run()` sync (L. 411-442), `run_async()` `asyncio.gather` (L. 444-500) |

---

## **§2.4 — HTTP Serving Layer**

### E-14 · `POST /predict` route — JSON parsing, inference call, response serialization

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/serving/routes.py` |
| **Lines** | 100 – 212 |
| **Shows** | `@router.post("/predict")`, JSON body parsing, `await apmoe_app.predict_async(inputs)`, HTTP 422/503/500 error handling |

### E-15 · `GET /health` and `GET /info` routes

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/serving/routes.py` |
| **Lines** | 214 – 290 |
| **Shows** | `GET /health` with `expert_registry.health_check()` and HTTP 503 on degraded state (L. 218-259); `GET /info` delegating to `apmoe_app.get_info()` (L. 265-289) |

### E-16 · `ExpertPlugin` ABC — four abstract methods

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/experts/base.py` |
| **Lines** | 57 – 201 |
| **Shows** | `class ExpertPlugin(ABC)`, `@abstractmethod name` (L. 82-94), `declared_modalities()` (L. 96-116), `load_weights()` (L. 118-136), `predict()` (L. 138-167), optional `get_info()` and `is_loaded` (L. 169-201) |

---

## **§2.5 — CLI**

### E-17 · `apmoe init` — scaffolding command with `--builtin` weight copy

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/cli/main.py` |
| **Lines** | 336 – 436 |
| **Shows** | `@cli.command("init")`, `--builtin` flag (L. 339-341), directory creation + weight copy loop (L. 372-388), all stub file writes (L. 392-431), `apmoe validate` / `apmoe serve` next-step hints |

### E-18 · `apmoe serve` — bootstrap + uvicorn launch

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/cli/main.py` |
| **Lines** | 443 – 534 |
| **Shows** | `@cli.command("serve")`, env-var override logic (L. 503-510), `APMoEApp.from_config(config)` (L. 513), `app.serve()` (L. 531) |

### E-19 · `apmoe validate` — three-pass pre-flight check

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/cli/main.py` |
| **Lines** | 699 – 757 |
| **Shows** | `@cli.command("validate")`, schema bootstrap (L. 727), `app.validate()` expert health call (L. 735), per-expert loaded/NOT LOADED coloured output (L. 747-754) |

---

## **§2.6 — Fine-Tuning Support (In Progress)**

### E-20 · `ExpertPlugin` ABC — `load_weights()` hook (existing entry point for fine-tuning)

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/experts/base.py` |
| **Lines** | 118 – 136 |
| **Shows** | `@abstractmethod load_weights(self, path: str)` — the lifecycle hook that will be extended with a symmetric `finetune()` + `save_weights()` to enable integrator-controlled retraining |

---

## **§2.7 — Below-Threshold Confidence Handling (In Progress)**

### E-21 · `ExpertOutput.confidence` and `Prediction.confidence` — existing confidence data model

| Field | Value |
| :--- | :--- |
| **File** | `src/apmoe/core/types.py` |
| **Lines** | 124 – 196 |
| **Shows** | `ExpertOutput.confidence` field with `-1.0` sentinel (L. 143-155); `Prediction.confidence` in `[0, 1]` (L. 177-196); `Prediction.metadata` dict (L. 182) — the field that will carry `"recommendations"` |

---

## **§4 — Testing**

### E-22 · Coverage enforcement — `pyproject.toml` `fail_under = 80`

| Field | Value |
| :--- | :--- |
| **File** | `pyproject.toml` |
| **Lines** | 86 – 100 |
| **Shows** | `[tool.pytest.ini_options]` with `asyncio_mode = "auto"` (L. 90), `[tool.coverage.report]` with `fail_under = 80` (L. 99) |

### E-23 · Face E2E test — full image pipeline with mock Keras model

| Field | Value |
| :--- | :--- |
| **File** | `tests/integration/test_face_e2e.py` |
| **Lines** | 1 – 184 |
| **Shows** | `_build_mock_expert()` (L. 45-51), `_run_pipeline()` 4-stage chain (L. 62-87), `test_rgb_image_full_pipeline()` (L. 89-97), age clamping tests (L. 114-124), batch dim verification (L. 126-140) |

### E-24 · Module boundary tests — 7 cross-module interface contracts

| Field | Value |
| :--- | :--- |
| **File** | `tests/integration/test_module_boundaries.py` |
| **Lines** | 1 – 50 |
| **Shows** | Test class names and the 7 interface contracts being validated across registry → instantiation → processing chain → metadata |

### E-25 · Unit test inventory — all 13 unit test files

| Field | Value |
| :--- | :--- |
| **File** | `tests/unit/` (directory listing) |
| **Shows** | `test_types.py`, `test_config.py`, `test_registry.py`, `test_modality.py`, `test_processing.py`, `test_experts.py`, `test_aggregation.py`, `test_aggregation_builtin.py`, `test_pipeline.py`, `test_image_modality.py`, `test_serving.py`, `test_cli.py` |

### E-26 · `pyproject.toml` — full dependency list and entry point

| Field | Value |
| :--- | :--- |
| **File** | `pyproject.toml` |
| **Lines** | 1 – 53 |
| **Shows** | `[project.dependencies]` (pydantic, fastapi, uvicorn, click, numpy, torch, onnxruntime, pillow, tensorflow), `[project.scripts] apmoe = "apmoe.cli.main:cli"` (L. 46), `artifacts = ["src/apmoe/weights/**"]` for bundled model shipping (L. 52) |