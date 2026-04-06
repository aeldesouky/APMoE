# MVP Deliverables

This document maps the APMoE framework's features and implementation status to the requirements specified in the MVP deliverables.

## 1. MVP Features Checklist

Status uses **fully implemented** for behavior that matches the stated design in code and docs today, and **partially implemented** when the core path works but behavior is optional, stubbed, or still being hardened.

| Feature area | Status | Evidence / gaps |
| :--- | :--- | :--- |
| **CLI & generated projects** | Fully implemented | Scaffold, validate, predict, serve; generated config, stubs, optional bundled weights. |
| **Inversion of Control & configuration** | Fully implemented | One validated JSON config wires components by path/name; no core edits required. |
| **Multimodal pipeline (built-ins + extensibility)** | Fully implemented | Built-in modalities work out of the box; swap or extend every stage via config and IoC. |
| **Mixture-of-experts inference & aggregation** | Partially implemented | MoE + default aggregator shipped; uneven confidence semantics; experts run sequentially. |
| **HTTP API & interactive docs** | Partially implemented | FastAPI, JSON prediction body, health/info, OpenAPI UI; no multipart in MVP. |
| **CORS, authentication & rate limiting** | Partially implemented (stubs / optional) | CORS/auth are wiring stubs, not full hardening; rate limit opt-in; correlation logging on. |
| **Developer documentation (primary DX artifact)** | Fully implemented | `docs/dev/`, config and integration notes, DX overview—authoritative for extension. |
| **Other project & user-facing docs** | To be updated for final delivery | Root README and narrative copy deferred to final project polish. |
| **Error handling & diagnostics (DX)** | Fully implemented | Typed errors, fast config validation, clear CLI/HTTP errors, request correlation. |

## 2. Demo Video Plan (2–3 minutes)

**1. Problem Context (0:00 - 0:20)**
* Briefly introduce the need for a multimodal Age Prediction Mixture of Experts (APMoE) system that handles both visual and keystroke data securely.

**2. System Running & User Workflow (0:20 - 1:00)**
* Run `apmoe init demo_project --builtin`.
* Inspect the generated `config.json` and default `weights/`.
* Run `apmoe validate --config config.json` to prove system health.

**3. Integration & Key Features (1:00 - 2:00)**
* Start the server with `apmoe serve --config config.json`.
* Open the browser to `/docs` (Swagger UI) to show the dashboard.
* Submit a multimodal request (image + keystroke JSON) via the Swagger UI.

**4. Output Display (2:00 - 2:30)**
* Highlight the prediction response containing aggregated age, confidence, and per-expert breakdowns.

## 3. Two-Slide MVP Summary

### Slide 1: MVP Overview
* **What the MVP does**: APMoE provides an extensible, multimodal Mixture of Experts framework for age prediction using visual and keystroke data.
* **Key Implemented Features**:
  * Complete CLI toolset (`init`, `validate`, `serve`, `predict`) for Developer Experience (DX).
  * Extremely Extensible Pipeline Architecture (Processors, Cleaners, Anonymizers, Embedders).
  * Deep Inversion of Control (IoC) allowing completely custom logic via dotted configuration strings.
  * Built-in Keras (Face) and ONNX (Keystroke) models for Age Prediction out-of-the-box (`--builtin`).
  * Auto-generated REST API featuring pre-built Rate Limiting, Logging, Auth bindings, and Swagger endpoints.
  * Hierarchical, resilient Error System (`APMoEError` subclasses).
* **System Architecture Snapshot**: 
  * JSON Config -> IoC Registries -> Pipeline Strategy (Modality Processors) -> Extensible Cleaner/Anonymization steps -> Expert Dispatcher -> Custom Aggregator -> Configurable FastAPI Serving Layer.

### Slide 2: Learning & Contributions
* **Key Technical Challenges**:
  * Bridging fundamentally different modalities (discrete keystrokes vs continuous image data) into a uniform processing pipeline.
  * Developing strict `Inversion of Control (IoC)` schemas allowing end-users (developers) to insert their logic securely via Python strings and configuration without editing framework core code.
  * Ensuring zero-coupling across pipeline boundaries by exchanging tightly-specified Data Classes (`ModalityData`, `ProcessedInput`, `Prediction`).
* **Team Learning**: 
  * Advanced Python architecture standards: Dependency Injection, the Factory/Strategy patterns, and Dynamic Registries.
  * Robust generic framework construction including deep error hierarchy handling (`APMoEError`, `PipelineError`, `ConfigurationError`).
  * Asynchronous web API provisioning via FastAPI while abstracting developer involvement away from Web sockets (CORS, Rate Limiting Middleware).
* **Responsibilities**: *(To be filled with actual team member names per your roster)*
  * *Member 1*: Core Framework Architecture & IoC Container.
  * *Member 2*: CLI, Tools, and Scaffolding (`apmoe init/validate`).
  * *Member 3*: ML Pipeline & Built-in Experts (Keras/ONNX integration).
  * *Member 4*: FastAPI Serving Layer & OpenAPI spec generation.

## 4. Live Demo Readiness

**Local Execution Steps:**
1. Clone / Install environment using `pip install -e ".[dev]"`.
2. `apmoe init live_demo --builtin`
3. `cd live_demo`
4. `apmoe serve -c config.json`
5. Test using REST client or Web UI (`http://127.0.0.1:8000/docs`).

**Key topics to prepare for:**
* **Architecture Decisions**: Why pure Inversion of Control (IoC) with JSON binding was chosen in favor of hard-coded structures, utilizing a dynamic registry pattern, isolating data definitions into pure container structures like `ModalityData`, and robust error hierarchies (`ModalityError`, `ExpertError`, `ServingError`).
* **System Boundaries**: How developers customize the tool via subclasses: Processors (data extraction), Cleaners (preparation), Anonymizers (privacy gating), Embedders (feature engineering), Experts (Inference/MoE), and Aggregators (confidence consensus).
* **Limitations**: The framework currently computes inference sequentially on single-node environments (no distributed GPU parallel clustering built-in). Out-of-the-box, it ships natively with Image processing and Keystroke integration via ONNX/Keras alone.