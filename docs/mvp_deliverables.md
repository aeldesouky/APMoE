# MVP Deliverables

This document maps the APMoE framework's features and implementation status to the requirements specified in the MVP deliverables.

## 1. MVP Features Checklist

| Feature Component | Status | Evidence |
| :--- | :--- | :--- |
| **Project Initialization (CLI)** | Implemented | `apmoe init` command creating project scaffolding with built-in weights (`--builtin`). |
| **Framework Extensibility (IoC)** | Implemented | `config.json` defining `ModalityProcessor`, `Cleaner`, `Anonymizer`, and custom `ExpertPlugin` paths. |
| **Configuration Validation** | Implemented | `apmoe validate` schema, path resolution, and health checks. |
| **Multimodal AI Pipeline** | Implemented | Image and Keystroke modality processors, cleaners, anonymizers, and optional embedders processing via `apmoe predict`. |
| **Model Prediction (MoE)** | Implemented | Keras (Face) and ONNX (Keystroke) expert plugins executing independently and aggregating results. |
| **Backend API (Serving)** | Implemented | `apmoe serve` starting FastAPI server with `/predict`, `/health`, and `/info` endpoints. |
| **Middleware & DX** | Implemented | Rate Limiting, CORS, Auth hooks, correlation logging, and rigid `APMoEError` handling. |
| **API Dashboard View** | Implemented | Swagger UI automatically available at `/docs` when serving. |

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