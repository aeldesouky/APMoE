# Developer Experience and Error Handling

APMoE is designed as a generic, Inversion of Control (IoC) framework. This document details how the developer experience (DX) and error handling have been optimized to make extending and debugging the framework as seamless as possible.

## 1. CLI Scaffolding and Fast Feedback

**Project Initialization**
The `apmoe init <project-name> --builtin` command gives developers an immediate, fully working project. It scaffolds:
- A `config.json` file pre-wired with valid processing strategies and models.
- Placeholder stub files (`custom_expert.py`, `custom_cleaner.py`, etc.) demonstrating how to subclass the framework's Abstract Base Classes (ABCs).
- A `weights/` directory populated with working ONNX and Keras models (when `--builtin` is passed).
- A `README.md` containing next steps.

**Configuration Validation**
To prevent deep runtime failures, developers can use `apmoe validate --config config.json`. This command acts as a comprehensive pre-flight health check, verifying:
- Pydantic schema validation for the JSON structure.
- Successful import and resolution of all Python classes specified via dotted paths.
- Existence of all required weight files on disk.
- Execution of the generic health checks (`get_info()`) against each `ExpertPlugin`.

## 2. Error Handling Hierarchy

APMoE uses a strongly typed, hierarchical exception system defined in `src/apmoe/core/exceptions.py` to ensure high visibility and predictability during API or CLI execution.

All errors inherit from `APMoEError`, allowing broad catch-all handling. This is then split into granular domains:

### **Configuration and Initialization Errors**
- `ConfigurationError`: Raised when the `config.json` structure is malformed, required fields are missing, or Pydantic validation fails.
- `RegistryError`: Raised when attempting to register duplicate plugin names or load an unregistered class via dotted path.

### **Inference Pipeline Errors**
- `PipelineError`: Raised when the orchestrator fails (e.g., an aggregator receives identical or empty output structures, or post-filtering leaves no valid experts).
- `ModalityError`: Raised specifically for data-level issues during the preparation phase (Cleaner, Anonymizer, Embedder). Also handles raw data validation failures.

### **Plugin Validation and Execution Errors**
- `ExpertError`: Thrown when an expert plugin fails to load properly, encounters an invalid payload shape during `predict()`, or attempts to return a malformed `ExpertOutput`.

### **HTTP Serving Errors**
- `ServingError`: Handles ASGI-level failures, framework bootstrapping errors within the FastAPI startup sequence, or missing middleware requirements.

## 3. API Transparency and Logging

**Standardized Outputs**
All successful pipeline executions guarantee an `apmoe.core.types.Prediction` dataclass serialization. This applies identically whether passing data via the `apmoe predict` CLI or the `POST /predict` HTTP endpoints. 

**Decorators and Extensibility**
Registering custom processing strategies or models is incredibly low-friction. Developers only need to use module-level singletons exposed via `apmoe`. For example:

```python
from apmoe import expert_registry, ExpertPlugin

@expert_registry.register("my_domain_expert")
class DomainExpert(ExpertPlugin):
    ...
```

**Developer-centric Messaging**
Failed dependency injections or validation checks log highly readable tracebacks pointing specifically to the component key and expected path formatting. The `message` and `context` attributes on all `APMoEError` subclasses are directly passed back to the user or API consumer, keeping abstraction boundaries strong but error states clear.

## 4. Complete Extensibility Strategy (Inversion of Control)

APMoE's true power lies in its dependency injection architecture. Rather than hard-coding how "images" or "keystrokes" are processed, APMoE uses configuration files to bind implementations to interfaces:

1. **ModalityProcessors**: Convert raw bytes (e.g., from an HTTP upload or disk) into typed `ModalityData`. E.g., `ImageProcessor` handles JPEG decoding into arrays.
2. **Pipelines**: Each modality has a chained processing pipeline.
   * **Cleaners**: Ensure data validity (e.g., Image Cleaner resizing or Keystroke filtering).
   * **Anonymizers**: Mutate the payload to preserve privacy before inference (e.g., applying pixelation/blur to images).
   * **Embedders (Optional)**: Can pre-compute vector embeddings if multiple experts require the same expensive feature extraction.
3. **Experts (MoE)**: Isolated inference modules (`ExpertPlugin`). The developer specifies required modalities (e.g., `["image"]`, `["keystroke", "audio"]`), and the framework automatically subsets the data and invokes each expert concurrently. Built-in experts handle Keras (`FaceAgeExpert`) and ONNX (`KeystrokeAgeExpert`) weights natively.
4. **Aggregators**: Combine multiple `ExpertOutput` structures into a final `Prediction`. The `WeightedAverageAggregator` natively handles simple confidence/value arithmetic, but custom heuristic combiners can be plugged in immediately.

## 5. Serving Middleware and Security

The `apmoe serve` endpoint automatically generates a robust FastAPI environment wrapped in multiple security layers:
- **CORS Middleware**: Native, configurable Cross-Origin settings to allow web dashboard connections immediately.
- **Request Logging**: Built-in correlation ID tracking for every request sent into the framework.
- **Rate Limiting**: Capable of protecting the heavy AI inference endpoints from DDOS or exhaustion out-of-the-box.
- **AuthPlugin Hook**: Complete abstraction for authentication. Developers can pass an `AuthPlugin` instance to `create_api()`, automatically gating endpoints without touching FastAPIs internals.