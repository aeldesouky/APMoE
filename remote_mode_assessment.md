# APMoE Remote Mode Assessment

The "Remote Mode" in APMoE allows the framework to delegate inference tasks to external models (such as local LLMs, OpenAI, or Hugging Face endpoints) rather than running inference on local weight files. 

After reviewing the codebase, here is the full picture of the current implementation and its capabilities.

## 1. Core Architecture: `RemoteExpert`
At the heart of the remote mode is `apmoe.experts.remote.RemoteExpert`. Instead of loading a local model (like Keras or ONNX), this expert serializes the modality data and `POST`s it to an HTTP endpoint.

**Key Features:**
*   **Security & Env Var Expansion:** To avoid hardcoding secrets in `config.json`, the expert supports `$VAR` substitution at bootstrap time for the endpoint URL, headers (e.g., `Bearer $TOKEN`), and static template fields.
*   **Templating Engine:** The `request_template` allows you to customize the JSON payload sent to the remote server using placeholders like `{{modalities.image}}` and `{{expert_name}}`. 
*   **Resilience & Circuit Breaking:** It includes a robust HTTP client built on `httpx` with:
    *   **Retry Policy:** Exponential backoff and jitter for transient errors (e.g., 502, 503).
    *   **Circuit Breaker:** A state machine (`closed` -> `open` -> `half_open`) that stops sending requests if an endpoint fails consistently, preventing cascading timeouts.
*   **Response Mapping:** Supports JSON-path-style extraction (e.g., `result.age` or `[0].age`) to easily pull the `predicted_age` out of generic API responses.

## 2. Built-in Providers
To make integration with popular services easier, the framework includes an `apmoe.experts.providers` namespace which houses ready-to-use subclasses of `RemoteExpert`.

**Current Provider: `LMStudioExpert`**
*   Designed specifically for LM Studio's local inference server (`/api/v1/chat`).
*   Overrides `_parse_response` to automatically dig through the chat completions schema, find the model's text response, and use a regular expression to extract the numeric age.
*   Collects rich metadata for observability, such as `input_tokens`, `output_tokens`, and `tokens_per_sec`, attaching them to the final `ExpertOutput`.

## 3. Modality Pre-processing for LLMs
Local models expect raw float tensors (e.g., `[200, 200, 3]` arrays for images), but remote LLMs expect images to be compact, Base64-encoded strings. To bridge this gap, the `apmoe.processing.llm` module provides specialized processing strategies:

*   **`Base64ImageCleaner`**: Takes the raw image, resizes it to fit within standard LLM token context windows (max 160px longest side), applies JPEG compression, and encodes it to a Base64 string.
*   **`PassthroughImageAnonymizer`**: A no-op anonymizer that satisfies the pipeline requirement while allowing the Base64 string to pass through to the expert unharmed.

## Summary & Assessment
The Remote Mode implementation is **highly mature and robust**. 
*   **Strengths**: The separation of concerns is excellent. The base `RemoteExpert` handles all the complex HTTP networking (retries, timeouts, security auditing), allowing providers (like `LMStudioExpert`) to be extremely thin—only needing to implement response parsing. The env-var templating is a great touch for security.
*   **Scalability**: The `apmoe.experts.providers` pattern is highly extensible. Adding support for OpenAI, Anthropic, or Hugging Face Inference API simply requires subclassing `RemoteExpert` and tweaking the response parser, exactly as was done for LM Studio.

> [!TIP]
> If you plan on adding more remote providers in the future, you just need to drop a new file into `src/apmoe/experts/providers/` and implement `_parse_response` or `_build_request_body`. The heavy lifting (circuit breakers, retries, serialization) will be handled automatically by the base class.
