# Remote Expert Endpoints

APMoE supports **local** experts (loading a weight file from disk) and
**remote** experts (calling an HTTP model server). Both modes are first-class
citizens and can be mixed freely within the same `config.json`.

A remote expert is declared exactly like a local one — the only difference is
that `"endpoint"` replaces `"weights"` and the class is
`"apmoe.experts.remote.RemoteExpert"`.

---

## Why remote experts?

| Scenario | Benefit |
|---|---|
| HuggingFace Inference API | Use hosted foundation models with zero local GPU |
| Another APMoE server | Horizontally scale a specific modality to a dedicated GPU node |
| OpenAI-compatible endpoint | Plug in any LLM-based age-estimation service |
| Local LLM (LM Studio, Ollama) | Run a vision-language model on-device, fully offline |
| Proprietary model API | Call vendor-hosted models without exporting weights |
| Development / mocking | Point at a local FastAPI stub during CI |

---

## Minimal config example

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "keystroke",
        "processor": "apmoe.modality.builtin.keystroke.KeystrokeProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.cleaners.KeystrokeCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.KeystrokeAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "keystroke_remote_expert",
        "class":      "apmoe.experts.remote.RemoteExpert",
        "modalities": ["keystroke"],
        "endpoint":   "$MY_MODEL_ENDPOINT"
      }
    ],
    "aggregation": { "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator" },
    "serving":     { "host": "0.0.0.0", "port": 8000 }
  }
}
```

> [!NOTE]
> `"weights"` is omitted entirely for remote experts — providing both
> `"weights"` **and** `"endpoint"` in the same entry raises a
> `ConfigurationError` at startup.

---

## All configuration fields

```json
{
  "name":              "<string>  — unique identifier, used in logs + /info",
  "class":             "apmoe.experts.remote.RemoteExpert",
  "modalities":        ["<modality_name>", "..."],
  "endpoint":          "$MY_ENDPOINT  — or a literal URL",
  "endpoint_headers":  { "<header>": "<value or $ENV_VAR>" },
  "endpoint_timeout":  10.0,
  "request_template":  { "…": "{{modalities.<name>}}" },
  "response_mapping":  { "predicted_age": "<dot-path>", "confidence": "<dot-path>" }
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `string` | **required** | Full HTTP/HTTPS URL; `$VAR` is expanded at bootstrap |
| `endpoint_headers` | `dict[str, str]` | `{}` | HTTP headers; `$VAR` values are expanded at bootstrap |
| `endpoint_timeout` | `float` | `10.0` | Read timeout in seconds |
| `request_template` | `dict \| null` | `null` | Body template; literal `$VAR` leaves expanded at bootstrap, `{{…}}` leaves expanded per-request |
| `response_mapping` | `dict \| null` | `null` | Response extraction dot-paths |

---

## Environment variable substitution

`$VAR` references are expanded from environment variables **at bootstrap time**
(when `APMoEApp.from_config()` calls `load_weights()`). Expansion applies to
three places:

| Field | Example config value | Resolved at |
|---|---|---|
| `endpoint` | `"$LLM_ENDPOINT"` | Bootstrap |
| `endpoint_headers` values | `"Bearer $HF_TOKEN"` | Bootstrap |
| Literal string leaves in `request_template` | `"$LLM_MODEL"` | Bootstrap |

> [!IMPORTANT]
> Strings containing `{{...}}` predict-time placeholders are **skipped** during
> env-var expansion so they are preserved for per-request substitution.
> The two syntaxes are entirely independent and can coexist in the same template.

### Two-pass substitution

The full substitution lifecycle for a `request_template` value is:

```
config.json value          bootstrap              predict-time
─────────────────────────────────────────────────────────────
"$LLM_MODEL"        ──▶  "google/gemma-4-e4b"    (unchanged)
"{{modalities.image}}"   (skipped)         ──▶   "<base64 string>"
"Bearer $HF_TOKEN"  ──▶  "Bearer hf_xxx…"        (unchanged)
42                        (unchanged)             (unchanged)
```

| Syntax | When | What it replaces |
|---|---|---|
| `$VARNAME` | Bootstrap (startup) | Environment variables — server URL, model ID, system prompts, API keys |
| `{{modalities.<name>}}` | Per-request (predict) | Serialised modality data for that specific inference call |
| `{{expert_name}}` | Per-request (predict) | The expert's configured `name` string |

### Setting variables before serving

```bash
export LLM_ENDPOINT="http://127.0.0.1:1234/api/v1/chat"
export LLM_MODEL="google/gemma-4-e4b"
export LLM_SYSTEM_PROMPT="Return only an integer age."
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
apmoe serve --config config.json
```

If a referenced variable is not set, `APMoEApp.from_config()` raises an
`ExpertError` immediately with a clear message — misconfigurations are caught
at startup, not mid-request.

---

## Request body

### Default schema (no `request_template`)

When `request_template` is omitted, the framework sends:

```json
{
  "expert_name": "keystroke_remote_expert",
  "modalities": {
    "keystroke": { "dur_8": [95.0, 102.0], "dur_13": [100.0] }
  }
}
```

Each modality value is the serialised output of the processing chain:
- `ModalityData` → its `.data` attribute (dict, list, str, etc.)
- `EmbeddingResult` → `.embedding.tolist()` (list of floats)

### Custom `request_template`

`request_template` is a nested JSON object. Each string leaf can be either:

- A **`$VAR`** reference → expanded from env at bootstrap (static per deployment)
- A **`{{placeholder}}`** expression → substituted per-request with live data
- A **plain literal** → passed through unchanged

Numbers, booleans, nested dicts/lists are always passed through unchanged.

#### Placeholder expressions (predict-time)

| Expression | Replaced with |
|---|---|
| `"{{expert_name}}"` | The expert's configured `name` string |
| `"{{modalities.<name>}}"` | The serialised value for that modality |

Placeholders are matched exactly (the entire leaf string must be the
expression). Partial interpolation within a longer string is not supported.

#### Examples

**HuggingFace Inference API** — single modality under `"inputs"`:
```json
"request_template": { "inputs": "{{modalities.keystroke}}" }
```

**LM Studio / local LLM** — model and system prompt from env vars:
```json
"request_template": {
  "model":         "$LLM_MODEL",
  "system_prompt": "$LLM_SYSTEM_PROMPT",
  "input":         "{{modalities.image}}"
}
```

**OpenAI-compatible** — image under `"content"` in a messages array:
```json
"request_template": {
  "model": "$LLM_MODEL",
  "messages": [{ "role": "user", "content": "{{modalities.image}}" }]
}
```

**Custom server** — mixed literal + placeholder:
```json
"request_template": {
  "source":  "{{expert_name}}",
  "payload": "{{modalities.keystroke}}",
  "version": 3
}
```

**Multiple modalities**:
```json
"request_template": {
  "face":      "{{modalities.image}}",
  "keystrokes":"{{modalities.keystroke}}"
}
```

---

## Response extraction

### Default (no `response_mapping`)

The framework expects the JSON response to be a flat object with these keys
at the top level:

```json
{
  "predicted_age": 34.5,
  "confidence":    0.82,
  "metadata":      {}
}
```

- `confidence` and `metadata` are optional — they default to `-1.0` and `{}`
  respectively if absent.

### Custom `response_mapping`

`response_mapping` maps `ExpertOutput` field names to **dot-paths** in the
remote response. Supported output keys:

| Key | Required | Default when absent |
|---|---|---|
| `"predicted_age"` | ✅ yes | — (raises `ExpertError`) |
| `"confidence"` | ❌ no | `-1.0` |
| `"metadata"` | ❌ no | `{}` |

#### Dot-path syntax

| Syntax | Navigates |
|---|---|
| `"predicted_age"` | Top-level key |
| `"result.age"` | Nested key |
| `"[0].age"` | First element of an array, then key `age` |
| `"[0]"` | First element of an array (scalar) |

#### Examples

**HuggingFace list response** — `[{"score": 0.9, "label": "26-35"}, ...]`:
```json
"response_mapping": {
  "predicted_age": "[0].age",
  "confidence":    "[0].score"
}
```

**Nested response** — `{"output": {"age": 30.1, "prob": 0.77}}`:
```json
"response_mapping": {
  "predicted_age": "output.age",
  "confidence":    "output.prob"
}
```

**Scalar array** — `[30.0]`:
```json
"response_mapping": { "predicted_age": "[0]" }
```

---

## LLM / vision-language model integration

Sending an image to a vision-language model requires a different image
processing pipeline from the one used by the built-in
`FaceAgeExpert` (which expects a normalised `float32 (200, 200, 3)` array).
LLMs need the raw image as a compact, base64-encoded JPEG string.

### `apmoe.processing.llm` module

A dedicated module at `src/apmoe/processing/llm/__init__.py` provides two
processors for this purpose. They are intentionally **separate** from the
core builtin processing pipeline so the standard modality chain is never
affected.

> [!WARNING]
> Do **not** use these processors with `FaceAgeExpert` or any other local
> expert. They produce a base64 string, not the numpy array those experts
> expect.

#### `Base64ImageCleaner`

| Class | `apmoe.processing.llm.Base64ImageCleaner` |
|---|---|
| Registry key | `"base64_image_cleaner"` |
| Input | Raw image bytes or numpy array from `ImageProcessor` |
| Output | Base64-encoded JPEG ASCII string |

Processing steps:
1. Decode raw bytes (or accept a decoded numpy array from `ImageProcessor`).
2. Convert to RGB (promotes grayscale, drops alpha channel).
3. **Resize** so the longest side is at most **160 px**, preserving aspect ratio.
4. **JPEG-compress** at quality **35** — produces ~1.3–2.5 KB / ~1000–1900 tokens.
5. Base64-encode to an ASCII string.

The 160 px / quality 35 defaults keep the image well within a 4096-token context
window while retaining enough detail for age estimation.

Config example:
```json
{
  "name": "image",
  "processor": "apmoe.modality.builtin.image.ImageProcessor",
  "pipeline": {
    "cleaner":    "apmoe.processing.llm.Base64ImageCleaner",
    "anonymizer": "apmoe.processing.llm.PassthroughImageAnonymizer"
  }
}
```

#### `PassthroughImageAnonymizer`

| Class | `apmoe.processing.llm.PassthroughImageAnonymizer` |
|---|---|
| Registry key | `"passthrough_image_anonymizer"` |
| Behaviour | No-op — returns data unchanged |

Exists solely to satisfy the mandatory `Cleaner → Anonymizer` pipeline contract.
If your deployment requires PII removal before sending to a remote server (e.g.
face-blurring before calling an external API), replace this with a custom
`AnonymizerStrategy` subclass that operates on the base64 string.

### Reference config: `configs/llm_remote.json`

The bundled `configs/llm_remote.json` is a fully portable, zero-hardcoding
configuration for a local vision-LLM (tested with LM Studio / Gemma 4):

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "image",
        "processor": "apmoe.modality.builtin.image.ImageProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.llm.Base64ImageCleaner",
          "anonymizer": "apmoe.processing.llm.PassthroughImageAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "llm_face_age_expert",
        "class":      "apmoe.experts.remote.RemoteExpert",
        "modalities": ["image"],
        "endpoint":   "$LLM_ENDPOINT",
        "endpoint_headers": { "Content-Type": "application/json" },
        "endpoint_timeout": 60.0,
        "request_template": {
          "model":         "$LLM_MODEL",
          "system_prompt": "$LLM_SYSTEM_PROMPT",
          "input":         "{{modalities.image}}"
        }
      }
    ],
    "aggregation": { "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator" },
    "serving":     { "host": "0.0.0.0", "port": 8000, "workers": 1 }
  }
}
```

Start it with:
```bash
export LLM_ENDPOINT="http://127.0.0.1:1234/api/v1/chat"
export LLM_MODEL="google/gemma-4-e4b"
export LLM_SYSTEM_PROMPT="Return ONLY a single integer representing the estimated age."
apmoe serve --config configs/llm_remote.json
```

### Handling non-standard LLM response schemas

Many local LLM servers return a response that does not match the flat
`{"predicted_age": ...}` schema. For example, LM Studio's `/api/v1/chat`
returns:

```json
{
  "output": [
    {"type": "reasoning", "content": "..."},
    {"type": "message",   "content": "34"}
  ],
  "stats": { "tokens_per_second": 52.7, ... }
}
```

The `response_mapping` dot-path system cannot filter arrays by field value, so
the recommended approach is to subclass `RemoteExpert` and override
`_parse_response`:

```python
import re
from typing import Any
from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput
from apmoe.experts.remote import RemoteExpert
from apmoe.experts.registry import expert_registry


@expert_registry.register("myapp.LMStudioExpert")
class LMStudioExpert(RemoteExpert):
    """RemoteExpert that parses the LM Studio /api/v1/chat response."""

    def _parse_response(
        self, data: Any, consumed_modalities: list[str]
    ) -> ExpertOutput:
        # Find the first output item with type=="message"
        message_content = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                message_content = item.get("content", "").strip()
                break

        if not message_content:
            raise ExpertError(
                "LMStudioExpert: no 'message' item in response.",
                context={"response": str(data)[:300]},
            )

        # Extract first integer (the predicted age)
        numbers = re.findall(r"\b\d+\b", message_content)
        if not numbers:
            raise ExpertError(
                f"LMStudioExpert: no integer found in '{message_content}'.",
                context={"content": message_content},
            )

        stats = data.get("stats", {})
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=consumed_modalities,
            predicted_age=float(numbers[0]),
            confidence=-1.0,   # LLM is a regressor — no calibrated probability
            metadata={
                "llm_response":   message_content,
                "model":          data.get("model_instance_id", "unknown"),
                "tokens_per_sec": stats.get("tokens_per_second"),
            },
        )
```

Reference it in config as `"class": "myapp.LMStudioExpert"`.

### End-to-end test script

`scripts/test_llm_remote.py` is a self-contained end-to-end test that:
1. Sets env vars via `os.environ.setdefault` (overridden by real shell exports).
2. Bootstraps APMoE from `configs/llm_remote.json`.
3. Loads `kmelsayed.jpg` as raw bytes and runs the full inference pipeline.
4. Prints the predicted age with model metadata and pipeline latency.

```bash
# Optionally override the defaults:
export LLM_ENDPOINT="http://127.0.0.1:1234/api/v1/chat"
export LLM_MODEL="google/gemma-4-e4b"

python scripts/test_llm_remote.py
```

Example output:
```
────────────────────────────────────────────────────────────
  APMoE × Local LLM (Gemma 4) — Age Estimation Test
────────────────────────────────────────────────────────────
  ✅  Predicted age : 68 years
  LLM raw response : "68"
  Model            : google/gemma-4-e4b
  Tokens/sec       : 52.95
  Pipeline latency : 5.93 s
────────────────────────────────────────────────────────────
```

---

## Mixed local + remote experts

Local and remote experts combine transparently. The aggregation strategy
treats `ExpertOutput` objects identically regardless of their origin.

```json
{
  "experts": [
    {
      "name":       "keystroke_age_expert",
      "class":      "apmoe.experts.builtin.KeystrokeAgeExpert",
      "weights":    "./weights/keystroke_age_expert.onnx",
      "modalities": ["keystroke"]
    },
    {
      "name":       "remote_face_expert",
      "class":      "apmoe.experts.remote.RemoteExpert",
      "modalities": ["image"],
      "endpoint":   "$REMOTE_FACE_ENDPOINT",
      "endpoint_headers": { "X-API-Key": "$INTERNAL_API_KEY" }
    }
  ],
  "aggregation": {
    "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator",
    "weights": {
      "keystroke_age_expert": 0.4,
      "remote_face_expert":   0.6
    }
  }
}
```

---

## Provider-specific examples

### HuggingFace Inference API

```json
{
  "name":       "hf_keystroke_expert",
  "class":      "apmoe.experts.remote.RemoteExpert",
  "modalities": ["keystroke"],
  "endpoint":   "https://api-inference.huggingface.co/models/my-org/age-model",
  "endpoint_headers": { "Authorization": "Bearer $HF_TOKEN" },
  "endpoint_timeout": 20.0,
  "request_template": { "inputs": "{{modalities.keystroke}}" },
  "response_mapping": { "predicted_age": "[0].age", "confidence": "[0].score" }
}
```

### LM Studio (local, fully offline)

```json
{
  "name":       "llm_face_age_expert",
  "class":      "myapp.LMStudioExpert",
  "modalities": ["image"],
  "endpoint":   "$LLM_ENDPOINT",
  "endpoint_headers": { "Content-Type": "application/json" },
  "endpoint_timeout": 60.0,
  "request_template": {
    "model":         "$LLM_MODEL",
    "system_prompt": "$LLM_SYSTEM_PROMPT",
    "input":         "{{modalities.image}}"
  }
}
```

Pair this with the `apmoe.processing.llm` image pipeline (see above).

### Another APMoE instance

```json
{
  "name":       "gpu_node_face_expert",
  "class":      "apmoe.experts.remote.RemoteExpert",
  "modalities": ["image"],
  "endpoint":   "$GPU_NODE_ENDPOINT",
  "endpoint_headers": { "X-Internal-Token": "$GPU_NODE_TOKEN" }
}
```

The remote APMoE server responds with the standard APMoE prediction schema,
which the default response mapping handles directly.

### Generic REST API (custom schema)

```json
{
  "name":       "vendor_age_api",
  "class":      "apmoe.experts.remote.RemoteExpert",
  "modalities": ["image"],
  "endpoint":   "$VENDOR_ENDPOINT",
  "endpoint_headers": {
    "Authorization": "ApiKey $VENDOR_KEY",
    "Content-Type":  "application/json"
  },
  "endpoint_timeout": 30.0,
  "request_template": {
    "image_data": "{{modalities.image}}",
    "options":    { "return_confidence": true }
  },
  "response_mapping": {
    "predicted_age": "result.estimated_age",
    "confidence":    "result.confidence_score",
    "metadata":      "result.details"
  }
}
```

---

## Health check and info

### `GET /v1/health`

Remote experts report `true` in the `experts` map when `load_weights()` has
been called successfully (i.e. `httpx` is importable, all env-var references
resolved, and endpoint URL was expanded). The actual remote server is **not**
probed — the health check is local only.

```json
{
  "status": "healthy",
  "experts": {
    "keystroke_age_expert": true,
    "llm_face_age_expert":  true
  }
}
```

### `GET /v1/info`

Remote experts include extra fields in the `experts` array:

```json
{
  "name":                  "llm_face_age_expert",
  "modalities":            ["image"],
  "expert_class":          "RemoteExpert",
  "backend":               "remote",
  "endpoint":              "http://127.0.0.1:1234/api/v1/chat",
  "has_request_template":  true,
  "has_response_mapping":  false,
  "loaded":                true
}
```

Note: `endpoint` is shown **after** env-var expansion, so the resolved URL
(not `$LLM_ENDPOINT`) appears in the info response.

---

## Error reference

| Error | When raised |
|---|---|
| `ConfigurationError` | Both `weights` and `endpoint` provided, or neither |
| `ConfigurationError` | `endpoint` set but class is not a `RemoteExpert` subclass |
| `ExpertError: httpx is required` | `httpx` is not installed |
| `ExpertError: … env var '…' not set` | A `$VAR` in endpoint, headers, or template cannot be resolved |
| `ExpertError: timed out after Xs` | Request exceeded `endpoint_timeout` |
| `ExpertError: HTTP 4xx/5xx` | Remote server returned a non-2xx response |
| `ExpertError: network error` | DNS failure, connection refused, etc. |
| `ExpertError: not valid JSON` | Response body is not parseable JSON |
| `ExpertError: cannot extract 'predicted_age'` | `response_mapping` path not found |
| `ExpertError: Unrecognised template placeholder` | `{{…}}` expression not recognised |

---

## Subclassing `RemoteExpert`

Advanced users can subclass `RemoteExpert` to override serialisation or
parsing behaviour for a specific API without reimplementing the HTTP layer.
Both `_build_request_body` and `_parse_response` are designed as extension
points.

```python
from apmoe.experts.remote import RemoteExpert
from apmoe.core.types import ExpertOutput
from typing import Any

class MyVendorExpert(RemoteExpert):
    """Override both request body construction and response parsing."""

    def _build_request_body(self, serialised: dict[str, Any]) -> dict[str, Any]:
        # Completely custom body — bypass the template system entirely
        return {
            "data": serialised.get("keystroke"),
        }

    def _parse_response(
        self, data: Any, consumed_modalities: list[str]
    ) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=consumed_modalities,
            predicted_age=float(data["age"]),
            confidence=float(data.get("prob", -1.0)),
        )
```

Register and reference it in config:
```json
{
  "class":    "myapp.MyVendorExpert",
  "endpoint": "$VENDOR_ENDPOINT"
}
```

Note: `$VAR` expansion in `endpoint`, `endpoint_headers`, and
`request_template` is performed by `load_weights()` in `RemoteExpert.`
Subclasses that call `super().load_weights()` inherit this behaviour
automatically.

---

## Implementation notes

The remote expert feature spans the following files:

| File | Role |
|---|---|
| `src/apmoe/core/config.py` | `ExpertConfig` extended with `endpoint`, `endpoint_headers`, `endpoint_timeout`, `request_template`, `response_mapping`; `weights` made optional; mutual-exclusion validator added |
| `src/apmoe/experts/remote.py` | `RemoteExpert` class; `_apply_template` (predict-time placeholders); `_resolve_path` (dot-path response extraction); `_expand_str` / `_expand_headers` / `_expand_template` (bootstrap-time env-var expansion across endpoint, headers, and template literals) |
| `src/apmoe/experts/registry.py` | `ExpertRegistry.from_configs` detects remote experts, passes constructor kwargs, lazy-imports `RemoteExpert` for auto-registration |
| `src/apmoe/core/app.py` | `validate()` skips weight-file existence check for remote experts |
| `pyproject.toml` | `httpx>=0.27` added to core dependencies; `[remote]` optional extras alias added |
| `src/apmoe/processing/llm/__init__.py` | **New** — `Base64ImageCleaner` (resize + JPEG compress + base64 encode for LLM context windows) and `PassthroughImageAnonymizer` (no-op to satisfy pipeline contract); kept entirely separate from the core builtin modality pipeline |
| `configs/llm_remote.json` | **New** — reference config for a local vision-LLM; fully driven by `$LLM_ENDPOINT`, `$LLM_MODEL`, `$LLM_SYSTEM_PROMPT` |
| `scripts/test_llm_remote.py` | **New** — self-contained end-to-end test; includes `LMStudioExpert` subclass that parses the LM Studio `/api/v1/chat` response format |
