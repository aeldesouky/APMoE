"""Pydantic models and OpenAPI metadata for the HTTP serving layer.

Keeps the FastAPI app's OpenAPI/Swagger documentation accurate, readable,
and populated with request/response examples.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# App-level OpenAPI (FastAPI constructor)
# ---------------------------------------------------------------------------

OPENAPI_DESCRIPTION = """\
## Multimodal age prediction (MVP)

Send **`POST /predict`** with **`Content-Type: application/json`**. The body must be a **JSON object** (not an array). Each **key** is a modality name from your `config.json`; each **value** is any JSON value your modality processor accepts (arrays, strings, nested objects, etc.). Missing modalities are skipped; experts that need them appear under `skipped_experts` in the response.

### Defaults in generated projects

Typical modality keys are **`image`** and **`keystroke`**. Values are normalised to bytes before the pipeline (JSON values are serialised; strings are UTF-8 encoded).

### Interactive docs

| URL | Purpose |
|-----|---------|
| **`/docs`** | Swagger UI — try requests |
| **`/redoc`** | ReDoc |
| **`/openapi.json`** | OpenAPI 3 schema |

### Errors (summary)

| Code | When |
|------|------|
| **422** | Body is not JSON, or root is not a JSON object |
| **503** | No expert could run (e.g. pipeline empty), or health degraded |
| **500** | Other framework errors |

Rate limiting and authentication are optional deployment concerns; see configuration.
"""

OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "Inference",
        "description": "Run the inference pipeline on a JSON multimodal payload.",
    },
    {
        "name": "Operations",
        "description": "Liveness/readiness and static framework metadata for the running process.",
    },
]


# ---------------------------------------------------------------------------
# POST /predict — body examples (Swagger "Examples" dropdown)
# ---------------------------------------------------------------------------

_PREDICT_EXAMPLES: dict[str, dict[str, Any]] = {
    "keystroke_triples": {
        "summary": "Keystroke triples",
        "description": "IKDD-style rows `[key1, key2, dwell_ms]`. Use the modality name from your config (often `keystroke`).",
        "value": {
            "keystroke": [
                [8, 0, 95.0],
                [13, 0, 100.0],
                [65, 83, 145.2],
                [79, 32, 261.0],
            ],
        },
    },
    "keystroke_ikdd_string": {
        "summary": "Keystroke as IKDD text",
        "description": "Newline-separated `key-key,ms` lines; processors accept this string form.",
        "value": {
            "keystroke": "8-0,95.0\n13-0,100.0\n65-83,145.2\n",
        },
    },
    "keystroke_feature_dict": {
        "summary": "Keystroke pre-computed features",
        "description": "Dict of feature name → list of values, as accepted by the built-in keystroke processor.",
        "value": {
            "keystroke": {
                "dur_8": [95.0, 102.0],
                "dur_13": [100.0],
            },
        },
    },
    "image_and_keystroke": {
        "summary": "Image + keystroke (two modalities)",
        "description": (
            "Use modality names from your config (often `image` and `keystroke`). "
            "The image value must be a string your ImageProcessor accepts (e.g. base64 image data or a file path); "
            "paste a real JPEG/PNG payload when executing from Swagger."
        ),
        "value": {
            "image": "<base64 or file path — see ImageProcessor>",
            "keystroke": [[8, 0, 95.0], [13, 0, 100.0]],
        },
    },
}


PredictRequestBody = Annotated[
    dict[str, Any],
    Body(
        ...,
        media_type="application/json",
        title="Multimodal request",
        description=(
            "JSON object mapping each configured modality name to its payload. "
            "Only keys you send are processed; omitting a modality skips experts that depend on it."
        ),
        openapi_examples=_PREDICT_EXAMPLES,
    ),
]


# ---------------------------------------------------------------------------
# POST /predict — response model
# ---------------------------------------------------------------------------


class ExpertOutputItem(BaseModel):
    """One expert's contribution before aggregation."""

    model_config = ConfigDict(extra="ignore")

    expert_name: str
    consumed_modalities: list[str]
    predicted_age: float
    confidence: float = Field(
        description="In [0, 1] or -1.0 when the expert does not report a score."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Aggregated prediction returned by ``POST /predict`` on success (HTTP 200)."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "predicted_age": 32.5,
                "confidence": 0.82,
                "confidence_interval": None,
                "per_expert_outputs": [
                    {
                        "expert_name": "keystroke_age_expert",
                        "consumed_modalities": ["keystroke"],
                        "predicted_age": 32.5,
                        "confidence": 0.82,
                        "metadata": {
                            "predicted_group": "26-35",
                            "features_observed_fraction": 0.45,
                        },
                    }
                ],
                "skipped_experts": ["face_age_expert"],
                "metadata": {
                    "pipeline_latency_s": 0.006,
                    "available_modalities": ["keystroke"],
                    "failed_modalities": {},
                    "confidence_threshold": 0.85,
                    "recommendations": [
                        "Expert 'keystroke_age_expert': keystroke session coverage is 45%"
                        " — collect at least 50 keystrokes for reliable age inference.",
                        "Aggregated confidence is 82%, below the configured threshold of"
                        " 85%. Suggestions: (1) supply additional modalities if available;"
                        " (2) extend the keystroke session length.",
                    ],
                },
            }
        },
    )

    predicted_age: float = Field(description="Aggregated age estimate in years.")
    confidence: float = Field(description="Aggregated confidence in [0, 1].")
    confidence_interval: list[float] | None = Field(
        default=None,
        description="Optional [lower, upper] age bounds; null if not available.",
    )
    per_expert_outputs: list[ExpertOutputItem]
    skipped_experts: list[str] = Field(
        default_factory=list,
        description="Experts not run because required modalities were missing.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Pipeline metadata: latency, available modalities, failures, "
            "confidence_threshold (float | null), and recommendations (list[str]) "
            "when confidence falls below the configured threshold."
        ),
    )


class HealthResponse(BaseModel):
    """`GET /health` JSON body (200 or 503)."""

    model_config = ConfigDict(extra="allow")

    status: str = Field(description='Overall status: "healthy" or "degraded".')
    experts: dict[str, bool] = Field(
        description="Expert name → whether weights loaded successfully.",
    )


class InfoResponse(BaseModel):
    """Loose schema for `GET /info` (shape mirrors `APMoEApp.get_info()`)."""

    model_config = ConfigDict(extra="allow")

    version: str
    experts: list[dict[str, Any]]
    modalities: list[str]
    aggregator: dict[str, Any]
    serving: dict[str, Any]
    confidence_threshold: float | None = Field(
        default=None,
        description=(
            "Confidence gate in [0.0, 1.0] below which the pipeline populates "
            "Prediction.metadata['recommendations'].  null when disabled."
        ),
    )
