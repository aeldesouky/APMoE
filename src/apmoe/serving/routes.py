"""FastAPI route handlers for the APMoE serving layer.

Three endpoints are provided:

* ``POST /predict`` — multimodal age prediction via a JSON request body.
* ``GET /health`` — readiness / liveness probe (checks all experts are loaded).
* ``GET /info`` — framework metadata (version, experts, modalities, config).

Routes are built by :func:`create_router`, which closes over the
:class:`~apmoe.core.app.APMoEApp` instance supplied by
:func:`~apmoe.serving.app_factory.create_api`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from apmoe.core.exceptions import APMoEError, PipelineError
from apmoe.core.types import Prediction
from apmoe.serving.openapi_schemas import (
    HealthResponse,
    InfoResponse,
    PredictRequestBody,
    PredictionResponse,
)

if TYPE_CHECKING:
    from apmoe.core.app import APMoEApp

logger = logging.getLogger("apmoe.serving.routes")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _prediction_to_dict(prediction: Prediction) -> dict[str, Any]:
    """Convert a :class:`~apmoe.core.types.Prediction` dataclass to a JSON-serialisable dict.

    Args:
        prediction: The aggregated prediction returned by the inference pipeline.

    Returns:
        A plain dict suitable for JSON serialisation, containing:

        * ``predicted_age``
        * ``confidence``
        * ``confidence_interval`` (``null`` when not set)
        * ``per_expert_outputs`` (list of per-expert breakdowns)
        * ``skipped_experts``
        * ``metadata``
    """
    return {
        "predicted_age": prediction.predicted_age,
        "confidence": prediction.confidence,
        "confidence_interval": (
            list(prediction.confidence_interval)
            if prediction.confidence_interval is not None
            else None
        ),
        "per_expert_outputs": [
            {
                "expert_name": eo.expert_name,
                "consumed_modalities": eo.consumed_modalities,
                "predicted_age": eo.predicted_age,
                "confidence": eo.confidence,
                "metadata": eo.metadata,
            }
            for eo in prediction.per_expert_outputs
        ],
        "skipped_experts": prediction.skipped_experts,
        "metadata": prediction.metadata,
    }


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_router(apmoe_app: APMoEApp) -> APIRouter:
    """Build and return an :class:`~fastapi.APIRouter` bound to *apmoe_app*.

    All three endpoints close over *apmoe_app* so they share a single
    framework instance without using global state.

    Args:
        apmoe_app: The bootstrapped :class:`~apmoe.core.app.APMoEApp` instance.

    Returns:
        A fully-configured :class:`~fastapi.APIRouter` ready to be included
        in a :class:`~fastapi.FastAPI` application.
    """
    router = APIRouter()

    # ------------------------------------------------------------------
    # POST /predict
    # ------------------------------------------------------------------

    @router.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Predict age from multimodal JSON",
        response_description=(
            "Aggregated age estimate, confidence, optional interval, per-expert rows, "
            "skipped experts, and pipeline metadata."
        ),
        tags=["Inference"],
        responses={
            422: {
                "description": (
                    "Malformed JSON, or JSON value that cannot be parsed as an object "
                    "(e.g. top-level array)."
                ),
            },
            503: {"description": "Pipeline could not run any expert (e.g. all skipped)."},
            500: {"description": "Framework error during inference."},
        },
    )
    async def predict(request: Request, body: PredictRequestBody) -> PredictionResponse:
        """Run inference. Request/response schemas and **Examples** are defined in OpenAPI."""
        correlation_id: str = getattr(request.state, "correlation_id", "-")

        # Serialise each value to UTF-8 JSON bytes so every modality processor
        # receives a consistent bytes payload regardless of the value type.
        inputs: dict[str, Any] = {}
        for modality, value in body.items():
            if isinstance(value, str):
                inputs[modality] = value.encode()
            elif isinstance(value, bytes):
                inputs[modality] = value
            else:
                inputs[modality] = json.dumps(value).encode()

        # --- Run inference ---
        try:
            prediction: Prediction = await apmoe_app.predict_async(inputs)
        except PipelineError as exc:
            logger.error(
                "[%s] 503 Service Unavailable — pipeline error: %s",
                correlation_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except APMoEError as exc:
            logger.error(
                "[%s] 500 Internal Server Error — framework error: %s",
                correlation_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return PredictionResponse.model_validate(_prediction_to_dict(prediction))

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="Health check",
        response_description='Overall `"healthy"` or `"degraded"` plus per-expert load status.',
        tags=["Operations"],
        responses={
            200: {"description": "All experts loaded (or none registered)."},
            503: {"description": "One or more experts failed to load weights."},
        },
    )
    async def health() -> dict[str, Any]:
        """Expert weight load status; returns **503** when any expert is not loaded."""
        expert_health: dict[str, bool] = apmoe_app.expert_registry.health_check()
        all_healthy = all(expert_health.values()) if expert_health else True
        status = "healthy" if all_healthy else "degraded"

        payload: dict[str, Any] = {
            "status": status,
            "experts": expert_health,
        }

        if not all_healthy:
            not_loaded = [name for name, ok in expert_health.items() if not ok]
            logger.error(
                "503 Service Unavailable — health check degraded. "
                "Experts not loaded: %s",
                not_loaded,
            )
            return JSONResponse(content=payload, status_code=503)

        return payload  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # GET /info
    # ------------------------------------------------------------------

    @router.get(
        "/info",
        response_model=InfoResponse,
        summary="Framework metadata",
        response_description="Version, experts, modalities, aggregator, and serving settings snapshot.",
        tags=["Operations"],
    )
    async def info() -> dict[str, Any]:
        """Runtime snapshot from ``APMoEApp.get_info()`` (version, experts, config)."""
        return apmoe_app.get_info()

    return router
