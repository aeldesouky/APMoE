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
        summary="Multimodal age prediction",
        response_description="Aggregated age prediction with per-expert breakdown.",
        tags=["Inference"],
    )
    async def predict(request: Request) -> dict[str, Any]:
        """Run the inference pipeline from a JSON request body.

        The body must be a JSON **object** where each key is a modality name
        and the value is the raw session data for that modality.

        **Keystroke example:**

        .. code-block:: json

            {
              "keystroke": [
                [8,  0,  95.0],
                [13, 0, 100.0],
                [65, 83, 145.2],
                [79, 32, 261.0]
              ]
            }

        Each value is serialised to UTF-8 JSON bytes and passed to the
        corresponding modality processor.  Accepted value shapes (auto-detected
        by the processor):

        * **List of triples** ``[[key1, key2, ms], ...]`` — keystroke sessions
        * **Dict of lists** ``{"dur_8": [95, 102], ...}`` — pre-computed features
        * **String** — raw IKDD text (``"8-0,95.0\\n13-0,100.0\\n..."``)

        Modalities absent from the body are skipped gracefully; the experts
        that required them are listed in ``skipped_experts``.

        Returns:
            JSON object containing:

            * ``predicted_age`` — best age estimate in years.
            * ``confidence`` — aggregated confidence in ``[0, 1]``.
            * ``confidence_interval`` — ``[lower, upper]`` or ``null``.
            * ``per_expert_outputs`` — list of per-expert prediction dicts.
            * ``skipped_experts`` — names of experts that could not run.
            * ``metadata`` — pipeline-level metadata.

        Raises:
            HTTPException 422: If the body is not valid JSON or not a JSON object.
            HTTPException 503: If the pipeline has no runnable experts.
            HTTPException 500: For unexpected framework errors.
        """
        correlation_id: str = getattr(request.state, "correlation_id", "-")

        # --- Parse JSON body ---
        try:
            body = await request.json()
        except Exception as exc:
            logger.warning(
                "[%s] 422 Bad request — body is not valid JSON: %s",
                correlation_id,
                exc,
            )
            raise HTTPException(
                status_code=422,
                detail=f"Request body is not valid JSON: {exc}",
            ) from exc

        if not isinstance(body, dict):
            logger.warning(
                "[%s] 422 Bad request — expected JSON object, got %s",
                correlation_id,
                type(body).__name__,
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "JSON body must be an object whose keys are modality names. "
                    f"Got {type(body).__name__}."
                ),
            )

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

        return _prediction_to_dict(prediction)

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @router.get(
        "/health",
        summary="Readiness / liveness probe",
        response_description="Expert health status and overall readiness.",
        tags=["Operations"],
    )
    async def health() -> dict[str, Any]:
        """Return the health status of all loaded expert plugins.

        Queries :meth:`~apmoe.experts.registry.ExpertRegistry.health_check`
        to determine whether every registered expert has successfully loaded
        its pretrained weights.

        Returns:
            JSON object with:

            * ``status`` — ``"healthy"`` if all experts are loaded, otherwise
              ``"degraded"``.
            * ``experts`` — mapping of expert name → boolean loaded status.

        Raises:
            HTTPException 503: If one or more experts are not loaded.
        """
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
        summary="Framework information",
        response_description="Version, loaded experts, active modalities, and serving config.",
        tags=["Operations"],
    )
    async def info() -> dict[str, Any]:
        """Return metadata about the running APMoE framework instance.

        Delegates to :meth:`~apmoe.core.app.APMoEApp.get_info`, which
        aggregates version, expert, modality, aggregator, and serving config
        information into a single JSON-serialisable dict.

        Returns:
            JSON object with:

            * ``version`` — framework version string.
            * ``experts`` — list of per-expert info dicts.
            * ``modalities`` — list of active modality names.
            * ``aggregator`` — aggregator info dict.
            * ``serving`` — serving config dict.
        """
        return apmoe_app.get_info()

    return router
