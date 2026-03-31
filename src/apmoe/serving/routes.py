"""FastAPI route handlers for the APMoE serving layer.

Three endpoints are provided:

* ``POST /predict`` — multimodal age prediction from file uploads.
* ``GET /health`` — readiness / liveness probe (checks all experts are loaded).
* ``GET /info`` — framework metadata (version, experts, modalities, config).

Routes are built by :func:`create_router`, which closes over the
:class:`~apmoe.core.app.APMoEApp` instance supplied by
:func:`~apmoe.serving.app_factory.create_api`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from apmoe.core.exceptions import APMoEError, PipelineError
from apmoe.core.types import Prediction

if TYPE_CHECKING:
    from apmoe.core.app import APMoEApp


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
        """Run the inference pipeline on multimodal file uploads.

        Accepts a ``multipart/form-data`` body where **each field name is a
        modality name** (e.g. ``visual``, ``audio``, ``eeg``) and the
        corresponding value is an uploaded file.  Text fields are treated as
        raw string inputs and passed through unchanged.

        Fields whose names do not match any configured modality are silently
        ignored.  Configured modalities whose files are absent cause dependent
        experts to be skipped (graceful degradation).

        Returns:
            JSON object containing:

            * ``predicted_age`` — best age estimate in years.
            * ``confidence`` — aggregated confidence in ``[0, 1]``.
            * ``confidence_interval`` — ``[lower, upper]`` or ``null``.
            * ``per_expert_outputs`` — list of per-expert prediction dicts.
            * ``skipped_experts`` — names of experts that could not run.
            * ``metadata`` — pipeline-level metadata (e.g. latency).

        Raises:
            HTTPException 422: If the multipart body cannot be parsed.
            HTTPException 503: If the pipeline has no runnable experts.
            HTTPException 500: For unexpected framework errors.
        """
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to parse multipart form data: {exc}",
            ) from exc

        inputs: dict[str, Any] = {}
        for field_name, field_value in form.multi_items():
            if isinstance(field_value, UploadFile):
                inputs[field_name] = await field_value.read()
            else:
                inputs[field_name] = field_value

        try:
            prediction: Prediction = await apmoe_app.predict_async(inputs)
        except PipelineError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except APMoEError as exc:
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
