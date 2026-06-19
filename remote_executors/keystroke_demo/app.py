"""Standalone keystroke remote executor for APMoE demos.

The service intentionally lives outside ``src/apmoe`` so the framework calls it
through the same HTTP ``RemoteExpert`` contract a production remote executor
would use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from apmoe.core.exceptions import APMoEError
from apmoe.core.types import ModalityData
from apmoe.experts.builtin import KeystrokeAgeExpert
from apmoe.modality.builtin.keystroke import KeystrokeProcessor
from apmoe.processing.builtin.anonymizers import KeystrokeAnonymizer
from apmoe.processing.builtin.cleaners import KeystrokeCleaner

CONTRACT_VERSION = "apmoe.remote-executor.v1"
DEFAULT_EXPERT_NAME = "remote_keystroke_age_expert"


class RemoteExecutorRequest(BaseModel):
    """APMoE ``RemoteExpert`` request body.

    ``RemoteExpert`` sends ``expert_name`` plus ``modalities`` by default. The
    direct ``keystroke`` and ``inputs`` fields are accepted for manual smoke
    tests and for configs that use a simple custom request template.
    """

    expert_name: str = DEFAULT_EXPERT_NAME
    modalities: dict[str, Any] | None = None
    keystroke: Any | None = None
    inputs: Any | None = None

    model_config = {"extra": "allow"}


class RemoteExecutorResponse(BaseModel):
    """Flat response shape consumed by ``apmoe.experts.remote.RemoteExpert``."""

    predicted_age: float
    confidence: float = Field(ge=-1.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _default_weights_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    env_path = os.environ.get("APMOE_KEYSTROKE_WEIGHTS")
    if env_path:
        return Path(env_path)
    return root / "weights" / "keystroke_age_expert.onnx"


class KeystrokeRemoteExecutor:
    """Loads and runs the bundled keystroke model behind an HTTP contract."""

    def __init__(self, weights_path: str | os.PathLike[str] | None = None) -> None:
        """Load the keystroke processor, cleaner, anonymizer, and expert."""
        self._weights_path = Path(weights_path) if weights_path else _default_weights_path()
        self._processor = KeystrokeProcessor()
        self._cleaner = KeystrokeCleaner()
        self._anonymizer = KeystrokeAnonymizer()
        self._expert = KeystrokeAgeExpert()
        self._expert.load_weights(str(self._weights_path))

    @property
    def is_ready(self) -> bool:
        """Return whether the underlying expert is loaded."""
        return self._expert.is_loaded

    def info(self) -> dict[str, Any]:
        """Return executor diagnostics and integration contract details."""
        expert_info = self._expert.get_info()
        return {
            "name": "keystroke_demo_remote_executor",
            "mode": "keystroke",
            "contract_version": CONTRACT_VERSION,
            "request_contract": {
                "default": {
                    "expert_name": "string",
                    "modalities": {"keystroke": "dict[str, list[float]]"},
                },
                "manual_smoke_test_aliases": ["keystroke", "inputs"],
            },
            "response_contract": {
                "predicted_age": "float",
                "confidence": "float in [0, 1]",
                "metadata": "object",
            },
            "weights": str(self._weights_path),
            "expert": expert_info,
        }

    def predict(self, request: RemoteExecutorRequest) -> RemoteExecutorResponse:
        """Run one keystroke inference request."""
        payload = self._extract_keystroke_payload(request)
        data = self._prepare_keystroke_data(payload)
        output = self._expert.predict({"keystroke": data})

        metadata = dict(output.metadata)
        metadata.update(
            {
                "executor": "keystroke_demo_remote_executor",
                "contract_version": CONTRACT_VERSION,
                "mode": "keystroke",
                "requested_expert_name": request.expert_name,
                "model_expert_name": output.expert_name,
                "input_features": len(data.data),
            }
        )
        return RemoteExecutorResponse(
            predicted_age=output.predicted_age,
            confidence=output.confidence,
            metadata=metadata,
        )

    def _extract_keystroke_payload(self, request: RemoteExecutorRequest) -> object:
        if request.modalities and "keystroke" in request.modalities:
            return request.modalities["keystroke"]
        if request.keystroke is not None:
            return request.keystroke
        if request.inputs is not None:
            return request.inputs
        raise HTTPException(
            status_code=422,
            detail=(
                "Keystroke payload required at modalities.keystroke, "
                "keystroke, or inputs."
            ),
        )

    def _prepare_keystroke_data(self, payload: object) -> ModalityData:
        if isinstance(payload, dict):
            modality_data = ModalityData(
                modality="keystroke",
                data=_normalise_feature_mapping(payload),
                metadata={
                    "source": "remote_executor_contract",
                    "num_features_observed": len(payload),
                },
            )
        elif isinstance(payload, list):
            modality_data = self._processor.preprocess(json.dumps(payload))
        elif isinstance(payload, str):
            modality_data = self._processor.preprocess(payload)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported keystroke payload type: {type(payload).__name__}.",
            )

        cleaned = self._cleaner.clean(modality_data)
        if not cleaned.data:
            raise HTTPException(
                status_code=422,
                detail="Keystroke payload contains no valid timings after cleaning.",
            )
        return self._anonymizer.anonymize(cleaned)


def _normalise_feature_mapping(payload: dict[str, Any]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for feature_name, values in payload.items():
        if isinstance(values, list):
            result[str(feature_name)] = [float(value) for value in values]
        else:
            result[str(feature_name)] = [float(values)]
    return result


def create_app(weights_path: str | os.PathLike[str] | None = None) -> FastAPI:
    """Create the FastAPI app for uvicorn or tests."""
    try:
        executor = KeystrokeRemoteExecutor(weights_path=weights_path)
    except APMoEError as exc:
        raise RuntimeError(f"Failed to initialise keystroke remote executor: {exc}") from exc

    app = FastAPI(
        title="APMoE Keystroke Remote Executor",
        version="1.0.0",
        description="Demo remote executor implementing the APMoE RemoteExpert contract.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "healthy" if executor.is_ready else "unhealthy",
            "mode": "keystroke",
            "contract_version": CONTRACT_VERSION,
            "expert_loaded": executor.is_ready,
        }

    @app.get("/info")
    def info() -> dict[str, Any]:
        return executor.info()

    @app.post("/predict", response_model=RemoteExecutorResponse)
    def predict(request: RemoteExecutorRequest) -> RemoteExecutorResponse:
        try:
            return executor.predict(request)
        except HTTPException:
            raise
        except (APMoEError, ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
