#!/usr/bin/env python3
"""End-to-end resilience checks for APMoE.

This script exercises the resilience layer through bootstrapped APMoE
applications and serving stores. It intentionally uses in-process fakes for
remote HTTP and Redis so it can run without network access or external
services.

Usage:
    python scripts/e2e_resilience.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.app import APMoEApp
from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput, ModalityData, Prediction, ProcessedInput
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import modality_registry
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    anonymizer_registry,
    cleaner_registry,
)
from apmoe.serving.middleware import RedisRateLimitStore, RedisTokenInvalidationStore

logging.getLogger("apmoe.pipeline").setLevel(logging.CRITICAL)
logging.getLogger("apmoe.serving").setLevel(logging.CRITICAL)


class _E2EVisualProcessor(ModalityProcessor):
    @property
    def modality_name(self) -> str:
        return "visual"

    def validate(self, data: object) -> bool:
        return data is not None

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality="visual", data=data, metadata={"processed": True})


class _PassCleaner(CleanerStrategy):
    def clean(self, data: ModalityData) -> ModalityData:
        return data


class _PassAnonymizer(AnonymizerStrategy):
    def anonymize(self, data: ModalityData) -> ModalityData:
        return data


class _HealthyExpert(ExpertPlugin):
    _loaded = False

    @property
    def name(self) -> str:
        return "healthy_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["visual"],
            predicted_age=42.0,
            confidence=0.8,
            metadata={"source": "healthy"},
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _FailingExpert(ExpertPlugin):
    _loaded = False

    @property
    def name(self) -> str:
        return "failing_expert"

    def declared_modalities(self) -> list[str]:
        return ["visual"]

    def load_weights(self, path: str) -> None:
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        raise ExpertError("synthetic expert outage", context={"expert": self.name})

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class _AverageAggregator(AggregatorStrategy):
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        age = sum(output.predicted_age for output in outputs) / len(outputs)
        confidence = sum(output.confidence for output in outputs) / len(outputs)
        return Prediction(
            predicted_age=age,
            confidence=confidence,
            per_expert_outputs=list(outputs),
            metadata={"aggregated_count": len(outputs)},
        )


def _register_components() -> None:
    modality_registry.register_class(
        "e2e.VisualProcessor", _E2EVisualProcessor, overwrite=True
    )
    cleaner_registry.register_class("e2e.PassCleaner", _PassCleaner, overwrite=True)
    anonymizer_registry.register_class(
        "e2e.PassAnonymizer", _PassAnonymizer, overwrite=True
    )
    expert_registry.register_class("e2e.HealthyExpert", _HealthyExpert, overwrite=True)
    expert_registry.register_class("e2e.FailingExpert", _FailingExpert, overwrite=True)
    aggregator_registry.register_class(
        "e2e.AverageAggregator", _AverageAggregator, overwrite=True
    )


def _base_config(tmp_dir: Path) -> dict[str, Any]:
    return {
        "apmoe": {
            "modalities": [
                {
                    "name": "visual",
                    "processor": "e2e.VisualProcessor",
                    "pipeline": {
                        "cleaner": "e2e.PassCleaner",
                        "anonymizer": "e2e.PassAnonymizer",
                    },
                }
            ],
            "experts": [],
            "aggregation": {"strategy": "e2e.AverageAggregator"},
            "serving": {
                "authentication_enabled": False,
                "authorization_enabled": False,
            },
            "security": {
                "remote_endpoint_allowlist": ["models.example.com"],
                "remote_enforce_https": True,
                "remote_allow_private_networks": False,
            },
        }
    }


def _write_config(tmp_dir: Path, data: dict[str, Any], name: str) -> Path:
    path = tmp_dir / name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _run_pipeline_fallback_check(tmp_dir: Path) -> None:
    healthy_weights = tmp_dir / "healthy.weights"
    failing_weights = tmp_dir / "failing.weights"
    healthy_weights.write_text("ok", encoding="utf-8")
    failing_weights.write_text("ok", encoding="utf-8")

    config = _base_config(tmp_dir)
    config["apmoe"]["expert_failure_policy"] = "skip_failed"
    config["apmoe"]["experts"] = [
        {
            "name": "healthy_expert",
            "class": "e2e.HealthyExpert",
            "weights": str(healthy_weights),
            "modalities": ["visual"],
        },
        {
            "name": "failing_expert",
            "class": "e2e.FailingExpert",
            "weights": str(failing_weights),
            "modalities": ["visual"],
        },
    ]

    app = APMoEApp.from_config(_write_config(tmp_dir, config, "pipeline.json"))

    prediction = app.predict({"visual": b"payload"})
    assert prediction.predicted_age == 42.0
    assert [output.expert_name for output in prediction.per_expert_outputs] == [
        "healthy_expert"
    ]
    assert "failing_expert" in prediction.metadata["failed_experts"]

    async_prediction = asyncio.run(app.predict_async({"visual": b"payload"}))
    assert async_prediction.predicted_age == 42.0
    assert "failing_expert" in async_prediction.metadata["failed_experts"]


def _mock_json_response(payload: dict[str, Any]) -> MagicMock:
    raw = json.dumps(payload).encode("utf-8")
    response = MagicMock()
    response.content = raw
    response.headers = {"content-type": "application/json"}
    response.text = raw.decode("utf-8")
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def _run_remote_retry_and_circuit_check(tmp_dir: Path) -> None:
    import httpx

    endpoint = "https://models.example.com/predict"
    config = _base_config(tmp_dir)
    config["apmoe"]["remote_retry"] = {
        "max_attempts": 2,
        "initial_delay_s": 0.01,
        "max_delay_s": 0.01,
        "backoff_multiplier": 2.0,
        "jitter": False,
    }
    config["apmoe"]["remote_circuit_breaker"] = {
        "enabled": True,
        "failure_threshold": 1,
        "recovery_timeout_s": 0.02,
    }
    config["apmoe"]["experts"] = [
        {
            "name": "remote_age",
            "class": "apmoe.experts.remote.RemoteExpert",
            "endpoint": endpoint,
            "endpoint_timeout": 0.1,
            "modalities": ["visual"],
        }
    ]

    app = APMoEApp.from_config(_write_config(tmp_dir, config, "remote.json"))

    retry_success = _mock_json_response(
        {"predicted_age": 33.0, "confidence": 0.7, "metadata": {"attempt": 2}}
    )
    timeout = httpx.TimeoutException("synthetic timeout")
    with (
        patch("httpx.post", side_effect=[timeout, retry_success]) as post,
        patch("apmoe.experts.remote.time.sleep", return_value=None),
    ):
        prediction = app.predict({"visual": b"payload"})
    assert post.call_count == 2
    assert prediction.predicted_age == 33.0

    request = httpx.Request("POST", endpoint)
    network_error = httpx.RequestError("synthetic network outage", request=request)
    with (
        patch("httpx.post", side_effect=network_error) as post,
        patch("apmoe.experts.remote.time.sleep", return_value=None),
    ):
        try:
            app.predict({"visual": b"payload"})
        except ExpertError:
            pass
        else:
            raise AssertionError("remote outage did not raise ExpertError")
    assert post.call_count == 2

    with patch("httpx.post") as post:
        try:
            app.predict({"visual": b"payload"})
        except ExpertError as exc:
            assert "circuit breaker is open" in str(exc)
        else:
            raise AssertionError("open circuit did not short-circuit the call")
    assert post.call_count == 0

    time.sleep(0.03)
    half_open_success = _mock_json_response(
        {"predicted_age": 35.0, "confidence": 0.75, "metadata": {"half_open": True}}
    )
    with patch("httpx.post", return_value=half_open_success) as post:
        prediction = app.predict({"visual": b"payload"})
    assert post.call_count == 1
    assert prediction.predicted_age == 35.0
    assert app.expert_registry.get("remote_age").get_info()["circuit_state"] == "closed"


def _run_redis_fallback_check() -> None:
    class _FailingPipeline:
        def zremrangebyscore(self, *args: Any) -> None:
            return None

        def zadd(self, *args: Any) -> None:
            return None

        def zcard(self, *args: Any) -> None:
            return None

        def expire(self, *args: Any) -> None:
            return None

        def execute(self) -> list[int]:
            raise RuntimeError("synthetic Redis outage")

    class _FailingRedis:
        def pipeline(self) -> _FailingPipeline:
            return _FailingPipeline()

        def exists(self, key: str) -> bool:
            raise RuntimeError("synthetic Redis outage")

        def setex(self, key: str, ttl_seconds: int, value: str) -> None:
            raise RuntimeError("synthetic Redis outage")

    fake_redis = SimpleNamespace(
        Redis=SimpleNamespace(from_url=lambda url: _FailingRedis())
    )
    with patch.dict(sys.modules, {"redis": fake_redis}):
        rate_store = RedisRateLimitStore("redis://unavailable/0")
        assert rate_store.allow_request("client", limit=1, window_seconds=60) is True
        assert rate_store.allow_request("client", limit=1, window_seconds=60) is False

        token_store = RedisTokenInvalidationStore("redis://unavailable/0")
        expires_at = datetime.now(UTC) + timedelta(minutes=5)
        assert token_store.is_invalid("token-1") is False
        token_store.invalidate("token-1", expires_at)
        assert token_store.is_invalid("token-1") is True


def main() -> int:
    _register_components()
    temp_root = ROOT / "tmp_e2e_resilience"
    temp_root.mkdir(exist_ok=True)

    try:
        _run_pipeline_fallback_check(temp_root)
        print("[PASS] pipeline skip_failed fallback")

        _run_remote_retry_and_circuit_check(temp_root)
        print("[PASS] remote retries and circuit breaker")

        _run_redis_fallback_check()
        print("[PASS] Redis process-local fallback")
    finally:
        for path in temp_root.glob("*"):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        try:
            temp_root.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
