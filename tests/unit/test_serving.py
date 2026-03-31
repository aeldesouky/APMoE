"""Unit tests for the APMoE serving layer (Phase 4).

Covers:

* ``POST /predict`` — success, pipeline error, multi-modality, text field.
* ``GET /health`` — healthy, degraded (503), no experts.
* ``GET /info`` — metadata structure.
* :class:`~apmoe.serving.middleware.RequestLoggingMiddleware` — correlation ID
  header uniqueness.
* :class:`~apmoe.serving.middleware.RateLimitMiddleware` — within limit, over
  limit, ``Retry-After`` header.
* CORS — wildcard and specific-origin configurations.
* :class:`~apmoe.serving.middleware.AuthMiddleware` — allow, deny, excluded
  paths bypass auth.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from apmoe.core.exceptions import APMoEError, PipelineError
from apmoe.core.types import ExpertOutput, Prediction
from apmoe.serving.app_factory import create_api
from apmoe.serving.middleware import AuthPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prediction(
    *,
    age: float = 35.0,
    confidence: float = 0.90,
    ci: tuple[float, float] | None = (30.0, 40.0),
) -> Prediction:
    """Build a minimal :class:`~apmoe.core.types.Prediction` for testing."""
    return Prediction(
        predicted_age=age,
        confidence=confidence,
        confidence_interval=ci,
        per_expert_outputs=[
            ExpertOutput(
                expert_name="test_expert",
                consumed_modalities=["visual"],
                predicted_age=age,
                confidence=confidence,
                metadata={"model": "stub"},
            )
        ],
        skipped_experts=[],
        metadata={"pipeline_ms": 1.0},
    )


def _make_serving_cfg(
    *,
    cors_origins: list[str] | None = None,
    rate_limit: int | None = None,
) -> MagicMock:
    """Return a mock ServingConfig."""
    cfg = MagicMock()
    cfg.cors_origins = cors_origins if cors_origins is not None else ["*"]
    cfg.rate_limit = rate_limit
    return cfg


def _make_app(
    *,
    cors_origins: list[str] | None = None,
    rate_limit: int | None = None,
    predict_result: Prediction | None = None,
    predict_side_effect: Exception | None = None,
    expert_health: dict[str, bool] | None = None,
) -> MagicMock:
    """Build a minimal MagicMock standing in for :class:`APMoEApp`."""
    mock = MagicMock()
    mock.config.apmoe.serving = _make_serving_cfg(
        cors_origins=cors_origins,
        rate_limit=rate_limit,
    )
    mock.expert_registry.health_check.return_value = (
        expert_health if expert_health is not None else {"test_expert": True}
    )
    mock.get_info.return_value = {
        "version": "0.1.0",
        "experts": [{"name": "test_expert", "modalities": ["visual"]}],
        "modalities": ["visual"],
        "aggregator": {"name": "WeightedAverageAggregator"},
        "serving": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "cors_origins": ["*"],
            "rate_limit": None,
            "log_level": "info",
        },
    }

    if predict_side_effect is not None:
        mock.predict_async = AsyncMock(side_effect=predict_side_effect)
    else:
        result = predict_result if predict_result is not None else _make_prediction()
        mock.predict_async = AsyncMock(return_value=result)

    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_app() -> MagicMock:
    """Default APMoEApp mock (no rate limit, wildcard CORS, healthy)."""
    return _make_app()


@pytest.fixture()
def client(mock_app: MagicMock) -> TestClient:
    """TestClient wrapping the default mock app."""
    return TestClient(create_api(mock_app), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    """Tests for the ``POST /predict`` endpoint."""

    def test_success_with_single_file_upload(self, client: TestClient) -> None:
        """A valid file upload returns 200 with the prediction structure."""
        response = client.post(
            "/predict",
            files={"visual": ("face.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_age"] == pytest.approx(35.0)
        assert body["confidence"] == pytest.approx(0.90)
        assert body["confidence_interval"] == [30.0, 40.0]
        assert len(body["per_expert_outputs"]) == 1
        assert body["per_expert_outputs"][0]["expert_name"] == "test_expert"
        assert body["skipped_experts"] == []
        assert "metadata" in body

    def test_file_bytes_forwarded_to_pipeline(
        self, client: TestClient, mock_app: MagicMock
    ) -> None:
        """Raw bytes of the uploaded file are passed to predict_async."""
        payload = b"\xde\xad\xbe\xef"
        client.post("/predict", files={"visual": ("f.bin", payload, "application/octet-stream")})
        mock_app.predict_async.assert_called_once()
        call_inputs: dict[str, Any] = mock_app.predict_async.call_args[0][0]
        assert call_inputs["visual"] == payload

    def test_multiple_modalities_all_forwarded(
        self, client: TestClient, mock_app: MagicMock
    ) -> None:
        """Multiple modality files are each forwarded under their field name."""
        client.post(
            "/predict",
            files={
                "visual": ("img.jpg", b"\xff\xd8", "image/jpeg"),
                "audio": ("clip.wav", b"RIFF", "audio/wav"),
            },
        )
        call_inputs = mock_app.predict_async.call_args[0][0]
        assert "visual" in call_inputs
        assert "audio" in call_inputs

    def test_pipeline_error_returns_503(self, client: TestClient, mock_app: MagicMock) -> None:
        """A PipelineError from predict_async maps to HTTP 503."""
        mock_app.predict_async.side_effect = PipelineError("No experts available.")
        response = client.post(
            "/predict",
            files={"visual": ("f.jpg", b"\xff\xd8", "image/jpeg")},
        )
        assert response.status_code == 503
        assert "No experts available" in response.json()["detail"]

    def test_generic_apmoe_error_returns_500(
        self, client: TestClient, mock_app: MagicMock
    ) -> None:
        """An unexpected APMoEError from predict_async maps to HTTP 500."""
        mock_app.predict_async.side_effect = APMoEError("Something went wrong.")
        response = client.post(
            "/predict",
            files={"visual": ("f.jpg", b"\xff\xd8", "image/jpeg")},
        )
        assert response.status_code == 500
        assert "Something went wrong" in response.json()["detail"]

    def test_confidence_interval_none_serialised_as_null(self) -> None:
        """A Prediction with no confidence interval produces null in the response."""
        app = _make_app(predict_result=_make_prediction(ci=None))
        c = TestClient(create_api(app), raise_server_exceptions=False)
        response = c.post("/predict", files={"visual": ("f.jpg", b"\xff", "image/jpeg")})
        assert response.status_code == 200
        assert response.json()["confidence_interval"] is None

    def test_text_field_forwarded_as_string(
        self, client: TestClient, mock_app: MagicMock
    ) -> None:
        """Non-file form fields are forwarded to the pipeline as plain strings."""
        client.post("/predict", data={"visual": "some-text-value"})
        call_inputs = mock_app.predict_async.call_args[0][0]
        assert call_inputs["visual"] == "some-text-value"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the ``GET /health`` endpoint."""

    def test_all_experts_loaded_returns_200(self, client: TestClient) -> None:
        """All experts healthy → 200 with status 'healthy'."""
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["experts"] == {"test_expert": True}

    def test_degraded_when_expert_not_loaded(self) -> None:
        """One unloaded expert → 503 with status 'degraded'."""
        app = _make_app(expert_health={"expert_a": True, "expert_b": False})
        c = TestClient(create_api(app), raise_server_exceptions=False)
        response = c.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "degraded"
        assert body["experts"]["expert_b"] is False

    def test_no_experts_considered_healthy(self) -> None:
        """Empty expert registry is treated as healthy (no experts to check)."""
        app = _make_app(expert_health={})
        c = TestClient(create_api(app), raise_server_exceptions=False)
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_multiple_experts_all_healthy(self) -> None:
        """Multiple fully-loaded experts → 200 healthy."""
        app = _make_app(expert_health={"a": True, "b": True, "c": True})
        c = TestClient(create_api(app), raise_server_exceptions=False)
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# GET /info
# ---------------------------------------------------------------------------


class TestInfoEndpoint:
    """Tests for the ``GET /info`` endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        """Info endpoint always returns 200."""
        assert client.get("/info").status_code == 200

    def test_response_contains_required_keys(self, client: TestClient) -> None:
        """Response includes version, experts, modalities, aggregator, serving."""
        body = client.get("/info").json()
        for key in ("version", "experts", "modalities", "aggregator", "serving"):
            assert key in body, f"Missing key: {key}"

    def test_delegates_to_get_info(self, client: TestClient, mock_app: MagicMock) -> None:
        """The route delegates to APMoEApp.get_info exactly once."""
        client.get("/info")
        mock_app.get_info.assert_called_once()

    def test_version_matches(self, client: TestClient) -> None:
        """The version field in the response matches the mock's value."""
        body = client.get("/info").json()
        assert body["version"] == "0.1.0"


# ---------------------------------------------------------------------------
# RequestLoggingMiddleware
# ---------------------------------------------------------------------------


class TestRequestLoggingMiddleware:
    """Tests for correlation ID injection and uniqueness."""

    def test_correlation_id_header_present(self, client: TestClient) -> None:
        """Every response includes an X-Correlation-ID header."""
        response = client.get("/health")
        assert "x-correlation-id" in response.headers

    def test_correlation_id_is_uuid4_format(self, client: TestClient) -> None:
        """The correlation ID is a 36-character UUID (8-4-4-4-12 format)."""
        cid = client.get("/health").headers["x-correlation-id"]
        parts = cid.split("-")
        assert len(parts) == 5
        assert len(cid) == 36

    def test_correlation_id_unique_per_request(self, client: TestClient) -> None:
        """Two consecutive requests receive distinct correlation IDs."""
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["x-correlation-id"] != r2.headers["x-correlation-id"]

    def test_correlation_id_on_predict_endpoint(self, client: TestClient) -> None:
        """POST /predict also receives a correlation ID header."""
        response = client.post(
            "/predict",
            files={"visual": ("f.jpg", b"\xff", "image/jpeg")},
        )
        assert "x-correlation-id" in response.headers


# ---------------------------------------------------------------------------
# RateLimitMiddleware
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    """Tests for the sliding-window rate limiter."""

    def test_requests_within_limit_all_succeed(self) -> None:
        """All requests up to the limit return 200."""
        app = _make_app(rate_limit=5)
        c = TestClient(create_api(app), raise_server_exceptions=False)
        for _ in range(5):
            assert c.get("/health").status_code == 200

    def test_request_over_limit_returns_429(self) -> None:
        """The (limit+1)-th request within the window returns 429."""
        app = _make_app(rate_limit=3)
        c = TestClient(create_api(app), raise_server_exceptions=False)
        for _ in range(3):
            c.get("/health")
        response = c.get("/health")
        assert response.status_code == 429

    def test_429_response_body_contains_detail(self) -> None:
        """The 429 response body includes a human-readable detail message."""
        app = _make_app(rate_limit=1)
        c = TestClient(create_api(app), raise_server_exceptions=False)
        c.get("/health")
        body = c.get("/health").json()
        assert "Rate limit exceeded" in body["detail"]

    def test_429_response_has_retry_after_header(self) -> None:
        """Rejected requests include a Retry-After response header."""
        app = _make_app(rate_limit=1)
        c = TestClient(create_api(app), raise_server_exceptions=False)
        c.get("/health")
        response = c.get("/health")
        assert response.status_code == 429
        assert "retry-after" in response.headers

    def test_no_rate_limit_when_not_configured(self) -> None:
        """Without a rate_limit config, many requests all succeed."""
        app = _make_app(rate_limit=None)
        c = TestClient(create_api(app), raise_server_exceptions=False)
        for _ in range(20):
            assert c.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


class TestCORSMiddleware:
    """Tests for CORS header configuration."""

    def test_wildcard_origin_reflected(self, client: TestClient) -> None:
        """With cors_origins=['*'], any origin header is accepted."""
        response = client.get("/health", headers={"Origin": "http://example.com"})
        acao = response.headers.get("access-control-allow-origin")
        # Starlette reflects the specific origin or echoes '*'
        assert acao in ("*", "http://example.com")

    def test_specific_allowed_origin_reflected(self) -> None:
        """A configured allowed origin is reflected in the ACAO header."""
        app = _make_app(cors_origins=["http://trusted.com"])
        c = TestClient(create_api(app), raise_server_exceptions=False)
        response = c.get("/health", headers={"Origin": "http://trusted.com"})
        assert response.headers.get("access-control-allow-origin") == "http://trusted.com"

    def test_disallowed_origin_not_reflected(self) -> None:
        """An origin not in the allowed list is not reflected."""
        app = _make_app(cors_origins=["http://trusted.com"])
        c = TestClient(create_api(app), raise_server_exceptions=False)
        response = c.get("/health", headers={"Origin": "http://evil.com"})
        acao = response.headers.get("access-control-allow-origin", "")
        assert "evil.com" not in acao


# ---------------------------------------------------------------------------
# AuthMiddleware
# ---------------------------------------------------------------------------


class _AllowAuth(AuthPlugin):
    """Always grants access."""

    def authenticate(self, request: Request) -> bool:
        return True


class _DenyAuth(AuthPlugin):
    """Always denies access with a custom 401 message."""

    def authenticate(self, request: Request) -> bool:
        return False

    def unauthenticated_response(self) -> Response:
        return JSONResponse(status_code=401, content={"detail": "Access denied."})


class _ApiKeyAuth(AuthPlugin):
    """Grants access only when the X-API-Key header matches."""

    def __init__(self, key: str) -> None:
        self._key = key

    def authenticate(self, request: Request) -> bool:
        return request.headers.get("X-API-Key") == self._key


class TestAuthMiddleware:
    """Tests for the developer-provided authentication hook."""

    def test_allow_plugin_passes_predict(self) -> None:
        """With an always-allow plugin, POST /predict succeeds."""
        app = _make_app()
        c = TestClient(create_api(app, auth_plugin=_AllowAuth()), raise_server_exceptions=False)
        response = c.post("/predict", files={"visual": ("f.jpg", b"\xff", "image/jpeg")})
        assert response.status_code == 200

    def test_deny_plugin_blocks_predict(self) -> None:
        """With an always-deny plugin, POST /predict returns 401."""
        app = _make_app()
        c = TestClient(create_api(app, auth_plugin=_DenyAuth()), raise_server_exceptions=False)
        response = c.post("/predict", files={"visual": ("f.jpg", b"\xff", "image/jpeg")})
        assert response.status_code == 401
        assert response.json()["detail"] == "Access denied."

    def test_health_bypasses_auth_by_default(self) -> None:
        """GET /health is excluded from auth by default."""
        app = _make_app()
        c = TestClient(create_api(app, auth_plugin=_DenyAuth()), raise_server_exceptions=False)
        assert c.get("/health").status_code == 200

    def test_info_bypasses_auth_by_default(self) -> None:
        """GET /info is excluded from auth by default."""
        app = _make_app()
        c = TestClient(create_api(app, auth_plugin=_DenyAuth()), raise_server_exceptions=False)
        assert c.get("/info").status_code == 200

    def test_custom_exclude_paths_respected(self) -> None:
        """Custom exclude_paths override the default excluded set."""
        app = _make_app()
        c = TestClient(
            create_api(app, auth_plugin=_DenyAuth(), auth_exclude_paths=frozenset({"/predict"})),
            raise_server_exceptions=False,
        )
        # /predict is now excluded → allowed
        response = c.post("/predict", files={"visual": ("f.jpg", b"\xff", "image/jpeg")})
        assert response.status_code == 200
        # /health is NOT in custom exclude → blocked
        assert c.get("/health").status_code == 401

    def test_api_key_auth_correct_key_allowed(self) -> None:
        """Correct API key header passes the ApiKeyAuth plugin."""
        app = _make_app()
        c = TestClient(
            create_api(app, auth_plugin=_ApiKeyAuth("secret")),
            raise_server_exceptions=False,
        )
        response = c.post(
            "/predict",
            files={"visual": ("f.jpg", b"\xff", "image/jpeg")},
            headers={"X-API-Key": "secret"},
        )
        assert response.status_code == 200

    def test_api_key_auth_wrong_key_rejected(self) -> None:
        """Wrong API key header is rejected by the ApiKeyAuth plugin."""
        app = _make_app()
        c = TestClient(
            create_api(app, auth_plugin=_ApiKeyAuth("secret")),
            raise_server_exceptions=False,
        )
        response = c.post(
            "/predict",
            files={"visual": ("f.jpg", b"\xff", "image/jpeg")},
            headers={"X-API-Key": "wrong"},
        )
        assert response.status_code == 401

    def test_no_auth_plugin_means_no_restriction(self, client: TestClient) -> None:
        """Without an auth plugin, all endpoints are freely accessible."""
        response = client.post("/predict", files={"visual": ("f.jpg", b"\xff", "image/jpeg")})
        assert response.status_code == 200
