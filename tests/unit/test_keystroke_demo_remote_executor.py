"""Tests for the standalone keystroke demo remote executor."""

from __future__ import annotations

from fastapi.testclient import TestClient
from remote_executors.keystroke_demo.app import CONTRACT_VERSION, create_app

SAMPLE_SESSION = [
    [8, 0, 95.0],
    [13, 0, 100.0],
    [65, 83, 145.0],
    [83, 68, 130.0],
    [68, 70, 125.0],
]


def _client() -> TestClient:
    return TestClient(create_app())


def test_health_reports_keystroke_mode() -> None:
    """The executor health endpoint exposes readiness and keystroke mode."""
    client = _client()

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["mode"] == "keystroke"
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["expert_loaded"] is True


def test_info_describes_remote_executor_contract() -> None:
    """The info endpoint documents the remote request/response contract."""
    client = _client()

    response = client.get("/info")

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "keystroke"
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["request_contract"]["default"]["modalities"]["keystroke"]
    assert body["response_contract"]["predicted_age"] == "float"


def test_predict_accepts_default_remote_expert_contract() -> None:
    """The executor accepts the default body sent by RemoteExpert."""
    client = _client()

    response = client.post(
        "/predict",
        json={
            "expert_name": "remote_keystroke_age_expert",
            "modalities": {"keystroke": SAMPLE_SESSION},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_age"] > 0
    assert 0 <= body["confidence"] <= 1
    assert body["metadata"]["mode"] == "keystroke"
    assert body["metadata"]["contract_version"] == CONTRACT_VERSION


def test_predict_rejects_missing_keystroke_payload() -> None:
    """A request without keystroke data fails with a clear validation error."""
    client = _client()

    response = client.post("/predict", json={"expert_name": "remote_keystroke_age_expert"})

    assert response.status_code == 422
    assert "Keystroke payload required" in response.text
