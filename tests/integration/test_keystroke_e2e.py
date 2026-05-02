"""End-to-end integration test for the keystroke age-prediction pipeline.

Boots the REAL APMoEApp (loads the actual ONNX model from disk), spins up a
FastAPI TestClient, and exercises the full pipeline via ``POST /predict`` with
a JSON body.  No mocks — every component runs for real.

Requirements
------------
* ``weights/keystroke_age_expert.onnx`` must exist relative to the project root.
* ``weights/keystroke_constants.json`` must exist relative to the project root.
* ``configs/keystroke.json`` must point to the above files.

Run with::

    uv run pytest tests/integration/test_keystroke_e2e.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apmoe.core.app import APMoEApp
from apmoe.serving.app_factory import create_api

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "keystroke.json"

# Age labels the model can emit — must match keystroke_constants.json
VALID_AGE_LABELS = {"18-25", "26-35", "36-45", "46+"}

# ---------------------------------------------------------------------------
# Simulated keystroke session
#
# Each triple is [key_code, next_key_code, timing_ms].
#   key2 == 0  →  hold time for key1   ("dur_<key1>")
#   key2 != 0  →  flight time key1→key2 ("dig_<key1>_<key2>")
#
# This session simulates a short typing passage with realistic timings.
# Keys use standard virtual-key codes (e.g. 84=T, 72=H, 69=E, 32=Space).
# ---------------------------------------------------------------------------

# fmt: off
SAMPLE_SESSION = [
    # T-H-E (hold + digraphs)
    [84,  0,   93.2],   # dur_84
    [84,  72, 102.5],   # dig_84_72
    [72,  0,   88.7],   # dur_72
    [72,  69, 130.1],   # dig_72_69
    [69,  0,   91.0],   # dur_69
    [69,  32, 189.3],   # dig_69_32  (space after "the")

    # Q-U-I-C-K
    [81,  0,   97.4],   # dur_81
    [81,  85, 115.2],   # dig_81_85
    [85,  0,   89.6],   # dur_85
    [85,  73, 108.7],   # dig_85_73
    [73,  0,   94.1],   # dur_73
    [73,  67, 121.3],   # dig_73_67
    [67,  0,   86.5],   # dur_67
    [67,  75, 133.0],   # dig_67_75
    [75,  0,   92.8],   # dur_75
    [75,  32, 175.6],   # dig_75_32  (space)

    # B-R-O-W-N
    [66,  0,  101.3],   # dur_66
    [66,  82, 118.4],   # dig_66_82
    [82,  0,   90.2],   # dur_82
    [82,  79, 107.9],   # dig_82_79
    [79,  0,   88.4],   # dur_79
    [79,  87, 124.6],   # dig_79_87
    [87,  0,   96.3],   # dur_87
    [87,  78, 109.1],   # dig_87_78
    [78,  0,   93.7],   # dur_78
    [78,  32, 182.5],   # dig_78_32  (space)

    # F-O-X
    [70,  0,   99.1],   # dur_70
    [70,  79, 113.2],   # dig_70_79
    [79,  0,   87.5],   # dur_79 (second observation)
    [79,  88, 131.4],   # dig_79_88
    [88,  0,   94.6],   # dur_88
    [88,  13, 221.7],   # dig_88_13  (Enter)

    # J-U-M-P-S
    [74,  0,   95.3],   # dur_74
    [74,  85, 117.8],   # dig_74_85
    [85,  0,   91.2],   # dur_85 (second observation)
    [85,  77, 112.6],   # dig_85_77
    [77,  0,   98.9],   # dur_77
    [77,  80, 126.3],   # dig_77_80
    [80,  0,   87.1],   # dur_80
    [80,  83, 108.4],   # dig_80_83
    [83,  0,   93.5],   # dur_83
    [83,  32, 179.2],   # dig_83_32  (space)

    # O-V-E-R
    [79,  0,   89.7],   # dur_79 (third observation)
    [79,  86, 122.5],   # dig_79_86
    [86,  0,   96.8],   # dur_86
    [86,  69, 110.3],   # dig_86_69
    [69,  0,   90.4],   # dur_69 (second observation)
    [69,  82, 127.9],   # dig_69_82
    [82,  0,   88.6],   # dur_82 (second observation)
    [82,  32, 184.1],   # dig_82_32  (space)

    # T-H-E  (repeated phrase — more observations per feature)
    [84,  0,   91.5],
    [84,  72, 104.7],
    [72,  0,   87.3],
    [72,  69, 128.9],
    [69,  0,   92.6],
    [69,  32, 191.0],

    # L-A-Z-Y  D-O-G
    [76,  0,  100.2],   # dur_76
    [76,  65, 119.6],   # dig_76_65
    [65,  0,   95.4],   # dur_65
    [65,  90, 132.1],   # dig_65_90
    [90,  0,   89.3],   # dur_90
    [90,  89, 114.7],   # dig_90_89
    [89,  0,   97.6],   # dur_89
    [89,  32, 177.4],   # dig_89_32  (space)
    [68,  0,  102.5],   # dur_68
    [68,  79, 120.8],   # dig_68_79
    [79,  0,   86.9],   # dur_79 (more observations)
    [79,  71, 109.3],   # dig_79_71
    [71,  0,   94.2],   # dur_71
    [71,  13, 215.6],   # dig_71_13  (Enter)
]
# fmt: on

# ---------------------------------------------------------------------------
# Session-scoped fixtures  (model loads once per pytest run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def keystroke_client() -> TestClient:  # type: ignore[return]
    """Bootstrap the real APMoEApp and return a live TestClient.

    Changes the working directory to the project root so that relative
    weight paths in the config resolve correctly, then restores the
    original directory after the test session ends.
    """
    if not CONFIG_PATH.exists():
        pytest.skip(f"Config file not found: {CONFIG_PATH}")

    onnx_path = PROJECT_ROOT / "weights" / "keystroke_age_expert.onnx"
    if not onnx_path.exists():
        pytest.skip(f"ONNX weights not found: {onnx_path}")

    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        app = APMoEApp.from_config(CONFIG_PATH)
        api = create_api(app)
        with TestClient(api, raise_server_exceptions=True) as client:
            yield client
    finally:
        os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# /health — expert loaded check
# ---------------------------------------------------------------------------


class TestKeystrokeHealth:
    """Verify the keystroke expert is reported as loaded."""

    def test_health_returns_200(self, keystroke_client: TestClient) -> None:
        """/health must be 200 when the model is loaded."""
        response = keystroke_client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_healthy(self, keystroke_client: TestClient) -> None:
        """/health must report 'healthy'."""
        body = keystroke_client.get("/health").json()
        assert body["status"] == "healthy"

    def test_keystroke_expert_is_loaded(self, keystroke_client: TestClient) -> None:
        """The keystroke_age_expert must be flagged as loaded (True)."""
        experts = keystroke_client.get("/health").json()["experts"]
        assert "keystroke_age_expert" in experts, (
            f"keystroke_age_expert not in health response: {experts}"
        )
        assert experts["keystroke_age_expert"] is True


# ---------------------------------------------------------------------------
# /info — metadata check
# ---------------------------------------------------------------------------


class TestKeystrokeInfo:
    """Verify /info reflects the keystroke expert configuration."""

    def test_info_returns_200(self, keystroke_client: TestClient) -> None:
        response = keystroke_client.get("/info")
        assert response.status_code == 200

    def test_info_contains_keystroke_modality(self, keystroke_client: TestClient) -> None:
        body = keystroke_client.get("/info").json()
        assert "keystroke" in body["modalities"]

    def test_info_contains_expert_entry(self, keystroke_client: TestClient) -> None:
        body = keystroke_client.get("/info").json()
        expert_names = [e["name"] for e in body["experts"]]
        assert "keystroke_age_expert" in expert_names

    def test_expert_info_has_num_features(self, keystroke_client: TestClient) -> None:
        """The expert must report a positive number of features."""
        body = keystroke_client.get("/info").json()
        expert = next(e for e in body["experts"] if e["name"] == "keystroke_age_expert")
        num_features = expert.get("num_features")
        assert isinstance(num_features, int) and num_features > 0, (
            f"Expected a positive integer for num_features, got {num_features!r}"
        )


# ---------------------------------------------------------------------------
# POST /predict — real inference
# ---------------------------------------------------------------------------


class TestKeystrokePredict:
    """Full end-to-end inference tests against the live ONNX model."""

    def test_predict_returns_200(self, keystroke_client: TestClient) -> None:
        """A valid keystroke session must yield HTTP 200."""
        response = keystroke_client.post(
            "/predict",
            json={"keystroke": SAMPLE_SESSION},
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )

    def test_prediction_has_required_keys(self, keystroke_client: TestClient) -> None:
        """Response must include all standard prediction fields."""
        body = keystroke_client.post(
            "/predict", json={"keystroke": SAMPLE_SESSION}
        ).json()
        for key in (
            "predicted_age",
            "confidence",
            "confidence_interval",
            "per_expert_outputs",
            "skipped_experts",
            "metadata",
        ):
            assert key in body, f"Missing key: {key}"

    def test_predicted_age_is_positive_number(self, keystroke_client: TestClient) -> None:
        """predicted_age must be a positive finite number."""
        body = keystroke_client.post(
            "/predict", json={"keystroke": SAMPLE_SESSION}
        ).json()
        age = body["predicted_age"]
        assert isinstance(age, (int, float))
        assert age > 0, f"predicted_age must be positive, got {age}"

    def test_confidence_is_in_unit_interval(self, keystroke_client: TestClient) -> None:
        """confidence must be in [0, 1]."""
        body = keystroke_client.post(
            "/predict", json={"keystroke": SAMPLE_SESSION}
        ).json()
        conf = body["confidence"]
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0, f"confidence out of range: {conf}"

    def test_per_expert_output_has_valid_label(self, keystroke_client: TestClient) -> None:
        """The keystroke expert must report one of the four valid age groups."""
        body = keystroke_client.post(
            "/predict", json={"keystroke": SAMPLE_SESSION}
        ).json()
        assert len(body["per_expert_outputs"]) >= 1, "Expected at least one expert output"
        expert_out = body["per_expert_outputs"][0]
        assert expert_out["expert_name"] == "keystroke_age_expert"
        # metadata carries the predicted group label
        label = expert_out["metadata"].get("predicted_group")
        assert label in VALID_AGE_LABELS, (
            f"Got unexpected age label: {label!r}. Expected one of {VALID_AGE_LABELS}"
        )

    def test_no_experts_skipped_for_keystroke_input(
        self, keystroke_client: TestClient
    ) -> None:
        """When keystroke data is provided, no experts should be skipped."""
        body = keystroke_client.post(
            "/predict", json={"keystroke": SAMPLE_SESSION}
        ).json()
        assert body["skipped_experts"] == [], (
            f"Experts were unexpectedly skipped: {body['skipped_experts']}"
        )

    def test_predict_with_minimal_session(self, keystroke_client: TestClient) -> None:
        """Even a very short session (1 hold + 1 digraph) must return 200.

        Missing features are filled with training-set medians, so the model
        should still produce a valid prediction.
        """
        minimal = [[84, 0, 95.0], [84, 72, 105.0]]
        response = keystroke_client.post("/predict", json={"keystroke": minimal})
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_age"] > 0
        assert 0.0 <= body["confidence"] <= 1.0

    def test_predict_with_ikdd_string_format(self, keystroke_client: TestClient) -> None:
        """IKDD text passed as a JSON string must also produce a valid prediction."""
        # Build IKDD text: "key1-key2,ms1,ms2,...\n"
        ikdd_lines = [
            "84-0,93.2,91.5",       # dur_84 (two observations)
            "84-72,102.5,104.7",    # dig_84_72
            "72-0,88.7,87.3",       # dur_72
            "72-69,130.1,128.9",    # dig_72_69
            "69-0,91.0,92.6",       # dur_69
            "69-32,189.3,191.0",    # dig_69_32
        ]
        ikdd_text = "\n".join(ikdd_lines)
        response = keystroke_client.post(
            "/predict", json={"keystroke": ikdd_text}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_age"] > 0

    def test_predict_with_precomputed_features_dict(
        self, keystroke_client: TestClient
    ) -> None:
        """A pre-computed feature dict (keyed by feature name) must be accepted."""
        features = {
            "dur_84": [93.2, 91.5],
            "dig_84_72": [102.5, 104.7],
            "dur_72": [88.7, 87.3],
            "dig_72_69": [130.1, 128.9],
            "dur_69": [91.0, 92.6],
            "dig_69_32": [189.3, 191.0],
        }
        response = keystroke_client.post("/predict", json={"keystroke": features})
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_age"] > 0

    def test_invalid_json_root_returns_422(self, keystroke_client: TestClient) -> None:
        """A JSON array at root level (not a dict) must return 422."""
        response = keystroke_client.post(
            "/predict",
            content=b"[[84,0,95.0]]",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_keystroke_key_results_in_skipped_expert(
        self, keystroke_client: TestClient
    ) -> None:
        """Sending an unknown modality means no experts can run → 503."""
        response = keystroke_client.post(
            "/predict",
            json={"unknown_modality": [[1, 2, 3.0]]},
        )
        # The keystroke expert requires 'keystroke' data; without it, the
        # pipeline has no runnable experts → PipelineError → 503.
        assert response.status_code == 503
