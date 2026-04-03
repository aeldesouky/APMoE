"""End-to-end integration test for the image modality + FaceAgeExpert.

Mirrors tests/integration/test_keystroke_e2e.py.

Uses a mock FaceAgeExpert (patched Keras model) to exercise the full pipeline:
    ImageProcessor → ImageCleaner → ImageAnonymizer → FaceAgeExpert → Prediction

No real Keras model, no network I/O, no filesystem access (besides the
synthetic PNG bytes generated in-memory).
"""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from apmoe.core.types import ExpertOutput, ModalityData, Prediction
from apmoe.experts.builtin import FaceAgeExpert
from apmoe.modality.builtin.image import ImageProcessor
from apmoe.processing.builtin.image_anonymizers import ImageAnonymizer
from apmoe.processing.builtin.image_cleaners import ImageCleaner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 64, height: int = 64, mode: str = "RGB") -> bytes:
    """Return valid in-memory PNG bytes."""
    from PIL import Image  # type: ignore[import-untyped]

    channels = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, 3)
    colour: Any = (100,) * channels if channels > 1 else 100
    img = Image.new(mode, (width, height), colour)  # type: ignore[arg-type]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_mock_expert(predicted_age: float = 30.0) -> FaceAgeExpert:
    """Return a FaceAgeExpert with a mock Keras model injected."""
    expert = FaceAgeExpert()
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[predicted_age]], dtype=np.float32)
    expert._model = mock_model
    return expert


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestFaceAgeE2E:
    """Full pipeline: bytes → ModalityData → cleaned → anonymised → ExpertOutput."""

    def _run_pipeline(
        self,
        image_bytes: bytes,
        expected_age: float = 30.0,
    ) -> ExpertOutput:
        proc = ImageProcessor()
        cleaner = ImageCleaner()
        anon = ImageAnonymizer()
        expert = _build_mock_expert(expected_age)

        # Stage 1: modality processor
        modal_data = proc.preprocess(image_bytes)
        assert isinstance(modal_data, ModalityData)

        # Stage 2: cleaner
        cleaned = cleaner.clean(modal_data)
        assert cleaned.data.shape == (200, 200, 3)
        assert cleaned.data.dtype == np.float32

        # Stage 3: anonymizer
        anonymised = anon.anonymize(cleaned)
        assert anonymised is cleaned  # pass-through

        # Stage 4: expert inference
        output = expert.predict({"image": anonymised})
        return output

    def test_rgb_image_full_pipeline(self) -> None:
        png = _make_png_bytes(64, 64, "RGB")
        output = self._run_pipeline(png, expected_age=28.7)
        assert output.expert_name == "face_age_expert"
        assert output.consumed_modalities == ["image"]
        assert output.predicted_age == float(round(28.7))
        assert output.confidence == 1.0
        assert "raw_output" in output.metadata
        assert "rounded_age" in output.metadata

    def test_grayscale_image_full_pipeline(self) -> None:
        png = _make_png_bytes(80, 80, "L")
        output = self._run_pipeline(png, expected_age=42.3)
        assert output.predicted_age == float(round(42.3))

    def test_rgba_image_full_pipeline(self) -> None:
        png = _make_png_bytes(100, 100, "RGBA")
        output = self._run_pipeline(png, expected_age=22.1)
        assert output.predicted_age == float(round(22.1))

    def test_large_image_resized_correctly(self) -> None:
        png = _make_png_bytes(800, 600, "RGB")
        output = self._run_pipeline(png, expected_age=55.0)
        assert output.predicted_age == 55.0

    def test_age_clamped_below(self) -> None:
        """Extremely negative model output is clamped to 1."""
        png = _make_png_bytes(32, 32, "RGB")
        output = self._run_pipeline(png, expected_age=-50.0)
        assert output.predicted_age == 1.0

    def test_age_clamped_above(self) -> None:
        """Extremely large model output is clamped to 120."""
        png = _make_png_bytes(32, 32, "RGB")
        output = self._run_pipeline(png, expected_age=999.0)
        assert output.predicted_age == 120.0

    def test_batch_dim_passed_to_model(self) -> None:
        """Verifies model.predict is called with shape (1, 200, 200, 3)."""
        png = _make_png_bytes(64, 64, "RGB")
        proc = ImageProcessor()
        cleaner = ImageCleaner()
        anon = ImageAnonymizer()
        expert = _build_mock_expert(30.0)

        cleaned = anon.anonymize(cleaner.clean(proc.preprocess(png)))
        expert.predict({"image": cleaned})

        call_args = expert._model.predict.call_args
        batch = call_args[0][0]
        assert batch.shape == (1, 200, 200, 3)
        assert batch.dtype == np.float32

    def test_model_not_loaded_raises(self) -> None:
        from apmoe.core.exceptions import ExpertError

        expert = FaceAgeExpert()
        dummy = ModalityData(
            modality="image",
            data=np.zeros((200, 200, 3), dtype=np.float32),
        )
        with pytest.raises(ExpertError, match="not loaded"):
            expert.predict({"image": dummy})

    def test_output_metadata_model_name(self) -> None:
        png = _make_png_bytes(64, 64, "RGB")
        output = self._run_pipeline(png, expected_age=35.0)
        assert "Face Age Prediction" in output.metadata["model"]


# ---------------------------------------------------------------------------
# FaceAgeExpert.get_info()
# ---------------------------------------------------------------------------


class TestFaceAgeExpertInfo:
    def test_get_info_not_loaded(self) -> None:
        expert = FaceAgeExpert()
        info = expert.get_info()
        assert info["name"] == "face_age_expert"
        assert info["modalities"] == ["image"]
        assert info["loaded"] is False
        assert info["input_shape"] == "(1, 200, 200, 3)"
        assert info["output_type"] == "regressor"

    def test_get_info_loaded(self) -> None:
        expert = _build_mock_expert()
        info = expert.get_info()
        assert info["loaded"] is True

    def test_is_loaded_false_when_no_model(self) -> None:
        assert FaceAgeExpert().is_loaded is False

    def test_is_loaded_true_when_model_present(self) -> None:
        assert _build_mock_expert().is_loaded is True
