"""Unit tests for the image modality slice.

Tests cover:
- ImageProcessor: validate() and preprocess() with synthetic images and bad inputs.
- ImageCleaner: grayscale→RGB, RGBA→RGB, resize, normalisation.
- ImageAnonymizer: pass-through contract.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest

from apmoe.core.exceptions import ModalityError
from apmoe.core.types import ModalityData
from apmoe.modality.builtin.image import ImageProcessor
from apmoe.processing.builtin.image_anonymizers import ImageAnonymizer
from apmoe.processing.builtin.image_cleaners import ImageCleaner


# ---------------------------------------------------------------------------
# Helpers — synthetic image bytes
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    """Return minimal valid PNG bytes for a solid-colour image.

    Uses stdlib only so no Pillow is needed at import time for this helper.
    Falls back to Pillow when mode != 'RGB' (for simplicity).
    """
    from PIL import Image  # type: ignore[import-untyped]

    channels = {"RGB": 3, "RGBA": 4, "L": 1}.get(mode, 3)
    colour: int | tuple[int, ...] = (128,) * channels if channels > 1 else 128
    img = Image.new(mode, (width, height), colour)  # type: ignore[arg-type]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_modality_data(arr: np.ndarray, mode: str = "RGB") -> ModalityData:
    """Wrap an array into a ModalityData matching ImageProcessor output."""
    h, w = arr.shape[:2]
    c = arr.shape[-1] if arr.ndim == 3 else 1
    return ModalityData(
        modality="image",
        data=arr,
        metadata={"width": w, "height": h, "channels": c, "mode": mode},
    )


# ---------------------------------------------------------------------------
# ImageProcessor tests
# ---------------------------------------------------------------------------


class TestImageProcessor:
    proc = ImageProcessor()

    def test_modality_name(self) -> None:
        assert self.proc.modality_name == "image"

    def test_validate_valid_rgb_png(self) -> None:
        png = _make_png_bytes(32, 32, "RGB")
        assert self.proc.validate(png) is True

    def test_validate_valid_rgba_png(self) -> None:
        png = _make_png_bytes(16, 16, "RGBA")
        assert self.proc.validate(png) is True

    def test_validate_valid_grayscale_png(self) -> None:
        png = _make_png_bytes(20, 20, "L")
        assert self.proc.validate(png) is True

    def test_validate_invalid_bytes(self) -> None:
        assert self.proc.validate(b"not an image") is False

    def test_validate_wrong_type(self) -> None:
        assert self.proc.validate(12345) is False

    def test_preprocess_rgb_shape_dtype(self) -> None:
        png = _make_png_bytes(64, 64, "RGB")
        result = self.proc.preprocess(png)
        assert result.modality == "image"
        assert result.data.shape == (64, 64, 3)
        assert result.data.dtype == np.uint8
        assert result.metadata["width"] == 64
        assert result.metadata["height"] == 64
        assert result.metadata["channels"] == 3
        assert result.metadata["mode"] == "RGB"

    def test_preprocess_rgba_shape(self) -> None:
        png = _make_png_bytes(32, 32, "RGBA")
        result = self.proc.preprocess(png)
        assert result.data.shape == (32, 32, 4)
        assert result.metadata["channels"] == 4

    def test_preprocess_grayscale_shape(self) -> None:
        png = _make_png_bytes(20, 20, "L")
        result = self.proc.preprocess(png)
        # PIL L-mode arrays are 2-D
        assert result.data.ndim == 2
        assert result.metadata["mode"] == "L"

    def test_preprocess_bad_bytes_raises(self) -> None:
        with pytest.raises(ModalityError):
            self.proc.preprocess(b"garbage")

    def test_preprocess_wrong_type_raises(self) -> None:
        with pytest.raises(ModalityError):
            self.proc.preprocess(9999)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ImageCleaner tests
# ---------------------------------------------------------------------------


class TestImageCleaner:
    cleaner = ImageCleaner()

    def _run(self, arr: np.ndarray, mode: str = "RGB") -> ModalityData:
        md = _make_modality_data(arr, mode)
        return self.cleaner.clean(md)

    def test_rgb_resize_and_normalise(self) -> None:
        arr = np.full((400, 300, 3), 128, dtype=np.uint8)
        result = self._run(arr)
        assert result.data.shape == (200, 200, 3)
        assert result.data.dtype == np.float32
        # 128 / 255 ≈ 0.502 — check values stay within [0, 1]
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0
        # All pixels should be approximately 128/255
        np.testing.assert_allclose(result.data, 128.0 / 255.0, atol=0.01)

    def test_normalisation_range(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:50, :] = 255
        result = self._run(arr)
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0

    def test_grayscale_converted_to_rgb(self) -> None:
        arr = np.full((50, 50), 200, dtype=np.uint8)
        result = self._run(arr, mode="L")
        assert result.data.shape == (200, 200, 3)
        assert "grayscale→RGB" in result.metadata["channel_adjustments"]

    def test_rgba_alpha_dropped(self) -> None:
        arr = np.full((60, 60, 4), 150, dtype=np.uint8)
        result = self._run(arr, mode="RGBA")
        assert result.data.shape == (200, 200, 3)
        assert "RGBA→RGB" in result.metadata["channel_adjustments"]

    def test_rgb_no_channel_adjustment(self) -> None:
        arr = np.full((80, 80, 3), 100, dtype=np.uint8)
        result = self._run(arr)
        assert result.metadata["channel_adjustments"] == []

    def test_metadata_updated(self) -> None:
        arr = np.full((300, 400, 3), 0, dtype=np.uint8)
        result = self._run(arr)
        assert result.metadata["output_size"] == (200, 200)
        assert result.metadata["normalised"] is True
        assert result.metadata["input_size"] == (400, 300)

    def test_too_small_raises(self) -> None:
        arr = np.full((2, 2, 3), 128, dtype=np.uint8)
        with pytest.raises(ModalityError):
            self._run(arr)

    def test_non_array_raises(self) -> None:
        md = ModalityData(modality="image", data="not an array")
        with pytest.raises(ModalityError):
            self.cleaner.clean(md)


# ---------------------------------------------------------------------------
# ImageAnonymizer tests
# ---------------------------------------------------------------------------


class TestImageAnonymizer:
    anon = ImageAnonymizer()

    def test_pass_through(self) -> None:
        arr = np.random.rand(200, 200, 3).astype(np.float32)
        md = _make_modality_data(arr)
        result = self.anon.anonymize(md)
        assert result is md  # exact same object — pure pass-through

    def test_modality_preserved(self) -> None:
        arr = np.zeros((200, 200, 3), dtype=np.float32)
        md = ModalityData(modality="image", data=arr, metadata={"foo": "bar"})
        result = self.anon.anonymize(md)
        assert result.modality == "image"
        assert result.metadata["foo"] == "bar"
