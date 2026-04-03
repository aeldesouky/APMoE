"""Built-in image modality processor for the APMoE framework.

Parses raw image bytes (JPEG, PNG, BMP, WEBP, …) or a filesystem path into
a NumPy ``uint8`` array that the downstream processing chain can clean,
anonymise, and hand off to a :class:`~apmoe.experts.builtin.FaceAgeExpert`.

Supported input formats
-----------------------

**Raw bytes** (primary — direct upload via ``POST /predict``):

    Any image format supported by `Pillow` (JPEG, PNG, BMP, WEBP, GIF, …).
    The bytes are decoded in-memory without touching the filesystem.

**File-path string** (secondary — batch inference via CLI):

    A filesystem path to an image file (absolute or relative to the working
    directory at inference time).  The file is read and decoded by Pillow.

Both inputs produce the same :class:`~apmoe.core.types.ModalityData` output:
a ``numpy.ndarray`` of shape ``(H, W, C)`` with ``dtype=uint8`` and
``C ∈ {1, 3, 4}`` depending on the original image mode.  Channel handling
(grayscale → RGB, RGBA → RGB) is performed by the downstream
:class:`~apmoe.processing.builtin.image_cleaners.ImageCleaner`.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from apmoe.core.exceptions import ModalityError
from apmoe.core.types import ModalityData
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import modality_registry


@modality_registry.register("image")
class ImageProcessor(ModalityProcessor):
    """Parse raw image bytes or a file path into a NumPy uint8 array.

    The ``data`` attribute of the returned
    :class:`~apmoe.core.types.ModalityData` is a ``numpy.ndarray`` of shape
    ``(H, W, C)`` with ``dtype=uint8``.  No resizing or normalisation is
    performed here — that is the responsibility of
    :class:`~apmoe.processing.builtin.image_cleaners.ImageCleaner`.

    Registered as ``"image"`` in
    :data:`~apmoe.modality.factory.modality_registry`.
    """

    @property
    def modality_name(self) -> str:
        """Return the canonical modality name ``"image"``."""
        return "image"

    def validate(self, data: object) -> bool:
        """Return ``True`` if *data* can be decoded as an image.

        Args:
            data: ``bytes`` (raw image file content) or ``str`` / ``Path``
                (filesystem path to an image file).

        Returns:
            ``True`` if the input is parseable as an image with at least
            one valid pixel; ``False`` otherwise.
        """
        try:
            arr = self._decode(data)
            return arr.size > 0
        except Exception:
            return False

    def preprocess(self, data: object) -> ModalityData:
        """Decode *data* into a :class:`~apmoe.core.types.ModalityData`.

        Args:
            data: Raw image — ``bytes`` (JPEG/PNG/…) or ``str``/``Path``
                (filesystem path to an image file).

        Returns:
            A :class:`~apmoe.core.types.ModalityData` with:

            * ``modality = "image"``
            * ``data`` = ``numpy.ndarray`` of shape ``(H, W, C)``,
              ``dtype=uint8``, colour order is as decoded by Pillow (RGB for
              most formats).
            * ``metadata["width"]`` = original image width (pixels).
            * ``metadata["height"]`` = original image height (pixels).
            * ``metadata["channels"]`` = number of channels.
            * ``metadata["mode"]`` = Pillow image mode string (e.g. ``"RGB"``,
              ``"L"``, ``"RGBA"``).

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If decoding fails
                or the result contains no pixels.
        """
        try:
            arr, meta = self._decode_with_meta(data)
        except Exception as exc:
            raise ModalityError(
                f"ImageProcessor: cannot decode input — {exc}",
                context={"input_type": type(data).__name__},
            ) from exc

        if arr.size == 0:
            raise ModalityError(
                "ImageProcessor: decoded image contains no pixels.",
                context={"input_type": type(data).__name__},
            )

        return ModalityData(
            modality=self.modality_name,
            data=arr,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(data: object) -> np.ndarray:
        """Decode *data* and return a uint8 NumPy array (no metadata)."""
        arr, _ = ImageProcessor._decode_with_meta(data)
        return arr

    @staticmethod
    def _decode_with_meta(data: object) -> tuple[np.ndarray, dict[str, Any]]:
        """Decode *data* → ``(array, metadata_dict)``.

        Args:
            data: ``bytes``, ``str``, or ``Path``.

        Returns:
            Tuple of ``(uint8 ndarray of shape H×W×C, metadata dict)``.

        Raises:
            ValueError: If the format is unrecognised or Pillow cannot open it.
        """
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ModalityError(
                "Pillow is required for ImageProcessor.  "
                "Install it with: pip install pillow",
                context={},
            ) from exc

        if isinstance(data, (bytes, bytearray)):
            img = Image.open(io.BytesIO(data))
        elif isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise ValueError(f"Image file not found: {path}")
            img = Image.open(path)
        else:
            raise ValueError(
                f"Expected bytes, str, or Path, got {type(data).__name__}."
            )

        mode: str = img.mode
        width, height = img.size
        arr: np.ndarray = np.array(img)

        meta: dict[str, Any] = {
            "width": width,
            "height": height,
            "channels": arr.shape[-1] if arr.ndim == 3 else 1,
            "mode": mode,
        }
        return arr, meta
