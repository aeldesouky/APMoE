"""LLM-compatible image processing strategies for use with remote vision experts.

These processors are **only** intended for configs that route the image modality
to a remote LLM / vision-API endpoint (e.g. LM Studio, HuggingFace Inference
API, OpenAI Vision).  They must **not** be used in local-inference pipelines —
the built-in experts (:class:`~apmoe.experts.builtin.FaceAgeExpert`) expect the
standard ``float32 (200, 200, 3)`` array produced by
:class:`~apmoe.processing.builtin.image_cleaners.ImageCleaner`.

Provided classes
----------------
* :class:`Base64ImageCleaner` — resizes and JPEG-compresses an image before
  base64-encoding it so it fits within an LLM context window.
* :class:`PassthroughImageAnonymizer` — no-op anonymizer that keeps the
  base64 string intact.

Typical config usage::

    {
      "name": "image",
      "processor": "apmoe.modality.builtin.image.ImageProcessor",
      "pipeline": {
        "cleaner":    "apmoe.processing.llm.image.Base64ImageCleaner",
        "anonymizer": "apmoe.processing.llm.image.PassthroughImageAnonymizer"
      }
    }
"""

from __future__ import annotations

import numpy as np

from apmoe.core.exceptions import ModalityError
from apmoe.core.types import ModalityData
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    anonymizer_registry,
    cleaner_registry,
)


# ---------------------------------------------------------------------------
# Base64ImageCleaner
# ---------------------------------------------------------------------------


@cleaner_registry.register("base64_image_cleaner")
class Base64ImageCleaner(CleanerStrategy):
    """Resize, JPEG-compress, and base64-encode an image for remote LLM dispatch.

    Transforms the raw image (bytes or decoded numpy array from
    :class:`~apmoe.modality.builtin.image.ImageProcessor`) into a compact
    base64-encoded JPEG string that can be embedded directly into an LLM
    request body via the ``request_template`` field of
    :class:`~apmoe.experts.remote.RemoteExpert`.

    Processing steps
    ----------------
    1. Open/decode the image (bytes → Pillow, or numpy → Pillow).
    2. Convert to ``RGB`` (drops alpha, promotes grayscale).
    3. Resize so the longest side is at most :attr:`MAX_SIZE` pixels,
       preserving the original aspect ratio.
    4. Re-encode as JPEG at :attr:`JPEG_QUALITY`.
    5. Base64-encode to an ASCII ``str``.

    The resulting string is typically ~1–3 KB and fits well within the 4096-
    token context windows common in locally-run quantised vision models.

    Registered as ``"base64_image_cleaner"`` in
    :data:`~apmoe.processing.base.cleaner_registry`.

    .. warning::
        Do **not** combine with the built-in
        :class:`~apmoe.experts.builtin.FaceAgeExpert`.  That expert requires
        the ``float32 (200, 200, 3)`` array produced by
        :class:`~apmoe.processing.builtin.image_cleaners.ImageCleaner`.
        Use this cleaner **only** with
        :class:`~apmoe.experts.remote.RemoteExpert` pointing at a vision LLM.
    """

    #: Maximum pixel length for the longest side after resizing.
    #: 160 px keeps base64 output around 2.5 KB / ~1900 tokens —
    #: well within a 4096-token context at JPEG_QUALITY 35.
    MAX_SIZE: int = 160

    #: JPEG quality (1–95) used when re-encoding.
    #: Lower values reduce file size; 35 is a good balance for LLM inputs
    #: where photographic fidelity matters less than token budget.
    JPEG_QUALITY: int = 35

    def clean(self, data: ModalityData) -> ModalityData:
        """Resize, compress, and base64-encode the image.

        Args:
            data: :class:`~apmoe.core.types.ModalityData` whose ``data``
                attribute is raw image bytes or a decoded ``numpy.ndarray``.

        Returns:
            A new :class:`~apmoe.core.types.ModalityData` whose ``data``
            is a plain ASCII ``str`` — the base64-encoded JPEG.
            ``metadata`` is updated with ``base64_bytes``, ``jpeg_quality``,
            and ``max_size`` for traceability.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If Pillow is not
                installed, the input type is unsupported, or encoding fails.
        """
        import base64
        import io

        try:
            from PIL import Image as PILImage  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ModalityError(
                "Pillow is required for Base64ImageCleaner.  "
                "Install it with: pip install pillow",
                context={"modality": data.modality},
            ) from exc

        raw = data.data
        try:
            if isinstance(raw, (bytes, bytearray)):
                img = PILImage.open(io.BytesIO(bytes(raw)))
                img = img.convert("RGB")
            elif isinstance(raw, np.ndarray):
                img = PILImage.fromarray(raw.astype(np.uint8)).convert("RGB")
            else:
                raise ModalityError(
                    f"Base64ImageCleaner expects bytes or numpy.ndarray, "
                    f"got {type(raw).__name__}.",
                    context={"modality": data.modality},
                )

            # Step 3: resize — preserve aspect ratio, cap longest side
            w, h = img.size
            scale = self.MAX_SIZE / max(w, h)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)  # type: ignore[attr-defined]

            # Step 4: re-encode as compressed JPEG
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.JPEG_QUALITY, optimize=True)

            # Step 5: base64-encode
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        except ModalityError:
            raise
        except Exception as exc:
            raise ModalityError(
                f"Base64ImageCleaner: failed to encode image: {exc}",
                context={"modality": data.modality},
            ) from exc

        result = data.with_data(encoded)
        result.metadata["base64_bytes"] = len(encoded)
        result.metadata["jpeg_quality"] = self.JPEG_QUALITY
        result.metadata["max_size"] = self.MAX_SIZE
        return result


# ---------------------------------------------------------------------------
# PassthroughImageAnonymizer
# ---------------------------------------------------------------------------


@anonymizer_registry.register("passthrough_image_anonymizer")
class PassthroughImageAnonymizer(AnonymizerStrategy):
    """No-op anonymizer for images already prepared for remote LLM dispatch.

    The :class:`Base64ImageCleaner` preceding this step has already produced
    a compact, resized base64 string.  No further anonymisation is applied
    here.  The class exists solely to satisfy the mandatory
    ``Cleaner → Anonymizer`` pipeline contract.

    If your deployment requires PII removal before sending the image to a
    remote server (e.g. face blurring before calling an external API),
    replace this with a custom :class:`~apmoe.processing.base.AnonymizerStrategy`
    subclass that operates on the base64 string or on a pre-blur step before
    :class:`Base64ImageCleaner`.

    Registered as ``"passthrough_image_anonymizer"`` in
    :data:`~apmoe.processing.base.anonymizer_registry`.
    """

    def anonymize(self, data: ModalityData) -> ModalityData:
        """Return *data* unchanged.

        Args:
            data: :class:`~apmoe.core.types.ModalityData` — a base64 string
                produced by :class:`Base64ImageCleaner`.

        Returns:
            The same *data* object, unmodified.
        """
        return data
