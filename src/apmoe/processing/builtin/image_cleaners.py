"""Built-in cleaner strategy for the image modality.

Implements the exact preprocessing pipeline described in
``docs/face_integration.md``:

1. Handle grayscale images (``ndim == 2``) → stack to 3 channels.
2. Handle RGBA images (``channels == 4``) → drop alpha, keep RGB.
3. Resize to **200 × 200** pixels.
4. Normalise pixel values to **[0, 1]** (divide by 255, cast to float32).

The resulting array has shape ``(200, 200, 3)`` and ``dtype=float32``,
ready for the :class:`~apmoe.experts.builtin.FaceAgeExpert` which adds the
batch dimension before calling ``model.predict``.
"""

from __future__ import annotations

import numpy as np

from apmoe.core.exceptions import ModalityError
from apmoe.core.types import ModalityData
from apmoe.processing.base import CleanerStrategy, cleaner_registry

#: Target spatial size required by the face age model.
_TARGET_SIZE: tuple[int, int] = (200, 200)

#: Minimum side length (pixels) accepted as a valid image.
_MIN_SIDE: int = 4


@cleaner_registry.register("image_cleaner")
class ImageCleaner(CleanerStrategy):
    """Preprocess a raw image array for the Face Age Expert.

    Operates on :class:`~apmoe.core.types.ModalityData` produced by
    :class:`~apmoe.modality.builtin.image.ImageProcessor`, where ``data``
    is a ``numpy.ndarray`` of shape ``(H, W)`` or ``(H, W, C)`` with
    ``dtype=uint8``.

    Processing steps (in order, matching ``docs/face_integration.md``):

    1. **Grayscale → RGB**: if ``ndim == 2``, stack the single channel
       three times so the array becomes ``(H, W, 3)``.
    2. **RGBA → RGB**: if ``channels == 4``, drop the alpha channel.
    3. **Resize to 200 × 200** using Pillow's ``LANCZOS`` filter.
    4. **Normalise to [0, 1]**: divide by 255 and cast to ``float32``.

    The ``metadata`` dict is updated with ``input_size``, ``output_size``,
    and ``channel_adjustments`` for traceability.

    Registered as ``"image_cleaner"`` in
    :data:`~apmoe.processing.base.cleaner_registry`.
    """

    def clean(self, data: ModalityData) -> ModalityData:
        """Apply the full preprocessing pipeline to *data*.

        Args:
            data: :class:`~apmoe.core.types.ModalityData` from
                :class:`~apmoe.modality.builtin.image.ImageProcessor`.
                ``data.data`` must be a ``numpy.ndarray``.

        Returns:
            A new :class:`~apmoe.core.types.ModalityData` (via
            :meth:`~apmoe.core.types.ModalityData.with_data`) whose
            ``data`` is a ``float32`` array of shape ``(200, 200, 3)``
            with values in ``[0.0, 1.0]``.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If the input is
                not a NumPy array, is too small to be a valid image, or if
                Pillow is unavailable.
        """
        try:
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ModalityError(
                "Pillow is required for ImageCleaner.  "
                "Install it with: pip install pillow",
                context={},
            ) from exc

        arr: np.ndarray = data.data
        if not isinstance(arr, np.ndarray):
            raise ModalityError(
                f"ImageCleaner expects a numpy.ndarray, got {type(arr).__name__}.",
                context={"modality": data.modality},
            )

        h, w = arr.shape[:2]
        if h < _MIN_SIDE or w < _MIN_SIDE:
            raise ModalityError(
                f"Image too small ({w}×{h}). Minimum side length is {_MIN_SIDE}px.",
                context={"modality": data.modality, "shape": arr.shape},
            )

        channel_adjustments: list[str] = []
        original_channels = arr.shape[-1] if arr.ndim == 3 else 1

        # --- Step 1: grayscale → RGB -----------------------------------------
        if arr.ndim == 2:
            arr = np.stack((arr,) * 3, axis=-1)
            channel_adjustments.append("grayscale→RGB")

        # --- Step 2: RGBA → RGB ----------------------------------------------
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
            channel_adjustments.append("RGBA→RGB")

        # --- Step 3: resize to 200 × 200 ------------------------------------
        pil_img = Image.fromarray(arr.astype(np.uint8))
        pil_img = pil_img.resize(_TARGET_SIZE, Image.LANCZOS)  # type: ignore[attr-defined]
        arr = np.array(pil_img)

        # --- Step 4: normalise to [0, 1] float32 ----------------------------
        arr = arr.astype(np.float32) / 255.0

        updated_meta = dict(data.metadata)
        updated_meta["input_size"] = (w, h)
        updated_meta["input_channels"] = original_channels
        updated_meta["output_size"] = _TARGET_SIZE
        updated_meta["output_channels"] = 3
        updated_meta["channel_adjustments"] = channel_adjustments
        updated_meta["normalised"] = True

        result = data.with_data(arr)
        result.metadata.update(updated_meta)
        return result
