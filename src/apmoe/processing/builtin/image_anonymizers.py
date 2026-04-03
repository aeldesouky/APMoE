"""Built-in anonymiser strategy for the image modality.

The Face Age Expert receives a patch-level abstraction of the face image
after the cleaner has resized and normalised it.  At this stage the image
is a 200 × 200 float32 array — no direct biometric identifiers (names,
IDs) are embedded in the pixel values, and the expert extracts only a
single age scalar from the image.

The anonymiser is therefore a deliberate **pass-through** for the current
model, consistent with the integration specification in
``docs/face_integration.md``.  It exists to:

* Satisfy the framework's mandatory ``Cleaner → Anonymiser`` pipeline
  contract.
* Provide a clear, documented extension point for teams that need
  additional privacy guarantees (e.g. face-region blurring, pixel
  perturbation, k-anonymity transforms).

To add blurring, subclass :class:`ImageAnonymizer` and override
:meth:`anonymize`.
"""

from __future__ import annotations

from apmoe.core.types import ModalityData
from apmoe.processing.base import AnonymizerStrategy, anonymizer_registry


@anonymizer_registry.register("image_anonymizer")
class ImageAnonymizer(AnonymizerStrategy):
    """Pass-through anonymiser for preprocessed face images.

    Returns *data* unchanged.  See module docstring for the rationale.

    Registered as ``"image_anonymizer"`` in
    :data:`~apmoe.processing.base.anonymizer_registry`.
    """

    def anonymize(self, data: ModalityData) -> ModalityData:
        """Return *data* unchanged.

        Args:
            data: :class:`~apmoe.core.types.ModalityData` from the image
                cleaner step — a float32 array of shape ``(200, 200, 3)``.

        Returns:
            The same *data* object, unmodified.
        """
        return data
