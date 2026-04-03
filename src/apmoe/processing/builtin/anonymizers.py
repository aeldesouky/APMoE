"""Built-in anonymizer strategies for the APMoE processing pipeline.

Anonymizers are the second step in each modality's processing chain
(``Cleaner → Anonymizer → optional Embedder``).  They strip or obfuscate
personally-identifying information (PII) before data reaches experts or
storage.

Currently provided:

* :class:`KeystrokeAnonymizer` — pass-through for keystroke timing data.
  Inter-key timing values and scan-codes do not constitute direct PII in
  the same sense as biometric images or voice recordings; the expert
  operates on aggregate statistical features, not raw identifiers.
"""

from __future__ import annotations

from apmoe.core.types import ModalityData
from apmoe.processing.base import AnonymizerStrategy, anonymizer_registry


@anonymizer_registry.register("keystroke_anonymizer")
class KeystrokeAnonymizer(AnonymizerStrategy):
    """Pass-through anonymizer for keystroke dynamics data.

    Keystroke inter-key timing values and scan-codes are behavioural
    biometrics that do not contain directly identifying information such
    as names, faces, or voice prints.  The expert downstream aggregates
    these timings into statistical features over the full session, further
    reducing any residual identifiability.

    This anonymizer is therefore a deliberate no-op: it returns *data*
    unchanged.  It exists to satisfy the framework's mandatory
    ``Cleaner → Anonymizer`` pipeline contract and to provide a clear
    extension point — subclass and override :meth:`anonymize` if your
    deployment requires additional suppression (e.g. rounding timings to
    the nearest 10 ms, or excluding specific key-code ranges).

    Registered as ``"keystroke_anonymizer"`` in
    :data:`~apmoe.processing.base.anonymizer_registry`.
    """

    def anonymize(self, data: ModalityData) -> ModalityData:
        """Return *data* unchanged.

        Args:
            data: :class:`~apmoe.core.types.ModalityData` from the keystroke
                cleaner step.

        Returns:
            The same *data* object, unmodified.
        """
        return data
