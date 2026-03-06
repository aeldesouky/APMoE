"""Abstract base class for modality processors.

Every modality (visual, audio, EEG, …) must be handled by a concrete
:class:`ModalityProcessor` that knows how to validate and pre-process raw
input bytes or objects into a normalised :class:`~apmoe.core.types.ModalityData`
that the rest of the pipeline can consume.

Implementing a custom processor::

    from apmoe.modality.base import ModalityProcessor
    from apmoe.core.types import ModalityData

    class MyProcessor(ModalityProcessor):
        @property
        def modality_name(self) -> str:
            return "my_modality"

        def validate(self, data: object) -> bool:
            return isinstance(data, bytes) and len(data) > 0

        def preprocess(self, data: object) -> ModalityData:
            # Convert raw bytes / object into a normalised payload.
            return ModalityData(modality=self.modality_name, data=data)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from apmoe.core.types import ModalityData


class ModalityProcessor(ABC):
    """Abstract base class that every modality processor must implement.

    A ``ModalityProcessor`` is responsible for two things:

    1. **Validation** — determine whether a raw input is structurally valid
       before it enters the pipeline.  This is a fast, cheap check (e.g., file
       format, shape, duration) that lets the pipeline fail early with a clear
       error rather than propagating garbage data to later stages.

    2. **Preprocessing** — convert the raw input into a normalised
       :class:`~apmoe.core.types.ModalityData` object that Cleaner and
       Anonymizer strategies can operate on.  This step may resize images,
       resample audio, filter EEG signals, etc.

    Concrete subclasses are registered via the framework's modality registry
    and resolved by :class:`~apmoe.modality.factory.ModalityProcessorFactory`
    at bootstrap time.

    The modality name returned by :attr:`modality_name` is the canonical key
    used throughout the pipeline to identify this modality (e.g. ``"visual"``,
    ``"audio"``, ``"eeg"``).  It **must** match the ``name`` field in the
    corresponding :class:`~apmoe.core.config.ModalityConfig`.

    Example::

        from apmoe.modality.base import ModalityProcessor
        from apmoe.core.types import ModalityData
        import numpy as np

        class VisualProcessor(ModalityProcessor):
            @property
            def modality_name(self) -> str:
                return "visual"

            def validate(self, data: object) -> bool:
                return isinstance(data, (bytes, np.ndarray))

            def preprocess(self, data: object) -> ModalityData:
                # … decode, resize, normalise …
                return ModalityData(modality="visual", data=data)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Canonical name of the modality this processor handles.

        Returns:
            A non-empty string identifier, e.g. ``"visual"``, ``"audio"``,
            ``"eeg"``.  Must match the ``name`` declared in the config.
        """

    @abstractmethod
    def validate(self, data: object) -> bool:
        """Check whether *data* is structurally valid for this modality.

        This is a **fast, cheap** pre-flight check.  It should not perform
        full preprocessing — just enough to confirm that the input can be
        processed without raising an unrecoverable error.

        Implementors should return ``False`` (not raise) for invalid input;
        the pipeline will convert a ``False`` return into a
        :class:`~apmoe.core.exceptions.ModalityError`.

        Args:
            data: The raw input object for this modality.  Typically ``bytes``
                from an HTTP upload, a ``numpy.ndarray``, a file path, or
                whatever the transport layer provides.

        Returns:
            ``True`` if the input is valid and can be passed to
            :meth:`preprocess`; ``False`` otherwise.
        """

    @abstractmethod
    def preprocess(self, data: object) -> ModalityData:
        """Convert raw *data* into a normalised :class:`~apmoe.core.types.ModalityData`.

        This method **must** call :meth:`validate` internally or the caller
        will have already validated.  Either way, ``preprocess`` may assume
        that *data* has passed validation.

        Typical operations:
        - Decode bytes to a numeric array (e.g. JPEG → RGB ``ndarray``).
        - Resize / crop / normalise images.
        - Resample audio to a standard sample rate.
        - Apply bandpass filter to EEG signals.

        Args:
            data: The raw input object (same type as accepted by
                :meth:`validate`).

        Returns:
            A :class:`~apmoe.core.types.ModalityData` containing the
            normalised payload and appropriate metadata.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If preprocessing
                fails for any reason.
        """

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def get_info(self) -> dict[str, object]:
        """Return metadata about this processor.

        Subclasses may override this to expose version, configuration, or
        supported input formats.  The default implementation returns only the
        modality name and the class name.

        Returns:
            A plain dict suitable for JSON serialisation.
        """
        return {
            "modality": self.modality_name,
            "processor_class": type(self).__qualname__,
        }
