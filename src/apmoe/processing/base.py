"""Abstract strategy base classes for the modality processing pipeline.

The per-modality processing chain consists of up to three sequential steps,
each represented by an abstract strategy:

1. **Cleaner** — removes noise, artefacts, or irrelevant content from the raw
   :class:`~apmoe.core.types.ModalityData` (e.g. denoising, normalisation).
2. **Anonymizer** — strips or obscures personally-identifying information
   (e.g. face blurring, voice perturbation).
3. **Embedder** *(optional)* — maps the cleaned and anonymised
   :class:`~apmoe.core.types.ModalityData` to a dense feature vector
   :class:`~apmoe.core.types.EmbeddingResult`.  When omitted, experts receive
   the preprocessed :class:`~apmoe.core.types.ModalityData` directly.

All three strategies are swappable per-modality via the framework config.
Concrete implementations live in ``apmoe.processing.builtin.*`` and are
registered with the global :data:`cleaner_registry`,
:data:`anonymizer_registry`, and :data:`embedder_registry` instances.

Implementing a custom strategy::

    from apmoe.processing.base import CleanerStrategy
    from apmoe.core.types import ModalityData
    from apmoe.processing.base import cleaner_registry

    @cleaner_registry.register("my_cleaner")
    class MyCleaner(CleanerStrategy):
        def clean(self, data: ModalityData) -> ModalityData:
            # … your logic …
            return data.with_data(cleaned_payload)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from apmoe.core.registry import Registry
from apmoe.core.types import EmbeddingResult, ModalityData

# ---------------------------------------------------------------------------
# Global strategy registries
# ---------------------------------------------------------------------------

#: Registry for :class:`CleanerStrategy` implementations.
cleaner_registry: Registry[CleanerStrategy] = Registry("cleaners")

#: Registry for :class:`AnonymizerStrategy` implementations.
anonymizer_registry: Registry[AnonymizerStrategy] = Registry("anonymizers")

#: Registry for :class:`EmbedderStrategy` implementations.
embedder_registry: Registry[EmbedderStrategy] = Registry("embedders")


# ---------------------------------------------------------------------------
# Abstract strategies
# ---------------------------------------------------------------------------


class CleanerStrategy(ABC):
    """Abstract strategy for the *clean* step of a modality's processing chain.

    A cleaner removes noise, normalises values, strips low-quality segments,
    or performs any other data-quality operation that should happen **before**
    anonymisation and embedding.

    The operation is **non-destructive with respect to metadata**: the returned
    :class:`~apmoe.core.types.ModalityData` should preserve the source modality
    name, timestamp, source, and metadata from the input.  Use
    :meth:`~apmoe.core.types.ModalityData.with_data` to create a copy with a
    new payload.

    Example::

        class PassThroughCleaner(CleanerStrategy):
            def clean(self, data: ModalityData) -> ModalityData:
                return data  # No-op cleaner (useful for testing)
    """

    @abstractmethod
    def clean(self, data: ModalityData) -> ModalityData:
        """Clean *data* and return a (possibly new) :class:`~apmoe.core.types.ModalityData`.

        Args:
            data: The raw or partially-processed modality payload, as produced
                by the :class:`~apmoe.modality.base.ModalityProcessor`.

        Returns:
            A :class:`~apmoe.core.types.ModalityData` with the cleaned payload.
            May be the same instance if no modifications were made.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If cleaning fails
                unrecoverably.
        """

    def get_info(self) -> dict[str, object]:
        """Return metadata about this cleaner.

        Returns:
            A plain dict with at least ``"cleaner_class"`` key.
        """
        return {"cleaner_class": type(self).__qualname__}


class AnonymizerStrategy(ABC):
    """Abstract strategy for the *anonymize* step of a modality's processing chain.

    An anonymizer removes or obfuscates personally-identifying information
    (PII) from a :class:`~apmoe.core.types.ModalityData` object so that
    downstream experts and storage never see raw biometric identifiers.

    Typical operations:
    - Detect and blur faces in images.
    - Perturb pitch / formants in audio to prevent speaker identification.
    - Remove subject-identifying EEG artifacts.

    The anonymizer **must not change the modality name or shape** in ways that
    would break downstream processing — only the data *content* should be
    anonymised.

    Example::

        class PassThroughAnonymizer(AnonymizerStrategy):
            def anonymize(self, data: ModalityData) -> ModalityData:
                return data  # No-op (useful for testing or non-PII modalities)
    """

    @abstractmethod
    def anonymize(self, data: ModalityData) -> ModalityData:
        """Anonymize *data* and return a (possibly new) :class:`~apmoe.core.types.ModalityData`.

        Args:
            data: The modality payload after the clean step.

        Returns:
            A :class:`~apmoe.core.types.ModalityData` with PII removed or
            obfuscated.  May be the same instance if no modifications were made.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If anonymisation
                fails unrecoverably.
        """

    def get_info(self) -> dict[str, object]:
        """Return metadata about this anonymizer.

        Returns:
            A plain dict with at least ``"anonymizer_class"`` key.
        """
        return {"anonymizer_class": type(self).__qualname__}


class EmbedderStrategy(ABC):
    """Abstract strategy for the optional *embed* step of a modality's processing chain.

    An embedder maps a cleaned and anonymised
    :class:`~apmoe.core.types.ModalityData` to a dense numeric
    :class:`~apmoe.core.types.EmbeddingResult`.  Experts that consume
    ``EmbeddingResult`` objects benefit from a lower-dimensional, semantically
    rich representation and do not need to perform their own feature extraction.

    If no embedder is configured for a modality, the pipeline skips this step
    and the expert receives the preprocessed
    :class:`~apmoe.core.types.ModalityData` directly.

    Embedders typically wrap a **pretrained** feature-extraction backbone
    (e.g. MobileNet for images, a mel-spectrogram extractor for audio).
    They are **inference-only** — no training occurs inside the framework.

    Example::

        import numpy as np
        from apmoe.processing.base import EmbedderStrategy
        from apmoe.core.types import ModalityData, EmbeddingResult

        class IdentityEmbedder(EmbedderStrategy):
            \"\"\"Flatten data array as a trivial embedding (for testing).\"\"\"
            def embed(self, data: ModalityData) -> EmbeddingResult:
                flat = np.array(data.data).flatten().astype(float)
                return EmbeddingResult(
                    modality=data.modality,
                    embedding=flat,
                    metadata={"embedder": "identity"},
                )
    """

    @abstractmethod
    def embed(self, data: ModalityData) -> EmbeddingResult:
        """Produce a feature embedding from *data*.

        Args:
            data: The cleaned and anonymised modality payload.

        Returns:
            An :class:`~apmoe.core.types.EmbeddingResult` containing the
            feature vector and relevant metadata.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If embedding fails.
        """

    def get_info(self) -> dict[str, object]:
        """Return metadata about this embedder.

        Returns:
            A plain dict with at least ``"embedder_class"`` key.  Subclasses
            should add model name, embedding dimension, and backbone version.
        """
        return {"embedder_class": type(self).__qualname__}
