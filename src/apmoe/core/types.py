"""Shared data types used across the entire APMoE framework.

These types flow through the inference pipeline:

1. Raw multi-modal input arrives as a mapping of modality-name → bytes / file-like.
2. Each modality's processor converts it into :class:`ModalityData`.
3. The optional Embedder step produces :class:`EmbeddingResult`.
4. :data:`ProcessedInput` is the union of both: whatever exits the processing chain.
5. Each expert outputs an :class:`ExpertOutput`.
6. The aggregator combines all :class:`ExpertOutput` instances into a :class:`Prediction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np


# ---------------------------------------------------------------------------
# Pipeline intermediate types
# ---------------------------------------------------------------------------


@dataclass
class ModalityData:
    """Wraps a single modality's data at any stage before embedding.

    This is produced by :class:`~apmoe.modality.base.ModalityProcessor` and
    is passed through the Cleaner → Anonymizer chain unchanged in shape.
    Experts that perform their own feature extraction receive this object
    when no Embedder is configured for the modality.

    Attributes:
        modality: Canonical modality name as declared in config (e.g. ``"visual"``).
        data: The actual payload — could be a ``numpy.ndarray``, a
            ``torch.Tensor``, raw ``bytes``, or any other type that the
            concrete processor and downstream expert understand.
        metadata: Arbitrary key/value pairs for tracing (source file,
            sample-rate, resolution, etc.).
        timestamp: Optional Unix timestamp (seconds) when the data was
            captured or received.
        source: Optional human-readable origin string (file path, device ID,
            stream URL, etc.).
    """

    modality: str
    data: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None
    source: str | None = None

    def with_data(self, new_data: Any) -> "ModalityData":
        """Return a copy of this instance with ``data`` replaced.

        Used by Cleaner and Anonymizer implementations to produce a new
        :class:`ModalityData` without mutating the original.

        Args:
            new_data: The replacement data payload.

        Returns:
            A new :class:`ModalityData` sharing all metadata with the original.
        """
        return ModalityData(
            modality=self.modality,
            data=new_data,
            metadata=dict(self.metadata),
            timestamp=self.timestamp,
            source=self.source,
        )


@dataclass
class EmbeddingResult:
    """Feature vector produced by an :class:`~apmoe.processing.base.EmbedderStrategy`.

    Experts that consume this type receive a dense numeric representation
    of the modality rather than raw or preprocessed data.

    Attributes:
        modality: Canonical modality name (same value as the source
            :class:`ModalityData`).
        embedding: A 1-D or 2-D ``numpy.ndarray`` of shape ``(D,)`` or
            ``(N, D)``.  Concrete embedders may also store a ``torch.Tensor``
            here as long as the consuming expert handles it.
        metadata: Arbitrary key/value pairs for tracing (embedder class,
            model name, layer name, etc.).
        embedding_dim: Dimensionality of a single embedding vector.  Inferred
            from ``embedding.shape[-1]`` when the array attribute is set.
    """

    modality: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_dim: int = field(default=0)

    def __post_init__(self) -> None:
        """Infer ``embedding_dim`` from the array shape when not provided."""
        if self.embedding_dim == 0 and hasattr(self.embedding, "shape"):
            shape = self.embedding.shape  # type: ignore[union-attr]
            self.embedding_dim = int(shape[-1]) if len(shape) > 0 else 0


# ---------------------------------------------------------------------------
# Union type alias
# ---------------------------------------------------------------------------

#: The output type of a modality's full processing chain
#: (``Cleaner → Anonymizer → optional Embedder``).
#:
#: * :class:`EmbeddingResult` — when an Embedder was configured for this modality.
#: * :class:`ModalityData` — when no Embedder was configured (expert receives
#:   the preprocessed data directly).
ProcessedInput = Union[ModalityData, EmbeddingResult]


# ---------------------------------------------------------------------------
# Expert and aggregation types
# ---------------------------------------------------------------------------


@dataclass
class ExpertOutput:
    """Age prediction produced by a single expert plugin.

    Attributes:
        expert_name: Unique registered name of the expert that produced this
            output (matches the ``name`` field in config).
        consumed_modalities: List of modality names the expert actually used.
        predicted_age: The expert's age estimate in years.
        confidence: A scalar in ``[0.0, 1.0]`` indicating the expert's
            self-reported confidence.  Higher is more confident.
        metadata: Arbitrary key/value pairs (inference time, internal scores,
            model version, etc.).
    """

    expert_name: str
    consumed_modalities: list[str]
    predicted_age: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that ``confidence`` is within ``[0.0, 1.0]``."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"ExpertOutput.confidence must be in [0.0, 1.0], got {self.confidence}"
            )


@dataclass
class Prediction:
    """Final, aggregated age prediction returned to the caller.

    Attributes:
        predicted_age: The framework's best estimate of the subject's age in
            years, produced by combining all expert outputs.
        confidence: Aggregated confidence scalar in ``[0.0, 1.0]``.
        confidence_interval: Optional ``(lower, upper)`` age bounds (e.g. a
            95 % CI).
        per_expert_outputs: List of individual :class:`ExpertOutput` instances
            that contributed to this prediction.  Useful for debugging and
            explainability.
        skipped_experts: Names of experts that were skipped during inference
            (e.g. because a required modality was absent in the input).
        metadata: Arbitrary framework-level metadata (pipeline version,
            total latency, etc.).
    """

    predicted_age: float
    confidence: float
    confidence_interval: tuple[float, float] | None = None
    per_expert_outputs: list[ExpertOutput] = field(default_factory=list)
    skipped_experts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate ``confidence`` and ``confidence_interval`` consistency."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Prediction.confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.confidence_interval is not None:
            lo, hi = self.confidence_interval
            if lo > hi:
                raise ValueError(
                    f"confidence_interval lower bound ({lo}) must be ≤ upper bound ({hi})"
                )
