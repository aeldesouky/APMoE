"""Minimal subclasses of built-in components for ``apmoe init`` extension tests.

Copied into a temporary project as ``proj_ext.py`` to verify dotted-path
resolution from a project-local module (CLI adds the config directory to
``sys.path``).
"""

from __future__ import annotations

import numpy as np

from apmoe.aggregation.builtin import WeightedAverageAggregator
from apmoe.core.types import EmbeddingResult, ModalityData
from apmoe.experts.builtin import FaceAgeExpert, KeystrokeAgeExpert
from apmoe.modality.builtin.image import ImageProcessor
from apmoe.modality.builtin.keystroke import KeystrokeProcessor
from apmoe.processing.base import EmbedderStrategy
from apmoe.processing.builtin.anonymizers import KeystrokeAnonymizer
from apmoe.processing.builtin.cleaners import KeystrokeCleaner
from apmoe.processing.builtin.image_anonymizers import ImageAnonymizer
from apmoe.processing.builtin.image_cleaners import ImageCleaner


class InitImageProcessor(ImageProcessor):
    """Subclass for dotted-path resolution tests (image modality)."""


class InitKeystrokeProcessor(KeystrokeProcessor):
    """Subclass for dotted-path resolution tests (keystroke modality)."""


class InitImageCleaner(ImageCleaner):
    """Subclass for dotted-path resolution tests."""


class InitImageAnonymizer(ImageAnonymizer):
    """Subclass for dotted-path resolution tests."""


class InitKeystrokeCleaner(KeystrokeCleaner):
    """Subclass for dotted-path resolution tests."""


class InitKeystrokeAnonymizer(KeystrokeAnonymizer):
    """Subclass for dotted-path resolution tests."""


class InitImageEmbedder(EmbedderStrategy):
    """Passes through the cleaned image array as an embedding (FaceAgeExpert supports this)."""

    def embed(self, data: ModalityData) -> EmbeddingResult:
        return EmbeddingResult(
            modality=data.modality,
            embedding=np.asarray(data.data, dtype=np.float32),
        )


class InitFaceAgeExpert(FaceAgeExpert):
    """Subclass for dotted-path resolution tests."""


class InitKeystrokeAgeExpert(KeystrokeAgeExpert):
    """Subclass for dotted-path resolution tests."""


class InitWeightedAverageAggregator(WeightedAverageAggregator):
    """Subclass for dotted-path resolution tests."""
