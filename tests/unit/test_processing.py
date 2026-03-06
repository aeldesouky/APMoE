"""Unit tests for apmoe.processing.base (Cleaner, Anonymizer, Embedder ABCs)."""

from __future__ import annotations

import numpy as np
import pytest

from apmoe.core.types import EmbeddingResult, ModalityData
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    EmbedderStrategy,
    anonymizer_registry,
    cleaner_registry,
    embedder_registry,
)


# ---------------------------------------------------------------------------
# Concrete helpers
# ---------------------------------------------------------------------------


class _PassThroughCleaner(CleanerStrategy):
    def clean(self, data: ModalityData) -> ModalityData:
        return data


class _UpperCaseCleaner(CleanerStrategy):
    """Cleaner that marks data as 'cleaned' via metadata."""

    def clean(self, data: ModalityData) -> ModalityData:
        new_meta = dict(data.metadata)
        new_meta["cleaned"] = True
        return ModalityData(
            modality=data.modality,
            data=data.data,
            metadata=new_meta,
            timestamp=data.timestamp,
            source=data.source,
        )


class _PassThroughAnonymizer(AnonymizerStrategy):
    def anonymize(self, data: ModalityData) -> ModalityData:
        return data


class _MarkingAnonymizer(AnonymizerStrategy):
    """Anonymizer that stamps metadata."""

    def anonymize(self, data: ModalityData) -> ModalityData:
        new_meta = dict(data.metadata)
        new_meta["anonymized"] = True
        return ModalityData(
            modality=data.modality,
            data=data.data,
            metadata=new_meta,
            timestamp=data.timestamp,
            source=data.source,
        )


class _FlatEmbedder(EmbedderStrategy):
    """Embedder that flattens data into a 1-D float array."""

    def embed(self, data: ModalityData) -> EmbeddingResult:
        flat = np.array(data.data, dtype=float).flatten()
        return EmbeddingResult(
            modality=data.modality,
            embedding=flat,
            metadata={"embedder": "flat"},
        )


# ---------------------------------------------------------------------------
# CleanerStrategy ABC
# ---------------------------------------------------------------------------


class TestCleanerStrategyABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            CleanerStrategy()  # type: ignore[abstract]

    def test_missing_clean_raises(self) -> None:
        class _Bad(CleanerStrategy):
            pass

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_passthrough_returns_same_instance(self) -> None:
        cleaner = _PassThroughCleaner()
        data = ModalityData(modality="visual", data=b"x")
        result = cleaner.clean(data)
        assert result is data

    def test_mutating_cleaner_preserves_modality(self) -> None:
        cleaner = _UpperCaseCleaner()
        data = ModalityData(modality="audio", data=b"raw", metadata={})
        result = cleaner.clean(data)
        assert result.modality == "audio"
        assert result.metadata["cleaned"] is True

    def test_get_info_default(self) -> None:
        cleaner = _PassThroughCleaner()
        info = cleaner.get_info()
        assert "_PassThroughCleaner" in str(info["cleaner_class"])

    def test_cleaner_registry_instance(self) -> None:
        assert cleaner_registry.name == "cleaners"

    def test_register_and_resolve_cleaner(self) -> None:
        cleaner_registry.register_class("_pt_cleaner", _PassThroughCleaner, overwrite=True)
        cls = cleaner_registry.resolve("_pt_cleaner")
        assert cls is _PassThroughCleaner


# ---------------------------------------------------------------------------
# AnonymizerStrategy ABC
# ---------------------------------------------------------------------------


class TestAnonymizerStrategyABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            AnonymizerStrategy()  # type: ignore[abstract]

    def test_missing_anonymize_raises(self) -> None:
        class _Bad(AnonymizerStrategy):
            pass

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_passthrough_returns_same_instance(self) -> None:
        anon = _PassThroughAnonymizer()
        data = ModalityData(modality="visual", data=b"img")
        assert anon.anonymize(data) is data

    def test_marking_anonymizer_sets_metadata(self) -> None:
        anon = _MarkingAnonymizer()
        data = ModalityData(modality="visual", data=b"img", metadata={})
        result = anon.anonymize(data)
        assert result.metadata["anonymized"] is True
        assert result.modality == "visual"

    def test_get_info_default(self) -> None:
        anon = _PassThroughAnonymizer()
        info = anon.get_info()
        assert "_PassThroughAnonymizer" in str(info["anonymizer_class"])

    def test_anonymizer_registry_instance(self) -> None:
        assert anonymizer_registry.name == "anonymizers"

    def test_register_and_resolve_anonymizer(self) -> None:
        anonymizer_registry.register_class(
            "_pt_anonymizer", _PassThroughAnonymizer, overwrite=True
        )
        cls = anonymizer_registry.resolve("_pt_anonymizer")
        assert cls is _PassThroughAnonymizer


# ---------------------------------------------------------------------------
# EmbedderStrategy ABC
# ---------------------------------------------------------------------------


class TestEmbedderStrategyABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            EmbedderStrategy()  # type: ignore[abstract]

    def test_missing_embed_raises(self) -> None:
        class _Bad(EmbedderStrategy):
            pass

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_embed_returns_embedding_result(self) -> None:
        embedder = _FlatEmbedder()
        data = ModalityData(modality="visual", data=np.array([1.0, 2.0, 3.0]))
        result = embedder.embed(data)
        assert isinstance(result, EmbeddingResult)
        assert result.modality == "visual"
        assert result.embedding.shape == (3,)

    def test_embedding_dim_inferred(self) -> None:
        embedder = _FlatEmbedder()
        data = ModalityData(modality="eeg", data=np.ones((4, 5)))
        result = embedder.embed(data)
        # _FlatEmbedder flattens (4, 5) → (20,); embedding_dim is inferred as last dim = 20
        assert result.embedding.shape == (20,)
        assert result.embedding_dim == 20

    def test_get_info_default(self) -> None:
        embedder = _FlatEmbedder()
        info = embedder.get_info()
        assert "_FlatEmbedder" in str(info["embedder_class"])

    def test_embedder_registry_instance(self) -> None:
        assert embedder_registry.name == "embedders"

    def test_register_and_resolve_embedder(self) -> None:
        embedder_registry.register_class("_flat_emb", _FlatEmbedder, overwrite=True)
        cls = embedder_registry.resolve("_flat_emb")
        assert cls is _FlatEmbedder


# ---------------------------------------------------------------------------
# Pipeline chaining: clean → anonymize → embed
# ---------------------------------------------------------------------------


class TestProcessingChain:
    """Integration smoke-test: chain the three strategies together."""

    def test_full_chain_produces_embedding_result(self) -> None:
        cleaner = _UpperCaseCleaner()
        anonymizer = _MarkingAnonymizer()
        embedder = _FlatEmbedder()

        raw = ModalityData(modality="visual", data=np.array([10.0, 20.0, 30.0]))
        cleaned = cleaner.clean(raw)
        anonymized = anonymizer.anonymize(cleaned)
        result = embedder.embed(anonymized)

        assert isinstance(result, EmbeddingResult)
        assert result.modality == "visual"
        assert result.embedding.shape == (3,)
        # Metadata should have been accumulated through the chain
        assert anonymized.metadata.get("cleaned") is True
        assert anonymized.metadata.get("anonymized") is True

    def test_chain_without_embedder_gives_modality_data(self) -> None:
        cleaner = _PassThroughCleaner()
        anonymizer = _PassThroughAnonymizer()

        raw = ModalityData(modality="audio", data=b"wave")
        cleaned = cleaner.clean(raw)
        result = anonymizer.anonymize(cleaned)

        assert isinstance(result, ModalityData)
        assert result.data == b"wave"
