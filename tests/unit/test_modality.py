"""Unit tests for apmoe.modality.base and apmoe.modality.factory."""

from __future__ import annotations

import pytest

from apmoe.core.exceptions import ModalityError
from apmoe.core.types import ModalityData
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import ModalityProcessorFactory, modality_registry


# ---------------------------------------------------------------------------
# Concrete helpers (minimal, valid implementations)
# ---------------------------------------------------------------------------


class _GoodProcessor(ModalityProcessor):
    """A minimal valid ModalityProcessor for 'visual'."""

    @property
    def modality_name(self) -> str:
        return "visual"

    def validate(self, data: object) -> bool:
        return isinstance(data, bytes)

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality=self.modality_name, data=data)


class _AnotherProcessor(ModalityProcessor):
    """A minimal valid ModalityProcessor for 'audio'."""

    @property
    def modality_name(self) -> str:
        return "audio"

    def validate(self, data: object) -> bool:
        return data is not None

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(modality=self.modality_name, data=data)


# ---------------------------------------------------------------------------
# ModalityProcessor ABC
# ---------------------------------------------------------------------------


class TestModalityProcessorABC:
    """Verify the ABC contract: subclasses must implement all abstract methods."""

    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            ModalityProcessor()  # type: ignore[abstract]

    def test_missing_modality_name_raises(self) -> None:
        class _Bad(ModalityProcessor):
            def validate(self, data: object) -> bool:
                return True

            def preprocess(self, data: object) -> ModalityData:
                return ModalityData(modality="x", data=data)

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_missing_validate_raises(self) -> None:
        class _Bad(ModalityProcessor):
            @property
            def modality_name(self) -> str:
                return "x"

            def preprocess(self, data: object) -> ModalityData:
                return ModalityData(modality="x", data=data)

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_missing_preprocess_raises(self) -> None:
        class _Bad(ModalityProcessor):
            @property
            def modality_name(self) -> str:
                return "x"

            def validate(self, data: object) -> bool:
                return True

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self) -> None:
        proc = _GoodProcessor()
        assert proc.modality_name == "visual"

    def test_validate_returns_true_for_valid_input(self) -> None:
        proc = _GoodProcessor()
        assert proc.validate(b"image bytes") is True

    def test_validate_returns_false_for_invalid_input(self) -> None:
        proc = _GoodProcessor()
        assert proc.validate("not bytes") is False

    def test_preprocess_returns_modality_data(self) -> None:
        proc = _GoodProcessor()
        result = proc.preprocess(b"data")
        assert isinstance(result, ModalityData)
        assert result.modality == "visual"
        assert result.data == b"data"

    def test_get_info_default_impl(self) -> None:
        proc = _GoodProcessor()
        info = proc.get_info()
        assert info["modality"] == "visual"
        assert "_GoodProcessor" in str(info["processor_class"])


# ---------------------------------------------------------------------------
# ModalityProcessorFactory — resolve
# ---------------------------------------------------------------------------


class TestModalityProcessorFactoryResolve:
    """Test that the factory resolves classes correctly."""

    def setup_method(self) -> None:
        """Register a fresh test processor before each test."""
        # Use overwrite=True to avoid conflicts across test runs
        modality_registry.register_class(
            "_test_good_proc", _GoodProcessor, overwrite=True
        )

    def test_resolve_by_registered_name(self) -> None:
        cls = ModalityProcessorFactory.resolve("_test_good_proc")
        assert cls is _GoodProcessor

    def test_resolve_by_dotted_path(self) -> None:
        cls = ModalityProcessorFactory.resolve(
            "tests.unit.test_modality._GoodProcessor"
        )
        assert cls is _GoodProcessor

    def test_resolve_unknown_raises_modality_error(self) -> None:
        with pytest.raises(ModalityError, match="Cannot resolve modality processor"):
            ModalityProcessorFactory.resolve("nonexistent.Module.Class")


# ---------------------------------------------------------------------------
# ModalityProcessorFactory — create
# ---------------------------------------------------------------------------


class TestModalityProcessorFactoryCreate:
    def setup_method(self) -> None:
        modality_registry.register_class(
            "_test_good_proc", _GoodProcessor, overwrite=True
        )

    def test_create_returns_instance(self) -> None:
        instance = ModalityProcessorFactory.create("_test_good_proc")
        assert isinstance(instance, ModalityProcessor)
        assert isinstance(instance, _GoodProcessor)

    def test_create_unknown_raises_modality_error(self) -> None:
        with pytest.raises(ModalityError):
            ModalityProcessorFactory.create("no.such.Processor")

    def test_create_non_processor_class_raises(self) -> None:
        """Registering a class that doesn't subclass ModalityProcessor should error on create."""
        modality_registry.register_class("_bad_class", int, overwrite=True)  # type: ignore[arg-type]
        with pytest.raises(ModalityError, match="does not subclass ModalityProcessor"):
            ModalityProcessorFactory.create("_bad_class")


# ---------------------------------------------------------------------------
# ModalityProcessorFactory — from_configs
# ---------------------------------------------------------------------------


class TestModalityProcessorFactoryFromConfigs:
    def setup_method(self) -> None:
        modality_registry.register_class(
            "_test_good_proc", _GoodProcessor, overwrite=True
        )
        modality_registry.register_class(
            "_test_audio_proc", _AnotherProcessor, overwrite=True
        )

    def _make_modality_config(self, name: str, processor: str):  # type: ignore[return]
        """Create a real ModalityConfig object."""
        from apmoe.core.config import ModalityConfig, PipelineConfig

        return ModalityConfig(
            name=name,
            processor=processor,
            pipeline=PipelineConfig(
                cleaner="dummy.Cleaner",
                anonymizer="dummy.Anonymizer",
            ),
        )

    def test_builds_processor_map(self) -> None:
        configs = [
            self._make_modality_config("visual", "_test_good_proc"),
            self._make_modality_config("audio", "_test_audio_proc"),
        ]
        result = ModalityProcessorFactory.from_configs(configs)
        assert set(result.keys()) == {"visual", "audio"}
        assert isinstance(result["visual"], _GoodProcessor)
        assert isinstance(result["audio"], _AnotherProcessor)

    def test_modality_name_mismatch_raises(self) -> None:
        """Config name 'audio' but processor declares 'visual' → error."""
        configs = [
            self._make_modality_config("audio", "_test_good_proc")  # _GoodProcessor → "visual"
        ]
        with pytest.raises(ModalityError, match="must match"):
            ModalityProcessorFactory.from_configs(configs)

    def test_empty_configs_returns_empty_dict(self) -> None:
        result = ModalityProcessorFactory.from_configs([])
        assert result == {}
