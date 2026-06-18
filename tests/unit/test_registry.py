"""Unit tests for apmoe.core.registry.Registry."""

from __future__ import annotations

import pytest

import apmoe.core.plugins as plugins
from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.exceptions import RegistryError
from apmoe.core.registry import Registry
from apmoe.core.types import ExpertOutput, Prediction


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Base:
    """Dummy base class for registry type parameter."""


class _ImplA(_Base):
    """First concrete implementation."""


class _ImplB(_Base):
    """Second concrete implementation."""


class _EntryPoint:
    """Small importlib.metadata.EntryPoint test double."""

    def __init__(self, name: str, group: str, loaded: object) -> None:
        self.name = name
        self.group = group
        self._loaded = loaded

    def load(self) -> object:
        return self._loaded


class _EntryPoints:
    """Entry-points collection with the modern ``select`` API."""

    def __init__(self, values: list[_EntryPoint]) -> None:
        self._values = values

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry_point for entry_point in self._values if entry_point.group == group]


class _PluginAggregator(AggregatorStrategy):
    """Aggregator exposed through a fake package entry point."""

    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        output = outputs[0]
        return Prediction(
            predicted_age=output.predicted_age,
            confidence=output.confidence,
            per_expert_outputs=list(outputs),
        )


@pytest.fixture()
def reg() -> Registry[_Base]:
    """Return a fresh, empty Registry instance for each test."""
    return Registry("test_registry")


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestRegistryInit:
    def test_name_is_stored(self, reg: Registry[_Base]) -> None:
        assert reg.name == "test_registry"

    def test_starts_empty(self, reg: Registry[_Base]) -> None:
        assert len(reg) == 0
        assert reg.list_registered() == []

    def test_repr_contains_name(self, reg: Registry[_Base]) -> None:
        assert "test_registry" in repr(reg)


# ---------------------------------------------------------------------------
# register decorator
# ---------------------------------------------------------------------------


class TestRegisterDecorator:
    def test_decorator_registers_class(self, reg: Registry[_Base]) -> None:
        @reg.register("impl_a")
        class MyImpl(_Base):
            pass

        assert "impl_a" in reg
        assert reg.get("impl_a") is MyImpl

    def test_decorator_returns_class_unchanged(self, reg: Registry[_Base]) -> None:
        @reg.register("impl_a")
        class MyImpl(_Base):
            pass

        assert MyImpl.__name__ == "MyImpl"

    def test_decorator_duplicate_raises(self, reg: Registry[_Base]) -> None:
        reg.register_class("impl_a", _ImplA)
        with pytest.raises(RegistryError, match="already registered"):
            reg.register_class("impl_a", _ImplB)

    def test_decorator_duplicate_with_overwrite(self, reg: Registry[_Base]) -> None:
        reg.register_class("impl_a", _ImplA)
        reg.register_class("impl_a", _ImplB, overwrite=True)
        assert reg.get("impl_a") is _ImplB


# ---------------------------------------------------------------------------
# register_class
# ---------------------------------------------------------------------------


class TestRegisterClass:
    def test_register_and_get(self, reg: Registry[_Base]) -> None:
        reg.register_class("a", _ImplA)
        assert reg.get("a") is _ImplA

    def test_multiple_classes(self, reg: Registry[_Base]) -> None:
        reg.register_class("a", _ImplA)
        reg.register_class("b", _ImplB)
        assert reg.get("a") is _ImplA
        assert reg.get("b") is _ImplB


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_unknown_raises(self, reg: Registry[_Base]) -> None:
        with pytest.raises(RegistryError, match="No component named 'unknown'"):
            reg.get("unknown")

    def test_get_unknown_lists_available(self, reg: Registry[_Base]) -> None:
        reg.register_class("alpha", _ImplA)
        with pytest.raises(RegistryError, match="alpha"):
            reg.get("missing")


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_registered_name(self, reg: Registry[_Base]) -> None:
        reg.register_class("impl_a", _ImplA)
        assert reg.resolve("impl_a") is _ImplA

    def test_resolve_dotted_path(self, reg: Registry[_Base]) -> None:
        # Resolve via fully-qualified path (standard library class as example)
        cls = reg.resolve("pathlib.Path")
        from pathlib import Path
        assert cls is Path

    def test_resolve_dotted_path_invalid_module(self, reg: Registry[_Base]) -> None:
        with pytest.raises(RegistryError, match="Cannot resolve"):
            reg.resolve("nonexistent.module.SomeClass")

    def test_resolve_dotted_path_missing_attr(self, reg: Registry[_Base]) -> None:
        with pytest.raises(RegistryError, match="Cannot resolve"):
            reg.resolve("pathlib.NonExistentClass")

    def test_resolve_plain_name_not_registered(self, reg: Registry[_Base]) -> None:
        with pytest.raises(RegistryError, match="not a registered name"):
            reg.resolve("unknown_no_dots")

    def test_resolve_legacy_keystroke_cleaner_path(self) -> None:
        """Old configs used ``keystroke_cleaners``; alias resolves to ``cleaners``."""
        from apmoe.processing.base import cleaner_registry
        from apmoe.processing.builtin.cleaners import KeystrokeCleaner

        cls = cleaner_registry.resolve(
            "apmoe.processing.builtin.keystroke_cleaners.KeystrokeCleaner",
        )
        assert cls is KeystrokeCleaner

    def test_resolve_legacy_keystroke_anonymizer_path(self) -> None:
        from apmoe.processing.base import anonymizer_registry
        from apmoe.processing.builtin.anonymizers import KeystrokeAnonymizer

        cls = anonymizer_registry.resolve(
            "apmoe.processing.builtin.keystroke_anonymizers.KeystrokeAnonymizer",
        )
        assert cls is KeystrokeAnonymizer


class TestEntryPointDiscovery:
    def test_discovers_aggregator_entry_point(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry_points = _EntryPoints(
            [
                _EntryPoint(
                    "unit_test_plugin_aggregator",
                    "apmoe.aggregators",
                    _PluginAggregator,
                )
            ]
        )
        monkeypatch.setattr(plugins.metadata, "entry_points", lambda: entry_points)

        plugins.discover_plugin_entry_points(force=True)

        assert aggregator_registry.resolve("unit_test_plugin_aggregator") is _PluginAggregator

    def test_rejects_entry_point_with_wrong_type(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry_points = _EntryPoints(
            [
                _EntryPoint(
                    "unit_test_bad_aggregator",
                    "apmoe.aggregators",
                    _ImplA,
                )
            ]
        )
        monkeypatch.setattr(plugins.metadata, "entry_points", lambda: entry_points)

        with pytest.raises(RegistryError, match="must load a subclass"):
            plugins.discover_plugin_entry_points(force=True)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


class TestIntrospection:
    def test_list_registered_sorted(self, reg: Registry[_Base]) -> None:
        reg.register_class("zebra", _ImplA)
        reg.register_class("apple", _ImplB)
        assert reg.list_registered() == ["apple", "zebra"]

    def test_contains_true(self, reg: Registry[_Base]) -> None:
        reg.register_class("a", _ImplA)
        assert "a" in reg

    def test_contains_false(self, reg: Registry[_Base]) -> None:
        assert "missing" not in reg

    def test_len(self, reg: Registry[_Base]) -> None:
        reg.register_class("a", _ImplA)
        reg.register_class("b", _ImplB)
        assert len(reg) == 2

    def test_iter(self, reg: Registry[_Base]) -> None:
        reg.register_class("a", _ImplA)
        reg.register_class("b", _ImplB)
        keys = list(reg)
        assert set(keys) == {"a", "b"}

    def test_empty_list_registered(self, reg: Registry[_Base]) -> None:
        assert reg.list_registered() == []
