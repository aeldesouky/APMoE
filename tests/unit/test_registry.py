"""Unit tests for apmoe.core.registry.Registry."""

from __future__ import annotations

import pytest

from apmoe.core.exceptions import RegistryError
from apmoe.core.registry import Registry


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Base:
    """Dummy base class for registry type parameter."""


class _ImplA(_Base):
    """First concrete implementation."""


class _ImplB(_Base):
    """Second concrete implementation."""


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
