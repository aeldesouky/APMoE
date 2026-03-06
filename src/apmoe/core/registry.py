"""Generic, type-safe component registry for the APMoE framework.

Every major extension point (modality processors, cleaners, anonymizers,
embedders, expert plugins, aggregators) uses its own :class:`Registry`
instance.  Developers register their custom components once — either via
the :meth:`Registry.register` decorator or by calling
:meth:`Registry.register_class` directly — and the framework resolves them
by name at bootstrap time.

The registry also supports **dotted-path resolution**: if a config file
references a class by its fully-qualified import path (e.g.
``"myproject.experts.CustomExpert"``) the registry will import and return the
class even if it was never explicitly registered.

Typical usage::

    from apmoe.core.registry import Registry

    # Create a registry for a specific extension point
    expert_registry: Registry[ExpertPlugin] = Registry("experts")

    # Register a class with a decorator
    @expert_registry.register("my_expert")
    class MyExpert(ExpertPlugin):
        ...

    # Look up by name
    cls = expert_registry.get("my_expert")
    instance = cls()

    # Or resolve by dotted import path
    cls = expert_registry.resolve("myproject.experts.MyExpert")
"""

from __future__ import annotations

import importlib
from typing import Callable, Generic, Iterator, TypeVar

from apmoe.core.exceptions import RegistryError

T = TypeVar("T")


class Registry(Generic[T]):
    """Generic name-to-class registry.

    Provides decorator-based registration, name-based lookup, and
    dotted-path import resolution.

    Type Parameters:
        T: The base type (abstract class) that all registered classes must
           be subclasses of.  Not enforced at runtime for flexibility, but
           enforced by static type checkers.

    Attributes:
        name: Human-readable name for this registry (e.g. ``"experts"``).
            Used in error messages.
    """

    def __init__(self, name: str) -> None:
        """Initialise an empty registry.

        Args:
            name: A descriptive label for this registry used in error messages
                (e.g. ``"modality_processors"``).
        """
        self.name = name
        self._store: dict[str, type[T]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Class decorator that registers a class under *key*.

        Args:
            key: The name under which the class will be stored and looked up.
                Must be unique within this registry.

        Returns:
            A decorator that registers the decorated class and returns it
            unchanged, so it can still be imported and used normally.

        Raises:
            RegistryError: If *key* is already registered.

        Example::

            @expert_registry.register("cnn_age_expert")
            class CNNAgeExpert(ExpertPlugin):
                ...
        """
        def decorator(cls: type[T]) -> type[T]:
            self.register_class(key, cls)
            return cls

        return decorator

    def register_class(self, key: str, cls: type[T], *, overwrite: bool = False) -> None:
        """Explicitly register *cls* under *key*.

        Args:
            key: The lookup name.
            cls: The class to register.
            overwrite: If ``True``, silently replace any existing registration
                for *key*.  Defaults to ``False`` (raises on duplicate).

        Raises:
            RegistryError: If *key* is already registered and *overwrite* is
                ``False``.
        """
        if key in self._store and not overwrite:
            existing = self._store[key].__qualname__
            raise RegistryError(
                f"Component '{key}' is already registered in registry '{self.name}' "
                f"(existing class: {existing}).  Use overwrite=True to replace it.",
                context={"registry": self.name, "key": key},
            )
        self._store[key] = cls

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, key: str) -> type[T]:
        """Return the class registered under *key*.

        Args:
            key: The name used when registering the class.

        Returns:
            The registered class (not an instance).

        Raises:
            RegistryError: If *key* is not found.
        """
        if key not in self._store:
            available = sorted(self._store.keys())
            raise RegistryError(
                f"No component named '{key}' in registry '{self.name}'.  "
                f"Available: {available}.",
                context={"registry": self.name, "key": key},
            )
        return self._store[key]

    def resolve(self, name_or_path: str) -> type[T]:
        """Resolve a class by registered name *or* dotted import path.

        First checks the local registry; if not found, attempts to import the
        module and retrieve the attribute.

        Args:
            name_or_path: Either a short registered name (``"cnn_age_expert"``)
                or a fully-qualified dotted path
                (``"myproject.experts.CNNAgeExpert"``).

        Returns:
            The class object.

        Raises:
            RegistryError: If the name is not registered and the dotted path
                cannot be imported.

        Example::

            cls = registry.resolve("apmoe.experts.builtin.CNNAgeExpert")
            instance = cls()
        """
        # 1. Check the in-memory store first (short names win).
        if name_or_path in self._store:
            return self._store[name_or_path]

        # 2. Attempt dotted-path import.
        if "." in name_or_path:
            try:
                module_path, attr_name = name_or_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                cls: type[T] = getattr(module, attr_name)
                return cls
            except (ImportError, AttributeError, ValueError) as exc:
                raise RegistryError(
                    f"Cannot resolve '{name_or_path}' in registry '{self.name}': {exc}",
                    context={"registry": self.name, "path": name_or_path},
                ) from exc

        # 3. Neither matched.
        available = sorted(self._store.keys())
        raise RegistryError(
            f"'{name_or_path}' is not a registered name and does not look like a "
            f"dotted import path (no '.' separator).  Registry '{self.name}' contains: "
            f"{available}.",
            context={"registry": self.name, "key": name_or_path},
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_registered(self) -> list[str]:
        """Return a sorted list of all registered component names.

        Returns:
            A sorted list of registered keys.
        """
        return sorted(self._store.keys())

    def __contains__(self, key: object) -> bool:
        """Support ``"name" in registry`` membership test."""
        return key in self._store

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered component names in insertion order."""
        return iter(self._store)

    def __repr__(self) -> str:
        keys = list(self._store.keys())
        return f"Registry(name={self.name!r}, registered={keys})"
