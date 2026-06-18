"""Entry-point based extension discovery for APMoE packages.

Third-party packages can expose framework components through Python package
entry points.  Discovery registers those classes into the existing in-process
registries, so config files may reference either dotted paths or entry-point
names.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.exceptions import RegistryError
from apmoe.core.registry import Registry
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import modality_registry
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    EmbedderStrategy,
    anonymizer_registry,
    cleaner_registry,
    embedder_registry,
)


@dataclass(frozen=True)
class PluginGroup:
    """Mapping from a Python entry-point group to an APMoE registry."""

    group: str
    registry: Registry[Any]
    expected_base: type[Any]


PLUGIN_GROUPS: tuple[PluginGroup, ...] = (
    PluginGroup("apmoe.modality_processors", modality_registry, ModalityProcessor),
    PluginGroup("apmoe.cleaners", cleaner_registry, CleanerStrategy),
    PluginGroup("apmoe.anonymizers", anonymizer_registry, AnonymizerStrategy),
    PluginGroup("apmoe.embedders", embedder_registry, EmbedderStrategy),
    PluginGroup("apmoe.experts", expert_registry, ExpertPlugin),
    PluginGroup("apmoe.aggregators", aggregator_registry, AggregatorStrategy),
)

_DISCOVERED = False


def discover_plugin_entry_points(*, force: bool = False) -> None:
    """Load installed APMoE extension entry points into framework registries.

    Args:
        force: Re-run discovery even if it already completed in this process.

    Raises:
        RegistryError: If an entry point cannot be loaded or does not expose a
            subclass of the expected APMoE base class.
    """
    global _DISCOVERED
    if _DISCOVERED and not force:
        return

    for plugin_group in PLUGIN_GROUPS:
        for entry_point in _entry_points_for_group(plugin_group.group):
            _register_entry_point(entry_point, plugin_group)

    _DISCOVERED = True


def _entry_points_for_group(group: str) -> list[metadata.EntryPoint]:
    """Return entry points for *group* across supported importlib APIs."""
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=group))
    return list(entry_points.get(group, []))  # type: ignore[union-attr]


def _register_entry_point(
    entry_point: metadata.EntryPoint,
    plugin_group: PluginGroup,
) -> None:
    """Load and register a single entry point."""
    try:
        component = entry_point.load()
    except Exception as exc:
        raise RegistryError(
            f"Cannot load entry point '{entry_point.name}' from group "
            f"'{plugin_group.group}': {exc}",
            context={"entry_point": entry_point.name, "group": plugin_group.group},
        ) from exc

    if not isinstance(component, type) or not issubclass(
        component,
        plugin_group.expected_base,
    ):
        raise RegistryError(
            f"Entry point '{entry_point.name}' from group '{plugin_group.group}' "
            f"must load a subclass of {plugin_group.expected_base.__name__}.",
            context={"entry_point": entry_point.name, "group": plugin_group.group},
        )

    if entry_point.name in plugin_group.registry:
        existing = plugin_group.registry.get(entry_point.name)
        if existing is component:
            return

    plugin_group.registry.register_class(entry_point.name, component)
