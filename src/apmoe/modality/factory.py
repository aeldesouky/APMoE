"""Factory for resolving and instantiating :class:`~apmoe.modality.base.ModalityProcessor` objects.

The :class:`ModalityProcessorFactory` bridges the gap between the framework
config (which stores processor names as dotted-path strings) and live
:class:`~apmoe.modality.base.ModalityProcessor` instances.

At bootstrap time, :meth:`ModalityProcessorFactory.from_configs` iterates
over all :class:`~apmoe.core.config.ModalityConfig` entries, resolves each
``processor`` string to a class via the underlying
:class:`~apmoe.core.registry.Registry`, instantiates it, and returns a
mapping of modality name → processor instance ready to use.

Usage::

    from apmoe.modality.factory import ModalityProcessorFactory, modality_registry
    from apmoe.core.config import load_config

    cfg = load_config("configs/default.json")
    processors = ModalityProcessorFactory.from_configs(cfg.apmoe.modalities)
    # processors == {"visual": <VisualProcessor>, "audio": <AudioProcessor>}

Custom processors can register themselves for short-name resolution::

    from apmoe.modality.factory import modality_registry
    from apmoe.modality.base import ModalityProcessor

    @modality_registry.register("my_processor")
    class MyProcessor(ModalityProcessor):
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmoe.core.exceptions import ModalityError
from apmoe.core.registry import Registry
from apmoe.modality.base import ModalityProcessor

if TYPE_CHECKING:
    from apmoe.core.config import ModalityConfig


#: The global modality-processor registry.
#:
#: Built-in processors are registered here automatically on import.  Third-party
#: processors can register via the ``@modality_registry.register("name")``
#: decorator or by calling ``modality_registry.register_class("name", cls)``.
modality_registry: Registry[ModalityProcessor] = Registry("modality_processors")


class ModalityProcessorFactory:
    """Resolves and instantiates :class:`~apmoe.modality.base.ModalityProcessor` objects.

    All methods are class-methods; the factory is stateless and is never
    instantiated directly.

    The factory delegates class resolution to :data:`modality_registry`, which
    supports both short registered names (e.g. ``"visual"``) and fully-qualified
    dotted import paths (e.g.
    ``"apmoe.modality.builtin.visual.VisualProcessor"``).
    """

    @classmethod
    def resolve(cls, processor_name: str) -> type[ModalityProcessor]:
        """Resolve a processor class by name or dotted path.

        Args:
            processor_name: Either a short name registered in
                :data:`modality_registry`, or a fully-qualified dotted import
                path.

        Returns:
            The :class:`~apmoe.modality.base.ModalityProcessor` subclass (not
            an instance).

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If the name cannot
                be resolved.
        """
        try:
            return modality_registry.resolve(processor_name)
        except Exception as exc:
            raise ModalityError(
                f"Cannot resolve modality processor '{processor_name}': {exc}",
                context={"processor": processor_name},
            ) from exc

    @classmethod
    def create(cls, processor_name: str) -> ModalityProcessor:
        """Resolve *processor_name* and return a fresh instance.

        Args:
            processor_name: Registered name or dotted import path of the
                processor class.

        Returns:
            A new :class:`~apmoe.modality.base.ModalityProcessor` instance.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If the class cannot
                be resolved or instantiation raises an exception.
        """
        processor_cls = cls.resolve(processor_name)
        try:
            instance = processor_cls()
        except Exception as exc:
            raise ModalityError(
                f"Failed to instantiate modality processor '{processor_name}': {exc}",
                context={"processor": processor_name},
            ) from exc

        if not isinstance(instance, ModalityProcessor):
            raise ModalityError(
                f"Class '{processor_name}' does not subclass ModalityProcessor.",
                context={"processor": processor_name},
            )
        return instance

    @classmethod
    def from_configs(
        cls,
        modality_configs: list[ModalityConfig],
    ) -> dict[str, ModalityProcessor]:
        """Build a modality-name → processor mapping from a list of configs.

        Iterates over each :class:`~apmoe.core.config.ModalityConfig`, creates
        the corresponding processor instance, and indexes it by
        :attr:`~apmoe.core.config.ModalityConfig.name`.

        Args:
            modality_configs: The list of
                :class:`~apmoe.core.config.ModalityConfig` objects from the
                framework config (``cfg.apmoe.modalities``).

        Returns:
            A dict mapping each modality name (e.g. ``"visual"``) to its live
            :class:`~apmoe.modality.base.ModalityProcessor` instance.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If any processor
                cannot be resolved or instantiated, or if the processor's
                :attr:`~apmoe.modality.base.ModalityProcessor.modality_name`
                does not match the config name.
        """
        processors: dict[str, ModalityProcessor] = {}
        for modality_cfg in modality_configs:
            processor = cls.create(modality_cfg.processor)
            if processor.modality_name != modality_cfg.name:
                raise ModalityError(
                    f"Processor '{modality_cfg.processor}' declares modality_name="
                    f"'{processor.modality_name}' but config entry uses name="
                    f"'{modality_cfg.name}'.  These must match.",
                    context={
                        "config_name": modality_cfg.name,
                        "processor_modality_name": processor.modality_name,
                    },
                )
            processors[modality_cfg.name] = processor
        return processors
