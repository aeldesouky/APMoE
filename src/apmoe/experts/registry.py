"""Specialised registry for :class:`~apmoe.experts.base.ExpertPlugin` instances.

While the generic :class:`~apmoe.core.registry.Registry` maps names to
*classes*, the :class:`ExpertRegistry` manages live *instances* of
:class:`~apmoe.experts.base.ExpertPlugin`.  It provides higher-level
functionality specific to the expert lifecycle:

- Tracking which modalities each expert requires.
- Validating that all required modalities are present in the framework config.
- Health checking (``is_loaded`` status for every expert).
- Dispatching: given a set of available modality names, returning only the
  experts whose required modalities are all satisfied.

The :data:`expert_registry` module-level singleton is the primary way to
register expert *classes* via the ``@expert_registry.register`` decorator.
At bootstrap time :class:`ExpertRegistry` resolves those class names to
instances via :meth:`ExpertRegistry.from_configs`.

Usage::

    from apmoe.experts.registry import expert_registry, ExpertRegistry

    # Register an expert class (typically done in experts/builtin.py or user code)
    @expert_registry.register("cnn_face_expert")
    class CNNFaceExpert(ExpertPlugin):
        ...

    # At bootstrap, build live instance registry from config
    from apmoe.core.config import load_config
    cfg = load_config("configs/default.json")
    registry = ExpertRegistry.from_configs(cfg.apmoe.experts, cfg.apmoe.modalities)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmoe.core.exceptions import ExpertError, RegistryError
from apmoe.core.registry import Registry
from apmoe.experts.base import ExpertPlugin

if TYPE_CHECKING:
    from apmoe.core.config import ExpertConfig, ModalityConfig


#: The global expert *class* registry.
#:
#: Maps short names or fully-qualified paths to :class:`~apmoe.experts.base.ExpertPlugin`
#: subclasses.  Built-in experts are auto-registered on import of
#: ``apmoe.experts.builtin``.  Third-party experts register via
#: ``@expert_registry.register("name")``.
expert_registry: Registry[ExpertPlugin] = Registry("experts")


class ExpertRegistry:
    """Manages live :class:`~apmoe.experts.base.ExpertPlugin` instances.

    Unlike the generic :class:`~apmoe.core.registry.Registry`, this class
    works with *instances* (not classes) and understands the expert lifecycle
    (load weights, health check, modality dispatch).

    Attributes:
        _experts: Ordered mapping of expert name → instance.

    Typical usage — build from config during bootstrap::

        registry = ExpertRegistry.from_configs(
            cfg.apmoe.experts, cfg.apmoe.modalities
        )
        healthy = registry.health_check()
        runnable = registry.get_runnable_experts({"visual", "audio"})
    """

    def __init__(self) -> None:
        """Initialise an empty instance registry."""
        self._experts: dict[str, ExpertPlugin] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_instance(self, instance: ExpertPlugin) -> None:
        """Add a live *instance* to the registry.

        Args:
            instance: A fully-constructed :class:`~apmoe.experts.base.ExpertPlugin`
                instance (weights not necessarily loaded yet).

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert with the
                same :attr:`~apmoe.experts.base.ExpertPlugin.name` is already
                registered.
        """
        key = instance.name
        if key in self._experts:
            raise ExpertError(
                f"Expert '{key}' is already registered in the ExpertRegistry.",
                context={"expert": key},
            )
        self._experts[key] = instance

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ExpertPlugin:
        """Return the expert instance registered under *name*.

        Args:
            name: The expert's :attr:`~apmoe.experts.base.ExpertPlugin.name`.

        Returns:
            The live :class:`~apmoe.experts.base.ExpertPlugin` instance.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If *name* is not
                registered.
        """
        if name not in self._experts:
            available = sorted(self._experts.keys())
            raise ExpertError(
                f"No expert named '{name}' in ExpertRegistry.  Available: {available}.",
                context={"expert": name},
            )
        return self._experts[name]

    def list_experts(self) -> list[str]:
        """Return a sorted list of all registered expert names.

        Returns:
            Sorted list of expert name strings.
        """
        return sorted(self._experts.keys())

    def all_instances(self) -> list[ExpertPlugin]:
        """Return all registered expert instances in insertion order.

        Returns:
            List of :class:`~apmoe.experts.base.ExpertPlugin` instances.
        """
        return list(self._experts.values())

    # ------------------------------------------------------------------
    # Modality dispatch
    # ------------------------------------------------------------------

    def get_runnable_experts(
        self, available_modalities: set[str]
    ) -> list[ExpertPlugin]:
        """Return experts whose *all* declared modalities are available.

        This is called by the inference pipeline to determine which experts can
        run given the modalities present in the current request.  Experts whose
        required modalities are missing are silently excluded here; the pipeline
        records their names in :attr:`~apmoe.core.types.Prediction.skipped_experts`.

        Args:
            available_modalities: The set of modality names for which processed
                data is available in the current request.

        Returns:
            A list of :class:`~apmoe.experts.base.ExpertPlugin` instances that
            can run (i.e. all their declared modalities are available).
        """
        runnable: list[ExpertPlugin] = []
        for expert in self._experts.values():
            required = set(expert.declared_modalities())
            if required.issubset(available_modalities):
                runnable.append(expert)
        return runnable

    def get_skipped_experts(
        self, available_modalities: set[str]
    ) -> list[str]:
        """Return the names of experts that cannot run due to missing modalities.

        Args:
            available_modalities: The set of modality names available in the
                current request.

        Returns:
            Sorted list of expert names that would be skipped.
        """
        skipped: list[str] = []
        for name, expert in self._experts.items():
            required = set(expert.declared_modalities())
            if not required.issubset(available_modalities):
                skipped.append(name)
        return sorted(skipped)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, bool]:
        """Return a mapping of expert name → ``is_loaded`` status.

        Uses the :attr:`~apmoe.experts.base.ExpertPlugin.is_loaded` property
        of each registered expert.  A ``False`` value indicates that
        :meth:`~apmoe.experts.base.ExpertPlugin.load_weights` either was not
        called or failed.

        Returns:
            Dict mapping expert name to its loaded status.
        """
        return {name: expert.is_loaded for name, expert in self._experts.items()}

    def all_healthy(self) -> bool:
        """Return ``True`` if every registered expert reports ``is_loaded=True``.

        Returns:
            ``True`` if all experts are loaded and healthy.
        """
        return all(expert.is_loaded for expert in self._experts.values())

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_expert_modalities(
        expert_configs: list[ExpertConfig],
        modality_configs: list[ModalityConfig],
    ) -> None:
        """Validate that every expert's modalities are declared in the config.

        This mirrors the Pydantic ``model_validator`` in
        :class:`~apmoe.core.config.APMoEConfig` but is re-exposed here for
        programmatic use (e.g. in the ``apmoe validate`` CLI command).

        Args:
            expert_configs: List of :class:`~apmoe.core.config.ExpertConfig`
                objects from the framework config.
            modality_configs: List of :class:`~apmoe.core.config.ModalityConfig`
                objects from the framework config.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If any expert
                references a modality not declared in *modality_configs*.
        """
        declared_modalities = {m.name for m in modality_configs}
        for expert_cfg in expert_configs:
            unknown = set(expert_cfg.modalities) - declared_modalities
            if unknown:
                raise ExpertError(
                    f"Expert '{expert_cfg.name}' references undeclared modalities: "
                    f"{sorted(unknown)}.  Declared: {sorted(declared_modalities)}.",
                    context={
                        "expert": expert_cfg.name,
                        "unknown_modalities": sorted(unknown),
                    },
                )

    # ------------------------------------------------------------------
    # Bootstrap factory
    # ------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        expert_configs: list[ExpertConfig],
        modality_configs: list[ModalityConfig],
    ) -> ExpertRegistry:
        """Build a live :class:`ExpertRegistry` from config objects.

        For each :class:`~apmoe.core.config.ExpertConfig`:

        1. Validates that the expert's modalities are declared (via
           :meth:`validate_expert_modalities`).
        2. Resolves the class via :data:`expert_registry` (supports dotted
           paths).
        3. Instantiates the class (constructor must accept no required args).
        4. Calls :meth:`~apmoe.experts.base.ExpertPlugin.load_weights` with
           the configured ``weights`` path.
        5. Registers the live instance.

        Args:
            expert_configs: The ``cfg.apmoe.experts`` list.
            modality_configs: The ``cfg.apmoe.modalities`` list (used for
                modality validation).

        Returns:
            A fully-populated :class:`ExpertRegistry` with all experts
            instantiated and weights loaded.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If any expert
                cannot be resolved, instantiated, or loaded.
        """
        cls.validate_expert_modalities(expert_configs, modality_configs)

        registry = cls()
        for expert_cfg in expert_configs:
            # Resolve class
            try:
                expert_cls: type[ExpertPlugin] = expert_registry.resolve(
                    expert_cfg.class_path
                )
            except RegistryError as exc:
                raise ExpertError(
                    f"Cannot resolve expert class '{expert_cfg.class_path}': {exc}",
                    context={"expert": expert_cfg.name, "class": expert_cfg.class_path},
                ) from exc

            # Instantiate
            try:
                instance: ExpertPlugin = expert_cls()
            except Exception as exc:
                raise ExpertError(
                    f"Failed to instantiate expert '{expert_cfg.name}' "
                    f"(class: {expert_cfg.class_path}): {exc}",
                    context={"expert": expert_cfg.name},
                ) from exc

            if not isinstance(instance, ExpertPlugin):
                raise ExpertError(
                    f"Class '{expert_cfg.class_path}' does not subclass ExpertPlugin.",
                    context={"expert": expert_cfg.name},
                )

            # Load weights
            try:
                instance.load_weights(expert_cfg.weights)
            except ExpertError:
                raise
            except Exception as exc:
                raise ExpertError(
                    f"Expert '{expert_cfg.name}' failed to load weights "
                    f"from '{expert_cfg.weights}': {exc}",
                    context={"expert": expert_cfg.name, "weights": expert_cfg.weights},
                ) from exc

            registry.register_instance(instance)

        return registry

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of registered experts."""
        return len(self._experts)

    def __contains__(self, name: object) -> bool:
        """Support ``"expert_name" in registry`` membership test."""
        return name in self._experts

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        names = list(self._experts.keys())
        return f"ExpertRegistry(experts={names})"
