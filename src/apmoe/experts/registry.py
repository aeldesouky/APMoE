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
from urllib.parse import urlparse

from apmoe.core.exceptions import ExpertError, RegistryError
from apmoe.core.registry import Registry
from apmoe.core.security import (
    emit_security_audit,
    redact_url,
    validate_remote_url,
    verify_local_sha256,
    verify_manifest_payload,
)
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

#: Sentinel class path for the built-in remote expert.
_REMOTE_EXPERT_CLASS_PATH: str = "apmoe.experts.remote.RemoteExpert"


def _expand_env_value(value: str, *, expert_name: str, field: str) -> str:
    """Expand a single $ENV_VAR value used by integrity config."""
    import os

    if value.startswith("$") and value.count("$") == 1:
        env_key = value[1:]
        env_value = os.environ.get(env_key)
        if env_value is None:
            raise ExpertError(
                f"Expert '{expert_name}': {field} references unset environment "
                f"variable '{env_key}'.",
                context={"expert": expert_name, "field": field, "env_var": env_key},
            )
        return env_value
    return value


def _origin(url: str) -> str:
    """Return scheme://host[:port] for a URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _verify_remote_manifest_if_configured(
    expert_cfg: ExpertConfig,
    *,
    endpoint: str,
    security_config: object | None,
    environment: str,
) -> None:
    """Fetch and verify an RSA-signed remote model manifest when configured."""
    integrity = expert_cfg.integrity
    if integrity is None or integrity.manifest_url is None:
        if integrity is not None and integrity.manifest_required:
            raise ExpertError(
                f"Expert '{expert_cfg.name}': manifest_required=true but manifest_url is missing.",
                context={"expert": expert_cfg.name},
            )
        return
    if not integrity.manifest_public_key:
        exc = ExpertError(
            f"Expert '{expert_cfg.name}': manifest_public_key is required for remote integrity.",
            context={"expert": expert_cfg.name},
        )
        emit_security_audit(
            "remote_manifest_integrity",
            "failure",
            expert_name=expert_cfg.name,
            reason="missing_public_key",
        )
        if _manifest_failure_blocks(integrity, environment):
            raise exc
        return

    manifest_url = _expand_env_value(
        integrity.manifest_url,
        expert_name=expert_cfg.name,
        field="integrity.manifest_url",
    )
    public_key = _expand_env_value(
        integrity.manifest_public_key,
        expert_name=expert_cfg.name,
        field="integrity.manifest_public_key",
    )
    if security_config is not None:
        allowlist = getattr(security_config, "remote_endpoint_allowlist", None)
        if allowlist is None and environment != "production":
            allowlist = ["*"]
        try:
            validate_remote_url(
                manifest_url,
                allowlist=allowlist or [],
                enforce_https=getattr(security_config, "remote_enforce_https", True),
                allow_private_networks=getattr(
                    security_config,
                    "remote_allow_private_networks",
                    False,
                ),
                purpose=f"Expert '{expert_cfg.name}' manifest",
            )
        except ExpertError as exc:
            emit_security_audit(
                "remote_manifest_integrity",
                "failure",
                expert_name=expert_cfg.name,
                reason="endpoint_policy_failed",
                metadata={"manifest_url": redact_url(manifest_url)},
            )
            if _manifest_failure_blocks(integrity, environment):
                raise
            return

    try:
        import httpx

        response = httpx.get(
            manifest_url,
            timeout=getattr(expert_cfg, "endpoint_timeout", 10.0),
        )
        response.raise_for_status()
        raw = response.content
        max_bytes = getattr(security_config, "remote_response_max_bytes", 1_048_576)
        if len(raw) > max_bytes:
            raise ExpertError(
                f"Expert '{expert_cfg.name}': remote model manifest exceeded {max_bytes} bytes.",
                context={"expert": expert_cfg.name, "manifest_url": redact_url(manifest_url)},
            )
        manifest = response.json()
        if not isinstance(manifest, dict):
            raise ExpertError(
                f"Expert '{expert_cfg.name}': remote model manifest must be a JSON object.",
                context={"expert": expert_cfg.name},
            )
        verify_manifest_payload(
            manifest,
            public_key_pem=public_key,
            expert_name=expert_cfg.name,
            endpoint_origin=_origin(endpoint),
        )
    except ExpertError:
        emit_security_audit(
            "remote_manifest_integrity",
            "failure",
            expert_name=expert_cfg.name,
            reason="verification_failed",
            metadata={"manifest_url": redact_url(manifest_url)},
        )
        if _manifest_failure_blocks(integrity, environment):
            raise
        return
    except Exception as exc:
        emit_security_audit(
            "remote_manifest_integrity",
            "failure",
            expert_name=expert_cfg.name,
            reason=type(exc).__name__,
            metadata={"manifest_url": redact_url(manifest_url)},
        )
        wrapped = ExpertError(
            f"Expert '{expert_cfg.name}': remote model manifest verification failed: {exc}",
            context={"expert": expert_cfg.name, "manifest_url": redact_url(manifest_url)},
        )
        if _manifest_failure_blocks(integrity, environment):
            raise wrapped from exc
        return

    emit_security_audit(
        "remote_manifest_integrity",
        "success",
        expert_name=expert_cfg.name,
        metadata={"manifest_url": redact_url(manifest_url)},
    )


def _manifest_failure_blocks(integrity: object, environment: str) -> bool:
    """Return whether a remote manifest failure should block startup."""
    return environment == "production" or bool(getattr(integrity, "manifest_required", False))


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
        security_config: object | None = None,
        environment: str = "development",
        remote_retry_config: object | None = None,
        remote_circuit_breaker_config: object | None = None,
    ) -> ExpertRegistry:
        """Build a live :class:`ExpertRegistry` from config objects.

        For each :class:`~apmoe.core.config.ExpertConfig`:

        **Local expert** (``weights`` is set):

        1. Validates that the expert's modalities are declared (via
           :meth:`validate_expert_modalities`).
        2. Resolves the class via :data:`expert_registry` (supports dotted
           paths).
        3. Instantiates the class (constructor must accept no required args).
        4. Calls :meth:`~apmoe.experts.base.ExpertPlugin.load_weights` with
           the configured ``weights`` path.
        5. Registers the live instance.

        **Remote expert** (``endpoint`` is set, class is
        ``apmoe.experts.remote.RemoteExpert``):

        1-2. Same validation and class resolution.
        3. Instantiates :class:`~apmoe.experts.remote.RemoteExpert` passing
           ``endpoint``, ``endpoint_headers``, ``endpoint_timeout``,
           ``request_template``, and ``response_mapping`` as constructor kwargs.
        4. Calls :meth:`~apmoe.experts.base.ExpertPlugin.load_weights` with
           an empty string (the method validates httpx is installed and expands
           env-var headers; no file is read).
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
            is_remote = expert_cfg.endpoint is not None

            # Resolve class ------------------------------------------------
            try:
                expert_cls: type[ExpertPlugin] = expert_registry.resolve(
                    expert_cfg.class_path
                )
            except RegistryError:
                if is_remote:
                    # Auto-import RemoteExpert so it self-registers
                    from apmoe.experts.remote import RemoteExpert  # noqa: F401
                    try:
                        expert_cls = expert_registry.resolve(expert_cfg.class_path)
                    except RegistryError as exc2:
                        raise ExpertError(
                            f"Cannot resolve remote expert class '{expert_cfg.class_path}': {exc2}",
                            context={"expert": expert_cfg.name, "class": expert_cfg.class_path},
                        ) from exc2
                else:
                    raise ExpertError(
                        f"Cannot resolve expert class '{expert_cfg.class_path}'",
                        context={"expert": expert_cfg.name, "class": expert_cfg.class_path},
                    )

            # Instantiate ---------------------------------------------------
            try:
                if is_remote:
                    from apmoe.experts.remote import RemoteExpert
                    if not issubclass(expert_cls, RemoteExpert):
                        raise ExpertError(
                            f"Expert '{expert_cfg.name}' has 'endpoint' set but "
                            f"class '{expert_cfg.class_path}' is not a RemoteExpert "
                            f"subclass.  Use class 'apmoe.experts.remote.RemoteExpert' "
                            f"or a subclass of it for remote experts.",
                            context={"expert": expert_cfg.name},
                        )
                    instance: ExpertPlugin = expert_cls(
                        expert_name=expert_cfg.name,
                        modalities=expert_cfg.modalities,
                        endpoint=expert_cfg.endpoint,
                        endpoint_headers=dict(expert_cfg.endpoint_headers),
                        endpoint_timeout=expert_cfg.endpoint_timeout,
                        request_template=expert_cfg.request_template,
                        response_mapping=expert_cfg.response_mapping,
                        endpoint_response_max_bytes=expert_cfg.endpoint_response_max_bytes,
                        security_config=security_config,
                        environment=environment,
                        retry_config=remote_retry_config,
                        circuit_breaker_config=remote_circuit_breaker_config,
                    )
                else:
                    instance = expert_cls()
            except ExpertError:
                raise
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

            # Load weights (or validate remote prerequisites) ---------------
            weights_path = expert_cfg.weights or ""
            try:
                if not is_remote and expert_cfg.integrity and expert_cfg.integrity.sha256:
                    verify_local_sha256(
                        weights_path,
                        expert_cfg.integrity.sha256,
                        expert_name=expert_cfg.name,
                    )
                    emit_security_audit(
                        "local_artifact_integrity",
                        "success",
                        expert_name=expert_cfg.name,
                        metadata={"algorithm": "sha256"},
                    )
                instance.load_weights(weights_path)
                if is_remote and expert_cfg.integrity:
                    _verify_remote_manifest_if_configured(
                        expert_cfg,
                        endpoint=getattr(instance, "endpoint", expert_cfg.endpoint or ""),
                        security_config=security_config,
                        environment=environment,
                    )
            except ExpertError:
                if not is_remote and expert_cfg.integrity and expert_cfg.integrity.sha256:
                    emit_security_audit(
                        "local_artifact_integrity",
                        "failure",
                        expert_name=expert_cfg.name,
                        metadata={"algorithm": "sha256"},
                    )
                raise
            except Exception as exc:
                raise ExpertError(
                    f"Expert '{expert_cfg.name}' failed to load weights "
                    f"from '{weights_path}': {exc}",
                    context={"expert": expert_cfg.name, "weights": weights_path},
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
