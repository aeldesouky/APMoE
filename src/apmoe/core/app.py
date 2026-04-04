"""APMoEApp: the IoC container and main entry point for the APMoE framework.

This module implements the Hollywood Principle: "Don't call us, we'll call
you."  Developers declare components in a JSON config file; :class:`APMoEApp`
owns the bootstrap lifecycle, wires all components together, and exposes a
simple ``predict`` / ``serve`` / ``validate`` API.

Typical usage::

    from apmoe.core.app import APMoEApp

    app = APMoEApp.from_config("configs/my_project.json")
    prediction = app.predict({"visual": image_bytes, "audio": audio_bytes})
    print(prediction.predicted_age)

Or as a long-running API server::

    app = APMoEApp.from_config("configs/my_project.json")
    app.serve()   # blocks until the server is stopped

Bootstrap order
---------------
1. Load and validate the JSON config file.
2. Build :class:`~apmoe.modality.base.ModalityProcessor` instances via
   :class:`~apmoe.modality.factory.ModalityProcessorFactory`.
3. Resolve :class:`~apmoe.processing.base.CleanerStrategy`,
   :class:`~apmoe.processing.base.AnonymizerStrategy`, and optionally
   :class:`~apmoe.processing.base.EmbedderStrategy` for each modality.
4. Build the :class:`~apmoe.experts.registry.ExpertRegistry` (resolves
   expert classes, instantiates them, loads pretrained weights).
5. Resolve the :class:`~apmoe.aggregation.base.AggregatorStrategy` (and
   optionally load its combiner weights).
6. Wire everything into an :class:`~apmoe.core.pipeline.InferencePipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.config import FrameworkConfig, load_config
from apmoe.core.exceptions import ConfigurationError, ExpertError, ServingError
from apmoe.core.pipeline import InferencePipeline, ModalityChain
from apmoe.core.registry import legacy_dotted_import_alias
from apmoe.core.types import Prediction
from apmoe.experts.registry import ExpertRegistry
from apmoe.modality.factory import ModalityProcessorFactory
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    EmbedderStrategy,
    anonymizer_registry,
    cleaner_registry,
    embedder_registry,
)


class APMoEApp:
    """IoC container and application lifecycle manager for the APMoE framework.

    :class:`APMoEApp` is the single façade that application code interacts with.
    It holds references to all wired framework components and exposes the three
    main operations:

    * :meth:`predict` — run the inference pipeline on raw inputs.
    * :meth:`serve` — start the FastAPI HTTP server (Phase 4).
    * :meth:`validate` — health-check all loaded components.

    Do not construct :class:`APMoEApp` directly.  Use the
    :meth:`from_config` class method which performs the full bootstrap
    sequence.

    Attributes:
        _config: The fully-validated :class:`~apmoe.core.config.FrameworkConfig`.
        _pipeline: The wired :class:`~apmoe.core.pipeline.InferencePipeline`.
        _expert_registry: Live :class:`~apmoe.experts.registry.ExpertRegistry`.
        _aggregator: The active :class:`~apmoe.aggregation.base.AggregatorStrategy`.
    """

    def __init__(
        self,
        config: FrameworkConfig,
        pipeline: InferencePipeline,
        expert_registry: ExpertRegistry,
        aggregator: AggregatorStrategy,
    ) -> None:
        """Initialise the app.  Prefer :meth:`from_config` over direct construction.

        Args:
            config: Validated framework config.
            pipeline: Fully-wired inference pipeline.
            expert_registry: Live expert instance registry.
            aggregator: Active aggregation strategy.
        """
        self._config = config
        self._pipeline = pipeline
        self._expert_registry = expert_registry
        self._aggregator = aggregator

    # ------------------------------------------------------------------
    # Bootstrap factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, path: str | Path) -> APMoEApp:
        """Load a JSON config file and bootstrap the full framework.

        This is the **primary constructor** for :class:`APMoEApp`.  It performs
        the complete bootstrap sequence (see module docstring) and returns a
        fully initialised app ready for inference.

        Args:
            path: Filesystem path to the JSON config file.

        Returns:
            A fully-initialised :class:`APMoEApp` instance.

        Raises:
            :class:`~apmoe.core.exceptions.ConfigurationError`: If the config
                file cannot be read, is malformed, or fails schema validation,
                or if any component class cannot be resolved or instantiated.
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert cannot
                be resolved, instantiated, or loaded.
            :class:`~apmoe.core.exceptions.ModalityError`: If a modality
                processor cannot be resolved or instantiated.
        """
        # 1. Load and validate config
        cfg = load_config(path)
        apmoe_cfg = cfg.apmoe

        # 2. Build modality processors
        processors = ModalityProcessorFactory.from_configs(apmoe_cfg.modalities)

        # 3. Build per-modality processing chains
        chains: dict[str, ModalityChain] = {}
        for modality_cfg in apmoe_cfg.modalities:
            mod_name = modality_cfg.name
            processor = processors[mod_name]
            pipeline_cfg = modality_cfg.pipeline

            # Resolve cleaner (rewrite legacy dotted paths before registry lookup)
            try:
                cleaner_cls = cleaner_registry.resolve(
                    legacy_dotted_import_alias(pipeline_cfg.cleaner),
                )
                cleaner: CleanerStrategy = cleaner_cls()
            except Exception as exc:  # noqa: BLE001
                raise ConfigurationError(
                    f"Cannot resolve cleaner '{pipeline_cfg.cleaner}' "
                    f"for modality '{mod_name}': {exc}",
                    context={"modality": mod_name, "cleaner": pipeline_cfg.cleaner},
                ) from exc

            # Resolve anonymizer (rewrite legacy dotted paths before registry lookup)
            try:
                anonymizer_cls = anonymizer_registry.resolve(
                    legacy_dotted_import_alias(pipeline_cfg.anonymizer),
                )
                anonymizer: AnonymizerStrategy = anonymizer_cls()
            except Exception as exc:  # noqa: BLE001
                raise ConfigurationError(
                    f"Cannot resolve anonymizer '{pipeline_cfg.anonymizer}' "
                    f"for modality '{mod_name}': {exc}",
                    context={"modality": mod_name, "anonymizer": pipeline_cfg.anonymizer},
                ) from exc

            # Optionally resolve embedder
            embedder: EmbedderStrategy | None = None
            if pipeline_cfg.embedder is not None:
                try:
                    embedder_cls = embedder_registry.resolve(pipeline_cfg.embedder)
                    embedder = embedder_cls()
                except Exception as exc:  # noqa: BLE001
                    raise ConfigurationError(
                        f"Cannot resolve embedder '{pipeline_cfg.embedder}' "
                        f"for modality '{mod_name}': {exc}",
                        context={"modality": mod_name, "embedder": pipeline_cfg.embedder},
                    ) from exc

            chains[mod_name] = ModalityChain(
                processor=processor,
                cleaner=cleaner,
                anonymizer=anonymizer,
                embedder=embedder,
            )

        # 4. Build expert registry (resolves classes, instantiates, loads weights)
        expert_reg = ExpertRegistry.from_configs(apmoe_cfg.experts, apmoe_cfg.modalities)

        # 5. Resolve aggregation strategy
        try:
            agg_cls = aggregator_registry.resolve(apmoe_cfg.aggregation.strategy)
            aggregator: AggregatorStrategy = agg_cls()
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                f"Cannot resolve aggregator '{apmoe_cfg.aggregation.strategy}': {exc}",
                context={"strategy": apmoe_cfg.aggregation.strategy},
            ) from exc

        # Load aggregator combiner weights if specified
        if apmoe_cfg.aggregation.weights_path is not None:
            try:
                aggregator.load_weights(apmoe_cfg.aggregation.weights_path)
            except Exception as exc:  # noqa: BLE001
                raise ConfigurationError(
                    f"Failed to load aggregator weights from "
                    f"'{apmoe_cfg.aggregation.weights_path}': {exc}",
                    context={"weights_path": apmoe_cfg.aggregation.weights_path},
                ) from exc

        # 6. Wire the inference pipeline
        pipeline = InferencePipeline(
            chains=chains,
            expert_registry=expert_reg,
            aggregator=aggregator,
        )

        return cls(
            config=cfg,
            pipeline=pipeline,
            expert_registry=expert_reg,
            aggregator=aggregator,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, inputs: dict[str, Any]) -> Prediction:
        """Run the inference pipeline on raw multi-modal inputs.

        Delegates directly to :meth:`~apmoe.core.pipeline.InferencePipeline.run`.

        Args:
            inputs: Mapping of modality name → raw data.  Keys that are not
                configured modalities are silently ignored.  Missing configured
                modalities cause dependent experts to be skipped.

        Returns:
            A :class:`~apmoe.core.types.Prediction` containing the final
            aggregated age estimate, per-expert breakdown, skipped expert names,
            and pipeline latency metadata.

        Raises:
            :class:`~apmoe.core.exceptions.PipelineError`: If no experts can
                produce output (all required modalities missing or failed).
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert raises
                an unhandled exception during inference.
        """
        return self._pipeline.run(inputs)

    async def predict_async(self, inputs: dict[str, Any]) -> Prediction:
        """Asynchronously run inference with parallel modality processing.

        Delegates to
        :meth:`~apmoe.core.pipeline.InferencePipeline.run_async`.

        Args:
            inputs: Same as :meth:`predict`.

        Returns:
            A :class:`~apmoe.core.types.Prediction`.

        Raises:
            :class:`~apmoe.core.exceptions.PipelineError`: If no experts can run.
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert raises.
        """
        return await self._pipeline.run_async(inputs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, Any]:
        """Validate the framework configuration and component health.

        Performs the following checks:

        * Expert ``is_loaded`` health status for every registered expert.
        * Existence of each expert's configured weight file path.
        * Existence of the aggregator's combiner weight file (if configured).

        Args: (none)

        Returns:
            A dict with keys:

            * ``"valid"`` (bool) — ``True`` iff no issues were found.
            * ``"expert_health"`` (dict[str, bool]) — per-expert loaded status.
            * ``"issues"`` (list[str]) — human-readable descriptions of any
              problems found.

        Raises:
            :class:`~apmoe.core.exceptions.ConfigurationError`: If any
                critical issue is found (issues list is non-empty).
        """
        issues: list[str] = []

        # Expert health check
        health = self._expert_registry.health_check()
        unhealthy = [name for name, ok in health.items() if not ok]
        if unhealthy:
            issues.append(f"Experts not loaded: {unhealthy}")

        # Expert weight file existence
        for expert_cfg in self._config.apmoe.experts:
            weights_path = Path(expert_cfg.weights)
            if not weights_path.exists():
                issues.append(
                    f"Weight file missing for expert '{expert_cfg.name}': {weights_path}"
                )

        # Aggregator combiner weight file (if configured)
        if self._config.apmoe.aggregation.weights_path is not None:
            agg_weights = Path(self._config.apmoe.aggregation.weights_path)
            if not agg_weights.exists():
                issues.append(f"Aggregator weight file missing: {agg_weights}")

        report: dict[str, Any] = {
            "valid": len(issues) == 0,
            "expert_health": health,
            "issues": issues,
        }

        if issues:
            raise ConfigurationError(
                f"Validation found {len(issues)} issue(s): " + "; ".join(issues),
                context={"issues": issues},
            )

        return report

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def serve(self, **uvicorn_kwargs: Any) -> None:
        """Start the FastAPI HTTP serving layer.

        Imports the :func:`~apmoe.serving.app_factory.create_api` factory
        (implemented in Phase 4) to build the FastAPI application, then
        launches ``uvicorn`` with the settings from the ``serving`` config
        block.

        This method **blocks** until the server is stopped.

        Args:
            **uvicorn_kwargs: Additional keyword arguments forwarded directly
                to ``uvicorn.run()`` (e.g. ``ssl_keyfile``, ``ssl_certfile``).

        Raises:
            :class:`~apmoe.core.exceptions.ServingError`: If the serving
                dependencies cannot be imported or uvicorn fails to start.
        """
        try:
            import uvicorn
            from apmoe.serving.app_factory import create_api  # type: ignore[import]
        except ImportError as exc:
            raise ServingError(
                f"Cannot start server: {exc}.  "
                f"Ensure serving dependencies (fastapi, uvicorn) are installed.",
                context={"error": str(exc)},
            ) from exc

        api = create_api(self)
        serving_cfg = self._config.apmoe.serving

        uvicorn.run(
            api,
            host=serving_cfg.host,
            port=serving_cfg.port,
            workers=serving_cfg.workers,
            log_level=serving_cfg.log_level,
            **uvicorn_kwargs,
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> FrameworkConfig:
        """The loaded and validated :class:`~apmoe.core.config.FrameworkConfig`."""
        return self._config

    @property
    def pipeline(self) -> InferencePipeline:
        """The wired :class:`~apmoe.core.pipeline.InferencePipeline`."""
        return self._pipeline

    @property
    def expert_registry(self) -> ExpertRegistry:
        """Live :class:`~apmoe.experts.registry.ExpertRegistry`."""
        return self._expert_registry

    @property
    def aggregator(self) -> AggregatorStrategy:
        """Active :class:`~apmoe.aggregation.base.AggregatorStrategy`."""
        return self._aggregator

    def get_info(self) -> dict[str, Any]:
        """Return a framework-level info summary.

        Useful for the ``GET /info`` endpoint (Phase 4) and for diagnostics.

        Returns:
            A JSON-serialisable dict containing:

            * ``"version"`` — framework version string.
            * ``"experts"`` — list of per-expert info dicts from
              :meth:`~apmoe.experts.base.ExpertPlugin.get_info`.
            * ``"modalities"`` — list of configured modality names.
            * ``"aggregator"`` — aggregator info dict.
            * ``"serving"`` — serving config dict.
        """
        import apmoe

        return {
            "version": apmoe.__version__,
            "experts": [
                expert.get_info() for expert in self._expert_registry.all_instances()
            ],
            "modalities": [m.name for m in self._config.apmoe.modalities],
            "aggregator": self._aggregator.get_info(),
            "serving": self._config.apmoe.serving.model_dump(),
        }

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        experts = self._expert_registry.list_experts()
        modalities = [m.name for m in self._config.apmoe.modalities]
        return (
            f"APMoEApp("
            f"modalities={modalities}, "
            f"experts={experts}, "
            f"aggregator={type(self._aggregator).__name__}"
            f")"
        )
