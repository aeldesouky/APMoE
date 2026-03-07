"""Inference pipeline orchestrator for the APMoE framework.

The :class:`InferencePipeline` is the heart of the framework's runtime.  It
owns the two-phase execution loop:

**Phase A — Modality Processing (parallel):**

1. Receives raw multi-modal input as ``dict[str, Any]`` (modality-name → raw data).
2. For each modality independently:

   * Routes to the appropriate :class:`~apmoe.modality.base.ModalityProcessor`
     (validate + preprocess).
   * Passes through the ``Cleaner → Anonymizer → (optional) Embedder`` chain.

3. Produces ``dict[str, ProcessedInput]`` (modality-name → processed output).

**Phase B — Expert Inference + Aggregation:**

1. For each registered expert, selects the subset of processed data matching
   its :meth:`~apmoe.experts.base.ExpertPlugin.declared_modalities`.
2. Calls ``expert.predict(inputs)`` and collects all
   :class:`~apmoe.core.types.ExpertOutput` objects.
3. Passes them to the :class:`~apmoe.aggregation.base.AggregatorStrategy` for
   final combination.
4. Returns a :class:`~apmoe.core.types.Prediction`.

Graceful degradation
--------------------
If a modality's raw input is absent from the request, or its processing chain
raises a :class:`~apmoe.core.exceptions.ModalityError`, that modality is
recorded in ``failed_modalities`` and excluded from the processed dict.
Experts whose required modalities are all satisfied still run; experts that
need an absent/failed modality are recorded in
:attr:`~apmoe.core.types.Prediction.skipped_experts`.

Hooks
-----
Four hook lists are provided for observability::

    pipeline.on_before_process.append(lambda name, data: logger.info(name))
    pipeline.on_after_embed.append(lambda name, result: metrics.record(name))
    pipeline.on_after_expert.append(lambda output: logger.info(output.expert_name))
    pipeline.on_after_aggregate.append(lambda pred: logger.info(pred.predicted_age))

Each list can hold multiple callables; they are called in order.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from apmoe.aggregation.base import AggregatorStrategy
from apmoe.core.exceptions import ExpertError, ModalityError, PipelineError
from apmoe.core.types import ExpertOutput, ModalityData, Prediction, ProcessedInput
from apmoe.experts.registry import ExpertRegistry
from apmoe.modality.base import ModalityProcessor
from apmoe.processing.base import AnonymizerStrategy, CleanerStrategy, EmbedderStrategy


# ---------------------------------------------------------------------------
# Modality chain descriptor
# ---------------------------------------------------------------------------


@dataclass
class ModalityChain:
    """Groups all processing components for a single modality.

    An instance of this class is built for each modality at bootstrap time
    (see :meth:`~apmoe.core.app.APMoEApp.from_config`) and stored in the
    pipeline's internal ``chains`` mapping.

    Attributes:
        processor: The :class:`~apmoe.modality.base.ModalityProcessor` that
            validates and preprocesses raw input for this modality.
        cleaner: The :class:`~apmoe.processing.base.CleanerStrategy` applied
            after preprocessing.
        anonymizer: The :class:`~apmoe.processing.base.AnonymizerStrategy`
            applied after cleaning.
        embedder: Optional :class:`~apmoe.processing.base.EmbedderStrategy`.
            When set, the output of the chain is an
            :class:`~apmoe.core.types.EmbeddingResult`; when ``None``, the
            output is the preprocessed
            :class:`~apmoe.core.types.ModalityData`.
    """

    processor: ModalityProcessor
    cleaner: CleanerStrategy
    anonymizer: AnonymizerStrategy
    embedder: EmbedderStrategy | None = None


# ---------------------------------------------------------------------------
# Hook type aliases
# ---------------------------------------------------------------------------

#: Called before a modality's processing chain starts.
#: Signature: ``(modality_name: str, raw_data: Any) -> None``
OnBeforeProcess = Callable[[str, Any], None]

#: Called after the full processing chain (including optional embedding) for
#: a modality completes.
#: Signature: ``(modality_name: str, result: ProcessedInput) -> None``
OnAfterEmbed = Callable[[str, ProcessedInput], None]

#: Called after each expert produces its output.
#: Signature: ``(output: ExpertOutput) -> None``
OnAfterExpert = Callable[[ExpertOutput], None]

#: Called after the aggregator produces the final prediction.
#: Signature: ``(prediction: Prediction) -> None``
OnAfterAggregate = Callable[[Prediction], None]


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------


@dataclass
class InferencePipeline:
    """Two-phase inference pipeline orchestrator.

    Owns the full execution lifecycle from raw multi-modal input to a final
    :class:`~apmoe.core.types.Prediction`.  Both synchronous and asynchronous
    execution are supported.

    Attributes:
        chains: Mapping of modality name → :class:`ModalityChain`.
        expert_registry: Live :class:`~apmoe.experts.registry.ExpertRegistry`
            with all experts instantiated and weights loaded.
        aggregator: The :class:`~apmoe.aggregation.base.AggregatorStrategy`
            used to combine expert outputs.
        on_before_process: Hook list fired before each modality's chain.
        on_after_embed: Hook list fired after each modality's chain completes.
        on_after_expert: Hook list fired after each expert produces output.
        on_after_aggregate: Hook list fired after the aggregator runs.

    Example::

        pipeline = InferencePipeline(
            chains={"visual": chain},
            expert_registry=registry,
            aggregator=WeightedAverageAggregator(),
        )
        prediction = pipeline.run({"visual": image_bytes})
    """

    chains: dict[str, ModalityChain]
    expert_registry: ExpertRegistry
    aggregator: AggregatorStrategy
    on_before_process: list[OnBeforeProcess] = field(default_factory=list)
    on_after_embed: list[OnAfterEmbed] = field(default_factory=list)
    on_after_expert: list[OnAfterExpert] = field(default_factory=list)
    on_after_aggregate: list[OnAfterAggregate] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_hooks(self, hooks: list[Callable[..., None]], *args: Any) -> None:
        """Invoke every callable in *hooks* with *args*.

        Args:
            hooks: List of callables to invoke.
            *args: Positional arguments forwarded to each callable.
        """
        for hook in hooks:
            hook(*args)

    def _process_one_modality(
        self,
        modality_name: str,
        raw_data: Any,
        chain: ModalityChain,
    ) -> ProcessedInput:
        """Run *raw_data* through the full processing chain for *modality_name*.

        Steps:
        1. Fire ``on_before_process`` hooks.
        2. Validate with ``chain.processor.validate()``.
        3. Preprocess with ``chain.processor.preprocess()``.
        4. Clean with ``chain.cleaner.clean()``.
        5. Anonymize with ``chain.anonymizer.anonymize()``.
        6. Optionally embed with ``chain.embedder.embed()``.
        7. Fire ``on_after_embed`` hooks.

        Args:
            modality_name: Canonical modality name (used in error messages).
            raw_data: Raw input object for this modality.
            chain: The :class:`ModalityChain` to apply.

        Returns:
            The :data:`~apmoe.core.types.ProcessedInput` produced by the chain.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If any step fails.
        """
        # 1. Before-process hooks
        self._run_hooks(self.on_before_process, modality_name, raw_data)

        # 2. Validate
        try:
            valid = chain.processor.validate(raw_data)
        except Exception as exc:  # noqa: BLE001
            raise ModalityError(
                f"Validation raised an exception for modality '{modality_name}': {exc}",
                context={"modality": modality_name},
            ) from exc

        if not valid:
            raise ModalityError(
                f"Input validation failed for modality '{modality_name}'.",
                context={"modality": modality_name},
            )

        # 3. Preprocess
        try:
            modality_data: ModalityData = chain.processor.preprocess(raw_data)
        except ModalityError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModalityError(
                f"Preprocessing failed for modality '{modality_name}': {exc}",
                context={"modality": modality_name},
            ) from exc

        # 4. Clean
        try:
            modality_data = chain.cleaner.clean(modality_data)
        except ModalityError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModalityError(
                f"Cleaning failed for modality '{modality_name}': {exc}",
                context={"modality": modality_name},
            ) from exc

        # 5. Anonymize
        try:
            modality_data = chain.anonymizer.anonymize(modality_data)
        except ModalityError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModalityError(
                f"Anonymization failed for modality '{modality_name}': {exc}",
                context={"modality": modality_name},
            ) from exc

        # 6. Optionally embed
        result: ProcessedInput
        if chain.embedder is not None:
            try:
                result = chain.embedder.embed(modality_data)
            except ModalityError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ModalityError(
                    f"Embedding failed for modality '{modality_name}': {exc}",
                    context={"modality": modality_name},
                ) from exc
        else:
            result = modality_data

        # 7. After-embed hooks
        self._run_hooks(self.on_after_embed, modality_name, result)

        return result

    def _phase_a_sync(
        self, raw_inputs: dict[str, Any]
    ) -> tuple[dict[str, ProcessedInput], dict[str, str]]:
        """Run Phase A (modality processing) synchronously.

        Args:
            raw_inputs: Mapping of modality name → raw data.

        Returns:
            A 2-tuple of (processed, failed_modalities) where:

            - ``processed``: modality-name → :data:`~apmoe.core.types.ProcessedInput`.
            - ``failed_modalities``: modality-name → error string for any
              modality that raised a :class:`~apmoe.core.exceptions.ModalityError`.
        """
        processed: dict[str, ProcessedInput] = {}
        failed: dict[str, str] = {}

        for modality_name, raw_data in raw_inputs.items():
            if modality_name not in self.chains:
                # Input for an unconfigured modality — ignore silently.
                continue
            chain = self.chains[modality_name]
            try:
                processed[modality_name] = self._process_one_modality(
                    modality_name, raw_data, chain
                )
            except ModalityError as exc:
                failed[modality_name] = str(exc)

        return processed, failed

    def _phase_b(
        self,
        processed: dict[str, ProcessedInput],
        failed_modalities: dict[str, str],
        start_time: float,
    ) -> Prediction:
        """Run Phase B (expert inference + aggregation).

        Args:
            processed: The ``dict[str, ProcessedInput]`` from Phase A.
            failed_modalities: Modalities that failed during Phase A.
            start_time: ``time.monotonic()`` value from before Phase A,
                used to compute total pipeline latency.

        Returns:
            The final :class:`~apmoe.core.types.Prediction`.

        Raises:
            :class:`~apmoe.core.exceptions.PipelineError`: If no experts
                produce output.
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert's
                ``predict()`` method raises an unhandled exception.
        """
        available = set(processed.keys())
        runnable = self.expert_registry.get_runnable_experts(available)
        skipped_names: list[str] = self.expert_registry.get_skipped_experts(available)

        expert_outputs: list[ExpertOutput] = []
        for expert in runnable:
            expert_inputs = {
                mod: processed[mod]
                for mod in expert.declared_modalities()
                if mod in processed
            }
            try:
                output = expert.predict(expert_inputs)
            except ExpertError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ExpertError(
                    f"Expert '{expert.name}' raised an unhandled exception: {exc}",
                    context={"expert": expert.name},
                ) from exc

            expert_outputs.append(output)
            self._run_hooks(self.on_after_expert, output)

        if not expert_outputs:
            raise PipelineError(
                "No experts produced output.  Cannot aggregate an empty list.",
                context={
                    "available_modalities": sorted(available),
                    "skipped_experts": skipped_names,
                    "failed_modalities": list(failed_modalities.keys()),
                },
            )

        # Aggregate
        raw_prediction = self.aggregator.aggregate(expert_outputs)

        # Enrich with pipeline-level metadata
        prediction = Prediction(
            predicted_age=raw_prediction.predicted_age,
            confidence=raw_prediction.confidence,
            confidence_interval=raw_prediction.confidence_interval,
            per_expert_outputs=raw_prediction.per_expert_outputs,
            skipped_experts=skipped_names,
            metadata={
                **raw_prediction.metadata,
                "pipeline_latency_s": round(time.monotonic() - start_time, 6),
                "available_modalities": sorted(available),
                "failed_modalities": failed_modalities,
            },
        )

        self._run_hooks(self.on_after_aggregate, prediction)

        return prediction

    # ------------------------------------------------------------------
    # Public execution interface
    # ------------------------------------------------------------------

    def run(self, raw_inputs: dict[str, Any]) -> Prediction:
        """Execute the full inference pipeline synchronously.

        Processes each configured modality present in *raw_inputs* through its
        ``Processor → Cleaner → Anonymizer → (optional) Embedder`` chain
        (sequentially), then dispatches processed data to the matching experts,
        and finally aggregates all expert outputs into a
        :class:`~apmoe.core.types.Prediction`.

        Missing modalities are handled gracefully: if a modality key is absent
        from *raw_inputs* or its processing chain fails, experts that
        require only those modalities are skipped and recorded in
        :attr:`~apmoe.core.types.Prediction.skipped_experts`.

        Args:
            raw_inputs: Mapping of modality name → raw data (bytes, ndarray,
                file path, etc. — whatever the concrete
                :class:`~apmoe.modality.base.ModalityProcessor` accepts).

        Returns:
            A :class:`~apmoe.core.types.Prediction` containing the aggregated
            age estimate, per-expert breakdown, and pipeline metadata.

        Raises:
            :class:`~apmoe.core.exceptions.PipelineError`: If no experts can
                run (all skipped or failed).
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert's
                ``predict()`` raises an unhandled exception.
        """
        start_time = time.monotonic()
        processed, failed = self._phase_a_sync(raw_inputs)
        return self._phase_b(processed, failed, start_time)

    async def run_async(self, raw_inputs: dict[str, Any]) -> Prediction:
        """Execute the pipeline with parallel modality processing.

        Phase A runs each modality's processing chain concurrently using
        ``asyncio`` and a thread-pool executor (since processing chains are
        typically CPU-bound and not natively async).  Phase B (expert
        inference + aggregation) runs sequentially after all modalities are
        processed.

        Args:
            raw_inputs: Same as :meth:`run`.

        Returns:
            A :class:`~apmoe.core.types.Prediction`.

        Raises:
            :class:`~apmoe.core.exceptions.PipelineError`: If no experts can run.
            :class:`~apmoe.core.exceptions.ExpertError`: If an expert raises.
        """
        start_time = time.monotonic()
        loop = asyncio.get_event_loop()

        modality_names = [name for name in raw_inputs if name in self.chains]

        async def _process_one(
            name: str,
        ) -> tuple[str, ProcessedInput | None, str | None]:
            chain = self.chains[name]
            try:
                result = await loop.run_in_executor(
                    None,
                    self._process_one_modality,
                    name,
                    raw_inputs[name],
                    chain,
                )
                return name, result, None
            except ModalityError as exc:
                return name, None, str(exc)

        results = await asyncio.gather(*[_process_one(name) for name in modality_names])

        processed: dict[str, ProcessedInput] = {}
        failed: dict[str, str] = {}
        for name, result, error in results:
            if result is not None:
                processed[name] = result
            else:
                failed[name] = error or "unknown error"

        return self._phase_b(processed, failed, start_time)
