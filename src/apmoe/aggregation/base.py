"""Abstract base class for aggregation strategies.

After all registered experts have produced their individual
:class:`~apmoe.core.types.ExpertOutput` objects, the framework passes them to
an :class:`AggregatorStrategy` which combines them into a single final
:class:`~apmoe.core.types.Prediction`.

This is where the *Mixture of Experts* "gating" logic lives.  Different
strategies implement different combination rules:

- **Mathematical combiners** (no learned parameters) — weighted average,
  median, confidence-weighted blending.
- **Learned combiners** — a small pretrained model that takes expert
  predictions and confidences as input and outputs the final prediction.

Built-in implementations live in :mod:`apmoe.aggregation.builtin` and are
registered with :data:`aggregator_registry`.

Implementing a custom strategy::

    from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
    from apmoe.core.types import ExpertOutput, Prediction

    @aggregator_registry.register("mean_aggregator")
    class MeanAggregator(AggregatorStrategy):
        def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
            age = sum(o.predicted_age for o in outputs) / len(outputs)
            conf = sum(o.confidence for o in outputs) / len(outputs)
            return Prediction(
                predicted_age=age,
                confidence=conf,
                per_expert_outputs=outputs,
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from apmoe.core.registry import Registry
from apmoe.core.types import ExpertOutput, Prediction

#: Global registry for :class:`AggregatorStrategy` implementations.
#:
#: Built-in aggregators are auto-registered when
#: ``apmoe.aggregation.builtin`` is imported.  Third-party strategies
#: register via ``@aggregator_registry.register("name")``.
aggregator_registry: Registry[AggregatorStrategy] = Registry("aggregators")


class AggregatorStrategy(ABC):
    """Abstract strategy that combines multiple expert predictions into one.

    The aggregator is the final stage of the inference pipeline.  It receives
    a list of :class:`~apmoe.core.types.ExpertOutput` objects — one from each
    expert that ran — and must return a single
    :class:`~apmoe.core.types.Prediction`.

    Framework contract
    ------------------
    - :meth:`aggregate` will **never be called with an empty list**.  The
      pipeline raises a :class:`~apmoe.core.exceptions.PipelineError` before
      reaching the aggregator if no experts produced output.
    - The aggregator **must** include all received
      :class:`~apmoe.core.types.ExpertOutput` objects in the
      :attr:`~apmoe.core.types.Prediction.per_expert_outputs` field so that
      callers can inspect per-expert contributions.
    - The :attr:`~apmoe.core.types.Prediction.confidence` field must be in
      ``[0.0, 1.0]``.
    - Optionally, the aggregator may populate
      :attr:`~apmoe.core.types.Prediction.confidence_interval`.

    Learned combiners
    -----------------
    Aggregators that use a pretrained combiner model should override
    :meth:`load_weights` (default: no-op) and be listed in the config with a
    ``weights_path`` entry.  The framework does **not** automatically call
    :meth:`load_weights`; it is the responsibility of the bootstrap layer
    (:class:`~apmoe.core.app.APMoEApp`) to call it during initialisation.

    Example::

        class UniformAverageAggregator(AggregatorStrategy):
            def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
                n = len(outputs)
                age = sum(o.predicted_age for o in outputs) / n
                conf = sum(o.confidence for o in outputs) / n
                return Prediction(
                    predicted_age=age,
                    confidence=min(conf, 1.0),
                    per_expert_outputs=list(outputs),
                )
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        """Combine *outputs* into a single :class:`~apmoe.core.types.Prediction`.

        This is the core method that every aggregation strategy must implement.

        Args:
            outputs: A **non-empty** list of
                :class:`~apmoe.core.types.ExpertOutput` instances, one per
                expert that successfully ran in the current request.  The list
                may contain outputs from single-modality and multi-modal experts
                alike.

        Returns:
            A :class:`~apmoe.core.types.Prediction` with:

            - :attr:`~apmoe.core.types.Prediction.predicted_age` — the final
              combined age estimate.
            - :attr:`~apmoe.core.types.Prediction.confidence` — in
              ``[0.0, 1.0]``.
            - :attr:`~apmoe.core.types.Prediction.per_expert_outputs` — must
              include all items from *outputs*.

        Raises:
            :class:`~apmoe.core.exceptions.PipelineError`: If aggregation fails
                for any reason.
        """

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> None:
        """Load pretrained combiner weights from *path*.

        The default implementation is a **no-op** and is suitable for
        mathematical aggregators that have no learned parameters.

        Subclasses implementing a *learned combiner* should override this method
        to load their pretrained combiner model.

        Args:
            path: Filesystem path to the weight file (e.g. ``.pt``, ``.onnx``).
        """

    def get_info(self) -> dict[str, object]:
        """Return metadata about this aggregator.

        Exposed via the framework's ``GET /info`` endpoint.  Override to add
        strategy-specific configuration info (e.g. weights, thresholds).

        Returns:
            A JSON-serialisable dict with at least ``"aggregator_class"``.
        """
        return {"aggregator_class": type(self).__qualname__}
