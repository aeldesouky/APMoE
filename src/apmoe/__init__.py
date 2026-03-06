"""APMoE — Age Prediction using Mixture of Experts.

This is the public API surface of the framework.  Import from here in
application code; do **not** import framework internals directly (they are
subject to change between minor versions).

Quick start::

    from apmoe.core.config import load_config
    from apmoe.core.registry import Registry
    from apmoe.core.types import ModalityData, EmbeddingResult, ExpertOutput, Prediction
    from apmoe.core.exceptions import APMoEError

Phase 1 public symbols
-----------------------
- Config loading and Pydantic models.
- Generic :class:`~apmoe.core.registry.Registry`.
- Shared data types.
- Exception hierarchy.

Phase 2 public symbols
-----------------------
- :class:`~apmoe.modality.base.ModalityProcessor` ABC.
- :class:`~apmoe.modality.factory.ModalityProcessorFactory`
  and :data:`~apmoe.modality.factory.modality_registry`.
- :class:`~apmoe.processing.base.CleanerStrategy`,
  :class:`~apmoe.processing.base.AnonymizerStrategy`,
  :class:`~apmoe.processing.base.EmbedderStrategy` ABCs + their registries.
- :class:`~apmoe.experts.base.ExpertPlugin` ABC.
- :class:`~apmoe.experts.registry.ExpertRegistry`
  and :data:`~apmoe.experts.registry.expert_registry`.
- :class:`~apmoe.aggregation.base.AggregatorStrategy` ABC
  and :data:`~apmoe.aggregation.base.aggregator_registry`.
"""

from apmoe.aggregation.base import AggregatorStrategy, aggregator_registry
from apmoe.core.config import (
    AggregationConfig,
    APMoEConfig,
    ExpertConfig,
    FrameworkConfig,
    ModalityConfig,
    PipelineConfig,
    ServingConfig,
    load_config,
)
from apmoe.core.exceptions import (
    APMoEError,
    ConfigurationError,
    ExpertError,
    ModalityError,
    PipelineError,
    RegistryError,
    ServingError,
)
from apmoe.core.registry import Registry
from apmoe.core.types import (
    EmbeddingResult,
    ExpertOutput,
    ModalityData,
    Prediction,
    ProcessedInput,
)
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import ExpertRegistry, expert_registry
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import ModalityProcessorFactory, modality_registry
from apmoe.processing.base import (
    AnonymizerStrategy,
    CleanerStrategy,
    EmbedderStrategy,
    anonymizer_registry,
    cleaner_registry,
    embedder_registry,
)

__version__ = "0.1.0"
__all__ = [
    "APMoEConfig",
    # Exceptions
    "APMoEError",
    # Config
    "AggregationConfig",
    # Aggregation
    "AggregatorStrategy",
    # Processing strategies
    "AnonymizerStrategy",
    "CleanerStrategy",
    "ConfigurationError",
    "EmbedderStrategy",
    # Types
    "EmbeddingResult",
    "ExpertConfig",
    "ExpertError",
    "ExpertOutput",
    # Expert layer
    "ExpertPlugin",
    "ExpertRegistry",
    "FrameworkConfig",
    "ModalityConfig",
    "ModalityData",
    "ModalityError",
    # Modality layer
    "ModalityProcessor",
    "ModalityProcessorFactory",
    "PipelineConfig",
    "PipelineError",
    "Prediction",
    "ProcessedInput",
    # Generic registry
    "Registry",
    "RegistryError",
    "ServingConfig",
    "ServingError",
    # Version
    "__version__",
    "aggregator_registry",
    "anonymizer_registry",
    "cleaner_registry",
    "embedder_registry",
    "expert_registry",
    "load_config",
    "modality_registry",
]
