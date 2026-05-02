"""APMoE configuration system.

The framework uses a two-step config approach:

1. A **JSON file** defines the full configuration (modalities, experts,
   aggregation strategy, serving parameters).
2. ``APMOE_``-prefixed **environment variables** can override individual
   leaf values at runtime without touching the file.

The JSON document is validated against a hierarchy of Pydantic models
(:class:`FrameworkConfig`) on load, so misconfiguration produces clear,
actionable error messages.

Usage::

    from apmoe.core.config import load_config

    cfg = load_config("configs/default.json")
    print(cfg.apmoe.serving.port)

Environment variable override examples::

    APMOE_SERVING_HOST=0.0.0.0
    APMOE_SERVING_PORT=9000
    APMOE_SERVING_WORKERS=8
    APMOE_SERVING_CORS_ORIGINS=http://localhost:3000,https://myapp.com
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from apmoe.core.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Configuration for a single modality's processing chain.

    Attributes:
        cleaner: Dotted import path or registered name of the
            :class:`~apmoe.processing.base.CleanerStrategy` to use.
        anonymizer: Dotted import path or registered name of the
            :class:`~apmoe.processing.base.AnonymizerStrategy` to use.
        embedder: Optional dotted import path or registered name of the
            :class:`~apmoe.processing.base.EmbedderStrategy`.  When omitted,
            experts for this modality receive preprocessed
            :class:`~apmoe.core.types.ModalityData` directly.
    """

    cleaner: str
    anonymizer: str
    embedder: str | None = None


class ModalityConfig(BaseModel):
    """Configuration for a single modality.

    Attributes:
        name: Canonical name used as the key throughout the pipeline
            (e.g. ``"visual"``, ``"audio"``, ``"eeg"``).  Must be unique
            across all modality entries.
        processor: Dotted import path or registered name of the
            :class:`~apmoe.modality.base.ModalityProcessor`.
        pipeline: The processing chain definition for this modality.
    """

    name: str
    processor: str
    pipeline: PipelineConfig

    @field_validator("name")
    @classmethod
    def name_must_be_non_empty(cls, v: str) -> str:
        """Ensure modality name is a non-empty, stripped identifier."""
        v = v.strip()
        if not v:
            raise ValueError("Modality name must not be empty.")
        return v


class ExpertConfig(BaseModel):
    """Configuration for a single expert plugin.

    Attributes:
        name: Unique identifier for this expert instance (e.g.
            ``"face_age_expert"``).
        class_path: Dotted import path or registered name of the
            :class:`~apmoe.experts.base.ExpertPlugin` implementation.
            Declared as ``"class"`` in JSON (mapped via ``alias``).
        weights: Filesystem path to the pretrained weight file (``.pt``,
            ``.onnx``, etc.).
        modalities: List of modality names this expert consumes.  Every
            entry must correspond to a modality declared in the
            :attr:`APMoEConfig.modalities` list.
        extra: Any additional expert-specific config keys that the concrete
            class may consume (e.g. ``temperature``, ``threshold``).
    """

    name: str
    class_path: str = Field(..., alias="class")
    weights: str
    modalities: list[str]
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @field_validator("modalities")
    @classmethod
    def modalities_must_be_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one modality is declared per expert."""
        if not v:
            raise ValueError("An expert must declare at least one modality.")
        return v


class AggregationConfig(BaseModel):
    """Configuration for the aggregation/combination strategy.

    Attributes:
        strategy: Dotted import path or registered name of the
            :class:`~apmoe.aggregation.base.AggregatorStrategy`.
        weights: Optional mapping of expert-name → numeric weight for
            weighted average strategies.
        weights_path: Optional path to a pretrained combiner model weight
            file (used by ``LearnedCombiner``).
        extra: Additional strategy-specific parameters.
    """

    strategy: str
    weights: dict[str, float] | None = None
    weights_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ServingConfig(BaseModel):
    """HTTP serving parameters.

    Attributes:
        host: Interface address for uvicorn to bind to.
        port: TCP port number.
        workers: Number of uvicorn worker processes.
        cors_origins: Allowed CORS origin patterns.
        rate_limit: Maximum requests per minute per client IP (``None``
            disables rate limiting).
        log_level: Uvicorn log level (``"debug"``, ``"info"``, ``"warning"``,
            ``"error"``, ``"critical"``).
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int | None = None
    log_level: str = "info"

    @field_validator("port")
    @classmethod
    def port_must_be_valid(cls, v: int) -> int:
        """Validate TCP port is in the usable range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}.")
        return v

    @field_validator("workers")
    @classmethod
    def workers_must_be_positive(cls, v: int) -> int:
        """Ensure at least one worker process."""
        if v < 1:
            raise ValueError(f"Workers must be ≥ 1, got {v}.")
        return v


class APMoEConfig(BaseModel):
    """Top-level framework configuration block (the ``"apmoe"`` JSON key).

    Attributes:
        modalities: Ordered list of modality definitions.
        experts: List of expert plugin definitions.
        aggregation: Aggregation strategy config.
        serving: HTTP serving config (defaults applied when absent).
        confidence_threshold: Optional confidence gate in ``[0.0, 1.0]``.
            When the aggregated :attr:`~apmoe.core.types.Prediction.confidence`
            falls below this value the pipeline populates
            ``Prediction.metadata["recommendations"]`` with a list of
            actionable improvement hints.  Set to ``null`` (or omit) to
            disable the recommendation engine.
    """

    modalities: list[ModalityConfig]
    experts: list[ExpertConfig]
    aggregation: AggregationConfig
    serving: ServingConfig = Field(default_factory=ServingConfig)
    confidence_threshold: float | None = None

    @field_validator("confidence_threshold")
    @classmethod
    def confidence_threshold_must_be_in_range(cls, v: float | None) -> float | None:
        """Ensure the threshold is a probability in ``[0.0, 1.0]`` when set."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {v}."
            )
        return v

    @model_validator(mode="after")
    def validate_expert_modalities_declared(self) -> "APMoEConfig":
        """Ensure every expert modality is declared in the modalities list."""
        declared = {m.name for m in self.modalities}
        for expert in self.experts:
            unknown = set(expert.modalities) - declared
            if unknown:
                raise ValueError(
                    f"Expert '{expert.name}' references undeclared modalities: "
                    f"{sorted(unknown)}.  Declared modalities: {sorted(declared)}."
                )
        return self

    @model_validator(mode="after")
    def validate_unique_modality_names(self) -> "APMoEConfig":
        """Ensure modality names are unique."""
        seen: set[str] = set()
        for m in self.modalities:
            if m.name in seen:
                raise ValueError(
                    f"Duplicate modality name '{m.name}'. Each modality must have a unique name."
                )
            seen.add(m.name)
        return self

    @model_validator(mode="after")
    def validate_unique_expert_names(self) -> "APMoEConfig":
        """Ensure expert names are unique."""
        seen: set[str] = set()
        for e in self.experts:
            if e.name in seen:
                raise ValueError(
                    f"Duplicate expert name '{e.name}'. Each expert must have a unique name."
                )
            seen.add(e.name)
        return self


class FrameworkConfig(BaseModel):
    """Root config document.  Maps to the top-level JSON object.

    Attributes:
        apmoe: The ``"apmoe"`` configuration block.
    """

    apmoe: APMoEConfig


# ---------------------------------------------------------------------------
# Environment variable override logic
# ---------------------------------------------------------------------------

_SERVING_ENV_MAP: dict[str, tuple[str, type[Any]]] = {
    "APMOE_SERVING_HOST": ("host", str),
    "APMOE_SERVING_PORT": ("port", int),
    "APMOE_SERVING_WORKERS": ("workers", int),
    "APMOE_SERVING_LOG_LEVEL": ("log_level", str),
    "APMOE_SERVING_RATE_LIMIT": ("rate_limit", int),
}


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Mutate *data* in-place with ``APMOE_``-prefixed environment variable overrides.

    Currently supports overriding the ``serving`` block.  Future phases may
    extend this to ``aggregation`` and other sections.

    Args:
        data: The raw deserialized JSON dict (will be mutated).
    """
    apmoe_section: dict[str, Any] = data.setdefault("apmoe", {})
    serving_section: dict[str, Any] = apmoe_section.setdefault("serving", {})

    for env_key, (field_name, cast) in _SERVING_ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        try:
            serving_section[field_name] = cast(raw)
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(
                f"Environment variable {env_key}={raw!r} cannot be cast to {cast.__name__}.",
                context={"env_key": env_key, "raw_value": raw},
            ) from exc

    # APMOE_SERVING_CORS_ORIGINS — comma-separated list
    cors_raw = os.environ.get("APMOE_SERVING_CORS_ORIGINS")
    if cors_raw is not None:
        serving_section["cors_origins"] = [o.strip() for o in cors_raw.split(",") if o.strip()]


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> FrameworkConfig:
    """Load, validate, and return the framework configuration.

    Reads the JSON file at *path*, applies ``APMOE_``-prefixed environment
    variable overrides, and validates the result against the Pydantic schema.

    Args:
        path: Filesystem path to the JSON config file.

    Returns:
        A fully-validated :class:`FrameworkConfig` instance.

    Raises:
        ConfigurationError: If the file cannot be read, the JSON is malformed,
            or the content fails schema validation.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            context={"path": str(config_path)},
        )

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read configuration file: {config_path}",
            context={"path": str(config_path), "error": str(exc)},
        ) from exc

    try:
        data: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Malformed JSON in configuration file: {config_path}",
            context={"path": str(config_path), "error": str(exc)},
        ) from exc

    _apply_env_overrides(data)

    try:
        return FrameworkConfig.model_validate(data)
    except PydanticValidationError as exc:
        # Re-raise with a friendlier message that includes the file path.
        raise ConfigurationError(
            f"Configuration validation failed for '{config_path}':\n{exc}",
            context={"path": str(config_path)},
        ) from exc
