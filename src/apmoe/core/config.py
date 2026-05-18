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
    APMOE_SERVING_TOKEN_INVALIDATION_STORE=redis
    APMOE_SERVING_TOKEN_INVALIDATION_REDIS_URL=redis://localhost:6379/0
    APMOE_SERVING_RATE_LIMIT_STORE=redis
    APMOE_SERVING_RATE_LIMIT_REDIS_URL=redis://localhost:6379/0
    APMOE_SERVING_AUTHENTICATION_ENABLED=true
    APMOE_SERVING_AUTHORIZATION_ENABLED=true
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

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

    Experts can run inference either **locally** (loading a weight file from
    disk) or **remotely** (calling an HTTP endpoint).  Exactly one of
    ``weights`` or ``endpoint`` must be provided.

    Attributes:
        name: Unique identifier for this expert instance (e.g.
            ``"face_age_expert"``).
        class_path: Dotted import path or registered name of the
            :class:`~apmoe.experts.base.ExpertPlugin` implementation.
            Declared as ``"class"`` in JSON (mapped via ``alias``).  Use
            ``"apmoe.experts.remote.RemoteExpert"`` for remote experts.
        weights: Filesystem path to the pretrained weight file (``.pt``,
            ``.onnx``, etc.).  Mutually exclusive with ``endpoint``.
        modalities: List of modality names this expert consumes.  Every
            entry must correspond to a modality declared in the
            :attr:`APMoEConfig.modalities` list.
        endpoint: Full HTTP/HTTPS URL of a remote model server.  When set,
            the framework POSTs the processed modality data to this URL at
            inference time instead of running local inference.  Mutually
            exclusive with ``weights``.
        endpoint_headers: HTTP headers to attach to every request.  Values
            that start with ``$`` are expanded from environment variables
            at bootstrap time (e.g. ``"$MY_API_KEY"``).
        endpoint_timeout: Read timeout in seconds for HTTP requests to the
            remote endpoint (default 10.0).
        request_template: A nested JSON-serialisable dict that forms the
            body sent to the remote endpoint.  Leaf string values may
            contain placeholder expressions of the form
            ``"{{modalities.<name>}}"`` (replaced with the serialised
            modality data) or ``"{{expert_name}}"`` (replaced with the
            expert's configured name).  When ``None``, the framework sends
            the default APMoE schema::

                {"expert_name": "...", "modalities": {"<name>": ...}}

            **HuggingFace example**::

                {"inputs": "{{modalities.keystroke}}"}

            **Multi-field example**::

                {"model": "age-v1", "data": "{{modalities.image}}"}

        response_mapping: Maps :class:`~apmoe.core.types.ExpertOutput`
            field names to dot-paths in the remote JSON response.  Supported
            keys: ``"predicted_age"`` (required), ``"confidence"``
            (optional, defaults to ``-1.0``), ``"metadata"`` (optional,
            defaults to ``{}``).

            **Example** — map HuggingFace response::

                {"predicted_age": "[0].age", "confidence": "[0].score"}

            When ``None`` (default), the framework expects the response to
            contain top-level ``predicted_age``, ``confidence``, and
            ``metadata`` keys directly.
        extra: Any additional expert-specific config keys that the concrete
            class may consume (e.g. ``temperature``, ``threshold``).
    """

    name: str
    class_path: str = Field(..., alias="class")
    weights: str | None = None
    modalities: list[str]
    # --- Remote-endpoint fields ---
    endpoint: str | None = None
    endpoint_headers: dict[str, str] = Field(default_factory=dict)
    endpoint_timeout: float = 10.0
    request_template: dict[str, Any] | None = None
    response_mapping: dict[str, str] | None = None
    endpoint_response_max_bytes: int | None = None
    integrity: "ExpertIntegrityConfig | None" = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @field_validator("modalities")
    @classmethod
    def modalities_must_be_non_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one modality is declared per expert."""
        if not v:
            raise ValueError("An expert must declare at least one modality.")
        return v

    @field_validator("endpoint_response_max_bytes")
    @classmethod
    def endpoint_response_max_bytes_must_be_positive(cls, v: int | None) -> int | None:
        """Validate optional per-expert remote response limit."""
        if v is not None and v < 1:
            raise ValueError("endpoint_response_max_bytes must be >= 1.")
        return v

    @model_validator(mode="after")
    def validate_weights_or_endpoint(self) -> "ExpertConfig":
        """Ensure exactly one of ``weights`` or ``endpoint`` is provided."""
        has_weights = self.weights is not None
        has_endpoint = self.endpoint is not None
        if has_weights and has_endpoint:
            raise ValueError(
                f"Expert '{self.name}': provide either 'weights' (local) or "
                f"'endpoint' (remote), not both."
            )
        if not has_weights and not has_endpoint:
            raise ValueError(
                f"Expert '{self.name}': one of 'weights' (local file path) or "
                f"'endpoint' (remote HTTP URL) is required."
            )
        return self


class ExpertIntegrityConfig(BaseModel):
    """Optional integrity policy for local or remote expert artifacts."""

    sha256: str | None = None
    manifest_url: str | None = None
    manifest_public_key: str | None = None
    manifest_required: bool = False
    signature_algorithm: str = "RSA-PSS-SHA256"

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_hex(cls, v: str | None) -> str | None:
        """Validate SHA-256 digest shape when configured."""
        if v is None:
            return v
        value = v.strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("sha256 must be a 64-character hex digest.")
        return value

    @field_validator("signature_algorithm")
    @classmethod
    def signature_algorithm_must_be_supported(cls, v: str) -> str:
        """Validate manifest signature algorithm."""
        if v != "RSA-PSS-SHA256":
            raise ValueError("Only RSA-PSS-SHA256 is supported.")
        return v


class RemoteRetryConfig(BaseModel):
    """Retry policy for remote expert inference calls."""

    max_attempts: int = 3
    initial_delay_s: float = 0.25
    max_delay_s: float = 2.0
    backoff_multiplier: float = 2.0
    jitter: bool = True

    @field_validator("max_attempts")
    @classmethod
    def max_attempts_must_be_positive(cls, v: int) -> int:
        """Require at least one attempt."""
        if v < 1:
            raise ValueError("remote_retry.max_attempts must be >= 1.")
        return v

    @field_validator("initial_delay_s", "max_delay_s", "backoff_multiplier")
    @classmethod
    def retry_numbers_must_be_positive(cls, v: float) -> float:
        """Require positive retry timing values."""
        if v <= 0:
            raise ValueError("remote_retry numeric values must be > 0.")
        return v

    @model_validator(mode="after")
    def validate_retry_delay_bounds(self) -> "RemoteRetryConfig":
        """Ensure the configured maximum delay can contain the initial delay."""
        if self.max_delay_s < self.initial_delay_s:
            raise ValueError("remote_retry.max_delay_s must be >= initial_delay_s.")
        return self


class RemoteCircuitBreakerConfig(BaseModel):
    """Circuit-breaker policy for remote expert inference calls."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0

    @field_validator("failure_threshold")
    @classmethod
    def failure_threshold_must_be_positive(cls, v: int) -> int:
        """Require at least one failure before opening a circuit."""
        if v < 1:
            raise ValueError("remote_circuit_breaker.failure_threshold must be >= 1.")
        return v

    @field_validator("recovery_timeout_s")
    @classmethod
    def recovery_timeout_must_be_positive(cls, v: float) -> float:
        """Require a positive recovery timeout."""
        if v <= 0:
            raise ValueError("remote_circuit_breaker.recovery_timeout_s must be > 0.")
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
        authentication_enabled: Whether the stateless authentication
            middleware is enabled. Defaults to fail-closed ``True``.
        authorization_enabled: Whether the authorization middleware is enabled.
            Defaults to fail-closed ``True``.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int | None = None
    log_level: str = "info"
    authentication_enabled: bool = True
    authorization_enabled: bool = True
    token_invalidation_store: str = "memory"
    token_invalidation_redis_url: str | None = None
    token_invalidation_key_prefix: str = "apmoe:jwt:invalid:"
    rate_limit_store: str = "memory"
    rate_limit_redis_url: str | None = None
    rate_limit_key_prefix: str = "apmoe:rate:"

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

    @field_validator("token_invalidation_store", "rate_limit_store")
    @classmethod
    def store_backend_must_be_supported(cls, v: str) -> str:
        """Validate shared-store backend names."""
        normalised = v.strip().lower()
        if normalised not in {"memory", "redis"}:
            raise ValueError(
                f"Store backend must be one of 'memory' or 'redis', got {v!r}."
            )
        return normalised

    @model_validator(mode="after")
    def validate_shared_store_urls(self) -> "ServingConfig":
        """Require Redis URLs when Redis-backed stores are selected."""
        if self.token_invalidation_store == "redis" and not self.token_invalidation_redis_url:
            raise ValueError(
                "serving.token_invalidation_redis_url is required when "
                "serving.token_invalidation_store='redis'."
            )
        if self.rate_limit_store == "redis" and not self.rate_limit_redis_url:
            raise ValueError(
                "serving.rate_limit_redis_url is required when "
                "serving.rate_limit_store='redis'."
            )
        return self


class SecurityConfig(BaseModel):
    """Framework-level security hardening settings."""

    remote_endpoint_allowlist: list[str] | None = None
    remote_enforce_https: bool = True
    remote_allow_private_networks: bool = False
    remote_response_max_bytes: int = 1_048_576
    audit_enabled: bool = True
    audit_success_events: bool = True

    @field_validator("remote_response_max_bytes")
    @classmethod
    def remote_response_max_bytes_must_be_positive(cls, v: int) -> int:
        """Validate remote response byte limit."""
        if v < 1:
            raise ValueError("remote_response_max_bytes must be >= 1.")
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
    environment: str = "development"
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    confidence_threshold: float | None = None
    expert_failure_policy: Literal["fail_fast", "skip_failed"] = "fail_fast"
    remote_retry: RemoteRetryConfig = Field(default_factory=RemoteRetryConfig)
    remote_circuit_breaker: RemoteCircuitBreakerConfig = Field(
        default_factory=RemoteCircuitBreakerConfig
    )

    @field_validator("environment")
    @classmethod
    def environment_must_be_supported(cls, v: str) -> str:
        """Validate environment mode."""
        normalised = v.strip().lower()
        if normalised not in {"development", "test", "staging", "production"}:
            raise ValueError(
                "environment must be one of: development, test, staging, production."
            )
        return normalised

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

    @model_validator(mode="after")
    def validate_production_remote_allowlist(self) -> "APMoEConfig":
        """Require explicit remote endpoint allowlist in production."""
        has_remote = any(expert.endpoint is not None for expert in self.experts)
        allowlist = self.security.remote_endpoint_allowlist
        unsafe = allowlist is None or not allowlist or "*" in allowlist
        if self.environment == "production" and has_remote and unsafe:
            raise ValueError(
                "Production remote experts require an explicit "
                "apmoe.security.remote_endpoint_allowlist without '*'."
            )
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
    "APMOE_SERVING_TOKEN_INVALIDATION_STORE": ("token_invalidation_store", str),
    "APMOE_SERVING_TOKEN_INVALIDATION_REDIS_URL": (
        "token_invalidation_redis_url",
        str,
    ),
    "APMOE_SERVING_TOKEN_INVALIDATION_KEY_PREFIX": (
        "token_invalidation_key_prefix",
        str,
    ),
    "APMOE_SERVING_RATE_LIMIT_STORE": ("rate_limit_store", str),
    "APMOE_SERVING_RATE_LIMIT_REDIS_URL": ("rate_limit_redis_url", str),
    "APMOE_SERVING_RATE_LIMIT_KEY_PREFIX": ("rate_limit_key_prefix", str),
}

_SERVING_BOOL_ENV_MAP: dict[str, str] = {
    "APMOE_SERVING_AUTHENTICATION_ENABLED": "authentication_enabled",
    "APMOE_SERVING_AUTHORIZATION_ENABLED": "authorization_enabled",
}


def _parse_bool_env(env_key: str, raw: str) -> bool:
    """Parse a boolean environment variable value.

    Accepts common deployment spellings so Docker, shell, and CI users do not
    have to remember one exact representation.
    """
    normalised = raw.strip().lower()
    if normalised in {"true", "1", "yes", "on"}:
        return True
    if normalised in {"false", "0", "no", "off"}:
        return False
    raise ConfigurationError(
        f"Environment variable {env_key}={raw!r} cannot be cast to bool. "
        "Use one of: true/false, 1/0, yes/no, on/off.",
        context={"env_key": env_key, "raw_value": raw},
    )


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Mutate *data* in-place with ``APMOE_``-prefixed environment variable overrides.

    Currently supports overriding the ``serving`` block.  Future phases may
    extend this to ``aggregation`` and other sections.

    Args:
        data: The raw deserialized JSON dict (will be mutated).
    """
    apmoe_section: dict[str, Any] = data.setdefault("apmoe", {})
    serving_section: dict[str, Any] = apmoe_section.setdefault("serving", {})

    env_mode = os.environ.get("APMOE_ENV")
    if env_mode is not None:
        apmoe_section["environment"] = env_mode

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

    for env_key, field_name in _SERVING_BOOL_ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is not None:
            serving_section[field_name] = _parse_bool_env(env_key, raw)

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
