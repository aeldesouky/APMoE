"""APMoE framework exception hierarchy.

All framework-level exceptions derive from :class:`APMoEError` so callers can
catch the entire family with a single ``except APMoEError`` clause, or target
specific sub-types for granular handling.
"""

from __future__ import annotations


class APMoEError(Exception):
    """Base class for all APMoE framework errors.

    Attributes:
        message: Human-readable description of the error.
        context: Optional mapping of additional diagnostic key/value pairs.
    """

    def __init__(self, message: str, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class ConfigurationError(APMoEError):
    """Raised when the framework configuration is invalid or cannot be loaded.

    Typical causes:
    - Missing required fields in the JSON config file.
    - Malformed JSON.
    - Environment variable overrides that fail type coercion.
    - Schema validation failures from Pydantic.
    """


class RegistryError(APMoEError):
    """Raised for errors in the component registry.

    Typical causes:
    - Duplicate registration of a component name.
    - Lookup of an unregistered component.
    - Failure to resolve a dotted-path import string.
    """


class PipelineError(APMoEError):
    """Raised when the inference pipeline cannot complete execution.

    Typical causes:
    - An unrecoverable error in a processing step.
    - No experts available to run after modality filtering.
    - Aggregator receives an empty list of expert outputs.
    """


class ModalityError(APMoEError):
    """Raised for errors within a modality processing branch.

    Typical causes:
    - Invalid or corrupt raw input data for a modality.
    - A Cleaner, Anonymizer, or Embedder step raises an unhandled exception.
    - The ModalityProcessor reports that the input fails validation.
    """


class ExpertError(APMoEError):
    """Raised for errors originating within an expert plugin.

    Typical causes:
    - Weight file not found or corrupt.
    - Expert's ``predict`` method raises an unhandled exception.
    - Required modality data is missing from the dispatch dict.
    """


class ServingError(APMoEError):
    """Raised for errors in the HTTP serving layer.

    Typical causes:
    - FastAPI app cannot start (port in use, invalid config).
    - Middleware misconfiguration.
    - Request parsing failures in the /predict endpoint.
    """
