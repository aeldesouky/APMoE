"""Remote expert plugin — delegates inference to an external HTTP endpoint.

Instead of loading a local weight file, :class:`RemoteExpert` serialises the
processed modality data and POSTs it to a user-configured URL.  This lets any
APMoE pipeline call an external model provider (HuggingFace Inference API,
OpenAI-compatible servers, other APMoE instances, custom FastAPI endpoints,
etc.) as a first-class expert.

Configuration
-------------
Set ``"class": "apmoe.experts.remote.RemoteExpert"`` and provide an
``"endpoint"`` URL instead of a ``"weights"`` path in your ``config.json``:

.. code-block:: json

    {
      "name": "keystroke_remote_expert",
      "class": "apmoe.experts.remote.RemoteExpert",
      "modalities": ["keystroke"],
      "endpoint": "https://api-inference.huggingface.co/models/my-org/age-model",
      "endpoint_headers": {"Authorization": "Bearer $HF_TOKEN"},
      "endpoint_timeout": 15.0,
      "request_template": {"inputs": "{{modalities.keystroke}}"},
      "response_mapping": {"predicted_age": "[0].age", "confidence": "[0].score"}
    }

Request template
----------------
``request_template`` is a nested JSON-serialisable dict.  Any *leaf string*
that is exactly a placeholder expression is replaced at predict-time:

* ``"{{modalities.<name>}}"`` — replaced with the serialised value of that
  modality's :class:`~apmoe.core.types.ProcessedInput`.
* ``"{{expert_name}}"`` — replaced with the expert's configured ``name``.

When ``request_template`` is ``None`` (the default) the framework sends:

.. code-block:: json

    {"expert_name": "...", "modalities": {"<name>": ...}}

Response mapping
----------------
``response_mapping`` maps :class:`~apmoe.core.types.ExpertOutput` field names
to dot-paths (or array-index paths) within the remote JSON response:

* ``"predicted_age"`` — **required** key.
* ``"confidence"`` — optional; defaults to ``-1.0`` when absent.
* ``"metadata"`` — optional; defaults to ``{}`` when absent.

Path syntax supports simple dot-notation (``"result.age"``) and leading
array-index notation (``"[0].age"``).  When ``response_mapping`` is ``None``
the framework expects the response to be a flat object with the keys
``predicted_age``, ``confidence``, and ``metadata`` at the top level.

Environment variable substitution
----------------------------------
``$VAR`` references are expanded from environment variables **at bootstrap
time** (in :meth:`RemoteExpert.load_weights`).  Expansion applies to:

* Every value in ``endpoint_headers`` (already documented).
* The ``endpoint`` URL itself — use ``$LLM_ENDPOINT`` to keep the server
  address out of the config file.
* Every **literal** string leaf in ``request_template`` that does not contain
  a ``{{...}}`` predict-time placeholder — e.g. ``"$LLM_MODEL"`` for the
  model name or ``"$LLM_SYSTEM_PROMPT"`` for the system prompt.

Both bare ``"$VAR"`` and inline ``"Bearer $TOKEN"`` patterns are supported.
A missing environment variable raises :class:`~apmoe.core.exceptions.ExpertError`
with a clear message so misconfigurations are caught at startup, not mid-request.

Dependencies
------------
``httpx`` must be installed::

    pip install httpx
    # or with the framework extras:
    pip install apmoe[remote]
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from apmoe.core.exceptions import ExpertError
from apmoe.core.security import (
    emit_security_audit,
    redact_url,
    redact_value,
    validate_remote_url,
)
from apmoe.core.types import EmbeddingResult, ExpertOutput, ProcessedInput
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry

logger = logging.getLogger("apmoe.experts.remote")

_TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


@dataclass(frozen=True)
class _RemoteRetryPolicy:
    """Resolved retry settings for one remote expert."""

    max_attempts: int = 3
    initial_delay_s: float = 0.25
    max_delay_s: float = 2.0
    backoff_multiplier: float = 2.0
    jitter: bool = True

    @classmethod
    def from_config(cls, config: Any | None) -> "_RemoteRetryPolicy":
        """Build a retry policy from a Pydantic config object or defaults."""
        if config is None:
            return cls()
        return cls(
            max_attempts=int(getattr(config, "max_attempts", cls.max_attempts)),
            initial_delay_s=float(
                getattr(config, "initial_delay_s", cls.initial_delay_s)
            ),
            max_delay_s=float(getattr(config, "max_delay_s", cls.max_delay_s)),
            backoff_multiplier=float(
                getattr(config, "backoff_multiplier", cls.backoff_multiplier)
            ),
            jitter=bool(getattr(config, "jitter", cls.jitter)),
        )


@dataclass(frozen=True)
class _CircuitBreakerPolicy:
    """Resolved circuit-breaker settings for one remote expert."""

    enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0

    @classmethod
    def from_config(cls, config: Any | None) -> "_CircuitBreakerPolicy":
        """Build a circuit-breaker policy from a Pydantic config object or defaults."""
        if config is None:
            return cls()
        return cls(
            enabled=bool(getattr(config, "enabled", cls.enabled)),
            failure_threshold=int(
                getattr(config, "failure_threshold", cls.failure_threshold)
            ),
            recovery_timeout_s=float(
                getattr(config, "recovery_timeout_s", cls.recovery_timeout_s)
            ),
        )


class _CircuitBreaker:
    """Small in-memory circuit breaker for a single remote expert instance."""

    def __init__(self, policy: _CircuitBreakerPolicy) -> None:
        self._policy = policy
        self._state = "closed"
        self._failure_count = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> str:
        """Return the current circuit state."""
        if (
            self._policy.enabled
            and self._state == "open"
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self._policy.recovery_timeout_s
        ):
            self._state = "half_open"
        return self._state

    def before_call(self) -> None:
        """Raise if the circuit is open and not ready for a trial call."""
        if not self._policy.enabled:
            return
        if self.state == "open":
            raise ExpertError(
                "remote circuit breaker is open",
                context={"circuit_state": "open"},
            )

    def record_success(self) -> None:
        """Close the circuit after a successful call."""
        if not self._policy.enabled:
            return
        self._state = "closed"
        self._failure_count = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call and open the circuit when the threshold is reached."""
        if not self._policy.enabled:
            return
        if self.state == "half_open":
            self._open()
            return
        self._failure_count += 1
        if self._failure_count >= self._policy.failure_threshold:
            self._open()

    def _open(self) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Helper: serialise a ProcessedInput to a JSON-safe value
# ---------------------------------------------------------------------------


def _serialise_input(processed: ProcessedInput) -> Any:
    """Convert a :class:`~apmoe.core.types.ProcessedInput` to a JSON-safe value.

    * :class:`~apmoe.core.types.ModalityData` — returns ``data`` directly when
      it is already JSON-serialisable (dict, list, str, int, float); otherwise
      calls ``str(data)`` as a best-effort fallback.
    * :class:`~apmoe.core.types.EmbeddingResult` — returns the embedding as a
      Python list of floats (suitable for JSON).

    Args:
        processed: The output of a modality's processing chain.

    Returns:
        A JSON-serialisable Python object.
    """
    if isinstance(processed, EmbeddingResult):
        try:
            return processed.embedding.tolist()
        except AttributeError:
            return list(processed.embedding)

    # ModalityData
    data = processed.data
    if isinstance(data, (dict, list, str, int, float, bool)) or data is None:
        return data
    # numpy / torch / arbitrary — attempt tolist(), else str
    try:
        return data.tolist()  # type: ignore[union-attr]
    except AttributeError:
        return str(data)


# ---------------------------------------------------------------------------
# Helper: resolve a dot/index path within a parsed JSON value
# ---------------------------------------------------------------------------


def _resolve_path(obj: Any, path: str) -> Any:
    """Navigate a parsed JSON object using a simple dot/index path.

    Syntax:

    * ``"predicted_age"`` — top-level key.
    * ``"result.age"`` — nested key.
    * ``"[0].age"`` — first element of a list, then key ``age``.
    * ``"[0]"`` — first element of a list.

    Args:
        obj: The parsed JSON value (dict, list, or scalar).
        path: Dot-path string as described above.

    Returns:
        The value at the specified path.

    Raises:
        KeyError / IndexError / TypeError: If the path does not match the
            structure of *obj*.
    """
    current = obj
    parts = _split_path(path)
    for part in parts:
        if part.startswith("[") and part.endswith("]"):
            idx = int(part[1:-1])
            current = current[idx]
        else:
            current = current[part]
    return current


def _split_path(path: str) -> list[str]:
    """Split a dot-path string into individual navigation tokens.

    Handles leading ``[n]`` array-index segments and nested dot-notation.

    Args:
        path: A path string such as ``"[0].age"`` or ``"result.score"``.

    Returns:
        List of string tokens, e.g. ``["[0]", "age"]`` or
        ``["result", "score"]``.
    """
    tokens: list[str] = []
    remaining = path.strip()
    while remaining:
        if remaining.startswith("["):
            close = remaining.index("]")
            tokens.append(remaining[: close + 1])
            remaining = remaining[close + 1 :].lstrip(".")
        else:
            dot = remaining.find(".")
            if dot == -1:
                tokens.append(remaining)
                break
            tokens.append(remaining[:dot])
            remaining = remaining[dot + 1 :]
    return tokens


# ---------------------------------------------------------------------------
# Helper: apply request_template placeholders
# ---------------------------------------------------------------------------

_PLACEHOLDER_PREFIX = "{{"
_PLACEHOLDER_SUFFIX = "}}"


def _apply_template(
    template: Any,
    context: dict[str, Any],
) -> Any:
    """Recursively substitute placeholder expressions in *template*.

    Traverses dicts and lists recursively.  For each leaf string value that
    *exactly* matches a ``{{...}}`` expression, replaces it with the
    corresponding value from *context*.

    Supported placeholders:

    * ``"{{expert_name}}"`` → ``context["expert_name"]`` (str)
    * ``"{{modalities.<name>}}"`` → ``context["modalities"][name]`` (any)

    Args:
        template: The template value (dict, list, or scalar).
        context: A dict containing ``"expert_name"`` (str) and
            ``"modalities"`` (dict[str, Any]) keys.

    Returns:
        The template with all recognised placeholders substituted.

    Raises:
        :class:`~apmoe.core.exceptions.ExpertError`: If a placeholder
            references an unknown modality or an unrecognised expression.
    """
    if isinstance(template, dict):
        return {k: _apply_template(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [_apply_template(item, context) for item in template]
    if isinstance(template, str):
        s = template.strip()
        if s.startswith(_PLACEHOLDER_PREFIX) and s.endswith(_PLACEHOLDER_SUFFIX):
            expr = s[len(_PLACEHOLDER_PREFIX) : -len(_PLACEHOLDER_SUFFIX)].strip()
            return _resolve_placeholder(expr, context)
    return template


def _resolve_placeholder(expr: str, context: dict[str, Any]) -> Any:
    """Resolve a single placeholder expression such as ``modalities.keystroke``.

    Args:
        expr: The inner expression string (without ``{{`` / ``}}``).
        context: Template context dict.

    Returns:
        Resolved value.

    Raises:
        :class:`~apmoe.core.exceptions.ExpertError`: If the expression is
            unrecognised or the modality is not available.
    """
    if expr == "expert_name":
        return context["expert_name"]

    if expr.startswith("modalities."):
        modality_name = expr[len("modalities."):]
        modalities: dict[str, Any] = context.get("modalities", {})
        if modality_name not in modalities:
            available = sorted(modalities.keys())
            raise ExpertError(
                f"Template placeholder '{{{{ {expr} }}}}' references modality "
                f"'{modality_name}' which is not available in this request.  "
                f"Available modalities: {available}.",
                context={"expr": expr, "available": available},
            )
        return modalities[modality_name]

    raise ExpertError(
        f"Unrecognised template placeholder expression: '{expr}'.  "
        f"Supported forms: '{{{{expert_name}}}}', '{{{{modalities.<name>}}}}'.",
        context={"expr": expr},
    )


def _expand_str(value: str, expert_name: str, field: str) -> str:
    """Expand all ``$VARNAME`` occurrences in *value* from environment variables.

    Supports both bare ``"$TOKEN"`` and inline ``"Bearer $TOKEN"`` patterns.

    Args:
        value: The raw string that may contain ``$VAR`` references.
        expert_name: Used for error messages only.
        field: Human-readable field name for error messages (e.g. ``"endpoint"``).

    Returns:
        The string with all ``$VAR`` occurrences replaced by their env values.

    Raises:
        :class:`~apmoe.core.exceptions.ExpertError`: If a referenced env var
            is not set.
    """
    import re

    _ENV_VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

    def _replace(match: re.Match[str]) -> str:  # type: ignore[type-arg]
        env_var = match.group(1)
        env_value = os.environ.get(env_var)
        if env_value is None:
            raise ExpertError(
                f"Expert '{expert_name}': {field} references environment variable "
                f"'{env_var}' which is not set.  "
                f"Set it before starting the server (e.g. export {env_var}=...).",
                context={"expert": expert_name, "field": field, "env_var": env_var},
            )
        return env_value

    return _ENV_VAR_RE.sub(_replace, value)


def _expand_headers(raw_headers: dict[str, str], expert_name: str) -> dict[str, str]:
    """Expand ``$VAR`` references in all header values from environment variables.

    Delegates per-value expansion to :func:`_expand_str`.

    Args:
        raw_headers: Header dict from config (values may contain ``$VAR`` refs).
        expert_name: Used for error messages only.

    Returns:
        A new dict with all ``$VAR`` occurrences replaced by actual env values.

    Raises:
        :class:`~apmoe.core.exceptions.ExpertError`: If a referenced env var
            is not set.
    """
    return {
        key: _expand_str(value, expert_name, f"header '{key}'")
        for key, value in raw_headers.items()
    }


def _expand_template(template: Any, expert_name: str) -> Any:
    """Recursively expand ``$VAR`` references in *non-placeholder* string leaves.

    Walks the template tree identically to :func:`_apply_template`.  For each
    leaf string that does **not** contain a ``{{...}}`` predict-time placeholder,
    expands any ``$VAR`` occurrences from environment variables at bootstrap time.
    Strings that are (or contain) ``{{...}}`` expressions are left unchanged so
    that predict-time substitution can proceed normally.

    Args:
        template: The raw template value as parsed from config.
        expert_name: Used for error messages only.

    Returns:
        The template with ``$VAR`` literals expanded; ``{{...}}`` placeholders
        are preserved verbatim for predict-time substitution.
    """
    if isinstance(template, dict):
        return {k: _expand_template(v, expert_name) for k, v in template.items()}
    if isinstance(template, list):
        return [_expand_template(item, expert_name) for item in template]
    if isinstance(template, str):
        # Leave predict-time placeholders intact
        if "{{" in template and "}}" in template:
            return template
        return _expand_str(template, expert_name, "request_template value")
    return template



# ---------------------------------------------------------------------------
# RemoteExpert
# ---------------------------------------------------------------------------


@expert_registry.register("apmoe.experts.remote.RemoteExpert")
class RemoteExpert(ExpertPlugin):
    """Expert that delegates inference to an external HTTP endpoint.

    POSTs serialised modality data to a remote URL and maps the JSON response
    back to an :class:`~apmoe.core.types.ExpertOutput`.  The request body and
    response extraction are fully customisable via ``request_template`` and
    ``response_mapping`` (see module docstring).

    Do **not** construct this class directly.  Use
    :meth:`~apmoe.core.app.APMoEApp.from_config` with the ``"endpoint"``
    field in your ``config.json``.
    """

    def __init__(
        self,
        *,
        expert_name: str,
        modalities: list[str],
        endpoint: str,
        endpoint_headers: dict[str, str] | None = None,
        endpoint_timeout: float = 10.0,
        request_template: dict[str, Any] | None = None,
        response_mapping: dict[str, str] | None = None,
        endpoint_response_max_bytes: int | None = None,
        security_config: Any | None = None,
        environment: str = "development",
        retry_config: Any | None = None,
        circuit_breaker_config: Any | None = None,
    ) -> None:
        """Initialise the remote expert.

        Args:
            expert_name: The ``name`` from the config (used in logs and
                placeholder substitution).
            modalities: The modality names this expert consumes (from config).
            endpoint: Full HTTP/HTTPS URL to POST to.
            endpoint_headers: Raw HTTP headers (``$VAR`` values are expanded
                from environment variables in :meth:`load_weights`).
            endpoint_timeout: Read timeout in seconds.
            request_template: Optional body template dict (see module docs).
            response_mapping: Optional response dot-path mapping (see module docs).
        """
        self._expert_name = expert_name
        self._modalities = modalities
        self._endpoint = endpoint
        self._raw_headers: dict[str, str] = endpoint_headers or {}
        self._timeout = endpoint_timeout
        self._request_template: dict[str, Any] | None = request_template
        self._response_mapping: dict[str, str] | None = response_mapping
        self._response_max_bytes = endpoint_response_max_bytes
        self._security_config = security_config
        self._environment = environment
        self._retry_policy = _RemoteRetryPolicy.from_config(retry_config)
        self._circuit_breaker = _CircuitBreaker(
            _CircuitBreakerPolicy.from_config(circuit_breaker_config)
        )
        self._headers: dict[str, str] = {}  # populated in load_weights
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # ExpertPlugin interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the configured expert name."""
        return self._expert_name

    def declared_modalities(self) -> list[str]:
        """Return the modality list from config."""
        return list(self._modalities)

    @property
    def endpoint(self) -> str:
        """Return the resolved remote endpoint URL."""
        return self._endpoint

    def load_weights(self, path: str) -> None:  # noqa: ARG002
        """Validate connectivity prerequisites and expand env-var references.

        No weight file is loaded for remote experts.  This method:

        1. Checks ``httpx`` is installed (raises a helpful error if not).
        2. Expands ``$ENV_VAR`` in ``endpoint_headers`` values.
        3. Expands ``$ENV_VAR`` in the ``endpoint`` URL itself.
        4. Expands ``$ENV_VAR`` in non-placeholder string leaves of
           ``request_template`` (strings containing ``{{...}}`` are left
           intact for predict-time substitution).

        Args:
            path: Ignored for remote experts (should be ``None`` / empty).
        """
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise ExpertError(
                "httpx is required for RemoteExpert.  "
                "Install it with:  pip install httpx  "
                "(or:  pip install apmoe[remote])",
                context={"expert": self._expert_name},
            ) from exc

        # Expand env vars in headers
        self._headers = _expand_headers(self._raw_headers, self._expert_name)

        # Expand env vars in the endpoint URL
        self._endpoint = _expand_str(self._endpoint, self._expert_name, "endpoint")
        self._validate_endpoint_policy()

        # Expand env vars in non-placeholder template values
        if self._request_template is not None:
            self._request_template = _expand_template(
                self._request_template, self._expert_name
            )

        self._loaded = True
        logger.debug(
            "RemoteExpert '%s' initialised — endpoint: %s",
            self._expert_name,
            redact_url(self._endpoint),
        )

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Serialise inputs, POST to the remote endpoint, map the response.

        Args:
            inputs: Dict of modality name → processed input (as provided by
                the framework pipeline).

        Returns:
            An :class:`~apmoe.core.types.ExpertOutput` populated from the
            remote response.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If ``load_weights``
                was not called, the HTTP request fails, the response is not
                valid JSON, or a required field cannot be extracted.
        """
        if not self._loaded:
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': call load_weights() before predict().",
                context={"expert": self._expert_name},
            )

        import httpx  # local import - availability checked in load_weights

        # --- Serialise modality inputs -----------------------------------
        serialised: dict[str, Any] = {
            mod: _serialise_input(processed) for mod, processed in inputs.items()
        }

        # --- Build request body ------------------------------------------
        body = self._build_request_body(serialised)

        # --- POST to remote endpoint -------------------------------------
        logger.debug(
            "RemoteExpert '%s' POST %s",
            self._expert_name,
            redact_url(self._endpoint),
        )
        try:
            response = self._post_with_resilience(httpx, body)
            data = self._parse_limited_json_response(response)
            self._circuit_breaker.record_success()
        except ExpertError:
            raise
        except Exception as exc:
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': response from {redact_url(self._endpoint)} "
                f"is not valid JSON: {exc}",
                context={"expert": self._expert_name, "endpoint": redact_url(self._endpoint)},
            ) from exc

        emit_security_audit(
            "remote_call",
            "success",
            expert_name=self._expert_name,
            endpoint_host=self._endpoint_host(),
        )

        return self._parse_response(data, list(inputs.keys()))

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` after :meth:`load_weights` has been called successfully."""
        return self._loaded

    def get_info(self) -> dict[str, object]:
        """Return diagnostic metadata for ``GET /info``."""
        return {
            "name": self.name,
            "modalities": self.declared_modalities(),
            "expert_class": type(self).__qualname__,
            "backend": "remote",
            "endpoint": redact_url(self._endpoint),
            "has_request_template": self._request_template is not None,
            "has_response_mapping": self._response_mapping is not None,
            "retry": {
                "max_attempts": self._retry_policy.max_attempts,
                "initial_delay_s": self._retry_policy.initial_delay_s,
                "max_delay_s": self._retry_policy.max_delay_s,
                "backoff_multiplier": self._retry_policy.backoff_multiplier,
                "jitter": self._retry_policy.jitter,
            },
            "circuit_state": self._circuit_breaker.state,
            "loaded": self.is_loaded,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_request_body(self, serialised: dict[str, Any]) -> dict[str, Any]:
        """Construct the HTTP request body.

        If a ``request_template`` was provided, applies placeholder
        substitution.  Otherwise returns the default APMoE schema.

        Args:
            serialised: Dict of modality name → JSON-safe value.

        Returns:
            The request body dict ready for ``json=`` in ``httpx.post``.
        """
        context: dict[str, Any] = {
            "expert_name": self._expert_name,
            "modalities": serialised,
        }

        if self._request_template is not None:
            filled = _apply_template(copy.deepcopy(self._request_template), context)
            # Ensure the result is a dict (template root must be an object)
            if not isinstance(filled, dict):
                raise ExpertError(
                    f"RemoteExpert '{self._expert_name}': request_template root must be "
                    f"a JSON object (dict), got {type(filled).__name__}.",
                    context={"expert": self._expert_name},
                )
            return filled

        # Default schema
        return {"expert_name": self._expert_name, "modalities": serialised}

    def _post_with_resilience(self, httpx: Any, body: dict[str, Any]) -> Any:
        """POST to the configured endpoint with retry/backoff and circuit checks."""
        try:
            self._circuit_breaker.before_call()
        except ExpertError as exc:
            emit_security_audit(
                "remote_circuit_breaker",
                "failure",
                expert_name=self._expert_name,
                endpoint_host=self._endpoint_host(),
                reason="open",
            )
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': circuit breaker is open for "
                f"{redact_url(self._endpoint)}",
                context={
                    "expert": self._expert_name,
                    "endpoint": redact_url(self._endpoint),
                    "circuit_state": "open",
                },
            ) from exc

        last_exc: Exception | None = None
        attempts = self._retry_policy.max_attempts
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    self._endpoint,
                    json=body,
                    headers=self._headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response
            except httpx.TimeoutException as exc:
                last_exc = exc
                if not self._should_retry(attempt, attempts):
                    self._circuit_breaker.record_failure()
                    emit_security_audit(
                        "remote_call",
                        "failure",
                        expert_name=self._expert_name,
                        endpoint_host=self._endpoint_host(),
                        reason="timeout",
                        metadata={"attempts": attempt},
                    )
                    raise ExpertError(
                        f"RemoteExpert '{self._expert_name}': request timed out after "
                        f"{self._timeout}s - {redact_url(self._endpoint)}",
                        context={
                            "expert": self._expert_name,
                            "endpoint": redact_url(self._endpoint),
                            "attempts": attempt,
                        },
                    ) from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code not in _TRANSIENT_STATUS_CODES:
                    emit_security_audit(
                        "remote_call",
                        "failure",
                        expert_name=self._expert_name,
                        endpoint_host=self._endpoint_host(),
                        reason=f"http_{status_code}",
                        metadata={"attempts": attempt},
                    )
                    raise ExpertError(
                        f"RemoteExpert '{self._expert_name}': HTTP {status_code} "
                        f"from {redact_url(self._endpoint)}: "
                        f"{redact_value(exc.response.text[:200])}",
                        context={
                            "expert": self._expert_name,
                            "endpoint": redact_url(self._endpoint),
                            "status_code": status_code,
                            "attempts": attempt,
                        },
                    ) from exc
                last_exc = exc
                if not self._should_retry(attempt, attempts):
                    self._circuit_breaker.record_failure()
                    emit_security_audit(
                        "remote_call",
                        "failure",
                        expert_name=self._expert_name,
                        endpoint_host=self._endpoint_host(),
                        reason=f"http_{status_code}",
                        metadata={"attempts": attempt},
                    )
                    raise ExpertError(
                        f"RemoteExpert '{self._expert_name}': HTTP {status_code} "
                        f"from {redact_url(self._endpoint)}: "
                        f"{redact_value(exc.response.text[:200])}",
                        context={
                            "expert": self._expert_name,
                            "endpoint": redact_url(self._endpoint),
                            "status_code": status_code,
                            "attempts": attempt,
                        },
                    ) from exc
            except httpx.RequestError as exc:
                last_exc = exc
                if not self._should_retry(attempt, attempts):
                    self._circuit_breaker.record_failure()
                    emit_security_audit(
                        "remote_call",
                        "failure",
                        expert_name=self._expert_name,
                        endpoint_host=self._endpoint_host(),
                        reason="network_error",
                        metadata={"attempts": attempt},
                    )
                    raise ExpertError(
                        f"RemoteExpert '{self._expert_name}': network error - {exc}",
                        context={
                            "expert": self._expert_name,
                            "endpoint": redact_url(self._endpoint),
                            "attempts": attempt,
                        },
                    ) from exc

            self._sleep_before_retry(attempt)

        self._circuit_breaker.record_failure()
        raise ExpertError(
            f"RemoteExpert '{self._expert_name}': request failed after {attempts} attempts.",
            context={
                "expert": self._expert_name,
                "endpoint": redact_url(self._endpoint),
                "attempts": attempts,
                "last_error": str(last_exc) if last_exc else None,
            },
        )

    def _should_retry(self, attempt: int, max_attempts: int) -> bool:
        """Return whether another retry attempt remains."""
        return attempt < max_attempts

    def _sleep_before_retry(self, failed_attempt: int) -> None:
        """Sleep using exponential backoff and optional jitter."""
        base_delay = min(
            self._retry_policy.initial_delay_s
            * (self._retry_policy.backoff_multiplier ** (failed_attempt - 1)),
            self._retry_policy.max_delay_s,
        )
        delay = (
            random.uniform(0.0, base_delay)
            if self._retry_policy.jitter
            else base_delay
        )
        time.sleep(delay)

    def _parse_response(
        self, data: Any, consumed_modalities: list[str]
    ) -> ExpertOutput:
        """Extract an :class:`~apmoe.core.types.ExpertOutput` from the response.

        Uses ``response_mapping`` dot-paths when provided, otherwise reads
        top-level ``predicted_age``, ``confidence``, ``metadata`` keys.

        Args:
            data: Parsed JSON response (any type).
            consumed_modalities: Modality names that were in the request.

        Returns:
            A fully-populated :class:`~apmoe.core.types.ExpertOutput`.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If ``predicted_age``
                cannot be extracted or the value types are wrong.
        """
        mapping = self._response_mapping

        # --- predicted_age (required) ------------------------------------
        try:
            if mapping and "predicted_age" in mapping:
                raw_age = _resolve_path(data, mapping["predicted_age"])
            else:
                raw_age = data["predicted_age"]
            predicted_age = float(raw_age)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            path_hint = (mapping or {}).get("predicted_age", "predicted_age")
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': cannot extract 'predicted_age' "
                f"from response using path '{path_hint}': {exc}.  "
                f"Response (truncated): {json.dumps(redact_value(data), default=str)[:200]}",
                context={"expert": self._expert_name, "endpoint": redact_url(self._endpoint)},
            ) from exc

        # --- confidence (optional, default -1.0) -------------------------
        confidence: float = -1.0
        try:
            if mapping and "confidence" in mapping:
                confidence = float(_resolve_path(data, mapping["confidence"]))
            elif isinstance(data, dict) and "confidence" in data:
                confidence = float(data["confidence"])
        except (KeyError, IndexError, TypeError, ValueError):
            confidence = -1.0  # non-fatal — treat as unreported

        # --- metadata (optional, default {}) -----------------------------
        metadata: dict[str, Any] = {}
        try:
            if mapping and "metadata" in mapping:
                metadata = dict(_resolve_path(data, mapping["metadata"]))
            elif isinstance(data, dict) and "metadata" in data:
                metadata = dict(data["metadata"])
        except (KeyError, IndexError, TypeError, ValueError):
            metadata = {}  # non-fatal

        metadata["backend"] = "remote"
        metadata["endpoint"] = redact_url(self._endpoint)

        return ExpertOutput(
            expert_name=self._expert_name,
            consumed_modalities=consumed_modalities,
            predicted_age=predicted_age,
            confidence=confidence,
            metadata=metadata,
        )

    def _validate_endpoint_policy(self) -> None:
        """Validate the resolved endpoint against configured security policy."""
        if self._security_config is None:
            return
        allowlist = self._security_config.remote_endpoint_allowlist
        if allowlist is None and self._environment != "production":
            allowlist = ["*"]
        if not allowlist:
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': remote endpoint allowlist is empty."
            )
        try:
            validate_remote_url(
                self._endpoint,
                allowlist=allowlist,
                enforce_https=self._security_config.remote_enforce_https,
                allow_private_networks=self._security_config.remote_allow_private_networks,
                purpose=f"RemoteExpert '{self._expert_name}' endpoint",
            )
        except ExpertError as exc:
            emit_security_audit(
                "remote_endpoint_policy",
                "block",
                expert_name=self._expert_name,
                endpoint_host=self._endpoint_host(),
                reason=str(exc),
            )
            raise
        emit_security_audit(
            "remote_endpoint_policy",
            "allow",
            expert_name=self._expert_name,
            endpoint_host=self._endpoint_host(),
        )

    def _effective_response_max_bytes(self) -> int:
        """Return per-expert or global remote response byte limit."""
        if self._response_max_bytes is not None:
            return self._response_max_bytes
        if self._security_config is not None:
            return self._security_config.remote_response_max_bytes
        return 1_048_576

    def _parse_limited_json_response(self, response: Any) -> Any:
        """Parse a remote JSON response after enforcing content type and byte limit."""
        content_type = ""
        headers = getattr(response, "headers", {}) or {}
        if hasattr(headers, "get"):
            content_type = headers.get("content-type", "") or headers.get("Content-Type", "")
        if not isinstance(content_type, str):
            content_type = ""
        if content_type and "json" not in content_type.lower():
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': response Content-Type is not JSON: "
                f"{content_type!r}",
                context={"expert": self._expert_name, "content_type": content_type},
            )

        max_bytes = self._effective_response_max_bytes()
        raw = getattr(response, "content", None)
        if isinstance(raw, bytes):
            if len(raw) > max_bytes:
                emit_security_audit(
                    "remote_response_limit",
                    "failure",
                    expert_name=self._expert_name,
                    endpoint_host=self._endpoint_host(),
                    reason="response_too_large",
                    metadata={"max_bytes": max_bytes, "actual_bytes": len(raw)},
                )
                raise ExpertError(
                    f"RemoteExpert '{self._expert_name}': response exceeded {max_bytes} bytes.",
                    context={"expert": self._expert_name, "max_bytes": max_bytes},
                )
            return json.loads(raw.decode("utf-8"))

        data = response.json()
        approx_size = len(json.dumps(data, default=str).encode("utf-8"))
        if approx_size > max_bytes:
            emit_security_audit(
                "remote_response_limit",
                "failure",
                expert_name=self._expert_name,
                endpoint_host=self._endpoint_host(),
                reason="response_too_large",
                metadata={"max_bytes": max_bytes, "actual_bytes": approx_size},
            )
            raise ExpertError(
                f"RemoteExpert '{self._expert_name}': response exceeded {max_bytes} bytes.",
                context={"expert": self._expert_name, "max_bytes": max_bytes},
            )
        return data

    def _endpoint_host(self) -> str | None:
        """Return endpoint host for audit records."""
        from urllib.parse import urlparse

        return urlparse(self._endpoint).hostname
