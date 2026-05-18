"""Security helpers for serving, remote experts, integrity, and audit logs."""

from __future__ import annotations

import base64
import contextvars
import hashlib
import ipaddress
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from apmoe.core.exceptions import ExpertError

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "apmoe_correlation_id",
    default=None,
)

_SAFE_CORRELATION_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SENSITIVE_KEYS = ("authorization", "api-key", "apikey", "token", "secret", "cookie", "key")
_PRIVATE_HOSTS = {"localhost"}

logger = logging.getLogger("apmoe.security")
audit_logger = logging.getLogger("apmoe.security.audit")


def get_correlation_id() -> str | None:
    """Return the current global correlation id, if any."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the current global correlation id."""
    _correlation_id.set(correlation_id)


def ensure_correlation_id(candidate: str | None = None) -> str:
    """Use a safe inbound correlation id or generate a UUID4 value."""
    if candidate and _SAFE_CORRELATION_RE.fullmatch(candidate):
        correlation_id = candidate
    else:
        correlation_id = str(uuid.uuid4())
    set_correlation_id(correlation_id)
    return correlation_id


def canonical_json(data: dict[str, Any]) -> bytes:
    """Return canonical JSON bytes for signing or verification."""
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def redact_url(raw_url: str) -> str:
    """Redact credentials and sensitive query parameters from a URL."""
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return "<redacted-url>"
    netloc = parsed.hostname or ""
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    query = urlencode(
        [
            (key, "***" if _is_sensitive_key(key) else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, query, parsed.fragment))


def redact_headers(headers: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive HTTP header values."""
    return {
        key: "***" if _is_sensitive_key(key) else value
        for key, value in headers.items()
    }


def redact_value(value: Any) -> Any:
    """Redact common secret-bearing strings and containers."""
    if isinstance(value, str):
        if "://" in value:
            return redact_url(value)
        return value
    if isinstance(value, dict):
        return {
            key: "***" if _is_sensitive_key(str(key)) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEYS)


def host_matches_allowlist(host: str, allowlist: list[str]) -> bool:
    """Return whether a hostname matches exact or wildcard allowlist entries."""
    host = host.lower().strip(".")
    for pattern in allowlist:
        candidate = pattern.lower().strip()
        if candidate == "*":
            return True
        if candidate.startswith("*."):
            suffix = candidate[1:]
            if host.endswith(suffix) and host != candidate[2:]:
                return True
        elif host == candidate.strip("."):
            return True
    return False


def validate_remote_url(
    url: str,
    *,
    allowlist: list[str],
    enforce_https: bool,
    allow_private_networks: bool,
    purpose: str,
) -> None:
    """Validate a remote URL against allowlist, HTTPS, and private-network policy."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not parsed.scheme or not host:
        raise ExpertError(f"{purpose}: remote URL must include a scheme and host.")
    if enforce_https and parsed.scheme != "https":
        if not (allow_private_networks and parsed.scheme == "http" and _is_private_host(host)):
            raise ExpertError(f"{purpose}: remote URL must use HTTPS: {redact_url(url)}")
    if not allow_private_networks and _is_private_host(host):
        raise ExpertError(f"{purpose}: private or local remote host is not allowed: {host}")
    if not host_matches_allowlist(host, allowlist):
        raise ExpertError(
            f"{purpose}: host '{host}' is not in remote endpoint allowlist.",
            context={"host": host, "allowlist": allowlist},
        )


def _is_private_host(host: str) -> bool:
    if host in _PRIVATE_HOSTS or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or str(ip) == "169.254.169.254"
    )


def sha256_file(path: str | Path) -> str:
    """Stream a file and return its SHA-256 hex digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_local_sha256(path: str | Path, expected: str, *, expert_name: str) -> None:
    """Verify a local model artifact hash."""
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ExpertError(
            f"Expert '{expert_name}': local model SHA-256 mismatch.",
            context={"expert": expert_name, "expected": expected, "actual": actual},
        )


def verify_rsa_pss_sha256(public_key_pem: str, payload: bytes, signature_b64: str) -> None:
    """Verify an RSA-PSS SHA-256 signature."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as exc:
        raise ExpertError(
            "cryptography is required for RSA signed model manifests. "
            "Install it with: pip install apmoe[security]"
        ) from exc

    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    try:
        public_key.verify(
            base64.b64decode(signature_b64),
            payload,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ExpertError("Remote model manifest signature verification failed.") from exc


def verify_manifest_payload(
    manifest: dict[str, Any],
    *,
    public_key_pem: str,
    expert_name: str,
    endpoint_origin: str,
) -> None:
    """Verify signed remote model manifest content."""
    required = {
        "expert_name",
        "model_id",
        "model_version",
        "endpoint_origin",
        "issued_at",
        "expires_at",
        "signature",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ExpertError(
            f"Remote model manifest missing required field(s): {missing}",
            context={"expert": expert_name},
        )
    if "model_digest" not in manifest and "artifact_digest" not in manifest:
        raise ExpertError(
            "Remote model manifest must include model_digest or artifact_digest.",
            context={"expert": expert_name},
        )
    if manifest["expert_name"] != expert_name:
        raise ExpertError(
            "Remote model manifest expert_name does not match config.",
            context={"expected": expert_name, "actual": manifest["expert_name"]},
        )
    if str(manifest["endpoint_origin"]).rstrip("/") != endpoint_origin.rstrip("/"):
        raise ExpertError(
            "Remote model manifest endpoint_origin does not match configured endpoint.",
            context={"expected": endpoint_origin, "actual": manifest["endpoint_origin"]},
        )
    expires_at = datetime.fromisoformat(str(manifest["expires_at"]).replace("Z", "+00:00"))
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        raise ExpertError("Remote model manifest has expired.", context={"expert": expert_name})

    signed = {key: value for key, value in manifest.items() if key != "signature"}
    verify_rsa_pss_sha256(public_key_pem, canonical_json(signed), str(manifest["signature"]))


@dataclass(frozen=True)
class SecurityAuditEvent:
    """Structured security audit event."""

    event_type: str
    outcome: str
    correlation_id: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    subject: str | None = None
    path: str | None = None
    method: str | None = None
    client_ip: str | None = None
    expert_name: str | None = None
    endpoint_host: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a redacted JSON-serialisable event dict."""
        return redact_value(asdict(self))


class SecurityAuditSink:
    """Abstract-ish sink interface for security audit events."""

    def emit(self, event: SecurityAuditEvent) -> None:
        """Emit one audit event."""
        raise NotImplementedError


class LoggingSecurityAuditSink(SecurityAuditSink):
    """Default audit sink writing structured JSON logs."""

    def emit(self, event: SecurityAuditEvent) -> None:
        """Log a structured audit event."""
        audit_logger.info(json.dumps(event.to_dict(), sort_keys=True))


def emit_security_audit(
    event_type: str,
    outcome: str,
    *,
    sinks: list[Any] | None = None,
    **kwargs: Any,
) -> SecurityAuditEvent:
    """Create and emit a security audit event."""
    event = SecurityAuditEvent(
        event_type=event_type,
        outcome=outcome,
        correlation_id=kwargs.pop("correlation_id", None) or get_correlation_id(),
        **kwargs,
    )
    targets = sinks if sinks is not None else [LoggingSecurityAuditSink()]
    for sink in targets:
        if hasattr(sink, "emit"):
            sink.emit(event)
        else:
            sink(event)
    return event

