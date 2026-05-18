"""Middleware components for the APMoE HTTP serving layer."""

from __future__ import annotations

import json
import logging
import inspect
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from apmoe.core.security import emit_security_audit, ensure_correlation_id, redact_url

logger = logging.getLogger("apmoe.serving")


def _audit_sinks(request: Request) -> list[Any] | None:
    """Return app-configured audit sinks, honoring audit disablement."""
    return getattr(request.app.state, "security_audit_hooks", None)


def _audit_success_enabled(request: Request) -> bool:
    """Return whether successful security events should be emitted."""
    return bool(getattr(request.app.state, "audit_success_events", True))


# ---------------------------------------------------------------------------
# Stateless authentication / authorization primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthContext:
    """Authenticated principal information attached to ``request.state``."""

    subject: str
    scopes: frozenset[str]
    token_id: str
    expires_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenInvalidationStore(ABC):
    """Abstract token invalidation store."""

    @abstractmethod
    def is_invalid(self, token_id: str) -> bool:
        """Return ``True`` when *token_id* has been invalidated."""

    @abstractmethod
    def invalidate(self, token_id: str, expires_at: datetime) -> None:
        """Mark *token_id* invalid until *expires_at*."""


class InMemoryTokenInvalidationStore(TokenInvalidationStore):
    """Process-local invalidation store suitable for local/single-worker use."""

    def __init__(self) -> None:
        self._invalidated: dict[str, datetime] = {}

    def _prune(self, now: datetime | None = None) -> None:
        current = now or datetime.now(UTC)
        expired = [
            token_id
            for token_id, expires_at in self._invalidated.items()
            if expires_at <= current
        ]
        for token_id in expired:
            del self._invalidated[token_id]

    def is_invalid(self, token_id: str) -> bool:
        """Return whether a token id is currently invalidated."""
        self._prune()
        return token_id in self._invalidated

    def invalidate(self, token_id: str, expires_at: datetime) -> None:
        """Invalidate a token id until its token expiry time."""
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        self._prune()
        if expires_at > datetime.now(UTC):
            self._invalidated[token_id] = expires_at
            emit_security_audit(
                "token_invalidation",
                "success",
                metadata={"token_id": token_id, "expires_at": expires_at.isoformat()},
            )


class RedisTokenInvalidationStore(TokenInvalidationStore):
    """Redis-backed JWT invalidation store shared across workers/nodes."""

    def __init__(self, redis_url: str, *, key_prefix: str = "apmoe:jwt:invalid:") -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisTokenInvalidationStore requires the redis package. "
                "Install it with: pip install apmoe[redis]"
            ) from exc

        self._client = redis.Redis.from_url(redis_url)
        self._key_prefix = key_prefix

    def _key(self, token_id: str) -> str:
        return f"{self._key_prefix}{token_id}"

    def is_invalid(self, token_id: str) -> bool:
        """Return whether a token id has an active invalidation key."""
        return bool(self._client.exists(self._key(token_id)))

    def invalidate(self, token_id: str, expires_at: datetime) -> None:
        """Invalidate a token id in Redis until its token expiry time."""
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        ttl_seconds = int((expires_at - datetime.now(UTC)).total_seconds())
        if ttl_seconds > 0:
            self._client.setex(self._key(token_id), ttl_seconds, "1")
            emit_security_audit(
                "token_invalidation",
                "success",
                metadata={"token_id": token_id, "expires_at": expires_at.isoformat()},
            )


class StatelessAuthProvider(ABC):
    """Abstract request credential verifier."""

    @abstractmethod
    def authenticate(self, request: Request) -> AuthContext | None:
        """Return an :class:`AuthContext` for valid credentials, else ``None``."""


class AuthorizationPolicy(ABC):
    """Abstract authorization policy for authenticated requests."""

    @abstractmethod
    def authorize(self, context: AuthContext, request: Request) -> bool:
        """Return ``True`` if *context* may access *request*."""


class JWTBearerAuthProvider(StatelessAuthProvider):
    """JWT Bearer credential verifier with token invalidation support."""

    def __init__(
        self,
        *,
        signing_key: str | bytes,
        algorithms: Sequence[str] = ("HS256",),
        issuer: str | None = None,
        audience: str | Sequence[str] | None = None,
        subject_claim: str = "sub",
        token_id_claim: str = "jti",
        scopes_claim: str | None = None,
        invalidation_store: TokenInvalidationStore | None = None,
    ) -> None:
        self._signing_key = signing_key
        self._algorithms = list(algorithms)
        self._issuer = issuer
        self._audience = audience
        self._subject_claim = subject_claim
        self._token_id_claim = token_id_claim
        self._scopes_claim = scopes_claim
        self._uses_default_invalidation_store = invalidation_store is None
        self._invalidation_store = invalidation_store or InMemoryTokenInvalidationStore()

    def set_invalidation_store(
        self,
        invalidation_store: TokenInvalidationStore,
        *,
        override: bool = False,
    ) -> None:
        """Attach the serving-layer invalidation store to this provider."""
        if override or self._uses_default_invalidation_store:
            self._invalidation_store = invalidation_store
            self._uses_default_invalidation_store = False

    def authenticate(self, request: Request) -> AuthContext | None:
        """Authenticate a request using its Bearer token."""
        auth_header = request.headers.get("Authorization", "")
        scheme, _, token = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None

        try:
            import jwt
        except ImportError as exc:
            raise RuntimeError(
                "PyJWT is required for JWTBearerAuthProvider. "
                "Install it with: pip install apmoe[security]"
            ) from exc

        try:
            payload: dict[str, Any] = jwt.decode(
                token,
                self._signing_key,
                algorithms=self._algorithms,
                issuer=self._issuer,
                audience=self._audience,
                options={
                    "require": ["exp", self._subject_claim, self._token_id_claim],
                    "verify_signature": True,
                    "verify_exp": True,
                },
            )
        except jwt.PyJWTError:
            return None

        subject = str(payload.get(self._subject_claim, "")).strip()
        token_id = str(payload.get(self._token_id_claim, "")).strip()
        if not subject or not token_id:
            return None
        if self._invalidation_store.is_invalid(token_id):
            return None

        expires_at = self._expiry_from_claim(payload.get("exp"))
        if expires_at is None:
            return None

        return AuthContext(
            subject=subject,
            scopes=frozenset(self._extract_scopes(payload)),
            token_id=token_id,
            expires_at=expires_at,
            metadata={
                key: value
                for key, value in payload.items()
                if key not in {self._subject_claim, self._token_id_claim}
            },
        )

    def _extract_scopes(self, payload: Mapping[str, Any]) -> set[str]:
        claim_names = [self._scopes_claim] if self._scopes_claim else ["scope", "scopes"]
        for claim_name in claim_names:
            if claim_name is None or claim_name not in payload:
                continue
            raw = payload[claim_name]
            if isinstance(raw, str):
                return {scope for scope in raw.split() if scope}
            if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
                return {str(scope) for scope in raw if str(scope)}
        return set()

    def _expiry_from_claim(self, raw_exp: Any) -> datetime | None:
        try:
            exp = float(raw_exp)
        except (TypeError, ValueError):
            return None
        return datetime.fromtimestamp(exp, UTC)


class ScopeAuthorizationPolicy(AuthorizationPolicy):
    """Route-to-scope authorization policy used by the default serving layer."""

    DEFAULT_REQUIRED_SCOPES: Mapping[tuple[str, str], frozenset[str]] = {
        ("POST", "/predict"): frozenset({"predict"}),
        ("POST", "/v1/predict"): frozenset({"predict"}),
        ("GET", "/info"): frozenset({"info:read"}),
        ("GET", "/v1/info"): frozenset({"info:read"}),
    }

    def __init__(
        self,
        required_scopes: Mapping[tuple[str, str], set[str] | frozenset[str]] | None = None,
    ) -> None:
        self._required_scopes: dict[tuple[str, str], frozenset[str]] = {
            key: frozenset(value)
            for key, value in (required_scopes or self.DEFAULT_REQUIRED_SCOPES).items()
        }

    def authorize(self, context: AuthContext, request: Request) -> bool:
        """Authorize based on route-required scopes."""
        required = self._required_scopes.get((request.method.upper(), request.url.path))
        if not required:
            return True
        return required.issubset(context.scopes)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authenticate requests and attach ``request.state.auth_context``."""

    _DEFAULT_EXCLUDED: frozenset[str] = frozenset({"/health", "/v1/health"})

    def __init__(
        self,
        app: ASGIApp,
        auth_provider: StatelessAuthProvider,
        exclude_paths: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._auth_provider = auth_provider
        self._excluded = exclude_paths if exclude_paths is not None else self._DEFAULT_EXCLUDED

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Authenticate then forward the request."""
        if request.url.path in self._excluded:
            return await call_next(request)

        context = self._auth_provider.authenticate(request)
        if inspect.isawaitable(context):
            context = await context
        if context is None:
            emit_security_audit(
                "authentication",
                "failure",
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else None,
                reason="invalid_or_missing_credentials",
                sinks=_audit_sinks(request),
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized."},
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.auth_context = context
        if _audit_success_enabled(request):
            emit_security_audit(
                "authentication",
                "success",
                subject=context.subject,
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else None,
                sinks=_audit_sinks(request),
            )
        return await call_next(request)


class AuthorizationMiddleware(BaseHTTPMiddleware):
    """Authorize authenticated requests using a configured policy."""

    _DEFAULT_EXCLUDED: frozenset[str] = frozenset({"/health", "/v1/health"})

    def __init__(
        self,
        app: ASGIApp,
        authorization_policy: AuthorizationPolicy,
        exclude_paths: frozenset[str] | None = None,
    ) -> None:
        super().__init__(app)
        self._authorization_policy = authorization_policy
        self._excluded = exclude_paths if exclude_paths is not None else self._DEFAULT_EXCLUDED

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Authorize then forward the request."""
        if request.url.path in self._excluded:
            return await call_next(request)

        context = getattr(request.state, "auth_context", None)
        if not isinstance(context, AuthContext):
            emit_security_audit(
                "authorization",
                "failure",
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else None,
                reason="missing_auth_context",
                sinks=_audit_sinks(request),
            )
            return JSONResponse(status_code=403, content={"detail": "Forbidden."})
        if not self._authorization_policy.authorize(context, request):
            emit_security_audit(
                "authorization",
                "failure",
                subject=context.subject,
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else None,
                reason="insufficient_scope",
                sinks=_audit_sinks(request),
            )
            return JSONResponse(status_code=403, content={"detail": "Forbidden."})
        if _audit_success_enabled(request):
            emit_security_audit(
                "authorization",
                "success",
                subject=context.subject,
                path=request.url.path,
                method=request.method,
                client_ip=request.client.host if request.client else None,
                sinks=_audit_sinks(request),
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Request Logging
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request as a structured JSON record."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request, attach a correlation ID, and log the outcome."""
        correlation_id = ensure_correlation_id(request.headers.get("X-Correlation-ID"))
        request.state.correlation_id = correlation_id
        started_at = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.error(
                "[%s] 500 Unhandled exception in %s %s after %.2f ms - %s: %s",
                correlation_id,
                request.method,
                request.url.path,
                duration_ms,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        status_code = response.status_code
        redacted_url = redact_url(str(request.url))
        redacted_query = ""
        if "?" in redacted_url:
            redacted_query = redacted_url.split("?", 1)[1].split("#", 1)[0]
        record: dict[str, object] = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query": redacted_query or None,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "client_host": request.client.host if request.client else None,
        }

        if status_code >= 500:
            logger.error(
                "[%s] %d %s %s - %.2f ms",
                correlation_id,
                status_code,
                request.method,
                request.url.path,
                duration_ms,
            )
        elif status_code >= 400:
            logger.warning(
                "[%s] %d %s %s - %.2f ms",
                correlation_id,
                status_code,
                request.method,
                request.url.path,
                duration_ms,
            )
        else:
            logger.info(json.dumps(record))

        response.headers["X-Correlation-ID"] = correlation_id
        return response


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------


class RateLimitStore(ABC):
    """Abstract store used by request rate limiting."""

    @abstractmethod
    def allow_request(self, key: str, *, limit: int, window_seconds: int) -> bool:
        """Record one request and return whether it is within the limit."""


class InMemoryRateLimitStore(RateLimitStore):
    """Process-local sliding-window rate-limit store."""

    def __init__(self) -> None:
        self._request_log: dict[str, list[float]] = defaultdict(list)

    def allow_request(self, key: str, *, limit: int, window_seconds: int) -> bool:
        """Record one request in the local sliding window."""
        now = time.monotonic()
        cutoff = now - window_seconds
        self._request_log[key] = [t for t in self._request_log[key] if t > cutoff]
        if len(self._request_log[key]) >= limit:
            return False
        self._request_log[key].append(now)
        return True


class RedisRateLimitStore(RateLimitStore):
    """Redis-backed sliding-window rate-limit store shared across workers/nodes."""

    def __init__(self, redis_url: str, *, key_prefix: str = "apmoe:rate:") -> None:
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "RedisRateLimitStore requires the redis package. "
                "Install it with: pip install apmoe[redis]"
            ) from exc

        self._client = redis.Redis.from_url(redis_url)
        self._key_prefix = key_prefix

    def _key(self, key: str) -> str:
        return f"{self._key_prefix}{key}"

    def allow_request(self, key: str, *, limit: int, window_seconds: int) -> bool:
        """Record one request in a Redis sorted-set sliding window."""
        now = time.time()
        redis_key = self._key(key)
        member = f"{now}:{uuid.uuid4()}"
        pipe = self._client.pipeline()
        pipe.zremrangebyscore(redis_key, 0, now - window_seconds)
        pipe.zadd(redis_key, {member: now})
        pipe.zcard(redis_key)
        pipe.expire(redis_key, window_seconds + 1)
        results = pipe.execute()
        return int(results[2]) <= limit


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter keyed by client IP address."""

    def __init__(
        self,
        app: ASGIApp,
        max_requests_per_minute: int,
        store: RateLimitStore | None = None,
    ) -> None:
        """Initialise the rate limiter."""
        super().__init__(app)
        self._limit: int = max_requests_per_minute
        self._window_seconds = 60
        self._store = store or InMemoryRateLimitStore()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Enforce the rate limit before forwarding to the next ASGI layer."""
        client_ip: str = request.client.host if request.client else "unknown"
        if not self._store.allow_request(
            client_ip,
            limit=self._limit,
            window_seconds=self._window_seconds,
        ):
            emit_security_audit(
                "rate_limit",
                "failure",
                path=request.url.path,
                method=request.method,
                client_ip=client_ip,
                reason="limit_exceeded",
                metadata={"limit": self._limit, "window_seconds": self._window_seconds},
                sinks=_audit_sinks(request),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {self._limit} requests per minute allowed."
                    )
                },
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# Legacy authentication hook
# ---------------------------------------------------------------------------


class AuthPlugin(ABC):
    """Backwards-compatible binary authentication hook."""

    @abstractmethod
    def authenticate(self, request: Request) -> bool:
        """Determine whether the request is authenticated."""

    def unauthenticated_response(self) -> Response:
        """Return the HTTP response sent when authentication fails."""
        return JSONResponse(status_code=401, content={"detail": "Unauthorized."})


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that delegates authentication to a legacy :class:`AuthPlugin`."""

    _DEFAULT_EXCLUDED: frozenset[str] = frozenset(
        {"/health", "/info", "/v1/health", "/v1/info"}
    )

    def __init__(
        self,
        app: ASGIApp,
        auth_plugin: AuthPlugin,
        exclude_paths: frozenset[str] | None = None,
    ) -> None:
        """Initialise the authentication middleware."""
        super().__init__(app)
        self._auth: AuthPlugin = auth_plugin
        self._excluded: frozenset[str] = (
            exclude_paths if exclude_paths is not None else self._DEFAULT_EXCLUDED
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Run the auth check before forwarding to the next ASGI layer."""
        if request.url.path in self._excluded:
            return await call_next(request)

        if not self._auth.authenticate(request):
            return self._auth.unauthenticated_response()

        return await call_next(request)
