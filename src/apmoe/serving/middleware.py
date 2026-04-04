"""Middleware components for the APMoE HTTP serving layer.

Three middleware classes are provided:

* :class:`RequestLoggingMiddleware` — structured JSON request logging with
  per-request correlation IDs (``X-Correlation-ID`` response header).
* :class:`RateLimitMiddleware` — sliding-window in-memory rate limiter keyed
  by client IP.  Configured via
  :attr:`~apmoe.core.config.ServingConfig.rate_limit`.
* :class:`AuthPlugin` / :class:`AuthMiddleware` — abstract authentication hook
  that framework users implement and attach via
  :func:`~apmoe.serving.app_factory.create_api`.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

if TYPE_CHECKING:
    pass

logger = logging.getLogger("apmoe.serving")

# ---------------------------------------------------------------------------
# Request Logging
# ---------------------------------------------------------------------------


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request as a structured JSON record.

    A UUID4 correlation ID is generated for each inbound request and
    propagated in two ways:

    * Stored on ``request.state.correlation_id`` so route handlers and
      downstream middleware can reference it.
    * Returned to the caller as the ``X-Correlation-ID`` response header.

    Log records are emitted at ``INFO`` level to the ``apmoe.serving``
    logger and contain:

    * ``correlation_id``
    * ``method`` (HTTP verb)
    * ``path``
    * ``query`` (raw query string, ``null`` when absent)
    * ``status_code``
    * ``duration_ms`` (rounded to 2 decimal places)
    * ``client_host``

    Example log line::

        {"correlation_id": "…", "method": "POST", "path": "/predict",
         "query": null, "status_code": 200, "duration_ms": 14.32,
         "client_host": "127.0.0.1"}
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process the request, attach a correlation ID, and log the outcome.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that passes the request to the next ASGI layer.

        Returns:
            The HTTP response with an added ``X-Correlation-ID`` header.
        """
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        started_at = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.error(
                "[%s] 500 Unhandled exception in %s %s after %.2f ms — %s: %s",
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
        record: dict[str, object] = {
            "correlation_id": correlation_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) or None,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "client_host": request.client.host if request.client else None,
        }

        if status_code >= 500:
            logger.error(
                "[%s] %d %s %s — %.2f ms",
                correlation_id,
                status_code,
                request.method,
                request.url.path,
                duration_ms,
            )
        elif status_code >= 400:
            logger.warning(
                "[%s] %d %s %s — %.2f ms",
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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window in-memory rate limiter keyed by client IP address.

    For each client IP, a list of request timestamps is maintained.  On every
    incoming request the list is pruned to the last 60 seconds; if the
    remaining count equals or exceeds *max_requests_per_minute* the request
    is rejected with ``HTTP 429 Too Many Requests`` and a
    ``Retry-After: 60`` header.

    .. note::
        This implementation is **in-process only**.  With multiple uvicorn
        workers (separate OS processes), each worker has its own counter, so
        the effective limit is ``max_requests_per_minute x workers``.  For
        production deployments requiring strict limits, use a shared store
        (Redis, etc.) outside the framework.

    Args:
        app: The ASGI application to wrap.
        max_requests_per_minute: Maximum number of requests allowed per client
            IP in any rolling 60-second window.
    """

    def __init__(self, app: ASGIApp, max_requests_per_minute: int) -> None:
        """Initialise the rate limiter.

        Args:
            app: The ASGI application to wrap.
            max_requests_per_minute: Request cap per client IP per 60 seconds.
        """
        super().__init__(app)
        self._limit: int = max_requests_per_minute
        self._window_seconds: float = 60.0
        self._request_log: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Enforce the rate limit before forwarding to the next ASGI layer.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that passes the request to the next layer.

        Returns:
            A ``429 Too Many Requests`` response if the limit is exceeded,
            otherwise the response from the next layer.
        """
        client_ip: str = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - self._window_seconds

        # Prune timestamps that fall outside the rolling window
        self._request_log[client_ip] = [
            t for t in self._request_log[client_ip] if t > cutoff
        ]

        if len(self._request_log[client_ip]) >= self._limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        f"Rate limit exceeded: {self._limit} requests per minute allowed."
                    )
                },
                headers={"Retry-After": "60"},
            )

        self._request_log[client_ip].append(now)
        return await call_next(request)


# ---------------------------------------------------------------------------
# Authentication hook
# ---------------------------------------------------------------------------


class AuthPlugin(ABC):
    """Abstract authentication hook for the APMoE serving layer.

    Subclass :class:`AuthPlugin` and pass an instance to
    :func:`~apmoe.serving.app_factory.create_api` to enforce custom
    authentication on every incoming request (except paths listed in
    ``exclude_paths``).

    Example::

        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from apmoe.serving.middleware import AuthPlugin

        class ApiKeyAuth(AuthPlugin):
            def __init__(self, valid_keys: set[str]) -> None:
                self._keys = valid_keys

            def authenticate(self, request: Request) -> bool:
                key = request.headers.get("X-API-Key", "")
                return key in self._keys

            def unauthenticated_response(self) -> Response:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key."},
                )
    """

    @abstractmethod
    def authenticate(self, request: Request) -> bool:
        """Determine whether the request is authenticated.

        Args:
            request: The incoming Starlette request.

        Returns:
            ``True`` to allow the request through; ``False`` to reject it.
        """

    def unauthenticated_response(self) -> Response:
        """Return the HTTP response sent when authentication fails.

        Override to customise the rejection payload and status code.
        The default implementation returns ``HTTP 401 Unauthorized``.

        Returns:
            A :class:`~starlette.responses.Response` (or subclass) returned
            directly to the client without invoking the route handler.
        """
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized."},
        )


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware that delegates authentication to a developer-provided :class:`AuthPlugin`.

    Paths listed in *exclude_paths* bypass authentication entirely — by
    default this includes ``/health`` and ``/info`` so monitoring systems can
    probe the service without credentials.

    Args:
        app: The ASGI application to wrap.
        auth_plugin: The :class:`AuthPlugin` instance that performs
            the authentication check.
        exclude_paths: Set of URL paths that bypass authentication.
            Defaults to ``{"/health", "/info"}``.
    """

    _DEFAULT_EXCLUDED: frozenset[str] = frozenset({"/health", "/info"})

    def __init__(
        self,
        app: ASGIApp,
        auth_plugin: AuthPlugin,
        exclude_paths: frozenset[str] | None = None,
    ) -> None:
        """Initialise the authentication middleware.

        Args:
            app: The ASGI application to wrap.
            auth_plugin: Plugin that performs the authentication check.
            exclude_paths: Optional override for paths that skip auth.
        """
        super().__init__(app)
        self._auth: AuthPlugin = auth_plugin
        self._excluded: frozenset[str] = (
            exclude_paths if exclude_paths is not None else self._DEFAULT_EXCLUDED
        )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Run the auth check before forwarding to the next ASGI layer.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that passes the request to the next layer.

        Returns:
            The response from :meth:`AuthPlugin.unauthenticated_response` if
            the request fails authentication; otherwise the response from the
            next layer.
        """
        if request.url.path in self._excluded:
            return await call_next(request)

        if not self._auth.authenticate(request):
            return self._auth.unauthenticated_response()

        return await call_next(request)
