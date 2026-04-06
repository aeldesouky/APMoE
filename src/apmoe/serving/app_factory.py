"""FastAPI application factory for the APMoE serving layer.

The single public symbol here is :func:`create_api`, which accepts a
bootstrapped :class:`~apmoe.core.app.APMoEApp` instance and returns a
fully-configured :class:`~fastapi.FastAPI` ASGI application.

:meth:`~apmoe.core.app.APMoEApp.serve` calls this factory internally —
application code should not need to call it directly unless embedding the
framework inside an existing FastAPI / ASGI application.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import apmoe
from apmoe.serving.middleware import (
    AuthMiddleware,
    AuthPlugin,
    RateLimitMiddleware,
    RequestLoggingMiddleware,
)
from apmoe.serving.openapi_schemas import OPENAPI_DESCRIPTION, OPENAPI_TAGS
from apmoe.serving.routes import create_router

if TYPE_CHECKING:
    from apmoe.core.app import APMoEApp

logger = logging.getLogger("apmoe.serving")


def create_api(
    app: APMoEApp,
    *,
    auth_plugin: AuthPlugin | None = None,
    auth_exclude_paths: frozenset[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application for the APMoE serving layer.

    Middleware is applied in the following order (outermost first, so the
    first listed intercepts the request first):

    1. **CORS** — handles ``OPTIONS`` preflight requests before any other layer.
    2. **Request Logging** — generates a correlation ID and logs the request.
    3. **Rate Limiting** — enforces per-IP request caps (only when configured).
    4. **Authentication** — delegates to the developer-provided plugin (optional).

    OpenAPI documentation is auto-generated and available at ``/docs``
    (Swagger UI) and ``/redoc`` (ReDoc).

    Args:
        app: The bootstrapped :class:`~apmoe.core.app.APMoEApp` instance.
            The serving config (CORS origins, rate limit) is read from
            ``app.config.apmoe.serving``.
        auth_plugin: Optional :class:`~apmoe.serving.middleware.AuthPlugin`
            implementation.  When provided, an
            :class:`~apmoe.serving.middleware.AuthMiddleware` is attached that
            calls :meth:`~apmoe.serving.middleware.AuthPlugin.authenticate` on
            every request not in *auth_exclude_paths*.
        auth_exclude_paths: Set of URL paths that bypass the auth plugin.
            Defaults to ``{"/health", "/info"}`` when *auth_plugin* is given.
            Has no effect when *auth_plugin* is ``None``.

    Returns:
        A fully-configured :class:`~fastapi.FastAPI` ASGI application ready
        to be passed to ``uvicorn.run()`` or used with a
        :class:`~fastapi.testclient.TestClient`.

    Example::

        from apmoe.core.app import APMoEApp
        from apmoe.serving.app_factory import create_api

        apmoe_app = APMoEApp.from_config("configs/default.json")
        api = create_api(apmoe_app)

        # Embed in an existing FastAPI app:
        parent_app.mount("/apmoe", api)
    """
    serving_cfg = app.config.apmoe.serving

    api = FastAPI(
        title="APMoE — Age Prediction using Mixture of Experts",
        description=OPENAPI_DESCRIPTION,
        version=apmoe.__version__,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # 1. CORS — must be outermost so OPTIONS preflight requests are handled
    #    before any other middleware inspects them.
    api.add_middleware(
        CORSMiddleware,
        allow_origins=serving_cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Structured JSON request logging with per-request correlation IDs.
    api.add_middleware(RequestLoggingMiddleware)

    # 3. Rate limiting — only attached when a limit is configured.
    if serving_cfg.rate_limit is not None:
        api.add_middleware(
            RateLimitMiddleware,
            max_requests_per_minute=serving_cfg.rate_limit,
        )

    # 4. Developer-provided authentication hook — only attached when supplied.
    if auth_plugin is not None:
        api.add_middleware(
            AuthMiddleware,
            auth_plugin=auth_plugin,
            exclude_paths=auth_exclude_paths,
        )

    # Catch-all handler for any exception that escapes the route layer.
    # FastAPI would otherwise return a bare 500 with no console output.
    @api.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        correlation_id: str = getattr(request.state, "correlation_id", "-")
        logger.error(
            "[%s] 500 Unhandled %s in %s %s: %s",
            correlation_id,
            type(exc).__name__,
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"},
        )

    # Mount route handlers
    router = create_router(app)
    api.include_router(router)

    return api
