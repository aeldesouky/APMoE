"""FastAPI application factory for the APMoE serving layer.

The single public symbol here is :func:`create_api`, which accepts a
bootstrapped :class:`~apmoe.core.app.APMoEApp` instance and returns a
fully-configured :class:`~fastapi.FastAPI` ASGI application.

:meth:`~apmoe.core.app.APMoEApp.serve` calls this factory internally —
application code should not need to call it directly unless embedding the
framework inside an existing FastAPI / ASGI application.

Multi-worker serving
--------------------
When ``serving.workers > 1`` uvicorn spawns multiple OS processes.  Because
each worker is a fresh interpreter, the app object from the parent process
cannot be shared — uvicorn requires a **string import path** together with the
``--factory`` flag so each worker calls :func:`create_worker_app` itself.

:func:`~apmoe.core.app.APMoEApp.serve` handles this automatically by:

1. Writing the config path to the ``APMOE_CONFIG_PATH`` environment variable.
2. Passing ``"apmoe.serving.app_factory:create_worker_app"`` as the app
   argument with ``factory=True``.

Each worker process then calls :func:`create_worker_app`, which reads the
environment variable, bootstraps a private :class:`~apmoe.core.app.APMoEApp`,
and returns a fully-configured :class:`~fastapi.FastAPI` instance.
"""

from __future__ import annotations

import logging
import os
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

#: Environment variable key used to pass the config path to worker processes.
_WORKER_CONFIG_ENV_KEY = "APMOE_CONFIG_PATH"


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


def create_worker_app() -> FastAPI:
    """Bootstrap a :class:`~apmoe.core.app.APMoEApp` and return a FastAPI app.

    This function is the **uvicorn factory entry point** for multi-worker
    (multi-process) deployments.  Each worker process calls it independently
    to construct its own ``APMoEApp`` — shared-memory app objects cannot be
    forked safely across uvicorn worker processes.

    The config file path is read from the ``APMOE_CONFIG_PATH`` environment
    variable, which :meth:`~apmoe.core.app.APMoEApp.serve` writes before
    handing control to uvicorn.

    Returns:
        A fully-configured :class:`~fastapi.FastAPI` ASGI application.

    Raises:
        :class:`~apmoe.core.exceptions.ConfigurationError`: If
            ``APMOE_CONFIG_PATH`` is not set or the config file is invalid.

    Usage (internal — called by uvicorn when ``factory=True``)::

        # APMoEApp.serve() passes the equivalent of:
        uvicorn.run(
            "apmoe.serving.app_factory:create_worker_app",
            factory=True,
            host="0.0.0.0",
            port=8000,
            workers=4,
        )
    """
    from apmoe.core.app import APMoEApp  # local import avoids circular dependency
    from apmoe.core.exceptions import ConfigurationError

    config_path = os.environ.get(_WORKER_CONFIG_ENV_KEY)
    if not config_path:
        raise ConfigurationError(
            f"Multi-worker bootstrap requires the '{_WORKER_CONFIG_ENV_KEY}' "
            f"environment variable to be set to the APMoE config file path.  "
            f"This variable is written automatically by APMoEApp.serve().",
            context={"env_key": _WORKER_CONFIG_ENV_KEY},
        )

    logger.info(
        "Worker process starting — loading config from '%s'", config_path
    )
    apmoe_app = APMoEApp.from_config(config_path)
    return create_api(apmoe_app)
