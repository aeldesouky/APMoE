"""Serving layer: FastAPI app factory, route handlers, and middleware.

Public symbols
--------------
- :func:`~apmoe.serving.app_factory.create_api` — build the FastAPI ASGI app.
- :class:`~apmoe.serving.middleware.AuthPlugin` — abstract auth hook.
- :class:`~apmoe.serving.middleware.AuthMiddleware` — auth middleware.
- :class:`~apmoe.serving.middleware.RateLimitMiddleware` — rate limiting.
- :class:`~apmoe.serving.middleware.RequestLoggingMiddleware` — request logging.
"""
