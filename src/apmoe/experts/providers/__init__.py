"""Built-in provider-specific RemoteExpert subclasses.

This package contains ready-to-use :class:`~apmoe.experts.remote.RemoteExpert`
subclasses that are pre-wired to the request/response schema of specific
popular LLM providers.  Each class handles one provider's API without
requiring the integrator to write any custom parsing code.

Provided providers
------------------
* :mod:`apmoe.experts.providers.lmstudio` — LM Studio local inference server
  (``/api/v1/chat`` schema with ``output[*].type`` array).

Adding a new provider
---------------------
1. Create ``src/apmoe/experts/providers/<name>.py``.
2. Subclass :class:`~apmoe.experts.remote.RemoteExpert`.
3. Override :meth:`~apmoe.experts.remote.RemoteExpert._parse_response` (and
   optionally :meth:`~apmoe.experts.remote.RemoteExpert._build_request_body`).
4. Decorate with ``@expert_registry.register("<dotted.module.ClassName>")``.
5. Add a reference config to ``configs/`` and unit tests to
   ``tests/unit/providers/``.

All ``$VAR`` env-var expansion in ``endpoint``, ``endpoint_headers``, and
``request_template`` is inherited from :class:`~apmoe.experts.remote.RemoteExpert`
— no extra wiring required in the subclass.
"""
