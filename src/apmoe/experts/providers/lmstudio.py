"""RemoteExpert subclass for the LM Studio local inference server.

LM Studio exposes a ``POST /api/v1/chat`` endpoint whose response schema
differs from the flat ``{"predicted_age": ...}`` convention expected by the
default :class:`~apmoe.experts.remote.RemoteExpert._parse_response`.  This
module provides :class:`LMStudioExpert`, a drop-in subclass that understands
that schema and extracts the age value automatically.

LM Studio response schema
--------------------------
.. code-block:: json

    {
      "output": [
        {"type": "reasoning", "content": "Let me look at the image…"},
        {"type": "message",   "content": "34"}
      ],
      "stats": {
        "input_tokens":        1920,
        "total_output_tokens": 3,
        "tokens_per_second":   52.7
      },
      "model_instance_id": "google/gemma-4-e4b"
    }

The expert finds the first item in ``output`` whose ``type`` is
``"message"``, then extracts the first integer from its ``content`` string
as the predicted age.  Inference statistics from ``stats`` are forwarded
verbatim into the ``ExpertOutput.metadata`` dict for observability.

Configuration example
---------------------
.. code-block:: json

    {
      "name":       "llm_face_age_expert",
      "class":      "apmoe.experts.providers.lmstudio.LMStudioExpert",
      "modalities": ["image"],
      "endpoint":   "$LLM_ENDPOINT",
      "endpoint_headers": {"Content-Type": "application/json"},
      "endpoint_timeout": 60.0,
      "request_template": {
        "model":         "$LLM_MODEL",
        "system_prompt": "$LLM_SYSTEM_PROMPT",
        "input":         "{{modalities.image}}"
      }
    }

Required environment variables
-------------------------------
* ``LLM_ENDPOINT`` — full URL, e.g. ``http://127.0.0.1:1234/api/v1/chat``
* ``LLM_MODEL`` — model identifier shown in LM Studio, e.g. ``google/gemma-4-e4b``
* ``LLM_SYSTEM_PROMPT`` — instruction prompt for the model

These are expanded by the base class :meth:`~apmoe.experts.remote.RemoteExpert.load_weights`
at bootstrap time — never hardcoded in the config file.

Reference config
----------------
See ``configs/llm_remote.json`` for a complete, ready-to-use configuration.

Pairing with the image pipeline
--------------------------------
When the image modality is consumed by this expert, the modality pipeline
must use :class:`~apmoe.processing.llm.Base64ImageCleaner` and
:class:`~apmoe.processing.llm.PassthroughImageAnonymizer` instead of the
standard ``ImageCleaner``/``ImageAnonymizer`` pair::

    "pipeline": {
      "cleaner":    "apmoe.processing.llm.Base64ImageCleaner",
      "anonymizer": "apmoe.processing.llm.PassthroughImageAnonymizer"
    }
"""

from __future__ import annotations

import re
from typing import Any

from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput
from apmoe.experts.registry import expert_registry
from apmoe.experts.remote import RemoteExpert


@expert_registry.register("apmoe.experts.providers.lmstudio.LMStudioExpert")
class LMStudioExpert(RemoteExpert):
    """RemoteExpert pre-wired to the LM Studio ``/api/v1/chat`` response format.

    Inherits all HTTP handling, retry/circuit-breaker logic, env-var
    substitution, and request templating from
    :class:`~apmoe.experts.remote.RemoteExpert`.  The only customisation is
    :meth:`_parse_response`, which navigates the ``output`` array and extracts
    the numeric age from the first ``"message"``-type item.

    Registered as ``"apmoe.experts.providers.lmstudio.LMStudioExpert"`` in
    :data:`~apmoe.experts.registry.expert_registry`.

    Example config key::

        "class": "apmoe.experts.providers.lmstudio.LMStudioExpert"
    """

    def _parse_response(
        self, data: Any, consumed_modalities: list[str]
    ) -> ExpertOutput:
        """Extract the predicted age from LM Studio's ``output`` array.

        Searches ``data["output"]`` for the first item whose ``"type"`` is
        ``"message"``, then uses a regex to find the first integer in its
        ``"content"`` string.  LM Studio inference statistics from
        ``data["stats"]`` are forwarded to ``ExpertOutput.metadata``.

        Args:
            data: Parsed JSON response from the LM Studio endpoint.
            consumed_modalities: Modality names that were sent in the request.

        Returns:
            :class:`~apmoe.core.types.ExpertOutput` with:

            * ``predicted_age`` — first integer found in the message content.
            * ``confidence`` — always ``-1.0`` (LLM regression, no calibrated
              probability).
            * ``metadata`` — includes ``llm_response``, ``model``,
              ``input_tokens``, ``output_tokens``, ``tokens_per_sec``,
              ``backend``, and ``endpoint``.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If no
                ``"message"``-type item exists in ``output``, or its
                ``"content"`` contains no recognisable integer.
        """
        # ── Locate the first "message"-type output item ───────────────────
        output_items: list[dict[str, str]] = data.get("output", [])
        message_content: str = ""
        for item in output_items:
            if item.get("type") == "message":
                message_content = item.get("content", "").strip()
                break

        if not message_content:
            raise ExpertError(
                "LMStudioExpert: no item with type='message' found in "
                "the response 'output' array.  Check that the model is "
                "returning text output and that the endpoint URL is correct.",
                context={"response_preview": str(data)[:300]},
            )

        # ── Extract the first integer (the predicted age) ─────────────────
        numbers = re.findall(r"\b\d+\b", message_content)
        if not numbers:
            raise ExpertError(
                f"LMStudioExpert: no integer found in the model response "
                f"content: '{message_content}'.  The system prompt may need "
                f"to be more specific about the output format.",
                context={"content": message_content},
            )

        predicted_age = float(numbers[0])

        # ── Collect inference statistics for observability ─────────────────
        stats: dict[str, Any] = data.get("stats", {})
        metadata: dict[str, Any] = {
            "backend":        "remote",
            "endpoint":       self._endpoint,
            "llm_response":   message_content,
            "model":          data.get("model_instance_id", "unknown"),
            "input_tokens":   stats.get("input_tokens"),
            "output_tokens":  stats.get("total_output_tokens"),
            "tokens_per_sec": stats.get("tokens_per_second"),
        }

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=consumed_modalities,
            predicted_age=predicted_age,
            confidence=-1.0,  # LLM regressor — no calibrated probability
            metadata=metadata,
        )
