#!/usr/bin/env python3
"""End-to-end test: run kmelsayed.jpg through the local Gemma 4 LLM
using APMoE's RemoteExpert pipeline.

Usage:
    python scripts/test_llm_remote.py

The script:
1. Bootstraps APMoE from configs/llm_remote.json
2. Reads kmelsayed.jpg as raw bytes
3. Runs the full APMoE inference pipeline (Base64ImageCleaner → RemoteExpert → LLM)
4. Prints the predicted age and full pipeline result

The local LLM (LM Studio) response schema is:
    {
      "output": [
        {"type": "reasoning", "content": "..."},
        {"type": "message",   "content": "<age as integer>"}
      ],
      ...
    }

We subclass RemoteExpert to parse the LLM's output array and extract
the numeric age from the first "message"-type item.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

# Add the project source to the path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput, ProcessedInput
from apmoe.experts.remote import RemoteExpert
from apmoe.experts.registry import expert_registry


# ---------------------------------------------------------------------------
# Custom RemoteExpert subclass that understands the LM Studio response format
# ---------------------------------------------------------------------------


@expert_registry.register("apmoe.experts.remote.LMStudioExpert")
class LMStudioExpert(RemoteExpert):
    """RemoteExpert that parses the LM Studio /api/v1/chat response format.

    The response is:
        {
          "output": [
            {"type": "reasoning", "content": "..."},
            {"type": "message",   "content": "<integer>"}
          ],
          "stats": {...}
        }

    This subclass overrides ``_parse_response`` to find the first
    "message"-type item in ``output`` and extract a numeric age from its
    ``content`` string.
    """

    def _parse_response(
        self, data: Any, consumed_modalities: list[str]
    ) -> ExpertOutput:
        """Extract age from LM Studio's output array."""
        # Find the first item with type=="message"
        output_items: list[dict[str, str]] = data.get("output", [])
        message_content: str = ""
        for item in output_items:
            if item.get("type") == "message":
                message_content = item.get("content", "").strip()
                break

        if not message_content:
            raise ExpertError(
                "LMStudioExpert: no 'message' item found in LLM response output.",
                context={"response": str(data)[:300]},
            )

        # Extract the first integer found in the message
        numbers = re.findall(r"\b\d+\b", message_content)
        if not numbers:
            raise ExpertError(
                f"LMStudioExpert: cannot find a numeric age in LLM response: "
                f"'{message_content}'",
                context={"content": message_content},
            )

        predicted_age = float(numbers[0])

        # Collect stats for metadata
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
            confidence=-1.0,  # LLM is a regressor — no calibrated probability
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    image_path = _PROJECT_ROOT / "kmelsayed.jpg"
    config_path = _PROJECT_ROOT / "configs" / "llm_remote.json"

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # --- Set env vars if not already present ----------------------------
    # In a real deployment these come from the shell / .env file / secrets manager.
    # We set defaults here only so the test is self-contained.
    os.environ.setdefault("LLM_ENDPOINT",       "http://127.0.0.1:1234/api/v1/chat")
    os.environ.setdefault("LLM_MODEL",          "google/gemma-4-e4b")
    os.environ.setdefault(
        "LLM_SYSTEM_PROMPT",
        "You are an age estimation model. The user provides a base64-encoded JPEG image. "
        "Return ONLY a single integer representing the estimated age of the person in the image. "
        "No explanation, no other text, just the number.",
    )

    # Patch the config to use LMStudioExpert instead of RemoteExpert
    # so we get the custom response parser
    raw_cfg = json.loads(config_path.read_text())
    raw_cfg["apmoe"]["experts"][0]["class"] = "apmoe.experts.remote.LMStudioExpert"

    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=_PROJECT_ROOT
    ) as tmp:
        json.dump(raw_cfg, tmp)
        tmp_path = tmp.name

    try:
        print("─" * 60)
        print("  APMoE × Local LLM (Gemma 4) — Age Estimation Test")
        print("─" * 60)
        print(f"  Image  : {image_path.name}")
        print(f"  Config : {config_path.name}")
        print(f"  LLM    : {raw_cfg['apmoe']['experts'][0]['endpoint']}")
        print("─" * 60)
        print("  Bootstrapping APMoE...")

        from apmoe.core.app import APMoEApp
        app = APMoEApp.from_config(tmp_path)

        print("  Bootstrap complete. Running inference...")
        print("  (LLM call may take a few seconds...)\n")

        image_bytes = image_path.read_bytes()
        result = app.predict({"image": image_bytes})

        print("─" * 60)
        print(f"  ✅  Predicted age : {result.predicted_age:.0f} years")
        print(f"  Confidence       : {result.confidence} (n/a — LLM regressor)")
        print(f"  Expert           : {result.per_expert_outputs[0].expert_name}")
        print()
        meta = result.per_expert_outputs[0].metadata
        print(f"  LLM raw response : \"{meta.get('llm_response', '')}\"")
        print(f"  Model            : {meta.get('model', '')}")
        print(f"  Tokens/sec       : {meta.get('tokens_per_sec', 'n/a')}")
        print(f"  Pipeline latency : {result.metadata.get('pipeline_latency_s', 'n/a')} s")
        print("─" * 60)

        if result.skipped_experts:
            print(f"\n  Skipped experts: {result.skipped_experts}")

    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    main()
