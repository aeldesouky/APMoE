#!/usr/bin/env python3
"""End-to-end smoke test: image → Base64ImageCleaner → LMStudioExpert → local LLM.

This script validates the full APMoE remote-expert pipeline against a locally
running LM Studio instance.  It is intentionally minimal — all provider logic
lives in :class:`apmoe.experts.providers.lmstudio.LMStudioExpert` and the
image preprocessing in :class:`apmoe.processing.llm.Base64ImageCleaner`.

Usage
-----
::

    # Optionally override defaults via env:
    export LLM_ENDPOINT="http://127.0.0.1:1234/api/v1/chat"
    export LLM_MODEL="google/gemma-4-e4b"
    export LLM_SYSTEM_PROMPT="Return ONLY a single integer age."

    python scripts/test_llm_remote.py [path/to/image.jpg]

The image argument defaults to ``kmelsayed.jpg`` in the project root if not
supplied.  The config used is always ``configs/llm_remote.json``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the package is importable when run from the project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Provider-specific expert — imported here so the registry entry is created
# before APMoEApp.from_config() resolves the dotted class path.
import apmoe.experts.providers.lmstudio  # noqa: F401  (side-effect: registry registration)

from apmoe.core.app import APMoEApp


def main() -> None:
    # ── Resolve the image path ─────────────────────────────────────────────
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1]).resolve()
    else:
        image_path = _PROJECT_ROOT / "kmelsayed.jpg"

    config_path = _PROJECT_ROOT / "configs" / "llm_remote.json"

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # ── Env-var defaults (overridden by real shell exports) ────────────────
    # In production these come from the shell / .env / secrets manager.
    # Defaults here make the script self-contained for local development.
    os.environ.setdefault("LLM_ENDPOINT", "http://127.0.0.1:1234/api/v1/chat")
    os.environ.setdefault("LLM_MODEL", "google/gemma-4-e4b")
    os.environ.setdefault(
        "LLM_SYSTEM_PROMPT",
        "You are an age estimation model. The user provides a base64-encoded JPEG image. "
        "Return ONLY a single integer representing the estimated age of the person in the image. "
        "No explanation, no other text, just the number.",
    )

    # ── Bootstrap ─────────────────────────────────────────────────────────
    print("─" * 60)
    print("  APMoE LM Studio end-to-end test")
    print("─" * 60)
    print(f"  Image  : {image_path.name}")
    print(f"  Config : {config_path.name}")
    print(f"  LLM    : {os.environ['LLM_ENDPOINT']}")
    print(f"  Model  : {os.environ['LLM_MODEL']}")
    print("─" * 60)
    print("  Bootstrapping APMoE...")

    app = APMoEApp.from_config(str(config_path))

    print("  Bootstrap complete.  Running inference...")
    print("  (LLM call may take a few seconds...)\n")

    # ── Inference ─────────────────────────────────────────────────────────
    image_bytes = image_path.read_bytes()
    result = app.predict({"image": image_bytes})

    # ── Results ───────────────────────────────────────────────────────────
    expert_out = result.per_expert_outputs[0]
    meta = expert_out.metadata

    print("─" * 60)
    print(f"  Predicted age    : {result.predicted_age:.0f} years")
    print(f"  Confidence       : {result.confidence} (n/a — LLM regressor)")
    print(f"  Expert           : {expert_out.expert_name}")
    print()
    print(f"  LLM raw response : \"{meta.get('llm_response', '')}\"")
    print(f"  Model            : {meta.get('model', '')}")
    print(f"  Tokens/sec       : {meta.get('tokens_per_sec', 'n/a')}")
    print(f"  Pipeline latency : {result.metadata.get('pipeline_latency_s', 'n/a')} s")
    print("─" * 60)

    if result.skipped_experts:
        print(f"\n  Skipped experts: {result.skipped_experts}")


if __name__ == "__main__":
    main()
