"""Run the keystroke demo remote executor with ``python -m``."""

from __future__ import annotations

import argparse

import uvicorn

from remote_executors.keystroke_demo.app import create_app


def main() -> None:
    """Start the demo executor HTTP server."""
    parser = argparse.ArgumentParser(
        description="APMoE keystroke remote executor demo service."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8010, type=int)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Optional path to keystroke_age_expert.onnx. Defaults to the "
            "repo's bundled weights/keystroke_age_expert.onnx."
        ),
    )
    args = parser.parse_args()

    uvicorn.run(
        create_app(weights_path=args.weights),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
