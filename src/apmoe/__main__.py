"""Allow ``python -m apmoe`` so the CLI uses the same interpreter as ``python``.

Use this when ``apmoe`` on ``PATH`` points at a different environment than the
interpreter you expect (e.g. after ``pip install -e .`` into the wrong Python).
"""

from __future__ import annotations

from apmoe.cli.main import cli

if __name__ == "__main__":
    cli()
