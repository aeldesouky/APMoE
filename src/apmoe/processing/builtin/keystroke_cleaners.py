"""Compatibility shim for older configs.

Historically some templates referenced ``KeystrokeCleaner`` under this module
name.  The canonical location is :mod:`apmoe.processing.builtin.cleaners`.
"""

from __future__ import annotations

from apmoe.processing.builtin.cleaners import KeystrokeCleaner

__all__ = ["KeystrokeCleaner"]
