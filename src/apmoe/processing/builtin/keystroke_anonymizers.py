"""Compatibility shim for older configs.

Historically some templates referenced ``KeystrokeAnonymizer`` under this module
name.  The canonical location is :mod:`apmoe.processing.builtin.anonymizers`.
"""

from __future__ import annotations

from apmoe.processing.builtin.anonymizers import KeystrokeAnonymizer

__all__ = ["KeystrokeAnonymizer"]
