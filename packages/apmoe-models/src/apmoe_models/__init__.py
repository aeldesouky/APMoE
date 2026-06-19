"""Packaged demo model artifacts for APMoE."""

from __future__ import annotations

from importlib import resources

__version__ = "0.1.3"


def weights() -> resources.abc.Traversable:
    """Return the package resource directory containing model artifacts."""
    return resources.files(__name__).joinpath("weights")
