"""Shared pytest fixtures for the APMoE test suite.

Fixtures here are available to all tests without explicit import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a minimal but valid raw config dict, with optional overrides merged."""
    base: dict[str, Any] = {
        "apmoe": {
            "modalities": [
                {
                    "name": "visual",
                    "processor": "myproject.processors.VisualProcessor",
                    "pipeline": {
                        "cleaner": "myproject.cleaners.ImageCleaner",
                        "anonymizer": "myproject.anonymizers.FaceAnonymizer",
                    },
                }
            ],
            "experts": [
                {
                    "name": "face_expert",
                    "class": "myproject.experts.FaceExpert",
                    "weights": "./weights/face.pt",
                    "modalities": ["visual"],
                }
            ],
            "aggregation": {
                "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
            },
        }
    }
    if overrides:
        # Deep-merge overrides into base (one level deep for simplicity)
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k].update(v)
            else:
                base[k] = v
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_config_dict() -> dict[str, Any]:
    """A minimal valid raw config dictionary (no file I/O)."""
    return _minimal_config()


@pytest.fixture()
def minimal_config_file(tmp_path: Path) -> Path:
    """Write a minimal valid config JSON file and return its path."""
    data = _minimal_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


@pytest.fixture()
def full_config_file(tmp_path: Path) -> Path:
    """Write a full multi-modality config JSON file and return its path."""
    data: dict[str, Any] = {
        "apmoe": {
            "modalities": [
                {
                    "name": "visual",
                    "processor": "myproject.processors.VisualProcessor",
                    "pipeline": {
                        "cleaner": "myproject.cleaners.ImageCleaner",
                        "anonymizer": "myproject.anonymizers.FaceAnonymizer",
                        "embedder": "myproject.embedders.MobileNetEmbedder",
                    },
                },
                {
                    "name": "audio",
                    "processor": "myproject.processors.AudioProcessor",
                    "pipeline": {
                        "cleaner": "myproject.cleaners.AudioCleaner",
                        "anonymizer": "myproject.anonymizers.VoiceAnonymizer",
                    },
                },
            ],
            "experts": [
                {
                    "name": "face_expert",
                    "class": "myproject.experts.FaceExpert",
                    "weights": "./weights/face.pt",
                    "modalities": ["visual"],
                },
                {
                    "name": "audio_expert",
                    "class": "myproject.experts.AudioExpert",
                    "weights": "./weights/audio.pt",
                    "modalities": ["audio"],
                },
                {
                    "name": "multi_expert",
                    "class": "myproject.experts.MultiExpert",
                    "weights": "./weights/multi.pt",
                    "modalities": ["visual", "audio"],
                },
            ],
            "aggregation": {
                "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator",
                "weights": {
                    "face_expert": 0.5,
                    "audio_expert": 0.3,
                    "multi_expert": 0.2,
                },
            },
            "serving": {
                "host": "127.0.0.1",
                "port": 9000,
                "workers": 2,
            },
        }
    }
    config_path = tmp_path / "full_config.json"
    config_path.write_text(json.dumps(data), encoding="utf-8")
    return config_path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any APMOE_ environment variables before each test."""
    apmoe_vars = [k for k in os.environ if k.startswith("APMOE_")]
    for var in apmoe_vars:
        monkeypatch.delenv(var, raising=False)
