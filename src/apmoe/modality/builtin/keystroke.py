"""Built-in keystroke-dynamics modality processor.

Parses raw keystroke session data into a per-feature timing dict that the
rest of the pipeline can clean, anonymise, and hand off to an expert.

Supported input formats
-----------------------

**IKDD text format** (primary — from keystroke logging tools):

.. code-block:: text

    # optional metadata / comment lines
    8-0,62.5,70.0,55.0
    13-0,98.0,95.0
    65-83,145.2,130.0,160.4

Each non-comment line is ``key1-key2,timing1,timing2,...``.
When ``key2 == 0`` the feature is a **hold time** (``dur_{key1}``);
when ``key2 != 0`` it is a **digraph flight time** (``dig_{key1}_{key2}``).

**JSON format** (secondary — for API or programmatic use):

.. code-block:: json

    [[8, 0, 62.5], [8, 0, 70.0], [13, 0, 98.0], [65, 83, 145.2]]

A list of ``[key1, key2, timing_ms]`` triples.

Both formats produce the same ``ModalityData`` output: a
``dict[str, list[float]]`` mapping feature names to their raw timing values.
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from apmoe.core.exceptions import ModalityError
from apmoe.core.types import ModalityData
from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import modality_registry


def _feature_name(key1: int, key2: int) -> str:
    """Return the canonical feature name for a ``(key1, key2)`` pair.

    Args:
        key1: First key scan-code.
        key2: Second key scan-code (``0`` = hold time / duration feature).

    Returns:
        ``"dur_{key1}"`` when ``key2 == 0``, else ``"dig_{key1}_{key2}"``.
    """
    return f"dur_{key1}" if key2 == 0 else f"dig_{key1}_{key2}"


@modality_registry.register("keystroke")
class KeystrokeProcessor(ModalityProcessor):
    """Parse raw keystroke session data into a feature-timing dict.

    Accepts IKDD-format text files or JSON-encoded lists of
    ``[key1, key2, timing_ms]`` triples (see module docstring).

    The ``data`` attribute of the returned
    :class:`~apmoe.core.types.ModalityData` is a
    ``dict[str, list[float]]`` mapping feature names (e.g.
    ``"dur_8"``, ``"dig_65_83"``) to their observed timing values (ms).
    Multiple timings for the same feature pair are all preserved so that
    the :class:`~apmoe.processing.builtin.cleaners.KeystrokeCleaner` can
    filter individual outlier values before the expert aggregates means.

    Registered as ``"keystroke"`` in
    :data:`~apmoe.modality.factory.modality_registry`.
    """

    @property
    def modality_name(self) -> str:
        """Return the canonical modality name ``"keystroke"``."""
        return "keystroke"

    def validate(self, data: object) -> bool:
        """Return ``True`` if *data* can be parsed as a keystroke session.

        Args:
            data: ``bytes`` or ``str`` (IKDD or JSON).

        Returns:
            ``True`` if the input is parseable and contains at least one
            valid timing row; ``False`` otherwise.
        """
        try:
            parsed = self._parse(data)
            return bool(parsed)
        except Exception:
            return False

    def preprocess(self, data: object) -> ModalityData:
        """Parse *data* into a :class:`~apmoe.core.types.ModalityData`.

        Args:
            data: Raw keystroke session — ``bytes`` or ``str`` in IKDD or
                JSON format.

        Returns:
            A :class:`~apmoe.core.types.ModalityData` with:

            * ``modality = "keystroke"``
            * ``data`` = ``dict[str, list[float]]`` mapping feature names
              to lists of raw timing values (ms).
            * ``metadata["num_features_observed"]`` = number of distinct
              features in this session.
            * ``metadata["num_raw_timings"]`` = total timing measurements.

        Raises:
            :class:`~apmoe.core.exceptions.ModalityError`: If parsing fails
                or the input contains no valid rows.
        """
        try:
            timings: dict[str, list[float]] = self._parse(data)
        except Exception as exc:
            raise ModalityError(
                f"KeystrokeProcessor: cannot parse input — {exc}",
                context={"input_type": type(data).__name__},
            ) from exc

        if not timings:
            raise ModalityError(
                "KeystrokeProcessor: input contains no valid timing rows.",
                context={"input_type": type(data).__name__},
            )

        num_raw = sum(len(v) for v in timings.values())

        return ModalityData(
            modality=self.modality_name,
            data=timings,
            metadata={
                "num_features_observed": len(timings),
                "num_raw_timings": num_raw,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(data: object) -> dict[str, list[float]]:
        """Decode *data* and return a ``{feature_name: [timings]}`` dict.

        Args:
            data: ``bytes`` or ``str`` — IKDD text or JSON list.

        Returns:
            Dict mapping feature names to lists of timing values.

        Raises:
            ValueError: If the format is unrecognised or structurally invalid.
        """
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(
                f"Expected bytes or str, got {type(data).__name__}."
            )

        text = text.strip()

        # JSON path: starts with '[' or '{'
        if text.startswith("["):
            return KeystrokeProcessor._parse_json_list(text)
        if text.startswith("{"):
            return KeystrokeProcessor._parse_json_dict(text)

        # IKDD text path
        return KeystrokeProcessor._parse_ikdd(text)

    @staticmethod
    def _parse_ikdd(text: str) -> dict[str, list[float]]:
        """Parse IKDD-format text into a feature-timing dict.

        Args:
            text: Multi-line IKDD string.

        Returns:
            ``{feature_name: [timings]}`` dict.
        """
        result: dict[str, list[float]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            key_part = parts[0].strip()
            try:
                k1_str, k2_str = key_part.split("-", 1)
                key1, key2 = int(k1_str), int(k2_str)
            except ValueError:
                continue
            feat = _feature_name(key1, key2)
            timings_for_feat = result.setdefault(feat, [])
            for v in parts[1:]:
                with contextlib.suppress(ValueError):
                    timings_for_feat.append(float(v.strip()))
        return result

    @staticmethod
    def _parse_json_list(text: str) -> dict[str, list[float]]:
        """Parse a JSON list of ``[key1, key2, timing_ms]`` triples.

        Args:
            text: JSON string containing a list of triples.

        Returns:
            ``{feature_name: [timings]}`` dict.
        """
        rows: list[Any] = json.loads(text)
        result: dict[str, list[float]] = {}
        for row in rows:
            key1, key2, timing = int(row[0]), int(row[1]), float(row[2])
            feat = _feature_name(key1, key2)
            result.setdefault(feat, []).append(timing)
        return result

    @staticmethod
    def _parse_json_dict(text: str) -> dict[str, list[float]]:
        """Parse a JSON dict of ``{feature_name: [timings]}`` directly.

        Args:
            text: JSON string of a feature-timing mapping.

        Returns:
            Validated ``{feature_name: [timings]}`` dict.
        """
        raw: dict[str, Any] = json.loads(text)
        return {k: [float(v) for v in vals] for k, vals in raw.items()}
