"""Built-in cleaner strategies for the APMoE processing pipeline.

Currently provided:

* :class:`KeystrokeCleaner` — filters invalid/outlier timing values from
  the per-feature timing lists produced by
  :class:`~apmoe.modality.builtin.keystroke.KeystrokeProcessor`.
"""

from __future__ import annotations

from apmoe.core.types import ModalityData
from apmoe.processing.base import CleanerStrategy, cleaner_registry

#: Minimum valid inter-key timing in milliseconds (exclusive).
_MIN_TIMING_MS: float = 0.0

#: Maximum plausible inter-key timing in milliseconds.
#: Intervals above 10 seconds almost certainly represent session breaks
#: or logging artefacts rather than genuine consecutive keystrokes.
_MAX_TIMING_MS: float = 10_000.0


@cleaner_registry.register("keystroke_cleaner")
class KeystrokeCleaner(CleanerStrategy):
    """Remove invalid timing values from keystroke feature lists.

    Operates on :class:`~apmoe.core.types.ModalityData` produced by
    :class:`~apmoe.modality.builtin.keystroke.KeystrokeProcessor`, where
    ``data`` is a ``dict[str, list[float]]`` mapping feature names to
    lists of raw timing values (ms).

    Each timing value is filtered independently:

    * ``timing <= 0`` — physically impossible; sensor or logging error.
    * ``timing > 10 000`` — longer than 10 seconds; almost certainly a
      session break captured as a single digraph.

    Features whose entire timing list is removed are dropped from the dict.
    The number of removed values is recorded in
    ``metadata["removed_timings"]``.

    Registered as ``"keystroke_cleaner"`` in
    :data:`~apmoe.processing.base.cleaner_registry`.
    """

    def clean(self, data: ModalityData) -> ModalityData:
        """Filter invalid timing values from each feature list.

        Args:
            data: :class:`~apmoe.core.types.ModalityData` from the
                keystroke processor.  ``data.data`` must be a
                ``dict[str, list[float]]``.

        Returns:
            A new :class:`~apmoe.core.types.ModalityData` (via
            :meth:`~apmoe.core.types.ModalityData.with_data`) containing
            only valid timing values, with updated metadata.
        """
        raw: dict[str, list[float]] = data.data
        cleaned: dict[str, list[float]] = {}
        removed = 0

        for feat, timings in raw.items():
            valid = [t for t in timings if _MIN_TIMING_MS < t <= _MAX_TIMING_MS]
            removed += len(timings) - len(valid)
            if valid:
                cleaned[feat] = valid

        updated_meta = dict(data.metadata)
        updated_meta["num_features_observed"] = len(cleaned)
        updated_meta["num_raw_timings"] = sum(len(v) for v in cleaned.values())
        updated_meta["removed_timings"] = removed

        result = data.with_data(cleaned)
        result.metadata.update(updated_meta)
        return result
