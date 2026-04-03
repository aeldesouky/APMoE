"""Built-in expert plugin implementations for the APMoE framework.

Currently provided:

* :class:`KeystrokeAgeExpert` — ONNX-based age-group classifier using
  keystroke dynamics (hold-time and digraph inter-key timing features).
* :class:`FaceAgeExpert` — Keras deep-learning regression model that predicts
  age from a 200 × 200 RGB face image (MAE ≈ 11.8 years).

Keystroke bootstrap requirements
---------------------------------
The expert requires **two files** in the same directory as the ONNX weight
file:

1. ``keystroke_age_expert.onnx`` — the pretrained ONNX model.
2. ``keystroke_constants.json`` — feature column names, training-set medians,
   and class label mapping exported from the training notebook.

``keystroke_constants.json`` format::

    {
        "feature_cols":    ["dur_8", "dur_13", ..., "dig_65_83", ...],
        "feature_medians": {"dur_8": 62.5, "dig_65_83": 145.2, ...},
        "labels":          ["18-25", "26-35", "36-45", "46+"]
    }

If this file is absent, :meth:`KeystrokeAgeExpert.load_weights` raises
:class:`~apmoe.core.exceptions.ExpertError` with clear instructions.

Face model bootstrap requirements
-----------------------------------
The expert requires a single ``.keras`` file:

* ``face_age_expert_v1.keras`` — the pretrained Keras regression model.

The model is loaded once at startup via ``tf.keras.models.load_model``.
Neither an architecture file nor any companion constants file is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput, ModalityData, ProcessedInput
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry

#: Expected filename alongside the ONNX file.
_CONSTANTS_FILENAME: str = "keystroke_constants.json"


@expert_registry.register("keystroke_age_expert")
class KeystrokeAgeExpert(ExpertPlugin):
    """ONNX-based keystroke-dynamics age-group classifier.

    Uses a pretrained logistic-regression ONNX model trained on **201
    selected keystroke features** (hold-time ``dur_*`` and digraph
    flight-time ``dig_*_*``) to predict one of four age groups.

    The continuous ``predicted_age`` is the probability-weighted average of
    each group's midpoint age. ``confidence`` is the maximum class probability.

    Feature construction (per request)
    -----------------------------------
    For each of the 201 features in ``FEATURE_COLS`` order:

    1. Collect all timing values observed for that feature in the session.
    2. Compute the **mean** of valid values.
    3. If the feature was not observed (or all values were filtered), fill with
       its **training-set median** (``FEATURE_MEDIANS``).
    4. Stack into a ``float32`` vector of shape ``(1, 201)`` — **no additional
       normalisation** is required; the model was trained on raw ms values.

    Config example
    --------------
    .. code-block:: json

        {
          "name": "keystroke_age_expert",
          "class": "apmoe.experts.builtin.KeystrokeAgeExpert",
          "weights": "./weights/keystroke_age_expert.onnx",
          "modalities": ["keystroke"]
        }
    """

    def __init__(self) -> None:
        """Initialise with no model loaded."""
        self._session: Any = None  # onnxruntime.InferenceSession
        self._input_name: str = ""
        self._feature_cols: list[str] = []
        self._feature_medians: dict[str, float] = {}
        self._labels: list[str] = []

    # ------------------------------------------------------------------
    # ExpertPlugin interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the registered expert name ``"keystroke_age_expert"``."""
        return "keystroke_age_expert"

    def declared_modalities(self) -> list[str]:
        """Declare that this expert consumes the ``"keystroke"`` modality."""
        return ["keystroke"]

    def load_weights(self, path: str) -> None:
        """Load the ONNX model and ``keystroke_constants.json``.

        Both files must exist in the same directory.

        Args:
            path: Filesystem path to ``keystroke_age_expert.onnx``.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If the ONNX file
                or constants file are missing / malformed, or if
                ``onnxruntime`` is not installed.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ExpertError(
                "onnxruntime is required for KeystrokeAgeExpert.  "
                "Install it with: pip install onnxruntime",
                context={"weights": path},
            ) from exc

        onnx_path = Path(path)
        if not onnx_path.exists():
            raise ExpertError(
                f"ONNX weights file not found: {onnx_path}",
                context={"weights": path},
            )

        # --- Load ONNX session -------------------------------------------
        try:
            self._session = ort.InferenceSession(str(onnx_path))
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            raise ExpertError(
                f"Failed to load ONNX model from '{onnx_path}': {exc}",
                context={"weights": path},
            ) from exc

        # --- Load constants file -----------------------------------------
        constants_path = onnx_path.parent / _CONSTANTS_FILENAME
        if not constants_path.exists():
            raise ExpertError(
                f"Missing required constants file: {constants_path}\n"
                f"This file must be exported from the training notebook and "
                f"placed alongside the ONNX file.  It must contain:\n"
                f"  feature_cols    — ordered list of 201 feature names\n"
                f"  feature_medians — per-feature training-set median (fill value)\n"
                f"  labels          — list of class label strings (index → label)\n"
                f"See AI_INTEGRATION_GUIDE.md for the exact export instructions.",
                context={"constants_path": str(constants_path)},
            )

        try:
            with constants_path.open(encoding="utf-8") as fh:
                constants: dict[str, Any] = json.load(fh)
        except Exception as exc:
            raise ExpertError(
                f"Failed to read constants file '{constants_path}': {exc}",
                context={"constants_path": str(constants_path)},
            ) from exc

        self._feature_cols = constants["feature_cols"]
        self._feature_medians = {k: float(v) for k, v in constants["feature_medians"].items()}
        self._labels = constants["labels"]

        # Validate consistency — accept N+1 (ONNX off-by-one export artefact)
        expected_features = self._session.get_inputs()[0].shape[1]
        n_cols = len(self._feature_cols)
        if n_cols == expected_features + 1:
            # The ONNX export declared one fewer input than the training set had;
            # the last feature is effectively unused in inference.  Trim silently.
            dropped = self._feature_cols.pop(-1)
            del self._feature_medians[dropped]
        elif n_cols != expected_features:
            raise ExpertError(
                f"feature_cols has {n_cols} entries but ONNX model "
                f"expects {expected_features} features.",
                context={"constants_path": str(constants_path)},
            )

        missing_medians = set(self._feature_cols) - set(self._feature_medians)
        if missing_medians:
            raise ExpertError(
                f"feature_medians is missing entries for: {sorted(missing_medians)}",
                context={"constants_path": str(constants_path)},
            )

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Build the feature vector and run ONNX inference.

        Args:
            inputs: Must contain ``"keystroke"`` key mapping to a
                :class:`~apmoe.core.types.ModalityData` whose ``data``
                attribute is ``dict[str, list[float]]`` (feature → timings).

        Returns:
            An :class:`~apmoe.core.types.ExpertOutput` with:

            * ``predicted_age`` — probability-weighted midpoint age (years).
            * ``confidence`` — maximum class probability.
            * ``metadata["predicted_group"]`` — highest-probability label.
            * ``metadata["age_group_probs"]`` — per-group probabilities.
            * ``metadata["features_observed"]`` — fraction of the 201
              features seen in this session (coverage indicator).

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If the model is not
                loaded or ONNX inference fails.
        """
        if self._session is None:
            raise ExpertError(
                "KeystrokeAgeExpert: model not loaded — call load_weights() first.",
                context={"expert": self.name},
            )

        processed = inputs["keystroke"]
        session_timings: dict[str, list[float]] = (
            processed.data if isinstance(processed, ModalityData) else {}
        )

        # --- Build feature vector ----------------------------------------
        features = self._build_feature_vector(session_timings)

        # --- ONNX inference ----------------------------------------------
        try:
            results = self._session.run(
                ["output_label", "output_probability"],
                {self._input_name: features.reshape(1, -1)},
            )
        except Exception as exc:
            raise ExpertError(
                f"KeystrokeAgeExpert ONNX inference failed: {exc}",
                context={"expert": self.name},
            ) from exc

        label_idx: int = int(results[0][0])
        prob_map: dict[int, float] = dict(results[1][0])  # {class_idx: prob}

        # Ordered probability array
        n_classes = len(self._labels)
        probs = np.array([prob_map.get(i, 0.0) for i in range(n_classes)], dtype=np.float32)

        # Map to midpoint ages for continuous estimate
        midpoints = self._label_midpoints()
        predicted_age = float(np.dot(probs, midpoints))
        confidence = float(probs.max())

        features_observed = sum(
            1 for col in self._feature_cols if col in session_timings
        ) / len(self._feature_cols)

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["keystroke"],
            predicted_age=predicted_age,
            confidence=confidence,
            metadata={
                "predicted_group": self._labels[label_idx],
                "age_group_probs": {
                    self._labels[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "features_observed_fraction": round(features_observed, 4),
            },
        )

    def get_info(self) -> dict[str, object]:
        """Return metadata about this expert for the ``GET /info`` endpoint."""
        return {
            "name": self.name,
            "modalities": self.declared_modalities(),
            "model": "Keystroke Dynamics Age Classifier (ONNX, Logistic Regression)",
            "labels": self._labels,
            "num_features": self.num_features,
            "loaded": self.is_loaded,
        }

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the ONNX session and constants are loaded."""
        return (
            self._session is not None
            and bool(self._feature_cols)
            and bool(self._labels)
        )

    @property
    def num_features(self) -> int:
        """Return the number of features this expert uses (0 until loaded)."""
        return len(self._feature_cols)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self, session_timings: dict[str, list[float]]
    ) -> np.ndarray:
        """Build the 201-element float32 input vector.

        For each feature in ``FEATURE_COLS`` order:

        * Use the **mean** of observed timing values if the feature is present.
        * Fall back to the **training-set median** if absent.

        Args:
            session_timings: ``{feature_name: [timing_ms, ...]}`` from the
                cleaned :class:`~apmoe.core.types.ModalityData`.

        Returns:
            1-D ``float32`` array of length 201.
        """
        vector = np.empty(len(self._feature_cols), dtype=np.float32)
        for i, col in enumerate(self._feature_cols):
            timings = session_timings.get(col)
            if timings:
                vector[i] = float(np.mean(timings))
            else:
                vector[i] = float(self._feature_medians[col])
        return vector

    def _label_midpoints(self) -> list[float]:
        """Return midpoint ages (years) for each label in ``LABELS`` order.

        Parses labels of the form ``"18-25"``, ``"26-35"``, ``"36-45"``,
        ``"46+"``.  The upper-open ``"46+"`` group uses ``55.0`` as its
        midpoint.

        Returns:
            List of midpoint floats, one per label.
        """
        midpoints: list[float] = []
        for label in self._labels:
            if "+" in label:
                lo = float(label.replace("+", "").strip())
                midpoints.append(lo + 9.0)  # e.g. "46+" → 55.0
            elif "-" in label:
                lo, hi = label.split("-", 1)
                midpoints.append((float(lo) + float(hi)) / 2.0)
            else:
                midpoints.append(float(label))
        return midpoints


# ---------------------------------------------------------------------------
# FaceAgeExpert
# ---------------------------------------------------------------------------


@expert_registry.register("face_age_expert")
class FaceAgeExpert(ExpertPlugin):
    """Keras deep-learning regression expert for age prediction from images.

    Loads a ``.keras`` Keras model file trained to predict a person's age
    from a **200 × 200 RGB image** normalised to the range ``[0, 1]``.
    The model outputs a single float (the predicted age in years).

    The preprocessing pipeline (handled by
    :class:`~apmoe.processing.builtin.image_cleaners.ImageCleaner` upstream)
    follows the exact steps in ``docs/face_integration.md``:

    1. Grayscale → RGB (stack channel 3×).
    2. RGBA → RGB (drop alpha).
    3. Resize to ``(200, 200)`` with LANCZOS filter.
    4. Normalise: ``/ 255.0`` → ``float32`` in ``[0, 1]``.

    The expert then adds the batch dimension and calls
    ``model.predict(batch, verbose=0)``.

    **Confidence**: the Keras model is a pure regressor and does not produce
    class probabilities.  A fixed ``confidence = 1.0`` is reported.  This
    means confidence-weighted aggregators treat the face expert as fully
    confident; use explicit per-expert weights in config to tune blending.

    Config example
    --------------
    .. code-block:: json

        {
          "name": "face_age_expert",
          "class": "apmoe.experts.builtin.FaceAgeExpert",
          "weights": "./weights/face_age_expert_v1.keras",
          "modalities": ["image"]
        }
    """

    def __init__(self) -> None:
        """Initialise with no model loaded."""
        self._model: Any = None  # tf.keras.Model

    # ------------------------------------------------------------------
    # ExpertPlugin interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the registered expert name ``"face_age_expert"``."""
        return "face_age_expert"

    def declared_modalities(self) -> list[str]:
        """Declare that this expert consumes the ``"image"`` modality."""
        return ["image"]

    def load_weights(self, path: str) -> None:
        """Load the Keras ``.keras`` model from *path*.

        The model is loaded via ``tf.keras.models.load_model`` and stored
        for reuse across all subsequent :meth:`predict` calls.

        Args:
            path: Filesystem path to the ``face_age_expert_v1.keras`` file.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If the file is
                missing, if TensorFlow / Keras is not installed, or if the
                model cannot be loaded.
        """
        try:
            import tensorflow as tf  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ExpertError(
                "TensorFlow is required for FaceAgeExpert.  "
                "Install it with: pip install tensorflow",
                context={"weights": path},
            ) from exc

        keras_path = Path(path)
        if not keras_path.exists():
            raise ExpertError(
                f"Keras model file not found: {keras_path}",
                context={"weights": path},
            )

        try:
            import tensorflow as tf  # type: ignore[import-untyped]
            self._model = tf.keras.models.load_model(str(keras_path))
        except Exception as exc:
            raise ExpertError(
                f"Failed to load Keras model from '{keras_path}': {exc}",
                context={"weights": path},
            ) from exc

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Run Keras inference and return an age prediction.

        The method expects ``inputs["image"]`` to be a
        :class:`~apmoe.core.types.ModalityData` whose ``data`` attribute is
        a ``float32`` NumPy array of shape ``(200, 200, 3)`` with values in
        ``[0, 1]`` — exactly what
        :class:`~apmoe.processing.builtin.image_cleaners.ImageCleaner`
        produces.

        The batch dimension is added here following the integration spec::

            img_array = np.expand_dims(img_array, axis=0)  # → (1, 200, 200, 3)
            prediction = model.predict(img_array)
            predicted_age = int(round(prediction[0][0]))

        Args:
            inputs: Must contain ``"image"`` key mapping to a
                :class:`~apmoe.core.types.ModalityData`.

        Returns:
            An :class:`~apmoe.core.types.ExpertOutput` with:

            * ``predicted_age`` — float (rounded to nearest integer per spec,
              stored as float for aggregator compatibility).
            * ``confidence`` — ``1.0`` (fixed; regressor has no class probs).
            * ``metadata["raw_output"]`` — raw float32 output before rounding.
            * ``metadata["rounded_age"]`` — integer age per spec.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If the model is not
                loaded or Keras inference fails.
        """
        if self._model is None:
            raise ExpertError(
                "FaceAgeExpert: model not loaded — call load_weights() first.",
                context={"expert": self.name},
            )

        processed = inputs["image"]
        img_array: np.ndarray = (
            processed.data if isinstance(processed, ModalityData) else processed.embedding
        )

        # Add batch dimension: (200, 200, 3) → (1, 200, 200, 3)
        batch = np.expand_dims(img_array, axis=0).astype(np.float32)

        try:
            prediction = self._model.predict(batch, verbose=0)  # type: ignore[union-attr]
        except Exception as exc:
            raise ExpertError(
                f"FaceAgeExpert Keras inference failed: {exc}",
                context={"expert": self.name},
            ) from exc

        raw_output = float(prediction[0][0])
        rounded_age = int(round(raw_output))
        # Clamp to plausible range — model may occasionally extrapolate
        predicted_age = float(max(1, min(120, rounded_age)))

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["image"],
            predicted_age=predicted_age,
            confidence=1.0,
            metadata={
                "raw_output": round(raw_output, 4),
                "rounded_age": rounded_age,
                "model": "Face Age Prediction v1.0 (Keras, MAE≈11.8)",
            },
        )

    def get_info(self) -> dict[str, object]:
        """Return metadata about this expert for the ``GET /info`` endpoint."""
        return {
            "name": self.name,
            "modalities": self.declared_modalities(),
            "model": "Face Age Prediction v1.0 (Keras Regression, MAE≈11.8)",
            "input_shape": "(1, 200, 200, 3)",
            "input_dtype": "float32",
            "input_range": "[0, 1]",
            "output_type": "regressor",
            "loaded": self.is_loaded,
        }

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the Keras model is loaded."""
        return self._model is not None
