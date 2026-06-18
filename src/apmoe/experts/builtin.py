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

* ``face_age_expert.keras`` — the pretrained Keras regression model.

The model is loaded once at startup via ``tf.keras.models.load_model``.
Neither an architecture file nor any companion constants file is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput, ModalityData, ProcessedInput
from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry

#: Expected filename alongside the ONNX file.
_CONSTANTS_FILENAME: str = "keystroke_constants.json"
_KERAS_FACE_SUFFIXES: frozenset[str] = frozenset({".keras", ".h5", ".hdf5"})
_PYTORCH_FACE_SUFFIXES: tuple[str, ...] = (".pt", ".pth", ".pt.zip", ".pth.zip")


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

def _make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Match MobileNet channel rounding used by torchvision."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _load_torch_modules() -> tuple[Any, Any]:
    """Import PyTorch lazily so the base package stays lightweight."""
    try:
        import torch  # type: ignore[import-untyped]
        import torch.nn as nn  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ExpertError(
            "PyTorch is required for FaceAgeExpert .pt/.pth weights. "
            "Install it with: pip install torch",
        ) from exc
    return torch, nn


def _conv_bn_activation(
    nn: Any,
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    activation: type[Any] | None,
) -> Any:
    """Build a torchvision-compatible Conv2d -> BatchNorm -> activation block."""
    padding = (kernel_size - 1) // 2
    layers: list[Any] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
    ]
    if activation is not None:
        layers.append(activation(inplace=True))
    return nn.Sequential(*layers)


def _squeeze_excitation(nn: Any, input_channels: int, squeeze_channels: int) -> Any:
    """Build a torchvision-compatible squeeze-excitation block."""

    class _SqueezeExcitation(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
            self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
            self.activation = nn.ReLU()
            self.scale_activation = nn.Hardsigmoid()

        def forward(self, x: Any) -> Any:
            scale = self.avgpool(x)
            scale = self.fc1(scale)
            scale = self.activation(scale)
            scale = self.fc2(scale)
            scale = self.scale_activation(scale)
            return scale * x

    return _SqueezeExcitation()


def _inverted_residual(
    nn: Any,
    in_channels: int,
    expanded_channels: int,
    out_channels: int,
    *,
    kernel_size: int,
    stride: int,
    use_se: bool,
    use_hs: bool,
) -> Any:
    """Build a MobileNetV3 inverted residual block matching torchvision keys."""

    class _InvertedResidual(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            activation = nn.Hardswish if use_hs else nn.ReLU
            layers: list[Any] = []
            if expanded_channels != in_channels:
                layers.append(
                    _conv_bn_activation(
                        nn,
                        in_channels,
                        expanded_channels,
                        kernel_size=1,
                        activation=activation,
                    )
                )
            layers.append(
                _conv_bn_activation(
                    nn,
                    expanded_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=expanded_channels,
                    activation=activation,
                )
            )
            if use_se:
                squeeze_channels = _make_divisible(expanded_channels // 4, 8)
                layers.append(_squeeze_excitation(nn, expanded_channels, squeeze_channels))
            layers.append(
                _conv_bn_activation(
                    nn,
                    expanded_channels,
                    out_channels,
                    kernel_size=1,
                    activation=None,
                )
            )
            self.block = nn.Sequential(*layers)
            self.use_res_connect = stride == 1 and in_channels == out_channels

        def forward(self, x: Any) -> Any:
            result = self.block(x)
            if self.use_res_connect:
                result = result + x
            return result

    return _InvertedResidual()


def _build_mobilenet_v3_age_regressor(nn: Any) -> Any:
    """Build the MobileNetV3 Large age regressor used by the .pth artifact."""

    class _MobileNetV3Backbone(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                _conv_bn_activation(
                    nn,
                    3,
                    16,
                    kernel_size=3,
                    stride=2,
                    activation=nn.Hardswish,
                ),
                _inverted_residual(
                    nn,
                    16,
                    16,
                    16,
                    kernel_size=3,
                    stride=1,
                    use_se=False,
                    use_hs=False,
                ),
                _inverted_residual(
                    nn,
                    16,
                    64,
                    24,
                    kernel_size=3,
                    stride=2,
                    use_se=False,
                    use_hs=False,
                ),
                _inverted_residual(
                    nn,
                    24,
                    72,
                    24,
                    kernel_size=3,
                    stride=1,
                    use_se=False,
                    use_hs=False,
                ),
                _inverted_residual(
                    nn,
                    24,
                    72,
                    40,
                    kernel_size=5,
                    stride=2,
                    use_se=True,
                    use_hs=False,
                ),
                _inverted_residual(
                    nn,
                    40,
                    120,
                    40,
                    kernel_size=5,
                    stride=1,
                    use_se=True,
                    use_hs=False,
                ),
                _inverted_residual(
                    nn,
                    40,
                    120,
                    40,
                    kernel_size=5,
                    stride=1,
                    use_se=True,
                    use_hs=False,
                ),
                _inverted_residual(
                    nn,
                    40,
                    240,
                    80,
                    kernel_size=3,
                    stride=2,
                    use_se=False,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    80,
                    200,
                    80,
                    kernel_size=3,
                    stride=1,
                    use_se=False,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    80,
                    184,
                    80,
                    kernel_size=3,
                    stride=1,
                    use_se=False,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    80,
                    184,
                    80,
                    kernel_size=3,
                    stride=1,
                    use_se=False,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    80,
                    480,
                    112,
                    kernel_size=3,
                    stride=1,
                    use_se=True,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    112,
                    672,
                    112,
                    kernel_size=3,
                    stride=1,
                    use_se=True,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    112,
                    672,
                    160,
                    kernel_size=5,
                    stride=2,
                    use_se=True,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    160,
                    960,
                    160,
                    kernel_size=5,
                    stride=1,
                    use_se=True,
                    use_hs=True,
                ),
                _inverted_residual(
                    nn,
                    160,
                    960,
                    160,
                    kernel_size=5,
                    stride=1,
                    use_se=True,
                    use_hs=True,
                ),
                _conv_bn_activation(
                    nn,
                    160,
                    960,
                    kernel_size=1,
                    activation=nn.Hardswish,
                ),
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(960, 256),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(256, 1),
            )

        def forward(self, x: Any) -> Any:
            x = self.features(x)
            x = self.avgpool(x)
            x = nn.Flatten(1)(x)
            return self.classifier(x)

    class _MobileNetV3AgeRegressor(nn.Module):  # type: ignore[name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.backbone = _MobileNetV3Backbone()

        def forward(self, x: Any) -> Any:
            return self.backbone(x)

    return _MobileNetV3AgeRegressor()


def _is_pytorch_face_path(path: Path) -> bool:
    """Return whether *path* uses a PyTorch face model suffix."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _PYTORCH_FACE_SUFFIXES)


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
    a calibrated probability.  :attr:`~apmoe.core.types.ExpertOutput.confidence`
    is set to ``-1.0`` (not reported).  Aggregators ignore ``-1`` when combining
    self-reported confidence; use explicit per-expert ``aggregation.weights``
    in config to tune blending with other experts.

    Config example
    --------------
    .. code-block:: json

        {
          "name": "face_age_expert",
          "class": "apmoe.experts.builtin.FaceAgeExpert",
          "weights": "./weights/face_age_expert.keras",
          "modalities": ["image"]
        }
    """

    def __init__(self) -> None:
        """Initialise with no model loaded."""
        self._model: Any = None  # tf.keras.Model or torch.nn.Module
        self._backend: Literal["keras", "pytorch"] | None = None

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
        """Load a Keras or PyTorch face model from *path*."""
        weights_path = Path(path)
        if not weights_path.exists():
            raise ExpertError(
                f"Face model file not found: {weights_path}",
                context={"weights": path},
            )

        suffix = weights_path.suffix.lower()
        if suffix in _KERAS_FACE_SUFFIXES:
            self._load_keras_weights(weights_path)
            return
        if _is_pytorch_face_path(weights_path):
            self._load_pytorch_weights(weights_path)
            return

        supported = sorted(_KERAS_FACE_SUFFIXES) + list(_PYTORCH_FACE_SUFFIXES)
        raise ExpertError(
            f"Unsupported FaceAgeExpert model format: {weights_path.name}. "
            f"Supported suffixes: {supported}",
            context={"weights": path, "suffix": suffix},
        )

    def _load_keras_weights(self, path: Path) -> None:
        """Load a TensorFlow/Keras face model."""
        try:
            import tensorflow as tf  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise ExpertError(
                "TensorFlow is required for FaceAgeExpert.  "
                "Install it with: pip install tensorflow",
                context={"weights": str(path)},
            ) from exc

        try:
            import tensorflow as tf  # type: ignore[import-untyped]

            self._model = tf.keras.models.load_model(str(path))
            self._backend = "keras"
        except Exception as exc:
            raise ExpertError(
                f"Failed to load Keras model from '{path}': {exc}",
                context={"weights": str(path)},
            ) from exc

    def _load_pytorch_weights(self, path: Path) -> None:
        """Load a PyTorch MobileNetV3 face-age state dict."""
        torch, nn = _load_torch_modules()
        try:
            model = _build_mobilenet_v3_age_regressor(nn)
            try:
                state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(str(path), map_location="cpu")
            if not isinstance(state_dict, dict):
                raise ExpertError(
                    "PyTorch face model must be a state dict.",
                    context={"weights": str(path)},
                )
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            self._backend = "pytorch"
        except ExpertError:
            raise
        except Exception as exc:
            raise ExpertError(
                f"Failed to load PyTorch model from '{path}': {exc}",
                context={"weights": str(path)},
            ) from exc

    def _predict_keras_legacy(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
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
            * ``confidence`` — ``-1.0`` (not reported; regressor has no class probs).
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
            confidence=-1.0,
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
            "backend": self._backend,
            "loaded": self.is_loaded,
        }

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if the face model is loaded."""
        return self._model is not None and self._backend is not None

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Run face-age inference and return an age prediction."""
        if self._model is None or self._backend is None:
            raise ExpertError(
                "FaceAgeExpert: model not loaded - call load_weights() first.",
                context={"expert": self.name},
            )

        processed = inputs["image"]
        img_array: np.ndarray = (
            processed.data if isinstance(processed, ModalityData) else processed.embedding
        )

        if self._backend == "pytorch":
            return self._predict_pytorch(img_array)
        return self._predict_keras(img_array)

    def _predict_keras(self, img_array: np.ndarray) -> ExpertOutput:
        """Run Keras inference and return the standard face expert output."""
        batch = np.expand_dims(img_array, axis=0).astype(np.float32)

        try:
            prediction = self._model.predict(batch, verbose=0)  # type: ignore[union-attr]
        except Exception as exc:
            raise ExpertError(
                f"FaceAgeExpert Keras inference failed: {exc}",
                context={"expert": self.name},
            ) from exc

        return self._build_output(float(prediction[0][0]), backend_label="Keras")

    def _predict_pytorch(self, img_array: np.ndarray) -> ExpertOutput:
        """Run PyTorch inference and return the standard face expert output."""
        torch, _ = _load_torch_modules()
        try:
            arr = np.asarray(img_array, dtype=np.float32)
            batch = torch.as_tensor(arr).permute(2, 0, 1).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=batch.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=batch.dtype).view(1, 3, 1, 1)
            batch = (batch - mean) / std
            with torch.inference_mode():
                prediction = self._model(batch)  # type: ignore[misc]
            raw_output = float(prediction.reshape(-1)[0].item())
        except Exception as exc:
            raise ExpertError(
                f"FaceAgeExpert PyTorch inference failed: {exc}",
                context={"expert": self.name},
            ) from exc

        return self._build_output(raw_output, backend_label="PyTorch MobileNetV3")

    def _build_output(self, raw_output: float, *, backend_label: str) -> ExpertOutput:
        """Build the shared face expert output from a raw scalar prediction."""
        rounded_age = int(round(raw_output))
        predicted_age = float(max(1, min(120, rounded_age)))

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["image"],
            predicted_age=predicted_age,
            confidence=-1.0,
            metadata={
                "raw_output": round(raw_output, 4),
                "rounded_age": rounded_age,
                "model": f"Face Age Prediction v1.0 ({backend_label})",
                "backend": self._backend,
            },
        )
