# Implementing `ExpertPlugin`

An `ExpertPlugin` is a pretrained age-prediction model that consumes one or
more modalities and produces an independent age estimate. The framework loads
its weights once at bootstrap and calls `predict()` for every request.

```
ABC:         apmoe.experts.base.ExpertPlugin
Registry:    apmoe.experts.registry.expert_registry
Config key:  experts[].class
```

---

## Abstract interface

```python
from abc import ABC, abstractmethod
from apmoe.core.types import ProcessedInput, ExpertOutput

class ExpertPlugin(ABC):

    @abstractmethod
    def declared_modalities(self) -> list[str]:
        """Return the list of modality names this expert requires.

        Called at bootstrap to validate the config and to build the
        dispatch map. The names must match modalities declared in config.

        Returns:
            A non-empty list of modality names, e.g. ["image"] or
            ["image", "keystroke"].
        """

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load pretrained weights from the filesystem.

        Called exactly once during APMoEApp bootstrap â€” never during
        inference. The framework resolves `path` from the config's
        `experts[].weights` field.

        Args:
            path: Absolute or CWD-relative path to the weight file
                  (.pt, .onnx, .pkl, etc.).

        Raises:
            ExpertError: If the file cannot be opened or parsed.
        """

    @abstractmethod
    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Run inference and return a single age prediction.

        Args:
            inputs: Dict mapping modality name â†’ ProcessedInput for every
                    modality in declared_modalities(). A ProcessedInput is
                    either a ModalityData (no embedder configured) or an
                    EmbeddingResult (embedder configured for that modality).

        Returns:
            An ExpertOutput with predicted_age and confidence in [0, 1].

        Raises:
            ExpertError: If inference fails unrecoverably.
        """

    def get_info(self) -> dict:
        """Return metadata about this expert (optional override).

        Exposed via the GET /info endpoint. Override to add architecture
        details, model version, training dataset, etc.

        Returns:
            A JSON-serialisable dict. Default implementation returns the
            class name and declared modalities.
        """
        return {
            "class": type(self).__qualname__,
            "modalities": self.declared_modalities(),
        }
```

---

## Single-modality expert

```python
# myproject/experts.py
import torch
import torch.nn as nn

from apmoe.experts.base import ExpertPlugin
from apmoe.experts.registry import expert_registry
from apmoe.core.types import ProcessedInput, EmbeddingResult, ExpertOutput
from apmoe.core.exceptions import ExpertError


@expert_registry.register("image_age_expert")
class ImageAgeExpert(ExpertPlugin):
    """Predicts age from a pre-computed MobileNet embedding."""

    def __init__(self) -> None:
        self.model: nn.Module | None = None

    @property
    def name(self) -> str:
        return "image_age_expert"

    def declared_modalities(self) -> list[str]:
        return ["image"]

    def load_weights(self, path: str) -> None:
        try:
            self.model = torch.load(path, map_location="cpu")
            self.model.eval()
        except Exception as exc:
            raise ExpertError(
                f"Failed to load weights: {exc}",
                context={"expert_name": self.name, "weights_path": path},
            ) from exc

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        assert self.model is not None, "load_weights() not called"

        inp = inputs["image"]

        # Accept either a pre-computed embedding or raw tensor
        if isinstance(inp, EmbeddingResult):
            features = torch.as_tensor(inp.embedding).unsqueeze(0)
        else:
            # Run the CNN on the raw tensor
            features = self.model.features(
                torch.as_tensor(inp.data).unsqueeze(0)
            )

        with torch.inference_mode():
            age_logit = self.model.head(features)
        age = float(age_logit.squeeze())

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["image"],
            predicted_age=age,
            confidence=0.88,
        )
```

---

## Multi-modal expert

A multi-modal expert declares two or more modalities. The framework dispatches
all of them together. The expert is responsible for combining them internally.

```python
@expert_registry.register("image_keystroke_age_expert")
class ImageKeystrokeAgeExpert(ExpertPlugin):
    """Fuses image and keystroke features for a joint age estimate."""

    def __init__(self) -> None:
        self.keystroke_branch: nn.Module | None = None
        self.image_branch: nn.Module | None = None
        self.fusion_head: nn.Module | None = None

    @property
    def name(self) -> str:
        return "image_keystroke_age_expert"

    def declared_modalities(self) -> list[str]:
        return ["image", "keystroke"]   # framework dispatches both

    def load_weights(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.keystroke_branch = checkpoint["keystroke_branch"]
        self.image_branch = checkpoint["image_branch"]
        self.fusion_head = checkpoint["fusion_head"]
        for m in (self.keystroke_branch, self.image_branch, self.fusion_head):
            m.eval()

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        image_feat = self._extract(inputs["image"], self.image_branch)
        key_feat = self._extract(inputs["keystroke"], self.keystroke_branch)

        with torch.inference_mode():
            age = float(self.fusion_head(torch.cat([image_feat, key_feat], dim=-1)))

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["image", "keystroke"],
            predicted_age=age,
            confidence=0.91,
        )

    def _extract(self, inp: ProcessedInput, branch: nn.Module) -> torch.Tensor:
        if isinstance(inp, EmbeddingResult):
            return torch.as_tensor(inp.embedding).unsqueeze(0)
        return branch(torch.as_tensor(inp.data).unsqueeze(0))
```

---

## Using `extra` config parameters

Any keys in the expert's config block beyond the standard four (`name`,
`class`, `weights`, `modalities`) land in `ExpertConfig.extra` and are
passed to the expert's constructor. Access them via `__init__`:

```json
{
  "name":        "calibrated_expert",
  "class":       "myproject.experts.CalibratedExpert",
  "weights":     "./weights/calibrated.pt",
  "modalities":  ["image"],
  "temperature": 1.5,
  "bias":        -0.3
}
```

```python
class CalibratedExpert(ExpertPlugin):
    def __init__(self, temperature: float = 1.0, bias: float = 0.0) -> None:
        self.temperature = temperature
        self.bias = bias
        self.model = None

    # ... rest of the implementation
```

---

## Contract rules

1. `declared_modalities()` is a **classmethod** â€” the framework may call it
   before instantiation to validate config.
2. `load_weights()` is called **once** at bootstrap. Store the loaded model on
   `self`. Do not load weights in `predict()`.
3. `predict()` must be **thread-safe** if `workers > 1` in serving config.
   Use `torch.inference_mode()` and avoid shared mutable state.
4. Return `confidence` in `[0.0, 1.0]`, or `-1.0` if the expert does not
   report a score (e.g. a regressor with no uncertainty head). Construction of
   `ExpertOutput` validates this and raises `ValueError` otherwise.
5. The `expert_name` in the returned `ExpertOutput` should match the `name`
   from config so aggregators can match weights correctly.

---

## Config wiring

```json
{
  "experts": [
    {
      "name":       "image_age_expert",
      "class":      "myproject.experts.ImageAgeExpert",
      "weights":    "./weights/image.pt",
      "modalities": ["image"]
    },
    {
      "name":       "image_keystroke_age_expert",
      "class":      "myproject.experts.ImageKeystrokeAgeExpert",
      "weights":    "./weights/image_keystroke_fusion.pt",
      "modalities": ["image", "keystroke"]
    }
  ]
}
```

---

## Graceful degradation

If a modality is absent from the request (for example, no image payload was sent), the
framework automatically skips every expert that lists that modality as
required. The skipped expert names appear in `Prediction.skipped_experts`.

Expert outputs that arrive from available experts are still aggregated normally.

---

## Built-in experts

| Class | Path | Modalities |
|---|---|---|
| `FaceAgeExpert` | `apmoe.experts.builtin.FaceAgeExpert` | `["image"]` |
| `KeystrokeAgeExpert` | `apmoe.experts.builtin.KeystrokeAgeExpert` | `["keystroke"]` |
| `RemoteExpert` | `apmoe.experts.remote.RemoteExpert` | Configured per expert |
| `LMStudioExpert` | `apmoe.experts.providers.lmstudio.LMStudioExpert` | Usually `["image"]` |

