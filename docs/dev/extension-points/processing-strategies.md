# Processing Strategies

The processing pipeline runs three strategies in sequence for each modality:

```
ModalityData
    │
    ▼  CleanerStrategy.clean()       ← required
ModalityData  (cleaned)
    │
    ▼  AnonymizerStrategy.anonymize() ← required
ModalityData  (anonymised)
    │
    ▼  EmbedderStrategy.embed()      ← optional (omit pipeline.embedder to skip)
EmbeddingResult  OR  ModalityData    (ProcessedInput)
```

Each strategy is independent, swappable per-modality via config, and must
not mutate its input — use `ModalityData.with_data()` to produce a copy.

---

## `CleanerStrategy`

```
ABC:         apmoe.processing.base.CleanerStrategy
Registry:    apmoe.processing.base.cleaner_registry
Config key:  modalities[].pipeline.cleaner
```

### Interface

```python
from abc import ABC, abstractmethod
from apmoe.core.types import ModalityData

class CleanerStrategy(ABC):

    @abstractmethod
    def clean(self, data: ModalityData) -> ModalityData:
        """Remove noise, artefacts, or invalid samples.

        Args:
            data: The preprocessed ModalityData from ModalityProcessor.

        Returns:
            A new ModalityData with cleaned payload. Use data.with_data()
            to preserve all metadata fields.

        Raises:
            ModalityError: If cleaning fails unrecoverably.
        """
```

### Example

```python
import numpy as np
from apmoe.processing.base import CleanerStrategy, cleaner_registry
from apmoe.core.types import ModalityData


@cleaner_registry.register("audio_cleaner")
class AudioCleaner(CleanerStrategy):
    """Trim silence and clip extreme amplitudes."""

    CLIP_DB = 60.0

    def clean(self, data: ModalityData) -> ModalityData:
        waveform: np.ndarray = data.data
        # Normalise amplitude
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        # Trim leading/trailing silence
        waveform = np.trim_zeros(waveform)
        return data.with_data(waveform)
```

---

## `AnonymizerStrategy`

```
ABC:         apmoe.processing.base.AnonymizerStrategy
Registry:    apmoe.processing.base.anonymizer_registry
Config key:  modalities[].pipeline.anonymizer
```

Anonymisers remove or obfuscate personally-identifiable information.
They run **after** the cleaner and receive the cleaned `ModalityData`.

### Interface

```python
class AnonymizerStrategy(ABC):

    @abstractmethod
    def anonymize(self, data: ModalityData) -> ModalityData:
        """Remove or obfuscate personally-identifiable information.

        Args:
            data: Cleaned ModalityData from CleanerStrategy.

        Returns:
            A new ModalityData with PII removed or perturbed.

        Raises:
            ModalityError: If anonymisation fails unrecoverably.
        """
```

### Example

```python
import numpy as np
from apmoe.processing.base import AnonymizerStrategy, anonymizer_registry
from apmoe.core.types import ModalityData


@anonymizer_registry.register("voice_anonymizer")
class VoiceAnonymizer(AnonymizerStrategy):
    """Shift pitch slightly to obscure speaker identity."""

    SHIFT_SEMITONES = 2

    def anonymize(self, data: ModalityData) -> ModalityData:
        waveform = data.data
        # Pitch shift via resampling approximation (simplified)
        factor = 2 ** (self.SHIFT_SEMITONES / 12)
        shifted = np.interp(
            np.arange(0, len(waveform), factor),
            np.arange(len(waveform)),
            waveform,
        ).astype(waveform.dtype)
        return data.with_data(shifted)
```

---

## `EmbedderStrategy`

```
ABC:         apmoe.processing.base.EmbedderStrategy
Registry:    apmoe.processing.base.embedder_registry
Config key:  modalities[].pipeline.embedder   (optional — omit to skip)
```

The Embedder step is **optional**. When `pipeline.embedder` is absent from
config for a modality, its processing chain ends after the Anonymizer and
experts receive a `ModalityData`. When present, experts receive an
`EmbeddingResult`.

This lets experts that do their own feature extraction (e.g. a CNN that
operates on raw image tensors) skip the embedding step, while experts that
expect a pre-computed feature vector use it.

### Interface

```python
from apmoe.core.types import ModalityData, EmbeddingResult

class EmbedderStrategy(ABC):

    @abstractmethod
    def embed(self, data: ModalityData) -> EmbeddingResult:
        """Compute a dense feature vector from preprocessed data.

        Args:
            data: Anonymised ModalityData from AnonymizerStrategy.

        Returns:
            An EmbeddingResult containing the feature vector and metadata.

        Raises:
            ModalityError: If embedding fails unrecoverably.
        """
```

### Example

```python
import numpy as np
import torch
from apmoe.processing.base import EmbedderStrategy, embedder_registry
from apmoe.core.types import ModalityData, EmbeddingResult


@embedder_registry.register("mobilenet_embedder")
class MobileNetEmbedder(EmbedderStrategy):
    """Extract 1280-d features using a pretrained MobileNetV3-Small."""

    def __init__(self) -> None:
        import torchvision.models as models
        backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
        # Strip the classifier head — keep only the feature extractor
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()

    def embed(self, data: ModalityData) -> EmbeddingResult:
        tensor = torch.as_tensor(data.data).unsqueeze(0)   # add batch dim
        with torch.inference_mode():
            features = self.model(tensor).squeeze()        # (1280,)
        embedding = features.numpy()
        return EmbeddingResult(
            modality=data.modality,
            embedding=embedding,
            metadata={"model": "mobilenet_v3_small", "layer": "features"},
        )
```

---

## Shared contract rules for all three strategies

1. **Never mutate the input.** Always return a new `ModalityData` using
   `data.with_data(new_payload)` (for Cleaner and Anonymizer) or a new
   `EmbeddingResult` (for Embedder).
2. **Raise `ModalityError`** (not bare exceptions) if the step fails — the
   framework catches and wraps, but explicit raises produce better context.
3. **Preserve `metadata`** — `with_data()` does this automatically. If you
   add keys, add them to the copy's `metadata` dict, not the original's.
4. **Keep it stateless** where possible. Strategies may hold pretrained model
   state (e.g. `MobileNetEmbedder`) but should not accumulate per-request
   state.

---

## Config wiring

```json
{
  "name": "visual",
  "processor": "myproject.processors.MyVisualProcessor",
  "pipeline": {
    "cleaner":    "myproject.cleaners.ImageCleaner",
    "anonymizer": "myproject.anonymizers.FaceAnonymizer",
    "embedder":   "myproject.embedders.MobileNetEmbedder"
  }
}
```

Each value is a dotted import path or a short registered name.  
`embedder` may be **omitted entirely** to skip the embedding step.

---

## Built-in implementations (Phase 6)

### Cleaners

| Class | Path | Modality |
|---|---|---|
| `ImageCleaner` | `apmoe.processing.builtin.cleaners.ImageCleaner` | `"visual"` |
| `AudioCleaner` | `apmoe.processing.builtin.cleaners.AudioCleaner` | `"audio"` |
| `EEGCleaner` | `apmoe.processing.builtin.cleaners.EEGCleaner` | `"eeg"` |

### Anonymisers

| Class | Path | Technique |
|---|---|---|
| `FaceAnonymizer` | `apmoe.processing.builtin.anonymizers.FaceAnonymizer` | Face detection + Gaussian blur |
| `VoiceAnonymizer` | `apmoe.processing.builtin.anonymizers.VoiceAnonymizer` | Pitch perturbation |
| `EEGAnonymizer` | `apmoe.processing.builtin.anonymizers.EEGAnonymizer` | Channel dropout + noise |

### Embedders

| Class | Path | Output dim | Modality |
|---|---|---|---|
| `MobileNetEmbedder` | `apmoe.processing.builtin.embedders.MobileNetEmbedder` | 1280 | `"visual"` |
| `MelSpectrogramEmbedder` | `apmoe.processing.builtin.embedders.MelSpectrogramEmbedder` | configurable | `"audio"` |
| `EEGEmbedder` | `apmoe.processing.builtin.embedders.EEGEmbedder` | configurable | `"eeg"` |
