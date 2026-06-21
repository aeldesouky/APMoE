# Implementing `ModalityProcessor`

`ModalityProcessor` is the entry point for a single input modality. It
receives raw bytes (or a file-like object) from the HTTP request and converts
them into a `ModalityData` object that the rest of the pipeline can work with.

```
ABC:         apmoe.modality.base.ModalityProcessor
Registry:    apmoe.modality.factory.modality_registry
Config key:  modalities[].processor
```

---

## Abstract interface

```python
from abc import ABC, abstractmethod
from apmoe.core.types import ModalityData

class ModalityProcessor(ABC):

    @property
    @abstractmethod
    def modality_name(self) -> str:
        """Return the canonical modality name declared in config."""

    @abstractmethod
    def validate(self, data: bytes) -> bool:
        """Check that raw input is acceptable before processing.

        Called before preprocess(). If this returns False, the framework
        raises ModalityError and skips this modality for the current request.

        Args:
            data: Raw bytes from the caller (e.g. HTTP JSON serialised to bytes, or CLI file reads).

        Returns:
            True if the input is valid and processing should continue.
            False to signal that this input is unusable.
        """

    @abstractmethod
    def preprocess(self, data: bytes) -> ModalityData:
        """Convert raw bytes into a ModalityData object.

        Only called when validate() returns True. This is the first
        transformation step â€” decode, resize, resample, normalise, etc.

        Args:
            data: Raw bytes from the caller (e.g. HTTP JSON serialised to bytes, or CLI file reads).

        Returns:
            A ModalityData with modality set to this processor's modality name.

        Raises:
            ModalityError: If preprocessing fails for any reason.
        """
```

---

## Minimal implementation

```python
# myproject/processors.py
import io
import numpy as np
from PIL import Image

from apmoe.modality.base import ModalityProcessor
from apmoe.modality.factory import modality_registry
from apmoe.core.types import ModalityData


@modality_registry.register("my_image_processor")
class MyImageProcessor(ModalityProcessor):

    TARGET_SIZE = (224, 224)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def modality_name(self) -> str:
        return "image"

    def validate(self, data: bytes) -> bool:
        # Reject empty payloads; try to open as an image
        if not data:
            return False
        try:
            Image.open(io.BytesIO(data)).verify()
            return True
        except Exception:
            return False

    def preprocess(self, data: bytes) -> ModalityData:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize(self.TARGET_SIZE)
        arr = (np.array(img, dtype=np.float32) / 255.0 - self.MEAN) / self.STD
        tensor = arr.transpose(2, 0, 1)   # HÃ—WÃ—C â†’ CÃ—HÃ—W
        return ModalityData(
            modality=self.modality_name,
            data=tensor,
            metadata={"original_size": img.size},
            source="http_upload",
        )
```

```json
{
  "modalities": [{
    "name":      "image",
    "processor": "myproject.processors.MyImageProcessor",
    "pipeline":  { ... }
  }]
}
```

---

## Responsibilities

| Responsibility | âœ… Yours | âŒ Framework's |
|---|---|---|
| Decode the raw bytes | âœ… | |
| Resize / resample / normalise | âœ… | |
| Set `ModalityData.modality` correctly | âœ… | |
| Call `clean()` on the result | | âŒ (framework) |
| Load pretrained weights | *(none needed)* | |

---

## Contract rules

1. `validate()` must be **fast and side-effect-free** â€” it may be called
   without `preprocess()` following.
2. `preprocess()` is only called after a `True` return from `validate()`.
3. The `modality` field of the returned `ModalityData` must match the
   `name` declared in config for this processor entry.
4. Do **not** run the Cleaner or Anonymizer inside `preprocess()` â€” that is
   the framework's job.
5. Raise `ModalityError` (not a bare exception) if preprocessing fails
   unrecoverably, so the framework can attach modality context.

---

## Config wiring

```json
{
  "name":      "image",
  "processor": "myproject.processors.MyImageProcessor",
  "pipeline": {
    "cleaner":    "myproject.cleaners.ImageCleaner",
    "anonymizer": "myproject.anonymizers.ImageAnonymizer"
  }
}
```

The `processor` value can be either a dotted import path or a short name
previously passed to `@modality_registry.register(...)`. See
[extension-points/index.md](index.md#registration) for details.

---

## Built-in processors

| Class | Dotted path | Modality |
|---|---|---|
| `ImageProcessor` | `apmoe.modality.builtin.image.ImageProcessor` | `"image"` |
| `KeystrokeProcessor` | `apmoe.modality.builtin.keystroke.KeystrokeProcessor` | `"keystroke"` |

