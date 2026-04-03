# AI Integration Guide — APMoE Framework

**For:** AI / ML Team  
**Project:** APMoE — Age Prediction using Mixture of Experts  
**Purpose:** This document defines exactly what the framework integration layer needs from each model. Read it in full, fill in every item marked **NEEDED**, and send the complete package back. Items marked **HAVE** are already confirmed and do not need to be re-sent.

---

## How the Framework Consumes Models

APMoE is an inference-only orchestration framework. It loads each model's weight file once at startup, runs the preprocessing pipeline you specify, calls the model, and combines results from all experts into a final prediction.

Each model maps to one **Expert Plugin**. The framework calls two methods:

1. `load_weights(path)` — loads the model file once at startup
2. `predict(inputs)` — runs the full pipeline once per request

Everything below defines what is needed to implement those two methods correctly.

---

## Model 1: Keystroke Age Expert

### What We Already Have

| Item | Status | Detail |
|------|--------|--------|
| ONNX weight file | **HAVE** | `Keystroke Age Prediction v1.0 [LR 70% accuracy].onnx` |
| Model architecture | **HAVE** | `SVMClassifier` (scikit-learn via ONNX) |
| Input name | **HAVE** | `float_input` |
| Input shape | **HAVE** | `(batch, 201)` — 201 float32 features per sample |
| Output 1 | **HAVE** | `output_label` — shape `(batch,)`, dtype `int64`, predicted age group class index |
| Output 2 | **HAVE** | `output_probability` — per-class probability map |
| Raw file format | **HAVE** | IKDD format: metadata line + `key1-key2,v1,v2,...` lines |
| Feature types | **HAVE** | `dur_{key1}` (hold time, key2==0) and `dig_{key1}_{key2}` (digraph, key2≠0) |
| Fill strategy | **HAVE** | Missing features filled with **training median**, not zero |

### What Is Still Needed — Keystroke Model

The ONNX file contains the trained weights but **not** the three constants that were in your training notebook. Without these, inference cannot be reproduced correctly. Please export them directly from your training notebook using the instructions below and send them as a Python file.

---

#### NEEDED 1 — `FEATURE_COLS`: Ordered Feature Name List

This is the ordered list of the 201 column names the model was trained on. In your notebook it lives in `X_imp.columns`.

**How to export from your notebook:**

```python
import json
print(json.dumps(X_imp.columns.tolist(), indent=2))
```

**What to send** — a Python file `keystroke_constants.py` containing:

```python
FEATURE_COLS = [
    "dur_8",
    "dur_13",
    "dur_32",
    "dig_65_83",
    # ... all 201 names in exact training order
]
```

This list determines which features map to which position in the 201-element input vector. If this list is wrong or out of order, every prediction will be incorrect.

---

#### NEEDED 2 — `FEATURE_MEDIANS`: Per-Feature Fill Values

When a user's session does not contain a particular key pair, the framework must fill that feature with the value seen during training — the training-set median, not zero. In your notebook this is `X_imp.median()`.

**How to export from your notebook:**

```python
import json
print(json.dumps(X_imp.median().to_dict(), indent=2))
```

**What to send** — add to `keystroke_constants.py`:

```python
FEATURE_MEDIANS = {
    "dur_8":    62.5,
    "dur_13":   98.0,
    "dig_65_83": 145.2,
    # ... one entry per feature in FEATURE_COLS
}
```

This dict must have exactly one key per entry in `FEATURE_COLS`. Every key in `FEATURE_COLS` must appear here.

---

#### NEEDED 3 — `LABELS`: Age Group Class Index Map

The model outputs an integer class index. The framework needs to know which string label each integer maps to. In your notebook this is `le.classes_`.

**How to export from your notebook:**

```python
import json
print(json.dumps(le.classes_.tolist(), indent=2))
```

**What to send** — add to `keystroke_constants.py`:

```python
# Index in this list = class index output by the model
# e.g. LABELS[0] = the label for class 0
LABELS = [
    "18-25",
    "26-35",
    "36-45",
    "46-55",
    "46+",
    # ... in exact order as exported from le.classes_
]
```

---

#### NEEDED 4 — Validation Test Case

Run your existing script on the sample file `any_ks&dl_user001_(1).txt` and record the output. Send the result as `keystroke_validation.json`:

```json
{
  "input_file": "any_ks&dl_user001_(1).txt",
  "actual_age_group": "18-25",
  "predicted_label_index": 0,
  "predicted_label_string": "18-25",
  "output_probability": {
    "0": 0.72,
    "1": 0.15,
    "2": 0.08,
    "3": 0.03,
    "4": 0.02
  }
}
```

This is used to write a unit test that verifies the integration produces the exact same output as your notebook.

---

#### Keystroke Model — Delivery Checklist

- [ ] `keystroke_constants.py` containing `FEATURE_COLS` (201 entries), `FEATURE_MEDIANS` (one entry per feature), and `LABELS` (one entry per class)
- [ ] `keystroke_validation.json` with predicted output on the sample file

That is all that is needed. Do not re-send the ONNX file.

---

## Model 2: Face Age Expert

We have no artifacts for this model yet. Everything below is needed.

---

### NEEDED — Weight File

Provide the trained model as a single file. Preferred formats in order:

| Format | How to save | Notes |
|--------|-------------|-------|
| **ONNX** (preferred) | `torch.onnx.export(model, ...)` | Self-contained, no class needed at load time |
| **TorchScript** | `torch.jit.save(torch.jit.script(model), "face_age_expert_v1.pt")` | Self-contained PyTorch |
| **PyTorch state dict** | `torch.save(model.state_dict(), "face_age_expert_v1.pt")` | Requires architecture class — see below |

**File naming:** `face_age_expert_v<version>.<ext>`

If you use PyTorch state dict, also include a `face_architecture.py` file with the full `torch.nn.Module` class definition — no internal imports, fully self-contained.

---

### NEEDED — Input Specification

#### Raw Input Format

Describe what the raw data looks like before any preprocessing:

- Accepted file formats (JPEG, PNG, other?)
- Color space (RGB or BGR?)
- Any minimum or maximum resolution requirements
- Single face per image, or does the model handle multi-face images?

#### Preprocessing Steps (in order)

List every transformation applied before the model call, with all parameters:

```
Example — fill in your actual values:
1. Decode image bytes to RGB uint8 array
2. Detect face bounding box using <which detector? MTCNN? RetinaFace? OpenCV Haar?>
3. Crop to bounding box with <N>px padding
4. Resize to <H>×<W>
5. Convert to float32, divide by 255
6. Normalize: mean=[?, ?, ?], std=[?, ?, ?]
7. Add batch dim → shape (1, 3, H, W)
```

Specify what happens when **no face is detected** — should the framework skip this expert for that request, or raise an error?

#### Final Input Tensor Specification

| Property | Value |
|----------|-------|
| Shape | e.g. `(1, 3, 224, 224)` |
| dtype | |
| Value range after normalization | e.g. `[-2.5, 2.5]` |
| Input node name (if ONNX) | |

---

### NEEDED — Output Specification

Is the model a **regressor** (outputs a single age float) or a **classifier** (outputs an age group index)?

#### If Classifier:

| Property | Value |
|----------|-------|
| Output node name (ONNX) | |
| Output shape | e.g. `(batch,)` |
| Output dtype | e.g. `int64` |

Provide the label map:

```python
FACE_LABELS = {
    0: "18-25",
    1: "26-35",
    # ...
}
```

Also provide the probability output node name and how to read per-class confidence.

#### If Regressor:

| Property | Value |
|----------|-------|
| Output node name (ONNX) | |
| Output shape | e.g. `(batch, 1)` |
| Output dtype | e.g. `float32` |
| Unit | years? |
| Valid output range | e.g. `[1, 100]` |
| Confidence output | second output node? fixed value? |

---

### NEEDED — Anonymization / Privacy Notes

State whether face blurring or masking is applied **before** feature extraction:

- Is the face cropped before being passed to the model (the full image is never stored)?
- Is any additional anonymization required, or is cropping sufficient?
- Should the framework blur non-face regions of the image before processing?

---

### NEEDED — Runtime Requirements

| Requirement | Value |
|-------------|-------|
| Runtime library and minimum version | e.g. `onnxruntime>=1.17`, `torch>=2.1` |
| Device | CPU only / GPU optional / GPU required |
| Approximate inference latency on CPU | |
| Any known incompatibilities | |

---

### NEEDED — Self-Contained Inference Snippet

Provide a runnable Python script that:

1. Loads the model from the weight file
2. Loads or constructs one example input (a real image or synthetic data)
3. Runs inference
4. Prints the predicted label/age and confidence

This snippet is the ground truth used to implement the Expert Plugin. It must run without modification.

```python
# face_inference_example.py — fill in your actual implementation
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("face_age_expert_v1.onnx")

# TODO: load a real image and preprocess it, or use random data for shape verification
x = np.random.rand(1, 3, 224, 224).astype(np.float32)

outputs = sess.run(None, {"<input_node_name>": x})
print("Predicted label:", outputs[0])
print("Probabilities:",   outputs[1])
```

---

### NEEDED — Validation Test Case

Provide one real example with a known correct output as `face_validation.json`:

```json
{
  "input_file": "face_sample.jpg",
  "actual_age_group": "26-35",
  "predicted_label_index": 1,
  "predicted_label_string": "26-35",
  "confidence": 0.81
}
```

Include the sample image file alongside the JSON.

---

### Face Model — Delivery Checklist

- [ ] Weight file (`face_age_expert_v1.<ext>`) in ONNX, TorchScript, or state dict format
- [ ] `face_architecture.py` (only if state dict format)
- [ ] Raw input format description
- [ ] Preprocessing steps in order with all parameters
- [ ] Final input tensor shape, dtype, and value range
- [ ] Output type (classifier or regressor), node names, dtype, shape
- [ ] Label map / confidence derivation
- [ ] Anonymization notes
- [ ] Runtime requirements
- [ ] `face_inference_example.py` — runnable, self-contained
- [ ] `face_validation.json` + sample image

---

## Package Structure to Send Back

```
keystroke_model/
  keystroke_constants.py        ← FEATURE_COLS, FEATURE_MEDIANS, LABELS
  keystroke_validation.json     ← predicted output on sample file

face_model/
  face_age_expert_v1.<ext>      ← weight file
  face_architecture.py          ← only if state dict
  face_inference_example.py     ← runnable snippet
  face_validation.json          ← expected output
  face_sample.<ext>             ← sample input image
  README.md                     ← fills in all NEEDED items above
```

---

## Questions

Direct any questions about the framework integration layer to the engineering team before finalizing deliverables. Do not make assumptions about parameters — every value must be explicit and verified against your training notebook.
