# Keystroke Dynamics — Integration Guide

This document is the complete reference for the keystroke age-prediction expert:
how it works internally, what input formats it accepts, how to supply data
through the HTTP API, and how to wire it into the framework.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Files Required](#files-required)
3. [Input Formats](#input-formats)  
   3a. [IKDD text format](#1-ikdd-text-format-primary)  
   3b. [JSON triples format](#2-json-triples-format)  
   3c. [JSON dict format](#3-json-dict-format)
4. [Key Code Reference](#key-code-reference)
5. [HTTP API Usage](#http-api-usage)  
   5a. [JSON body — cURL](#curl)  
   5b. [JSON body — Python requests](#python-requests)  
   5c. [JSON body — JavaScript fetch](#javascript-fetch)  
   5d. [JSON body — alternative value shapes](#json-body--alternative-value-shapes)  
   5e. [Multipart form-data](#multipart-form-data-for-file-uploads)
6. [Programmatic Usage](#programmatic-usage)
7. [Configuration](#configuration)
8. [Response Format](#response-format)
9. [Pipeline Internals](#pipeline-internals)
10. [Capturing Keystrokes in a Client App](#capturing-keystrokes-in-a-client-app)
11. [Troubleshooting](#troubleshooting)

---

## How It Works

The keystroke expert predicts the **age group** of a user based on their
typing dynamics — specifically how long they hold each key (*hold time*)
and how fast they move between pairs of keys (*flight time*).

```
Raw session data
      │
      ▼
KeystrokeProcessor          parses IKDD / JSON → feature dict
      │
      ▼
KeystrokeCleaner            removes timings ≤ 0 ms or > 10,000 ms
      │
      ▼
KeystrokeAnonymizer         pass-through (timings have no direct PII)
      │
      ▼
KeystrokeAgeExpert          builds 201-feature vector → ONNX SVM → age group
      │
      ▼
WeightedAverageAggregator   combines experts (solo here) → final prediction
```

**Model details**

| Property | Value |
|---|---|
| Architecture | SVM Classifier (sklearn → ONNX) |
| Input features | 201 (`dur_*` hold times + `dig_*_*` flight times) |
| Output classes | `"18-25"`, `"26-35"`, `"36-45"`, `"46+"` |
| Reported accuracy | 70 % |
| Fill strategy for missing features | Training-set median (not zero) |

---

## Files Required

All three files must be present **in the same directory** before the server starts:

```
weights/
  keystroke_age_expert.onnx      ← trained SVM model (ONNX format)
  full_digraph_index.json        ← complete digraph vocabulary (6,539 pairs)
  keystroke_constants.json       ← 201 feature names + training medians + labels
```

`keystroke_constants.json` structure:

```json
{
  "feature_cols": [
    "dur_13", "dur_16", "dur_32", ...,
    "dig_8_73", "dig_8_79", "dig_8_82", "dig_8_84"
  ],
  "feature_medians": {
    "dur_13": 98.81,
    "dur_32": 107.73,
    "dig_8_73": 625.83,
    ...
  },
  "labels": ["18-25", "26-35", "36-45", "46+"]
}
```

---

## Input Formats

The `POST /predict` endpoint accepts the keystroke session as the value of
the **`"keystroke"` key** in a JSON body (recommended), or as an uploaded
file field in a multipart request. Three data formats are supported for the
session payload itself:

### 1. IKDD Text Format (primary)

Plain-text format produced by keystroke logging tools.

```
# Any line starting with '#' is a comment and is ignored
# Format per data line:  key1-key2,timing1,timing2,...
#   key2 == 0  →  hold time  (how long key1 was held down, in ms)
#   key2 != 0  →  flight time (time from key1 release to key2 press, in ms)

8-0,95.0,102.0,88.0
13-0,100.0,97.0
32-0,108.0,115.0
65-0,120.0,118.0
65-83,145.2,130.0,160.4
79-32,261.0,255.0
8-73,620.0,640.0
```

**Rules:**
- One entry per line: `key1-key2,v1,v2,...`
- Multiple timing measurements for the same pair on the **same line** (comma-separated) — all are recorded and averaged
- The same pair can also appear on **multiple lines** — all measurements are pooled
- Lines starting with `#` are ignored; blank lines are ignored
- Timings outside `(0, 10 000]` ms are discarded by the cleaner

**Mapping to feature names:**

| Line | Feature name | Interpretation |
|---|---|---|
| `8-0,95.0` | `dur_8` | User held key 8 (Backspace) for 95 ms |
| `65-0,120.0` | `dur_65` | User held key 65 (A) for 120 ms |
| `65-83,145.2` | `dig_65_83` | 145 ms between releasing A and pressing S |
| `79-32,261.0` | `dig_79_32` | 261 ms between releasing O and pressing Space |

---

### 2. JSON Triples Format

A JSON array where each element is `[key1, key2, timing_ms]`.
Useful for APIs and programmatic generation.

```json
[
  [8,  0,  95.0],
  [8,  0,  102.0],
  [13, 0,  100.0],
  [32, 0,  108.0],
  [65, 0,  120.0],
  [65, 83, 145.2],
  [65, 83, 130.0],
  [79, 32, 261.0],
  [8,  73, 620.0]
]
```

Each triple `[key1, key2, timing_ms]`:
- `key1`, `key2` — integer key scan-codes (see [Key Code Reference](#key-code-reference))
- `timing_ms` — timing measurement in milliseconds (float)
- When `key2 == 0`, this is a **hold time** for `key1`
- When `key2 != 0`, this is a **flight time** from `key1` to `key2`

Multiple rows with the same `(key1, key2)` pair are pooled (averaged) before inference.

---

### 3. JSON Dict Format

Pre-aggregated dict mapping feature names directly to timing lists.
Useful when you compute feature names on the client side.

```json
{
  "dur_8":   [95.0, 102.0, 88.0],
  "dur_13":  [100.0, 97.0],
  "dur_65":  [120.0, 118.0],
  "dig_65_83": [145.2, 130.0, 160.4],
  "dig_79_32": [261.0, 255.0],
  "dig_8_73":  [620.0, 640.0]
}
```

Feature name convention:
- `dur_{key}` — hold time for key with scan-code `key`
- `dig_{key1}_{key2}` — flight time from key `key1` to key `key2`

---

## Key Code Reference

The framework uses **virtual key codes** as integers (same as Windows VK codes
and the values logged by most keystroke capture libraries).

Common keys relevant to typing tasks:

| Key | Code | Feature prefix |
|---|---|---|
| Backspace | 8 | `dur_8`, `dig_8_*`, `dig_*_8` |
| Tab | 9 | `dur_9` |
| Enter | 13 | `dur_13`, `dig_13_*` |
| Shift | 16 | `dur_16` |
| Alt | 18 | `dur_18` |
| CapsLock | 20 | `dur_20` |
| Space | 32 | `dur_32`, `dig_32_*`, `dig_*_32` |
| A–Z | 65–90 | `dur_65`–`dur_90`, `dig_65_*`–`dig_90_*` |
| 0–9 (number row) | 48–57 | `dur_48`–`dur_57` |
| Left arrow | 37 | `dur_37` |
| Right arrow | 39 | `dur_39` |

> Only the 201 features listed in `keystroke_constants.json` affect the
> prediction. Any key pairs not in that list are parsed but silently ignored
> when building the feature vector (they have no column in the trained model).

---

## HTTP API Usage

Start the server with the keystroke config:

```bash
uv run python -m apmoe serve --config configs/keystroke.json
```

The endpoint is `POST /predict`. It accepts **two content types**:

| Content-Type | Use when |
|---|---|
| `application/json` | Keystroke data, structured features — **recommended** |
| `multipart/form-data` | Binary file uploads (images, audio, etc.) |

The **JSON format is preferred** for keystroke data — no file encoding
overhead, works natively from any HTTP client.

---

### JSON body (recommended)

The body is a JSON object where each key is a modality name and the value is
the session data. For the keystroke modality, the value is a list of
`[key1, key2, timing_ms]` triples:

```
POST /predict
Content-Type: application/json

{
  "keystroke": [
    [8,  0,  95.0],
    [8,  0, 102.0],
    [13, 0, 100.0],
    [65, 83, 145.2],
    [79, 32, 261.0],
    [8,  73, 620.0]
  ]
}
```

#### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "keystroke": [
      [8,  0,  95.0],
      [13, 0, 100.0],
      [65, 83, 145.2],
      [79, 32, 261.0],
      [8,  73, 620.0]
    ]
  }'
```

#### Python requests

```python
import requests

session = [
    [8,  0,  95.0],
    [8,  0, 102.0],
    [13, 0, 100.0],
    [65, 83, 145.2],
    [79, 32, 261.0],
    [8,  73, 620.0],
]

resp = requests.post(
    "http://localhost:8000/predict",
    json={"keystroke": session},   # sets Content-Type: application/json automatically
)
print(resp.json())
```

#### JavaScript fetch

```javascript
const session = [
  [8,  0,  95.0],
  [13, 0, 100.0],
  [65, 83, 145.2],
  [79, 32, 261.0],
  [8,  73, 620.0],
];

const resp = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ keystroke: session }),
});
const result = await resp.json();
console.log(result);
```

#### axios

```javascript
const { data } = await axios.post("http://localhost:8000/predict", {
  keystroke: [
    [8, 0, 95.0],
    [13, 0, 100.0],
    [65, 83, 145.2],
  ],
});
```

---

### JSON body — alternative value shapes

All three input formats from [Input Formats](#input-formats) work as the JSON
value (the server auto-detects which one based on whether the value starts
with `[` or `{`):

```json
// List of triples (most common)
{ "keystroke": [[8, 0, 95], [13, 0, 100], [65, 83, 145]] }

// Pre-aggregated feature dict
{ "keystroke": { "dur_8": [95, 102], "dig_65_83": [145, 130] } }

// Raw IKDD text as a string
{ "keystroke": "8-0,95.0\n13-0,100.0\n65-83,145.2" }
```

---

### Multipart form-data (for file uploads)

Use multipart only when uploading **binary files** (images, audio).
For keystroke data the JSON format is always simpler.

```bash
# IKDD text file
curl -X POST http://localhost:8000/predict \
  -F "keystroke=@session.txt"

# JSON file
curl -X POST http://localhost:8000/predict \
  -F "keystroke=@session.json;type=application/json"
```

```python
# Python requests — file upload
with open("session.txt", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/predict",
        files={"keystroke": ("session.txt", f, "text/plain")},
    )
```

---

## Programmatic Usage

Use the pipeline directly in Python without the HTTP server:

```python
from apmoe.modality.builtin.keystroke import KeystrokeProcessor
from apmoe.processing.builtin.cleaners import KeystrokeCleaner
from apmoe.processing.builtin.anonymizers import KeystrokeAnonymizer
from apmoe.experts.builtin import KeystrokeAgeExpert
from apmoe.aggregation.builtin import WeightedAverageAggregator

# --- 1. Build the pipeline ---
processor  = KeystrokeProcessor()
cleaner    = KeystrokeCleaner()
anonymizer = KeystrokeAnonymizer()

expert = KeystrokeAgeExpert()
expert.load_weights("./weights/keystroke_age_expert.onnx")
# load_weights also reads weights/keystroke_constants.json automatically

aggregator = WeightedAverageAggregator()

# --- 2. Prepare raw input (any of the three formats) ---
raw_input = b"""
# typing session
8-0,95.0,102.0,88.0
13-0,100.0,97.0
65-0,120.0,118.0
65-83,145.2,130.0
79-32,261.0,255.0
8-73,620.0,640.0
"""

# --- 3. Run the pipeline ---
data = processor.preprocess(raw_input)       # parse
data = cleaner.clean(data)                   # filter invalid timings
data = anonymizer.anonymize(data)            # no-op for keystroke

output = expert.predict({"keystroke": data}) # ONNX inference
prediction = aggregator.aggregate([output])  # combine (single expert here)

# --- 4. Read results ---
print(f"Predicted age:   {prediction.predicted_age:.1f} years")
print(f"Confidence:      {prediction.confidence:.1%}")
print(f"Predicted group: {output.metadata['predicted_group']}")
print(f"Group probs:     {output.metadata['age_group_probs']}")
print(f"Coverage:        {output.metadata['features_observed_fraction']:.1%}")
```

Using the full `APMoEApp` IoC container from config:

```python
from apmoe.core.app import APMoEApp

app = APMoEApp.from_config("configs/keystroke.json")
# app.bootstrap() loads all weights automatically

import asyncio
result = asyncio.run(app.predict_async({"keystroke": raw_input}))
print(result.predicted_age, result.confidence)
```

---

## Configuration

`configs/keystroke.json` wires up the full pipeline:

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "keystroke",
        "processor": "apmoe.modality.builtin.keystroke.KeystrokeProcessor",
        "pipeline": {
          "cleaner":    "apmoe.processing.builtin.cleaners.KeystrokeCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.KeystrokeAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name":       "keystroke_age_expert",
        "class":      "apmoe.experts.builtin.KeystrokeAgeExpert",
        "weights":    "./weights/keystroke_age_expert.onnx",
        "modalities": ["keystroke"]
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
    },
    "serving": {
      "host":         "0.0.0.0",
      "port":         8000,
      "workers":      1,
      "cors_origins": ["*"],
      "log_level":    "info"
    }
  }
}
```

No embedder is configured for the `keystroke` modality — the expert receives
`ModalityData` directly and builds the feature vector internally.

---

## Response Format

A successful `POST /predict` returns HTTP 200 with:

```json
{
  "predicted_age": 51.0,
  "confidence": 0.8136,
  "confidence_interval": null,
  "per_expert_outputs": [
    {
      "expert_name": "keystroke_age_expert",
      "consumed_modalities": ["keystroke"],
      "predicted_age": 51.0,
      "confidence": 0.8136,
      "metadata": {
        "predicted_group": "46+",
        "age_group_probs": {
          "18-25": 0.0425,
          "26-35": 0.0496,
          "36-45": 0.0942,
          "46+":   0.8136
        },
        "features_observed_fraction": 0.055
      }
    }
  ],
  "skipped_experts": [],
  "metadata": {}
}
```

| Field | Description |
|---|---|
| `predicted_age` | Continuous age estimate in years (probability-weighted midpoint: 21.5 / 30.5 / 40.5 / 55.0) |
| `confidence` | Maximum class probability `[0, 1]` |
| `predicted_group` | Highest-probability age group label |
| `age_group_probs` | Per-class SVM probability for all four groups |
| `features_observed_fraction` | Fraction of the 201 model features that appeared in this session; low values mean more features were filled with medians |

### Error responses

| Code | Cause |
|---|---|
| 422 | Body could not be parsed — invalid JSON, non-object JSON root, or malformed multipart |
| 503 | No runnable experts (e.g. model weights not loaded) |
| 500 | Unexpected framework error |

---

## Pipeline Internals

### Feature construction

For each of the **201 features** in `feature_cols` order:

1. Collect every timing measurement for that `(key1, key2)` pair from the session
2. Discard values `≤ 0 ms` or `> 10 000 ms` (cleaner step)
3. Compute the **mean** of remaining values
4. If no valid measurement exists, substitute the **training-set median** from `feature_medians`
5. Stack the 201 values into a `float32` vector and feed it to the ONNX model

No z-score normalisation is applied — the ONNX model was exported with raw
millisecond values (the SVM decision boundaries are in ms-space).

### Model output

The SVM produces:
- `output_label`: integer class index `0`–`3`
- `output_probability`: `{0: p0, 1: p1, 2: p2, 3: p3}` — per-class probabilities

The continuous age is computed as a probability-weighted average:

```
predicted_age = p0 × 21.5 + p1 × 30.5 + p2 × 40.5 + p3 × 55.0
```

### Coverage

The `features_observed_fraction` field in the response tells you how much of
the session's data the model actually saw. A session where only 5 % of the 201
features were typed will rely heavily on medians and will be **less accurate**.

For best accuracy, aim for sessions where the user types at least **200–500
keystrokes** covering a mix of common letter pairs and function keys.

---

## Capturing Keystrokes in a Client App

### Browser (JavaScript)

```javascript
// --- Step 1: capture events ---
const events = [];

document.addEventListener("keydown", (e) => {
  events.push({ code: e.keyCode, time: performance.now(), type: "down" });
});
document.addEventListener("keyup", (e) => {
  events.push({ code: e.keyCode, time: performance.now(), type: "up" });
});

// --- Step 2: convert to [key1, key2, timing_ms] triples ---
function buildSession(events) {
  const session = [];
  const downTimes = {};

  for (const ev of events) {
    if (ev.type === "down" && downTimes[ev.code] === undefined) {
      downTimes[ev.code] = ev.time;
    } else if (ev.type === "up" && downTimes[ev.code] !== undefined) {
      // Hold time: key2 = 0
      session.push([ev.code, 0, ev.time - downTimes[ev.code]]);
      delete downTimes[ev.code];
    }
  }

  // Flight times between consecutive key-down events
  const downs = events.filter(e => e.type === "down");
  for (let i = 0; i < downs.length - 1; i++) {
    session.push([downs[i].code, downs[i + 1].code, downs[i + 1].time - downs[i].time]);
  }

  return session;
}

// --- Step 3: send as a JSON POST ---
async function submitSession() {
  const session = buildSession(events);

  const resp = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keystroke: session }),
  });

  const result = await resp.json();
  console.log("Predicted age:", result.predicted_age);
  console.log("Age group:", result.per_expert_outputs[0].metadata.predicted_group);
}
```

### Python (desktop app)

```python
from pynput import keyboard
import time

events = []

def on_press(key):
    try:
        code = key.vk
    except AttributeError:
        code = key.value.vk
    events.append({"code": code, "time": time.perf_counter() * 1000, "type": "down"})

def on_release(key):
    try:
        code = key.vk
    except AttributeError:
        code = key.value.vk
    events.append({"code": code, "time": time.perf_counter() * 1000, "type": "up"})

# Start listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()  # run until stopped
```

Then convert `events` to IKDD or JSON triples using the same logic shown in
the JavaScript example above.

---

## Troubleshooting

### `ExpertError: Missing required constants file`

The file `weights/keystroke_constants.json` was not found next to
`keystroke_age_expert.onnx`. Ensure both files are in the same directory and
the `weights` path in `configs/keystroke.json` is correct.

### `ExpertError: feature_cols has N entries but ONNX model expects 201`

The constants file has the wrong number of features. It must have exactly 201
entries in `feature_cols`. If you regenerated it from the training notebook,
check that `X_imp.columns.tolist()` returns 201 items after all preprocessing
steps are complete.

### Low `features_observed_fraction` (< 10 %)

The session is too short or the user typed very few of the 201 selected key
pairs. More than ~80 % of features will be filled with medians, reducing
accuracy. Ask users to type freely for at least 30–60 seconds before
submitting.

### All probabilities concentrate on one class

This is normal for short sessions — the model defaults heavily toward the
most common training class when most features are unseen. Longer typing
sessions distribute more probability mass across classes.

### Negative or extreme timing values in input

Values `≤ 0 ms` and `> 10 000 ms` are automatically discarded by
`KeystrokeCleaner`. If many values are being removed
(`removed_timings` in cleaner metadata is high), check that your client is
computing timings correctly (milliseconds, not seconds or microseconds).
