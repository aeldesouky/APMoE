# apmoe-models

Demo model artifacts for APMoE built-in experts.

Install through the main package:

```bash
pip install "apmoe[models]"
```

Or install directly:

```bash
pip install apmoe-models
```

The `apmoe` CLI can install this package automatically when a user runs:

```bash
apmoe download-models --dest weights
apmoe init my_app --download-models
```

Included artifacts:

- `face_age_expert.keras`
- `keystroke_age_expert.onnx`
- `keystroke_constants.json`

These artifacts are provided for demos and prototyping. Confirm dataset/model
redistribution rights before production use.
