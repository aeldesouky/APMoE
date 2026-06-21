# Keystroke Demo Remote Executor

Standalone HTTP executor for the APMoE remote-expert demo. It runs in
keystroke mode, loads the bundled ONNX keystroke model, and implements the
default `apmoe.experts.remote.RemoteExpert` contract.

Run it from the repository root:

```bash
python -m remote_executors.keystroke_demo --host 127.0.0.1 --port 8010
```

Then point APMoE at it:

```bash
apmoe serve --config configs/remote_keystroke_demo.json
```

Contract:

```json
{
  "expert_name": "remote_keystroke_age_expert",
  "modalities": {
    "keystroke": {
      "dur_8": [95.0, 102.0],
      "dig_65_83": [145.0]
    }
  }
}
```

Response:

```json
{
  "predicted_age": 34.5,
  "confidence": 0.82,
  "metadata": {}
}
```

