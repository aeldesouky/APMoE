# AI Integration Guide

This document records the model handoff contract between the ML work and the APMoE framework. It is kept for project traceability; the current repository already includes the demo artifacts used by the built-in experts.

## Current Artifact Status

| Model | Current implementation | Artifact status |
|---|---|---|
| Keystroke age expert | `apmoe.experts.builtin.KeystrokeAgeExpert` | `keystroke_age_expert.onnx` and `keystroke_constants.json` are included in package artifacts. |
| Face age expert | `apmoe.experts.builtin.FaceAgeExpert` | `face_age_expert.keras` is included in package artifacts. |

The package build includes demo artifacts from `src/apmoe/weights/`. The root `weights/` folder is for local development and demos.

## Framework Contract

APMoE is inference-only. Each model is wrapped by an `ExpertPlugin` with two runtime responsibilities:

1. `load_weights(path)` loads local model artifacts once during bootstrap.
2. `predict(inputs)` receives processed modality inputs and returns an `ExpertOutput`.

Remote experts use `endpoint` configuration instead of `weights`; see [Configuration reference](../dev/configuration.md#remote-primary-with-local-fallback).

## Keystroke Expert Contract

The built-in keystroke expert expects:

- ONNX model file: `keystroke_age_expert.onnx`
- constants file: `keystroke_constants.json`
- input modality: `keystroke`
- model input shape: one row of ordered numeric features
- output: age-group class/probability data mapped into an age estimate and confidence

The constants file must contain the ordered feature list, training medians, and labels required to reproduce the training-time feature vector. See [Keystroke integration](keystroke-integration.md) for supported request formats and troubleshooting.

## Face Expert Contract

The built-in face expert expects:

- Keras model file: `face_age_expert.keras`
- input modality: `image`
- image input that can be decoded and normalized by the image processor and cleaner
- output: one regression age value

The face model does not report calibrated confidence, so `FaceAgeExpert` sets per-expert confidence to `-1.0`. Aggregators treat that as "not reported". See [Face integration](face-integration.md).

## Adding or Replacing Models

When replacing a model artifact, provide:

- model file format and exact filename
- required preprocessing steps and input shape
- output tensor names/shapes or response schema
- label mapping, class mapping, or regression interpretation
- validation sample and expected output
- runtime dependencies
- privacy/anonymization assumptions
- SHA-256 digest or remote signed manifest configuration for production

Update the relevant integration guide and configuration examples after any model replacement.
