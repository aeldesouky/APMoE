# Licensing Information

APMoE is licensed under the MIT License.

This repository contains framework source code, documentation, configuration
examples, tests, scripts, and demo model artifacts used for local development.
PyPI wheels intentionally do not include the demo model artifacts; users must
download them explicitly or provide their own weights. Unless a file states
otherwise, project-authored materials may be used, copied,
modified, merged, published, distributed, sublicensed, and sold under the MIT
License terms.

## MIT License Summary

The MIT License is a permissive open-source license. In practical terms:

- You may use APMoE in academic, research, private, or commercial projects.
- You may modify and redistribute the framework.
- You may include APMoE in larger proprietary or open-source systems.
- You must keep the copyright notice and MIT permission notice with copies or
  substantial portions of the software.
- The software is provided as-is, without warranty.

The canonical license text is in [LICENSE](../LICENSE).

## Datasets Are Not Redistributed

The datasets referenced by APMoE are not included in this repository. They must
be obtained from their original publishers and used under their own license
terms. The MIT License for APMoE does not grant rights to third-party datasets.

Operators and researchers are responsible for verifying dataset permissions for
training, evaluation, commercial use, retention, redistribution, and derived
model publication.

Project dataset references from the README:

| Dataset | Used for | Repository status | Licensing action before production |
|---|---|---|---|
| Facial Age Dataset (Kaggle) | face/image age modeling | Not included | Check Kaggle dataset terms, redistribution permission, and commercial-use rights. |
| IKDD Keystroke Dynamics Dataset | keystroke timing model | Not included | Check upstream GitHub/dataset terms and whether derived model weights may be redistributed. |
| Mozilla Common Voice | possible speech modality | Not included | Follow Common Voice license and attribution rules if a speech expert is added. |
| OU-ISR Gait Dataset | possible gait modality | Not included | Confirm research/commercial restrictions before training or publishing gait experts. |
| TUH EEG Corpus | possible EEG modality | Not included | Confirm clinical data access, citation, and redistribution limits. |
| MIMIC | possible EHR modality | Not included | Confirm PhysioNet credentialing, DUA restrictions, and whether derived artifacts may be shared. |

Only the framework, tests, docs, configs, scripts, and local demo artifacts are
in this repository. Raw dataset records should not be committed to the repo or
packaged into distributions.

## Model Artifacts

Demo model artifacts are available in source checkouts to make local scaffolding
and demo flows runnable. They are excluded from PyPI wheels and must be acquired
with `apmoe download-models`, copied from a configured `APMOE_MODEL_SOURCE_DIR`,
or replaced with operator-owned weights. Before distributing a product or hosted
service that uses any model artifact, confirm:

- the source training data allows the intended use;
- the trained artifact may be redistributed or hosted;
- model cards, citations, or attribution requirements are satisfied;
- privacy, consent, and age-estimation compliance obligations are met.

If a downstream deployment replaces the bundled experts with vendor-hosted or
customer-owned models, those models remain governed by their own license and
service terms.

Current demo artifact inventory:

| Artifact | Location | Used by | Notes |
|---|---|---|---|
| Keystroke ONNX model | `weights/keystroke_age_expert.onnx`, source checkout `src/apmoe/weights/keystroke_age_expert.onnx` | `KeystrokeAgeExpert` in `configs/keystroke.json` and `configs/multimodal.json` | Derived from keystroke training data. Confirm the training-data license before external redistribution. |
| Keystroke constants | `weights/keystroke_constants.json`, source checkout `src/apmoe/weights/keystroke_constants.json` | `KeystrokeAgeExpert` feature ordering, medians, labels | Operationally part of the keystroke model package. Treat with the same license/provenance as the ONNX model. |
| Face Keras model | `weights/face_age_expert.keras`, source checkout `src/apmoe/weights/face_age_expert.keras` | `FaceAgeExpert` in `configs/multimodal.json` | Derived from face age training data. Confirm rights for commercial deployment and redistribution. |
| Legacy/source face or keystroke artifacts | repository root or historical paths | Documentation and migration context | Do not assume these inherit MIT rights unless the project owner confirms provenance. |

Recommended production packaging rule:

- Keep framework code under MIT.
- Give every model bundle its own `MODEL_LICENSE.md` or model card.
- Record training dataset, model version, artifact digest, intended use,
  limitations, and redistribution status.
- If model redistribution is unclear, ship deployment instructions that require
  operators to provide their own weights instead of bundling those artifacts.

## Third-Party Dependencies

APMoE depends on third-party packages such as FastAPI, Pydantic, Uvicorn, ONNX
Runtime, TensorFlow/Keras, Pillow, PyJWT, Redis clients, and HTTPX. Those
dependencies keep their own licenses. Redistributing an application that embeds
APMoE should include any third-party notices required by the dependency set
actually shipped with that application.

The default `pip install apmoe` runtime includes the framework, serving,
remote, security, Redis client, image, ONNX, TensorFlow, and Torch
dependencies. Runtime extra names remain as compatibility aliases. The
`models` extra installs the separate `apmoe-models` artifact package.

| Install path | Runtime dependencies likely present | Notice impact |
|---|---|---|
| `apmoe` | Click, Pydantic, NumPy, FastAPI, Uvicorn, python-multipart, Pillow, ONNX Runtime, TensorFlow/Keras, Torch, HTTPX, PyJWT, cryptography, redis client | Include notices for all runtime dependencies; comply with crypto-library export/security review policies where applicable. |
| Runtime alias extras such as `apmoe[serve]`, `apmoe[security]`, or `apmoe[redis]` | Same as `apmoe`; dependencies are already included by default | Same notice set as `apmoe`. |
| `apmoe[models]` / `apmoe-models` | Packaged demo model artifacts | Include model provenance and redistribution notices; confirm production rights before redistributing. |
| `apmoe[dev]` | pytest, ruff, mypy, pre-commit | Usually development-only; not included in production redistribution unless packaged into the image. |

## Recommended Distribution Checklist

Before publishing an APMoE-based package, container image, service, or SDK:

- include the APMoE MIT license text;
- include required third-party dependency notices;
- list any model artifacts and their separate license or provenance;
- do not redistribute datasets unless their licenses explicitly permit it;
- document whether age predictions are research-only, advisory, or used in an
  automated decision flow;
- document privacy and retention behavior for biometric or behavioral inputs.

For an APMoE vendor deployment, include these project-specific files or
equivalents in the release artifact:

- `LICENSE` for the MIT framework license;
- `docs/licensing.md` or an adapted licensing notice;
- a model card for each configured expert in `config.json`;
- checksums for every local `weights` artifact;
- attribution or citation text required by training datasets;
- remote provider terms for any `RemoteExpert` endpoint;
- a statement that raw datasets are not redistributed with the service.

## Open License Decisions To Confirm

The repository now documents MIT licensing as requested. The remaining release
metadata that should be confirmed by the project owners is:

- the exact copyright holder name to use in the MIT notice;
- whether bundled model artifacts are also MIT-licensed or should carry
  separate notices;
- whether any institution, course, sponsor, or dataset attribution text must be
  included in a `NOTICE` or citation file.
