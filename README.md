# Age Prediction using Mixture of Experts (APMoE)

Predicting age using anonymous biometric data with a Mixture of Experts (MoE) model. This project focuses on privacy-preserving age verification without the need for identifiable personal data, integrating advanced Human-Computer Interaction (HCI) implementations to ensure seamless user experience.

## Project Overview

Age verification is critical for protecting minors and ensuring compliance with legal regulations online. Current solutions often require sensitive personal identity documents, raising privacy and security concerns. APMoE offers a novel, privacy-respecting approach by estimating user age from non-identifiable data modalities such as facial images, voice samples, gait, EEG signals, and keystroke dynamics, processed through lightweight deep learning architectures. The project aims to balance accuracy, efficiency, and privacy protection, enabling scalable deployment across platforms and devices.

The system uses ensemble modeling via a mixture of experts, combining predictions from multiple distinct data sources to improve robustness and reliability without compromising user anonymity.

## Features

- Multi-modal age prediction using anonymized biometric and behavioral data
- Lightweight deep learning models like MobileNet and EfficientNet for efficient processing
- Ensemble modeling (Mixture of Experts) for enhanced accuracy
- Evaluation of model fairness, latency, and usability
- Focus on privacy adherence and ethical data handling
- User-friendly verification interface with strong emphasis on data security
- Deployment-ready APIs with containerization support

## Data Sources

Datasets are **not included** in this repository due to licensing restrictions but should be obtained separately from their original sources. The project uses the following datasets:

- Facial Age Dataset (Kaggle): https://www.kaggle.com/datasets/frabbisw/facial-age
- MIMIC Electronic Health Records (EHR): https://mimic.physionet.org/
- OU-ISR Gait Dataset: https://islab.ou.edu/datasets/
- Mozilla Common Voice Speech Dataset: https://commonvoice.mozilla.org/en/datasets
- IKDD Keystroke Dynamics Dataset: https://github.com/MachineLearningVisionRG/IKDD.git
- TUH EEG Corpus: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

## Installation

APMoE requires **Python 3.11+**. The base package is intentionally lightweight:
it installs the framework core, public types, registries, config loader, and CLI
without shipping model artifacts or heavyweight ML backends.

```bash
pip install apmoe
```

For the built-in serving stack and demo experts, install the relevant extras:

```bash
pip install "apmoe[serve,models]"
```

Common extras:

| Extra | Adds |
|---|---|
| `serve` | FastAPI, Uvicorn, multipart upload support |
| `image` | Pillow image decoding |
| `onnx` | ONNX Runtime for `KeystrokeAgeExpert` |
| `tensorflow` | TensorFlow / TensorFlow macOS for `FaceAgeExpert` |
| `remote` | HTTPX for remote experts |
| `security` | JWT and cryptography support |
| `redis` | Redis-backed auth invalidation and rate limiting |
| `models` | Runtime dependencies for the bundled/demo experts |
| `dev` | Test, lint, type-check, build, and publishing tools |

### From Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aeldesouky/APMoE.git
   cd APMoE
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify the installation**:
   ```bash
   apmoe --help
   ```

## Usage

APMoE is designed as a framework package: install the runtime you need, create a
project scaffold, then provide or download model artifacts explicitly.

### 1. Scaffolding a Local Project
Create a runnable configuration:
```bash
apmoe init my_app
cd my_app
```

In an interactive terminal, `apmoe init` asks whether you want to acquire demo
model artifacts immediately. For scripts, use `--download-models` or
`--no-download-models` explicitly.

To populate demo model artifacts for the built-in experts:

```bash
apmoe download-models --dest weights
```

When running from a source checkout, `apmoe init my_app --builtin` uses the same
model acquisition path and can copy local demo artifacts into the scaffold. PyPI
wheels do not bundle model files; set `APMOE_MODEL_SOURCE_DIR` or provide your
own weights when using a lightweight wheel.

The scaffold is local-only by default. When you run `apmoe validate`,
`apmoe serve`, or `apmoe predict`, the CLI prints an expert summary showing
whether each expert is `[local]`, `[remote]`, or `[local fallback]`.

### 2. Validating the Configuration
Before starting the server, ensure your configuration and weights are valid:
```bash
apmoe validate --config config.json
```

### 3. Serving the API
Start the high-performance ASGI inference server:
```bash
apmoe serve --config config.json --workers 1
```
The server will bind to `127.0.0.1:8000`. You can interact with the live Swagger UI immediately at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Running the Test Suite (Development)
If you are developing or contributing to APMoE, ensure you run the comprehensive test suite:
```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Extending APMoE

Local projects can keep using dotted import paths in `config.json`:

```json
{ "class": "myproject.experts.MyExpert" }
```

Installed extension packages can also expose components through Python entry
points:

```toml
[project.entry-points."apmoe.experts"]
my_expert = "my_package.experts:MyExpert"

[project.entry-points."apmoe.aggregators"]
my_aggregator = "my_package.aggregation:MyAggregator"
```

After installation, configs may use those short names:

```json
{ "class": "my_expert" }
```

### API versioning

The HTTP API is versioned under `/v1` (for example, `POST /v1/predict`).
Legacy unversioned endpoints remain temporarily and return `Deprecation`
and `Sunset` headers, plus `X-API-Version: 1`, to signal the migration
window for clients.

### Security

APMoE includes framework-level security controls for stateless JWT
authentication, scope authorization, shared Redis-backed token invalidation and
rate limiting with process-local fallback on Redis outages, remote expert
endpoint allowlists, remote retries and circuit breakers, configurable
prediction fallback, paired remote-to-local expert fallback, remote response limits, local SHA-256 model checks,
RSA-signed remote model manifests, correlation IDs, auditable security logs,
and secret redaction. See
[`docs/dev/security.md`](docs/dev/security.md) for the full reference and
production checklist.

### Operations and deployment

Deployment trade-offs, Lambda-style serverless guidance, dedicated
infrastructure guidance, fallback behavior, hot swapping, rollout/rollback,
autoscaling, bandwidth estimates, and vendor SLA recommendations are documented
in [`docs/deployment_sla_fallback.md`](docs/deployment_sla_fallback.md).

### Confidence scores

Per-expert outputs include a `confidence` field. Values are in **`[0.0, 1.0]`** when the model reports a meaningful score (for example, the keystroke ONNX classifier uses the maximum class probability). The bundled **face (Keras) regressor** does not produce a calibrated confidence: it reports **`-1.0`**, meaning *not applicable / not reported*. The aggregated prediction’s `confidence` remains in `[0.0, 1.0]`. See `docs/face_integration.md` and `docs/dev/core/types.md` for details.

## Ethical Considerations

This project adheres strictly to data privacy laws and ethical guidelines. No personally identifiable information (PII) is stored, shared, or processed beyond anonymized signals. Predicted ages are used solely for research and prototype development, not for decisions impacting users without additional consent.

## Citation

Please cite this project and datasets appropriately when using or referring to results.

***

## License

APMoE is licensed under the MIT License. See [LICENSE](LICENSE) for the
canonical license text and [`docs/licensing.md`](docs/licensing.md) for
dataset, model artifact, and redistribution guidance.

Datasets referenced by the project are not redistributed and remain governed by
their original licenses.
