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

## Installation (From Source)

APMoE requires **Python 3.11+**. Complete environment isolation using Docker is currently in progress. In the meantime, you can easily run the system locally from source.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/APMoE.git
   cd APMoE
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e .
   ```
   *This automatically installs all required dependencies (FastAPI, ONNX Runtime, Keras, etc.) specified in `pyproject.toml`.*

4. **Verify the installation**:
   ```bash
   apmoe --help
   ```

## Usage

APMoE is designed to be frictionless out-of-the-box. You do not need to manually configure models to run a smoke test.

### 1. Scaffolding a Local Project
Create a runnable configuration by initializing a project with bundled weights:
```bash
apmoe init my_app --builtin
cd my_app
```
*(This generates a `config.json` and copies working ONNX/Keras models into a `weights/` directory).*

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
pip install -e ".[test]"
pytest tests/ -v
```

### API versioning

The HTTP API is versioned under `/v1` (for example, `POST /v1/predict`).
Legacy unversioned endpoints remain temporarily and return `Deprecation`
and `Sunset` headers, plus `X-API-Version: 1`, to signal the migration
window for clients.

### Confidence scores

Per-expert outputs include a `confidence` field. Values are in **`[0.0, 1.0]`** when the model reports a meaningful score (for example, the keystroke ONNX classifier uses the maximum class probability). The bundled **face (Keras) regressor** does not produce a calibrated confidence: it reports **`-1.0`**, meaning *not applicable / not reported*. The aggregated prediction’s `confidence` remains in `[0.0, 1.0]`. See `docs/face_integration.md` and `docs/dev/core/types.md` for details.

## Ethical Considerations

This project adheres strictly to data privacy laws and ethical guidelines. No personally identifiable information (PII) is stored, shared, or processed beyond anonymized signals. Predicted ages are used solely for research and prototype development, not for decisions impacting users without additional consent.

## Citation

Please cite this project and datasets appropriately when using or referring to results.

***

## License

The code in this repository is licensed under the Apache 2.0 License for academic and research purposes.

**IMPORTANT:**

- The datasets used in this project are subject to their respective license agreements.
- Redistribution of datasets is **not permitted**; users must obtain datasets directly from their sources.
- This project and its code are for **non-commercial use only**.
- Users **must provide attribution** to the original dataset owners and project creators before using any derived models or results.

By using this repository, you agree to adhere to these licensing and attribution requirements.
