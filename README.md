# Age Prediction using Mixture of Experts (APMoE)

APMoE is an inference-only Python framework for privacy-preserving age prediction using a configurable Mixture of Experts pipeline. The current project supports image and keystroke modalities, local model experts, remote HTTP/LLM experts, FastAPI serving, CLI workflows, security controls, Redis-backed operational stores, and deployment guidance.

## Team Members

Submitted by:

| Name | Student ID | Program |
|---|---:|---|
| Ahmed M. Eldesouky | 202201114 | DSAI |
| Ahmed M. Abdelrahim | 202201983 | SWD/HCI |
| Seif Eldin H. Khashaba | 202200973 | DSAI |
| Mohamed A. Elnaggar | 202201974 | SWD/HCI |

Supervisor: Dr. / Prof. ______________________

## Problem Statement

Online age verification often relies on identity documents or personal data that can expose users to privacy and security risk. APMoE addresses this by estimating age from non-identifying signals, currently facial image data and keystroke dynamics, then combining expert predictions into one result. The framework is built for research and prototype deployment where privacy, modularity, reproducibility, and operational safety matter.

## Features

- Config-driven Mixture of Experts inference pipeline.
- Built-in image and keystroke modality processors.
- Built-in `FaceAgeExpert` using Keras and `KeystrokeAgeExpert` using ONNX.
- Weighted average, confidence-weighted, and median aggregation strategies.
- Remote expert support for external HTTP model providers and local LLM endpoints.
- Remote retry, circuit breaker, response-size limit, endpoint allowlist, and local fallback support.
- FastAPI serving with versioned `/v1` endpoints, Swagger UI, CORS, auth, authorization, rate limiting, request IDs, and audit logs.
- CLI commands for project scaffolding, model artifact acquisition, validation, prediction, and serving.
- Redis-backed rate-limit and JWT invalidation stores with process-local fallback.
- Unit, integration, end-to-end, resilience, and load-test scripts.

## System Architecture

APMoE loads a single JSON configuration, builds the requested modality pipelines, runs compatible experts, and aggregates their outputs into a final prediction.

```text
Client or CLI
  -> FastAPI/CLI entry point
  -> APMoEApp config bootstrap
  -> modality processors
  -> cleaner and anonymizer strategies
  -> local or remote expert plugins
  -> aggregation strategy
  -> prediction response
```

Architecture documentation:

- [Documentation index](docs/index.md)
- [System architecture](docs/architecture/system-architecture.md)
- [Dataflow design](docs/architecture/dataflow-design.md)
- [Inference pipeline diagram](docs/architecture/inference-pipeline-diagram.md)
- [Developer documentation](docs/dev/index.md)

## Technologies Used

| Area | Technologies |
|---|---|
| Language and package | Python 3.11-3.12, Hatchling, uv, pip |
| Backend and API | FastAPI, Uvicorn, Pydantic, python-multipart |
| AI/ML frameworks | ONNX Runtime, TensorFlow/Keras, PyTorch, NumPy, Pillow |
| Security | PyJWT, cryptography, signed remote manifests, audit logging |
| Remote integrations | httpx, custom REST experts, LM Studio-compatible provider |
| Database/cache | Redis for shared rate limiting and token invalidation |
| DevOps and quality | pytest, pytest-asyncio, pytest-cov, ruff, mypy, twine, GitHub Actions |

## Setup Instructions

APMoE requires Python `>=3.11,<3.13`.

1. Clone the repository.

```bash
git clone https://github.com/aeldesouky/APMoE.git
cd APMoE
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install for development.

```bash
pip install -e ".[dev]"
```

4. Verify the CLI.

```bash
apmoe --help
```

5. Run tests.

```bash
pytest
```

For the full walkthrough, see [User guide](docs/getting-started/user-guide.md) and [CLI reference](docs/dev/cli.md).

## Deployment Instructions

For local serving:

```bash
apmoe init my_app --download-models
cd my_app
apmoe validate --config config.json
apmoe serve --config config.json --host 127.0.0.1 --port 8000
```

The API is available at:

- `POST http://127.0.0.1:8000/v1/predict`
- `GET http://127.0.0.1:8000/v1/health`
- `GET http://127.0.0.1:8000/v1/info`
- Swagger UI: `http://127.0.0.1:8000/docs`

For production, package a validated config and immutable model artifacts, enable authentication/authorization, configure CORS, use Redis for shared rate limits and token invalidation, and place APMoE behind a TLS-terminating ingress or load balancer. See [Deployment, SLA, and fallback guidance](docs/operations/deployment-sla-fallback.md) and [Security reference](docs/dev/security.md).

## Usage Guide

Run a local prediction from files:

```bash
apmoe predict --config config.json --input data/
```

Send a keystroke prediction request to the HTTP API:

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d "{\"keystroke\": [[65, 0, 120], [65, 83, 80]]}"
```

Detailed usage docs:

- [User guide](docs/getting-started/user-guide.md)
- [Keystroke integration](docs/integrations/keystroke-integration.md)
- [Face integration](docs/integrations/face-integration.md)
- [Configuration reference](docs/dev/configuration.md)
- [Serving layer](docs/dev/serving.md)
- [OpenAPI reference](docs/dev/openapi.md)

## Screenshots / Demo

The repository includes generated architecture and performance visuals:

- [Performance dashboard](docs/assets/graphs/performance_dashboard.png)
- [Throughput vs concurrency](docs/assets/graphs/throughput_vs_concurrency.png)
- [P95 latency vs concurrency](docs/assets/graphs/p95_latency_vs_concurrency.png)
- [Average latency vs concurrency](docs/assets/graphs/average_latency_vs_concurrency.png)
- [Inference pipeline SVG](docs/assets/graphs/inference_pipeline.svg)

See [Performance testing graphs](docs/assets/graphs/README.md) for context.

## License

APMoE is licensed under the MIT License. See [LICENSE](LICENSE) and [Licensing information](docs/operations/licensing.md). Referenced datasets are not redistributed and remain governed by their original licenses.

