# APMoE Developer Documentation

APMoE is an inference-only, Inversion-of-Control framework for age prediction with a Mixture of Experts architecture. Application code supplies processors, cleaning/anonymization strategies, expert plugins, and aggregators; the framework owns configuration loading, pipeline orchestration, prediction, validation, and HTTP serving.

## How It Works

```text
Application code
  -> defines or installs extension classes
  -> config.json names those classes or entry points
  -> APMoEApp resolves registries and loads weights
  -> InferencePipeline runs modality chains
  -> experts produce ExpertOutput records
  -> aggregator returns a Prediction
```

The current built-in demo path supports:

- `image` through `ImageProcessor`, `ImageCleaner`, `ImageAnonymizer`, and `FaceAgeExpert`.
- `keystroke` through `KeystrokeProcessor`, `KeystrokeCleaner`, `KeystrokeAnonymizer`, and `KeystrokeAgeExpert`.
- Remote HTTP/LLM experts through `RemoteExpert` and `LMStudioExpert`.

## Pipeline Data Flow

```text
Raw request values
  -> ModalityProcessor.preprocess()
  -> CleanerStrategy.clean()
  -> AnonymizerStrategy.anonymize()
  -> optional EmbedderStrategy.embed()
  -> ExpertPlugin.predict()
  -> AggregatorStrategy.aggregate()
  -> Prediction
```

Experts receive only the modalities they declare. A single expert can consume one modality or multiple modalities, but the framework does not fuse all modality embeddings before expert inference.

## Documentation Map

| Document | What it covers |
|---|---|
| [Documentation hub](../index.md) | Top-level navigation for all project docs |
| [User guide](../getting-started/user-guide.md) | End-to-end application workflow: install, scaffold, configure, fallback, Redis, serve |
| [Configuration reference](configuration.md) | JSON config schema, recipes, and environment overrides |
| [CLI reference](cli.md) | `init`, `download-models`, `serve`, `predict`, `validate`, and exit behavior |
| [Serving layer](serving.md) | FastAPI routes, middleware, auth, rate limiting, and versioned endpoints |
| [OpenAPI reference](openapi.md) | Swagger UI, ReDoc, raw schema URL, examples, response models, and documented headers |
| [Security reference](security.md) | Authn/authz, Redis stores, remote allowlists, model integrity, audit logs |
| [Testing strategy](testing.md) | Unit, boundary, integration, resilience, and load-test coverage |
| [Publishing guide](publishing.md) | Maintainer release flow for GitHub Actions and PyPI Trusted Publishing |
| [Developer experience guide](developer-experience.md) | Extensibility, diagnostics, remote/LLM integration, and DX notes |
| [Core types](core/types.md) | `ModalityData`, `EmbeddingResult`, `ExpertOutput`, and `Prediction` |
| [Core pipeline](core/pipeline.md) | `InferencePipeline` and `ModalityChain` execution |
| [Core app](core/app.md) | `APMoEApp` lifecycle, prediction API, validation, and serving |
| [Core registry](core/registry.md) | Component registration and dotted-path resolution |
| [Core exceptions](core/exceptions.md) | Error hierarchy and handling guidance |
| [Extension points](extension-points/index.md) | All application-owned interfaces and registration options |
| [Deployment guidance](../operations/deployment-sla-fallback.md) | Deployment profiles, fallback behavior, rollout, rollback, autoscaling, and SLA planning |
| [Licensing](../operations/licensing.md) | MIT license, dataset/model boundaries, and redistribution checklist |

## Quickstart

Install from PyPI:

```bash
pip install apmoe
```

Install for contributor work from this checkout:

```bash
pip install -e ".[dev]"
```

Create and validate a scaffold:

```bash
apmoe init my_app --download-models
cd my_app
apmoe validate --config config.json
```

Serve the API:

```bash
apmoe serve --config config.json
# http://localhost:8000/v1/predict
# http://localhost:8000/v1/health
# http://localhost:8000/v1/info
# http://localhost:8000/docs
```

For the full application-owner path, start with the [User guide](../getting-started/user-guide.md).

## Minimal Config Example

```json
{
  "apmoe": {
    "modalities": [
      {
        "name": "image",
        "processor": "apmoe.modality.builtin.image.ImageProcessor",
        "pipeline": {
          "cleaner": "apmoe.processing.builtin.image_cleaners.ImageCleaner",
          "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name": "face_age_expert",
        "class": "apmoe.experts.builtin.FaceAgeExpert",
        "weights": "./weights/face_age_expert.keras",
        "modalities": ["image"]
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
    }
  }
}
```

## Key Design Constraints

1. No pre-prediction fusion: each expert receives only the modalities it declares.
2. Experts are not restricted to one modality; multi-modal experts combine their own inputs internally.
3. Embedding is optional per modality; omit `pipeline.embedder` when an expert consumes preprocessed `ModalityData` directly.
4. APMoE loads and runs pretrained artifacts; it does not train models.
