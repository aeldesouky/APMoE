# APMoE Documentation

This is the documentation hub for the APMoE project. Start with the README for
submission-level context, then use these folders for implementation details.

## Getting Started

- [User guide](getting-started/user-guide.md): install, scaffold, configure,
  validate, predict, and serve an APMoE app.

## Architecture

- [System architecture](architecture/system-architecture.md): high-level
  components, security boundaries, and data flow.
- [Dataflow design](architecture/dataflow-design.md): request lifecycle and
  core data structures.
- [Inference pipeline diagram](architecture/inference-pipeline-diagram.md):
  Mermaid source and presenter notes.
- [System design diagram](architecture/system-design.md): compact strategy and
  plugin flow diagram.

## Integrations

- [Keystroke integration](integrations/keystroke-integration.md): keystroke
  formats, API examples, model behavior, and troubleshooting.
- [Face integration](integrations/face-integration.md): image model inputs,
  preprocessing, outputs, and integration notes.
- [AI integration guide](integrations/ai-integration-guide.md): historical
  handoff checklist for model artifacts and inference specifications.
- [Keystroke remote executor](../remote_executors/keystroke_demo/README.md):
  reference remote executor used by tests and demos.

## Operations

- [Deployment, SLA, and fallback guidance](operations/deployment-sla-fallback.md):
  serverless and dedicated deployment patterns, fallback behavior, rollout, and
  SLA planning.
- [Licensing information](operations/licensing.md): MIT license, dataset
  boundaries, model artifact guidance, and redistribution checklist.
- [Performance graphs](assets/graphs/README.md): benchmark context and graph
  image links.

## Developer Reference

- [Developer documentation](dev/index.md): framework internals and contributor
  map.
- [CLI reference](dev/cli.md)
- [Configuration reference](dev/configuration.md)
- [Serving layer](dev/serving.md)
- [OpenAPI reference](dev/openapi.md)
- [Security reference](dev/security.md)
- [Testing strategy](dev/testing.md)
- [Extension points](dev/extension-points/index.md)
- [Core module reference](dev/core/index.md)
- [Developer experience guide](dev/developer-experience.md)
- [Publishing guide](dev/publishing.md)


