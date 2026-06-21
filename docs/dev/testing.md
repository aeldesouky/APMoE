# Testing Strategy

APMoE uses unit tests for individual modules, integration tests for module boundaries and end-to-end app bootstrap, plus standalone scripts for load and resilience checks.

## Test Layout

```text
tests/
  conftest.py
  unit/
    test_aggregation.py
    test_aggregation_builtin.py
    test_cli.py
    test_config.py
    test_experts.py
    test_image_modality.py
    test_keystroke_demo_remote_executor.py
    test_lmstudio_expert.py
    test_modality.py
    test_packaging.py
    test_pipeline.py
    test_processing.py
    test_registry.py
    test_remote_expert.py
    test_security.py
    test_serving.py
    test_types.py
  integration/
    test_app.py
    test_face_e2e.py
    test_init_project_extensions.py
    test_keystroke_e2e.py
    test_module_boundaries.py
```

## Test Layers

| Layer | Location | Purpose |
|---|---|---|
| Unit tests | `tests/unit/` | Validate module behavior with focused fakes and temporary files. |
| Boundary tests | `tests/integration/test_module_boundaries.py` | Check contracts between registries, processors, strategies, experts, pipeline, and app wiring. |
| App integration tests | `tests/integration/test_app.py` | Exercise `APMoEApp.from_config()`, prediction, validation, metadata, and error paths. |
| End-to-end model path tests | `tests/integration/test_face_e2e.py`, `test_keystroke_e2e.py` | Verify the built-in image and keystroke paths with repository artifacts or controlled fixtures. |
| Packaging tests | `tests/unit/test_packaging.py` | Guard package metadata, included artifacts, and install-facing behavior. |
| Security and serving tests | `tests/unit/test_security.py`, `test_serving.py` | Cover auth, authorization, rate limiting, audit events, headers, and route behavior. |
| Remote expert tests | `tests/unit/test_remote_expert.py`, `test_lmstudio_expert.py`, `test_keystroke_demo_remote_executor.py` | Cover remote request templating, response mapping, retries, circuit breaking, and demo executor behavior. |

## Running Tests

```bash
# Full suite
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# Integration tests only
uv run pytest tests/integration/

# With coverage
uv run pytest --cov --cov-report=term-missing

# Lint
uv run ruff check .
```

The project enforces an 80% coverage floor through `pyproject.toml`.

## Resilience Smoke Script

The repository includes an additional standalone smoke script:

```bash
python scripts/e2e_resilience.py
```

It checks skip-failed expert policy, remote retries and circuit breaking, and Redis fallback behavior using in-process fakes.

## Load-Test Scripts

Use these scripts against a running local server:

```bash
python scripts/load_test.py --url http://127.0.0.1:8000/v1/health --users 20 --duration 15
python scripts/load_test_predict.py http://127.0.0.1:8000/v1/predict 10 15
python scripts/load_test_multimodal.py http://127.0.0.1:8000/v1/predict 5 15
```

Performance graph outputs are documented in [../assets/graphs/README.md](../assets/graphs/README.md).

## Contributor Guidelines

- Keep unit tests focused on one module at a time.
- Use local fakes or temporary files instead of external services.
- Add integration coverage when changing config contracts, registry resolution, app bootstrap, serving routes, or cross-module data flow.
- Add or update E2E coverage when changing built-in face, keystroke, remote, or packaging behavior.
- Run `uv run pytest` before publishing or opening a release PR.
