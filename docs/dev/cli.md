# CLI Reference (`apmoe.cli.main`)

The APMoE CLI is the fastest way to scaffold projects, validate configuration,
run local predictions, and start the HTTP service.

For the end-to-end install and project workflow, see
[`../user_guide.md`](../user_guide.md).

---

## Command map

| Command | Purpose |
|---|---|
| `apmoe init [PROJECT_NAME]` | Scaffold a new project folder with starter files |
| `apmoe download-models --dest weights` | Acquire demo model artifacts for built-in experts |
| `apmoe serve --config <path>` | Bootstrap `APMoEApp` and run the FastAPI/uvicorn server |
| `apmoe predict --config <path> --input <path>` | Run local inference from files/manifest |
| `apmoe validate --config <path>` | Validate config + bootstrap + expert health |
| `apmoe --version` | Print installed framework version |

---

## `apmoe init`

Creates a starter project directory.

```bash
apmoe init my_apmoe_project
```

Generated structure:

```text
my_apmoe_project/
  config.json
  custom_processor.py
  custom_cleaner.py
  custom_anonymizer.py
  custom_embedder.py
  custom_expert.py
  custom_aggregator.py
  weights/
    .gitkeep
  README.md
```

Notes:
- If the target directory already exists, the command exits non-zero.
- Project names are normalized to Python-safe package names in templates
  (`my-project` -> `my_project`).
- Generated projects use local-only experts by default. The init output prints
  `Expert mode: local-only (default)`.
- In an interactive terminal, `apmoe init` asks whether to acquire demo model
  artifacts. Use `--download-models` or `--no-download-models` for
  non-interactive scripts.
- `apmoe init --builtin` uses the same acquisition helper as
  `apmoe download-models`. Source checkouts can copy local demo artifacts;
  PyPI wheels copy the demo artifacts bundled in the installed `apmoe` package,
  then fall back to release-hosted artifact URLs if package resources are not
  available.

---

## `apmoe download-models`

Copies or downloads demo model artifacts for the built-in experts.

```bash
apmoe download-models --dest weights --model all
```

Options:

- `--model all|face|keystroke` selects the artifact group.
- `--dest <dir>` chooses the output directory.
- `--force` overwrites existing files after acquisition.
- `--skip-existing/--no-skip-existing` controls existing-file behavior.

The command verifies SHA-256 checksums after every copy or download. PyPI
wheels include model binaries in the main `apmoe` package; by default the
command copies those bundled resources. Configure `APMOE_MODEL_SOURCE_DIR` to
point at a directory containing the expected filenames, set
`APMOE_MODEL_SOURCE_REF` to override the Git ref used for release URL fallback,
or use per-artifact variables such as `APMOE_MODEL_SOURCE_FACE`.

---

## `apmoe serve`

Bootstraps the framework from config, then starts HTTP serving.

```bash
apmoe serve --config config.json
```

Optional overrides:

```bash
apmoe serve --config config.json --host 127.0.0.1 --port 9000 --workers 2 --log-level debug
```

The command applies CLI overrides through environment variables before loading
config:

- `--host` -> `APMOE_SERVING_HOST`
- `--port` -> `APMOE_SERVING_PORT`
- `--workers` -> `APMOE_SERVING_WORKERS`
- `--log-level` -> `APMOE_SERVING_LOG_LEVEL`

Before starting uvicorn, `serve` prints an expert summary with `[local]`,
`[remote]`, and `[local fallback]` labels, remote fallback policy, endpoint
redaction, and fallback warnings.

Endpoints exposed by the server:
- `POST /predict`
- `GET /health`
- `GET /info`
- `GET /docs`

---

## `apmoe predict`

Runs inference without starting the API server.

```bash
apmoe predict --config config.json --input data/
```

Supported input modes:

1. Directory mode: file stem must match modality name.
   - Example: `image.jpg` is used for modality `image`.
2. JSON manifest mode (`.json`): maps modality names to file paths.
   - Example: `{"image": "face.jpg", "keystroke": "session.json"}`

Output behavior:
- Without `--output`, prediction JSON is printed to stdout.
- With `--output`, JSON is written to the provided file.
- Expert summary and input diagnostics are printed to stderr so stdout remains
  valid JSON for shell pipelines.

Rules and edge cases:
- Unknown modalities in a manifest are skipped with a warning.
- Missing manifest files are skipped with a warning.
- If no usable modality inputs are found, command exits non-zero.

---

## `apmoe validate`

Checks that a config is ready for production inference.

```bash
apmoe validate --config config.json
```

Validation stages:
1. Config file exists and parses as JSON.
2. Pydantic schema validation passes.
3. All configured classes are resolvable/importable (`APMoEApp.from_config`).
4. Expert weight files exist and experts report healthy (`app.validate()`).

The command prints expert mode, backend labels, fallback pairings, per-expert
load status, and exits non-zero on failure.

---

## Exit behavior

- Success paths exit `0`.
- Bootstrap, validation, and prediction errors are caught as `APMoEError`
  subclasses and reported as user-facing messages.
- Invalid CLI arguments or missing required options are handled by Click
  with a non-zero exit code.
