# CLI Reference (`apmoe.cli.main`)

The APMoE CLI is the fastest way to scaffold projects, validate configuration,
run local predictions, and start the HTTP service.

---

## Command map

| Command | Purpose |
|---|---|
| `apmoe init [PROJECT_NAME]` | Scaffold a new project folder with starter files |
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
  custom_expert.py
  weights/
    .gitkeep
  README.md
```

Notes:
- If the target directory already exists, the command exits non-zero.
- Project names are normalized to Python-safe package names in templates
  (`my-project` -> `my_project`).

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
   - Example: `visual.jpg` is used for modality `visual`.
2. JSON manifest mode (`.json`): maps modality names to file paths.
   - Example: `{"visual": "face.jpg", "audio": "clip.wav"}`

Output behavior:
- Without `--output`, prediction JSON is printed to stdout.
- With `--output`, JSON is written to the provided file.

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

The command prints per-expert load status and exits non-zero on failure.

---

## Exit behavior

- Success paths exit `0`.
- Bootstrap, validation, and prediction errors are caught as `APMoEError`
  subclasses and reported as user-facing messages.
- Invalid CLI arguments or missing required options are handled by Click
  with a non-zero exit code.
