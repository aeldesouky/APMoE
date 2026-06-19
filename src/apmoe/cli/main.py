"""Click-based command-line interface for the APMoE framework.

Commands
--------
``apmoe init [project-name]``
    Scaffold a new project directory with a config template and starter stubs
    for every major extension point.

``apmoe serve --config <path>``
    Load pretrained models from a JSON config file and start the FastAPI
    HTTP server.

``apmoe predict --config <path> --input <path>``
    Run inference on local files found in the input path.

``apmoe validate --config <path>``
    Validate a config file: schema correctness, weight file existence, and
    expert health.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
import click

# Short and long help flags for the group and every subcommand (Click defaults to ``--help`` only).
_CLI_CONTEXT_SETTINGS: dict[str, object] = {"help_option_names": ["-h", "--help"]}

if TYPE_CHECKING:
    from apmoe.core.types import Prediction

# ---------------------------------------------------------------------------
# Scaffolding templates (used by ``init``)
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE: str = """\
{
  "apmoe": {
    "modalities": [
      {
        "name": "image",
        "processor": "apmoe.modality.builtin.image.ImageProcessor",
        "pipeline": {
          "cleaner": "apmoe.processing.builtin.image_cleaners.ImageCleaner",
          "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer",
          "embedder": null
        }
      },
      {
        "name": "keystroke",
        "processor": "apmoe.modality.builtin.keystroke.KeystrokeProcessor",
        "pipeline": {
          "cleaner": "apmoe.processing.builtin.cleaners.KeystrokeCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.KeystrokeAnonymizer",
          "embedder": null
        }
      }
    ],
    "experts": [
      {
        "name": "face_age_expert",
        "class": "apmoe.experts.builtin.FaceAgeExpert",
        "weights": "./weights/face_age_expert.keras",
        "modalities": ["image"]
      },
      {
        "name": "keystroke_age_expert",
        "class": "apmoe.experts.builtin.KeystrokeAgeExpert",
        "weights": "./weights/keystroke_age_expert.onnx",
        "modalities": ["keystroke"]
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
    },
    "serving": {
      "host": "0.0.0.0",
      "port": 8000,
      "workers": 1,
      "log_level": "info",
      "cors_origins": ["*"],
      "rate_limit": null,
      "authentication_enabled": false,
      "authorization_enabled": false,
      "token_invalidation_store": "memory",
      "token_invalidation_redis_url": null,
      "rate_limit_store": "memory",
      "rate_limit_redis_url": null
    },
    "environment": "development",
    "security": {
      "remote_endpoint_allowlist": null,
      "remote_enforce_https": true,
      "remote_allow_private_networks": false,
      "remote_response_max_bytes": 1048576,
      "audit_enabled": true,
      "audit_success_events": true
    },
    "confidence_threshold": null,
    "expert_failure_policy": "fail_fast",
    "remote_fallback_policy": "transient_only",
    "remote_retry": {
      "max_attempts": 3,
      "initial_delay_s": 0.25,
      "max_delay_s": 2.0,
      "backoff_multiplier": 2.0,
      "jitter": true
    },
    "remote_circuit_breaker": {
      "enabled": true,
      "failure_threshold": 5,
      "recovery_timeout_s": 30.0
    }
  }
}
"""

_PROCESSOR_TEMPLATE: str = '''\
"""Optional custom modality processors for {project_name}.

Use this file when you want a custom :class:`~apmoe.modality.base.ModalityProcessor`.
Point a modality ``"processor"`` entry in ``config.json`` at
``"custom_processor.YourClassName"``.
"""

from __future__ import annotations

# Example imports when you add your own processor:
# from apmoe.core.types import ModalityData
# from apmoe.modality.base import ModalityProcessor
#
#
# class MyCustomProcessor(ModalityProcessor):
#     @property
#     def modality_name(self) -> str:
#         return "image"
#
#     def validate(self, data: object) -> bool:
#         return data is not None
#
#     def preprocess(self, data: object) -> ModalityData:
#         return ModalityData(modality=self.modality_name, data=data)
'''

_CLEANER_TEMPLATE: str = '''\
"""Optional custom cleaners for {project_name}.

Use this file when you want a custom :class:`~apmoe.processing.base.CleanerStrategy`.
Point a modality pipeline ``"cleaner"`` entry in ``config.json`` at
``"custom_cleaner.YourClassName"``.
"""

from __future__ import annotations

# Example imports when you add your own cleaner:
# from apmoe.core.types import ModalityData
# from apmoe.processing.base import CleanerStrategy
#
#
# class MyCustomCleaner(CleanerStrategy):
#     def clean(self, data: ModalityData) -> ModalityData:
#         return data
'''

_ANONYMIZER_TEMPLATE: str = '''\
"""Optional custom anonymizers for {project_name}.

Use this file when you want a custom :class:`~apmoe.processing.base.AnonymizerStrategy`.
Point a modality pipeline ``"anonymizer"`` entry in ``config.json`` at
``"custom_anonymizer.YourClassName"``.
"""

from __future__ import annotations

# Example imports when you add your own anonymizer:
# from apmoe.core.types import ModalityData
# from apmoe.processing.base import AnonymizerStrategy
#
#
# class MyCustomAnonymizer(AnonymizerStrategy):
#     def anonymize(self, data: ModalityData) -> ModalityData:
#         return data
'''

_EMBEDDER_TEMPLATE: str = '''\
"""Optional custom embedders for {project_name}.

Use this file when you want a custom :class:`~apmoe.processing.base.EmbedderStrategy`.
Point a modality pipeline ``"embedder"`` entry in ``config.json`` at
``"custom_embedder.YourClassName"``.
"""

from __future__ import annotations

# Example imports when you add your own embedder:
# import numpy as np
# from apmoe.core.types import EmbeddingResult, ModalityData
# from apmoe.processing.base import EmbedderStrategy
#
#
# class MyCustomEmbedder(EmbedderStrategy):
#     def embed(self, data: ModalityData) -> EmbeddingResult:
#         return EmbeddingResult(modality=data.modality, embedding=np.array([0.0]))
'''

_EXPERT_TEMPLATE: str = '''\
"""Optional custom experts for {project_name}.

The default ``config.json`` from ``apmoe init`` uses the built-in experts
(:class:`~apmoe.experts.builtin.FaceAgeExpert` and
:class:`~apmoe.experts.builtin.KeystrokeAgeExpert`) with the bundled weights in
``weights/`` — those run real Keras / ONNX inference.

Use this file when you want a **custom** :class:`~apmoe.ExpertPlugin`:
subclass it, implement ``load_weights`` and ``predict``, then set the
``"class"`` field in ``config.json`` to ``"custom_expert.YourClassName"``.
"""

from __future__ import annotations

# Example imports when you add your own expert:
# from apmoe import ExpertOutput, ExpertPlugin, ProcessedInput
#
#
# class MyCustomExpert(ExpertPlugin):
#     @property
#     def name(self) -> str:
#         return "my_custom_expert"
#
#     def declared_modalities(self) -> list[str]:
#         return ["image"]
#
#     def load_weights(self, path: str) -> None:
#         ...
#
#     def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
#         ...
'''

_AGGREGATOR_TEMPLATE: str = '''\
"""Optional custom aggregators for {project_name}.

Use this file when you want a custom :class:`~apmoe.aggregation.base.AggregatorStrategy`.
Point the aggregation ``"strategy"`` entry in ``config.json`` at
``"custom_aggregator.YourClassName"``.
"""

from __future__ import annotations

# Example imports when you add your own aggregator:
# from apmoe.aggregation.base import AggregatorStrategy
# from apmoe.core.types import ExpertOutput, Prediction
#
#
# class MyCustomAggregator(AggregatorStrategy):
#     def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
#         output = outputs[0]
#         return Prediction(
#             predicted_age=output.predicted_age,
#             confidence=output.confidence,
#             per_expert_outputs=list(outputs),
#         )
'''

_SECURITY_TEMPLATE: str = '''\
"""Optional custom authentication and authorization for {project_name}.

APMoE provides two security models — choose one:

1. **Stateless JWT/Bearer** (recommended for production):
   Implement :class:`~apmoe.serving.middleware.StatelessAuthProvider` and pass
   it to :func:`~apmoe.serving.app_factory.create_api` via ``security_provider``.
   Enable in ``config.json``:
       "serving": {{ "authentication_enabled": true, "authorization_enabled": true }}

2. **Legacy AuthPlugin** (simple binary allow/deny):
   Subclass :class:`~apmoe.serving.middleware.AuthPlugin` and pass it via
   ``auth_plugin`` in :func:`~apmoe.serving.app_factory.create_api`.
   (Mutually exclusive with the stateless provider.)

3. **Custom AuthorizationPolicy**:
   Subclass :class:`~apmoe.serving.middleware.AuthorizationPolicy` to define
   which JWT scopes are required for each route, then pass via
   ``authorization_policy`` in :func:`~apmoe.serving.app_factory.create_api`.
"""

from __future__ import annotations

# --- Example 1: JWT Bearer provider (stateless) ---
# from apmoe.serving.middleware import AuthContext, StatelessAuthProvider
# from starlette.requests import Request
#
# class MyJWTProvider(StatelessAuthProvider):
#     def authenticate(self, request: Request) -> AuthContext | None:
#         token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
#         # validate token, return None to reject
#         return None  # replace with real validation

# --- Example 2: Legacy AuthPlugin ---
# from apmoe.serving.middleware import AuthPlugin
# from starlette.requests import Request
#
# class MyAuthPlugin(AuthPlugin):
#     def authenticate(self, request: Request) -> bool:
#         return request.headers.get("X-API-Key") == "my-secret-key"

# --- Example 3: Custom scope-based AuthorizationPolicy ---
# from apmoe.serving.middleware import AuthContext, AuthorizationPolicy
# from starlette.requests import Request
#
# class MyAuthorizationPolicy(AuthorizationPolicy):
#     def authorize(self, context: AuthContext, request: Request) -> bool:
#         if request.url.path.startswith("/v1/predict"):
#             return "predict" in context.scopes
#         return True
'''

_README_TEMPLATE: str = """\
# {project_name}

An APMoE project for age prediction using Mixture of Experts.

The generated project uses **local-only experts by default**. Add remote experts
explicitly when you want APMoE to call external model endpoints.

## Quick Start

1. **Configure**: Edit `config.json` to point to your processor, cleaner,
   anonymizer, embedder, expert, and aggregator implementations.

2. **Weights**: Default models are already under `weights/`. Replace or add
   files there if you train your own checkpoints (and update paths in `config.json`).

3. **Validate**: Check the configuration is correct:
   ```
   apmoe validate --config config.json
   ```

4. **Serve**: Start the HTTP API:
   ```
   apmoe serve --config config.json
   ```

5. **Predict**: Run inference on local files:
   ```
   apmoe predict --config config.json --input data/
   ```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/predict` | Multimodal age prediction (versioned) |
| `GET`  | `/v1/health`  | Readiness/liveness probe (versioned) |
| `GET`  | `/v1/info`    | Framework metadata snapshot (versioned) |
| `POST` | `/predict`    | Legacy (deprecated, use `/v1/predict`) |
| `GET`  | `/health`     | Legacy health check |
| `GET`  | `/info`       | Legacy info |
| `GET`  | `/docs`       | Swagger UI (OpenAPI) |
| `GET`  | `/redoc`      | ReDoc (OpenAPI) |

## Security

By default (`authentication_enabled: false`) the server accepts all requests.
For production, enable JWT Bearer authentication:

```json
"serving": {{
  "authentication_enabled": true,
  "authorization_enabled": true,
  "token_invalidation_store": "redis",
  "token_invalidation_redis_url": "redis://localhost:6379/0"
}}
```

Then pass a `StatelessAuthProvider` (e.g. `JWTBearerAuthProvider`) to
`create_api()`. See `custom_security.py` for stubs.

Rate limiting:
```json
"serving": {{ "rate_limit": 60, "rate_limit_store": "memory" }}
```

CORS:
```json
"serving": {{ "cors_origins": ["https://myapp.com"] }}
```

Remote expert security (`apmoe.security`):
```json
"security": {{
  "remote_endpoint_allowlist": ["api.example.com"],
  "remote_enforce_https": true,
  "remote_allow_private_networks": false
}}
```

## Key Configuration Options

| Key | Default | Description |
|-----|---------|-------------|
| `environment` | `"development"` | One of: development, test, staging, production |
| `confidence_threshold` | `null` | Gate below which recommendations are added |
| `expert_failure_policy` | `"fail_fast"` | `"fail_fast"` or `"skip_failed"` |
| `remote_fallback_policy` | `"transient_only"` | paired remote-to-local fallback behavior |
| `remote_retry.max_attempts` | `3` | Retries for remote expert calls |
| `remote_circuit_breaker.enabled` | `true` | Circuit breaker for remote experts |

## Project Structure

```
{project_name}/
  config.json           # Framework configuration (built-in Keras + ONNX experts)
  custom_processor.py   # Optional: your own ModalityProcessor stubs
  custom_cleaner.py     # Optional: your own CleanerStrategy stubs
  custom_anonymizer.py  # Optional: your own AnonymizerStrategy stubs
  custom_embedder.py    # Optional: your own EmbedderStrategy stubs
  custom_expert.py      # Optional: your own ExpertPlugin (default config uses builtins)
  custom_aggregator.py  # Optional: your own AggregatorStrategy stubs
  custom_security.py    # Optional: your own AuthPlugin / StatelessAuthProvider stubs
  weights/              # face_age_expert.keras, keystroke_*.onnx, keystroke_constants.json
  README.md             # This file
```

## Extending the Framework

- Subclass `ModalityProcessor` in `custom_processor.py`.
- Subclass `CleanerStrategy` in `custom_cleaner.py`.
- Subclass `AnonymizerStrategy` in `custom_anonymizer.py`.
- Subclass `EmbedderStrategy` in `custom_embedder.py` when you need embeddings.
- Subclass `ExpertPlugin` in `custom_expert.py`.
- Subclass `AggregatorStrategy` in `custom_aggregator.py`.
- Implement `StatelessAuthProvider` or `AuthPlugin` in `custom_security.py`.
- Reference your custom classes in `config.json` with dotted paths like `"custom_expert.MyCustomExpert"`.
- See the [APMoE documentation](https://github.com/aeldesouky/APMoE/tree/main/docs) for details.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prediction_to_json(prediction: Prediction) -> str:
    """Serialize a :class:`~apmoe.core.types.Prediction` dataclass to JSON.

    Uses :func:`dataclasses.asdict` for conversion, with ``str`` as the
    fallback serialiser for any type that ``json`` cannot handle natively
    (e.g. ``numpy`` arrays, ``torch.Tensor``).

    Args:
        prediction: A :class:`~apmoe.core.types.Prediction` instance.

    Returns:
        A pretty-printed JSON string.
    """
    d = dataclasses.asdict(prediction)
    return json.dumps(d, indent=2, default=str)


def _expert_summary_mode(experts: list[Any]) -> str:
    """Return a concise local/remote mode label for CLI output."""
    active = [e for e in experts if not getattr(e, "fallback_only", False)]
    has_remote = any(getattr(e, "endpoint", None) is not None for e in active)
    has_local = any(getattr(e, "endpoint", None) is None for e in active)
    has_fallbacks = any(getattr(e, "fallback_expert", None) for e in experts)
    if has_fallbacks:
        return "mixed with local fallbacks"
    if has_remote and has_local:
        return "mixed local+remote"
    if has_remote:
        return "remote-only"
    return "local-only (default)"


def _fallback_parent_name(experts: list[Any], fallback_name: str) -> str | None:
    """Return the remote expert name that points at *fallback_name*, if any."""
    for expert in experts:
        if getattr(expert, "fallback_expert", None) == fallback_name:
            return getattr(expert, "name", None)
    return None


def _render_expert_summary(
    apmoe_cfg: Any,
    *,
    health: dict[str, bool] | None = None,
    err: bool = False,
) -> None:
    """Print a CLI-facing local/remote expert summary."""
    from apmoe.core.security import redact_url

    try:
        experts = list(apmoe_cfg.experts)
    except TypeError:
        return
    if not experts:
        click.echo("Expert mode: local-only (default)", err=err)
        click.echo("  (no experts configured)", err=err)
        return

    retry = getattr(apmoe_cfg, "remote_retry", None)
    circuit = getattr(apmoe_cfg, "remote_circuit_breaker", None)
    fallback_policy = getattr(apmoe_cfg, "remote_fallback_policy", "transient_only")

    click.echo(f"Expert mode: {_expert_summary_mode(experts)}", err=err)
    click.echo(f"Remote fallback policy: {fallback_policy}", err=err)
    for expert in experts:
        name = getattr(expert, "name", "<unknown>")
        endpoint = getattr(expert, "endpoint", None)
        fallback_only = bool(getattr(expert, "fallback_only", False))
        if fallback_only:
            parent = _fallback_parent_name(experts, name) or "(unpaired)"
            label = click.style("[local fallback]", fg="cyan")
            loaded_suffix = ""
            if health is not None:
                loaded_suffix = " loaded=" + ("yes" if health.get(name) else "no")
            click.echo(
                f"  {label} {name}: weights={getattr(expert, 'weights', None)} "
                f"fallback_for={parent} standby{loaded_suffix}",
                err=err,
            )
            continue

        if endpoint is not None:
            label = click.style("[remote]", fg="magenta")
            fallback = getattr(expert, "fallback_expert", None) or "(none)"
            retry_text = (
                f"retry={getattr(retry, 'max_attempts', '?')} attempts"
                if retry is not None
                else "retry=?"
            )
            circuit_text = (
                "circuit="
                + ("on" if getattr(circuit, "enabled", False) else "off")
                if circuit is not None
                else "circuit=?"
            )
            click.echo(
                f"  {label} {name}: endpoint={redact_url(str(endpoint))} "
                f"{retry_text} {circuit_text} fallback={fallback}",
                err=err,
            )
        else:
            label = click.style("[local]", fg="green")
            loaded_suffix = ""
            if health is not None:
                loaded_suffix = " loaded=" + ("yes" if health.get(name) else "no")
            click.echo(
                f"  {label} {name}: weights={getattr(expert, 'weights', None)}{loaded_suffix}",
                err=err,
            )

    warnings: list[str] = []
    for expert in experts:
        endpoint = getattr(expert, "endpoint", None)
        if endpoint is None:
            continue
        if getattr(expert, "fallback_expert", None) is None:
            warnings.append(
                f"Remote expert '{getattr(expert, 'name', '<unknown>')}' has no fallback_expert."
            )
        elif fallback_policy == "disabled":
            warnings.append(
                f"Remote expert '{getattr(expert, 'name', '<unknown>')}' has a fallback, "
                "but remote_fallback_policy='disabled'."
            )
    if health is not None:
        for expert in experts:
            if bool(getattr(expert, "fallback_only", False)) and not health.get(
                getattr(expert, "name", ""),
                False,
            ):
                warnings.append(
                    f"Fallback expert '{getattr(expert, 'name', '<unknown>')}' is not loaded."
                )
    if warnings:
        click.echo("Expert warnings:", err=err)
        for warning in warnings:
            click.echo(click.style(f"  Warning: {warning}", fg="yellow"), err=err)


def _should_download_demo_models(
    *,
    builtin: bool,
    download_models: bool | None,
) -> bool:
    """Return whether ``apmoe init`` should acquire demo model artifacts."""
    if builtin:
        return True
    if download_models is not None:
        return download_models
    if not sys.stdin.isatty():
        return False
    return click.confirm(
        "Download demo model artifacts for the built-in experts now?",
        default=False,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings=_CLI_CONTEXT_SETTINGS)
@click.version_option(package_name="apmoe")
def cli() -> None:
    """APMoE — Age Prediction using Mixture of Experts.

    Use this CLI to initialise new projects, start the API server,
    run batch inference, and validate your configuration.
    """


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS, short_help="Scaffold a new APMoE project directory.")
@click.argument("project_name", default="my_apmoe_project", metavar="[PROJECT_NAME]")
@click.option(
    "--builtin",
    is_flag=True,
    help=(
        "Download or copy demo face and keystroke models into the generated "
        "project's weights directory."
    ),
)
@click.option(
    "--download-models/--no-download-models",
    default=None,
    help=(
        "Explicitly choose whether to acquire demo model artifacts. "
        "When omitted in an interactive terminal, APMoE asks."
    ),
)
def init(project_name: str, builtin: bool, download_models: bool | None) -> None:
    """Scaffold a new APMoE project directory.

    Creates PROJECT_NAME/ with a local-only config template covering all framework
    features (serving, security, rate limiting, CORS, remote experts,
    circuit breaker, retry policy), starter stubs for every extension point,
    bundled default weights, and a README.

    \b
    Files created:
      config.json                        — minimal configuration template
      custom_processor.py                — placeholder for optional custom processors
      custom_cleaner.py                  — placeholder for optional custom cleaners
      custom_anonymizer.py               — placeholder for optional custom anonymizers
      custom_embedder.py                 — placeholder for optional custom embedders
      custom_expert.py                   — placeholder for optional custom experts
      custom_aggregator.py               — placeholder for optional custom aggregators
      weights/keystroke_age_expert.onnx  — default keystroke age model
      weights/keystroke_constants.json   — keystroke feature constants
      weights/face_age_expert.keras      — default face age model
      README.md                          — quick-start instructions
    """
    project_dir = Path(project_name)

    if project_dir.exists():
        click.echo(
            click.style(
                f"Error: directory '{project_name}' already exists. "
                "Choose a different name or remove the existing directory.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    project_dir.mkdir(parents=True)
    weights_dest = project_dir / "weights"
    weights_dest.mkdir()

    copied: list[str] = []
    should_download_models = _should_download_demo_models(
        builtin=builtin,
        download_models=download_models,
    )

    if should_download_models:
        from apmoe.core.exceptions import ConfigurationError
        from apmoe.core.models import download_model_artifacts

        try:
            copied = [
                path.name
                for path in download_model_artifacts(
                    weights_dest,
                    model="all",
                    force=False,
                    skip_existing=True,
                    install_model_package=True,
                )
            ]
        except ConfigurationError as exc:
            click.echo(click.style("Could not acquire demo model artifacts:", fg="red"), err=True)
            click.echo(f"  {exc}", err=True)
            click.echo(
                "Install `apmoe[models]`, allow release artifact downloads, set "
                "APMOE_MODEL_SOURCE_DIR, or rerun `apmoe download-models --dest "
                "weights` with network access.",
                err=True,
            )
            sys.exit(1)

    if not copied:
        # Fallback: write a .gitkeep so the directory is not entirely empty.
        (weights_dest / ".gitkeep").write_text("", encoding="utf-8")

    (project_dir / "config.json").write_text(_CONFIG_TEMPLATE, encoding="utf-8")

    template_files = {
        "custom_processor.py": _PROCESSOR_TEMPLATE,
        "custom_cleaner.py": _CLEANER_TEMPLATE,
        "custom_anonymizer.py": _ANONYMIZER_TEMPLATE,
        "custom_embedder.py": _EMBEDDER_TEMPLATE,
        "custom_expert.py": _EXPERT_TEMPLATE,
        "custom_aggregator.py": _AGGREGATOR_TEMPLATE,
        "custom_security.py": _SECURITY_TEMPLATE,
    }
    for filename, template in template_files.items():
        content = template.replace("{project_name}", project_name)
        (project_dir / filename).write_text(content, encoding="utf-8")

    readme_content = _README_TEMPLATE.replace("{project_name}", project_name)
    (project_dir / "README.md").write_text(readme_content, encoding="utf-8")

    click.echo(click.style(f"Created project '{project_name}/'", fg="green"))
    click.echo("Expert mode: local-only (default)")
    click.echo(f"  {project_name}/config.json          — full config (edit to configure)")
    click.echo(f"  {project_name}/custom_processor.py  — optional ModalityProcessor stubs")
    click.echo(f"  {project_name}/custom_cleaner.py    — optional CleanerStrategy stubs")
    click.echo(f"  {project_name}/custom_anonymizer.py — optional AnonymizerStrategy stubs")
    click.echo(f"  {project_name}/custom_embedder.py   — optional EmbedderStrategy stubs")
    click.echo(f"  {project_name}/custom_expert.py     — optional ExpertPlugin stubs")
    click.echo(f"  {project_name}/custom_aggregator.py — optional AggregatorStrategy stubs")
    click.echo(f"  {project_name}/custom_security.py   — optional AuthPlugin/StatelessAuthProvider stubs")
    if copied:
        for name in copied:
            click.echo(f"  {project_name}/weights/{name}")
    else:
        click.echo(f"  {project_name}/weights/             — place pretrained model files here")
    click.echo(f"  {project_name}/README.md            — quick-start + API endpoint reference")
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  apmoe validate --config config.json")
    click.echo("  apmoe serve --config config.json")
    click.echo()
    click.echo(click.style("Tip:", fg="cyan") + " authentication is disabled by default.")
    click.echo("  Set \"authentication_enabled\": true in config.json and implement")
    click.echo("  StatelessAuthProvider in custom_security.py for production use.")


# ---------------------------------------------------------------------------
# download-models
# ---------------------------------------------------------------------------


@cli.command(
    "download-models",
    context_settings=_CLI_CONTEXT_SETTINGS,
    short_help="Download or copy demo model artifacts.",
)
@click.option(
    "--dest",
    default="weights",
    type=click.Path(file_okay=False, dir_okay=True),
    show_default=True,
    help="Directory to place model artifacts in.",
)
@click.option(
    "--model",
    "model_name",
    default="all",
    type=click.Choice(["all", "face", "keystroke"]),
    show_default=True,
    help="Model artifact group to acquire.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files after downloading or copying.",
)
@click.option(
    "--skip-existing/--no-skip-existing",
    default=True,
    show_default=True,
    help="Skip files that already exist unless --force is set.",
)
@click.option(
    "--install-package/--no-install-package",
    default=True,
    show_default=True,
    help=(
        "Try installing the version-matched apmoe-models package from PyPI "
        "before falling back to release artifact URLs."
    ),
)
def download_models(
    dest: str,
    model_name: str,
    force: bool,
    skip_existing: bool,
    install_package: bool,
) -> None:
    """Download or copy demo model artifacts for built-in experts."""
    from apmoe.core.exceptions import ConfigurationError
    from apmoe.core.models import download_model_artifacts, selected_artifacts

    try:
        written = download_model_artifacts(
            dest,
            model=model_name,  # type: ignore[arg-type]
            force=force,
            skip_existing=skip_existing,
            install_model_package=install_package,
        )
    except ConfigurationError as exc:
        click.echo(click.style("Model acquisition failed:", fg="red"), err=True)
        click.echo(f"  {exc}", err=True)
        sys.exit(1)

    selected = selected_artifacts(model_name)  # type: ignore[arg-type]
    click.echo(click.style(f"Model directory ready: {Path(dest)}", fg="green"))
    if written:
        for path in written:
            click.echo(f"  wrote {path.name}")
    else:
        click.echo("  no files written; selected artifacts already exist")
    click.echo("Artifacts:")
    for artifact in selected:
        click.echo(f"  {artifact.filename} ({artifact.size_bytes} bytes)")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS, short_help="Load pretrained models and start the API server.")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the JSON configuration file.",
)
@click.option(
    "--host",
    default=None,
    help="Host to bind the API server to (e.g. 0.0.0.0 for all interfaces). "
    "Overrides config and APMOE_SERVING_HOST env.",
)
@click.option(
    "--port",
    "-p",
    default=None,
    type=int,
    help="Override the TCP port (env: APMOE_SERVING_PORT).",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help="Override the number of uvicorn workers (env: APMOE_SERVING_WORKERS).",
)
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Override the uvicorn log level (env: APMOE_SERVING_LOG_LEVEL).",
)
def serve(
    config: str,
    host: str | None,
    port: int | None,
    workers: int | None,
    log_level: str | None,
) -> None:
    """Load pretrained models and start the APMoE API server.

    The server exposes:

    \b
      POST /predict   — multimodal age prediction
      GET  /health    — readiness/liveness probe
      GET  /info      — framework metadata
      GET  /docs      — OpenAPI Swagger UI

    Command-line options override the corresponding values in the config file.
    """
    from apmoe.core.app import APMoEApp
    from apmoe.core.exceptions import APMoEError

    app_dir = str(Path(config).resolve().parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Apply CLI overrides via environment variables so load_config picks them up.
    if host is not None:
        os.environ["APMOE_SERVING_HOST"] = host
    if port is not None:
        os.environ["APMOE_SERVING_PORT"] = str(port)
    if workers is not None:
        os.environ["APMOE_SERVING_WORKERS"] = str(workers)
    if log_level is not None:
        os.environ["APMOE_SERVING_LOG_LEVEL"] = log_level

    try:
        app = APMoEApp.from_config(config)
    except APMoEError as exc:
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        sys.exit(1)

    serving_cfg = app.config.apmoe.serving
    _render_expert_summary(app.config.apmoe)
    click.echo()
    click.echo(
        click.style(
            f"Starting APMoE server on http://{serving_cfg.host}:{serving_cfg.port}",
            fg="green",
        )
    )
    click.echo(f"  Workers  : {serving_cfg.workers}")
    click.echo(f"  Log level: {serving_cfg.log_level}")
    click.echo(f"  Docs     : http://{serving_cfg.host}:{serving_cfg.port}/docs")
    click.echo()

    try:
        app.serve()
    except APMoEError as exc:
        click.echo(click.style(f"Server error: {exc}", fg="red"), err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS, short_help="Run inference on local files.")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the JSON configuration file.",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help=(
        "Input path. A directory: expects a test directory. If manifest.json "
        "exists, it will be used. Otherwise falls back to file stem matching "
        "(e.g. 'image.jpg' for 'image' modality). A .json manifest file: "
        "explicitly maps modality names to file paths."
    ),
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write the prediction JSON to this file instead of printing to stdout.",
)
def predict(config: str, input_path: str, output: str | None) -> None:
    """Run inference on local files.

    Directory input — expects a test directory containing files.
    If a `manifest.json` is provided inside the directory, APMoE will use it
    to map modality names to paths.
    If no manifest is provided, APMoE falls back to searching for files named
    exactly after the modalities (e.g. `image.jpg` or `keystroke.json`).

    \b
      data/
        manifest.json   (optional)
        image.jpg       -> image modality
        keystroke.json  -> keystroke modality

    JSON manifest input — you can also explicitly pass a .json file:

    \b
      {"image": "data/face.jpg", "keystroke": "data/typing.json"}

    The resulting Prediction is printed as JSON, or written to --output.
    """
    from apmoe.core.app import APMoEApp
    from apmoe.core.config import load_config
    from apmoe.core.exceptions import APMoEError

    app_dir = str(Path(config).resolve().parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Load config first so we know which modality names are configured.
    try:
        cfg = load_config(config)
    except APMoEError as exc:
        click.echo(click.style(f"Configuration error: {exc}", fg="red"), err=True)
        sys.exit(1)

    configured_modalities = {m.name for m in cfg.apmoe.modalities}
    input_p = Path(input_path)
    inputs: dict[str, Any] = {}

    manifest_file = None
    if input_p.is_file() and input_p.suffix.lower() == ".json":
        manifest_file = input_p
    elif input_p.is_dir() and (input_p / "manifest.json").is_file():
        manifest_file = input_p / "manifest.json"

    if manifest_file:
        # JSON manifest: {"modality": "path/to/file"}
        try:
            manifest: dict[str, str] = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            click.echo(
                click.style(f"Cannot read manifest '{manifest_file}': {exc}", fg="red"),
                err=True,
            )
            sys.exit(1)

        for modality, file_path_str in manifest.items():
            if modality not in configured_modalities:
                click.echo(
                    click.style(
                        f"Warning: manifest modality '{modality}' is not configured; skipping.",
                        fg="yellow",
                    ),
                    err=True,
                )
                continue
            file_p = Path(file_path_str)
            if not file_p.is_absolute():
                file_p = manifest_file.parent / file_p
            if not file_p.exists():
                click.echo(
                    click.style(f"Warning: '{file_p}' not found; skipping.", fg="yellow"),
                    err=True,
                )
                continue
            inputs[modality] = file_p.read_bytes()

    elif input_p.is_dir():
        # Directory: look for files whose stem matches a configured modality name.
        for file_p in sorted(input_p.iterdir()):
            if file_p.is_file() and file_p.stem in configured_modalities:
                inputs[file_p.stem] = file_p.read_bytes()

    else:
        click.echo(
            click.style(
                f"Error: --input must be a directory or a .json manifest, got '{input_p}'.",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)
    if not inputs:
        example_mod = sorted(configured_modalities)[0] if configured_modalities else "visual"
        click.echo(
            click.style(
                f"No matching files found in '{input_path}' for modalities "
                f"{sorted(configured_modalities)}. At least one modality file must be provided. "
                f"Name files after their modality (e.g. '{example_mod}.jpg' for '{example_mod}').",
                fg="red",
            ),
            err=True,
        )
        sys.exit(1)

    click.echo(f"Inputs detected: {sorted(inputs.keys())}", err=True)

    try:
        app = APMoEApp.from_config(config)
    except APMoEError as exc:
        click.echo(click.style(f"Bootstrap error: {exc}", fg="red"), err=True)
        sys.exit(1)

    _render_expert_summary(app.config.apmoe, err=True)

    try:
        result = app.predict(inputs)
    except APMoEError as exc:
        click.echo(click.style(f"Prediction error: {exc}", fg="red"), err=True)
        sys.exit(1)

    result_json = _prediction_to_json(result)

    if output is not None:
        output_p = Path(output)
        try:
            output_p.write_text(result_json, encoding="utf-8")
        except OSError as exc:
            click.echo(
                click.style(f"Cannot write to '{output}': {exc}", fg="red"),
                err=True,
            )
            sys.exit(1)
        click.echo(click.style(f"Result written to '{output}'.", fg="green"))
    else:
        click.echo(result_json)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS, short_help="Validate a configuration file and verify all components.")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the JSON configuration file.",
)
def validate(config: str) -> None:
    """Validate a configuration file and verify all components are ready.

    Checks:

    \b
      * JSON syntax and Pydantic schema correctness
      * All component classes can be resolved and imported
      * All expert weight files exist on disk
      * All expert plugins report as loaded
      * Security config coherence (authentication/authorization settings)
      * Remote expert endpoint URL format and allowlist policy
      * Serving config: CORS, rate limiting, token/rate stores
      * Environment, confidence_threshold, expert_failure_policy
      * Remote retry and circuit breaker policy
    """
    from apmoe.core.app import APMoEApp
    from apmoe.core.exceptions import APMoEError

    app_dir = str(Path(config).resolve().parent)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    # Bootstrap validates schema + resolves all component classes.
    try:
        app = APMoEApp.from_config(config)
    except APMoEError as exc:
        click.echo(click.style("Bootstrap failed:", fg="red"), err=True)
        click.echo(f"  {exc}", err=True)
        sys.exit(1)

    # App-level validate() checks weight files + expert health.
    try:
        report: dict[str, Any] = app.validate()
    except APMoEError as exc:
        click.echo(click.style("Validation failed:", fg="red"), err=True)
        click.echo(f"  {exc}", err=True)
        issues: list[str] = exc.context.get("issues", [])  # type: ignore[assignment]
        for issue in issues:
            click.echo(f"    * {issue}", err=True)
        sys.exit(1)

    click.echo(click.style("Configuration is valid.", fg="green"))
    click.echo()
    _render_expert_summary(app.config.apmoe, health=report["expert_health"])
    click.echo()

    # --- Expert health ---
    click.echo("Expert health:")
    health: dict[str, bool] = report["expert_health"]
    for name, loaded in health.items():
        status = (
            click.style("loaded", fg="green")
            if loaded
            else click.style("NOT LOADED", fg="red")
        )
        click.echo(f"  {name}: {status}")
    if not health:
        click.echo("  (no experts registered)")

    # --- Config summary ---
    apmoe_cfg = app.config.apmoe
    serving_cfg = apmoe_cfg.serving
    security_cfg = apmoe_cfg.security

    click.echo()
    click.echo("Configuration summary:")
    click.echo(f"  environment          : {apmoe_cfg.environment}")
    click.echo(f"  expert_failure_policy: {apmoe_cfg.expert_failure_policy}")
    click.echo(
        f"  confidence_threshold : "
        f"{apmoe_cfg.confidence_threshold if apmoe_cfg.confidence_threshold is not None else '(disabled)'}"
    )

    # --- Serving ---
    click.echo()
    click.echo("Serving:")
    click.echo(f"  host        : {serving_cfg.host}:{serving_cfg.port}")
    click.echo(f"  workers     : {serving_cfg.workers}")
    click.echo(f"  log_level   : {serving_cfg.log_level}")
    click.echo(f"  cors_origins: {serving_cfg.cors_origins}")
    click.echo(
        f"  rate_limit  : "
        f"{serving_cfg.rate_limit} req/min" if serving_cfg.rate_limit else "  rate_limit  : (disabled)"
    )

    # --- Security / Auth ---
    click.echo()
    click.echo("Security / Authentication:")
    auth_enabled = serving_cfg.authentication_enabled
    authz_enabled = serving_cfg.authorization_enabled
    click.echo(
        f"  authentication : "
        + (click.style("enabled", fg="green") if auth_enabled else click.style("disabled", fg="yellow"))
    )
    click.echo(
        f"  authorization  : "
        + (click.style("enabled", fg="green") if authz_enabled else click.style("disabled", fg="yellow"))
    )
    click.echo(f"  token_invalidation_store: {serving_cfg.token_invalidation_store}")
    click.echo(f"  rate_limit_store        : {serving_cfg.rate_limit_store}")

    # Warn about insecure combinations
    warnings: list[str] = []
    if apmoe_cfg.environment == "production" and not auth_enabled:
        warnings.append(
            "Production environment detected but authentication_enabled=false. "
            "Enable authentication for production deployments."
        )
    if authz_enabled and not auth_enabled:
        warnings.append(
            "authorization_enabled=true but authentication_enabled=false — "
            "authorization has no effect without authentication."
        )
    if auth_enabled and serving_cfg.token_invalidation_store == "memory" and serving_cfg.workers > 1:
        warnings.append(
            "token_invalidation_store='memory' with multiple workers — "
            "token invalidation will not be shared across workers. Use 'redis' for production."
        )
    if serving_cfg.rate_limit is not None and serving_cfg.rate_limit_store == "memory" and serving_cfg.workers > 1:
        warnings.append(
            "rate_limit_store='memory' with multiple workers — "
            "rate limits will not be coordinated across workers. Use 'redis' for production."
        )
    if serving_cfg.cors_origins == ["*"] and apmoe_cfg.environment == "production":
        warnings.append(
            "cors_origins=['*'] allows all origins. "
            "Restrict to specific domains in production."
        )

    # --- Remote experts ---
    remote_experts = [e for e in apmoe_cfg.experts if e.endpoint is not None]
    if remote_experts:
        click.echo()
        click.echo("Remote experts:")
        for e in remote_experts:
            click.echo(f"  {e.name}: {e.endpoint}")
        click.echo(f"  remote_enforce_https     : {security_cfg.remote_enforce_https}")
        click.echo(f"  remote_allow_private_nets: {security_cfg.remote_allow_private_networks}")
        allowlist = security_cfg.remote_endpoint_allowlist
        if allowlist:
            click.echo(f"  remote_endpoint_allowlist: {allowlist}")
        else:
            click.echo(f"  remote_endpoint_allowlist: " + click.style("(none — all hosts allowed)", fg="yellow"))
            if apmoe_cfg.environment == "production":
                warnings.append(
                    "Remote experts configured in production without an explicit "
                    "security.remote_endpoint_allowlist."
                )

    # --- Retry / Circuit breaker ---
    retry = apmoe_cfg.remote_retry
    cb = apmoe_cfg.remote_circuit_breaker
    click.echo()
    click.echo("Remote retry policy:")
    click.echo(f"  max_attempts     : {retry.max_attempts}")
    click.echo(f"  initial_delay_s  : {retry.initial_delay_s}")
    click.echo(f"  max_delay_s      : {retry.max_delay_s}")
    click.echo(f"  backoff_multiplier: {retry.backoff_multiplier}")
    click.echo(f"  jitter           : {retry.jitter}")
    click.echo()
    click.echo("Circuit breaker:")
    click.echo(f"  enabled          : " + (click.style("yes", fg="green") if cb.enabled else click.style("no", fg="yellow")))
    click.echo(f"  failure_threshold: {cb.failure_threshold}")
    click.echo(f"  recovery_timeout : {cb.recovery_timeout_s}s")

    # --- Audit ---
    click.echo()
    click.echo("Audit logging:")
    click.echo(f"  audit_enabled      : " + (click.style("yes", fg="green") if security_cfg.audit_enabled else click.style("no", fg="yellow")))
    click.echo(f"  audit_success_events: {security_cfg.audit_success_events}")

    # --- Print collected warnings ---
    if warnings:
        click.echo()
        click.echo(click.style("Warnings:", fg="yellow"))
        for w in warnings:
            click.echo(click.style(f"  ⚠  {w}", fg="yellow"))

