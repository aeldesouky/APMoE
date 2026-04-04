"""Click-based command-line interface for the APMoE framework.

Commands
--------
``apmoe init [project-name]``
    Scaffold a new project directory with a config template and an example
    custom expert implementation.

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
          "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
        }
      }
    ],
    "experts": [
      {
        "name": "face_age_expert",
        "class": "custom_expert.FaceAgeExpert",
        "weights": "./weights/{package}_face_age_expert.pt",
        "modalities": ["image"]
      }
    ],
    "aggregation": {
      "strategy": "apmoe.aggregation.builtin.WeightedAverageAggregator"
    },
    "serving": {
      "host": "0.0.0.0",
      "port": 8000,
      "workers": 1,
      "log_level": "info"
    }
  }
}
"""

_EXPERT_TEMPLATE: str = '''\
"""Example custom expert for {project_name}.

Extend :class:`~apmoe.ExpertPlugin` to create your own age-prediction expert.
Register your class in ``config.json`` under the ``"experts"`` section.

Steps
-----
1. Implement :meth:`load_weights` to load your pretrained model from *path*.
2. Implement :meth:`predict` to run inference and return an
   :class:`~apmoe.ExpertOutput`.
3. Update ``"class"`` in ``config.json`` to point at this module and class.
"""

from __future__ import annotations

from apmoe import ExpertOutput, ExpertPlugin, ProcessedInput


class FaceAgeExpert(ExpertPlugin):
    """CNN-based age estimation expert that consumes the image modality.

    Replace the stub body with real model loading and inference code.
    """

    @property
    def name(self) -> str:
        return "face_age_expert"

    def declared_modalities(self) -> list[str]:
        """Return the list of modalities this expert requires.

        Returns:
            ``["image"]`` — this expert only consumes image data.
        """
        return ["image"]

    def load_weights(self, path: str) -> None:
        """Load pretrained model weights from *path*.

        Called once at bootstrap.  Replace the stub with real loading code.

        Args:
            path: Filesystem path to the weight file (e.g. a ``.pt`` file).
        """
        # Example: self._model = torch.load(path, map_location="cpu")
        # self._model.eval()
        self._loaded = True  # type: ignore[attr-defined]

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Run inference and return an age prediction.

        Args:
            inputs: Mapping of modality name to processed data.  The
                ``"image"`` key holds either an
                :class:`~apmoe.EmbeddingResult` (if an embedder was
                configured) or a :class:`~apmoe.ModalityData`
                (preprocessed image tensor).

        Returns:
            An :class:`~apmoe.ExpertOutput` with predicted age and confidence.
        """
        _image_data = inputs["image"]  # noqa: F841
        # Replace with real inference, e.g.:
        # embedding = image_data.embedding
        # age = float(self._model(embedding).item())
        predicted_age: float = 35.0  # stub
        confidence: float = 0.85  # stub

        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=self.declared_modalities(),
            predicted_age=predicted_age,
            confidence=confidence,
            metadata={"model": "FaceAgeExpert-stub"},
        )

    def get_info(self) -> dict[str, object]:
        """Return metadata about this expert.

        Returns:
            A JSON-serialisable dict with ``name`` and ``modalities`` keys.
        """
        return {
            "name": self.name,
            "modalities": self.declared_modalities(),
        }
'''

_README_TEMPLATE: str = """\
# {project_name}

An APMoE project for age prediction using Mixture of Experts.

## Quick Start

1. **Configure**: Edit `config.json` to point to your processor, cleaner,
   anonymizer, embedder, and expert implementations.

2. **Add weights**: Place pretrained weight files in the `weights/` directory
   (or update paths in `config.json`).

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

## Project Structure

```
{project_name}/
  config.json          # Framework configuration
  custom_expert.py     # Example ExpertPlugin implementation
  weights/             # Pretrained weight files (add .pt files here)
    .gitkeep           # Ensures weights/ is tracked when empty
  README.md            # This file
```

## Extending the Framework

- Subclass `ExpertPlugin` in `custom_expert.py`.
- Register your class in `config.json` under `"experts"`.
- See the [APMoE documentation](https://github.com/your-org/apmoe) for details.
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


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS)
@click.argument("project_name", default="my_apmoe_project", metavar="[PROJECT_NAME]")
def init(project_name: str) -> None:
    r"""Scaffold a new APMoE project directory.

    Creates PROJECT_NAME/ with a config template, an example expert
    implementation, an empty weights directory, and a README.

    \b
    Files created:
      config.json        — minimal configuration template
      custom_expert.py   — example ExpertPlugin implementation
      weights/           — directory for pretrained model files
      README.md          — quick-start instructions
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

    # Derive a Python-safe package name from the project name.
    package_name = project_name.replace("-", "_").replace(" ", "_")

    project_dir.mkdir(parents=True)
    (project_dir / "weights").mkdir()
    (project_dir / "weights" / ".gitkeep").write_text("", encoding="utf-8")

    config_content = _CONFIG_TEMPLATE.replace("{package}", package_name)
    (project_dir / "config.json").write_text(config_content, encoding="utf-8")

    expert_content = _EXPERT_TEMPLATE.replace("{project_name}", project_name)
    (project_dir / "custom_expert.py").write_text(expert_content, encoding="utf-8")

    readme_content = _README_TEMPLATE.replace("{project_name}", project_name)
    (project_dir / "README.md").write_text(readme_content, encoding="utf-8")

    click.echo(click.style(f"Created project '{project_name}/'", fg="green"))
    click.echo(f"  {project_name}/config.json        — edit to configure your components")
    click.echo(f"  {project_name}/custom_expert.py   — example ExpertPlugin implementation")
    click.echo(f"  {project_name}/weights/            — place pretrained .pt files here")
    click.echo(f"  {project_name}/README.md           — quick-start instructions")
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  cd {project_name}")
    click.echo("  apmoe validate --config config.json")
    click.echo("  apmoe serve --config config.json")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS)
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
    help="Override serving host (env: APMOE_SERVING_HOST).",
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
    r"""Load pretrained models and start the APMoE API server.

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


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS)
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
        "Input path.  A directory: files whose name stem matches a configured "
        "modality are used (e.g. 'visual.jpg' for the 'visual' modality).  "
        "A .json manifest file: maps modality names to file paths."
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
    r"""Run inference on local files.

    \b
    Directory input — scan for files whose name stem matches a configured
    modality name (e.g. 'visual.jpg' is used for the 'visual' modality):

      data/
        visual.jpg    ->  visual modality
        audio.wav     ->  audio modality

    JSON manifest input — a .json file that maps modality names to paths:

      {"visual": "face.jpg", "audio": "recording.wav"}

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

    if input_p.is_file() and input_p.suffix.lower() == ".json":
        # JSON manifest: {"modality": "path/to/file"}
        try:
            manifest: dict[str, str] = json.loads(input_p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            click.echo(
                click.style(f"Cannot read manifest '{input_p}': {exc}", fg="red"),
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
        click.echo(
            click.style(
                f"No matching files found in '{input_path}' for modalities "
                f"{sorted(configured_modalities)}.  "
                "Name files after their modality (e.g. 'visual.jpg' for 'visual').",
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


@cli.command(context_settings=_CLI_CONTEXT_SETTINGS)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the JSON configuration file.",
)
def validate(config: str) -> None:
    r"""Validate a configuration file and verify all components are ready.

    Checks:

    \b
      * JSON syntax and Pydantic schema correctness
      * All component classes can be resolved and imported
      * All expert weight files exist on disk
      * All expert plugins report as loaded
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
