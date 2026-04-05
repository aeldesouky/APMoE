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
          "anonymizer": "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
        }
      },
      {
        "name": "keystroke",
        "processor": "apmoe.modality.builtin.keystroke.KeystrokeProcessor",
        "pipeline": {
          "cleaner": "apmoe.processing.builtin.cleaners.KeystrokeCleaner",
          "anonymizer": "apmoe.processing.builtin.anonymizers.KeystrokeAnonymizer"
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
      "log_level": "info"
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

_README_TEMPLATE: str = """\
# {project_name}

An APMoE project for age prediction using Mixture of Experts.

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

## Project Structure

```
{project_name}/
  config.json          # Framework configuration (built-in Keras + ONNX experts)
  custom_processor.py  # Optional: your own ModalityProcessor stubs
  custom_cleaner.py    # Optional: your own CleanerStrategy stubs
  custom_anonymizer.py # Optional: your own AnonymizerStrategy stubs
  custom_embedder.py   # Optional: your own EmbedderStrategy stubs
  custom_expert.py     # Optional: your own ExpertPlugin (default config uses builtins)
  custom_aggregator.py # Optional: your own AggregatorStrategy stubs
  weights/             # face_age_expert.keras, keystroke_*.onnx, keystroke_constants.json
  README.md            # This file
```

## Extending the Framework

- Subclass `ModalityProcessor` in `custom_processor.py`.
- Subclass `CleanerStrategy` in `custom_cleaner.py`.
- Subclass `AnonymizerStrategy` in `custom_anonymizer.py`.
- Subclass `EmbedderStrategy` in `custom_embedder.py` when you need embeddings.
- Subclass `ExpertPlugin` in `custom_expert.py`.
- Subclass `AggregatorStrategy` in `custom_aggregator.py`.
- Reference your custom classes in `config.json` with dotted paths like `"custom_expert.MyCustomExpert"`.
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

    Creates PROJECT_NAME/ with a config template wired to the built-in Keras
    and ONNX experts, bundled default weights, starter stubs for every major
    extension point, and a README.

    \b
    Files created:
      config.json                        — minimal configuration template
      custom_expert.py                   — placeholder for optional custom experts
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

    # Copy default pretrained models bundled with the package.
    _pkg_weights = Path(__file__).parent.parent / "weights"
    copied: list[str] = []
    if _pkg_weights.is_dir():
        for src_file in sorted(_pkg_weights.iterdir()):
            if src_file.is_file():
                shutil.copy2(src_file, weights_dest / src_file.name)
                copied.append(src_file.name)

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
    }
    for filename, template in template_files.items():
        content = template.replace("{project_name}", project_name)
        (project_dir / filename).write_text(content, encoding="utf-8")

    readme_content = _README_TEMPLATE.replace("{project_name}", project_name)
    (project_dir / "README.md").write_text(readme_content, encoding="utf-8")

    click.echo(click.style(f"Created project '{project_name}/'", fg="green"))
    click.echo(
        f"  {project_name}/custom_processor.py - optional custom ModalityProcessor stubs"
    )
    click.echo(
        f"  {project_name}/custom_cleaner.py   - optional custom CleanerStrategy stubs"
    )
    click.echo(
        f"  {project_name}/custom_anonymizer.py - optional custom AnonymizerStrategy stubs"
    )
    click.echo(
        f"  {project_name}/custom_embedder.py  - optional custom EmbedderStrategy stubs"
    )
    click.echo(
        f"  {project_name}/custom_aggregator.py - optional custom AggregatorStrategy stubs"
    )
    click.echo(f"  {project_name}/config.json        — edit to configure your components")
    click.echo(f"  {project_name}/custom_expert.py   — optional custom ExpertPlugin stubs")
    if copied:
        for name in copied:
            click.echo(f"  {project_name}/weights/{name}")
    else:
        click.echo(f"  {project_name}/weights/            — place pretrained model files here")
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
