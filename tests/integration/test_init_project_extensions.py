"""Integration tests for extending projects created by ``apmoe init``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from apmoe.cli.main import cli
from apmoe.core.app import APMoEApp


_CUSTOM_COMPONENTS_TEMPLATE = '''\
from __future__ import annotations

import numpy as np

from apmoe.aggregation.base import AggregatorStrategy
from apmoe.core.types import EmbeddingResult, ExpertOutput, ModalityData, Prediction, ProcessedInput
from apmoe.experts.base import ExpertPlugin
from apmoe.modality.base import ModalityProcessor
from apmoe.processing.base import AnonymizerStrategy, CleanerStrategy, EmbedderStrategy


class CustomProcessor(ModalityProcessor):
    @property
    def modality_name(self) -> str:
        return "image"

    def validate(self, data: object) -> bool:
        return data is not None

    def preprocess(self, data: object) -> ModalityData:
        return ModalityData(
            modality="image",
            data={"raw": data, "steps": ["processor"]},
            metadata={"processor": "custom"},
        )


class CustomCleaner(CleanerStrategy):
    def clean(self, data: ModalityData) -> ModalityData:
        steps = [*data.data["steps"], "cleaner"]
        return data.with_data({"raw": data.data["raw"], "steps": steps})


class CustomAnonymizer(AnonymizerStrategy):
    def anonymize(self, data: ModalityData) -> ModalityData:
        steps = [*data.data["steps"], "anonymizer"]
        return data.with_data({"raw": data.data["raw"], "steps": steps})


class CustomEmbedder(EmbedderStrategy):
    def embed(self, data: ModalityData) -> EmbeddingResult:
        steps = [*data.data["steps"], "embedder"]
        return EmbeddingResult(
            modality=data.modality,
            embedding=np.array([1.0, 2.0, 3.0], dtype=float),
            metadata={"steps": steps},
        )


class CustomExpert(ExpertPlugin):
    def __init__(self) -> None:
        self._loaded = False
        self.loaded_from: str | None = None

    @property
    def name(self) -> str:
        return "custom_expert"

    def declared_modalities(self) -> list[str]:
        return ["image"]

    def load_weights(self, path: str) -> None:
        self.loaded_from = path
        self._loaded = True

    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        image_input = inputs["image"]
        if isinstance(image_input, EmbeddingResult):
            stage = "embedding"
            trace = image_input.metadata.get("steps", [])
        else:
            stage = "modality_data"
            trace = image_input.data.get("steps", [])
        return ExpertOutput(
            expert_name=self.name,
            consumed_modalities=["image"],
            predicted_age=41.5,
            confidence=0.8,
            metadata={"input_stage": stage, "trace": trace},
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class CustomAggregator(AggregatorStrategy):
    def aggregate(self, outputs: list[ExpertOutput]) -> Prediction:
        output = outputs[0]
        return Prediction(
            predicted_age=output.predicted_age + 1.0,
            confidence=output.confidence,
            per_expert_outputs=list(outputs),
            metadata={"aggregator": "custom", "expert_trace": output.metadata.get("trace", [])},
        )
'''


def _init_project(tmp_path: Path, name: str = "generated_project") -> Path:
    """Run ``apmoe init`` and return the created project path."""
    runner = CliRunner()
    project_dir = tmp_path / name
    result = runner.invoke(cli, ["init", str(project_dir)])
    assert result.exit_code == 0, result.output
    return project_dir


def _write_custom_components(project_dir: Path) -> str:
    """Write a unique module containing all custom component doubles."""
    module_name = f"custom_components_{abs(hash(str(project_dir)))}"
    (project_dir / f"{module_name}.py").write_text(
        _CUSTOM_COMPONENTS_TEMPLATE,
        encoding="utf-8",
    )
    return module_name


def _write_custom_config(
    project_dir: Path,
    module_name: str,
    *,
    use_custom_processor: bool = True,
    use_custom_cleaner: bool = True,
    use_custom_anonymizer: bool = True,
    use_custom_embedder: bool = False,
    use_custom_expert: bool = True,
    use_custom_aggregator: bool = True,
) -> Path:
    """Replace the generated config with a lightweight custom bootstrap config."""
    weights_dir = project_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    (weights_dir / "dummy.bin").write_bytes(b"dummy")

    config = {
        "apmoe": {
            "modalities": [
                {
                    "name": "image",
                    "processor": (
                        f"{module_name}.CustomProcessor"
                        if use_custom_processor
                        else "apmoe.modality.builtin.image.ImageProcessor"
                    ),
                    "pipeline": {
                        "cleaner": (
                            f"{module_name}.CustomCleaner"
                            if use_custom_cleaner
                            else "apmoe.processing.builtin.image_cleaners.ImageCleaner"
                        ),
                        "anonymizer": (
                            f"{module_name}.CustomAnonymizer"
                            if use_custom_anonymizer
                            else "apmoe.processing.builtin.image_anonymizers.ImageAnonymizer"
                        ),
                    },
                }
            ],
            "experts": [
                {
                    "name": "custom_expert",
                    "class": (
                        f"{module_name}.CustomExpert"
                        if use_custom_expert
                        else "apmoe.experts.builtin.FaceAgeExpert"
                    ),
                    "weights": "./weights/dummy.bin",
                    "modalities": ["image"],
                }
            ],
            "aggregation": {
                "strategy": (
                    f"{module_name}.CustomAggregator"
                    if use_custom_aggregator
                    else "apmoe.aggregation.builtin.WeightedAverageAggregator"
                )
            },
            "serving": {"host": "127.0.0.1", "port": 9000, "workers": 1},
        }
    }
    if use_custom_embedder:
        config["apmoe"]["modalities"][0]["pipeline"]["embedder"] = (
            f"{module_name}.CustomEmbedder"
        )

    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _bootstrap_project(
    project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    **config_kwargs: bool,
) -> tuple[APMoEApp, object]:
    """Create custom files, rewrite config, and bootstrap the app."""
    module_name = _write_custom_components(project_dir)
    config_path = _write_custom_config(project_dir, module_name, **config_kwargs)
    monkeypatch.syspath_prepend(str(project_dir))
    monkeypatch.chdir(project_dir)
    module = importlib.import_module(module_name)
    app = APMoEApp.from_config(config_path.name)
    return app, module


class TestInitProjectExtensions:
    """Coverage for extension points inside generated projects."""

    def test_generated_project_bootstraps_custom_processor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, module = _bootstrap_project(project_dir, monkeypatch)
        assert isinstance(app.pipeline.chains["image"].processor, module.CustomProcessor)

    def test_generated_project_bootstraps_custom_cleaner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, module = _bootstrap_project(project_dir, monkeypatch)
        assert isinstance(app.pipeline.chains["image"].cleaner, module.CustomCleaner)

    def test_generated_project_bootstraps_custom_anonymizer(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, module = _bootstrap_project(project_dir, monkeypatch)
        assert isinstance(app.pipeline.chains["image"].anonymizer, module.CustomAnonymizer)

    def test_generated_project_bootstraps_custom_embedder(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, module = _bootstrap_project(project_dir, monkeypatch, use_custom_embedder=True)
        assert isinstance(app.pipeline.chains["image"].embedder, module.CustomEmbedder)

    def test_generated_project_bootstraps_custom_expert(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, module = _bootstrap_project(project_dir, monkeypatch)
        expert = app.expert_registry.get("custom_expert")
        assert isinstance(expert, module.CustomExpert)
        assert expert.is_loaded is True
        assert expert.loaded_from == "./weights/dummy.bin"

    def test_generated_project_bootstraps_custom_aggregator(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, module = _bootstrap_project(project_dir, monkeypatch)
        assert isinstance(app.aggregator, module.CustomAggregator)

    def test_generated_project_supports_all_custom_components_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        project_dir = _init_project(tmp_path)
        app, _ = _bootstrap_project(project_dir, monkeypatch, use_custom_embedder=True)

        prediction = app.predict({"image": b"fake-image"})

        assert prediction.predicted_age == 42.5
        assert prediction.metadata["aggregator"] == "custom"
        assert prediction.metadata["expert_trace"] == [
            "processor",
            "cleaner",
            "anonymizer",
            "embedder",
        ]
        assert prediction.per_expert_outputs[0].metadata["input_stage"] == "embedding"

    def test_init_scaffold_includes_placeholders_for_all_extension_points(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression test for the current scaffold gap reported by the user."""
        project_dir = _init_project(tmp_path)

        expected_files = {
            "custom_processor.py",
            "custom_cleaner.py",
            "custom_anonymizer.py",
            "custom_embedder.py",
            "custom_expert.py",
            "custom_aggregator.py",
        }
        actual_files = {path.name for path in project_dir.iterdir() if path.is_file()}

        assert expected_files.issubset(actual_files)
