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
