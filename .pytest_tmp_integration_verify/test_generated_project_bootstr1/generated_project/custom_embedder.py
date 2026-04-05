"""Optional custom embedders for D:\coding_projects\APMoE\.pytest_tmp_integration_verify\test_generated_project_bootstr1\generated_project.

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
