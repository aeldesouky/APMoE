"""Optional custom modality processors for D:\coding_projects\APMoE\.pytest_tmp_integration_verify\test_init_scaffold_includes_pl0\generated_project.

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
