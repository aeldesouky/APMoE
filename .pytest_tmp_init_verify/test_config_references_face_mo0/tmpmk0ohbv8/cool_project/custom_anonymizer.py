"""Optional custom anonymizers for cool_project.

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
