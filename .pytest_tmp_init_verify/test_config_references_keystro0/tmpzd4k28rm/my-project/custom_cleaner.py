"""Optional custom cleaners for my-project.

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
