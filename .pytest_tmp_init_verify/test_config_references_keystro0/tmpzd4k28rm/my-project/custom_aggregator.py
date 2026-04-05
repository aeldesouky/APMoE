"""Optional custom aggregators for my-project.

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
