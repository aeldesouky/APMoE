"""Optional custom experts for D:\coding_projects\APMoE\.pytest_tmp_integration_verify\test_generated_project_bootstr1\generated_project.

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
