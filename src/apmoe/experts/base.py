"""Abstract base class for expert plugins.

An *expert plugin* is a self-contained, pretrained age-prediction model.
Each expert declares which modalities it consumes, loads its own pretrained
weights at bootstrap time, and produces an :class:`~apmoe.core.types.ExpertOutput`
when :meth:`ExpertPlugin.predict` is called.

Key design principles
---------------------
- **Inference only**: experts arrive with pretrained weights.  No training loop
  exists in the framework.
- **Multi-modal capable**: an expert may declare one *or more* modalities.
  A single-modality expert receives one processed input; a multi-modal expert
  receives several and is free to combine them internally however it likes.
- **Typed inputs**: each input in the ``inputs`` dict is a
  :data:`~apmoe.core.types.ProcessedInput`, which is either a
  :class:`~apmoe.core.types.EmbeddingResult` (when an embedder is configured
  for that modality) or a :class:`~apmoe.core.types.ModalityData` (when no
  embedder is configured).

Implementing a custom expert::

    from apmoe.experts.base import ExpertPlugin
    from apmoe.core.types import ProcessedInput, ExpertOutput

    class MyAgeExpert(ExpertPlugin):
        @property
        def name(self) -> str:
            return "my_age_expert"

        def declared_modalities(self) -> list[str]:
            return ["visual"]

        def load_weights(self, path: str) -> None:
            import torch
            self._model = torch.load(path, map_location="cpu")
            self._model.eval()

        def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
            visual = inputs["visual"]
            age, conf = self._model(visual.embedding)
            return ExpertOutput(
                expert_name=self.name,
                consumed_modalities=["visual"],
                predicted_age=float(age),
                confidence=float(conf),
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from apmoe.core.types import ExpertOutput, ProcessedInput


class ExpertPlugin(ABC):
    """Abstract base class for age-prediction expert plugins.

    Every expert in the framework must subclass :class:`ExpertPlugin` and
    implement the four abstract members below.  The framework calls them in a
    strict lifecycle order:

    1. :meth:`declared_modalities` is called at **config validation** time to
       verify that each modality the expert needs is defined in the config.
    2. :meth:`load_weights` is called **once at bootstrap** after the config is
       validated.  The expert should load (and optionally warm-up) its
       pretrained model here.
    3. :meth:`predict` is called **at inference time** for every incoming
       request.
    4. :meth:`get_info` may be called at any time for diagnostics or the
       ``/info`` API endpoint.

    Attributes can be added freely to concrete subclasses; the framework only
    interacts with the abstract interface defined here.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this expert instance.

        This **must** match the ``name`` field in the corresponding
        :class:`~apmoe.core.config.ExpertConfig`.  The name is used as the
        key in logs, the ``/info`` endpoint, and
        :attr:`~apmoe.core.types.ExpertOutput.expert_name`.

        Returns:
            A non-empty string identifier, e.g. ``"face_age_expert"``.
        """

    @abstractmethod
    def declared_modalities(self) -> list[str]:
        """Return the list of modality names this expert consumes.

        The framework calls this method during **config validation** to ensure
        all required modalities are declared.  It is also called by the
        :class:`~apmoe.experts.registry.ExpertRegistry` to map experts to
        modality outputs.

        An expert may consume **one or more** modalities.  The modality names
        returned here must match the ``name`` fields declared in the
        :class:`~apmoe.core.config.APMoEConfig.modalities` list.

        Returns:
            An ordered list of modality name strings.  Must be non-empty.

        Example::

            def declared_modalities(self) -> list[str]:
                return ["visual", "audio"]  # multi-modal expert
        """

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load pretrained model weights from *path*.

        Called **once** by the framework at bootstrap time, after config
        validation.  The expert should store the loaded model (or any derived
        state) as instance attributes for use in :meth:`predict`.

        Implementors **must not** perform any inference here.  This method is
        only for initialisation / warm-loading.

        Args:
            path: Filesystem path to the weight file as specified in
                :attr:`~apmoe.core.config.ExpertConfig.weights`.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If the file is
                missing, corrupt, or incompatible with this expert.
        """

    @abstractmethod
    def predict(self, inputs: dict[str, ProcessedInput]) -> ExpertOutput:
        """Run inference and return an age prediction.

        Called **once per request** by the inference pipeline.  The framework
        provides exactly the modalities listed in :meth:`declared_modalities`
        as the keys of *inputs* (provided those modalities were present in the
        incoming request).

        Each value is a :data:`~apmoe.core.types.ProcessedInput` — either an
        :class:`~apmoe.core.types.EmbeddingResult` (when an embedder is
        configured for that modality) or a
        :class:`~apmoe.core.types.ModalityData` (when no embedder is
        configured).

        Multi-modal experts receive multiple keys and are responsible for
        combining the inputs internally however the model requires.

        Args:
            inputs: Dict mapping modality name → processed input.  Keys are a
                subset of :meth:`declared_modalities` (missing keys indicate
                that modality was absent from the current request).

        Returns:
            An :class:`~apmoe.core.types.ExpertOutput` containing the age
            estimate and confidence score.

        Raises:
            :class:`~apmoe.core.exceptions.ExpertError`: If inference fails.
        """

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def get_info(self) -> dict[str, object]:
        """Return diagnostic metadata about this expert.

        Exposed via the framework's ``GET /info`` endpoint and in log output.
        Override to add model architecture, version, or hyperparameter info.

        Returns:
            A JSON-serialisable dict.  The default implementation returns
            the expert name, consumed modalities, and class name.
        """
        return {
            "name": self.name,
            "modalities": self.declared_modalities(),
            "expert_class": type(self).__qualname__,
        }

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if :meth:`load_weights` has been called successfully.

        The default implementation always returns ``True`` (trusts the
        subclass).  Subclasses may override this to expose a proper health
        check (e.g. check that ``self._model is not None``).

        Returns:
            Boolean health status.
        """
        return True
