"""Unit tests for apmoe.experts.providers.lmstudio.LMStudioExpert."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apmoe.core.exceptions import ExpertError
from apmoe.core.types import ExpertOutput, ModalityData
from apmoe.experts.providers.lmstudio import LMStudioExpert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_expert(**kwargs: Any) -> LMStudioExpert:
    """Construct a LMStudioExpert with sensible defaults."""
    defaults: dict[str, Any] = {
        "expert_name": "lm_studio_test",
        "modalities": ["image"],
        "endpoint": "http://127.0.0.1:1234/api/v1/chat",
        "endpoint_headers": {"Content-Type": "application/json"},
        "endpoint_timeout": 30.0,
        "request_template": {
            "model": "test-model",
            "system_prompt": "Return ONLY an integer age.",
            "input": "{{modalities.image}}",
        },
        "response_mapping": None,
    }
    defaults.update(kwargs)
    return LMStudioExpert(**defaults)


def _loaded_expert(**kwargs: Any) -> LMStudioExpert:
    """Return a LMStudioExpert that has already called load_weights."""
    expert = _make_expert(**kwargs)
    expert.load_weights("")
    return expert


def _lmstudio_response(
    message: str = "34",
    *,
    with_reasoning: bool = True,
    model_id: str = "google/gemma-4-e4b",
    input_tokens: int = 1920,
    output_tokens: int = 3,
    tokens_per_second: float = 52.7,
) -> dict[str, Any]:
    """Build a realistic LM Studio /api/v1/chat response payload."""
    output = []
    if with_reasoning:
        output.append({"type": "reasoning", "content": "Let me look at the image."})
    output.append({"type": "message", "content": message})
    return {
        "output": output,
        "stats": {
            "input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
        },
        "model_instance_id": model_id,
    }


def _mock_http_response(payload: Any) -> MagicMock:
    import json

    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-type": "application/json"}
    resp.content = json.dumps(payload).encode()
    resp.json = MagicMock(return_value=payload)
    return resp


def _image_input() -> ModalityData:
    return ModalityData(modality="image", data="base64encodedstring==")


# ---------------------------------------------------------------------------
# Class identity & registry
# ---------------------------------------------------------------------------


class TestLMStudioExpertIdentity:
    def test_is_subclass_of_remote_expert(self) -> None:
        from apmoe.experts.remote import RemoteExpert

        assert issubclass(LMStudioExpert, RemoteExpert)

    def test_registered_under_canonical_dotted_path(self) -> None:
        from apmoe.experts.registry import expert_registry

        key = "apmoe.experts.providers.lmstudio.LMStudioExpert"
        assert key in expert_registry.list_registered()

    def test_name_matches_config(self) -> None:
        expert = _make_expert(expert_name="my_lm_expert")
        assert expert.name == "my_lm_expert"

    def test_declared_modalities(self) -> None:
        expert = _make_expert(modalities=["image"])
        assert expert.declared_modalities() == ["image"]

    def test_get_info_backend_is_remote(self) -> None:
        expert = _make_expert()
        assert expert.get_info()["backend"] == "remote"


# ---------------------------------------------------------------------------
# _parse_response — happy paths
# ---------------------------------------------------------------------------


class TestParseResponseHappyPath:
    def _parse(
        self, expert: LMStudioExpert, payload: Any
    ) -> ExpertOutput:
        return expert._parse_response(payload, ["image"])

    def test_extracts_integer_age_from_message(self) -> None:
        expert = _loaded_expert()
        output = self._parse(expert, _lmstudio_response("34"))
        assert output.predicted_age == pytest.approx(34.0)

    def test_works_without_reasoning_item(self) -> None:
        expert = _loaded_expert()
        payload = _lmstudio_response("55", with_reasoning=False)
        output = self._parse(expert, payload)
        assert output.predicted_age == pytest.approx(55.0)

    def test_extracts_first_integer_from_verbose_content(self) -> None:
        """If the model returns 'The person is about 42 years old.' we get 42."""
        expert = _loaded_expert()
        payload = _lmstudio_response("The person is about 42 years old.")
        output = self._parse(expert, payload)
        assert output.predicted_age == pytest.approx(42.0)

    def test_confidence_is_minus_one(self) -> None:
        expert = _loaded_expert()
        output = self._parse(expert, _lmstudio_response("30"))
        assert output.confidence == pytest.approx(-1.0)

    def test_consumed_modalities_forwarded(self) -> None:
        expert = _loaded_expert()
        output = expert._parse_response(_lmstudio_response("28"), ["image", "keystroke"])
        assert output.consumed_modalities == ["image", "keystroke"]

    def test_expert_name_in_output(self) -> None:
        expert = _loaded_expert(expert_name="face_llm")
        output = self._parse(expert, _lmstudio_response("25"))
        assert output.expert_name == "face_llm"

    def test_metadata_includes_llm_response(self) -> None:
        expert = _loaded_expert()
        output = self._parse(expert, _lmstudio_response("67"))
        assert output.metadata["llm_response"] == "67"

    def test_metadata_includes_model_id(self) -> None:
        expert = _loaded_expert()
        payload = _lmstudio_response("40", model_id="meta/llama-3")
        output = self._parse(expert, payload)
        assert output.metadata["model"] == "meta/llama-3"

    def test_metadata_includes_tokens_per_sec(self) -> None:
        expert = _loaded_expert()
        payload = _lmstudio_response("40", tokens_per_second=99.5)
        output = self._parse(expert, payload)
        assert output.metadata["tokens_per_sec"] == pytest.approx(99.5)

    def test_metadata_includes_backend_and_endpoint(self) -> None:
        expert = _loaded_expert()
        output = self._parse(expert, _lmstudio_response("30"))
        assert output.metadata["backend"] == "remote"
        assert "endpoint" in output.metadata

    def test_metadata_includes_token_counts(self) -> None:
        expert = _loaded_expert()
        payload = _lmstudio_response("50", input_tokens=512, output_tokens=1)
        output = self._parse(expert, payload)
        assert output.metadata["input_tokens"] == 512
        assert output.metadata["output_tokens"] == 1

    def test_missing_stats_key_defaults_to_none(self) -> None:
        """A response without a 'stats' key should not crash."""
        expert = _loaded_expert()
        payload = {
            "output": [{"type": "message", "content": "35"}],
            "model_instance_id": "test-model",
            # 'stats' intentionally omitted
        }
        output = self._parse(expert, payload)
        assert output.predicted_age == pytest.approx(35.0)
        assert output.metadata["tokens_per_sec"] is None


# ---------------------------------------------------------------------------
# _parse_response — error cases
# ---------------------------------------------------------------------------


class TestParseResponseErrors:
    def _parse(self, expert: LMStudioExpert, payload: Any) -> ExpertOutput:
        return expert._parse_response(payload, ["image"])

    def test_empty_output_array_raises(self) -> None:
        expert = _loaded_expert()
        with pytest.raises(ExpertError, match="type='message'"):
            self._parse(expert, {"output": []})

    def test_only_reasoning_items_raises(self) -> None:
        expert = _loaded_expert()
        payload = {"output": [{"type": "reasoning", "content": "Hmm…"}]}
        with pytest.raises(ExpertError, match="type='message'"):
            self._parse(expert, payload)

    def test_message_with_no_integer_raises(self) -> None:
        expert = _loaded_expert()
        payload = {"output": [{"type": "message", "content": "unknown age"}]}
        with pytest.raises(ExpertError, match="no integer"):
            self._parse(expert, payload)

    def test_empty_message_content_raises(self) -> None:
        expert = _loaded_expert()
        payload = {"output": [{"type": "message", "content": ""}]}
        with pytest.raises(ExpertError, match="type='message'"):
            self._parse(expert, payload)

    def test_missing_output_key_raises(self) -> None:
        expert = _loaded_expert()
        with pytest.raises(ExpertError, match="type='message'"):
            self._parse(expert, {"result": "something"})


# ---------------------------------------------------------------------------
# Full predict integration (mocked HTTP)
# ---------------------------------------------------------------------------


class TestLMStudioExpertPredict:
    def test_full_predict_round_trip(self) -> None:
        """End-to-end: predict() → HTTP POST → _parse_response → ExpertOutput."""
        expert = _loaded_expert()
        payload = _lmstudio_response("45")

        with patch("httpx.post", return_value=_mock_http_response(payload)):
            output = expert.predict({"image": _image_input()})

        assert isinstance(output, ExpertOutput)
        assert output.predicted_age == pytest.approx(45.0)
        assert output.confidence == pytest.approx(-1.0)
        assert output.metadata["llm_response"] == "45"

    def test_request_body_uses_template(self) -> None:
        """The request body sent to the LLM follows the request_template."""
        expert = _loaded_expert()
        payload = _lmstudio_response("30")

        with patch("httpx.post", return_value=_mock_http_response(payload)) as mock_post:
            expert.predict({"image": _image_input()})

        sent = mock_post.call_args.kwargs["json"]
        assert "model" in sent
        assert "system_prompt" in sent
        # {{modalities.image}} should have been replaced with the actual data
        assert sent.get("input") == "base64encodedstring=="

    def test_env_var_expansion_in_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """$VAR in endpoint is resolved at bootstrap, not at predict time."""
        monkeypatch.setenv("TEST_LLM_URL", "http://resolved-host:9999/chat")
        expert = _make_expert(endpoint="$TEST_LLM_URL")
        expert.load_weights("")
        assert expert._endpoint == "http://resolved-host:9999/chat"

    def test_env_var_expansion_in_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """$VAR in request_template literals is resolved at bootstrap."""
        monkeypatch.setenv("TEST_MODEL_ID", "my-org/my-model-v2")
        expert = _make_expert(
            request_template={
                "model": "$TEST_MODEL_ID",
                "input": "{{modalities.image}}",
            }
        )
        expert.load_weights("")
        assert expert._request_template["model"] == "my-org/my-model-v2"  # type: ignore[index]
        # Predict-time placeholder should be preserved
        assert expert._request_template["input"] == "{{modalities.image}}"  # type: ignore[index]
