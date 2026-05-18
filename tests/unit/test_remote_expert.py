"""Unit tests for apmoe.experts.remote.RemoteExpert."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from apmoe.core.exceptions import ExpertError
from apmoe.core.config import SecurityConfig
from apmoe.core.security import host_matches_allowlist
from apmoe.core.types import EmbeddingResult, ExpertOutput, ModalityData
from apmoe.experts.remote import (
    RemoteExpert,
    _apply_template,
    _expand_headers,
    _resolve_path,
    _serialise_input,
    _split_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_expert(**kwargs: Any) -> RemoteExpert:
    """Construct a RemoteExpert with sensible defaults, overridable via kwargs."""
    defaults: dict[str, Any] = {
        "expert_name": "test_remote",
        "modalities": ["keystroke"],
        "endpoint": "http://test-server/predict",
        "endpoint_headers": {},
        "endpoint_timeout": 5.0,
        "request_template": None,
        "response_mapping": None,
    }
    defaults.update(kwargs)
    return RemoteExpert(**defaults)


def _loaded_expert(**kwargs: Any) -> RemoteExpert:
    """Return a RemoteExpert that has already had load_weights called."""
    expert = _make_expert(**kwargs)
    with patch("httpx.post"):  # httpx import-check only; no actual call in load_weights
        expert.load_weights("")
    return expert


# ---------------------------------------------------------------------------
# _split_path
# ---------------------------------------------------------------------------


class TestSplitPath:
    def test_simple_key(self) -> None:
        assert _split_path("predicted_age") == ["predicted_age"]

    def test_dot_path(self) -> None:
        assert _split_path("result.age") == ["result", "age"]

    def test_array_index(self) -> None:
        assert _split_path("[0]") == ["[0]"]

    def test_array_then_key(self) -> None:
        assert _split_path("[0].age") == ["[0]", "age"]

    def test_nested_dot_path(self) -> None:
        assert _split_path("a.b.c") == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_top_level_key(self) -> None:
        assert _resolve_path({"predicted_age": 30.0}, "predicted_age") == 30.0

    def test_nested_key(self) -> None:
        data = {"result": {"age": 25.0}}
        assert _resolve_path(data, "result.age") == 25.0

    def test_array_index(self) -> None:
        data = [{"age": 40.0}]
        assert _resolve_path(data, "[0].age") == 40.0

    def test_array_index_only(self) -> None:
        data = [99.0]
        assert _resolve_path(data, "[0]") == 99.0

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            _resolve_path({"a": 1}, "b")


# ---------------------------------------------------------------------------
# _serialise_input
# ---------------------------------------------------------------------------


class TestSerialiseInput:
    def test_modality_data_dict(self) -> None:
        data = {"dur_8": [95.0, 102.0]}
        md = ModalityData(modality="keystroke", data=data)
        result = _serialise_input(md)
        assert result == data

    def test_modality_data_str(self) -> None:
        md = ModalityData(modality="keystroke", data="hello")
        assert _serialise_input(md) == "hello"

    def test_modality_data_list(self) -> None:
        md = ModalityData(modality="keystroke", data=[[8, 0, 95.0]])
        assert _serialise_input(md) == [[8, 0, 95.0]]

    def test_embedding_result(self) -> None:
        import numpy as np
        emb = EmbeddingResult(modality="image", embedding=np.array([0.1, 0.2, 0.3]))
        result = _serialise_input(emb)
        assert result == pytest.approx([0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# _expand_headers
# ---------------------------------------------------------------------------


class TestExpandHeaders:
    def test_literal_value_unchanged(self) -> None:
        result = _expand_headers({"Content-Type": "application/json"}, "test")
        assert result["Content-Type"] == "application/json"

    def test_env_var_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_TOKEN", "secret123")
        result = _expand_headers({"Authorization": "Bearer $MY_TOKEN"}, "test")
        assert result["Authorization"] == "Bearer secret123"

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ExpertError, match="MISSING_VAR"):
            _expand_headers({"X-Key": "$MISSING_VAR"}, "test")


# ---------------------------------------------------------------------------
# _apply_template
# ---------------------------------------------------------------------------


class TestApplyTemplate:
    def _ctx(self, **modalities: Any) -> dict[str, Any]:
        return {"expert_name": "my_expert", "modalities": modalities}

    def test_default_schema_passthrough(self) -> None:
        ctx = self._ctx(keystroke=[[8, 0, 95.0]])
        tmpl = {"expert_name": "{{expert_name}}", "data": "{{modalities.keystroke}}"}
        result = _apply_template(tmpl, ctx)
        assert result["expert_name"] == "my_expert"
        assert result["data"] == [[8, 0, 95.0]]

    def test_huggingface_style(self) -> None:
        ctx = self._ctx(keystroke={"dur_8": [95.0]})
        tmpl = {"inputs": "{{modalities.keystroke}}"}
        result = _apply_template(tmpl, ctx)
        assert result == {"inputs": {"dur_8": [95.0]}}

    def test_literal_values_pass_through(self) -> None:
        ctx = self._ctx(keystroke=[])
        tmpl = {"model": "age-v1", "version": 2}
        result = _apply_template(tmpl, ctx)
        assert result == {"model": "age-v1", "version": 2}

    def test_missing_modality_raises(self) -> None:
        ctx = self._ctx()
        tmpl = {"inputs": "{{modalities.image}}"}
        with pytest.raises(ExpertError, match="image"):
            _apply_template(tmpl, ctx)

    def test_unknown_placeholder_raises(self) -> None:
        ctx = self._ctx()
        tmpl = {"x": "{{unknown_field}}"}
        with pytest.raises(ExpertError, match="Unrecognised"):
            _apply_template(tmpl, ctx)

    def test_nested_dict(self) -> None:
        ctx = self._ctx(ks=[[1, 2, 3]])
        tmpl = {"outer": {"inner": "{{modalities.ks}}"}}
        result = _apply_template(tmpl, ctx)
        assert result == {"outer": {"inner": [[1, 2, 3]]}}

    def test_list_of_dicts(self) -> None:
        ctx = self._ctx(ks="hello")
        tmpl = [{"key": "{{modalities.ks}}"}]
        result = _apply_template(tmpl, ctx)
        assert result == [{"key": "hello"}]


# ---------------------------------------------------------------------------
# RemoteExpert — lifecycle
# ---------------------------------------------------------------------------


class TestRemoteExpertLifecycle:
    def test_name_matches_config(self) -> None:
        expert = _make_expert(expert_name="my_expert")
        assert expert.name == "my_expert"

    def test_declared_modalities(self) -> None:
        expert = _make_expert(modalities=["image", "keystroke"])
        assert expert.declared_modalities() == ["image", "keystroke"]

    def test_is_loaded_false_before_load_weights(self) -> None:
        expert = _make_expert()
        assert expert.is_loaded is False

    def test_load_weights_sets_loaded(self) -> None:
        expert = _make_expert()
        expert.load_weights("")
        assert expert.is_loaded is True

    def test_load_weights_raises_without_httpx(self) -> None:
        expert = _make_expert()
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ExpertError, match="httpx"):
                expert.load_weights("")

    def test_predict_before_load_raises(self) -> None:
        expert = _make_expert()
        md = ModalityData(modality="keystroke", data={})
        with pytest.raises(ExpertError, match="load_weights"):
            expert.predict({"keystroke": md})

    def test_get_info_includes_backend(self) -> None:
        expert = _make_expert()
        info = expert.get_info()
        assert info["backend"] == "remote"
        assert "endpoint" in info


# ---------------------------------------------------------------------------
# RemoteExpert — predict (mocked httpx)
# ---------------------------------------------------------------------------


def _mock_response(payload: Any) -> MagicMock:
    """Return a mock httpx.Response with a .json() method."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    resp.headers = {"content-type": "application/json"}
    return resp


class TestRemoteExpertPredict:
    def _keystroke_input(self) -> ModalityData:
        return ModalityData(modality="keystroke", data={"dur_8": [95.0]})

    def test_default_schema_success(self) -> None:
        """Default schema + flat response → correct ExpertOutput."""
        expert = _make_expert()
        expert.load_weights("")

        response_payload = {"predicted_age": 34.5, "confidence": 0.82, "metadata": {}}
        with patch("httpx.post", return_value=_mock_response(response_payload)) as mock_post:
            output = expert.predict({"keystroke": self._keystroke_input()})

        # Check what was sent
        call_kwargs = mock_post.call_args
        sent_body = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
        assert sent_body["expert_name"] == "test_remote"
        assert "keystroke" in sent_body["modalities"]

        assert isinstance(output, ExpertOutput)
        assert output.predicted_age == pytest.approx(34.5)
        assert output.confidence == pytest.approx(0.82)
        assert output.expert_name == "test_remote"

    def test_custom_request_template_huggingface(self) -> None:
        """HuggingFace-style template wraps the modality under 'inputs'."""
        expert = _make_expert(
            request_template={"inputs": "{{modalities.keystroke}}"},
            response_mapping={"predicted_age": "[0].age", "confidence": "[0].score"},
        )
        expert.load_weights("")

        response_payload = [{"age": 28.0, "score": 0.75}]
        with patch("httpx.post", return_value=_mock_response(response_payload)) as mock_post:
            output = expert.predict({"keystroke": self._keystroke_input()})

        sent_body = mock_post.call_args.kwargs.get("json", mock_post.call_args.args[1] if len(mock_post.call_args.args) > 1 else {})
        assert "inputs" in sent_body
        assert "expert_name" not in sent_body  # custom template — no default keys

        assert output.predicted_age == pytest.approx(28.0)
        assert output.confidence == pytest.approx(0.75)

    def test_missing_predicted_age_raises(self) -> None:
        expert = _make_expert()
        expert.load_weights("")

        with patch("httpx.post", return_value=_mock_response({"score": 0.9})):
            with pytest.raises(ExpertError, match="predicted_age"):
                expert.predict({"keystroke": self._keystroke_input()})

    def test_http_error_raises_expert_error(self) -> None:
        import httpx

        expert = _make_expert()
        expert.load_weights("")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        http_err = httpx.HTTPStatusError("500", request=MagicMock(), response=mock_resp)

        with patch("httpx.post", side_effect=http_err):
            with pytest.raises(ExpertError, match="HTTP 500"):
                expert.predict({"keystroke": self._keystroke_input()})

    def test_timeout_raises_expert_error(self) -> None:
        import httpx

        expert = _make_expert()
        expert.load_weights("")

        with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
            with pytest.raises(ExpertError, match="timed out"):
                expert.predict({"keystroke": self._keystroke_input()})

    def test_network_error_raises_expert_error(self) -> None:
        import httpx

        expert = _make_expert()
        expert.load_weights("")

        with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
            with pytest.raises(ExpertError, match="network error"):
                expert.predict({"keystroke": self._keystroke_input()})

    def test_non_json_response_raises(self) -> None:
        expert = _make_expert()
        expert.load_weights("")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(side_effect=ValueError("not JSON"))

        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(ExpertError, match="not valid JSON"):
                expert.predict({"keystroke": self._keystroke_input()})

    def test_confidence_defaults_to_minus_one_when_absent(self) -> None:
        expert = _make_expert()
        expert.load_weights("")

        with patch("httpx.post", return_value=_mock_response({"predicted_age": 40.0})):
            output = expert.predict({"keystroke": self._keystroke_input()})

        assert output.confidence == pytest.approx(-1.0)

    def test_metadata_includes_backend_and_endpoint(self) -> None:
        expert = _make_expert()
        expert.load_weights("")

        with patch(
            "httpx.post",
            return_value=_mock_response({"predicted_age": 30.0, "metadata": {"model_ver": "2"}}),
        ):
            output = expert.predict({"keystroke": self._keystroke_input()})

        assert output.metadata["backend"] == "remote"
        assert output.metadata["endpoint"] == "http://test-server/predict"

    def test_response_mapping_nested(self) -> None:
        """Dot-path response mapping resolves nested keys."""
        expert = _make_expert(
            response_mapping={"predicted_age": "output.age", "confidence": "output.conf"},
        )
        expert.load_weights("")

        payload = {"output": {"age": 52.1, "conf": 0.65}}
        with patch("httpx.post", return_value=_mock_response(payload)):
            output = expert.predict({"keystroke": self._keystroke_input()})

        assert output.predicted_age == pytest.approx(52.1)
        assert output.confidence == pytest.approx(0.65)

    def test_correct_headers_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolved headers are forwarded to httpx."""
        monkeypatch.setenv("MY_TOKEN", "tok123")
        expert = _make_expert(endpoint_headers={"Authorization": "Bearer $MY_TOKEN"})
        expert.load_weights("")

        with patch(
            "httpx.post",
            return_value=_mock_response({"predicted_age": 25.0}),
        ) as mock_post:
            expert.predict({"keystroke": self._keystroke_input()})

        sent_headers = mock_post.call_args.kwargs.get("headers", {})
        assert sent_headers.get("Authorization") == "Bearer tok123"

    def test_response_over_limit_raises(self) -> None:
        expert = _loaded_expert(endpoint_response_max_bytes=10)
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "application/json"}
        resp.content = b'{"predicted_age": 25, "extra": "too-large"}'
        with patch("httpx.post", return_value=resp):
            with pytest.raises(ExpertError, match="response exceeded"):
                expert.predict({"keystroke": self._keystroke_input()})

    def test_non_json_content_type_raises(self) -> None:
        expert = _loaded_expert()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "text/html"}
        resp.content = b"<html></html>"
        with patch("httpx.post", return_value=resp):
            with pytest.raises(ExpertError, match="Content-Type"):
                expert.predict({"keystroke": self._keystroke_input()})


class TestRemoteEndpointPolicy:
    def test_allowlist_exact_host_allows_https_endpoint(self) -> None:
        expert = _make_expert(
            endpoint="https://models.example.com/predict",
            security_config=SecurityConfig(remote_endpoint_allowlist=["models.example.com"]),
        )
        expert.load_weights("")
        assert expert.is_loaded is True

    def test_allowlist_wildcard_matches_subdomain(self) -> None:
        assert host_matches_allowlist("api.example.com", ["*.example.com"]) is True

    def test_disallowed_host_raises(self) -> None:
        expert = _make_expert(
            endpoint="https://evil.example.net/predict",
            security_config=SecurityConfig(remote_endpoint_allowlist=["models.example.com"]),
        )
        with pytest.raises(ExpertError, match="allowlist"):
            expert.load_weights("")

    def test_http_endpoint_rejected_when_https_enforced(self) -> None:
        expert = _make_expert(
            endpoint="http://models.example.com/predict",
            security_config=SecurityConfig(remote_endpoint_allowlist=["models.example.com"]),
        )
        with pytest.raises(ExpertError, match="HTTPS"):
            expert.load_weights("")

    def test_loopback_rejected_unless_private_networks_allowed(self) -> None:
        blocked = _make_expert(
            endpoint="http://127.0.0.1:8000/predict",
            security_config=SecurityConfig(remote_endpoint_allowlist=["*"]),
        )
        with pytest.raises(ExpertError, match="HTTPS"):
            blocked.load_weights("")

        allowed = _make_expert(
            endpoint="http://127.0.0.1:8000/predict",
            security_config=SecurityConfig(
                remote_endpoint_allowlist=["*"],
                remote_allow_private_networks=True,
            ),
        )
        allowed.load_weights("")
        assert allowed.is_loaded is True
