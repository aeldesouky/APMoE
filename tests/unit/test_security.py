"""Unit tests for framework security helpers."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta

import pytest

from apmoe.core.exceptions import ExpertError
from apmoe.core.security import (
    canonical_json,
    host_matches_allowlist,
    redact_headers,
    redact_url,
    validate_remote_url,
    verify_manifest_payload,
)


def _rsa_manifest_signature(payload: dict[str, object]) -> tuple[str, str]:
    cryptography = pytest.importorskip("cryptography")
    assert cryptography is not None

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    signature = private_key.sign(
        canonical_json(payload),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return public_pem, base64.b64encode(signature).decode("ascii")


def _manifest_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "expert_name": "remote_age",
        "model_id": "age-model",
        "model_version": "2026.05.18",
        "endpoint_origin": "https://models.example.com",
        "model_digest": "sha256:abc123",
        "issued_at": datetime.now(UTC).isoformat(),
        "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
    }
    payload.update(overrides)
    return payload


class TestRemoteUrlPolicy:
    def test_exact_and_wildcard_allowlist_matching(self) -> None:
        assert host_matches_allowlist("models.example.com", ["models.example.com"])
        assert host_matches_allowlist("api.models.example.com", ["*.models.example.com"])
        assert not host_matches_allowlist("models.example.com", ["*.models.example.com"])

    def test_allowed_https_endpoint_passes(self) -> None:
        validate_remote_url(
            "https://models.example.com/predict",
            allowlist=["*.example.com"],
            enforce_https=True,
            allow_private_networks=False,
            purpose="test",
        )

    def test_disallowed_host_fails(self) -> None:
        with pytest.raises(ExpertError, match="allowlist"):
            validate_remote_url(
                "https://evil.example.net/predict",
                allowlist=["models.example.com"],
                enforce_https=True,
                allow_private_networks=False,
                purpose="test",
            )

    def test_http_and_private_hosts_fail_by_default(self) -> None:
        with pytest.raises(ExpertError, match="HTTPS"):
            validate_remote_url(
                "http://models.example.com/predict",
                allowlist=["models.example.com"],
                enforce_https=True,
                allow_private_networks=False,
                purpose="test",
            )
        with pytest.raises(ExpertError, match="private"):
            validate_remote_url(
                "https://127.0.0.1/predict",
                allowlist=["*"],
                enforce_https=True,
                allow_private_networks=False,
                purpose="test",
            )

    def test_private_http_allowed_when_explicitly_enabled(self) -> None:
        validate_remote_url(
            "http://127.0.0.1:9000/predict",
            allowlist=["*"],
            enforce_https=True,
            allow_private_networks=True,
            purpose="test",
        )


class TestRedaction:
    def test_url_credentials_and_sensitive_query_params_redacted(self) -> None:
        redacted = redact_url(
            "https://user:pass@example.com/model?token=secret&safe=value&api_key=abc"
        )
        assert "user:pass" not in redacted
        assert "token=%2A%2A%2A" in redacted
        assert "api_key=%2A%2A%2A" in redacted
        assert "safe=value" in redacted

    def test_sensitive_headers_redacted(self) -> None:
        assert redact_headers({"Authorization": "Bearer x", "Accept": "json"}) == {
            "Authorization": "***",
            "Accept": "json",
        }


class TestSignedManifestVerification:
    def test_valid_rsa_pss_manifest_passes(self) -> None:
        payload = _manifest_payload()
        public_pem, signature = _rsa_manifest_signature(payload)
        verify_manifest_payload(
            {**payload, "signature": signature},
            public_key_pem=public_pem,
            expert_name="remote_age",
            endpoint_origin="https://models.example.com",
        )

    def test_invalid_signature_fails(self) -> None:
        payload = _manifest_payload()
        public_pem, signature = _rsa_manifest_signature(payload)
        manifest = {**payload, "signature": signature, "model_version": "tampered"}
        with pytest.raises(ExpertError, match="signature"):
            verify_manifest_payload(
                manifest,
                public_key_pem=public_pem,
                expert_name="remote_age",
                endpoint_origin="https://models.example.com",
            )

    def test_expired_manifest_fails(self) -> None:
        payload = _manifest_payload(
            expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        )
        public_pem, signature = _rsa_manifest_signature(payload)
        with pytest.raises(ExpertError, match="expired"):
            verify_manifest_payload(
                {**payload, "signature": signature},
                public_key_pem=public_pem,
                expert_name="remote_age",
                endpoint_origin="https://models.example.com",
            )

    def test_endpoint_origin_mismatch_fails(self) -> None:
        payload = _manifest_payload(endpoint_origin="https://other.example.com")
        public_pem, signature = _rsa_manifest_signature(payload)
        with pytest.raises(ExpertError, match="endpoint_origin"):
            verify_manifest_payload(
                {**payload, "signature": signature},
                public_key_pem=public_pem,
                expert_name="remote_age",
                endpoint_origin="https://models.example.com",
            )
