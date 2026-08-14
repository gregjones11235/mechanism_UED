"""Offline fake tests for the security-bounded Qwen D3 adapter."""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from d3_qwen_provider import (  # noqa: E402
    CredentialEchoError,
    CredentialMissingError,
    EnvDeclaration,
    MetadataBudgetExceeded,
    MetadataGateBlocked,
    QwenMetadataClient,
    QwenProviderConfig,
    QwenProviderError,
    parse_env_file,
    public_json,
)


SECRET = "qwen-test-secret-not-for-output"
DECL = EnvDeclaration("QWEN_PROVIDER", "QWEN_BASE_URL", "QWEN_MODEL", "QWEN_API_KEY")


def _config(tmp_path: Path, *, credential: str = SECRET) -> QwenProviderConfig:
    path = tmp_path / "experiment_llm.env"
    path.write_text(
        "QWEN_PROVIDER=qwen\n"
        "QWEN_BASE_URL=https://qwen.example/v1\n"
        "QWEN_MODEL=qwen-plus\n"
        f"QWEN_API_KEY={credential}\n",
        encoding="utf-8",
    )
    return QwenProviderConfig.from_snapshot(parse_env_file(path), DECL)


class _Response:
    status = 200

    def __init__(self, payload: object):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


class _RawResponse(_Response):
    def __init__(self, body: bytes):
        self._body = body


def test_env_loader_does_not_execute_shell_or_serialize_secret(tmp_path):
    path = tmp_path / "experiment_llm.env"
    path.write_text(
        "# values are data, not shell\n"
        "QWEN_PROVIDER='qwen'\n"
        "QWEN_BASE_URL=\"https://qwen.example/v1\"\n"
        "QWEN_MODEL=qwen-plus\n"
        f"QWEN_API_KEY={SECRET}\n",
        encoding="utf-8",
    )
    snapshot = parse_env_file(path)
    config = QwenProviderConfig.from_snapshot(snapshot, DECL)
    serialized = public_json(snapshot) + public_json(config)
    assert SECRET not in serialized
    assert "https://qwen.example" not in serialized
    assert "QWEN_API_KEY" in serialized
    assert config.credential_present is True


def test_loader_rejects_shell_syntax_and_export(tmp_path):
    path = tmp_path / "bad.env"
    path.write_text("export QWEN_API_KEY=bad\n", encoding="utf-8")
    with pytest.raises(QwenProviderError):
        parse_env_file(path)
    path.write_text("QWEN_API_KEY=$(cat /key)\n", encoding="utf-8")
    with pytest.raises(QwenProviderError):
        parse_env_file(path)
    path.write_text('QWEN_API_KEY="\\x60"\n', encoding="utf-8")
    with pytest.raises(QwenProviderError):
        parse_env_file(path)
    path.write_text('QWEN_API_KEY="\\x24("\n', encoding="utf-8")
    with pytest.raises(QwenProviderError):
        parse_env_file(path)


def test_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.env"
    path.write_text("QWEN_MODEL=one\nQWEN_MODEL=two\n", encoding="utf-8")
    with pytest.raises(QwenProviderError):
        parse_env_file(path)


def test_missing_credential_fails_before_metadata_request(tmp_path):
    path = tmp_path / "experiment_llm.env"
    path.write_text(
        "QWEN_PROVIDER=qwen\nQWEN_BASE_URL=https://qwen.example/v1\nQWEN_MODEL=qwen-plus\nQWEN_API_KEY=\n",
        encoding="utf-8",
    )
    with pytest.raises(CredentialMissingError):
        QwenProviderConfig.from_snapshot(parse_env_file(path), DECL)


def test_metadata_request_is_exactly_one_and_model_is_required(tmp_path):
    config = _config(tmp_path)
    calls = []

    def fake_urlopen(request, timeout=0):
        calls.append((request.full_url, dict(request.header_items()), timeout))
        return _Response({"data": [{"id": "qwen-plus"}]})

    client = QwenMetadataClient(config, urlopen=fake_urlopen)
    result = client.fetch_models(timeout_s=1.5)
    assert result.gate_status == "PASS"
    assert result.request_count == 1
    assert client.requests_used == 1
    assert calls[0][0] == "https://qwen.example/v1/models"
    assert calls[0][1]["Authorization"] == f"Bearer {SECRET}"
    with pytest.raises(MetadataBudgetExceeded):
        client.fetch_models()
    assert len(calls) == 1


def test_401_and_secret_echo_fail_closed(tmp_path):
    config = _config(tmp_path)

    def unauthorized(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    with pytest.raises(MetadataGateBlocked) as exc:
        QwenMetadataClient(config, urlopen=unauthorized).fetch_models()
    assert exc.value.reason == "unauthorized"

    def echoed(request, timeout=0):
        return _Response({"error": SECRET})

    with pytest.raises(CredentialEchoError):
        QwenMetadataClient(config, urlopen=echoed).fetch_models()


def test_transport_and_invalid_json_fail_closed(tmp_path):
    config = _config(tmp_path)

    def transport_error(request, timeout=0):
        raise urllib.error.URLError("connection refused")

    client = QwenMetadataClient(config, urlopen=transport_error)
    with pytest.raises(MetadataGateBlocked) as exc:
        client.fetch_models()
    assert exc.value.reason == "transport_error"
    assert client.requests_used == 1

    invalid = QwenMetadataClient(config, urlopen=lambda *a, **k: _RawResponse(b"{not-json"))
    with pytest.raises(MetadataGateBlocked) as exc:
        invalid.fetch_models()
    assert exc.value.reason == "invalid_json"
    assert invalid.requests_used == 1


def test_model_missing_blocks_and_completion_payload_requires_pass(tmp_path):
    config = _config(tmp_path)

    def missing(request, timeout=0):
        return _Response({"data": [{"id": "other-model"}]})

    client = QwenMetadataClient(config, urlopen=missing)
    with pytest.raises(MetadataGateBlocked) as exc:
        client.fetch_models()
    assert exc.value.reason == "model_missing"
    with pytest.raises(MetadataGateBlocked):
        client.build_chat_payload([{"role": "user", "content": "x"}])

    passed = QwenMetadataClient(config, urlopen=lambda *a, **k: _Response({"data": [{"id": "qwen-plus"}]}))
    passed.fetch_models()
    payload = passed.build_chat_payload([{"role": "user", "content": "x"}])
    assert payload["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 8192
