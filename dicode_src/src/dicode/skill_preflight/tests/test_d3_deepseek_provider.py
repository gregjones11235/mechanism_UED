"""Offline fake tests for the secure D3 DeepSeek provider adapter."""
from __future__ import annotations

import io
import json
import pickle
import sys
import traceback
import urllib.error
import uuid
from pathlib import Path

import pytest

EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[4]
    / "experiments"
    / "performance"
    / "llm_research_d"
)
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from d3_deepseek_provider import (  # noqa: E402
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_COMPLETION_REFERENCE,
    DEEPSEEK_LIST_MODELS_REFERENCE,
    DEEPSEEK_MODEL_ID,
    DEEPSEEK_PRICING_REFERENCE,
    DEEPSEEK_UPDATES_REFERENCE,
    OFFICIAL_REFERENCES,
    CredentialEchoError,
    CredentialMissingError,
    DeepSeekConfigError,
    DeepSeekMetadataClient,
    DeepSeekProviderConfig,
    EnvDeclaration,
    MetadataBudgetExceeded,
    MetadataGateBlocked,
    parse_env_file,
    public_json,
)


DECLARATION = EnvDeclaration(
    "DS_PROVIDER", "DS_BASE_URL", "DS_MODEL", "DS_API_CREDENTIAL"
)


def _unique_secret() -> str:
    return "credential-" + uuid.uuid4().hex + uuid.uuid4().hex + uuid.uuid4().hex


def _config(tmp_path: Path, secret: str) -> DeepSeekProviderConfig:
    env_path = tmp_path / "declared.env"
    env_path.write_text(
        "DS_PROVIDER=deepseek\n"
        f"DS_BASE_URL={DEEPSEEK_BASE_URL}\n"
        f"DS_MODEL={DEEPSEEK_MODEL_ID}\n"
        f"DS_API_CREDENTIAL={secret}\n",
        encoding="utf-8",
    )
    return DeepSeekProviderConfig.from_snapshot(parse_env_file(env_path), DECLARATION)


def _assert_sanitized_failure(error, secret: str, config: DeepSeekProviderConfig) -> None:
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    public_output = public_json(config) + json.dumps(
        {"reason": error.reason, "http_status": error.http_status}, sort_keys=True
    )
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in rendered
    assert secret not in public_output
    assert error.__cause__ is None
    assert error.__context__ is None


class _Response:
    status = 200

    def __init__(self, payload: object):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class _RawResponse(_Response):
    def __init__(self, body: bytes, status: int = 200):
        self.body = body
        self.status = status


def test_exact_constants_and_official_references():
    assert DEEPSEEK_MODEL_ID == "deepseek-v4-flash"
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert OFFICIAL_REFERENCES == (
        DEEPSEEK_LIST_MODELS_REFERENCE,
        DEEPSEEK_CHAT_COMPLETION_REFERENCE,
        DEEPSEEK_PRICING_REFERENCE,
        DEEPSEEK_UPDATES_REFERENCE,
    )
    assert OFFICIAL_REFERENCES == (
        "https://api-docs.deepseek.com/api/list-models",
        "https://api-docs.deepseek.com/api/create-chat-completion",
        "https://api-docs.deepseek.com/quick_start/pricing",
        "https://api-docs.deepseek.com/updates/",
    )


def test_secret_stays_private_and_cannot_be_publicly_serialized(tmp_path):
    secret = _unique_secret()
    env_path = tmp_path / "declared.env"
    env_path.write_text(
        "DS_PROVIDER='deepseek'\n"
        f'DS_BASE_URL="{DEEPSEEK_BASE_URL}"\n'
        f"DS_MODEL={DEEPSEEK_MODEL_ID}\n"
        f"DS_API_CREDENTIAL={secret}\n",
        encoding="utf-8",
    )
    snapshot = parse_env_file(env_path)
    config = DeepSeekProviderConfig.from_snapshot(snapshot, DECLARATION)
    visible = public_json(snapshot) + public_json(config) + repr(snapshot) + repr(config)
    assert secret not in visible
    assert config.credential_present is True
    assert not hasattr(config, "__dict__")
    with pytest.raises(TypeError):
        public_json({"credential": secret})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pickle.dumps(config)


@pytest.mark.parametrize(
    "hostile",
    [
        "export DS_API_CREDENTIAL=value\n",
        "EXPORT\tDS_API_CREDENTIAL=value\n",
        "DS_API_CREDENTIAL=$(read forbidden)\n",
        "DS_API_CREDENTIAL=`read forbidden`\n",
        'DS_API_CREDENTIAL="\\x60read forbidden\\x60"\n',
        'DS_API_CREDENTIAL="\\u0024\\u0028read forbidden)"\n',
        "DS_MODEL=first\nDS_MODEL=second\n",
    ],
)
def test_dotenv_rejects_hostile_shell_duplicate_and_decoded_forms(tmp_path, hostile):
    env_path = tmp_path / "hostile.env"
    env_path.write_text(hostile, encoding="utf-8")
    with pytest.raises(DeepSeekConfigError):
        parse_env_file(env_path)


def test_declaration_does_not_guess_and_exact_configuration_is_required(tmp_path):
    secret = _unique_secret()
    cases = (
        (
            "DS_PROVIDER=deepseek\n"
            f"DS_BASE_URL={DEEPSEEK_BASE_URL}\n"
            "DS_MODEL=deepseek-v4-pro\n"
            f"DS_API_CREDENTIAL={secret}\n"
        ),
        (
            "DS_PROVIDER=deepseek\n"
            "DS_BASE_URL=https://api.deepseek.com/v1\n"
            f"DS_MODEL={DEEPSEEK_MODEL_ID}\n"
            f"DS_API_CREDENTIAL={secret}\n"
        ),
        (
            f"DS_BASE_URL={DEEPSEEK_BASE_URL}\n"
            f"DS_MODEL={DEEPSEEK_MODEL_ID}\n"
            f"DS_API_CREDENTIAL={secret}\n"
        ),
    )
    env_path = tmp_path / "declared.env"
    for contents in cases:
        env_path.write_text(contents, encoding="utf-8")
        with pytest.raises(DeepSeekConfigError):
            DeepSeekProviderConfig.from_snapshot(parse_env_file(env_path), DECLARATION)


def test_missing_credential_fails_before_any_request(tmp_path):
    env_path = tmp_path / "declared.env"
    env_path.write_text(
        "DS_PROVIDER=deepseek\n"
        f"DS_BASE_URL={DEEPSEEK_BASE_URL}\n"
        f"DS_MODEL={DEEPSEEK_MODEL_ID}\n"
        "DS_API_CREDENTIAL=\n",
        encoding="utf-8",
    )
    with pytest.raises(CredentialMissingError):
        DeepSeekProviderConfig.from_snapshot(parse_env_file(env_path), DECLARATION)


def test_metadata_gate_uses_exactly_one_get_and_exact_model(tmp_path):
    secret = _unique_secret()
    config = _config(tmp_path, secret)
    calls = []

    def opener(request, timeout=0):
        calls.append(
            (request.full_url, request.get_method(), dict(request.header_items()), timeout)
        )
        return _Response({"object": "list", "data": [{"id": DEEPSEEK_MODEL_ID}]})

    client = DeepSeekMetadataClient(config, urlopen=opener)
    result = client.fetch_models(timeout_s=1.25)
    assert result.gate_status == "PASS"
    assert result.request_count == 1
    assert calls == [
        (
            "https://api.deepseek.com/models",
            "GET",
            {"Accept": "application/json", "Authorization": f"Bearer {secret}"},
            1.25,
        )
    ]
    with pytest.raises(MetadataBudgetExceeded):
        client.fetch_models()
    assert len(calls) == 1


def test_pro_metadata_does_not_satisfy_flash_gate_or_unlock_completion(tmp_path):
    client = DeepSeekMetadataClient(
        _config(tmp_path, _unique_secret()),
        urlopen=lambda *args, **kwargs: _Response({"data": [{"id": "deepseek-v4-pro"}]}),
    )
    with pytest.raises(MetadataGateBlocked) as caught:
        client.fetch_models()
    assert caught.value.reason == "model_missing"
    assert client.gate_passed is False
    with pytest.raises(MetadataGateBlocked) as blocked:
        client.build_chat_payload([{"role": "user", "content": "hello"}])
    assert blocked.value.reason == "completion_blocked_until_metadata_pass"


def test_unauthorized_and_other_http_fail_closed(tmp_path):
    config = _config(tmp_path, _unique_secret())

    def unauthorized(request, timeout=0):
        raise urllib.error.HTTPError(request.full_url, 401, "ignored", {}, None)

    with pytest.raises(MetadataGateBlocked) as caught:
        DeepSeekMetadataClient(config, urlopen=unauthorized).fetch_models()
    assert caught.value.reason == "unauthorized"
    assert caught.value.http_status == 401

    client = DeepSeekMetadataClient(config, urlopen=lambda *a, **k: _RawResponse(b"{}", 503))
    with pytest.raises(MetadataGateBlocked) as caught:
        client.fetch_models()
    assert caught.value.reason == "http_error"


def test_secret_bearing_http_error_is_sanitized_without_exception_chain(tmp_path):
    secret = _unique_secret()
    config = _config(tmp_path, secret)

    def unauthorized(request, timeout=0):
        body = json.dumps({"error": secret}).encode("utf-8")
        raise urllib.error.HTTPError(
            request.full_url, 401, secret, {}, io.BytesIO(body)
        )

    with pytest.raises(CredentialEchoError) as caught:
        DeepSeekMetadataClient(config, urlopen=unauthorized).fetch_models()
    _assert_sanitized_failure(caught.value, secret, config)


def test_transport_invalid_json_and_invalid_model_list_fail_closed(tmp_path):
    secret = _unique_secret()
    config = _config(tmp_path, secret)

    def transport(*args, **kwargs):
        raise urllib.error.URLError(secret)

    client = DeepSeekMetadataClient(config, urlopen=transport)
    with pytest.raises(MetadataGateBlocked) as caught:
        client.fetch_models()
    assert caught.value.reason == "transport_error"
    _assert_sanitized_failure(caught.value, secret, config)
    assert client.requests_used == 1

    for body, reason in ((b"{broken", "invalid_json"), (b'{"data":{}}', "invalid_model_list")):
        client = DeepSeekMetadataClient(
            config, urlopen=lambda *a, body=body, **k: _RawResponse(body)
        )
        with pytest.raises(MetadataGateBlocked) as caught:
            client.fetch_models()
        assert caught.value.reason == reason
        _assert_sanitized_failure(caught.value, secret, config)
        assert client.gate_passed is False


def test_plain_and_json_escaped_credential_echo_fail_closed_without_leak(tmp_path):
    secret = _unique_secret() + '"suffix'
    config = _config(tmp_path, secret)
    bodies = (secret.encode("utf-8"), json.dumps({"echo": secret}).encode("utf-8"))
    for body in bodies:
        client = DeepSeekMetadataClient(
            config, urlopen=lambda *a, body=body, **k: _RawResponse(body)
        )
        with pytest.raises(CredentialEchoError) as caught:
            client.fetch_models()
        _assert_sanitized_failure(caught.value, secret, config)
        assert client.gate_passed is False


@pytest.mark.parametrize(
    "body_template",
    [
        '{{"data":[{{"id":"{encoded}"}},{{"id":"deepseek-v4-flash"}}]}}',
        (
            '{{"meta":[{{"{encoded}":"safe"}}],'
            '"data":[{{"id":"deepseek-v4-flash"}}]}}'
        ),
        (
            '{{"meta":{{"safe":["prefix-{encoded}-suffix"]}},'
            '"data":[{{"id":"deepseek-v4-flash"}}]}}'
        ),
    ],
)
def test_fully_unicode_escaped_recursive_echo_never_enters_metadata(
    tmp_path, body_template
):
    secret = _unique_secret()
    config = _config(tmp_path, secret)
    encoded = "".join(f"\\u{ord(character):04x}" for character in secret)
    body = body_template.format(encoded=encoded).encode("ascii")
    client = DeepSeekMetadataClient(
        config, urlopen=lambda *args, **kwargs: _RawResponse(body)
    )
    with pytest.raises(CredentialEchoError) as caught:
        client.fetch_models()
    _assert_sanitized_failure(caught.value, secret, config)
    assert client.gate_passed is False
    assert client.requests_used == 1


def test_completion_requires_gate_and_payload_is_exact(tmp_path):
    config = _config(tmp_path, _unique_secret())
    messages = [
        {"role": "system", "content": "reason carefully"},
        {"role": "user", "content": "solve"},
    ]
    client = DeepSeekMetadataClient(config, urlopen=lambda *a, **k: _Response({"data": []}))
    with pytest.raises(MetadataGateBlocked):
        client.build_chat_payload(messages)

    passed = DeepSeekMetadataClient(
        config, urlopen=lambda *a, **k: _Response({"data": [{"id": DEEPSEEK_MODEL_ID}]})
    )
    passed.fetch_models()
    assert passed.build_chat_payload(messages) == {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "thinking": {"type": "enabled"},
    }
    serialized = json.dumps(passed.build_chat_payload(messages), sort_keys=True)
    assert "enable_thinking" not in serialized
    assert "chat_template_kwargs" not in serialized
    assert "extra_body" not in serialized
    assert "stream" not in serialized
