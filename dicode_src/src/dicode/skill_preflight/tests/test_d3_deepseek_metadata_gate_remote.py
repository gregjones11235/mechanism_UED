"""Offline tests for the audited DeepSeek metadata-only gate wrapper."""
from __future__ import annotations

import io
import json
import sys
import urllib.error
import uuid
from datetime import datetime, timezone
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

import d3_deepseek_metadata_gate_remote as gate  # noqa: E402


FIXED_NOW = datetime(2026, 8, 14, 3, 4, 5, tzinfo=timezone.utc)


def _secret() -> str:
    return "credential-" + uuid.uuid4().hex + uuid.uuid4().hex + uuid.uuid4().hex


def _env(tmp_path: Path, secret: str, *, model: str = "deepseek-v4-flash") -> Path:
    path = tmp_path / "experiment.env"
    path.write_text(
        "EXP_DEEPSEEK_PROVIDER=deepseek\n"
        "EXP_DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
        f"EXP_DEEPSEEK_MODEL={model}\n"
        f"EXP_DEEPSEEK_API_KEY={secret}\n",
        encoding="utf-8",
    )
    return path


class _Response:
    status = 200

    def __init__(self, payload=None, *, body: bytes | None = None):
        self._body = body if body is not None else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _serialized(artifact) -> str:
    return json.dumps(artifact, sort_keys=True, ensure_ascii=False)


def _recompute_artifact_hash(artifact) -> None:
    artifact["artifact_sha256"] = gate.canonical_json_sha256(
        {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    )


def test_success_is_exactly_one_models_get_and_redacted(tmp_path):
    secret = _secret()
    calls = []

    def opener(request, timeout=0):
        calls.append((request.full_url, request.get_method(), timeout))
        return _Response(
            {
                "data": [
                    {"id": "cross-provider-model-id"},
                    {"id": "deepseek-v4-flash"},
                    {"id": "deepseek-other-private-id"},
                ]
            }
        )

    artifact = gate.run_metadata_gate(_env(tmp_path, secret), urlopen=opener, now=FIXED_NOW)
    visible = _serialized(artifact)
    assert calls == [("https://api.deepseek.com/models", "GET", 5.0)]
    assert artifact["status"] == "PASS"
    assert artifact["reason"] is None
    assert artifact["model"] == "deepseek-v4-flash"
    assert artifact["base_url"] == "https://api.deepseek.com"
    assert artifact["exact_model_advertised"] is True
    assert artifact["request_count"] == 1
    assert artifact["deepseek_models_endpoint_requests"] == 1
    assert secret not in visible
    assert "cross-provider-model-id" not in visible
    assert "deepseek-other-private-id" not in visible
    gate.verify_artifact(artifact)


def test_missing_credential_blocks_without_opening_transport(tmp_path):
    calls = []
    artifact = gate.run_metadata_gate(
        _env(tmp_path, ""), urlopen=lambda *args, **kwargs: calls.append(args), now=FIXED_NOW
    )
    assert artifact["status"] == "BLOCKED"
    assert artifact["reason"] == "credential_missing"
    assert artifact["credential_present"] is False
    assert artifact["request_count"] == 0
    assert calls == []


@pytest.mark.parametrize(
    ("kind", "expected", "http_status"),
    [
        ("unauthorized", "unauthorized", 401),
        ("transport", "transport_error", None),
        ("invalid_json", "invalid_json", 200),
        ("echo", "credential_echo_detected", None),
    ],
)
def test_failures_are_sanitized_and_never_retry(tmp_path, kind, expected, http_status):
    secret = _secret()
    calls = []

    def opener(request, timeout=0):
        calls.append(request.full_url)
        if kind == "unauthorized":
            raise urllib.error.HTTPError(request.full_url, 401, secret, {}, None)
        if kind == "transport":
            raise urllib.error.URLError(secret)
        if kind == "invalid_json":
            return _Response(body=b"{invalid")
        return _Response({"error": secret})

    artifact = gate.run_metadata_gate(_env(tmp_path, secret), urlopen=opener, now=FIXED_NOW)
    visible = _serialized(artifact)
    assert artifact["status"] == "BLOCKED"
    assert artifact["reason"] == expected
    assert artifact["http_status"] == http_status
    assert artifact["request_count"] == 1
    assert calls == ["https://api.deepseek.com/models"]
    assert secret not in visible
    gate.verify_artifact(artifact)


def test_http_error_body_echo_is_not_serialized(tmp_path):
    secret = _secret()

    def opener(request, timeout=0):
        body = json.dumps({"echo": secret}).encode("utf-8")
        raise urllib.error.HTTPError(request.full_url, 401, secret, {}, io.BytesIO(body))

    artifact = gate.run_metadata_gate(_env(tmp_path, secret), urlopen=opener, now=FIXED_NOW)
    assert artifact["reason"] == "credential_echo_detected"
    assert secret not in _serialized(artifact)


def test_exact_model_base_and_fixed_zero_request_fields(tmp_path):
    artifact = gate.run_metadata_gate(
        _env(tmp_path, _secret(), model="deepseek-v4-pro"),
        urlopen=lambda *args, **kwargs: pytest.fail("configuration must block before network"),
        now=FIXED_NOW,
    )
    assert artifact["reason"] == "configuration_invalid"
    assert artifact["model"] == "deepseek-v4-flash"
    assert artifact["base_url"] == "https://api.deepseek.com"
    assert artifact["qwen_endpoint_requests"] == 0
    assert artifact["deepseek_other_endpoint_requests"] == 0
    assert artifact["completion_requests"] == 0
    assert artifact["embedding_requests"] == 0
    assert artifact["credential_value_serialized"] is False
    assert artifact["authorization_header_serialized"] is False
    assert artifact["response_body_serialized"] is False


def test_artifact_write_load_and_refuse_overwrite(tmp_path):
    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    output = tmp_path / "gate.json"
    gate.atomic_write_refusing_overwrite(output, artifact)
    assert gate.load_artifact(output) == artifact
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        gate.atomic_write_refusing_overwrite(output, artifact)
    assert output.read_bytes() == original


def test_loader_rejects_canonical_tampering(tmp_path):
    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    artifact["status"] = "PASS"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(gate.GateArtifactError, match="hash mismatch"):
        gate.load_artifact(path)

    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    artifact["model"] = "deepseek-v4-pro"
    _recompute_artifact_hash(artifact)
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(gate.GateArtifactError, match="fixed field mismatch"):
        gate.load_artifact(path)


def test_provenance_and_official_references_are_bound(tmp_path):
    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    assert artifact["observed_utc"] == "2026-08-14T03:04:05Z"
    assert artifact["official_references"] == [
        "https://api-docs.deepseek.com/api/list-models",
        "https://api-docs.deepseek.com/api/create-chat-completion",
        "https://api-docs.deepseek.com/quick_start/pricing",
        "https://api-docs.deepseek.com/updates/",
    ]
    observed = artifact["provenance"]["observed_source_files"]
    assert observed["tool"]["source"] == gate.TOOL_SOURCE
    assert observed["adapter"]["source"] == gate.ADAPTER_SOURCE
    for binding in observed.values():
        assert binding["hash_algorithm"] == "sha256"
        assert binding["identity_claim"] == (
            "observed_file_bytes_only_not_executing_code_identity"
        )
        assert len(binding["observed_file_bytes_sha256"]) == 64
    runtime = artifact["provenance"]["runtime_callable_fingerprint"]
    assert runtime["algorithm"] == "python_code_objects_canonical_sha256_v1"
    assert "adapter.parse_env_file" in runtime["scope"]
    assert "adapter.DeepSeekMetadataClient.fetch_models" in runtime["scope"]
    assert runtime["python_implementation"]
    assert runtime["python_version"]
    assert len(runtime["sha256"]) == 64


@pytest.mark.parametrize(
    ("mapping_path", "forbidden_key"),
    [
        ((), "response_body"),
        (("environment_declaration",), "credential_length"),
        (("provenance",), "authorization_header"),
        (("provenance", "observed_source_files"), "model_ids"),
        (("provenance", "observed_source_files", "tool"), "credential_hash"),
        (("provenance", "observed_source_files", "adapter"), "credential_prefix"),
        (("provenance", "runtime_callable_fingerprint"), "credential_value"),
    ],
)
def test_closed_schema_rejects_rehashed_unknown_fields(
    tmp_path, mapping_path, forbidden_key
):
    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    target = artifact
    for key in mapping_path:
        target = target[key]
    target[forbidden_key] = "forbidden-extension"
    _recompute_artifact_hash(artifact)
    with pytest.raises(gate.GateArtifactError, match="schema mismatch"):
        gate.verify_artifact(artifact)
    output = tmp_path / f"{forbidden_key}.json"
    with pytest.raises(gate.GateArtifactError, match="schema mismatch"):
        gate.atomic_write_refusing_overwrite(output, artifact)
    assert not output.exists()


def test_runtime_callable_monkeypatch_rejected_even_with_rehashed_artifact(
    tmp_path, monkeypatch
):
    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    original = gate.provider.parse_env_file
    monkeypatch.setattr(
        gate.provider,
        "parse_env_file",
        lambda path: original(path),
    )
    _recompute_artifact_hash(artifact)
    with pytest.raises(gate.GateArtifactError, match="provenance mismatch"):
        gate.verify_artifact(artifact)


def test_observed_source_file_hash_change_is_rejected(tmp_path):
    artifact = gate.run_metadata_gate(_env(tmp_path, ""), now=FIXED_NOW)
    artifact["provenance"]["observed_source_files"]["tool"][
        "observed_file_bytes_sha256"
    ] = "0" * 64
    _recompute_artifact_hash(artifact)
    with pytest.raises(gate.GateArtifactError, match="provenance mismatch"):
        gate.verify_artifact(artifact)


def test_cli_has_no_model_or_base_override_and_no_completion_path(tmp_path, capsys):
    secret = _secret()
    output = tmp_path / "result.json"
    with pytest.raises(SystemExit):
        gate.main(
            [
                "--env-file",
                str(_env(tmp_path, secret)),
                "--output",
                str(output),
                "--model",
                "deepseek-v4-pro",
                "--base-url",
                "https://example.invalid",
            ],
            urlopen=lambda *args, **kwargs: pytest.fail("unknown option must not run"),
            now=FIXED_NOW,
        )
    assert not output.exists()
    source = Path(gate.__file__).read_text(encoding="utf-8")
    assert "build_chat_payload" not in source
    assert "subprocess" not in source
    assert "nvidia-smi" not in source
    assert "curl" not in source
    assert secret not in capsys.readouterr().err
