#!/usr/bin/env python3
"""Audited one-request DeepSeek metadata gate for remote execution.

This module performs no environment or remote access at import time.  It
delegates dotenv parsing, credential handling, echo detection, and the single
``GET /models`` request to :mod:`d3_deepseek_provider`.  It has no completion,
embedding, shell, GPU, or cross-provider request path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import CodeType
from typing import Any, Callable, Mapping, Sequence

import d3_deepseek_provider as provider


CLASSIFICATION = "D3_DEEPSEEK_METADATA_GATE"
SCHEMA_VERSION = 1
CANONICAL_ALGORITHM = "canonical_json_sha256"
CANONICAL_SCOPE = "ALL_FIELDS_EXCLUDING_ARTIFACT_SHA256"
TOOL_SOURCE = "d3_deepseek_metadata_gate_remote.py"
ADAPTER_SOURCE = "d3_deepseek_provider.py"
FILE_HASH_ALGORITHM = "sha256"
FILE_HASH_CLAIM = "observed_file_bytes_only_not_executing_code_identity"
RUNTIME_FINGERPRINT_ALGORITHM = "python_code_objects_canonical_sha256_v1"
_RUNTIME_CALLABLE_BASELINE: dict[str, Any] | None = None

PROVIDER_VARIABLE = "EXP_DEEPSEEK_PROVIDER"
BASE_URL_VARIABLE = "EXP_DEEPSEEK_BASE_URL"
MODEL_VARIABLE = "EXP_DEEPSEEK_MODEL"
CREDENTIAL_VARIABLE = "EXP_DEEPSEEK_API_KEY"
ENV_DECLARATION = provider.EnvDeclaration(
    PROVIDER_VARIABLE,
    BASE_URL_VARIABLE,
    MODEL_VARIABLE,
    CREDENTIAL_VARIABLE,
)

TOP_LEVEL_KEYS = frozenset(
    {
        "artifact_sha256",
        "artifact_sha256_algorithm",
        "artifact_sha256_scope",
        "authorization_header_serialized",
        "base_url",
        "classification",
        "completion_requests",
        "credential_present",
        "credential_value_serialized",
        "deepseek_models_endpoint_requests",
        "deepseek_other_endpoint_requests",
        "embedding_requests",
        "environment_declaration",
        "exact_model_advertised",
        "http_status",
        "metadata_method",
        "metadata_path",
        "model",
        "observed_utc",
        "official_references",
        "provenance",
        "provider",
        "qwen_endpoint_requests",
        "reason",
        "request_count",
        "response_body_serialized",
        "schema_version",
        "status",
    }
)
ENVIRONMENT_DECLARATION_KEYS = frozenset(
    {"provider_variable", "base_url_variable", "model_variable", "credential_variable"}
)
PROVENANCE_KEYS = frozenset({"observed_source_files", "runtime_callable_fingerprint"})
OBSERVED_SOURCE_FILES_KEYS = frozenset({"tool", "adapter"})
OBSERVED_FILE_BINDING_KEYS = frozenset(
    {"source", "hash_algorithm", "observed_file_bytes_sha256", "identity_claim"}
)
RUNTIME_FINGERPRINT_KEYS = frozenset(
    {"algorithm", "scope", "python_implementation", "python_version", "sha256"}
)


class GateArtifactError(RuntimeError):
    """A sanitized artifact creation or verification failure."""


def canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonical(item) for item in value]
    raise GateArtifactError("artifact contains a non-canonical value")


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        canonical(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        raise GateArtifactError("unable to hash gate provenance") from None


def _constant_descriptor(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_constant_descriptor(item) for item in value]}
    if isinstance(value, frozenset):
        members = [_constant_descriptor(item) for item in value]
        members.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"frozenset": members}
    if isinstance(value, CodeType):
        return {"code": _code_descriptor(value)}
    if value is Ellipsis:
        return {"singleton": "Ellipsis"}
    return {"constant_type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _code_descriptor(code: CodeType) -> dict[str, Any]:
    """Normalize executable code without claiming whole-module identity."""
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "exceptiontable_hex": getattr(code, "co_exceptiontable", b"").hex(),
        "constants": [_constant_descriptor(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _selected_runtime_callables() -> list[tuple[str, Callable[..., Any]]]:
    """Resolve callables dynamically so monkeypatching changes the digest."""
    return [
        ("gate._artifact_payload", _artifact_payload),
        ("gate._base_artifact", _base_artifact),
        ("gate._code_descriptor", _code_descriptor),
        ("gate._constant_descriptor", _constant_descriptor),
        ("gate._current_provenance", _current_provenance),
        ("gate._file_sha256", _file_sha256),
        ("gate._runtime_callable_fingerprint", _runtime_callable_fingerprint),
        ("gate._seal", _seal),
        ("gate._selected_runtime_callables", _selected_runtime_callables),
        ("gate._utc_text", _utc_text),
        ("gate.canonical", canonical),
        ("gate.canonical_json_sha256", canonical_json_sha256),
        ("gate.run_metadata_gate", run_metadata_gate),
        ("gate.verify_artifact", verify_artifact),
        ("adapter._parse_env_value", provider._parse_env_value),
        ("adapter._reject_shell_tokens", provider._reject_shell_tokens),
        ("adapter.DeepSeekMetadataClient.__init__", provider.DeepSeekMetadataClient.__init__),
        ("adapter.DeepSeekMetadataClient._check_echo", provider.DeepSeekMetadataClient._check_echo),
        (
            "adapter.DeepSeekMetadataClient._decoded_body_echoes",
            provider.DeepSeekMetadataClient._decoded_body_echoes,
        ),
        (
            "adapter.DeepSeekMetadataClient.fetch_models",
            provider.DeepSeekMetadataClient.fetch_models,
        ),
        (
            "adapter.DeepSeekProviderConfig._credential_echoes",
            provider.DeepSeekProviderConfig._credential_echoes,
        ),
        (
            "adapter.DeepSeekProviderConfig._decoded_credential_echoes",
            provider.DeepSeekProviderConfig._decoded_credential_echoes,
        ),
        (
            "adapter.DeepSeekProviderConfig.from_snapshot",
            provider.DeepSeekProviderConfig.from_snapshot,
        ),
        ("adapter.EnvSnapshot.__init__", provider.EnvSnapshot.__init__),
        ("adapter.parse_env_file", provider.parse_env_file),
    ]


def _runtime_callable_fingerprint() -> dict[str, Any]:
    implementation = platform.python_implementation()
    version = platform.python_version()
    descriptors: dict[str, Any] = {}
    scope: list[str] = []
    for label, callable_value in _selected_runtime_callables():
        target = getattr(callable_value, "__func__", callable_value)
        code = getattr(target, "__code__", None)
        if not isinstance(code, CodeType):
            raise GateArtifactError("runtime callable fingerprint unavailable")
        scope.append(label)
        descriptors[label] = _code_descriptor(code)
    digest_payload = {
        "python_implementation": implementation,
        "python_version": version,
        "callables": descriptors,
    }
    return {
        "algorithm": RUNTIME_FINGERPRINT_ALGORITHM,
        "scope": scope,
        "python_implementation": implementation,
        "python_version": version,
        "sha256": canonical_json_sha256(digest_payload),
    }


def _observed_file_binding(source: str, path: Path) -> dict[str, str]:
    return {
        "source": source,
        "hash_algorithm": FILE_HASH_ALGORITHM,
        "observed_file_bytes_sha256": _file_sha256(path),
        "identity_claim": FILE_HASH_CLAIM,
    }


def _current_provenance() -> dict[str, Any]:
    adapter_path = Path(provider.__file__).resolve()
    tool_path = Path(__file__).resolve()
    runtime_fingerprint = _runtime_callable_fingerprint()
    if (
        _RUNTIME_CALLABLE_BASELINE is not None
        and runtime_fingerprint != _RUNTIME_CALLABLE_BASELINE
    ):
        raise GateArtifactError("gate artifact provenance mismatch: runtime callable changed")
    return {
        "observed_source_files": {
            "tool": _observed_file_binding(TOOL_SOURCE, tool_path),
            "adapter": _observed_file_binding(ADAPTER_SOURCE, adapter_path),
        },
        "runtime_callable_fingerprint": runtime_fingerprint,
    }


def _utc_text(now: datetime | None) -> str:
    observed = now if now is not None else datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise GateArtifactError("observed UTC must be timezone-aware")
    return observed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _base_artifact(*, observed_utc: str, credential_present: bool) -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "reason": "not_run",
        "provider": provider.DEEPSEEK_PROVIDER,
        "model": provider.DEEPSEEK_MODEL_ID,
        "base_url": provider.DEEPSEEK_BASE_URL,
        "metadata_method": "GET",
        "metadata_path": provider.DEEPSEEK_METADATA_PATH,
        "official_references": list(provider.OFFICIAL_REFERENCES),
        "environment_declaration": {
            "provider_variable": PROVIDER_VARIABLE,
            "base_url_variable": BASE_URL_VARIABLE,
            "model_variable": MODEL_VARIABLE,
            "credential_variable": CREDENTIAL_VARIABLE,
        },
        "credential_present": bool(credential_present),
        "credential_value_serialized": False,
        "authorization_header_serialized": False,
        "response_body_serialized": False,
        "request_count": 0,
        "http_status": None,
        "exact_model_advertised": False,
        "deepseek_models_endpoint_requests": 0,
        "deepseek_other_endpoint_requests": 0,
        "qwen_endpoint_requests": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
        "observed_utc": observed_utc,
        "provenance": _current_provenance(),
        "artifact_sha256_algorithm": CANONICAL_ALGORITHM,
        "artifact_sha256_scope": CANONICAL_SCOPE,
    }


def _seal(artifact: dict[str, Any]) -> dict[str, Any]:
    sealed = canonical(artifact)
    sealed["artifact_sha256"] = canonical_json_sha256(sealed)
    verify_artifact(sealed)
    return sealed


def run_metadata_gate(
    env_file: str | Path,
    *,
    urlopen: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run at most one DeepSeek ``GET /models`` and return redacted evidence."""
    credential_present = False
    snapshot: provider.EnvSnapshot | None = None
    try:
        snapshot = provider.parse_env_file(env_file)
        credential_present = bool(snapshot.presence.get(CREDENTIAL_VARIABLE, False))
        config = provider.DeepSeekProviderConfig.from_snapshot(snapshot, ENV_DECLARATION)
    except provider.CredentialMissingError:
        reason = "credential_missing"
        config = None
    except provider.DeepSeekProviderError:
        reason = "configuration_invalid"
        config = None
    artifact = _base_artifact(
        observed_utc=_utc_text(now), credential_present=credential_present
    )
    if config is None:
        artifact["reason"] = reason
        return _seal(artifact)

    client = provider.DeepSeekMetadataClient(config, urlopen=urlopen)
    try:
        result = client.fetch_models()
    except provider.MetadataGateBlocked as exc:
        artifact["reason"] = exc.reason
        artifact["http_status"] = exc.http_status
    except provider.CredentialMissingError:
        artifact["reason"] = "credential_missing"
    else:
        artifact["status"] = "PASS"
        artifact["reason"] = None
        artifact["http_status"] = result.http_status
        artifact["exact_model_advertised"] = True
    artifact["request_count"] = client.requests_used
    artifact["deepseek_models_endpoint_requests"] = client.requests_used
    return _seal(artifact)


def _artifact_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "artifact_sha256"}


def _require_exact_keys(value: Any, allowed: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != allowed:
        raise GateArtifactError(f"gate artifact {label} schema mismatch")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GateArtifactError(f"gate artifact {label} SHA256 invalid")
    return value


def verify_artifact(artifact: Any) -> dict[str, Any]:
    """Validate a recursively closed schema, integrity, and safety invariants."""
    artifact = _require_exact_keys(artifact, TOP_LEVEL_KEYS, "top-level")
    environment = _require_exact_keys(
        artifact["environment_declaration"],
        ENVIRONMENT_DECLARATION_KEYS,
        "environment declaration",
    )
    provenance = _require_exact_keys(artifact["provenance"], PROVENANCE_KEYS, "provenance")
    observed_files = _require_exact_keys(
        provenance["observed_source_files"],
        OBSERVED_SOURCE_FILES_KEYS,
        "observed source files",
    )
    tool_binding = _require_exact_keys(
        observed_files["tool"], OBSERVED_FILE_BINDING_KEYS, "tool source binding"
    )
    adapter_binding = _require_exact_keys(
        observed_files["adapter"], OBSERVED_FILE_BINDING_KEYS, "adapter source binding"
    )
    runtime_fingerprint = _require_exact_keys(
        provenance["runtime_callable_fingerprint"],
        RUNTIME_FINGERPRINT_KEYS,
        "runtime callable fingerprint",
    )
    if artifact.get("artifact_sha256_algorithm") != CANONICAL_ALGORITHM:
        raise GateArtifactError("gate artifact hash algorithm mismatch")
    if artifact.get("artifact_sha256_scope") != CANONICAL_SCOPE:
        raise GateArtifactError("gate artifact hash scope mismatch")
    artifact_sha256 = _require_sha256(artifact["artifact_sha256"], "canonical")
    if canonical_json_sha256(_artifact_payload(artifact)) != artifact_sha256:
        raise GateArtifactError("gate artifact hash mismatch")
    expected_fixed = {
        "classification": CLASSIFICATION,
        "schema_version": SCHEMA_VERSION,
        "provider": provider.DEEPSEEK_PROVIDER,
        "model": provider.DEEPSEEK_MODEL_ID,
        "base_url": provider.DEEPSEEK_BASE_URL,
        "metadata_method": "GET",
        "metadata_path": provider.DEEPSEEK_METADATA_PATH,
        "official_references": list(provider.OFFICIAL_REFERENCES),
        "credential_value_serialized": False,
        "authorization_header_serialized": False,
        "response_body_serialized": False,
        "deepseek_other_endpoint_requests": 0,
        "qwen_endpoint_requests": 0,
        "completion_requests": 0,
        "embedding_requests": 0,
    }
    for key, expected in expected_fixed.items():
        if artifact.get(key) != expected:
            raise GateArtifactError("gate artifact fixed field mismatch")
    if type(artifact["schema_version"]) is not int:
        raise GateArtifactError("gate artifact schema version type invalid")
    for redaction_flag in (
        "credential_value_serialized",
        "authorization_header_serialized",
        "response_body_serialized",
    ):
        if type(artifact[redaction_flag]) is not bool or artifact[redaction_flag] is not False:
            raise GateArtifactError("gate artifact redaction flag invalid")
    if environment != {
        "provider_variable": PROVIDER_VARIABLE,
        "base_url_variable": BASE_URL_VARIABLE,
        "model_variable": MODEL_VARIABLE,
        "credential_variable": CREDENTIAL_VARIABLE,
    }:
        raise GateArtifactError("gate artifact env declaration mismatch")
    count = artifact.get("request_count")
    if (
        type(count) is not int
        or count not in {0, 1}
        or type(artifact.get("deepseek_models_endpoint_requests")) is not int
        or artifact.get("deepseek_models_endpoint_requests") != count
    ):
        raise GateArtifactError("gate artifact request count mismatch")
    for zero_count_field in (
        "deepseek_other_endpoint_requests",
        "qwen_endpoint_requests",
        "completion_requests",
        "embedding_requests",
    ):
        if type(artifact[zero_count_field]) is not int or artifact[zero_count_field] != 0:
            raise GateArtifactError("gate artifact zero request field invalid")
    status = artifact.get("status")
    if status not in {"PASS", "BLOCKED"}:
        raise GateArtifactError("gate artifact status invalid")
    blocked_reasons = {
        "configuration_invalid",
        "credential_echo_detected",
        "credential_missing",
        "http_error",
        "invalid_json",
        "invalid_model_list",
        "metadata_request_budget_exhausted",
        "model_missing",
        "transport_error",
        "unauthorized",
    }
    if status == "PASS":
        if (
            artifact.get("reason") is not None
            or artifact.get("exact_model_advertised") is not True
            or count != 1
            or artifact.get("credential_present") is not True
            or not isinstance(artifact.get("http_status"), int)
            or not 200 <= artifact["http_status"] < 300
        ):
            raise GateArtifactError("gate artifact pass fields invalid")
    elif (
        type(artifact.get("reason")) is not str
        or artifact.get("reason") not in blocked_reasons
        or artifact.get("exact_model_advertised") is not False
    ):
        raise GateArtifactError("gate artifact blocked fields invalid")
    if type(artifact.get("credential_present")) is not bool:
        raise GateArtifactError("gate artifact credential presence invalid")
    if artifact["http_status"] is not None and type(artifact["http_status"]) is not int:
        raise GateArtifactError("gate artifact HTTP status invalid")
    observed_utc = artifact.get("observed_utc")
    if not isinstance(observed_utc, str) or not observed_utc.endswith("Z"):
        raise GateArtifactError("gate artifact UTC invalid")
    try:
        parsed_utc = datetime.fromisoformat(observed_utc.removesuffix("Z") + "+00:00")
    except ValueError:
        raise GateArtifactError("gate artifact UTC invalid") from None
    if parsed_utc.utcoffset() != timezone.utc.utcoffset(parsed_utc):
        raise GateArtifactError("gate artifact UTC invalid")
    for binding, expected_source in (
        (tool_binding, TOOL_SOURCE),
        (adapter_binding, ADAPTER_SOURCE),
    ):
        if (
            binding["source"] != expected_source
            or binding["hash_algorithm"] != FILE_HASH_ALGORITHM
            or binding["identity_claim"] != FILE_HASH_CLAIM
        ):
            raise GateArtifactError("gate artifact observed file binding invalid")
        _require_sha256(binding["observed_file_bytes_sha256"], "observed file bytes")
    if (
        runtime_fingerprint["algorithm"] != RUNTIME_FINGERPRINT_ALGORITHM
        or not isinstance(runtime_fingerprint["scope"], list)
        or not all(isinstance(item, str) for item in runtime_fingerprint["scope"])
        or not isinstance(runtime_fingerprint["python_implementation"], str)
        or not isinstance(runtime_fingerprint["python_version"], str)
    ):
        raise GateArtifactError("gate artifact runtime fingerprint invalid")
    _require_sha256(runtime_fingerprint["sha256"], "runtime callable fingerprint")
    if provenance != _current_provenance():
        raise GateArtifactError("gate artifact provenance mismatch")
    return artifact


# Capture the actual loaded code objects once all selected callables exist.
# This is pure runtime introspection: it reads no env file, key, network, or GPU.
_RUNTIME_CALLABLE_BASELINE = _runtime_callable_fingerprint()


def load_artifact(path: str | Path) -> dict[str, Any]:
    try:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise GateArtifactError("unable to load gate artifact") from None
    return verify_artifact(parsed)


def atomic_write_refusing_overwrite(path: str | Path, artifact: Mapping[str, Any]) -> None:
    target = Path(path)
    verify_artifact(dict(artifact))
    text = json.dumps(canonical(artifact), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    for _ in range(100):
        candidate = target.parent / f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}"
        try:
            fd = os.open(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        temporary = candidate
        break
    else:
        raise GateArtifactError("unable to allocate private artifact temporary")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError("refusing to overwrite existing gate artifact") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the audited DeepSeek metadata-only gate")
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    urlopen: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> int:
    args = _parser().parse_args(argv)
    artifact = run_metadata_gate(args.env_file, urlopen=urlopen, now=now)
    atomic_write_refusing_overwrite(args.output, artifact)
    sys.stdout.write(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if artifact["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
