#!/usr/bin/env python3
"""Audited one-request DeepSeek metadata gate for remote execution.

This module performs no work at import time.  It delegates dotenv parsing,
credential handling, echo detection, and the single ``GET /models`` request
to :mod:`d3_deepseek_provider`.  It has no completion, embedding, shell, GPU,
or cross-provider request path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import d3_deepseek_provider as provider


CLASSIFICATION = "D3_DEEPSEEK_METADATA_GATE"
SCHEMA_VERSION = 1
CANONICAL_ALGORITHM = "canonical_json_sha256"
CANONICAL_SCOPE = "ALL_FIELDS_EXCLUDING_ARTIFACT_SHA256"
TOOL_SOURCE = "d3_deepseek_metadata_gate_remote.py"
ADAPTER_SOURCE = "d3_deepseek_provider.py"

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


def _current_provenance() -> dict[str, dict[str, str]]:
    adapter_path = Path(provider.__file__).resolve()
    tool_path = Path(__file__).resolve()
    return {
        "tool": {"source": TOOL_SOURCE, "sha256": _file_sha256(tool_path)},
        "adapter": {"source": ADAPTER_SOURCE, "sha256": _file_sha256(adapter_path)},
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


def verify_artifact(artifact: Any) -> dict[str, Any]:
    """Validate canonical integrity and all fixed safety invariants."""
    if not isinstance(artifact, dict):
        raise GateArtifactError("gate artifact must be an object")
    if artifact.get("artifact_sha256_algorithm") != CANONICAL_ALGORITHM:
        raise GateArtifactError("gate artifact hash algorithm mismatch")
    if artifact.get("artifact_sha256_scope") != CANONICAL_SCOPE:
        raise GateArtifactError("gate artifact hash scope mismatch")
    if canonical_json_sha256(_artifact_payload(artifact)) != artifact.get("artifact_sha256"):
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
    if artifact.get("environment_declaration") != {
        "provider_variable": PROVIDER_VARIABLE,
        "base_url_variable": BASE_URL_VARIABLE,
        "model_variable": MODEL_VARIABLE,
        "credential_variable": CREDENTIAL_VARIABLE,
    }:
        raise GateArtifactError("gate artifact env declaration mismatch")
    count = artifact.get("request_count")
    if count not in {0, 1} or artifact.get("deepseek_models_endpoint_requests") != count:
        raise GateArtifactError("gate artifact request count mismatch")
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
        artifact.get("reason") not in blocked_reasons
        or artifact.get("exact_model_advertised") is not False
    ):
        raise GateArtifactError("gate artifact blocked fields invalid")
    if not isinstance(artifact.get("credential_present"), bool):
        raise GateArtifactError("gate artifact credential presence invalid")
    observed_utc = artifact.get("observed_utc")
    if not isinstance(observed_utc, str) or not observed_utc.endswith("Z"):
        raise GateArtifactError("gate artifact UTC invalid")
    try:
        parsed_utc = datetime.fromisoformat(observed_utc.removesuffix("Z") + "+00:00")
    except ValueError:
        raise GateArtifactError("gate artifact UTC invalid") from None
    if parsed_utc.utcoffset() != timezone.utc.utcoffset(parsed_utc):
        raise GateArtifactError("gate artifact UTC invalid")
    if artifact.get("provenance") != _current_provenance():
        raise GateArtifactError("gate artifact provenance mismatch")
    return artifact


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
