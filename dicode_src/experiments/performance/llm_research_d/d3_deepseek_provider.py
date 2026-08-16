#!/usr/bin/env python3
"""Offline-testable, fail-closed DeepSeek adapter for the D3 experiment.

Importing this module performs no environment, filesystem, or network access.
The caller must explicitly declare the dotenv variable names and pass a
``urlopen`` implementation when offline validation is required.
"""
from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


DEEPSEEK_PROVIDER = "deepseek"
DEEPSEEK_MODEL_ID = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_METADATA_PATH = "/models"
METADATA_REQUEST_LIMIT = 1

DEEPSEEK_LIST_MODELS_REFERENCE = "https://api-docs.deepseek.com/api/list-models"
DEEPSEEK_CHAT_COMPLETION_REFERENCE = (
    "https://api-docs.deepseek.com/api/create-chat-completion"
)
DEEPSEEK_PRICING_REFERENCE = "https://api-docs.deepseek.com/quick_start/pricing"
DEEPSEEK_UPDATES_REFERENCE = "https://api-docs.deepseek.com/updates/"
OFFICIAL_REFERENCES = (
    DEEPSEEK_LIST_MODELS_REFERENCE,
    DEEPSEEK_CHAT_COMPLETION_REFERENCE,
    DEEPSEEK_PRICING_REFERENCE,
    DEEPSEEK_UPDATES_REFERENCE,
)


class DeepSeekProviderError(RuntimeError):
    """Base error whose message contains only a fixed, non-secret reason."""


class DeepSeekConfigError(DeepSeekProviderError, ValueError):
    pass


class CredentialMissingError(DeepSeekProviderError):
    pass


class MetadataGateBlocked(DeepSeekProviderError):
    """The metadata gate failed closed with a safe classification."""

    def __init__(self, reason: str, *, http_status: int | None = None):
        self.reason = str(reason)
        self.http_status = http_status
        super().__init__(self.reason)


class CredentialEchoError(MetadataGateBlocked):
    def __init__(self):
        super().__init__("credential_echo_detected")


class MetadataBudgetExceeded(MetadataGateBlocked):
    def __init__(self):
        super().__init__("metadata_request_budget_exhausted")


def _safe_key(name: str) -> str:
    key = str(name).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise DeepSeekConfigError("invalid environment variable name")
    return key


def _reject_shell_tokens(value: str, *, line_no: int) -> None:
    if "$(" in value or "`" in value:
        raise DeepSeekConfigError(f"shell expansion syntax at line {line_no}")


def _parse_env_value(raw: str, *, line_no: int) -> str:
    value = raw.strip()
    if "\x00" in value or "\n" in value or "\r" in value:
        raise DeepSeekConfigError(f"invalid env value at line {line_no}")
    _reject_shell_tokens(value, line_no=line_no)
    if not value or value[0] not in {"'", '"'}:
        return value
    try:
        decoded = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        raise DeepSeekConfigError(f"invalid quoted env value at line {line_no}") from None
    if not isinstance(decoded, str) or any(char in decoded for char in ("\x00", "\n", "\r")):
        raise DeepSeekConfigError(f"invalid quoted env value at line {line_no}")
    # Quoted escapes can decode to shell metacharacters, so inspect both forms.
    _reject_shell_tokens(decoded, line_no=line_no)
    return decoded


@dataclass(frozen=True, slots=True)
class EnvDeclaration:
    """Explicit mapping of configuration roles to dotenv variable names."""

    provider_var: str
    base_url_var: str
    model_var: str
    credential_var: str

    def __post_init__(self) -> None:
        names = tuple(_safe_key(name) for name in self.variable_names)
        if len(set(names)) != len(names):
            raise DeepSeekConfigError("environment variable roles must be unique")

    @property
    def variable_names(self) -> tuple[str, str, str, str]:
        return (self.provider_var, self.base_url_var, self.model_var, self.credential_var)


class EnvSnapshot:
    """Dotenv data whose values stay in a private, immutable in-memory map."""

    __slots__ = ("path", "variable_names", "presence", "__values")

    def __init__(self, path: str, values: Mapping[str, str]):
        copied = dict(values)
        self.path = str(path)
        self.variable_names = tuple(sorted(copied))
        self.presence = MappingProxyType(
            {name: bool(copied[name]) for name in self.variable_names}
        )
        self.__values = MappingProxyType(copied)

    def _declared_value(self, variable: str) -> str:
        return self.__values.get(_safe_key(variable), "")

    def public_metadata(self) -> dict[str, Any]:
        return {
            "env_path": self.path,
            "variable_names": list(self.variable_names),
            "presence": {name: self.presence[name] for name in self.variable_names},
            "secret_values_serialized": False,
        }

    def __repr__(self) -> str:
        return f"EnvSnapshot(path={self.path!r}, variables={len(self.variable_names)})"

    def __reduce__(self) -> Any:
        raise TypeError("EnvSnapshot serialization is disabled")


def parse_env_file(path: str | Path) -> EnvSnapshot:
    """Parse dotenv assignments as plain data without expansion or evaluation."""
    env_path = Path(path)
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        raise DeepSeekConfigError("unable to read declared env file") from None
    values: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"(?i)^export\s+", line):
            raise DeepSeekConfigError(f"shell export syntax at line {line_no}")
        if "=" not in line:
            raise DeepSeekConfigError(f"invalid env assignment at line {line_no}")
        raw_key, raw_value = line.split("=", 1)
        key = _safe_key(raw_key)
        if key in values:
            raise DeepSeekConfigError(f"duplicate environment variable at line {line_no}")
        values[key] = _parse_env_value(raw_value, line_no=line_no)
    return EnvSnapshot(str(env_path), values)


class DeepSeekProviderConfig:
    """Exact DeepSeek configuration with a private, non-serializable secret."""

    __slots__ = (
        "provider_var",
        "base_url_var",
        "model_var",
        "credential_var",
        "__credential",
    )

    def __init__(self, declaration: EnvDeclaration, credential: str):
        self.provider_var = declaration.provider_var
        self.base_url_var = declaration.base_url_var
        self.model_var = declaration.model_var
        self.credential_var = declaration.credential_var
        self.__credential = credential

    @classmethod
    def from_snapshot(
        cls, snapshot: EnvSnapshot, declaration: EnvDeclaration
    ) -> "DeepSeekProviderConfig":
        if any(name not in snapshot.variable_names for name in declaration.variable_names):
            raise DeepSeekConfigError("declared DeepSeek variable is absent from env file")
        provider = snapshot._declared_value(declaration.provider_var).strip()
        base_url = snapshot._declared_value(declaration.base_url_var).strip()
        model_id = snapshot._declared_value(declaration.model_var).strip()
        credential = snapshot._declared_value(declaration.credential_var)
        if provider != DEEPSEEK_PROVIDER:
            raise DeepSeekConfigError("declared DeepSeek provider is not exact")
        if base_url != DEEPSEEK_BASE_URL:
            raise DeepSeekConfigError("declared DeepSeek base URL is not exact")
        if model_id != DEEPSEEK_MODEL_ID:
            raise DeepSeekConfigError("declared DeepSeek model is not exact")
        if not credential or not credential.strip():
            raise CredentialMissingError("declared DeepSeek credential is missing")
        if credential != credential.strip() or any(ord(char) < 0x20 for char in credential):
            raise DeepSeekConfigError("declared DeepSeek credential is invalid")
        return cls(declaration, credential)

    @property
    def model_id(self) -> str:
        return DEEPSEEK_MODEL_ID

    @property
    def credential_present(self) -> bool:
        return bool(self.__credential)

    def _authorization_header(self) -> str:
        return f"Bearer {self.__credential}"

    def _credential_echoes(self, body: bytes) -> bool:
        direct = self.__credential.encode("utf-8")
        escaped = json.dumps(self.__credential, ensure_ascii=True)[1:-1].encode("utf-8")
        return direct in body or escaped in body

    def _decoded_credential_echoes(self, value: Any) -> bool:
        """Recursively inspect decoded JSON before any value becomes public."""
        if isinstance(value, str):
            return self.__credential in value
        if isinstance(value, Mapping):
            return any(
                self._decoded_credential_echoes(key)
                or self._decoded_credential_echoes(item)
                for key, item in value.items()
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return any(self._decoded_credential_echoes(item) for item in value)
        return False

    def public_metadata(self) -> dict[str, Any]:
        return {
            "provider_variable": self.provider_var,
            "base_url_variable": self.base_url_var,
            "model_variable": self.model_var,
            "credential_variable": self.credential_var,
            "provider": DEEPSEEK_PROVIDER,
            "base_url": DEEPSEEK_BASE_URL,
            "model_id": DEEPSEEK_MODEL_ID,
            "credential_present": self.credential_present,
            "credential_value_serialized": False,
            "official_references": list(OFFICIAL_REFERENCES),
        }

    def __repr__(self) -> str:
        return (
            "DeepSeekProviderConfig("
            f"model_id={DEEPSEEK_MODEL_ID!r}, credential_present={self.credential_present})"
        )

    def __reduce__(self) -> Any:
        raise TypeError("DeepSeekProviderConfig serialization is disabled")


@dataclass(frozen=True, slots=True)
class MetadataResult:
    http_status: int
    model_ids: tuple[str, ...]
    expected_model: str
    request_count: int
    gate_status: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "http_status": self.http_status,
            "model_ids": list(self.model_ids),
            "expected_model": self.expected_model,
            "request_count": self.request_count,
            "gate_status": self.gate_status,
        }


Urlopen = Callable[..., Any]


class DeepSeekMetadataClient:
    """One-GET ``/models`` gate; there is deliberately no retry path."""

    __slots__ = ("config", "_urlopen", "_requests", "_gate_passed")

    def __init__(
        self, config: DeepSeekProviderConfig, *, urlopen: Urlopen | None = None
    ) -> None:
        self.config = config
        self._urlopen = urlopen or urllib.request.urlopen
        self._requests = 0
        self._gate_passed = False

    @property
    def requests_used(self) -> int:
        return self._requests

    @property
    def gate_passed(self) -> bool:
        return self._gate_passed

    def fetch_models(self, *, timeout_s: float = 5.0) -> MetadataResult:
        if not self.config.credential_present:
            raise CredentialMissingError("declared DeepSeek credential is missing")
        if self._requests >= METADATA_REQUEST_LIMIT:
            raise MetadataBudgetExceeded()
        self._requests += 1
        request = urllib.request.Request(
            DEEPSEEK_BASE_URL + DEEPSEEK_METADATA_PATH,
            headers={
                "Accept": "application/json",
                "Authorization": self.config._authorization_header(),
            },
            method="GET",
        )
        body: bytes = b""
        status = 0
        failure_reason: str | None = None
        failure_status: int | None = None
        try:
            with self._urlopen(request, timeout=float(timeout_s)) as response:
                status = int(getattr(response, "status", 200))
                body = response.read()
        except urllib.error.HTTPError as exc:
            try:
                status = int(exc.code)
                body = exc.read()
            except Exception:
                body = b""
            failure_reason = "unauthorized" if status == 401 else "http_error"
            failure_status = status
        except Exception:
            # Do not retain or chain a possibly secret-bearing exception.
            failure_reason = "transport_error"
        # Raising only after the handlers have exited guarantees that source
        # exceptions are neither causes nor contexts of sanitized gate errors.
        if not isinstance(body, bytes):
            body = b""
        self._check_echo(body)
        if failure_reason is not None:
            if self._decoded_body_echoes(body):
                raise CredentialEchoError()
            raise MetadataGateBlocked(failure_reason, http_status=failure_status)
        if status == 401:
            raise MetadataGateBlocked("unauthorized", http_status=status)
        if not 200 <= status < 300:
            raise MetadataGateBlocked("http_error", http_status=status)
        invalid_json = False
        try:
            decoded = json.loads(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
            invalid_json = True
        if invalid_json:
            raise MetadataGateBlocked("invalid_json", http_status=status)
        if self.config._decoded_credential_echoes(decoded):
            raise CredentialEchoError()
        if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
            raise MetadataGateBlocked("invalid_model_list", http_status=status)
        model_ids = tuple(
            item["id"]
            for item in decoded["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        )
        if DEEPSEEK_MODEL_ID not in model_ids:
            raise MetadataGateBlocked("model_missing", http_status=status)
        self._gate_passed = True
        return MetadataResult(status, model_ids, DEEPSEEK_MODEL_ID, self._requests, "PASS")

    def _check_echo(self, body: bytes) -> None:
        if self.config._credential_echoes(body):
            raise CredentialEchoError()

    def _decoded_body_echoes(self, body: bytes) -> bool:
        """Inspect a JSON HTTP-error body without retaining parse failures."""
        try:
            decoded = json.loads(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return self.config._decoded_credential_echoes(decoded)

    def build_chat_payload(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        """Build the exact thinking payload, but only after the metadata gate."""
        if not self._gate_passed:
            raise MetadataGateBlocked("completion_blocked_until_metadata_pass")
        if not messages or isinstance(messages, (str, bytes)):
            raise DeepSeekConfigError("chat messages must be non-empty")
        copied_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise DeepSeekConfigError("chat messages must be mappings")
            copied_messages.append(dict(message))
        return {
            "model": DEEPSEEK_MODEL_ID,
            "messages": copied_messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 8192,
            "thinking": {"type": "enabled"},
        }


def public_json(value: EnvSnapshot | DeepSeekProviderConfig | MetadataResult) -> str:
    """Serialize only an adapter object's explicitly public metadata."""
    if not isinstance(value, (EnvSnapshot, DeepSeekProviderConfig, MetadataResult)):
        raise TypeError("only explicitly public adapter metadata can be serialized")
    return json.dumps(
        value.public_metadata(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
