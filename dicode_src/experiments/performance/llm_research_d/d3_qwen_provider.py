#!/usr/bin/env python3
"""Security-bounded Qwen provider adapter for the D3 comparison.

This module deliberately contains no shell evaluation and no provider call at
import time.  The caller supplies the variable names declared by the existing
launcher; this adapter never guesses a credential, base URL, or model name.
Environment values are retained only in a private in-memory mapping.  Public
metadata contains variable names, presence bits, and the model id, but never a
credential, URL, or authorization header.

The metadata client has a hard one-request budget and is suitable for a later
server gate.  It is exercised by offline fake-openers in ``test_d3_qwen_provider``;
this file itself does not perform a network request.
"""
from __future__ import annotations

import ast
import hashlib
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


QWEN_PROVIDER_KIND = "qwen_api"
QWEN_METADATA_PATH = "/models"
QWEN_DEFAULT_THINKING = True
METADATA_REQUEST_LIMIT = 1


class QwenProviderError(RuntimeError):
    """Base error whose text is guaranteed not to contain secret values."""


class QwenConfigError(QwenProviderError, ValueError):
    pass


class CredentialMissingError(QwenProviderError):
    pass


class MetadataGateBlocked(QwenProviderError):
    """A metadata gate failed closed; ``reason`` is a safe classification."""

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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_key(name: str) -> str:
    key = str(name).strip()
    if not key or not key.replace("_", "a").isalnum() or not (key[0].isalpha() or key[0] == "_"):
        raise QwenConfigError("invalid environment variable name")
    return key


def _parse_env_value(raw: str, *, line_no: int) -> str:
    """Parse a dotenv value without expansion or shell evaluation.

    Only plain values and fully quoted Python-style strings are accepted.  A
    value containing ``$()``, backticks, or a newline is retained literally in
    neither case; these constructs are rejected to prevent accidental shell
    semantics from entering the experiment configuration.
    """
    value = raw.strip()
    if not value:
        return ""
    if "\x00" in value or "\n" in value or "\r" in value:
        raise QwenConfigError(f"invalid env value at line {line_no}")
    _reject_shell_tokens(value, line_no=line_no)
    if value[0] in ("'", '"'):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise QwenConfigError(f"invalid quoted env value at line {line_no}") from exc
        if not isinstance(parsed, str) or "\n" in parsed or "\r" in parsed:
            raise QwenConfigError(f"invalid quoted env value at line {line_no}")
        # literal_eval decodes escapes (for example ``"\\x60"`` and
        # ``"\\x24("``), so repeat the shell-token check on the decoded
        # value before retaining it in memory.
        _reject_shell_tokens(parsed, line_no=line_no)
        return parsed
    # Inline comments are not stripped: silently changing an API URL or key is
    # less safe than requiring the launcher to quote a value that contains '#'.
    return value


def _reject_shell_tokens(value: str, *, line_no: int) -> None:
    """Reject shell expansion markers before and after quoted decoding."""
    if "$" + "(" in value or "`" in value:
        raise QwenConfigError(f"shell expansion syntax at line {line_no}")


@dataclass(frozen=True)
class EnvDeclaration:
    """Explicit launcher declaration of the four Qwen configuration roles."""

    provider_var: str
    base_url_var: str
    model_var: str
    credential_var: str

    def __post_init__(self) -> None:
        for name in (self.provider_var, self.base_url_var, self.model_var, self.credential_var):
            _safe_key(name)

    @property
    def variable_names(self) -> tuple[str, ...]:
        return (self.provider_var, self.base_url_var, self.model_var, self.credential_var)


@dataclass(frozen=True)
class EnvSnapshot:
    """Parsed env file with private values and safe public metadata."""

    path: str
    variable_names: tuple[str, ...]
    presence: Mapping[str, bool]
    _values: Mapping[str, str] = field(repr=False, compare=False)

    def value(self, variable: str) -> str:
        _safe_key(variable)
        return str(self._values.get(variable, ""))

    def public_metadata(self) -> dict[str, Any]:
        return {
            "env_path": self.path,
            "variable_names": list(self.variable_names),
            "presence": {k: bool(self.presence[k]) for k in self.variable_names},
            "secret_values_serialized": False,
        }

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction
        return f"EnvSnapshot(path={self.path!r}, variables={len(self.variable_names)})"


def parse_env_file(path: str | Path) -> EnvSnapshot:
    """Read a dotenv-like file as data; never invoke a shell or expand values."""
    env_path = Path(path)
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QwenConfigError("unable to read declared env file") from exc
    values: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            raise QwenConfigError(f"shell export syntax at line {line_no}")
        if "=" not in line:
            raise QwenConfigError(f"invalid env assignment at line {line_no}")
        key_raw, value_raw = line.split("=", 1)
        key = _safe_key(key_raw)
        if key in values:
            raise QwenConfigError(f"duplicate environment variable at line {line_no}")
        values[key] = _parse_env_value(value_raw, line_no=line_no)
    names = tuple(sorted(values))
    return EnvSnapshot(
        path=str(env_path),
        variable_names=names,
        presence={key: bool(values[key]) for key in names},
        _values=values,
    )


def _safe_base_url(value: str) -> str:
    raw = str(value).strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise QwenConfigError("declared Qwen base URL is invalid")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise QwenConfigError("declared Qwen base URL contains credentials or query data")
    if any(ord(ch) < 0x20 for ch in raw):
        raise QwenConfigError("declared Qwen base URL contains control data")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


@dataclass(frozen=True)
class QwenProviderConfig:
    """Private Qwen configuration; repr and public evidence are redacted."""

    provider_var: str
    base_url_var: str
    model_var: str
    credential_var: str
    provider: str = field(repr=False)
    _base_url: str = field(repr=False)
    model_id: str
    _credential: str = field(repr=False)
    thinking: bool = QWEN_DEFAULT_THINKING

    @classmethod
    def from_snapshot(cls, snapshot: EnvSnapshot, declaration: EnvDeclaration) -> "QwenProviderConfig":
        missing = [name for name in declaration.variable_names if name not in snapshot.variable_names]
        if missing:
            raise QwenConfigError("declared Qwen variable is absent from env file")
        provider = snapshot.value(declaration.provider_var).strip()
        base_url = _safe_base_url(snapshot.value(declaration.base_url_var))
        model = snapshot.value(declaration.model_var).strip()
        credential = snapshot.value(declaration.credential_var)
        if not provider or not model:
            raise QwenConfigError("declared Qwen provider/model is empty")
        if not credential:
            raise CredentialMissingError("declared Qwen credential is missing")
        if any(ord(ch) < 0x20 for ch in credential):
            raise QwenConfigError("declared Qwen credential contains control data")
        return cls(
            provider_var=declaration.provider_var,
            base_url_var=declaration.base_url_var,
            model_var=declaration.model_var,
            credential_var=declaration.credential_var,
            provider=provider,
            _base_url=base_url,
            model_id=model,
            _credential=credential,
        )

    @property
    def base_url(self) -> str:
        """Internal use only; callers must not serialize this value."""
        return self._base_url

    @property
    def credential_present(self) -> bool:
        return bool(self._credential)

    def metadata_url(self) -> str:
        return self._base_url + QWEN_METADATA_PATH

    def public_metadata(self) -> dict[str, Any]:
        return {
            "provider_variable": self.provider_var,
            "provider_present": bool(self.provider),
            "base_url_variable": self.base_url_var,
            "base_url_present": bool(self._base_url),
            "model_variable": self.model_var,
            "model_present": bool(self.model_id),
            "credential_variable": self.credential_var,
            "credential_present": self.credential_present,
            "model_id": self.model_id,
            "thinking_default": bool(self.thinking),
            "credential_value_serialized": False,
            "base_url_serialized": False,
        }

    def _authorization_header(self) -> str:
        return f"Bearer {self._credential}"

    def __repr__(self) -> str:  # pragma: no cover - defensive redaction
        return f"QwenProviderConfig(model_id={self.model_id!r}, credential_present={self.credential_present})"


@dataclass(frozen=True)
class MetadataResult:
    http_status: int
    model_ids: tuple[str, ...]
    expected_model: str
    request_count: int
    gate_status: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "http_status": int(self.http_status),
            "model_ids": list(self.model_ids),
            "expected_model": self.expected_model,
            "request_count": int(self.request_count),
            "gate_status": self.gate_status,
        }


Urlopen = Callable[..., Any]


class QwenMetadataClient:
    """One-shot Qwen ``/models`` metadata gate.

    ``urlopen`` is injectable for offline tests.  There is deliberately no
    retry path: a 401, missing credential, model mismatch, or transport error
    blocks the arm and no completion endpoint is reachable through this class.
    """

    def __init__(self, config: QwenProviderConfig, *, urlopen: Urlopen | None = None,
                 request_limit: int = METADATA_REQUEST_LIMIT):
        if int(request_limit) != METADATA_REQUEST_LIMIT:
            raise QwenConfigError("D3 Qwen metadata request limit must be exactly one")
        self.config = config
        self._urlopen = urlopen or urllib.request.urlopen
        self._request_limit = int(request_limit)
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
            raise CredentialMissingError("declared Qwen credential is missing")
        if self._requests >= self._request_limit:
            raise MetadataBudgetExceeded()
        self._requests += 1
        request = urllib.request.Request(
            self.config.metadata_url(),
            headers={"Accept": "application/json", "Authorization": self.config._authorization_header()},
            method="GET",
        )
        try:
            with self._urlopen(request, timeout=float(timeout_s)) as response:
                status = int(getattr(response, "status", 200))
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            # Read only to detect an accidental credential echo; never retain
            # or include the body in an error/report.
            try:
                body = exc.read()
            except Exception:
                body = b""
            self._check_echo(body)
            raise MetadataGateBlocked("unauthorized" if status == 401 else "http_error", http_status=status)
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise MetadataGateBlocked("transport_error") from exc
        self._check_echo(body)
        if status == 401:
            raise MetadataGateBlocked("unauthorized", http_status=status)
        if status < 200 or status >= 300:
            raise MetadataGateBlocked("http_error", http_status=status)
        try:
            decoded = json.loads(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MetadataGateBlocked("invalid_json", http_status=status) from exc
        if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), list):
            raise MetadataGateBlocked("invalid_model_list", http_status=status)
        model_ids = tuple(
            str(item.get("id")) for item in decoded["data"]
            if isinstance(item, dict) and item.get("id") is not None
        )
        if self.config.model_id not in model_ids:
            raise MetadataGateBlocked("model_missing", http_status=status)
        self._gate_passed = True
        return MetadataResult(status, model_ids, self.config.model_id, self._requests, "PASS")

    def _check_echo(self, body: bytes) -> None:
        if self.config._credential and self.config._credential.encode("utf-8") in body:
            raise CredentialEchoError()

    def build_chat_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Build (but never send) a production-default thinking payload."""
        if not self._gate_passed:
            raise MetadataGateBlocked("completion_blocked_until_metadata_pass")
        if not messages:
            raise QwenConfigError("chat messages must be non-empty")
        if max_tokens <= 0:
            raise QwenConfigError("max_tokens must be positive")
        return {
            "model": self.config.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }


def public_json(value: Any) -> str:
    """Serialize only explicitly public metadata; reject private config objects."""
    if isinstance(value, (EnvSnapshot, QwenProviderConfig)):
        value = value.public_metadata()
    elif isinstance(value, MetadataResult):
        value = value.public_metadata()
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
