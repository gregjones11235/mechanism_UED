#!/usr/bin/env python3
"""D3Q slot executor -- remote runner (stdlib-only, no jax/openai imports).

Phase 1 of the D3Q small-vs-large model matrix.  This module executes one
matrix slot end-to-end on the remote training server:

1. loads the frozen prompt bytes from ``FROZEN_MANIFEST.json`` (system prompt +
   ``user_prompts[prompt_index]``) and verifies the frozen hashes;
2. builds the exact arm payload (small arm: Ollama OpenAI-compatible, no
   thinking fields; large arm: DeepSeek official with ``thinking`` enabled);
3. reserves one POST from the frozen shared budget (``d3q_budget.D3QLedger``)
   *before* every POST -- the 4th POST of a slot or the 109th POST of a
   provider fails closed;
4. sends the request over raw ``http.client`` (no SDK, no implicit retry,
   max_in_flight=1, connect timeout 30s / read timeout 600s) and records
   HTTP status, request-id header, monotonic timings, token usage,
   finish_reason and the exact model field;
5. runs the fixed pipeline per response: save request metadata ->
   extract_code -> static_lint (syntax / forbidden import / Craftax enums /
   Inventory kwargs / dangerous capabilities) -> isolated CPU-JAX validation
   subprocess -> (if invalid and budget allows) semantic repair with the frozen
   repair template -> final freeze (raw responses are never overwritten);
6. writes the per-slot result JSON, the final code and the raw responses under
   ``<exec_root>/slots/<slot_id>/``.

The main runner process never imports ``jax``.  CPU-JAX validation runs in a
child process (``d3q_cpu_validate_remote.py``) with ``CUDA_VISIBLE_DEVICES=''``,
``JAX_PLATFORMS=cpu`` and ``PYTHONPATH=<mason worktree>/dicode_src/src``.

All static-analysis and classification helpers are self-contained and
offline-testable (see ``test_d3q_slot_executor.py``); ``static_lint`` accepts an
injectable enum/inventory map so local tests run without craftax installed.
"""
from __future__ import annotations

import ast
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Frozen D3Q contract constants (mirror D3Q_MATRIX_BINDING.json).
# ---------------------------------------------------------------------------

FROZEN_MANIFEST_SHA256 = "2066515499c305e263023e84a607c93ee4569708da57f23e56c8927c5b690d01"
FROZEN_MANIFEST_RAW_SHA256 = "6369c48540236d49f816f9ef98e82ac9adf6a0d9fc954539b31f32547ebe659e"
FROZEN_REPAIR_TEMPLATE_SHA256 = "beff6ea4307bd0ec19c14caf3034231b35e0e204aa84bafaaa68189389007c43"

SMALL_MODEL = "qwen2.5-coder:14b"
LARGE_MODEL = "deepseek-v4-flash"

SMALL_BASE_URL = "http://127.0.0.1:11434/v1"
LARGE_BASE_URL = "https://api.deepseek.com"

SMALL_PROVIDER = "ollama"
LARGE_PROVIDER = "deepseek_official"

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192}

CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 600.0
CPU_VALIDATION_TIMEOUT_S = 300.0

MAX_POSTS_PER_SLOT = 3
MAX_REPAIRS_PER_SLOT = 2
MAX_PROVIDER_POSTS = 108

SLOT_ID_RE = re.compile(r"^slot_r([123])_(small|large)_p(\d{2})$")

ARM_ORDER = (
    ("small", "r1"), ("large", "r1"),
    ("large", "r2"), ("small", "r2"),
    ("small", "r3"), ("large", "r3"),
)

# Exact per-arm payload shape.  "large" carries the frozen thinking marker and
# nothing else; "small" carries no thinking / extra_body / chat_template_kwargs
# fields at all.
PAYLOAD_EXPECTED_KEYS = {
    "large": frozenset({"model", "messages", "temperature", "top_p", "max_tokens", "thinking"}),
    "small": frozenset({"model", "messages", "temperature", "top_p", "max_tokens"}),
}
PAYLOAD_FORBIDDEN_KEYS = frozenset(
    {"extra_body", "chat_template_kwargs", "enable_thinking", "reasoning_effort"}
)

# Error classes recorded per attempt / per slot.
RETRYABLE_TRANSPORT_CLASSES = frozenset(
    {"connection_error", "timeout", "http_5xx", "empty_response"}
)
FATAL_API_STATUSES = frozenset({401, 402, 403})

# ---------------------------------------------------------------------------
# Secret handling.
# ---------------------------------------------------------------------------

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"(?i)\bauthorization\s*[:=]\s*Bearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]{40,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{24,}"),
    re.compile(r"BEGIN (RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY"),
)


def contains_secret(text: str) -> bool:
    """True when a secret-shaped token (sk-..., Bearer ..., api_key=...) appears."""
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def sanitize_text(text: str) -> str:
    """Strip secret-shaped tokens from error text before it reaches any artifact."""
    if not isinstance(text, str):
        return ""
    cleaned = text
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(value: Any) -> str:
    """Deterministic JSON serialization used for artifact hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_arm(arm: str) -> str:
    if arm not in ("small", "large"):
        raise ValueError(f"invalid arm {arm!r}; expected small or large")
    return arm


def arm_to_provider_model(arm: str) -> tuple[str, str, str]:
    """Return (provider, model, base_url) for the arm."""
    validate_arm(arm)
    if arm == "small":
        return SMALL_PROVIDER, SMALL_MODEL, SMALL_BASE_URL
    return LARGE_PROVIDER, LARGE_MODEL, LARGE_BASE_URL


def parse_slot_id(slot_id: str) -> tuple[str, str, int]:
    match = SLOT_ID_RE.fullmatch(slot_id)
    if not match:
        raise ValueError(f"invalid slot_id {slot_id!r}")
    repeat = match.group(1)
    arm = match.group(2)
    prompt_index = int(match.group(3))
    return repeat, arm, prompt_index


def load_frozen_manifest(path: str | os.PathLike) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != FROZEN_MANIFEST_RAW_SHA256:
        raise ValueError("FROZEN_MANIFEST.json raw sha256 mismatch (fail-closed)")
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("manifest_sha256") != FROZEN_MANIFEST_SHA256:
        raise ValueError("FROZEN_MANIFEST.json manifest_sha256 mismatch (fail-closed)")
    if not isinstance(manifest.get("system_prompt"), str) or not manifest["system_prompt"]:
        raise ValueError("FROZEN_MANIFEST.json missing system_prompt")
    user_prompts = manifest.get("user_prompts")
    if not isinstance(user_prompts, list) or len(user_prompts) != 12:
        raise ValueError("FROZEN_MANIFEST.json must contain exactly 12 user_prompts")
    return manifest


def load_frozen_repair_template(path: str | os.PathLike) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("classification") != "D3Q_FROZEN_REPAIR_TEMPLATE":
        raise ValueError("repair template classification mismatch")
    if data.get("template_sha256") != FROZEN_REPAIR_TEMPLATE_SHA256:
        raise ValueError("repair template sha256 mismatch (fail-closed)")
    template_text = data.get("template_text")
    if not isinstance(template_text, str) or not template_text:
        raise ValueError("repair template missing template_text")
    if sha256_text(template_text) != FROZEN_REPAIR_TEMPLATE_SHA256:
        raise ValueError("repair template_text sha256 mismatch (fail-closed)")
    return data


# ---------------------------------------------------------------------------
# Dotenv parsing (mirrors d3_deepseek_provider.parse_env_file semantics:
# plain data, no export, no expansion, private values stay in memory).
# ---------------------------------------------------------------------------


def _safe_key(name: str) -> str:
    key = str(name).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        raise ValueError("invalid environment variable name")
    return key


def _parse_env_value(raw: str, *, line_no: int) -> str:
    value = raw.strip()
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"invalid env value at line {line_no}")
    if "$(" in value or "`" in value:
        raise ValueError(f"shell expansion syntax at line {line_no}")
    if not value or value[0] not in {"'", '"'}:
        return value
    try:
        decoded = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        raise ValueError(f"invalid quoted env value at line {line_no}") from None
    if not isinstance(decoded, str) or any(char in decoded for char in ("\x00", "\n", "\r")):
        raise ValueError(f"invalid quoted env value at line {line_no}")
    if "$(" in decoded or "`" in decoded:
        raise ValueError(f"shell expansion syntax at line {line_no}")
    return decoded


def parse_env_file(path: str | os.PathLike) -> dict[str, str]:
    """Parse dotenv assignments into an in-memory dict (values never serialized)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        raise ValueError("unable to read declared env file") from None
    values: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"(?i)^export\s+", line):
            raise ValueError(f"shell export syntax at line {line_no}")
        if "=" not in line:
            raise ValueError(f"invalid env assignment at line {line_no}")
        raw_key, raw_value = line.split("=", 1)
        key = _safe_key(raw_key)
        if key in values:
            raise ValueError(f"duplicate environment variable at line {line_no}")
        values[key] = _parse_env_value(raw_value, line_no=line_no)
    return values


# ---------------------------------------------------------------------------
# Exact payload builders.
# ---------------------------------------------------------------------------


def build_payload(arm: str, system_prompt: str, user_prompt: str) -> dict[str, Any]:
    """Build the exact frozen payload for the arm (verified by tests)."""
    validate_arm(arm)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    base: dict[str, Any] = {
        "model": SMALL_MODEL if arm == "small" else LARGE_MODEL,
        "messages": messages,
        "temperature": SAMPLING["temperature"],
        "top_p": SAMPLING["top_p"],
        "max_tokens": SAMPLING["max_tokens"],
    }
    if arm == "large":
        base["thinking"] = {"type": "enabled"}
    if set(base) != PAYLOAD_EXPECTED_KEYS[arm]:
        raise ValueError(f"payload keys not exact for arm {arm!r}")
    if set(base) & PAYLOAD_FORBIDDEN_KEYS:
        raise ValueError(f"payload contains forbidden keys for arm {arm!r}")
    return base


def payload_to_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

# ---------------------------------------------------------------------------
# Code extraction (mirrors llm_replay_harness.extract_code).
# ---------------------------------------------------------------------------


def _strip_code_fences(code: str | None) -> str | None:
    if code is None:
        return None
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_code(content: str | None) -> str | None:
    """Extract Python code from a response, mirroring production ``_extract_file``."""
    if not content:
        return None
    match = re.search(r"<code>\s*(.*?)\s*</code>", content, re.DOTALL)
    extracted = match.group(1).strip() if match else content
    return _strip_code_fences(extracted)


# ---------------------------------------------------------------------------
# Static lint (mirrors production EnvGenerator._static_lint semantics plus the
# D3Q forbidden-import / dangerous-capability checks).
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "os", "sys", "subprocess", "socket", "requests", "urllib", "urllib3",
        "pathlib", "shutil", "ctypes", "importlib", "http", "ftplib", "smtplib",
        "poplib", "telnetlib", "xmlrpc", "pickle", "marshal", "shelve", "sqlite3",
        "multiprocessing", "pdb", "code", "pty", "cffi", "builtins", "gc", "inspect",
        "tracemalloc", "faulthandler", "resource", "signal",
    }
)

DANGEROUS_BUILTIN_CALLS = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "memoryview"}
)


def static_lint(
    code: str,
    *,
    enum_members: Optional[Mapping[str, set[str]]] = None,
    inventory_fields: Optional[set[str]] = None,
) -> tuple[str, str]:
    """Static check returning ``(error_class, message)``.

    ``("", "")`` means pass.  Mirrors the frozen production lint for syntax /
    Craftax enums / Inventory kwargs and adds forbidden-import and dangerous
    capability checks.  When the craftax environment is unavailable and no map
    is injected the check fails closed with ``environment_unavailable``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return "syntax_error", f"Compilation error: {exc}"

    if enum_members is None or inventory_fields is None:
        try:
            from craftax.craftax.constants import BlockType, Achievement  # type: ignore
            from craftax.craftax.craftax_state import Inventory  # type: ignore
            from dataclasses import fields  # type: ignore

            enum_members = {
                "BlockType": set(BlockType.__members__),
                "Achievement": set(Achievement.__members__),
            }
            try:
                inventory_fields = {field.name for field in fields(Inventory)}
            except Exception:
                inventory_fields = set(getattr(Inventory, "__annotations__", {}))
        except Exception as exc:
            return "environment_unavailable", f"Static lint environment unavailable: {exc}"

    aliases: dict[str, str] = {}
    inventory_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                root = name.name.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    return "dangerous_import", f"forbidden import: {name.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    return "dangerous_import", f"forbidden import: {node.module}"
                for name in node.names:
                    if node.module.endswith("constants") and name.name in enum_members:
                        aliases[name.asname or name.name] = name.name
                    if node.module.endswith("craftax_state") and name.name == "Inventory":
                        inventory_aliases.add(name.asname or name.name)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in aliases:
                if node.attr not in enum_members[aliases[node.value.id]]:
                    return (
                        "api_enum_error",
                        f"invalid {aliases[node.value.id]} member {node.attr}",
                    )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_BUILTIN_CALLS:
                return "dangerous_capability", f"dangerous builtin call: {node.func.id}"
        if inventory_fields and isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in inventory_aliases:
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in inventory_fields:
                        return (
                            "inventory_error",
                            f"invalid Inventory kwarg {keyword.arg}",
                        )
    return "", ""


# ---------------------------------------------------------------------------
# Provider exception classification (mirrors llm_replay_harness.classify_error).
# ---------------------------------------------------------------------------


def classify_exception(exc: BaseException) -> str:
    name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in name:
        return "timeout"
    if isinstance(exc, (ConnectionError, OSError, ssl.SSLError)) or any(
        token in name for token in ("connection", "connect", "network", "http")
    ):
        return "connection_error"
    return "unknown_error"


# ---------------------------------------------------------------------------
# Raw HTTP POST via http.client (no SDK, no implicit retry).
# ---------------------------------------------------------------------------


def _open_connection(url: str, connect_timeout_s: float):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        context = ssl.create_default_context()
        connection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=connect_timeout_s, context=context
        )
    elif parsed.scheme == "http":
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=connect_timeout_s
        )
    else:
        raise ValueError(f"unsupported URL scheme {parsed.scheme!r}")
    connection.connect()
    return connection, parsed


def post_chat_completion(
    url: str,
    payload_bytes: bytes,
    headers: Mapping[str, str],
    *,
    connect_timeout_s: float = CONNECT_TIMEOUT_S,
    read_timeout_s: float = READ_TIMEOUT_S,
) -> dict[str, Any]:
    """One raw chat-completion POST.  Returns a fixed metadata dict; never raises
    transport errors (they are classified into ``error_class``)."""
    start_ns = time.monotonic_ns()
    result: dict[str, Any] = {
        "http_status": 0,
        "request_id": None,
        "response_headers": {},
        "body_bytes": b"",
        "error_class": None,
        "decoded": None,
        "finish_reason": None,
        "model_field": None,
        "connect_phase_s": None,
        "read_phase_s": None,
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": None,
    }
    connection = None
    try:
        connection, parsed = _open_connection(url, connect_timeout_s)
        connect_end_ns = time.monotonic_ns()
        result["connect_phase_s"] = (connect_end_ns - start_ns) / 1e9
        if connection.sock is not None:
            connection.sock.settimeout(read_timeout_s)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection.request("POST", path, body=payload_bytes, headers=dict(headers))
        response = connection.getresponse()
        status = int(response.status)
        header_map = {
            str(key).lower(): str(value) for key, value in response.getheaders()
        }
        result["http_status"] = status
        result["response_headers"] = header_map
        for header_name in ("x-request-id", "request-id", "request_id"):
            if header_name in header_map:
                result["request_id"] = header_map[header_name]
                break
        body = response.read()
        read_end_ns = time.monotonic_ns()
        result["read_phase_s"] = (read_end_ns - connect_end_ns) / 1e9
        result["body_bytes"] = body
    except (TimeoutError, socket.timeout) as exc:
        result["error_class"] = "timeout"
    except Exception as exc:
        result["error_class"] = classify_exception(exc)
    finally:
        result["end_monotonic_ns"] = time.monotonic_ns()
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    if result["error_class"] is None and 200 <= result["http_status"] < 300:
        body = result["body_bytes"]
        if not body:
            result["error_class"] = "empty_response"
        else:
            try:
                decoded = json.loads(body.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                decoded = None
                result["error_class"] = "invalid_json"
            if decoded is not None:
                if not isinstance(decoded, dict):
                    result["error_class"] = "invalid_json"
                else:
                    result["decoded"] = decoded
                    result["model_field"] = decoded.get("model")
                    choices = decoded.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0]
                        if isinstance(first, dict):
                            result["finish_reason"] = first.get("finish_reason")
    elif result["error_class"] is None:
        if 400 <= result["http_status"] < 500:
            result["error_class"] = "http_4xx"
        elif 500 <= result["http_status"] < 600:
            result["error_class"] = "http_5xx"
        else:
            result["error_class"] = "unknown_error"
    return result


def extract_usage(decoded: Mapping[str, Any]) -> dict[str, int]:
    """Token usage -> (prompt, completion, cached) with 0 defaults."""
    usage = decoded.get("usage")
    prompt = 0
    completion = 0
    cached = 0
    if isinstance(usage, dict):
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        if "prompt_cache_hit_tokens" in usage:
            cached = int(usage.get("prompt_cache_hit_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and not cached:
            cached = int(details.get("cached_tokens", 0) or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached}


# ---------------------------------------------------------------------------
# Repair prompt assembly (frozen template bytes + sanitized error information).
# ---------------------------------------------------------------------------

REPAIR_SUFFIX = (
    "\n\n=== PREVIOUS CANDIDATE (INVALID) ===\n"
    "{source}\n\n"
    "=== VALIDATION ERROR (SANITIZED) ===\n"
    "{error}\n\n"
    "Fix the candidate above so that it passes every validation check. "
    "Return the complete fixed file inside <code>...</code> tags."
)


def assemble_repair_user_prompt(
    template_text: str, source_text: str, error_message: str
) -> str:
    """Frozen formula: ``template_text + sanitized_error_information``."""
    if not isinstance(template_text, str) or not template_text:
        raise ValueError("repair template text missing")
    sanitized = sanitize_text(str(error_message))
    return template_text + REPAIR_SUFFIX.format(source=str(source_text), error=sanitized)


# ---------------------------------------------------------------------------
# Per-slot attempt record.
# ---------------------------------------------------------------------------


def empty_attempt(attempt_index: int, kind: str) -> dict[str, Any]:
    return {
        "attempt_index": int(attempt_index),
        "kind": kind,
        "http_status": 0,
        "request_id": None,
        "model_field": None,
        "model_field_exact": None,
        "finish_reason": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "generation_wall_s": 0.0,
        "error_class": None,
        "empty_response": 0,
        "timeout": 0,
        "connection_error": 0,
        "http_4xx": 0,
        "http_5xx": 0,
        "invalid_json": 0,
        "extract_error": 0,
        "syntax_error": 0,
        "api_enum_error": 0,
        "inventory_error": 0,
        "dangerous_import": 0,
        "dangerous_capability": 0,
        "cpu_jax_error": 0,
        "duplicate_code": 0,
        "code_sha256": None,
        "validation_valid": None,
        "validation_message": "",
        "cpu_validation_wall_s": 0.0,
        "payload_sha256": None,
    }

# ---------------------------------------------------------------------------
# Slot execution.
# ---------------------------------------------------------------------------


class D3QSlotRunnerError(RuntimeError):
    """Classified slot failure; message is a fixed, non-secret label."""


class CredentialEchoError(D3QSlotRunnerError):
    pass


class D3QSlotRunner:
    """Executes one matrix slot against a provider with the frozen pipeline."""

    def __init__(
        self,
        *,
        exec_root: str | os.PathLike,
        slot_id: str,
        manifest_path: str | os.PathLike,
        repair_template_path: str | os.PathLike,
        ledger_path: str | os.PathLike,
        env_file_path: str | os.PathLike | None,
        remote_python: str,
        cpu_validate_script: str | os.PathLike,
        mason_src_path: str | os.PathLike,
        run_id: str,
    ) -> None:
        self.exec_root = Path(exec_root)
        self.slot_id = str(slot_id)
        self.repeat, self.arm, self.prompt_index = parse_slot_id(self.slot_id)
        self.provider, self.model, self.base_url = arm_to_provider_model(self.arm)
        self.manifest_path = Path(manifest_path)
        self.repair_template_path = Path(repair_template_path)
        self.ledger_path = Path(ledger_path)
        self.env_file_path = Path(env_file_path) if env_file_path else None
        self.remote_python = str(remote_python)
        self.cpu_validate_script = Path(cpu_validate_script)
        self.mason_src_path = Path(mason_src_path)
        self.run_id = str(run_id)
        self.slot_dir = self.exec_root / "slots" / self.slot_id
        self.slot_dir.mkdir(parents=True, exist_ok=False)

        self.manifest = load_frozen_manifest(self.manifest_path)
        self.repair_template = load_frozen_repair_template(self.repair_template_path)
        self.credential: str | None = None
        if self.arm == "large":
            env = parse_env_file(self.env_file_path)
            credential = env.get("EXP_DEEPSEEK_API_KEY", "")
            if not credential:
                raise D3QSlotRunnerError("credential_missing")
            self.credential = credential

        from d3q_budget import D3QLedger  # deployed next to this module

        self.ledger = D3QLedger(self.ledger_path)
        self.ledger.load()

        self.attempts: list[dict[str, Any]] = []
        self.repair_requests = 0
        self.repair_success = 0
        self.generation_wall_s = 0.0
        self.repair_wall_s = 0.0
        self.cpu_validation_wall_s = 0.0
        self._seen_code_sha256: set[str] = set()
        self._first_candidate_seen = False
        self.final_code: str | None = None
        self.final_code_sha256: str | None = None
        self.initial_valid = False
        self.final_valid = False
        self.slot_counts: dict[str, int] = {
            key: 0
            for key in (
                "empty_response", "timeout", "connection_error", "http_4xx",
                "http_5xx", "invalid_json", "extract_error", "syntax_error",
                "api_enum_error", "inventory_error", "dangerous_import",
                "dangerous_capability", "cpu_jax_error", "duplicate_code",
            )
        }
        self.fatal_api_blocked = False

    # -- request helpers ---------------------------------------------------

    def _system_prompt(self) -> str:
        return self.manifest["system_prompt"]

    def _user_prompt(self) -> str:
        return self.manifest["user_prompts"][self.prompt_index]

    def _prompt_slot_name(self) -> str:
        order = self.manifest.get("request_order") or []
        for entry in order:
            if isinstance(entry, dict) and entry.get("index") == self.prompt_index:
                return str(entry.get("slot", ""))
        return ""

    def _verify_prompt_binding(self) -> None:
        order = self.manifest.get("request_order")
        if not isinstance(order, list):
            raise D3QSlotRunnerError("manifest_request_order_missing")
        entry = next(
            (
                item
                for item in order
                if isinstance(item, dict) and item.get("index") == self.prompt_index
            ),
            None,
        )
        if entry is None or entry.get("kind") != "code":
            raise D3QSlotRunnerError("manifest_request_order_mismatch")
        if not str(entry.get("slot", "")).startswith("task_"):
            raise D3QSlotRunnerError("manifest_request_order_mismatch")

    def _headers(self) -> dict[str, str]:
        if self.arm == "large":
            assert self.credential is not None
            authorization = f"Bearer {self.credential}"
        else:
            authorization = "Bearer ollama"
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": authorization,
        }

    def _endpoint_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def _check_echo_bytes(self, body: bytes) -> None:
        if self.arm != "large" or self.credential is None:
            return
        direct = self.credential.encode("utf-8")
        escaped = json.dumps(self.credential, ensure_ascii=True)[1:-1].encode("utf-8")
        if direct in body or escaped in body:
            raise CredentialEchoError("credential_echo_detected")

    def _iter_echo(self, value: Any) -> bool:
        if self.credential is None:
            return False
        if isinstance(value, str):
            return self.credential in value
        if isinstance(value, Mapping):
            return any(self._iter_echo(k) or self._iter_echo(v) for k, v in value.items())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return any(self._iter_echo(item) for item in value)
        return False

    # -- CPU-JAX validation (isolated subprocess) --------------------------

    def _cpu_validate(self, code_file: Path, result_file: Path) -> dict[str, Any]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
        env["PYTHONPATH"] = str(self.mason_src_path)
        argv = [
            self.remote_python,
            str(self.cpu_validate_script),
            "--code-file",
            str(code_file),
            "--result-file",
            str(result_file),
        ]
        start_ns = time.monotonic_ns()
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=CPU_VALIDATION_TIMEOUT_S,
                env=env,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic_ns() - start_ns) / 1e9
            return {
                "valid": False,
                "error_class": "cpu_jax_error",
                "message": "cpu jax validation timed out",
                "cpu_validation_wall_s": elapsed,
                "compile_s": None,
                "execute_s": None,
            }
        elapsed = (time.monotonic_ns() - start_ns) / 1e9
        result = {
            "valid": False,
            "error_class": "cpu_jax_error",
            "message": "cpu jax validation subprocess failed",
            "cpu_validation_wall_s": elapsed,
            "compile_s": None,
            "execute_s": None,
        }
        if completed.returncode == 0 and result_file.exists():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict):
                for key in ("valid", "message", "compile_s", "execute_s"):
                    if key in data:
                        result[key] = data[key]
                result["error_class"] = None if data.get("valid") else "cpu_jax_error"
        return result

    # -- per-response pipeline ---------------------------------------------

    def _pipeline(self, content: str) -> tuple[str | None, dict[str, Any]]:
        """Run extract -> static lint -> CPU-JAX.  Returns (code, outcome).

        ``outcome`` is normalized: keys are ``valid`` (bool),
        ``message`` (sanitized), ``error_class`` (str or None) and a
        ``count_keys`` mapping of slot counters to increment.
        """
        code = extract_code(content)
        if not code:
            return None, {
                "valid": False,
                "message": "no code block found in response",
                "error_class": "extract_error",
                "count_keys": {"extract_error": 1},
            }
        code_sha = sha256_text(code)
        if code_sha in self._seen_code_sha256:
            self.slot_counts["duplicate_code"] += 1
            self.attempts[-1]["duplicate_code"] = 1
        self._seen_code_sha256.add(code_sha)

        error_class, message = static_lint(code)
        if error_class:
            return code, {
                "valid": False,
                "message": sanitize_text(message),
                "error_class": error_class,
                "count_keys": {error_class: 1},
            }

        code_file = self.slot_dir / f"candidate_attempt_{len(self.attempts)}.py"
        code_file.write_text(code, encoding="utf-8")
        result_file = self.slot_dir / f"cpu_validate_attempt_{len(self.attempts)}.json"
        outcome = self._cpu_validate(code_file, result_file)
        count_keys = {"cpu_jax_error": 0 if outcome.get("valid") else 1}
        return code, {
            "valid": bool(outcome.get("valid")),
            "message": sanitize_text(str(outcome.get("message", ""))),
            "error_class": None if outcome.get("valid") else "cpu_jax_error",
            "cpu_validation_wall_s": outcome.get("cpu_validation_wall_s", 0.0),
            "compile_s": outcome.get("compile_s"),
            "execute_s": outcome.get("execute_s"),
            "count_keys": count_keys,
        }

    # -- exclusive writes (raw responses are never overwritten) -------------

    def _write_exclusive(self, path: Path, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(path, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _write_json_exclusive(self, path: Path, value: Any) -> None:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        self._write_exclusive(path, text.encode("utf-8"))

    # -- single POST --------------------------------------------------------

    def _post_once(
        self, kind: str, attempt_index: int, user_prompt: str
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Reserve budget, POST, record metadata, classify.

        Returns ``(attempt, decoded)``.  The decoded response object is kept
        out of every persisted artifact.
        """
        from d3q_budget import BudgetExceededError  # noqa: F401

        event = self.ledger.reserve(
            ts_utc=utc_now_iso(),
            slot_id=self.slot_id,
            model=self.model,
            provider=self.provider,
            kind=kind,
            attempt_index=attempt_index,
        )
        payload = build_payload(self.arm, self._system_prompt(), user_prompt)
        payload_bytes = payload_to_bytes(payload)
        attempt = empty_attempt(attempt_index, kind)
        attempt["payload_sha256"] = sha256_bytes(payload_bytes)
        start_ns = time.monotonic_ns()
        post = post_chat_completion(
            self._endpoint_url(),
            payload_bytes,
            self._headers(),
            connect_timeout_s=CONNECT_TIMEOUT_S,
            read_timeout_s=READ_TIMEOUT_S,
        )
        end_ns = time.monotonic_ns()
        wall_s = (end_ns - start_ns) / 1e9
        attempt["generation_wall_s"] = wall_s
        self.generation_wall_s += wall_s
        if kind == "semantic_repair":
            self.repair_wall_s += wall_s

        body = post["body_bytes"]
        decoded = post["decoded"]
        self._check_echo_bytes(body)
        if decoded is not None and self._iter_echo(decoded):
            raise CredentialEchoError("credential_echo_detected")

        attempt["http_status"] = post["http_status"]
        attempt["request_id"] = post["request_id"]
        attempt["error_class"] = post["error_class"]
        if post["error_class"] in (
            "empty_response", "timeout", "connection_error", "http_4xx",
            "http_5xx", "invalid_json",
        ):
            attempt[post["error_class"]] = 1
            self.slot_counts[post["error_class"]] += 1
        if decoded is not None:
            attempt["model_field"] = post["model_field"]
            attempt["model_field_exact"] = post["model_field"] == self.model
            if self.model != post["model_field"]:
                raise D3QSlotRunnerError("model_field_mismatch")
            attempt["finish_reason"] = post["finish_reason"]
            usage = extract_usage(decoded)
            attempt["prompt_tokens"] = usage["prompt_tokens"]
            attempt["completion_tokens"] = usage["completion_tokens"]
            attempt["cached_tokens"] = usage["cached_tokens"]

        raw_path = self.slot_dir / f"raw_{self.slot_id}_a{attempt_index}.txt"
        self._write_exclusive(raw_path, body)
        meta_path = self.slot_dir / f"request_{self.slot_id}_a{attempt_index}.json"
        metadata = {
            "classification": "D3Q_REQUEST_METADATA",
            "run_id": self.run_id,
            "slot_id": self.slot_id,
            "arm": self.arm,
            "repeat": self.repeat,
            "provider": self.provider,
            "model": self.model,
            "kind": kind,
            "attempt_index": attempt_index,
            "prompt_index": self.prompt_index,
            "prompt_slot": self._prompt_slot_name(),
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "generation_wall_s": wall_s,
            "http_status": post["http_status"],
            "request_id": post["request_id"],
            "finish_reason": attempt["finish_reason"],
            "model_field": attempt["model_field"],
            "model_field_exact": attempt["model_field_exact"],
            "usage": {
                "prompt_tokens": attempt["prompt_tokens"],
                "completion_tokens": attempt["completion_tokens"],
                "cached_tokens": attempt["cached_tokens"],
            },
            "error_class": post["error_class"],
            "payload_sha256": attempt["payload_sha256"],
            "raw_response_sha256": sha256_bytes(body),
            "raw_response_file": raw_path.name,
            "ledger_event": dict(event),
            "timeouts_frozen": {"connect_s": CONNECT_TIMEOUT_S, "read_s": READ_TIMEOUT_S},
        }
        self._write_json_exclusive(meta_path, metadata)
        self.attempts.append(attempt)
        return attempt, decoded

    # -- repair -------------------------------------------------------------

    def _repair_prompt(self, source_text: str, error_message: str) -> str:
        template_text = self.repair_template["template_text"]
        return assemble_repair_user_prompt(template_text, source_text, error_message)

    # -- outcome application ------------------------------------------------

    def _apply_outcome(self, attempt: dict[str, Any], outcome: dict[str, Any]) -> None:
        attempt["validation_valid"] = bool(outcome.get("valid"))
        attempt["validation_message"] = sanitize_text(str(outcome.get("message", "")))
        if "cpu_validation_wall_s" in outcome:
            wall = float(outcome.get("cpu_validation_wall_s") or 0.0)
            attempt["cpu_validation_wall_s"] = wall
            self.cpu_validation_wall_s += wall
        for key, count in (outcome.get("count_keys") or {}).items():
            self.slot_counts[key] = self.slot_counts.get(key, 0) + count
            attempt[key] = attempt.get(key, 0) + count

    # -- main flow ----------------------------------------------------------

    def run(self) -> dict[str, Any]:
        self._verify_prompt_binding()
        user_prompt = self._user_prompt()
        attempt_index = 1
        kind = "initial"
        best_source = ""
        best_message = ""

        while attempt_index <= MAX_POSTS_PER_SLOT:
            attempt, decoded = self._post_once(kind, attempt_index, user_prompt)

            if attempt["http_status"] in FATAL_API_STATUSES:
                self.fatal_api_blocked = True
                break

            if attempt["error_class"] in RETRYABLE_TRANSPORT_CLASSES:
                if attempt_index < MAX_POSTS_PER_SLOT:
                    attempt_index += 1
                    kind = "transport_retry"
                    continue
                break

            if attempt["error_class"] in ("http_4xx", "invalid_json"):
                # HTTP 4xx is never retried; an unparseable 2xx body is not a
                # semantic candidate and is not repaired.
                break

            content = ""
            if attempt["error_class"] is None and isinstance(decoded, dict):
                choices = decoded.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
                    if isinstance(message, dict):
                        content = str(message.get("content") or "")

            code = None
            if content:
                code, outcome = self._pipeline(content)
            else:
                outcome = {
                    "valid": False,
                    "message": "empty generation content",
                    "error_class": "extract_error",
                    "count_keys": {"extract_error": 1},
                }

            self._apply_outcome(attempt, outcome)

            if not self._first_candidate_seen:
                self._first_candidate_seen = True
                self.initial_valid = bool(outcome.get("valid"))

            if outcome.get("valid"):
                self.final_valid = True
                if kind == "semantic_repair":
                    self.repair_success += 1
                if code is not None:
                    self.final_code = code
                    self.final_code_sha256 = sha256_text(code)
                break

            # invalid candidate -> semantic repair (same model, frozen template)
            if code is not None:
                best_source = code
            else:
                best_source = content or "(no extractable code)"
            best_message = str(outcome.get("message", ""))
            if self.repair_requests >= MAX_REPAIRS_PER_SLOT:
                break
            if attempt_index >= MAX_POSTS_PER_SLOT:
                break
            self.repair_requests += 1
            user_prompt = self._repair_prompt(best_source, best_message)
            attempt_index += 1
            kind = "semantic_repair"

        if self.final_valid and self.final_code:
            final_path = self.slot_dir / "final_code.py"
            self._write_exclusive(final_path, self.final_code.encode("utf-8"))

        return self._result()

    # -- result -------------------------------------------------------------

    def _result(self) -> dict[str, Any]:
        return {
            "classification": "D3Q_SLOT_RESULT",
            "schema_version": 1,
            "run_id": self.run_id,
            "slot_id": self.slot_id,
            "arm": self.arm,
            "repeat": self.repeat,
            "prompt_index": self.prompt_index,
            "prompt_slot": self._prompt_slot_name(),
            "provider": self.provider,
            "model": self.model,
            "initial_valid": self.initial_valid,
            "final_valid": self.final_valid,
            "attempts": len(self.attempts),
            "repair_requests": self.repair_requests,
            "repair_success": self.repair_success,
            "empty_response": self.slot_counts["empty_response"],
            "timeout": self.slot_counts["timeout"],
            "connection_error": self.slot_counts["connection_error"],
            "http_4xx": self.slot_counts["http_4xx"],
            "http_5xx": self.slot_counts["http_5xx"],
            "invalid_json": self.slot_counts["invalid_json"],
            "extract_error": self.slot_counts["extract_error"],
            "syntax_error": self.slot_counts["syntax_error"],
            "api_enum_error": self.slot_counts["api_enum_error"],
            "inventory_error": self.slot_counts["inventory_error"],
            "dangerous_import": self.slot_counts["dangerous_import"],
            "dangerous_capability": self.slot_counts["dangerous_capability"],
            "cpu_jax_error": self.slot_counts["cpu_jax_error"],
            "duplicate_code": self.slot_counts["duplicate_code"],
            "prompt_tokens": sum(a["prompt_tokens"] for a in self.attempts),
            "completion_tokens": sum(a["completion_tokens"] for a in self.attempts),
            "cached_tokens": sum(a["cached_tokens"] for a in self.attempts),
            "generation_wall_s": round(self.generation_wall_s, 6),
            "repair_wall_s": round(self.repair_wall_s, 6),
            "cpu_validation_wall_s": round(self.cpu_validation_wall_s, 6),
            "final_code_sha256": self.final_code_sha256,
            "fatal_api_blocked": self.fatal_api_blocked,
            "timeouts_frozen": {"connect_s": CONNECT_TIMEOUT_S, "read_s": READ_TIMEOUT_S},
            "max_posts_per_slot": MAX_POSTS_PER_SLOT,
            "max_repairs_per_slot": MAX_REPAIRS_PER_SLOT,
            "attempts_detail": list(self.attempts),
            "ledger_counts": {
                "slot": self.ledger.slot_post_count(self.slot_id),
                "provider": self.ledger.provider_post_count(self.provider),
            },
            "final_code_file": "final_code.py" if self.final_valid else None,
            "manifest_sha256": FROZEN_MANIFEST_SHA256,
        }


# ---------------------------------------------------------------------------
# Filesystem ops used by the launcher over SSH (stdlib, path-validated).
# ---------------------------------------------------------------------------

_SAFE_EXEC_ROOT = re.compile(r"^/tmp/d3q_exec_[0-9]{8}T[0-9]{6}Z$")


def validate_exec_root(path: str) -> str:
    if not _SAFE_EXEC_ROOT.fullmatch(path):
        raise ValueError(f"invalid exec root {path!r}")
    return path


def fs_path_exists(path: str) -> bool:
    return Path(path).exists()


def fs_create_dir_exclusive(path: str) -> None:
    Path(path).mkdir(parents=False, exist_ok=False)


def fs_remove_tree(path: str) -> None:
    target = Path(validate_exec_root(path))
    shutil.rmtree(target)


def fs_sha256_file(path: str) -> str:
    return sha256_file(path)


def fs_ledger_summary(path: str) -> dict[str, Any]:
    from d3q_budget import D3QLedger

    ledger = D3QLedger(path)
    ledger.load()
    slot_counts = {}
    for slot_id, budget in sorted(getattr(ledger, "_slots", {}).items()):
        slot_counts[slot_id] = budget.post_count
    provider_counts = {}
    for provider, budget in sorted(getattr(ledger, "_providers", {}).items()):
        provider_counts[provider] = budget.post_count
    return {"exists": Path(path).exists(),
            "slot_counts": slot_counts, "provider_counts": provider_counts}


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exec-root", required=True)
    parser.add_argument("--slot-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repair-template", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--env-file", default="")
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--cpu-validate-script", required=True)
    parser.add_argument("--mason-src", required=True)
    args = parser.parse_args(argv)

    runner = D3QSlotRunner(
        exec_root=args.exec_root,
        slot_id=args.slot_id,
        manifest_path=args.manifest,
        repair_template_path=args.repair_template,
        ledger_path=args.ledger,
        env_file_path=args.env_file or None,
        remote_python=args.remote_python,
        cpu_validate_script=args.cpu_validate_script,
        mason_src_path=args.mason_src,
        run_id=args.run_id,
    )
    try:
        result = runner.run()
    except D3QSlotRunnerError as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc)}, sort_keys=True))
        return 3
    except Exception as exc:
        print(json.dumps({"status": "FAILED",
                          "reason": type(exc).__name__.lower()}, sort_keys=True))
        return 4
    result_path = runner.slot_dir / f"{args.slot_id}.result.json"
    runner._write_json_exclusive(result_path, result)
    print(json.dumps(
        {"status": "DONE", "slot_id": args.slot_id,
         "final_valid": result["final_valid"], "attempts": result["attempts"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
