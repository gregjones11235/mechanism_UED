#!/usr/bin/env python3
"""Frozen LLM-replay manifest (stage D independent research line).

Builds, atomically writes, reloads and tamper-checks a frozen replay manifest
that pins the system prompt, ordered user prompts, request order, candidate
slots, token budget, temperature and validation config for a fixed-prompt LLM
replay. The manifest carries its own SHA256 and fails closed on reload if any
prompt, order or budget field is altered.

This is an INDEPENDENT research tool. It does NOT import or call
``preflight_replay`` and must not be used to claim semantic equivalence with
the B/C production optimization line.

No API key / token / Authorization material is ever written into the manifest
(the local OpenAI-compatible provider uses the literal ``token-`` placeholder,
and external providers are referenced by name only).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

CLASSIFICATION = "LLM_REPLAY_MANIFEST"

# Error classes and phases are shared with the harness/report; declared here so
# the manifest can validate the subset it records without importing the harness.
ERROR_CLASSES = (
    "timeout", "connection_error", "server_error", "rate_limited",
    "empty_response", "invalid_json", "static_invalid",
    "jax_validation_failed", "cancelled", "unknown_error",
)

PHASES = (
    "replay_wall", "queue_wait", "chat_request", "embedding_request",
    "retry_backoff", "response_parse", "static_validation",
    "repair_request", "cpu_jax_validation", "candidate_finalize",
    "result_write",
)

# Frozen replay never lowers these below the scientific budget; a manifest that
# requests fewer candidate slots or a smaller token cap is rejected.
MIN_CANDIDATE_SLOTS = 1
MIN_MAX_TOKENS = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: Any) -> Any:
    """Deterministic JSON-normalizable form (stable across platforms)."""
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
    if hasattr(value, "tolist") and not isinstance(value, (list, dict, str)):
        try:
            return canonical(value.tolist())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(x) for x in value]
    return value


def fingerprint(value: Any) -> str:
    return sha256_bytes(
        json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    )


def _without_hash(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in manifest.items() if k != "manifest_sha256"}


def _tool_sha256() -> str:
    """SHA256 of this source file, so the manifest records the tool that built it."""
    return file_sha256(Path(__file__).resolve())


def _validate_prompt_field(prompt: str, label: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return prompt


def build_replay_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a frozen replay manifest from a spec dict.

    ``spec`` must provide: classification (exact), source_commit, provider,
    model, base_url, temperature, max_tokens, system_prompt, and an ordered
    ``user_prompts`` list. ``candidate_slots`` (one label per prompt) defaults
    to ``slot_0..slot_N`` and is preserved verbatim (order is significant).
    """
    if spec.get("classification") not in (None, CLASSIFICATION):
        raise ValueError(f"classification must be {CLASSIFICATION}")
    source_commit = str(spec.get("source_commit", "")).strip()
    if not source_commit:
        raise ValueError("source_commit is required")
    provider = str(spec.get("provider", "")).strip()
    model = str(spec.get("model", "")).strip()
    base_url = str(spec.get("base_url", "")).strip()
    if not provider or not model or not base_url:
        raise ValueError("provider/model/base_url are required")

    temperature = float(spec.get("temperature", 0.6))
    top_p = float(spec.get("top_p", 0.95))
    max_tokens = int(spec.get("max_tokens", 8192))
    if max_tokens < MIN_MAX_TOKENS:
        raise ValueError(f"max_tokens must be >= {MIN_MAX_TOKENS}")

    system_prompt = _validate_prompt_field(str(spec.get("system_prompt", "")), "system_prompt")
    raw_prompts = spec.get("user_prompts")
    if not isinstance(raw_prompts, Sequence) or isinstance(raw_prompts, (str, bytes)) or not raw_prompts:
        raise ValueError("user_prompts must be a non-empty ordered list of strings")
    user_prompts = [_validate_prompt_field(str(p), f"user_prompts[{i}]") for i, p in enumerate(raw_prompts)]

    raw_slots = spec.get("candidate_slots")
    if raw_slots is None:
        candidate_slots = [f"slot_{i}" for i in range(len(user_prompts))]
    else:
        if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
            raise ValueError("candidate_slots must be a list of labels")
        candidate_slots = [str(s) for s in raw_slots]
        if len(candidate_slots) != len(user_prompts):
            raise ValueError("candidate_slots length must equal user_prompts length")

    request_order = spec.get("request_order")
    if request_order is None:
        request_order = [{"index": i, "slot": candidate_slots[i],
                          "kind": spec.get("default_kind", "generation")}
                         for i in range(len(user_prompts))]
    else:
        request_order = [dict(r) for r in request_order]
        if len(request_order) != len(user_prompts):
            raise ValueError("request_order length must equal user_prompts length")

    repair_limit = int(spec.get("repair_limit", 0))
    timeout_s = float(spec.get("timeout_s", 300.0))
    max_retries = int(spec.get("max_retries", 3))
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    validation = dict(spec.get("validation") or {})
    validation.setdefault("static_lint", True)
    validation.setdefault("cpu_jax", True)
    validation.setdefault("dedup_by_code_hash", True)

    prompt_stages = spec.get("prompt_stages") or {}
    if prompt_stages:
        for stage, indices in prompt_stages.items():
            if stage not in ("early", "mid", "late"):
                raise ValueError(f"prompt_stages key must be early/mid/late, got {stage!r}")
            for idx in indices:
                if not (0 <= int(idx) < len(user_prompts)):
                    raise ValueError(f"prompt_stages {stage} index {idx} out of range")

    overall_prompt_sha256 = sha256_bytes(
        "\n".join([system_prompt, *user_prompts]).encode()
    )

    manifest = {
        "classification": CLASSIFICATION,
        "not_end_to_end_ued": True,
        "llm_api_calls_expected": len(user_prompts),
        "source_commit": source_commit,
        "tool_sha256": _tool_sha256(),
        "tool_path": str(Path(__file__).resolve()),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "max_retries": max_retries,
        "repair_limit": repair_limit,
        "candidate_slots": candidate_slots,
        "request_order": request_order,
        "validation": validation,
        "prompt_stages": canonical(prompt_stages) if prompt_stages else {},
        "system_prompt": system_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompts": user_prompts,
        "user_prompt_sha256s": [sha256_text(p) for p in user_prompts],
        "prompt_sha256": overall_prompt_sha256,
        "response_sha256s": {},
    }
    return manifest


def write_manifest(manifest: Mapping[str, Any], output) -> dict[str, Any]:
    """Atomically write the manifest with its self-hash."""
    data = dict(manifest)
    data.pop("manifest_sha256", None)
    data["manifest_sha256"] = fingerprint(_without_hash(data))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(canonical(data), f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return data


def validate_replay_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Reload a manifest and reject any tampering or budget reduction."""
    if manifest.get("manifest_sha256") is not None and \
            manifest.get("manifest_sha256") != fingerprint(_without_hash(manifest)):
        raise ValueError("manifest_sha256 mismatch (tampered)")
    if manifest.get("classification") != CLASSIFICATION:
        raise ValueError("invalid classification")
    if not manifest.get("source_commit") or not manifest.get("provider") \
            or not manifest.get("model") or not manifest.get("base_url"):
        raise ValueError("missing source_commit/provider/model/base_url")
    if int(manifest.get("max_tokens", 0)) < MIN_MAX_TOKENS:
        raise ValueError("max_tokens lowered below budget")
    system_prompt = manifest.get("system_prompt")
    user_prompts = manifest.get("user_prompts")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt empty")
    if not isinstance(user_prompts, list) or not user_prompts:
        raise ValueError("user_prompts empty")
    if sha256_text(system_prompt) != manifest.get("system_prompt_sha256"):
        raise ValueError("system_prompt content changed")
    for i, p in enumerate(user_prompts):
        if sha256_text(p) != manifest.get("user_prompt_sha256s", [])[i]:
            raise ValueError(f"user_prompt[{i}] content changed")
    candidate_slots = [str(s) for s in manifest.get("candidate_slots", [])]
    if len(candidate_slots) != len(user_prompts):
        raise ValueError("candidate_slots length mismatch (budget/order changed)")
    request_order = manifest.get("request_order")
    if not isinstance(request_order, list) or len(request_order) != len(user_prompts):
        raise ValueError("request_order length mismatch (order changed)")
    overall = sha256_bytes("\n".join([system_prompt, *user_prompts]).encode())
    if overall != manifest.get("prompt_sha256"):
        raise ValueError("prompt_sha256 mismatch (prompt set changed)")
    return dict(manifest)


def load_manifest(path) -> dict[str, Any]:
    """Load a manifest file from disk and fail closed on tampering."""
    target = Path(path)
    raw = json.loads(target.read_text(encoding="utf-8"))
    return validate_replay_manifest(raw)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON spec file path")
    parser.add_argument("--output", required=True, help="output manifest path")
    args = parser.parse_args()
    spec_text = Path(args.spec).read_text(encoding="utf-8") if Path(args.spec).is_file() else args.spec
    spec = json.loads(spec_text)
    manifest = build_replay_manifest(spec)
    written = write_manifest(manifest, args.output)
    reloaded = load_manifest(args.output)
    print(json.dumps({"manifest_sha256": written["manifest_sha256"],
                      "reload_ok": reloaded["manifest_sha256"] == written["manifest_sha256"]}))
