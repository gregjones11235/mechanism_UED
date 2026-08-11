"""The REAL authorized six-role LLM runtime.

Transport: the server-authorized OpenAI-compatible endpoint (DashScope /
Qwen) configured through the process environment. Keys are NEVER
printed, copied or committed; the runtime only checks presence and
routes calls. A missing authorization fails closed.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, List, Sequence

REQUIRED_ENV_VARS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "QWEN_MODEL")


class LLMRuntimeError(RuntimeError):
    """Fail-closed LLM runtime violation."""


class RealSixRoleLLMClient:
    """A real LLM client with the E1 query surface.

    ``query(system_prompt, user_prompts, *, cache_key, role)`` performs a
    REAL transport call (one per role invocation) and returns
    ``[{"content": str}]``. Every call is journaled (role, cache_key,
    request id, latency, token usage when provided) for the compute
    ledger; duplicate cache_keys within one run hard-fail (idempotent
    billing protection).
    """

    def __init__(self, *, model: str, base_url: str, journal: List[dict],
                 context: str = "real-six-role-client"):
        self._model = model
        self._base_url = base_url
        self._journal = journal
        self._context = context
        self._seen_keys: set = set()

    def query(self, system_prompt: str, user_prompts: Sequence[str], *,
              cache_key: str, role: str) -> List[Dict[str, Any]]:
        if not isinstance(cache_key, str) or not cache_key:
            raise LLMRuntimeError(
                f"{self._context}: cache_key is required")
        if cache_key in self._seen_keys:
            raise LLMRuntimeError(
                f"{self._context}: duplicate paid call for cache_key "
                f"{cache_key!r} role={role!r} — idempotent billing "
                "protection (hard fail, never re-bill)")
        self._seen_keys.add(cache_key)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMRuntimeError(
                f"{self._context}: OPENAI_API_KEY absent — the real "
                "transport is unauthorized (fail closed)")
        started = time.time()
        content = self._call_transport(system_prompt, list(user_prompts))
        self._journal.append({
            "role": role,
            "cache_key": cache_key,
            "model": self._model,
            "wall_seconds": round(time.time() - started, 6),
            "content_sha256": hashlib.sha256(
                content.encode("utf-8")).hexdigest(),
        })
        # diagnostic echo of the REAL reply (never the API key); the
        # smoke evidence keeps a per-role response record
        self._journal.append({
            "role_reply": role,
            "content": content[:4000],
        })
        return [{"content": content}]

    def _call_transport(self, system_prompt: str,
                        user_prompts: List[str]) -> str:
        """One REAL transport call through the OpenAI-compatible
        endpoint. Retries transient failures with exponential backoff
        (5xx / timeouts) and records the request id."""
        import json as _json
        import urllib.request

        messages = [{"role": "system", "content": system_prompt}]
        for prompt in user_prompts:
            messages.append({"role": "user", "content": prompt})
        payload = _json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": 0.0,
        }).encode("utf-8")
        url = self._base_url.rstrip("/") + "/chat/completions"
        last_error = None
        #: generous budget for the env-code generation transport: each
        #: DeepSeek call can take 60-120s+ (large JSON-escaped craftax
        #: modules), and the endpoint intermittently drops long-lived
        #: connections — a single drop must NOT kill a whole smoke, so
        #: retry with exponential backoff (capped) across a wider window.
        for attempt in range(6):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer "
                                     + os.environ.get("OPENAI_API_KEY", ""),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as resp:
                    body = _json.loads(resp.read().decode("utf-8"))
                    request_id = resp.headers.get("x-request-id", "")
                    if request_id:
                        self._journal.append({
                            "request_id": request_id,
                            "attempt": attempt,
                        })
                    return body["choices"][0]["message"]["content"]
            except Exception as exc:  # transient -> backoff, retry
                last_error = exc
                time.sleep(min(2 ** attempt, 32))
        raise LLMRuntimeError(
            f"{self._context}: real transport failed after retries: "
            f"{last_error!r}")


class DeepSeekEnvCoderClient(RealSixRoleLLMClient):
    """A REAL DeepSeek client for env-code generation (separate from the
    six-role board transport). Reads the server-authorized DeepSeek
    credentials (EXP_DEEPSEEK_API_KEY / EXP_DEEPSEEK_BASE_URL /
    EXP_GENERATOR_MODEL_ID) — never printed, never stored."""

    def __init__(self, *, journal: List[dict]):
        api_key = os.environ.get("EXP_DEEPSEEK_API_KEY", "")
        base_url = os.environ.get("EXP_DEEPSEEK_BASE_URL",
                                  "https://api.deepseek.com")
        model = os.environ.get("EXP_GENERATOR_MODEL_ID",
                               "deepseek-v4-pro")
        if not api_key:
            raise LLMRuntimeError(
                "DEEPSEEK_ENVCODER_UNAUTHORIZED: EXP_DEEPSEEK_API_KEY is "
                "absent in the environment (never printed); fail closed")
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["QWEN_MODEL"] = model
        super().__init__(model=model, base_url=base_url, journal=journal,
                         context="deepseek-envcoder-client")


class AuthorizedSixRoleLLMRuntime:
    """The authorized six-role LLM runtime object (registry asset).

    ``make_client()`` returns a REAL client ONLY when the transport
    authorization (env vars) is present; otherwise it fails closed.
    The runtime never prints or stores key material.
    """

    def __init__(self):
        self.journal: List[dict] = []
        self.object_identity_hash = hashlib.sha256(
            b"shared_runtime.authorized_six_role_llm_runtime.v1"
        ).hexdigest()
        self.registry_identity = self.object_identity_hash

    def transport_authorized(self) -> bool:
        return all(
            str(os.environ.get(name, "")).strip()
            for name in REQUIRED_ENV_VARS
        )

    def require_transport(self) -> None:
        missing = [name for name in REQUIRED_ENV_VARS
                   if not str(os.environ.get(name, "")).strip()]
        if missing:
            raise LLMRuntimeError(
                "SIX_ROLE_LLM_UNAUTHORIZED: real LLM transport requires "
                f"{sorted(missing)} in the environment (keys are never "
                "printed); fail closed")

    def make_client(self) -> RealSixRoleLLMClient:
        self.require_transport()
        return RealSixRoleLLMClient(
            model=os.environ["QWEN_MODEL"],
            base_url=os.environ["OPENAI_BASE_URL"],
            journal=self.journal,
        )
