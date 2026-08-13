#!/usr/bin/env python3
"""Independent LLM replay harness (stage D research line).

Sends frozen prompts to an OpenAI-compatible chat/embeddings endpoint with a
controllable ``max_in_flight`` concurrency limit, recording monotonic-clock
events to an append-only JSONL. It does NOT import the production ``LLM``
client, ``gen_manager`` orchestration, or ``preflight_replay``. The only
production objects touched are the ``Task`` loader and the static-lint logic,
imported lazily inside the validation functions so the timing core stays fully
independent (and so a provider outage fails closed before any validation runs).

Provider outage is fail-closed: ``ProviderUnavailableError`` is raised by
``health_check`` and the benchmark aborts rather than fabricating results.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# ``openai`` is imported lazily inside ``LLMReplayClient.__init__`` so the
# timing/classification core remains importable (and unit-testable) without the
# third-party SDK installed.

from llm_replay_manifest import (
    PHASES,
    ERROR_CLASSES,
    sha256_text,
)

EVENT_FIELDS = (
    "run_id", "replay_id", "stage", "provider", "model", "max_in_flight",
    "request_id", "candidate_slot", "phase", "parent_phase",
    "start_monotonic_ns", "end_monotonic_ns", "duration_s", "status",
    "attempt", "http_status", "error_class", "prompt_sha256",
    "response_sha256", "overlap_group",
)


class ProviderUnavailableError(RuntimeError):
    """Raised when the target provider cannot be reached (fail-closed)."""


def classify_error(exc: BaseException) -> tuple[str, int | None]:
    """Map a provider exception to a stable (error_class, http_status) tuple."""
    name = type(exc).__name__.lower()
    code = getattr(exc, "status_code", None)
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
        return "timeout", code
    if "connection" in name or "connect" in name or "network" in name:
        return "connection_error", code
    if isinstance(code, int) and code == 429:
        return "rate_limited", code
    if isinstance(code, int) and 500 <= code < 600:
        return "server_error", code
    if isinstance(code, int) and 400 <= code < 500:
        return "server_error", code  # client 4xx still surfaces as a failed attempt
    return "unknown_error", code


def extract_code(content: str | None) -> str | None:
    """Extract Python code from a response, mirroring production ``_extract_file``."""
    if not content:
        return None
    import re
    m = re.search(r"<code>\s*(.*?)\s*</code>", content, re.DOTALL)
    extracted = m.group(1).strip() if m else content
    return _strip_code_fences(extracted)


def _strip_code_fences(code: str | None) -> str | None:
    if code is None:
        return None
    lines = code.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def static_lint(code: str) -> tuple[bool, str]:
    """AST static check mirroring production ``EnvGenerator._static_lint``.

    Detects invalid Craftax enum members (BlockType/Achievement) and invalid
    Inventory kwargs — the two "invalid Craftax API candidate" classes.
    """
    import ast
    try:
        tree = ast.parse(code)
        from craftax.craftax.constants import BlockType, Achievement
        enum_members = {"BlockType": set(BlockType.__members__),
                        "Achievement": set(Achievement.__members__)}
        aliases: dict[str, str] = {}
        inventory_aliases: set[str] = set()
        try:
            from craftax.craftax.craftax_state import Inventory
            from dataclasses import fields
            inventory_fields = {f.name for f in fields(Inventory)}
        except Exception:
            inventory_fields = set(getattr(Inventory, "__annotations__", {})) \
                if "Inventory" in dir() else set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for name in node.names:
                    if node.module.endswith("constants") and name.name in enum_members:
                        aliases[name.asname or name.name] = name.name
                    if node.module.endswith("craftax_state") and name.name == "Inventory":
                        inventory_aliases.add(name.asname or name.name)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in aliases:
                if node.attr not in enum_members[aliases[node.value.id]]:
                    return False, f"invalid {aliases[node.value.id]} member {node.attr}"
            if inventory_fields and isinstance(node, ast.Call) \
                    and isinstance(node.func, ast.Name) and node.func.id in inventory_aliases:
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in inventory_fields:
                        return False, f"invalid Inventory kwarg {keyword.arg}"
        return True, ""
    except SyntaxError as exc:
        return False, f"Compilation error: {exc}"


def cpu_jax_validation(code: str, *, device: str = "cpu") -> tuple[bool, str]:
    """CPU JAX compile+run validation mirroring ``_check_compilation_uncached``.

    Lazily imports ``Task`` from the production gen_manager so the timing core
    stays independent; runs in an isolated temp file and cleans it up. Returns
    ``(False, ...)`` with a clear message if the craftax/jax environment is
    unavailable (fail-closed, never a silent success).
    """
    temp_file = None
    module_name = None
    try:
        import jax
        import jax.numpy as jnp
        try:
            from dicode.dreaming.gen_manager import Task
        except Exception as e:
            return False, f"Task import failed (environment unavailable): {e}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            cpu_device = jax.devices("cpu")[0]
        except IndexError:
            cpu_device = jax.local_devices(backend="cpu")[0]

        with jax.default_device(cpu_device):
            temp_task = Task(temp_file)
            env = temp_task.env
            module_name = temp_task.task.__module__
            params = env.default_params
            key = jax.random.PRNGKey(0)

            def _validate_on_cpu_impl(rng):
                rng, reset_key = jax.random.split(rng)
                obs, state = env.reset(reset_key, params)
                action = env.action_space(params).sample(rng)
                obs, state, reward, done, info = env.step(rng, state, action, params)
                for field_name, value in state.inventory.__dict__.items():
                    if hasattr(value, "dtype") and value.dtype != jnp.int32:
                        raise ValueError(
                            f"Inventory field '{field_name}' has type {value.dtype}, expected int32."
                        )
                return reward

            _validate_on_cpu = jax.jit(_validate_on_cpu_impl, backend="cpu")
            _ = _validate_on_cpu(key)
            _.block_until_ready()
        return True, ""
    except Exception as e:
        return False, f"Compilation error: {e}"
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        if module_name:
            import sys
            if module_name in sys.modules:
                del sys.modules[module_name]


class EventSink:
    """Append-only JSONL event recorder. No-op when profiling is disabled."""

    def __init__(self, *, output_jsonl: str | None, enabled: bool, run_id: str,
                 replay_id: str, provider: str, model: str, max_in_flight: int):
        self.enabled = bool(enabled)
        self.output_jsonl = os.fspath(output_jsonl) if output_jsonl else None
        self.run_id = run_id
        self.replay_id = replay_id
        self.provider = provider
        self.model = model
        self.max_in_flight = max_in_flight
        self._parent_stack: list[str] = []

    def _base(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "replay_id": self.replay_id,
            "stage": "llm_replay",
            "provider": self.provider,
            "model": self.model,
            "max_in_flight": self.max_in_flight,
        }

    def record(self, phase: str, *, start_monotonic_ns: int,
               end_monotonic_ns: int | None = None, status: str = "ok",
               attempt: int = 1, http_status: int | None = None,
               error_class: str | None = None, prompt_sha256: str | None = None,
               response_sha256: str | None = None, request_id: str | None = None,
               candidate_slot: str | None = None,
               parent_phase: str | None = None,
               overlap_group: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {}
        end = int(end_monotonic_ns if end_monotonic_ns is not None else time.monotonic_ns())
        parent = parent_phase if parent_phase is not None else (self._parent_stack[-1] if self._parent_stack else None)
        event = dict(self._base())
        event.update({
            "phase": phase,
            "parent_phase": parent,
            "start_monotonic_ns": int(start_monotonic_ns),
            "end_monotonic_ns": end,
            "duration_s": max(0, end - int(start_monotonic_ns)) / 1e9,
            "status": status,
            "attempt": attempt,
            "http_status": http_status,
            "error_class": error_class,
            "prompt_sha256": prompt_sha256,
            "response_sha256": response_sha256,
            "request_id": request_id,
            "candidate_slot": candidate_slot,
            "overlap_group": overlap_group,
        })
        row = {k: event.get(k) for k in EVENT_FIELDS}
        if self.output_jsonl:
            parent = os.path.dirname(self.output_jsonl)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.output_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return event

    @contextmanager
    def span(self, phase: str, **kwargs):
        start = time.monotonic_ns()
        token_holder = [None]
        self._parent_stack.append(phase)
        status = kwargs.pop("status", "ok")
        try:
            token_holder[0] = yield
        except Exception:
            status = "error"
            raise
        finally:
            self._parent_stack.pop()
            self.record(phase, start_monotonic_ns=start, status=status, **kwargs)
        _ = token_holder


class LLMReplayClient:
    """Async OpenAI-compatible client with a bounded in-flight window."""

    def __init__(self, *, base_url: str, model: str, provider: str,
                 temperature: float, top_p: float, max_tokens: int,
                 timeout_s: float, max_in_flight: int, sink: EventSink,
                 api_key: str = "token-", llm_type: str = "generation"):
        self.base_url = base_url
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_in_flight = max(1, int(max_in_flight))
        self.sink = sink
        self.llm_type = llm_type
        from openai import AsyncOpenAI  # lazy import (see module docstring)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._semaphore = asyncio.Semaphore(self.max_in_flight)
        self._counter = 0

    def next_request_id(self) -> str:
        self._counter += 1
        return f"{self.sink.run_id[:8]}-{self._counter:04d}"

    def _extra_body(self) -> dict[str, Any]:
        # Mirrors llm.py thinking-off/on per-model switch for faithful requests.
        m = self.model.lower()
        if "deepseek" in m:
            return {"reasoning_effort": "none"}
        if "qwen" in m or "glm" in m or "zai" in m:
            return {"chat_template_kwargs": {"enable_thinking": False}}
        return {"reasoning_effort": "none"}

    async def health_check(self) -> None:
        """Fail closed if the provider cannot be reached."""
        try:
            if self.llm_type == "embedding":
                await self.client.embeddings.create(model=self.model, input=["health"])
            else:
                await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "health"}],
                    max_tokens=1,
                )
        except Exception as e:
            raise ProviderUnavailableError(
                f"provider {self.provider} model {self.model} at {self.base_url} "
                f"unreachable: {type(e).__name__}: {e}"
            ) from e

    async def chat_once(self, system_prompt: str, user_prompt: str, *,
                        slot: str, request_id: str, attempt: int,
                        prompt_sha256: str) -> dict[str, Any]:
        """One chat attempt. Records queue_wait + chat_request."""
        q_start = time.monotonic_ns()
        async with self._semaphore:
            q_end = time.monotonic_ns()
            self.sink.record("queue_wait", start_monotonic_ns=q_start,
                             end_monotonic_ns=q_end, status="ok", attempt=attempt,
                             request_id=request_id, candidate_slot=slot,
                             prompt_sha256=prompt_sha256,
                             overlap_group=self.sink.replay_id)
            c_start = time.monotonic_ns()
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                extra_body = self._extra_body()
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    extra_body=extra_body,
                )
                content = resp.choices[0].message.content
                c_end = time.monotonic_ns()
                status = "ok" if content is not None and str(content).strip() else "empty"
                error_class = None if status == "ok" else "empty_response"
                response_sha256 = sha256_text(str(content)) if content is not None else None
                self.sink.record("chat_request", start_monotonic_ns=c_start,
                                 end_monotonic_ns=c_end, status=status, attempt=attempt,
                                 request_id=request_id, candidate_slot=slot,
                                 prompt_sha256=prompt_sha256,
                                 response_sha256=response_sha256,
                                 error_class=error_class,
                                 overlap_group=self.sink.replay_id)
                return {"content": content, "error_class": error_class,
                        "http_status": 200, "response_sha256": response_sha256,
                        "attempts": attempt}
            except Exception as e:
                c_end = time.monotonic_ns()
                error_class, http_status = classify_error(e)
                self.sink.record("chat_request", start_monotonic_ns=c_start,
                                 end_monotonic_ns=c_end, status="error", attempt=attempt,
                                 request_id=request_id, candidate_slot=slot,
                                 prompt_sha256=prompt_sha256,
                                 error_class=error_class, http_status=http_status,
                                 overlap_group=self.sink.replay_id)
                return {"content": None, "error_class": error_class,
                        "http_status": http_status, "response_sha256": None,
                        "attempts": attempt}

    async def chat_with_retries(self, system_prompt: str, user_prompt: str, *,
                                slot: str, request_id: str, prompt_sha256: str,
                                max_retries: int = 3) -> dict[str, Any]:
        """Chat with transport/empty retry (mirrors ``_query_with_retries``)."""
        last: dict[str, Any] = {}
        for attempt in range(1, max_retries + 1):
            last = await self.chat_once(system_prompt, user_prompt, slot=slot,
                                        request_id=request_id, attempt=attempt,
                                        prompt_sha256=prompt_sha256)
            if last.get("error_class") is None and last.get("content"):
                return last
            if attempt < max_retries:
                delay = 2 * (2 ** (attempt - 1))
                b_start = time.monotonic_ns()
                await asyncio.sleep(delay)
                self.sink.record("retry_backoff", start_monotonic_ns=b_start,
                                 end_monotonic_ns=time.monotonic_ns(), status="ok",
                                 attempt=attempt, request_id=request_id,
                                 candidate_slot=slot, prompt_sha256=prompt_sha256)
        return last

    async def embed_once(self, texts: list[str], *, slot: str, request_id: str,
                         attempt: int, prompt_sha256: str) -> dict[str, Any]:
        """One batched embedding attempt. Records queue_wait + embedding_request."""
        q_start = time.monotonic_ns()
        async with self._semaphore:
            q_end = time.monotonic_ns()
            self.sink.record("queue_wait", start_monotonic_ns=q_start,
                             end_monotonic_ns=q_end, status="ok", attempt=attempt,
                             request_id=request_id, candidate_slot=slot,
                             prompt_sha256=prompt_sha256,
                             overlap_group=self.sink.replay_id)
            e_start = time.monotonic_ns()
            try:
                resp = await self.client.embeddings.create(model=self.model, input=texts)
                e_end = time.monotonic_ns()
                status = "ok" if resp.data else "empty"
                self.sink.record("embedding_request", start_monotonic_ns=e_start,
                                 end_monotonic_ns=e_end, status=status, attempt=attempt,
                                 request_id=request_id, candidate_slot=slot,
                                 prompt_sha256=prompt_sha256,
                                 overlap_group=self.sink.replay_id)
                return {"content": resp.data, "error_class": None,
                        "http_status": 200, "response_sha256": None, "attempts": attempt}
            except Exception as e:
                e_end = time.monotonic_ns()
                error_class, http_status = classify_error(e)
                self.sink.record("embedding_request", start_monotonic_ns=e_start,
                                 end_monotonic_ns=e_end, status="error", attempt=attempt,
                                 request_id=request_id, candidate_slot=slot,
                                 prompt_sha256=prompt_sha256, error_class=error_class,
                                 http_status=http_status,
                                 overlap_group=self.sink.replay_id)
                return {"content": None, "error_class": error_class,
                        "http_status": http_status, "response_sha256": None,
                        "attempts": attempt}
