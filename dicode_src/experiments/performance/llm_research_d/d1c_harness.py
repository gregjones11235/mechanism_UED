#!/usr/bin/env python3
"""D1c production-shape embedding retry harness (independent research line).

Sends Mason-shape BATCHED embedding requests (one list of texts per request)
under three client-lifecycle arms — persistent/contiguous, fresh-client-per-
request, and persistent-with-idle-gap — to test whether the historical 574/575
SDK transport retries were caused by a stale keep-alive connection.

Reuses the SDKRetryCounter (which observes OpenAI-SDK internal retries) and the
GPU sampler. Does NOT import production llm.py / gen_manager, and never calls
JAX/PPO/preflight.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from llm_replay_harness import enable_sdk_retry_counting
from llm_replay_manifest import sha256_text

EVENT_FIELDS = (
    "run_id", "arm", "repeat", "request_id", "batch_index", "batch_size",
    "client_lifecycle", "idle_gap_s", "start_monotonic_ns", "end_monotonic_ns",
    "duration_s", "attempt", "sdk_retry_count", "outer_retry_count",
    "retry_backoff_s", "status", "http_status", "error_class", "exception_class",
    "response_item_count", "expected_item_count", "embedding_shape",
    "input_sha256", "response_content_sha256", "ollama_pid_before", "ollama_pid_after",
)

ERROR_CLASSES = (
    "timeout", "connection_error", "connection_reset", "remote_protocol_error",
    "server_error", "rate_limited", "empty_result", "item_count_mismatch",
    "shape_mismatch", "non_finite_embedding", "unknown_error",
)


def classify_error(exc: BaseException) -> tuple[str, str, int | None]:
    name = type(exc).__name__.lower()
    code = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) or "timeout" in name:
        return "timeout", type(exc).__name__, code
    if "reset" in name or "reset" in msg:
        return "connection_reset", type(exc).__name__, code
    if "remoteprotocol" in name or "protocol" in name or "incomplete" in msg:
        return "remote_protocol_error", type(exc).__name__, code
    if "connection" in name or "connect" in name or "network" in name:
        return "connection_error", type(exc).__name__, code
    if isinstance(code, int) and code == 429:
        return "rate_limited", type(exc).__name__, code
    if isinstance(code, int) and 500 <= code < 600:
        return "server_error", type(exc).__name__, code
    return "unknown_error", type(exc).__name__, code


def format_input(texts: list[str], instruction: str | None) -> list[str]:
    if instruction:
        return [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    return list(texts)


def validate_embedding(data: Any, expected_count: int, embedding_size: int):
    """Return (ok, error_class, item_count, shape). Enforces that every item's
    embedding dimension equals the configured embedding_size (fail-closed)."""
    items = list(getattr(data, "data", []) or [])
    item_count = len(items)
    if item_count == 0:
        return False, "empty_result", item_count, None
    if item_count != expected_count:
        return False, "item_count_mismatch", item_count, None
    shapes = set()
    for it in items:
        emb = getattr(it, "embedding", None)
        if emb is None:
            return False, "empty_result", item_count, None
        dim = len(emb)
        if dim != embedding_size:
            # configured dimension mismatch: fail-closed even if all items agree
            return False, "shape_mismatch", item_count, dim
        shapes.add(dim)
        try:
            if not all(__import__("math").isfinite(float(x)) for x in emb):
                return False, "non_finite_embedding", item_count, dim
        except Exception:
            return False, "non_finite_embedding", item_count, dim
    if len(shapes) != 1:
        return False, "shape_mismatch", item_count, sorted(shapes)
    return True, None, item_count, shapes.pop()


class EventWriter:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write(self, event: dict):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({k: event.get(k) for k in EVENT_FIELDS}, sort_keys=True, default=str) + "\n")


class D1CEmbeddingClient:
    def __init__(self, *, base_url, model, api_key, timeout_s, max_retries, embedding_size):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.embedding_size = embedding_size
        self._client = None

    def _get(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key,
                                       timeout=self.timeout_s, max_retries=self.max_retries)
        return self._client

    def reset(self):
        self._client = None

    async def aclose(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    async def embed(self, texts, instruction, sdk_counter, writer, *, run_id, arm,
                    repeat, request_id, batch_index, batch_size, client_lifecycle,
                    idle_gap_s, ollama_pid_before, ollama_pid_after_getter):
        sdk_counter.reset()
        formatted = format_input(texts, instruction)
        input_sha256 = sha256_text("\n".join(formatted))
        client = self._get()
        start = time.monotonic_ns()
        status = "ok"
        error_class = None
        exception_class = None
        http_status = None
        response_item_count = None
        embedding_shape = None
        response_content_sha256 = None
        attempt = 1
        try:
            resp = await client.embeddings.create(model=self.model, input=formatted)
            ok, err_cls, item_count, shape = validate_embedding(resp, len(texts), self.embedding_size)
            if not ok:
                status = "error"
                error_class = err_cls
            else:
                response_item_count = item_count
                embedding_shape = shape
                response_content_sha256 = sha256_text(
                    json.dumps([list(getattr(i, "embedding", [])) for i in resp.data], default=str))
        except Exception as e:
            status = "error"
            error_class, exception_class, http_status = classify_error(e)
        end = time.monotonic_ns()
        # sample the AFTER pid only once the await has returned/raised
        ollama_pid_after = ollama_pid_after_getter() if callable(ollama_pid_after_getter) else ollama_pid_after_getter
        writer.write({
            "run_id": run_id, "arm": arm, "repeat": repeat, "request_id": request_id,
            "batch_index": batch_index, "batch_size": batch_size,
            "client_lifecycle": client_lifecycle, "idle_gap_s": idle_gap_s,
            "start_monotonic_ns": start, "end_monotonic_ns": end,
            "duration_s": (end - start) / 1e9, "attempt": attempt,
            "sdk_retry_count": sdk_counter.count(), "outer_retry_count": 0,
            "retry_backoff_s": 0.0, "status": status, "http_status": http_status,
            "error_class": error_class, "exception_class": exception_class,
            "response_item_count": response_item_count,
            "expected_item_count": len(texts), "embedding_shape": embedding_shape,
            "input_sha256": input_sha256,
            "response_content_sha256": response_content_sha256,
            "ollama_pid_before": ollama_pid_before,
            "ollama_pid_after": ollama_pid_after,
        })
        return {
            "status": status, "error_class": error_class, "exception_class": exception_class,
            "http_status": http_status, "sdk_retry_count": sdk_counter.count(),
            "response_item_count": response_item_count, "embedding_shape": embedding_shape,
        }
