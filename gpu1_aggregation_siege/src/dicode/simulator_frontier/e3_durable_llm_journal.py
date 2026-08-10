"""Dependency-light durable journal for paid E3 LLM responses."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "simulator_frontier.e3-durable-llm-journal/v1"
MAX_SUCCESS_KEYS = 302
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def implementation_hash(client_path: str, journal_path: str) -> str:
    """Hash both complete implementation files for authorization binding."""
    h = hashlib.sha256()
    for path in (client_path, journal_path):
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _sha(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class DurablePaidCallJournal:
    def __init__(self, path: str, *, max_success_keys: int = MAX_SUCCESS_KEYS) -> None:
        self.path = Path(path)
        self.max_success_keys = int(max_success_keys)
        if self.max_success_keys <= 0:
            raise ValueError("max_success_keys must be positive")

    @staticmethod
    def composite_key(*, source_commit: str, candidate: str, session: int,
                      evidence_hash: str, role: str, provider: str,
                      requested_model: str, client_implementation_hash: str) -> str:
        return _sha({"source_commit": source_commit, "candidate": candidate,
                     "session": int(session), "evidence_hash": evidence_hash,
                     "role": role, "provider": provider,
                     "requested_model": requested_model,
                     "client_implementation_hash": client_implementation_hash})

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": SCHEMA, "entries": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"durable LLM journal unreadable: {exc!r}") from exc
        if payload.get("schema") != SCHEMA or not isinstance(payload.get("entries"), Mapping):
            raise ValueError("durable LLM journal schema invalid")
        entries = dict(payload["entries"])
        for key, entry in entries.items():
            if not self._valid_entry(key, entry):
                raise ValueError(f"durable LLM journal entry tampered: {key}")
        return {"schema": SCHEMA, "entries": entries}

    def _valid_entry(self, key: str, entry: Any) -> bool:
        if not isinstance(entry, Mapping) or entry.get("status") != "SUCCESS":
            return False
        required = ("key", "source_commit", "candidate", "session", "evidence_hash",
                    "role", "provider", "requested_model", "client_implementation_hash",
                    "content", "key_identity", "returned_model", "usage",
                    "response_content_hash", "validated_output",
                    "validated_output_hash")
        if any(k not in entry for k in required) or entry.get("key") != key:
            return False
        usage = entry.get("usage")
        if not isinstance(usage, Mapping):
            return False
        if any((not isinstance(usage.get(k), int) or usage.get(k) < 0)
               for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            return False
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            return False
        if (usage["total_tokens"] <= 0 or usage["completion_tokens"] > 1024
                or usage["total_tokens"] > 20000
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
            return False
        if entry.get("returned_model") != entry.get("requested_model"):
            return False
        output = entry.get("validated_output")
        if entry.get("requested_model") != entry.get("key_identity", {}).get("requested_model"):
            return False
        if entry.get("response_content_hash") != _sha(entry.get("content")):
            return False
        if _sha(entry.get("key_identity")) != key:
            return False
        return entry.get("validated_output_hash") == _sha(output)

    def lookup(self, key: str, *, identity: Mapping[str, Any] | None = None) -> Mapping[str, Any] | None:
        entry = self._load().get("entries", {}).get(key)
        if entry is None:
            return None
        if identity is not None and dict(entry.get("key_identity", {})) != dict(identity):
            raise ValueError("durable LLM journal identity mismatch")
        return dict(entry)

    def record_success(self, *, key: str, identity: Mapping[str, Any],
                       returned_model: str, usage: Mapping[str, int],
                       validated_output: Mapping[str, Any], raw_response: Any = None,
                       response_content: str | None = None) -> dict[str, Any]:
        identity = dict(identity)
        required = ("source_commit", "candidate", "session", "evidence_hash", "role",
                    "provider", "requested_model", "client_implementation_hash")
        if any(k not in identity for k in required):
            raise ValueError("durable LLM identity incomplete")
        if self.composite_key(**identity) != key:
            raise ValueError("durable LLM composite key does not match identity")
        if (not isinstance(usage, Mapping)
                or any(k not in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens"))):
            raise ValueError("LLM usage is missing")
        clean_usage = {k: int(usage[k]) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        if (clean_usage["total_tokens"] <= 0 or clean_usage["completion_tokens"] > 1024
                or clean_usage["total_tokens"] > 20000
                or clean_usage["total_tokens"] != clean_usage["prompt_tokens"] + clean_usage["completion_tokens"]):
            raise ValueError("LLM usage is missing or exceeds the signed token contract")
        with self._file_lock():
            payload = self._load()
            entries = payload["entries"]
            if key in entries:
                if not self._valid_entry(key, entries[key]) or dict(entries[key].get("key_identity", {})) != identity:
                    raise ValueError("existing durable journal entry is invalid/tampered")
                return dict(entries[key])
            if len(entries) >= self.max_success_keys:
                raise ValueError("E3 durable paid-call journal success ceiling exceeded")
            entry = {
            "schema": SCHEMA, "status": "SUCCESS", "key": key,
            "source_commit": identity["source_commit"], "candidate": identity["candidate"],
            "session": identity["session"], "evidence_hash": identity["evidence_hash"],
            "role": identity["role"], "provider": identity["provider"],
            "client_implementation_hash": identity["client_implementation_hash"],
            "key_identity": identity, "requested_model": str(identity["requested_model"]),
            "returned_model": str(returned_model), "usage": clean_usage,
            "content": str(response_content if response_content is not None else ""),
            "response_content_hash": _sha(str(response_content if response_content is not None else "")),
            "validated_output_hash": _sha(validated_output),
            "validated_output": dict(validated_output),
            "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
            if raw_response is not None:
                entry["raw_response"] = raw_response
            entries[key] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2, default=str)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name): os.unlink(tmp_name)
            return dict(entry)

    @contextmanager
    def _file_lock(self):
        lock_path = str(self.path) + ".lock"
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(lock_path, threading.Lock())
        thread_lock.acquire()
        fh = open(lock_path, "a+b")
        unlock = lambda: None
        try:
            try:
                import msvcrt
                fh.seek(0); fh.write(b"0"); fh.flush(); fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                unlock = lambda: msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            except ImportError:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                unlock = lambda: fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            yield
        finally:
            try: unlock()
            except Exception: pass
            fh.close()
            thread_lock.release()
