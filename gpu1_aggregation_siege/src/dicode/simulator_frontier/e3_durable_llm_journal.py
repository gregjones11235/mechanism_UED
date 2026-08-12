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
ROLE_COMPLETION_CAPS = {
    "frontier_evidence_diagnostician": 1024,
    "curriculum_search_planner": 4096,
}
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
        result = {"schema": SCHEMA, "entries": entries}
        installed = payload.get("installed_target_keys", [])
        if not isinstance(installed, list) or len(installed) != len(set(installed)):
            raise ValueError("installed preseed key list invalid")
        if any(k not in entries for k in installed):
            raise ValueError("installed preseed key missing from entries")
        result["installed_target_keys"] = list(installed)
        provenance = payload.get("preseed_provenance")
        if provenance is not None:
            if not isinstance(provenance, Mapping):
                raise ValueError("durable LLM preseed provenance invalid")
            result["preseed_provenance"] = dict(provenance)
        return result

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
        cap = ROLE_COMPLETION_CAPS.get(str(entry.get("role")))
        if cap is None:
            return False
        if (usage["total_tokens"] <= 0 or usage["completion_tokens"] > cap
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
        cap = ROLE_COMPLETION_CAPS.get(str(identity.get("role")))
        if cap is None:
            raise ValueError("unknown E3 LLM role")
        if (clean_usage["total_tokens"] <= 0 or clean_usage["completion_tokens"] > cap
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
            installed_count = len(payload.get("installed_target_keys", []))
            if len(entries) - installed_count >= self.max_success_keys:
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

    def install_preseed(self, *, source_entry: Mapping[str, Any],
                        identity: Mapping[str, Any],
                        provenance: Mapping[str, Any]) -> dict[str, Any]:
        """Atomically install one validated diagnostician success.

        The source response is copied without changing content/output hashes;
        only the composite identity is re-keyed for the current signed run.
        """
        installed = self.install_preseed_entries(
            entries=[source_entry], identities=[identity], provenance=provenance)
        return installed[0]

    def install_preseed_entries(self, *, entries: list[Mapping[str, Any]],
                                identities: list[Mapping[str, Any]],
                                provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Atomically install a validated contiguous paid-call prefix."""
        if len(entries) != len(identities) or not entries or len(entries) > 302:
            raise ValueError("preseed entries must contain one through 302 roles")
        source_entries = [dict(item) for item in entries]
        identities = [dict(item) for item in identities]
        roles = []
        for source_entry, identity in zip(source_entries, identities):
            old_key = str(source_entry.get("key", ""))
            if not old_key or not self._valid_entry(old_key, source_entry):
                raise ValueError("preseed entry invalid")
            role = str(source_entry.get("role", ""))
            role_session = (role, int(identity.get("session", 0)))
            if role not in ROLE_COMPLETION_CAPS or role_session in roles:
                raise ValueError("preseed roles must be unique diagnostician/planner")
            roles.append(role_session)
            required = ("source_commit", "candidate", "session", "evidence_hash",
                        "role", "provider", "requested_model",
                        "client_implementation_hash")
            if any(k not in identity for k in required):
                raise ValueError("preseed identity incomplete")
            if identity["role"] != role or not isinstance(identity["session"], int) or identity["session"] < 1:
                raise ValueError("preseed role/session mismatch")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError("preseed migration provenance missing")
        if not any(role == "frontier_evidence_diagnostician" for role, _ in roles):
            raise ValueError("preseed must contain diagnostician success")
        by_session = {}
        for entry, identity in zip(source_entries, identities):
            session = int(identity["session"])
            by_session.setdefault(session, set()).add(str(entry.get("role", "")))
        max_session = max(by_session)
        if set(by_session) != set(range(1, max_session + 1)):
            raise ValueError("preseed sessions must be contiguous")
        for session, session_roles in by_session.items():
            if "curriculum_search_planner" in session_roles and "frontier_evidence_diagnostician" not in session_roles:
                raise ValueError("preseed planner requires same-session diagnostician")
            if session < max_session and session_roles != {"frontier_evidence_diagnostician", "curriculum_search_planner"}:
                raise ValueError("all completed preseed sessions require both roles")
        with self._file_lock():
            payload = self._load()
            if payload.get("entries"):
                raise ValueError("preseed target journal is not empty")
            new_entries = {}
            installed = []
            for source_entry, identity in zip(source_entries, identities):
                key = self.composite_key(**identity)
                entry = dict(source_entry)
                entry.update({"key": key, **identity, "key_identity": identity})
                if not self._valid_entry(key, entry):
                    raise ValueError("rekeyed preseed entry invalid")
                new_entries[key] = entry
                installed.append(dict(entry))
            payload = {"schema": SCHEMA, "entries": new_entries,
                       "installed_target_keys": list(new_entries),
                       "preseed_provenance": dict(provenance)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, sort_keys=True,
                              indent=2, default=str)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(tmp_name, self.path)
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            return installed

    def install_continuation_prefix(self, *, prefix_entries: list[Mapping[str, Any]],
                                    diagnostician_entry: Mapping[str, Any],
                                    diagnostician_identity: Mapping[str, Any],
                                    provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Atomically install immutable s1..s29 entries plus rekeyed s30 diag."""
        required_provenance = ("manifest_hash", "legacy_journal_sha256", "quarantine_key")
        if (len(prefix_entries) != 58 or not isinstance(provenance, Mapping)
                or any(not str(provenance.get(k, "")) for k in required_provenance)):
            raise ValueError("continuation prefix requires exactly 58 entries")
        all_entries = [dict(e) for e in prefix_entries] + [dict(diagnostician_entry)]
        if any(str(e.get("role")) not in ROLE_COMPLETION_CAPS for e in all_entries):
            raise ValueError("continuation contains unknown role")
        sessions = [(int(e.get("session", 0)), str(e.get("role"))) for e in prefix_entries]
        if set(sessions) != {(s, r) for s in range(1, 30) for r in ROLE_COMPLETION_CAPS}:
            raise ValueError("continuation prefix sessions/roles incomplete")
        for entry in prefix_entries:
            key = str(entry.get("key", ""))
            identity = entry.get("key_identity")
            if not isinstance(identity, Mapping) or dict(identity) != {
                    k: entry.get(k) for k in ("source_commit", "candidate", "session", "evidence_hash",
                                               "role", "provider", "requested_model", "client_implementation_hash")
            } or self.composite_key(**dict(identity)) != key or not self._valid_entry(key, entry):
                raise ValueError("continuation prefix entry identity invalid")
        if int(diagnostician_entry.get("session", 0)) != 30 or diagnostician_entry.get("role") != "frontier_evidence_diagnostician":
            raise ValueError("continuation session30 diagnostician required")
        if not self._valid_entry(str(diagnostician_entry.get("key", "")), diagnostician_entry):
            raise ValueError("continuation diagnostician entry invalid")
        with self._file_lock():
            payload = self._load()
            if payload.get("entries"):
                raise ValueError("continuation target journal must be empty")
            entries = {str(e["key"]): e for e in prefix_entries}
            if len(entries) != 58:
                raise ValueError("continuation prefix duplicate keys")
            identity = dict(diagnostician_identity)
            key = self.composite_key(**identity)
            diag = dict(diagnostician_entry)
            diag.update({"key": key, **identity, "key_identity": identity})
            if not self._valid_entry(key, diag):
                raise ValueError("continuation rekeyed diagnostician invalid")
            entries[key] = diag
            payload = {"schema": SCHEMA, "entries": entries,
                       "installed_target_keys": list(entries),
                       "preseed_provenance": dict(provenance)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=self.path.name + ".", dir=str(self.path.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=2, default=str)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
            return list(entries.values())

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
