"""P0-1: the real-LLM call journal (audit-grade, append-only, hash-chained).

Master-directive LLM call rules: every real call must record role, model,
backend, logical/API request id, prompt/response hash, parse status, token
usage and retry count; UNBOUNDED RETRIES AND DUPLICATE SUCCESSFUL CALLS ARE
FORBIDDEN. This module is that journal for the production path:

* entries are append-only and hash-chained (every ``entry_hash`` folds in
  its predecessor's) — the journal is evidence and cannot be rewritten;
* once a logical call id carries a PARSED schema outcome, ANY further entry
  under the same id is refused (``DUPLICATE_SUCCESSFUL_CALL``) — a
  successful call is never repeated;
* every transport entry's ``retry_count`` is checked against the round cap
  — an unbounded retry loop cannot even be recorded;
* ``window`` / ``sequence`` / ``artifact_binding`` are unknown to a backend
  (it only ever sees role+prompt); transport entries carry sentinel
  defaults until the caller binds them through
  :meth:`RealCallJournalEntry.bound_copy`. The bound copy is a DERIVED
  artifact stamped onto the window record / envelope side; the journal
  chain itself keeps the transport entry.

Mock / replay backends NEVER journal: journaling is a property of real,
paid calls only. The journal does not store prompt or response TEXT — only
their sha256 identities — and it never sees credentials (those live solely
inside the transport closure).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.schemas.common import CanonicalModel, is_sha256_hex

# ---------------------------------------------------------------------------
# journal vocabulary
# ---------------------------------------------------------------------------
ENTRY_KIND_TRANSPORT = "TRANSPORT"
ENTRY_KIND_SCHEMA_OUTCOME = "SCHEMA_OUTCOME"
ENTRY_KINDS = frozenset({ENTRY_KIND_TRANSPORT, ENTRY_KIND_SCHEMA_OUTCOME})

#: output schema status of a real call (master-directive "parse" field)
OUTPUT_SCHEMA_PENDING = "PENDING_PARSE"
OUTPUT_SCHEMA_PARSED = "PARSED"
OUTPUT_SCHEMA_FAILED = "SCHEMA_FAILED"
OUTPUT_SCHEMA_STATUSES = frozenset({OUTPUT_SCHEMA_PENDING,
                                    OUTPUT_SCHEMA_PARSED,
                                    OUTPUT_SCHEMA_FAILED})

#: token-usage honesty: a transport that does not report usage is labelled,
#: never silently zero-filled as though measured
TOKEN_USAGE_PROVIDED = "PROVIDED_BY_TRANSPORT"
TOKEN_USAGE_NOT_PROVIDED = "NOT_PROVIDED_BY_TRANSPORT"
TOKEN_USAGE_STATUSES = frozenset({TOKEN_USAGE_PROVIDED,
                                  TOKEN_USAGE_NOT_PROVIDED})

#: default retry cap (matches RealBackendAdapter's default max_retries)
DEFAULT_RETRY_CAP = 2

#: genesis previous-hash of an empty journal chain
GENESIS_ENTRY_HASH = canonical_sha256({"journal": "real_call_journal.v1",
                                       "state": "empty"})


class JournalBlocked(RuntimeError):
    """Fail-closed refusal of the real call journal."""


class DuplicateSuccessfulCall(JournalBlocked):
    """A logical call id that already PARSED successfully may never be
    called (or journaled) again — duplicate successful calls are forbidden."""


# ---------------------------------------------------------------------------
# transport result normalization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RealTransportResult:
    """What a real transport may return: the raw text plus honest metadata.

    ``request_id`` is the API-side request id when the provider exposes one;
    ``token_usage`` the provider-reported usage (prompt/completion tokens).
    Both are OPTIONAL honesty channels — a bare ``str`` response is legal
    and is labelled, not silently upgraded.
    """

    raw: str
    request_id: str = ""
    token_usage: Optional[Dict[str, int]] = None


def normalize_transport_result(result: object
                               ) -> Tuple[str, str, Dict[str, int], str]:
    """Reduce a transport return value to (raw, request_id, token_usage,
    token_usage_status), fail closed on unknown shapes.

    A bare ``str`` carries no API request id: one is DERIVED deterministically
    from the response hash (``derived:<sha16>``) and marked as such — the
    journal never pretends a provider id existed.
    """
    if isinstance(result, RealTransportResult):
        raw = result.raw
        request_id = result.request_id
        token_usage: Dict[str, int] = {}
        if result.token_usage is not None:
            for key, value in result.token_usage.items():
                if not isinstance(value, int) or isinstance(value, bool) \
                        or value < 0:
                    raise ValueError(
                        f"ILLEGAL_TOKEN_USAGE: {key}={value!r} — token "
                        "usage values must be non-negative ints")
                token_usage[str(key)] = value
            usage_status = TOKEN_USAGE_PROVIDED
        else:
            usage_status = TOKEN_USAGE_NOT_PROVIDED
    elif isinstance(result, str):
        raw = result
        request_id = ""
        token_usage = {}
        usage_status = TOKEN_USAGE_NOT_PROVIDED
    else:
        raise ValueError(
            "REAL_TRANSPORT_RESULT_TYPE_UNKNOWN: a real transport must "
            f"return str or RealTransportResult, got {type(result).__name__}")
    if not isinstance(raw, str) or not raw:
        raise ValueError("EMPTY_REAL_LLM_RESPONSE")
    if not request_id:
        request_id = f"derived:{text_sha256(raw)[:16]}"
    return raw, request_id, token_usage, usage_status


def default_logical_call_id(role: str, prompt: str, backend_id: str) -> str:
    """Content-bound logical call id: (role, prompt, backend) -> sha256.

    A repair re-call carries a DIFFERENT prompt (the failure context is
    appended) and therefore a different logical id — identical content is
    exactly what the duplicate-success guard must catch.
    """
    return canonical_sha256({"role": role,
                             "prompt_sha256": text_sha256(prompt),
                             "backend_id": backend_id})


# ---------------------------------------------------------------------------
# journal entries
# ---------------------------------------------------------------------------
class RealCallJournalEntry(CanonicalModel):
    """One append-only journal record (transport or schema outcome)."""

    logical_call_id: str = Field(min_length=1)
    entry_kind: str = Field(min_length=1)
    role: str = Field(min_length=1)
    backend_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    #: API-side request id (or the honest ``derived:<sha16>`` stand-in)
    request_id: str = Field(min_length=1)
    prompt_sha256: str = ""
    response_sha256: str = ""
    token_usage: Dict[str, int] = Field(default_factory=dict)
    token_usage_status: str = TOKEN_USAGE_NOT_PROVIDED
    retry_count: int = Field(default=0, ge=0)
    output_schema_status: str = OUTPUT_SCHEMA_PENDING
    #: bound later by the caller (unknown to a backend); -1/"" = unbound
    window: int = Field(default=-1, ge=-1)
    sequence: int = Field(default=-1, ge=-1)
    artifact_binding: str = ""
    previous_entry_hash: str = ""
    entry_hash: str = ""

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "RealCallJournalEntry":
        if self.entry_kind not in ENTRY_KINDS:
            raise ValueError(f"ILLEGAL_JOURNAL_ENTRY_KIND: {self.entry_kind!r}")
        for field_name in ("prompt_sha256", "response_sha256"):
            value = getattr(self, field_name)
            if value and not is_sha256_hex(value):
                raise ValueError(
                    f"JOURNAL_HASH_NOT_SHA256: {field_name}={value!r}")
        if self.token_usage_status not in TOKEN_USAGE_STATUSES:
            raise ValueError(
                f"ILLEGAL_TOKEN_USAGE_STATUS: {self.token_usage_status!r}")
        if self.output_schema_status not in OUTPUT_SCHEMA_STATUSES:
            raise ValueError(
                f"ILLEGAL_OUTPUT_SCHEMA_STATUS: "
                f"{self.output_schema_status!r}")
        for key, value in self.token_usage.items():
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < 0:
                raise ValueError(
                    f"ILLEGAL_TOKEN_USAGE: {key}={value!r}")
        computed = canonical_sha256(
            {k: v for k, v in self.model_dump().items()
             if k != "entry_hash"})
        if self.entry_hash and self.entry_hash != computed:
            raise ValueError(
                f"CONTENT_HASH_MISMATCH: RealCallJournalEntry carried "
                f"entry_hash={self.entry_hash!r} but its content "
                f"recomputes to {computed!r}")
        object.__setattr__(self, "entry_hash", computed)
        return self

    def bound_copy(self, *, window: int, sequence: int,
                   artifact_binding: str) -> "RealCallJournalEntry":
        """Derived artifact binding the transport entry to a window/envelope.

        The copy recomputes its own entry_hash (it is NOT appended to the
        chain); the journal keeps the unbound transport entry as evidence.
        """
        payload = self.model_dump()
        payload.pop("entry_hash", None)
        payload.update(window=window, sequence=sequence,
                       artifact_binding=artifact_binding)
        return RealCallJournalEntry(**payload)


# ---------------------------------------------------------------------------
# the journal
# ---------------------------------------------------------------------------
class RealCallJournal:
    """Append-only, hash-chained journal of real LLM calls."""

    def __init__(self, *, retry_cap: int = DEFAULT_RETRY_CAP) -> None:
        if retry_cap < 0:
            raise ValueError(f"NEGATIVE_JOURNAL_RETRY_CAP: {retry_cap}")
        self._retry_cap = retry_cap
        self._entries: List[RealCallJournalEntry] = []
        #: logical_call_id -> True once a PARSED outcome exists
        self._completed: Dict[str, bool] = {}
        #: logical_call_id -> latest transport entry (outcome source)
        self._last_transport: Dict[str, RealCallJournalEntry] = {}
        self._chain_head = GENESIS_ENTRY_HASH

    # ------------------------------------------------------------- queries
    @property
    def retry_cap(self) -> int:
        return self._retry_cap

    @property
    def entries(self) -> Tuple[RealCallJournalEntry, ...]:
        return tuple(self._entries)

    @property
    def chain_head(self) -> str:
        return self._chain_head

    def is_completed(self, logical_call_id: str) -> bool:
        return bool(self._completed.get(logical_call_id, False))

    def assert_open(self, logical_call_id: str) -> None:
        """Refuse any further activity under an already-successful id."""
        if self.is_completed(logical_call_id):
            raise DuplicateSuccessfulCall(
                f"DUPLICATE_SUCCESSFUL_CALL: logical_call_id="
                f"{logical_call_id!r} already carries a PARSED outcome; a "
                "successful real call is never repeated")

    # -------------------------------------------------------------- append
    def _append(self, entry: RealCallJournalEntry) -> RealCallJournalEntry:
        self._entries.append(entry)
        self._chain_head = entry.entry_hash
        return entry

    def record_transport(self, *, logical_call_id: str, role: str,
                         backend_id: str, model_id: str, request_id: str,
                         prompt_sha256: str, response_sha256: str,
                         token_usage: Optional[Dict[str, int]] = None,
                         token_usage_status: str = TOKEN_USAGE_NOT_PROVIDED,
                         retry_count: int = 0) -> RealCallJournalEntry:
        """Journal one served transport call (hashes only, never text)."""
        self.assert_open(logical_call_id)
        if retry_count > self._retry_cap:
            raise JournalBlocked(
                f"JOURNAL_RETRY_CAP_EXCEEDED: logical_call_id="
                f"{logical_call_id!r} retry_count={retry_count} exceeds the "
                f"round cap {self._retry_cap} — unbounded retries are "
                "forbidden")
        entry = RealCallJournalEntry(
            logical_call_id=logical_call_id,
            entry_kind=ENTRY_KIND_TRANSPORT,
            role=role, backend_id=backend_id, model_id=model_id,
            request_id=request_id,
            prompt_sha256=prompt_sha256,
            response_sha256=response_sha256,
            token_usage=dict(token_usage or {}),
            token_usage_status=token_usage_status,
            retry_count=retry_count,
            output_schema_status=OUTPUT_SCHEMA_PENDING,
            previous_entry_hash=self._chain_head)
        self._last_transport[logical_call_id] = entry
        return self._append(entry)

    def record_schema_outcome(self, logical_call_id: str, *, status: str,
                              window: int = -1, sequence: int = -1,
                              artifact_binding: str = ""
                              ) -> RealCallJournalEntry:
        """Journal the parse verdict of a previously transported call.

        ``PARSED`` completes the logical id (any later entry under it is
        refused); ``SCHEMA_FAILED`` leaves it open for a bounded repair
        re-call (which, carrying a different prompt, gets a NEW logical id).
        """
        if status not in (OUTPUT_SCHEMA_PARSED, OUTPUT_SCHEMA_FAILED):
            raise ValueError(
                f"ILLEGAL_SCHEMA_OUTCOME_STATUS: {status!r} — an outcome "
                "entry must be PARSED or SCHEMA_FAILED")
        self.assert_open(logical_call_id)
        transport = self._last_transport.get(logical_call_id)
        if transport is None:
            raise JournalBlocked(
                f"JOURNAL_NO_TRANSPORT_FOR_OUTCOME: logical_call_id="
                f"{logical_call_id!r} has no transport entry in this "
                "journal — an outcome cannot be journaled for a call this "
                "journal never saw")
        entry = RealCallJournalEntry(
            logical_call_id=logical_call_id,
            entry_kind=ENTRY_KIND_SCHEMA_OUTCOME,
            role=transport.role, backend_id=transport.backend_id,
            model_id=transport.model_id, request_id=transport.request_id,
            prompt_sha256=transport.prompt_sha256,
            response_sha256=transport.response_sha256,
            token_usage=dict(transport.token_usage),
            token_usage_status=transport.token_usage_status,
            retry_count=transport.retry_count,
            output_schema_status=status,
            window=window, sequence=sequence,
            artifact_binding=artifact_binding,
            previous_entry_hash=self._chain_head)
        if status == OUTPUT_SCHEMA_PARSED:
            self._completed[logical_call_id] = True
        return self._append(entry)

    def dump(self) -> List[dict]:
        return [e.model_dump() for e in self._entries]


__all__ = [
    "ENTRY_KIND_TRANSPORT", "ENTRY_KIND_SCHEMA_OUTCOME", "ENTRY_KINDS",
    "OUTPUT_SCHEMA_PENDING", "OUTPUT_SCHEMA_PARSED", "OUTPUT_SCHEMA_FAILED",
    "OUTPUT_SCHEMA_STATUSES", "TOKEN_USAGE_PROVIDED",
    "TOKEN_USAGE_NOT_PROVIDED", "TOKEN_USAGE_STATUSES", "DEFAULT_RETRY_CAP",
    "GENESIS_ENTRY_HASH", "JournalBlocked", "DuplicateSuccessfulCall",
    "RealTransportResult", "normalize_transport_result",
    "default_logical_call_id", "RealCallJournalEntry", "RealCallJournal",
]
