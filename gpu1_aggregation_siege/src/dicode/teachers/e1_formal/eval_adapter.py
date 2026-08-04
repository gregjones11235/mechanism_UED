"""Candidate-evaluation adapter registry and immutable dual-probe
results (C15, supervisor REQUEST_CHANGES fix).

The first C15 iteration exposed ``record_dual_probe_attestation``, a
public method accepting ANY caller-shaped mapping — any caller could
mint a fake attestation with an arbitrary ``adapter_id`` and then
certify REUSE with it. That seam is GONE. Minting is now bound to:

* an INTERNAL ``CandidateEvalAdapterRegistry``: adapters enter only
  through a fail-closed registration of a signed adapter spec
  (adapter id/version/hash + the pinned dual-probe capability);
* an IMMUTABLE ``DualProbeResult`` frozen dataclass: the only object
  the teacher accepts as probe evidence. It carries the full evidence
  chain — Student/Reference candidate ids, Student/Reference
  CHECKPOINT hashes, probe ids + sha256 hashes, the review window id
  and hash, the ordered candidate-set hash, and the episode reset
  protocol id + hash.

Issuance happens ONLY inside the registry
(``issue_dual_probe_result``, keyword-only scalar arguments — NO
mapping parameter exists anywhere in the chain) and ONLY for a
REGISTERED adapter. A ``DualProbeResult`` constructed directly (or
mutated via ``dataclasses.replace``) is never a member of the issued
set, so the teacher refuses it as an unknown/forged result. Even a
mapping whose fields are ALL valid is rejected on sight: the teacher
consumes instances, never mappings.

Pure stdlib; no file/network I/O, no jax/craftax, no training.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, List, Mapping, Tuple

from .schemas import E1SchemaError

# fail-closed codes (greppable)
EVAL_ADAPTER_BAD_TYPE = "EVAL_ADAPTER_BAD_TYPE"
EVAL_ADAPTER_MISSING_FIELD = "EVAL_ADAPTER_MISSING_FIELD"
EVAL_ADAPTER_UNKNOWN_FIELD = "EVAL_ADAPTER_UNKNOWN_FIELD"
EVAL_ADAPTER_MISMATCH = "EVAL_ADAPTER_MISMATCH"
#: issuance cited an adapter that was never registered (fake adapter)
EVAL_ADAPTER_UNKNOWN = "EVAL_ADAPTER_UNKNOWN"

#: the ONLY capability that may issue dual-probe results; any other
#: capability string is refused at registration (pinned, not guessed)
CANDIDATE_EVAL_ADAPTER_CAPABILITY = (
    "candidate_evaluation_dual_probe_v1"
)

_ADAPTER_SPEC_FIELDS = (
    "adapter_id",
    "adapter_version",
    "adapter_hash",
    "capability",
)

#: immutable dual-probe result fields — the full evidence chain
_RESULT_FIELDS = (
    "adapter_id",
    "student_candidate_id",
    "student_checkpoint_hash",
    "student_probe_id",
    "student_probe_hash",
    "reference_candidate_id",
    "reference_checkpoint_hash",
    "reference_probe_id",
    "reference_probe_hash",
    "window_id",
    "window_hash",
    "candidate_set_hash",
    "episode_reset_protocol_id",
    "episode_reset_protocol_hash",
)

_RESULT_HASH_FIELDS = (
    "student_checkpoint_hash",
    "student_probe_hash",
    "reference_checkpoint_hash",
    "reference_probe_hash",
    "window_hash",
    "candidate_set_hash",
    "episode_reset_protocol_hash",
)

_RESULT_ID_FIELDS = tuple(
    name for name in _RESULT_FIELDS if name not in _RESULT_HASH_FIELDS
)

_HEX_DIGITS = frozenset("0123456789abcdef")

#: placeholder/wildcard values are guesses and are refused everywhere
_PLACEHOLDER_VALUES = frozenset(
    {"todo", "latest", "auto", "tbd", "unknown", "<fill-me>", "${"}
)


class CandidateEvalAdapterError(E1SchemaError):
    """Fail-closed adapter-registry violation; ``code`` is greppable."""


def _is_sha256_hex(value: Any) -> bool:
    """True iff ``value`` is a 64-char lowercase sha256 hex string."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX_DIGITS for c in value)
    )


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str):
        raise CandidateEvalAdapterError(
            EVAL_ADAPTER_BAD_TYPE,
            f"{ctx}: {name} must be a str, got {type(value).__name__}",
        )
    if not value.strip():
        raise CandidateEvalAdapterError(
            EVAL_ADAPTER_MISSING_FIELD,
            f"{ctx}: {name} must be a non-empty string",
        )
    return value


def _refuse_placeholder(value: str, name: str, ctx: str) -> None:
    lowered = value.strip().lower()
    for bad in _PLACEHOLDER_VALUES:
        if lowered == bad or lowered.startswith(bad):
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_MISMATCH,
                f"{ctx}: {name} looks guessed/placeholder "
                f"({value!r}) — adapter identities are never guessed",
            )


@dataclass(frozen=True)
class RegisteredAdapter:
    """A registered candidate-evaluation adapter identity (immutable)."""

    adapter_id: str
    adapter_version: str
    adapter_hash: str
    capability: str


@dataclass(frozen=True)
class DualProbeResult:
    """IMMUTABLE result of one adapter dual-probe evaluation.

    The ONLY object the teacher accepts as probe evidence (C15 fix).
    Every field is mandatory; construction validates fail-closed, so
    even direct construction with malformed values raises. Instances
    still need registry ISSUANCE membership to be consumed — valid
    fields alone never suffice.
    """

    adapter_id: str
    student_candidate_id: str
    student_checkpoint_hash: str
    student_probe_id: str
    student_probe_hash: str
    reference_candidate_id: str
    reference_checkpoint_hash: str
    reference_probe_id: str
    reference_probe_hash: str
    window_id: str
    window_hash: str
    candidate_set_hash: str
    episode_reset_protocol_id: str
    episode_reset_protocol_hash: str

    def __post_init__(self) -> None:
        ctx = "candidate_eval.DualProbeResult"
        assert tuple(f.name for f in dataclass_fields(self)) == (
            _RESULT_FIELDS
        )
        for name in _RESULT_FIELDS:
            _require_non_empty_str(getattr(self, name), name, ctx)
        for name in _RESULT_ID_FIELDS:
            _refuse_placeholder(getattr(self, name), name, ctx)
        for name in _RESULT_HASH_FIELDS:
            value = getattr(self, name)
            if not _is_sha256_hex(value):
                raise CandidateEvalAdapterError(
                    EVAL_ADAPTER_BAD_TYPE,
                    f"{ctx}: {name} must be lowercase sha256 hex "
                    f"(64 chars), got {value!r}",
                )
        if self.student_probe_id == self.reference_probe_id:
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_MISMATCH,
                f"{ctx}: Student and Reference probes must be "
                "DISTINCT (identical probe ids — a swapped or "
                "degenerate probe pair)",
            )
        if self.student_probe_hash == self.reference_probe_hash:
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_MISMATCH,
                f"{ctx}: Student and Reference probes must be "
                "DISTINCT (identical probe hashes — a swapped or "
                "degenerate probe pair)",
            )


class CandidateEvalAdapterRegistry:
    """Internal registry: registered adapters + issued results.

    The ONLY mint path for dual-probe evidence. Registration consumes
    a signed adapter spec fail-closed; issuance takes keyword-only
    scalar arguments (NO mapping is ever accepted), requires a
    REGISTERED adapter, and records every issued immutable result.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, RegisteredAdapter] = {}
        self._issued: List[DualProbeResult] = []

    # ------------------------------------------------------------------
    # adapter registration (signed-spec seam, fail-closed)
    # ------------------------------------------------------------------
    def register_adapter(self, spec: Any) -> RegisteredAdapter:
        """Register ONE candidate-evaluation adapter (fail-closed).

        Requires the exact spec field set, non-empty non-placeholder
        id/version, a sha256-hex adapter hash, and the pinned
        dual-probe capability. Re-registering an id with a DIFFERENT
        spec is a conflict and is refused; an identical re-registration
        is idempotent.
        """
        ctx = "candidate_eval.register_adapter"
        if not isinstance(spec, Mapping):
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_BAD_TYPE,
                f"{ctx}: adapter spec must be a mapping, got "
                f"{type(spec).__name__}",
            )
        unknown = sorted(
            k for k in spec if k not in _ADAPTER_SPEC_FIELDS
        )
        if unknown:
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_UNKNOWN_FIELD,
                f"{ctx}: unknown adapter spec field(s) {unknown}",
            )
        for name in _ADAPTER_SPEC_FIELDS:
            if name not in spec:
                raise CandidateEvalAdapterError(
                    EVAL_ADAPTER_MISSING_FIELD,
                    f"{ctx}: adapter spec missing field {name!r}",
                )
        adapter_id = _require_non_empty_str(
            spec["adapter_id"], "adapter_id", ctx
        )
        _refuse_placeholder(adapter_id, "adapter_id", ctx)
        adapter_version = _require_non_empty_str(
            spec["adapter_version"], "adapter_version", ctx
        )
        _refuse_placeholder(adapter_version, "adapter_version", ctx)
        adapter_hash = spec["adapter_hash"]
        if not _is_sha256_hex(adapter_hash):
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_BAD_TYPE,
                f"{ctx}: adapter_hash must be lowercase sha256 hex "
                f"(64 chars), got {adapter_hash!r}",
            )
        capability = spec["capability"]
        if capability != CANDIDATE_EVAL_ADAPTER_CAPABILITY:
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_MISMATCH,
                f"{ctx}: capability must be exactly "
                f"{CANDIDATE_EVAL_ADAPTER_CAPABILITY!r}, got "
                f"{capability!r}",
            )
        cleaned = RegisteredAdapter(
            adapter_id=adapter_id.strip(),
            adapter_version=adapter_version.strip(),
            adapter_hash=adapter_hash,
            capability=capability,
        )
        existing = self._adapters.get(cleaned.adapter_id)
        if existing is not None and existing != cleaned:
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_MISMATCH,
                f"{ctx}: adapter id {cleaned.adapter_id!r} is already "
                "registered with a DIFFERENT spec (conflict)",
            )
        self._adapters[cleaned.adapter_id] = cleaned
        return cleaned

    @property
    def registered_adapter_ids(self) -> Tuple[str, ...]:
        """Sorted registered adapter ids (audit)."""
        return tuple(sorted(self._adapters))

    @property
    def issued_results(self) -> Tuple[DualProbeResult, ...]:
        """Every issued immutable result, in issue order (audit)."""
        return tuple(self._issued)

    # ------------------------------------------------------------------
    # issuance — the ONLY mint path (keyword-only scalars, no mapping)
    # ------------------------------------------------------------------
    def issue_dual_probe_result(self, **kwargs: Any) -> DualProbeResult:
        """Issue ONE immutable dual-probe result for a REGISTERED
        adapter.

        Keyword-only scalar arguments (exactly ``_RESULT_FIELDS``);
        there is NO mapping parameter anywhere in the mint chain. The
        adapter must already be registered — a fake/unknown adapter id
        fails closed BEFORE any object exists. Construction then
        validates every field fail-closed. Exact duplicates are issued
        once.
        """
        ctx = "candidate_eval.issue_dual_probe_result"
        unknown = sorted(set(kwargs) - set(_RESULT_FIELDS))
        if unknown:
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_UNKNOWN_FIELD,
                f"{ctx}: unknown result field(s) {unknown}",
            )
        for name in _RESULT_FIELDS:
            if name not in kwargs:
                raise CandidateEvalAdapterError(
                    EVAL_ADAPTER_MISSING_FIELD,
                    f"{ctx}: result missing field {name!r}",
                )
        adapter_id = kwargs["adapter_id"]
        if not isinstance(adapter_id, str) or adapter_id not in (
            self._adapters
        ):
            raise CandidateEvalAdapterError(
                EVAL_ADAPTER_UNKNOWN,
                f"{ctx}: adapter {adapter_id!r} is NOT registered — a "
                "fake/unknown adapter can never issue dual-probe "
                "results",
            )
        result = DualProbeResult(**kwargs)  # fail-closed shape check
        if result not in self._issued:
            self._issued.append(result)
        return result

    # ------------------------------------------------------------------
    # issuance membership — direct construction never counts
    # ------------------------------------------------------------------
    def lookup_result(self, result: Any) -> bool:
        """True iff ``result`` is an IMMUTABLE object this registry
        ITSELF issued (direct construction / mutation never counts)."""
        return isinstance(result, DualProbeResult) and result in (
            self._issued
        )
