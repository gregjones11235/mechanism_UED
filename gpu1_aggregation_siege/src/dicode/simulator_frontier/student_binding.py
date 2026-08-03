"""Student identity binding for archive entries and branch outcomes (R3/R9).

The simulator_frontier core never imports concrete network classes: binding
works on identity strings/hashes supplied by whoever owns the StudentAdapter
(the shared student_adapters package).  Every missing binding raises — an
unbound entry can never be silently treated as bound.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from .archive_schema import FrontierArchiveEntry
from .errors import SimulatorFrontierError
from .memory_modes import MemoryCompatibilityReport, MemoryRestoreRequest, validate_memory_request
from .search_statistics import BranchOutcome


REQUIRED_ENTRY_BINDING_FIELDS = (
    "source_student_identity_hash",
    "source_parameter_hash",
    "source_memory_spec_hash",
    "capture_student_id",
    "discovery_provenance",
)

UNBOUND_STUDENT = "NONE"


def _require(name: str, value: Any) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise SimulatorFrontierError(f"student binding field {name} is missing or empty (never guess)")
    if isinstance(value, str) and value.upper() in {"PENDING", "UNKNOWN", "TODO"}:
        raise SimulatorFrontierError(f"student binding field {name} carries placeholder {value!r}; bind real evidence first")
    return str(value)


def bind_capture_entry(entry: FrontierArchiveEntry, *,
                       student_identity_hash: str,
                       parameter_hash: str,
                       memory_spec_hash: str,
                       capture_student_id: str,
                       discovery_provenance: str) -> FrontierArchiveEntry:
    """Return a copy of ``entry`` with all R3 binding fields set (fail-closed)."""
    return dataclasses.replace(
        entry,
        source_student_identity_hash=_require("source_student_identity_hash", student_identity_hash),
        source_parameter_hash=_require("source_parameter_hash", parameter_hash),
        source_memory_spec_hash=_require("source_memory_spec_hash", memory_spec_hash),
        capture_student_id=_require("capture_student_id", capture_student_id),
        discovery_provenance=_require("discovery_provenance", discovery_provenance),
    )


def assert_entry_bound(entry: FrontierArchiveEntry) -> None:
    """Raise if any binding field is empty/unbound."""
    for name in REQUIRED_ENTRY_BINDING_FIELDS:
        _require(name, getattr(entry, name))


def bind_branch_outcome(outcome: BranchOutcome, *,
                        capture_student_id: str,
                        search_student_id: str,
                        train_student_id: str,
                        memory_compatibility_status: str) -> BranchOutcome:
    """Bind R9 same-Student tracking fields; cross-policy search stays explicit.

    ``cross_policy_search`` is derived, never taken on faith: it is True iff
    the three Student ids are not all identical.
    """
    cap = _require("capture_student_id", capture_student_id)
    sea = _require("search_student_id", search_student_id)
    trn = _require("train_student_id", train_student_id)
    status = _require("memory_compatibility_status", memory_compatibility_status)
    return dataclasses.replace(
        outcome,
        capture_student_id=cap,
        search_student_id=sea,
        train_student_id=trn,
        cross_policy_search=not (cap == sea == trn),
        memory_compatibility_status=status,
    )


def assert_outcome_bound(outcome: BranchOutcome) -> None:
    """Raise if the outcome's Student tracking fields are unbound."""
    for name in ("capture_student_id", "search_student_id", "train_student_id"):
        value = getattr(outcome, name)
        if value is None or not str(value).strip() or str(value) == UNBOUND_STUDENT:
            raise SimulatorFrontierError(f"branch outcome field {name} is unbound")
    _require("memory_compatibility_status", outcome.memory_compatibility_status)
    if outcome.memory_compatibility_status == "UNSPECIFIED":
        raise SimulatorFrontierError("memory_compatibility_status must be resolved before use")


def check_bound_entry_memory_request(entry: FrontierArchiveEntry,
                                     request: MemoryRestoreRequest) -> MemoryCompatibilityReport:
    """Bridge to memory_modes: a bound entry constrains the restore request.

    The request's checkpoint id must match the entry's source checkpoint and
    the request must be internally valid; zero memory is only a diagnostic
    mode and is reported as such by validate_memory_request.
    """
    assert_entry_bound(entry)
    return validate_memory_request(request, checkpoint_id=entry.source_checkpoint_id)


def identity_mapping_hash_fields(identity: Mapping[str, Any]) -> dict:
    """Extract the three hash fields expected by binding from an identity mapping."""
    out = {}
    for key in ("identity_hash", "params_sha256", "memory_spec_hash"):
        value = identity.get(key, "")
        out[key] = _require(key, value)
    return out
