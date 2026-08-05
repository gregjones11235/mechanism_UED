"""Dual-Student selection, carry semantics and cross-student isolation (E3-DS).

The E3 consumer can mount ONE of two frozen primary Students per run:

    rmt16_persistent_98304  -> PERSISTENT_RMT16_ORIGINAL_VTRACE_98304
                              memory.mode = PERSISTENT  (carry_mode = persistent)
    rmt16_reset128_98304    -> RESET128_RMT16_ORIGINAL_VTRACE_98304
                              memory.mode = RESET128    (carry_mode = reset128)

They share the same RMT16 network family and the same memory TREE, but their
carry semantics differ: PERSISTENT keeps ``rmt.mem_tokens`` across the
128-step segment boundary while RESET128 clears them to zero at every
boundary.  They are TWO INDEPENDENT experiment starting points — never a
Student/Reference pair — so one E3 run must bind exactly ONE selected
student everywhere (capture == search == train == director-selected), and a
Frontier state captured by one arm can never be handed to the other's
training (and its memory can never be restored under the other's carry rule).

This module freezes the allowed set (fail closed for unknown candidates),
classifies the carry semantics and provides the cross-student guards.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .errors import InvalidEvidenceError, ProvenanceViolationError

DUAL_STUDENT_VERSION = "e3-dual-student/v1"

# Director-frozen set of admissible primary (training) Student candidates.
ALLOWED_PRIMARY_STUDENT_IDS = frozenset({
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "RESET128_RMT16_ORIGINAL_VTRACE_98304",
})

# Frozen profile-name -> candidate_id map (the shared registry supports both).
PROFILE_NAME_TO_CANDIDATE_ID = {
    "rmt16_persistent_98304": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "rmt16_reset128_98304": "RESET128_RMT16_ORIGINAL_VTRACE_98304",
}

# Frozen candidate_id -> memory carry mode (from the frozen CC4 contract).
CANDIDATE_CARRY_MODE = {
    "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304": "PERSISTENT",
    "RESET128_RMT16_ORIGINAL_VTRACE_98304": "RESET128",
}

# Memory restore mode values carried by the profiles.
PROFILE_MEMORY_MODE_TO_CANDIDATE = {
    "PERSISTENT": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "RESET128": "RESET128_RMT16_ORIGINAL_VTRACE_98304",
}

# Errors required by the director handoff (E3-DS section 3).
E3_FRONTIER_STUDENT_IDENTITY_MISMATCH = "E3_FRONTIER_STUDENT_IDENTITY_MISMATCH"
E3_FRONTIER_MEMORY_SPEC_MISMATCH = "E3_FRONTIER_MEMORY_SPEC_MISMATCH"
E3_FRONTIER_CARRY_MODE_MISMATCH = "E3_FRONTIER_CARRY_MODE_MISMATCH"

# Read-only mount vs training runtime readiness (honest, never self-upgraded).
E3_STUDENT_READ_ONLY_MOUNT_READY = False  # set true when the selected adapter
                                          # mounts read-only on a check-only run
E3_STUDENT_TRAINING_RUNTIME_READY = False  # only when a CanonicalDiCodeOneUpdateRuntime
                                           # is bound for the SELECTED candidate
E3_REAL_SMOKE_READY = False


def validate_primary_student_candidate(candidate_id: Any) -> str:
    """Fail closed on any candidate that is not in the director-frozen set."""
    text = str(candidate_id)
    if text not in ALLOWED_PRIMARY_STUDENT_IDS:
        raise InvalidEvidenceError(
            f"unknown primary Student candidate {text!r}: the director-frozen "
            f"allowed set is {sorted(ALLOWED_PRIMARY_STUDENT_IDS)} (fail closed; "
            "no defaulting to the first candidate)")
    return text


def candidate_from_profile_name(profile_name: Any) -> str:
    """The frozen candidate id for a profile name (fail closed)."""
    name = str(profile_name)
    if name not in PROFILE_NAME_TO_CANDIDATE_ID:
        raise InvalidEvidenceError(
            f"profile name {name!r} is not a frozen dual-student profile; "
            f"allowed: {sorted(PROFILE_NAME_TO_CANDIDATE_ID)}")
    return PROFILE_NAME_TO_CANDIDATE_ID[name]


def carry_mode_for_candidate(candidate_id: Any) -> str:
    """The frozen memory carry mode for the selected candidate."""
    candidate = validate_primary_student_candidate(candidate_id)
    return CANDIDATE_CARRY_MODE[candidate]


def memory_mode_for_candidate(candidate_id: Any) -> str:
    """The frozen profile memory.mode (PERSISTENT or RESET128)."""
    candidate = validate_primary_student_candidate(candidate_id)
    mode = CANDIDATE_CARRY_MODE[candidate]
    if mode not in PROFILE_MEMORY_MODE_TO_CANDIDATE:
        raise InvalidEvidenceError(
            f"unresolvable memory mode for candidate {candidate!r}")
    return mode


def assert_same_run_student(*, selected_candidate_id: Any,
                            capture_student_id: Any,
                            search_student_id: Any,
                            train_student_id: Any) -> None:
    """One E3 run binds EXACTLY one selected Student everywhere.

    capture == search == train == director-selected candidate.  The
    Reference is never one of these ids (it is mounted separately and marked
    as the Reference source).
    """
    selected = validate_primary_student_candidate(selected_candidate_id)
    ids = {
        "capture_student_id": str(capture_student_id),
        "search_student_id": str(search_student_id),
        "train_student_id": str(train_student_id),
    }
    for label, value in ids.items():
        if value != selected:
            raise ProvenanceViolationError(
                f"{E3_FRONTIER_STUDENT_IDENTITY_MISMATCH}: {label} {value!r} "
                f"!= selected candidate {selected!r} (one run, one Student; "
                "Reference is mounted separately)")


def assert_memory_binding_for_student(*, candidate_id: Any,
                                      memory_mode: Any,
                                      carry_mode: Any,
                                      memory_spec_hash: Any,
                                      expected_memory_spec_hash: Any) -> None:
    """The memory surface must belong to the selected Student's carry semantics.

    Rejects: a memory spec that is not the selected Student's own
    (E3_FRONTIER_MEMORY_SPEC_MISMATCH) and a carry mode that contradicts the
    selected candidate (E3_FRONTIER_CARRY_MODE_MISMATCH) — Persistent memory
    can never restore a Reset128 state and vice versa.
    """
    candidate = validate_primary_student_candidate(candidate_id)
    expected_mode = memory_mode_for_candidate(candidate)
    if str(memory_mode) != expected_mode:
        raise ProvenanceViolationError(
            f"{E3_FRONTIER_CARRY_MODE_MISMATCH}: candidate {candidate} requires "
            f"memory mode {expected_mode}, got {memory_mode!r} (Persistent memory "
            "never restores a Reset128 state and vice versa; fail closed)")
    expected_carry = carry_mode_for_candidate(candidate)
    if str(carry_mode) != expected_carry:
        raise ProvenanceViolationError(
            f"{E3_FRONTIER_CARRY_MODE_MISMATCH}: candidate {candidate} requires "
            f"carry mode {expected_carry}, got {carry_mode!r}")
    if str(memory_spec_hash) != str(expected_memory_spec_hash):
        raise ProvenanceViolationError(
            f"{E3_FRONTIER_MEMORY_SPEC_MISMATCH}: memory spec hash "
            f"{str(memory_spec_hash)[:16]}… != the selected Student's own "
            f"{str(expected_memory_spec_hash)[:16]}… (cross-policy memory "
            "rejected; fail closed)")


def memory_carry_rule(memory_mode: Any) -> str:
    """Classify the carry rule the selected Student must execute.

    * PERSISTENT  -> "PERSISTENT_CARRY": ``rmt.mem_tokens`` are PRESERVED
      across the 128-step segment boundary (the segment counter keeps
      progressing);
    * RESET128    -> "RESET128_CARRY": ``rmt.mem_tokens`` are CLEARED to zero
      at every 128-step segment boundary (a real segment reset).

    The two rules differ by construction — the shared memory TREE shape is
    NOT the semantics.
    """
    if str(memory_mode) == "PERSISTENT":
        return "PERSISTENT_CARRY"
    if str(memory_mode) == "RESET128":
        return "RESET128_CARRY"
    raise InvalidEvidenceError(
        f"unknown memory mode {memory_mode!r}; only PERSISTENT and RESET128 "
        "are admissible dual-student carry modes")


def carry_semantics_snapshot(candidate_id: Any) -> dict[str, Any]:
    """Honest report projection for the selected candidate."""
    candidate = validate_primary_student_candidate(candidate_id)
    mode = memory_mode_for_candidate(candidate)
    return {
        "candidate_id": candidate,
        "memory_mode": mode,
        "carry_mode": CANDIDATE_CARRY_MODE[candidate],
        "carry_rule": memory_carry_rule(mode),
        "network_family": "RMT16",
        "allowed_set": sorted(ALLOWED_PRIMARY_STUDENT_IDS),
    }


def assert_reference_is_not_a_primary_student(reference_candidate_id: Any) -> None:
    """The Reference is never one of the two frozen primary Students."""
    text = str(reference_candidate_id)
    if text in ALLOWED_PRIMARY_STUDENT_IDS:
        raise ProvenanceViolationError(
            f"reference candidate {text!r} is a primary Student id — the two "
            "top Students are two independent experiment starting points, NOT "
            "a Student/Reference pair (fail closed)")
