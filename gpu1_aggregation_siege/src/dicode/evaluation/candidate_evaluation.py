"""Candidate evaluation seam (C11): Student/Reference dual-probe
evaluation gated by the G1 ReferenceIdentityContract.

Pipeline stage 6 of the E1 formal direction. Gate order is FIXED and
fail-closed (plan D5 degradation chain)::

    G1 reference contract gate   (unfrozen/absent => BLOCKED)
      => StudentAdapter/state gate (no adapter/state => SKIPPED)
      => config enable gate        (disabled => DISABLED)
      => real dual probes (requires the CC4 shared StudentAdapter;
         NOT implemented this round and never stubbed: reaching this
         point raises NotImplementedError rather than fabricating
         results)

Nothing in the blocked/skipped path ever stamps the
``CANDIDATE_EVALUATION`` provenance label — a stamp belongs only to
evaluations that actually ran. This module performs NO file I/O, NO
network I/O and NO rollout of any kind this round; it is pure gate
logic plus an honest unimplemented-real-probe marker.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from dicode.teachers.e1_formal.reference_contract import (
    ReferenceIdentityContract,
)

# fail-closed seam codes (greppable)
EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN = (
    "EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN"
)
EVAL_BLOCKED_REFERENCE_CONTRACT_BAD_TYPE = (
    "EVAL_BLOCKED_REFERENCE_CONTRACT_BAD_TYPE"
)
EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER = "EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER"
EVAL_DISABLED_BY_CONFIG = "EVAL_DISABLED_BY_CONFIG"
EVAL_BAD_CANDIDATE_SET = "EVAL_BAD_CANDIDATE_SET"

#: provenance stamp for evaluations that ACTUALLY ran (never stamped
#: on blocked/skipped results)
CANDIDATE_EVALUATION_PROVENANCE = "CANDIDATE_EVALUATION"


class CandidateEvaluationError(Exception):
    """Fail-closed seam violation; ``code`` is greppable."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _blocked(
    status: str,
    reason: str,
    candidate_task_ids: Tuple[str, ...],
    gates_checked: Sequence[str],
) -> Dict[str, Any]:
    """A blocked/skipped result: evaluated=False, empty results, and
    NO provenance stamp (nothing was evaluated)."""
    return {
        "status": status,
        "evaluated": False,
        "results": (),
        "reason": reason,
        "candidate_task_ids": tuple(candidate_task_ids),
        "gates_checked": list(gates_checked),
    }


def evaluate_candidates_with_reference(
    config: Any,
    rng: Any,
    candidate_task_ids: Any,
    archive: Any,
    embedding_model: Any,
    student_train_state: Any,
    reference_train_state: Any,
    flags: Any,
    reference_contract: Optional[ReferenceIdentityContract],
) -> Dict[str, Any]:
    """Evaluate candidate tasks with the pinned Student + Reference.

    THIS ROUND the function is an honest gate: every reachable path
    returns a blocked/skipped result (contract unfrozen, no CC4
    adapter, seam disabled). The real dual-probe rollout path raises
    ``NotImplementedError`` — it is deliberately NOT stubbed, because
    a stub could be mistaken for real evidence. ``rng``, ``archive``
    and ``embedding_model`` are consumed only by the (future) real
    path and are read by nothing this round.
    """
    del rng, archive, embedding_model  # consumed only by the real path
    ctx = "evaluation.candidate_evaluation"

    # ---- fail-closed input validation --------------------------------
    if not isinstance(candidate_task_ids, (list, tuple)):
        raise CandidateEvaluationError(
            EVAL_BAD_CANDIDATE_SET,
            f"{ctx}: candidate_task_ids must be a sequence, got "
            f"{type(candidate_task_ids).__name__}",
        )
    cleaned_ids = []
    for i, task_id in enumerate(candidate_task_ids):
        if not isinstance(task_id, str) or not task_id.strip():
            raise CandidateEvaluationError(
                EVAL_BAD_CANDIDATE_SET,
                f"{ctx}: candidate_task_ids[{i}] must be a non-empty "
                f"str, got {task_id!r}",
            )
        cleaned_ids.append(task_id.strip())
    if len(set(cleaned_ids)) != len(cleaned_ids):
        raise CandidateEvaluationError(
            EVAL_BAD_CANDIDATE_SET,
            f"{ctx}: duplicate candidate task id in {cleaned_ids}",
        )
    cleaned = tuple(cleaned_ids)
    gates: list = []

    # ---- GATE 1 (G1): reference identity contract FIRST --------------
    gates.append("G1_reference_contract")
    if reference_contract is None:
        return _blocked(
            EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN,
            "G1 ReferenceIdentityContract is not frozen (no contract "
            "instance exists); the supervisor must freeze the "
            "Reference identity before any evaluation seam may run. "
            "E1 never guesses a Reference.",
            cleaned,
            gates,
        )
    if not isinstance(reference_contract, ReferenceIdentityContract):
        return _blocked(
            EVAL_BLOCKED_REFERENCE_CONTRACT_BAD_TYPE,
            f"{ctx}: reference_contract must be a consumed "
            "ReferenceIdentityContract (frozen by construction), got "
            f"{type(reference_contract).__name__}",
            cleaned,
            gates,
        )

    # ---- GATE 2: the CC4 shared StudentAdapter / train states --------
    gates.append("student_adapter_state")
    if student_train_state is None or reference_train_state is None:
        return _blocked(
            EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER,
            "no CC4 StudentAdapter-bound train state for Student "
            "and/or Reference; the seam is skipped (D5). E1 builds no "
            "loader/registry of its own.",
            cleaned,
            gates,
        )
    if flags is None or not hasattr(flags, "real_student_reference_eval"):
        return _blocked(
            EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER,
            f"{ctx}: flags must be the E1Flags contract consumer "
            "(real_student_reference_eval attribute required)",
            cleaned,
            gates,
        )

    # ---- GATE 3: the config enable knob (conf/teacher/e1_formal.yaml)
    gates.append("config_enable")
    enabled = False
    if config is not None and hasattr(config, "get"):
        candidate_eval = config.get("candidate_eval")
        if isinstance(candidate_eval, dict):
            enabled = candidate_eval.get("enabled") is True
    if not enabled:
        return _blocked(
            EVAL_DISABLED_BY_CONFIG,
            "candidate_eval.enabled is not exactly true in the "
            "teacher config; the seam stays fail-closed regardless "
            "of the other gates.",
            cleaned,
            gates,
        )

    # ---- real dual-probe path (CC4 round; NEVER stubbed) -------------
    raise NotImplementedError(
        f"{ctx}: real dual-probe evaluation requires the CC4 shared "
        "StudentAdapter + frozen Reference checkpoint (reduced-size "
        "dual probes, Wilson CI, dual scoring, "
        f"{CANDIDATE_EVALUATION_PROVENANCE} stamp). None of that "
        "exists this round and no substitute is provided: this "
        "branch is reached only when G1 is frozen, adapter states "
        "exist and the seam is enabled — all false this round."
    )
