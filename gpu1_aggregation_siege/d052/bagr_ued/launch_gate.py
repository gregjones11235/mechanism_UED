"""Strong-typed, UNBYPASSABLE LaunchGate (CC3 fix2, task §4-§8).

fix1 carried the launch decision as a plain dict whose single boolean
``training_launch_authorized`` conflated two DIFFERENT concepts:

  * STRUCTURAL batch readiness — every structural condition of the 12+4 /
    16x128=2048 batch plan holds at once;
  * DIRECTOR training authorization — the out-of-package director decision
    (TRAINING_AUTHORIZED) that permits real training at all.

This module replaces that dict with an unambiguous frozen dataclass:

    final_training_launch_authorized =
        structural_batch_ready AND director_training_authorized

and THIS ROUND ``director_training_authorized`` is ALWAYS false (the director
flag is never set inside this package), so even a structurally perfect batch
keeps ``final_training_launch_authorized = false``. A field named
``training_launch_authorized`` expressing both concepts at once is FORBIDDEN.

Hash binding (task §6): the gate carries four content hashes — batch plan,
selected descriptors, guard report, legality report. ``ProposalArchive.commit``
requires a LaunchGate (keyword-required, non-None, isinstance-verified,
version-checked) and re-verifies ALL four hashes against the CURRENT values;
any mismatch -> ARCHIVE_COMMIT_REJECTED. There is no commit path without a
gate: ``archive.commit(..., launch_gate=None)`` as a bypass interface is
FORBIDDEN — the parameter has no default and None fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from d052.bagr_ued import constants as C
from d052.bagr_ued.hashing import canonical_sha256

GATE_VERSION = "bagr_ued.launch_gate.v1"


# ---------------------------------------------------------------------------
# current-state hash helpers — the SINGLE source of truth shared by the gate
# evaluator and the archive commit verifier (so "gate hash == current" is a
# literal recomputation, not a copy).
# ---------------------------------------------------------------------------

def compute_batch_plan_hash(batch_plan) -> str:
    return canonical_sha256(batch_plan.model_dump())


def compute_selected_descriptor_hash(selected_descriptors) -> str:
    return canonical_sha256(
        [d.model_dump() for d in sorted(selected_descriptors,
                                        key=lambda d: d.descriptor_id)])


def compute_guard_report_hash(board_out) -> str:
    return canonical_sha256(dict(
        supervision_guard_status=board_out.supervision_guard_status,
        leakage_guard_status=board_out.leakage_guard_status))


def compute_legality_report_hash(legal_ids: Sequence[str],
                                 rejected: Sequence[dict]) -> str:
    return canonical_sha256(dict(
        legal_ids=sorted(legal_ids),
        rejected=sorted(rejected,
                        key=lambda r: r.get("descriptor_id", ""))))


@dataclass(frozen=True)
class LaunchGate:
    """The unambiguous final batch/launch decision (CC3 fix2 §4).

    Deliberately a frozen dataclass, not a dict: strong type, no default
    construction, version-verifiable. ``final_training_launch_authorized`` is
    the ONLY authorization to treat as "may train" — and it is false this
    round by construction.
    """

    structural_batch_ready: bool
    director_training_authorized: bool
    final_training_launch_authorized: bool
    batch_plan_hash: str
    selected_descriptor_hash: str
    guard_report_hash: str
    legality_report_hash: str
    reasons: Tuple[str, ...] = field(default_factory=tuple)
    gate_version: str = GATE_VERSION

    def __post_init__(self) -> None:
        # the defining contract, enforced at construction so no caller can
        # ever hand out a gate that violates it
        expected = self.structural_batch_ready and \
            self.director_training_authorized
        if self.final_training_launch_authorized != expected:
            raise ValueError(
                "LAUNCH_GATE_CONTRACT_VIOLATED: final_training_launch_"
                "authorized must equal structural_batch_ready AND "
                "director_training_authorized")
        if self.gate_version != GATE_VERSION:
            raise ValueError(
                f"LAUNCH_GATE_VERSION_MISMATCH: {self.gate_version!r} != "
                f"{GATE_VERSION!r}")


def _critic_hard_reject_ids(board_out) -> set:
    """Intervention ids the Critic/Skeptic hard-rejected (task §5: a SELECTED
    proposal carrying a hard-rejected source intervention blocks readiness)."""
    for e in board_out.envelopes:
        if e.role == C.ROLE_CRITIC_SKEPTIC:
            return set(e.parsed_json.get("critic_reject_intervention_ids", []))
    return set()


def evaluate_launch_gate(budget_plan, batch_plan, selected_descriptors,
                         rejected_descriptors, board_out, *,
                         director_training_authorized: bool =
                             C.TRAINING_AUTHORIZED,
                         legal_ids=None) -> LaunchGate:
    """Evaluate the structural batch gate + director authorization.

    ``selected_descriptors`` = the legal descriptors whose ids are the budget
    plan's UED slots; ``rejected_descriptors`` = LegalityGate rejection
    records. CC3 fix2 §15-§16: UNSELECTED illegal proposals are recorded but
    DO NOT block a structurally-satisfied LEGAL batch — only selected-side
    violations block (selected descriptor illegal, selector referencing a
    rejected candidate, fewer than 12 legal candidates, ...).
    """
    reasons: List[str] = []
    selected_ids = list(budget_plan.ued_slots)
    rejected_ids = sorted({r.get("descriptor_id", "")
                           for r in rejected_descriptors})
    legal_ids = sorted(legal_ids) if legal_ids is not None else \
        sorted(d.descriptor_id for d in selected_descriptors)

    # -- budget plan status / shortfall -------------------------------------
    if budget_plan.status != "OK":
        reasons.append(f"budget_plan_status={budget_plan.status}")
    if budget_plan.shortfall_note:
        reasons.append(f"unresolved_shortfall: {budget_plan.shortfall_note}")

    # -- slot / anchor structure --------------------------------------------
    if len(selected_ids) != C.UED_ACTIVE_SLOTS:
        reasons.append(f"selected_ued_slots={len(selected_ids)} "
                       f"!= {C.UED_ACTIVE_SLOTS}")
    if len(set(selected_ids)) != len(selected_ids):
        reasons.append("duplicate_selected_ued_slots_forbidden")
    if list(budget_plan.anchor_slots) != list(C.GLOBAL_CANONICAL_ANCHOR_IDS):
        reasons.append(
            f"canonical_anchor_slots={list(budget_plan.anchor_slots)} "
            f"!= the {C.GLOBAL_CANONICAL_ANCHORS} fixed global anchors")

    # -- batch arithmetic ----------------------------------------------------
    if batch_plan.num_envs != C.NUM_ENVS:
        reasons.append(f"total_envs={batch_plan.num_envs} != {C.NUM_ENVS}")
    if batch_plan.rollout_length != C.ROLLOUT_LENGTH:
        reasons.append(f"rollout_length={batch_plan.rollout_length} "
                       f"!= {C.ROLLOUT_LENGTH}")
    if batch_plan.transitions_per_update != C.TRANSITIONS_PER_UPDATE:
        reasons.append(
            f"transitions_per_update={batch_plan.transitions_per_update} "
            f"!= {C.TRANSITIONS_PER_UPDATE}")

    # -- legality (SELECTED side only — §15/§16 semantics) --------------------
    inter = sorted(set(selected_ids) & set(rejected_ids))
    if inter:
        reasons.append(
            f"selected_descriptor_illegal: selector references rejected "
            f"candidate(s) {inter}")
    missing_legal = sorted(set(selected_ids) - set(legal_ids))
    if missing_legal:
        reasons.append(
            f"selected_descriptor_without_legality_evidence: {missing_legal}")
    if len(legal_ids) < C.UED_ACTIVE_SLOTS and len(inter) == 0 \
            and not missing_legal:
        reasons.append(
            f"legal_candidates={len(legal_ids)} < {C.UED_ACTIVE_SLOTS}")

    # -- selected descriptors consistent with the recorded selection ---------
    sel_desc_ids = sorted(d.descriptor_id for d in selected_descriptors)
    if sel_desc_ids != sorted(set(selected_ids)):
        reasons.append(
            f"selected_descriptors_inconsistent_with_budget_slots: "
            f"descriptors={sel_desc_ids} slots={sorted(set(selected_ids))}")

    # -- critic hard reject on a SELECTED proposal ----------------------------
    critic_rejects = _critic_hard_reject_ids(board_out)
    for d in selected_descriptors:
        src = set(d.provenance.get("source_intervention_ids", []))
        hit = sorted(src & critic_rejects)
        if hit:
            reasons.append(
                f"selected_proposal_critic_hard_reject: "
                f"{d.descriptor_id} <- intervention(s) {hit}")

    # -- unresolved guard violations ------------------------------------------
    if not (board_out.supervision_guard_status == "PASS"
            and board_out.leakage_guard_status == "PASS"):
        reasons.append(
            f"unresolved_guard_violation: supervision="
            f"{board_out.supervision_guard_status} leakage="
            f"{board_out.leakage_guard_status}")

    # -- required provenance / hashes present ---------------------------------
    batch_plan_hash = compute_batch_plan_hash(batch_plan)
    selected_descriptor_hash = compute_selected_descriptor_hash(
        selected_descriptors)
    guard_report_hash = compute_guard_report_hash(board_out)
    legality_report_hash = compute_legality_report_hash(legal_ids,
                                                        rejected_descriptors)
    for name, h in (("batch_plan_hash", batch_plan_hash),
                    ("selected_descriptor_hash", selected_descriptor_hash),
                    ("guard_report_hash", guard_report_hash),
                    ("legality_report_hash", legality_report_hash)):
        if not (isinstance(h, str) and len(h) == 64):
            reasons.append(f"missing_required_provenance_hash: {name}")

    structural = not reasons
    director = bool(director_training_authorized)
    return LaunchGate(
        structural_batch_ready=structural,
        director_training_authorized=director,
        final_training_launch_authorized=structural and director,
        batch_plan_hash=batch_plan_hash,
        selected_descriptor_hash=selected_descriptor_hash,
        guard_report_hash=guard_report_hash,
        legality_report_hash=legality_report_hash,
        reasons=tuple(reasons),
        gate_version=GATE_VERSION)
