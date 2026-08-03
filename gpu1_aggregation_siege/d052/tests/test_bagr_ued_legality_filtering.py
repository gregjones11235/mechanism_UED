"""CC3 audit fix2 §15-§16 — legality rejection semantics (+ §3 cases I/J).

Raw proposals MAY contain illegal candidates: the LegalityGate rejects and
RECORDS them, and only legal candidates enter scoring/selection. The FINAL
structural gate checks ONLY the selected side:

  * all SELECTED descriptors legal,
  * selected_ids ∩ rejected_ids == ∅,
  * selected count == 12.

An UNSELECTED illegal proposal is recorded (reason, no archive, no selector)
but does NOT block a structurally-satisfied LEGAL batch. Blocking happens
only when: a selected descriptor is illegal; fewer than 12 legal candidates
exist; the selector references a rejected candidate; a rejected candidate
would enter the archive; legality evidence/hash is inconsistent.

Cases A-E (§16) + I/J (§3):
  A. 13 proposals (12 legal + 1 illegal UNSELECTED) -> illegal recorded,
     the 12 legal form a structural-ready batch;
  B. 11 legal + 1 illegal -> shortfall, structural=false;
  C. selector tries a rejected candidate -> fail closed;
  D. rejected candidate in the commit set -> archive hash binding rejects;
  E. rejected_ids ∩ selected_ids empty -> proceed (structural true);
  I. (§3) an illegal candidate cannot be selected via a high score;
  J. (§3) the critic penalty softens ranking but NEVER substitutes the
     legality hard gate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.archive import ProposalArchive
from d052.bagr_ued.batch_planner import BatchPlanner
from d052.bagr_ued.budget_allocator import BudgetAllocator, BudgetPlan
from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.launch_gate import (
    LaunchContext,
    compute_selected_descriptor_hash,
    evaluate_launch_context,
    evaluate_launch_gate,
)
from d052.bagr_ued.legality_gate import LegalityGate
from d052.bagr_ued.soft_copeland import EnvironmentScoreBundle, soft_copeland_rank

BATCH = BatchPlanner().plan(8)


def _all_true_context(gate):
    """SYNTHETIC fully-authorized context (CC3 fix3 §1/§3) — unit tests only;
    the package TRAINING_AUTHORIZED backstop still refuses every commit."""
    return LaunchContext(
        structural_batch_ready=True, review_certificate_valid=True,
        provenance_valid=True, guards_passed=True,
        simulator_probe_complete=True, selection_complete=True,
        director_training_authorized=True,
        final_training_launch_authorized=True,
        batch_plan_hash=gate.batch_plan_hash,
        selected_descriptor_hash=gate.selected_descriptor_hash,
        legality_report_hash=gate.legality_report_hash,
        guard_report_hash=gate.guard_report_hash,
        critic_report_hash=gate.critic_report_hash,
        director_authorization_hash=gate.director_authorization_hash,
        clip_batch_hash=canonical_sha256([]),
        reasons=())


def _board():
    critic = SimpleNamespace(
        role=C.ROLE_CRITIC_SKEPTIC,
        parsed_json={"critic_reject_intervention_ids": []})
    return SimpleNamespace(supervision_guard_status="PASS",
                           leakage_guard_status="PASS",
                           envelopes=[critic])


def _legal_descriptor(i):
    return TaskParamsDescriptor(
        descriptor_id=f"tpd:env{i:02d}",
        mock_env_family="threat_distance_family",
        mock_axis_values={"threat_distance_grading": "graded"},
        mock_variant_id=f"var:env{i:02d}",
        mock_variant_kind="single_axis",
        mutation_axes=["threat_distance_grading"],
        provenance={"source_intervention_ids": []})


def _illegal_descriptor():
    # a real LegalityGate rejection the descriptor SCHEMA itself admits:
    # supervision-forbidden content smuggled into provenance (the schema
    # treats provenance as an opaque dict; the gate's supervision scan
    # catches it). Illegal mutation axes / invented fields are refused even
    # earlier — at schema construction — which is the prior fail-closed
    # layer of the same contract.
    return TaskParamsDescriptor(
        descriptor_id="tpd:illegal",
        mock_env_family="threat_distance_family",
        mock_variant_id="var:illegal",
        mock_variant_kind="single_axis",
        mutation_axes=["threat_distance_grading"],
        provenance={"suggested_actions": ["wait_here"]})


def _score_bundle(descriptor_id, *, front=0.5, glob=0.5, penalty=0.0):
    return EnvironmentScoreBundle(
        environment_id=descriptor_id,
        front_regret=front, global_regret=glob, behavioral_gap=0.2,
        learning_progress=0.3, learnability=0.4, diversity=0.5,
        global_retention=0.6, critic_penalty=penalty, alpha_front=0.5)


GATE = LegalityGate()


# ---------------------------------------------------------------------------
# A. 13 proposals (12 legal + 1 illegal unselected) -> structural ready
# ---------------------------------------------------------------------------

def test_case_a_unselected_illegal_recorded_but_legal_12_ready():
    proposals = [_legal_descriptor(i) for i in range(12)] + \
        [_illegal_descriptor()]
    legal, rejected = GATE.screen(proposals)
    assert len(legal) == 12
    assert len(rejected) == 1
    assert rejected[0]["descriptor_id"] == "tpd:illegal"
    assert rejected[0]["code"] == "ILLEGAL_PROPOSAL_SUPERVISION"

    # scoring + selection over the LEGAL set only
    ranking = soft_copeland_rank([_score_bundle(d.descriptor_id)
                                  for d in legal])
    plan = BudgetAllocator().allocate(ranking)
    assert plan.status == "OK" and len(plan.ued_slots) == 12

    selected = [d for d in legal if d.descriptor_id in set(plan.ued_slots)]
    gate = evaluate_launch_gate(plan, BATCH, selected, rejected, _board(),
                                legal_ids=[d.descriptor_id for d in legal])
    # the unselected illegal proposal does NOT block the legal batch
    assert gate.structural_batch_ready is True
    assert gate.reasons == ()
    # ... but stays recorded in the legality hash (evidence, not silence)
    assert len(gate.legality_report_hash) == 64


# ---------------------------------------------------------------------------
# B. 11 legal + 1 illegal -> shortfall, structural false
# ---------------------------------------------------------------------------

def test_case_b_eleven_legal_plus_one_illegal_is_shortfall():
    proposals = [_legal_descriptor(i) for i in range(11)] + \
        [_illegal_descriptor()]
    legal, rejected = GATE.screen(proposals)
    assert len(legal) == 11 and len(rejected) == 1

    ranking = soft_copeland_rank([_score_bundle(d.descriptor_id)
                                  for d in legal])
    plan = BudgetAllocator().allocate(ranking)
    assert plan.status == "INSUFFICIENT"
    assert len(plan.ued_slots) == 11

    selected = [d for d in legal if d.descriptor_id in set(plan.ued_slots)]
    gate = evaluate_launch_gate(plan, BATCH, selected, rejected, _board(),
                                legal_ids=[d.descriptor_id for d in legal])
    assert gate.structural_batch_ready is False
    assert any("selected_ued_slots=11" in r for r in gate.reasons)
    assert any("legal_candidates=11" in r for r in gate.reasons)


# ---------------------------------------------------------------------------
# C. selector references a rejected candidate -> fail closed
# ---------------------------------------------------------------------------

def test_case_c_selector_referencing_rejected_candidate_fails_closed():
    legal = [_legal_descriptor(i) for i in range(12)]
    illegal = _illegal_descriptor()
    _, rejected = GATE.screen([illegal])
    # a forged budget plan that "selected" the rejected candidate
    forged = BudgetPlan(
        ued_slots=[d.descriptor_id for d in legal[:11]] +
                  [illegal.descriptor_id],
        anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
        status="OK")
    gate = evaluate_launch_gate(forged, BATCH, legal, rejected, _board(),
                                legal_ids=[d.descriptor_id for d in legal])
    assert gate.structural_batch_ready is False
    assert any("selected_descriptor_illegal" in r for r in gate.reasons)
    # the missing-legality-evidence check fires too (defense in depth)
    assert any("selected_descriptor_without_legality_evidence" in r
               for r in gate.reasons)


# ---------------------------------------------------------------------------
# D. rejected candidate in the archive commit set -> hash binding rejects
# ---------------------------------------------------------------------------

def test_case_d_rejected_candidate_in_commit_set_rejected_by_hash_binding():
    legal = [_legal_descriptor(i) for i in range(12)]
    illegal = _illegal_descriptor()
    _, rejected = GATE.screen([illegal])
    legal_ids = [d.descriptor_id for d in legal]

    plan = BudgetPlan(ued_slots=legal_ids,
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="OK")
    gate = evaluate_launch_gate(plan, BATCH, legal, rejected, _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids)
    assert gate.structural_batch_ready is True

    # trying to commit the legal set PLUS the rejected candidate: the
    # selected_descriptor_hash no longer matches the gated selection
    commit_set = legal + [illegal]
    assert compute_selected_descriptor_hash(commit_set) != \
        gate.selected_descriptor_hash
    # CC3 fix3 (§3): the commit rides with a synthetic all-true context so
    # the flow reaches the selected-descriptor hash binding, which rejects
    with pytest.raises(AssertionError,
                       match="ARCHIVE_COMMIT_REJECTED.*selected_descriptor"):
        ProposalArchive().commit(commit_set, {}, launch_gate=gate,
                                 launch_context=_all_true_context(gate),
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids,
                                 rejected_descriptors=rejected)


# ---------------------------------------------------------------------------
# E. rejected_ids ∩ selected_ids == ∅ -> proceed
# ---------------------------------------------------------------------------

def test_case_e_disjoint_rejected_and_selected_proceeds():
    legal = [_legal_descriptor(i) for i in range(12)]
    _, rejected = GATE.screen([_illegal_descriptor()])
    legal_ids = [d.descriptor_id for d in legal]
    plan = BudgetPlan(ued_slots=legal_ids,
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="OK")
    assert not (set(legal_ids) &
                {r["descriptor_id"] for r in rejected})
    gate = evaluate_launch_gate(plan, BATCH, legal, rejected, _board(),
                                legal_ids=legal_ids)
    assert gate.structural_batch_ready is True


# ---------------------------------------------------------------------------
# I (§3). an illegal candidate cannot be selected via a high score
# ---------------------------------------------------------------------------

def test_case_i_illegal_candidate_cannot_ride_a_high_score_into_selection():
    legal = [_legal_descriptor(i) for i in range(12)]
    illegal = _illegal_descriptor()
    screened_legal, rejected = GATE.screen(legal + [illegal])
    assert rejected and rejected[0]["descriptor_id"] == "tpd:illegal"

    # even a MAXIMAL score on the illegal candidate is irrelevant: scoring
    # only ever sees the legal set (screening precedes scoring)
    bundles = [_score_bundle(d.descriptor_id) for d in screened_legal]
    ranking = soft_copeland_rank(bundles)
    ranked_ids = {e.environment_id for e in ranking.entries}
    assert "tpd:illegal" not in ranked_ids

    plan = BudgetAllocator().allocate(ranking)
    assert "tpd:illegal" not in set(plan.ued_slots)
    selected = [d for d in screened_legal
                if d.descriptor_id in set(plan.ued_slots)]
    gate = evaluate_launch_gate(plan, BATCH, selected, rejected, _board(),
                                legal_ids=[d.descriptor_id
                                           for d in screened_legal])
    assert gate.structural_batch_ready is True


# ---------------------------------------------------------------------------
# J (§3). critic penalty must NOT substitute the legality hard gate
# ---------------------------------------------------------------------------

def test_case_j_critic_penalty_never_substitutes_legality_gate():
    # a zero-penalty illegal candidate is STILL illegal: the critic penalty
    # is a soft ranking criterion, never a legality verdict
    illegal = _illegal_descriptor()
    _, rejected = GATE.screen([illegal])
    assert rejected     # legality verdict independent of any penalty

    legal = [_legal_descriptor(i) for i in range(12)]
    legal_ids = [d.descriptor_id for d in legal]
    # critic_penalty=0.0 on everything does not launder the rejected id
    forged = BudgetPlan(
        ued_slots=[d.descriptor_id for d in legal[:11]] +
                  [illegal.descriptor_id],
        anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
        status="OK")
    gate = evaluate_launch_gate(forged, BATCH, legal, rejected, _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids)
    assert gate.structural_batch_ready is False
    assert gate.final_training_launch_authorized is False
    # CC3 fix3 (§1/§3): the context assembled from this gate inherits
    # structural=false, so the commit fails closed at the structural check
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(
            legal, {}, launch_gate=gate,
            launch_context=evaluate_launch_context(gate, _board(),
                                                   symbolic_payloads=()),
            batch_plan=BATCH, board_out=_board(),
            legal_ids=legal_ids,
            rejected_descriptors=rejected)


# ---------------------------------------------------------------------------
# end-to-end: the synthetic pipeline records rejections honestly and keeps
# the legal batch structurally ready
# ---------------------------------------------------------------------------

def test_controller_records_rejections_and_filters_them():
    from d052.bagr_ued.controller import BAGRUEdController
    from d052.bagr_ued.synthetic_traces import build_unsafe_rest_raw_rollout
    result = BAGRUEdController().run_dry_run(build_unsafe_rest_raw_rollout())
    d = result.model_dump()
    selected_ids = {s["descriptor_id"] for s in d["descriptors"]}
    rejected_ids = {r["descriptor_id"] for r in d["rejected_descriptors"]}
    assert not (selected_ids & rejected_ids)
    assert set(d["budget_plan"]["ued_slots"]) <= selected_ids
    # the gate's legality hash binds BOTH sides (legal ids + rejections)
    assert len(d["launch_gate"]["legality_report_hash"]) == 64
