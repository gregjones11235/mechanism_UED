"""CC3 audit fix2 §4-§8 — strong-typed, UNBYPASSABLE LaunchGate tests.

Contract-driven rewrite of the fix1 dict-gate tests (documented in
reports/behavior_aware_bagr_ued_v2_fix2): the launch decision is now a
frozen dataclass LaunchGate with THREE unambiguous booleans —

    structural_batch_ready / director_training_authorized /
    final_training_launch_authorized (= the first two ANDed)

— plus four hash bindings (batch plan / selected descriptors / guard report /
legality report) that archive.commit re-verifies against the current state.

Cases A-J (§8):
  A. 12+4 structural complete, director unauthorized -> structural=true,
     director=false, final=false, archive commit REJECTED;
  B. 12+4 + director true (SYNTHETIC unit test only — no real training; the
     package flag C.TRAINING_AUTHORIZED stays false) -> final=true, gate hash
     verification passes (commit then meets the package authorization
     backstop, which is the correct two-layer separation);
  C. 11+4 -> structural=false, final=false;
  D. commit without a gate -> TypeError fail closed;
  E. commit with launch_gate=None -> fail closed;
  F. refresh(dry_run=False, launch_gate=None) -> immediate fail closed;
  G. gate hash vs batch plan mismatch -> fail closed;
  H. gate from another descriptor batch -> fail closed;
  I. gate passed but one SELECTED descriptor illegal -> fail closed
     (structural gate refuses it in the first place);
  J. dry_run=true, no gate -> diagnostic plan allowed, archive untouched,
     training_authorized=false.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.archive import ProposalArchive
from d052.bagr_ued.batch_planner import BatchPlanner
from d052.bagr_ued.budget_allocator import BudgetPlan
from d052.bagr_ued.controller import BAGRUEdController
from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.launch_gate import (
    CONTEXT_VERSION,
    GATE_VERSION,
    LaunchContext,
    LaunchGate,
    compute_batch_plan_hash,
    compute_selected_descriptor_hash,
    evaluate_launch_context,
    evaluate_launch_gate,
)
from d052.bagr_ued.synthetic_traces import build_unsafe_rest_raw_rollout


def _context(gate, board=None):
    """The LaunchContext assembled from the SAME gate (dry-run conditions)."""
    return evaluate_launch_context(gate, board or _board(),
                                   symbolic_payloads=())


def _all_true_context(gate):
    """SYNTHETIC fully-authorized context (unit tests ONLY — the package
    flags stay false; the archive backstop still refuses the commit)."""
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


def _ued_ids(n):
    return [f"CF_ENV_{i:02d}" for i in range(n)]


def _ok_plan(n=12):
    return BudgetPlan(ued_slots=_ued_ids(n),
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="OK")


def _board(passed=True, critic_rejects=()):
    status = "PASS" if passed else "FAIL"
    critic = SimpleNamespace(
        role=C.ROLE_CRITIC_SKEPTIC,
        parsed_json={"critic_reject_intervention_ids": list(critic_rejects)})
    return SimpleNamespace(supervision_guard_status=status,
                           leakage_guard_status=status,
                           envelopes=[critic])


def _descriptor(env_id, intervention_ids=()):
    return TaskParamsDescriptor(
        descriptor_id=env_id,
        mock_env_family="threat_distance_family",
        mock_axis_values={"threat_distance_grading": "graded"},
        mock_variant_id=f"var:{env_id}",
        mock_variant_kind="single_axis",
        mutation_axes=["threat_distance_grading"],
        provenance={"source_intervention_ids": list(intervention_ids)})


CTRL = BAGRUEdController()
BATCH = BatchPlanner().plan(8)


# ---------------------------------------------------------------------------
# A. 12 UED + 4 anchors, director unauthorized -> structural true / final
#    false / archive commit rejected
# ---------------------------------------------------------------------------

def test_case_a_structural_ready_but_director_unauthorized():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    gate = evaluate_launch_gate(plan, BATCH, selected, [], _board(),
                                legal_ids=[d.descriptor_id for d in selected])
    assert gate.structural_batch_ready is True
    assert gate.director_training_authorized is False
    assert gate.final_training_launch_authorized is False
    assert gate.reasons == ()
    assert gate.gate_version == GATE_VERSION
    # the package flag is never set inside this package
    assert C.TRAINING_AUTHORIZED is False
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(
            selected, {}, launch_gate=gate, launch_context=_context(gate),
            batch_plan=BATCH, board_out=_board(),
            legal_ids=[d.descriptor_id for d in selected])


# ---------------------------------------------------------------------------
# B. 12 UED + 4 anchors + director TRUE (synthetic unit test ONLY) -> final
#    true and the gate hash verification passes; the package authorization
#    backstop (C.TRAINING_AUTHORIZED=false) then still refuses the commit —
#    the two layers are strictly separate. NO real training is started.
# ---------------------------------------------------------------------------

def test_case_b_director_true_synthetic_gate_verifies():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    legal_ids = [d.descriptor_id for d in selected]
    gate = evaluate_launch_gate(plan, BATCH, selected, [], _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids)
    assert gate.structural_batch_ready is True
    assert gate.director_training_authorized is True
    assert gate.final_training_launch_authorized is True
    # hash verification passes (no ARCHIVE_COMMIT_REJECTED); the refusal is
    # the package-level authorization backstop, proving separation of layers.
    # CC3 fix3: the SYNTHETIC all-true context rides along with the gate —
    # commit requires BOTH (a gate-only commit path no longer exists).
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_UNAUTHORIZED"):
        ProposalArchive().commit(
            selected, {}, launch_gate=gate,
            launch_context=_all_true_context(gate),
            batch_plan=BATCH, board_out=_board(), legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# C. 11 UED + 4 anchors -> structural false, final false
# ---------------------------------------------------------------------------

def test_case_c_eleven_slots_structural_false():
    plan = BudgetPlan(ued_slots=_ued_ids(11),
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="INSUFFICIENT",
                      shortfall_note="only 11 eligible descriptors for 12 UED "
                                     "slots; NO backfill / NO k-reduction / "
                                     "NO re-LLM")
    gate = evaluate_launch_gate(plan, BATCH, [], [], _board(), legal_ids=[])
    assert gate.structural_batch_ready is False
    assert gate.final_training_launch_authorized is False
    assert any("selected_ued_slots=11" in r for r in gate.reasons)
    assert any("budget_plan_status=INSUFFICIENT" in r for r in gate.reasons)
    assert any("unresolved_shortfall" in r for r in gate.reasons)


# ---------------------------------------------------------------------------
# D. commit without a gate -> type-level fail closed (keyword required)
# ---------------------------------------------------------------------------

def test_case_d_commit_without_gate_is_type_error():
    with pytest.raises(TypeError):
        ProposalArchive().commit([], {})          # no launch_gate at all


# ---------------------------------------------------------------------------
# E. commit with launch_gate=None -> fail closed (None is NOT a bypass)
# ---------------------------------------------------------------------------

def test_case_e_commit_with_none_gate_fails_closed():
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit([], {}, launch_gate=None,
                                 launch_context=None)
    # a plain dict gate (the fix1 shape) is rejected too — strong type only
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit([], {}, launch_gate={"batch_plan_ready": True},
                                 launch_context=None)
    # CC3 fix3 (§3): a dict / None context is rejected the same way even
    # next to a structurally perfect gate (constructed in test_case_b shape)
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    legal_ids = [d.descriptor_id for d in selected]
    gate = evaluate_launch_gate(plan, BATCH, selected, [], _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids)
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 launch_context=None,
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# F. refresh(dry_run=False, launch_gate=None) -> immediate fail closed
# ---------------------------------------------------------------------------

def test_case_f_refresh_non_dry_run_requires_gate():
    with pytest.raises(AssertionError, match="REFRESH_GATE_REQUIRED"):
        ProposalArchive().refresh([], {}, dry_run=False, launch_gate=None)
    with pytest.raises(AssertionError, match="REFRESH_GATE_REQUIRED"):
        ProposalArchive().refresh([], {}, dry_run=False)   # default None


# ---------------------------------------------------------------------------
# G. gate hash vs CURRENT batch plan mismatch -> fail closed
# ---------------------------------------------------------------------------

def test_case_g_batch_plan_hash_mismatch_fails_closed():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    legal_ids = [d.descriptor_id for d in selected]
    gate = evaluate_launch_gate(plan, BATCH, selected, [], _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids)
    other_batch = BatchPlanner().plan(4)     # different plan -> different hash
    assert compute_batch_plan_hash(other_batch) != gate.batch_plan_hash
    with pytest.raises(AssertionError,
                       match="ARCHIVE_COMMIT_REJECTED.*batch_plan_hash"):
        ProposalArchive().commit(
            selected, {}, launch_gate=gate,
            launch_context=_all_true_context(gate),
            batch_plan=other_batch,
            board_out=_board(), legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# H. gate from ANOTHER descriptor batch -> fail closed
# ---------------------------------------------------------------------------

def test_case_h_gate_from_another_descriptor_batch_fails_closed():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    legal_ids = [d.descriptor_id for d in selected]
    gate = evaluate_launch_gate(plan, BATCH, selected, [], _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids)
    foreign = [_descriptor(f"OTHER_ENV_{i:02d}") for i in range(12)]
    assert compute_selected_descriptor_hash(foreign) != \
        gate.selected_descriptor_hash
    with pytest.raises(AssertionError,
                       match="ARCHIVE_COMMIT_REJECTED.*selected_descriptor"):
        ProposalArchive().commit(
            foreign, {}, launch_gate=gate,
            launch_context=_all_true_context(gate),
            batch_plan=BATCH,
            board_out=_board(), legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# I. one SELECTED descriptor illegal -> the structural gate itself refuses
# ---------------------------------------------------------------------------

def test_case_i_selected_illegal_descriptor_blocks_structural_gate():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    legal_ids = [d.descriptor_id for d in selected]
    rejected = [dict(descriptor_id="CF_ENV_07",
                     code="ILLEGAL_PROPOSAL_FIELD",
                     message="illegal")]
    gate = evaluate_launch_gate(plan, BATCH, selected, rejected, _board(),
                                legal_ids=legal_ids)
    assert gate.structural_batch_ready is False
    assert gate.final_training_launch_authorized is False
    assert any("selected_descriptor_illegal" in r for r in gate.reasons)
    # ... and even a forged director=true cannot authorize it
    gate2 = evaluate_launch_gate(plan, BATCH, selected, rejected, _board(),
                                 director_training_authorized=True,
                                 legal_ids=legal_ids)
    assert gate2.final_training_launch_authorized is False
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(selected, {}, launch_gate=gate2,
                                 launch_context=_context(gate2),
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# J. dry_run=true without a gate -> diagnostic plan, archive untouched
# ---------------------------------------------------------------------------

def test_case_j_dry_run_gateless_is_diagnostic_only():
    archive = ProposalArchive()
    plan = _ok_plan(3)
    selected = [_descriptor(e) for e in plan.ued_slots]
    out = archive.refresh(selected, {d.descriptor_id: 0.5 for d in selected},
                          dry_run=True)          # launch_gate=None allowed
    assert out["dry_run"] is True
    assert out["training_authorized"] is False
    assert out["would_add"] and len(out["would_add"]) == 3
    assert archive.entries == {}                 # active archive untouched


# ---------------------------------------------------------------------------
# contract + provenance invariants
# ---------------------------------------------------------------------------

def test_launch_gate_contract_enforced_at_construction():
    # final != structural AND director -> refused at construction
    with pytest.raises(ValueError, match="LAUNCH_GATE_CONTRACT_VIOLATED"):
        LaunchGate(structural_batch_ready=False,
                   director_training_authorized=True,
                   final_training_launch_authorized=True,
                   batch_plan_hash="0" * 64, selected_descriptor_hash="0" * 64,
                   guard_report_hash="0" * 64, legality_report_hash="0" * 64)
    with pytest.raises(ValueError, match="LAUNCH_GATE_VERSION_MISMATCH"):
        LaunchGate(structural_batch_ready=True,
                   director_training_authorized=True,
                   final_training_launch_authorized=True,
                   batch_plan_hash="0" * 64, selected_descriptor_hash="0" * 64,
                   guard_report_hash="0" * 64, legality_report_hash="0" * 64,
                   gate_version="forged.v0")


def test_critic_hard_reject_on_selected_blocks_structural_gate():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    # CF_ENV_03 descends from an intervention the critic hard-rejected
    selected[3] = _descriptor("CF_ENV_03", intervention_ids=["itv:rejected:1"])
    legal_ids = [d.descriptor_id for d in selected]
    board = _board(critic_rejects=["itv:rejected:1"])
    gate = evaluate_launch_gate(plan, BATCH, selected, [], board,
                                legal_ids=legal_ids)
    assert gate.structural_batch_ready is False
    assert any("selected_proposal_critic_hard_reject" in r
               for r in gate.reasons)


def test_guard_violation_blocks_structural_gate():
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    gate = evaluate_launch_gate(plan, BATCH, selected, [], _board(False),
                                legal_ids=[d.descriptor_id
                                           for d in selected])
    assert gate.structural_batch_ready is False
    assert any("unresolved_guard_violation" in r for r in gate.reasons)


def test_three_anchors_refused_at_schema_and_gate():
    with pytest.raises(ValueError, match="ANCHOR_COUNT"):
        BudgetPlan(ued_slots=_ued_ids(12),
                   anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS)[:3],
                   status="OK")
    plan_like = SimpleNamespace(ued_slots=_ued_ids(12),
                                anchor_slots=list(
                                    C.GLOBAL_CANONICAL_ANCHOR_IDS)[:3],
                                status="OK", shortfall_note="")
    gate = evaluate_launch_gate(plan_like, BATCH, [], [], _board(),
                                legal_ids=[])
    assert gate.structural_batch_ready is False
    assert any("canonical_anchor_slots" in r for r in gate.reasons)


# ---------------------------------------------------------------------------
# end-to-end: synthetic window is structurally ready, never trains
# ---------------------------------------------------------------------------

def test_full_dry_run_gate_ready_but_training_forbidden():
    result = CTRL.run_dry_run(build_unsafe_rest_raw_rollout())
    gate = result.launch_gate
    assert gate["structural_batch_ready"] is True
    assert gate["director_training_authorized"] is False
    assert gate["final_training_launch_authorized"] is False
    assert gate["gate_version"] == GATE_VERSION
    assert all(len(gate[k]) == 64 for k in (
        "batch_plan_hash", "selected_descriptor_hash",
        "guard_report_hash", "legality_report_hash"))
    # CC3 fix3 (§2): the gate now carries the FULL six-way hash binding
    assert len(gate["critic_report_hash"]) == 64
    assert len(gate["director_authorization_hash"]) == 64
    # CC3 fix3 (§1): the strong-typed LaunchContext rides with the result —
    # assembled from the SAME gate (one binding, never two records), with the
    # extra window conditions fail-closed FALSE in a dry run
    ctx = result.launch_context
    assert ctx["context_version"] == CONTEXT_VERSION
    assert ctx["structural_batch_ready"] is True
    assert ctx["director_training_authorized"] is False
    assert ctx["final_training_launch_authorized"] is False
    assert ctx["review_certificate_valid"] is False
    assert ctx["provenance_valid"] is False
    assert ctx["guards_passed"] is True      # both board guards PASS here
    assert ctx["simulator_probe_complete"] is False
    assert ctx["selection_complete"] is False
    assert len(ctx["clip_batch_hash"]) == 64
    assert ctx["reasons"]                    # fail-closed reasons recorded
    for key in ("batch_plan_hash", "selected_descriptor_hash",
                "legality_report_hash", "guard_report_hash",
                "critic_report_hash", "director_authorization_hash"):
        assert ctx[key] == gate[key]         # ONE identical binding
    cert = result.dry_run_certificate
    assert cert["run_class"] == "ENGINEERING_DRY_RUN"
    assert cert["structural_batch_ready"] is True
    assert cert["director_training_authorized"] is False
    assert cert["final_training_launch_authorized"] is False
    assert cert["training_authorized"] is False
    assert cert["training_started"] is False
    assert cert["launch_block_reasons"] == []
    assert cert["real_llm_calls"] == 0
    assert cert["real_environment_rollouts"] == 0
    # CC3 fix3 (§2/§1): certificate mirrors the six-way binding + context
    assert cert["gate_critic_report_hash"] == gate["critic_report_hash"]
    assert cert["gate_director_authorization_hash"] == \
        gate["director_authorization_hash"]
    assert cert["context_version"] == CONTEXT_VERSION
    assert cert["context_review_certificate_valid"] is False
    assert cert["context_provenance_valid"] is False
    assert cert["context_guards_passed"] is True
    assert cert["context_simulator_probe_complete"] is False
    assert cert["context_selection_complete"] is False
    assert cert["context_final_training_launch_authorized"] is False
    # CC3 fix3 (§10): board and certificate bind ONE shared clip batch
    assert cert["clip_payload_batch_hash"] == ctx["clip_batch_hash"]
    assert cert["board_symbolic_clip_batch_hash"] == ctx["clip_batch_hash"]
    assert cert["board_and_certificate_share_clip_batch"] is True
    # CC3 fix3 (§7): clip caps surfaced with an explicit drop count
    assert cert["clips_dropped_by_caps"] == result.clips_dropped
    assert cert["max_clips_per_episode"] == C.MAX_CLIPS_PER_EPISODE
    assert cert["max_clips_per_review_window"] == \
        C.MAX_CLIPS_PER_REVIEW_WINDOW
    assert cert["symbolic_behavior_clip_count"] == \
        len(result.symbolic_behavior_clips)
    assert cert["symbolic_behavior_clip_count"] <= \
        C.MAX_CLIPS_PER_REVIEW_WINDOW
    # the forbidden double-meaning field name must not exist anywhere
    assert "training_launch_authorized" not in cert
    assert "training_launch_authorized" not in gate
    assert "training_launch_authorized" not in ctx
