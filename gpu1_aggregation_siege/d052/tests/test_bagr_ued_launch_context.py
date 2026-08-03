"""CC3 audit fix3 §1/§2/§3/§10 — strong-typed, UNOMITTABLE LaunchContext tests.

The LaunchContext binds the WHOLE review-window decision state around the
batch gate: seven conditions ANDed into final_training_launch_authorized
(enforced in __post_init__), the FULL six-way hash binding carried over from
the SAME LaunchGate (one binding, never two divergent records), and the hash
of the shared symbolic clip payload batch. ``archive.commit`` and
``archive.refresh(dry_run=False)`` require it ALONGSIDE the gate — there is
no gate-only commit path.

Contract coverage:
  1. commit omitting launch_context -> TypeError (keyword-required);
  2. commit with None/dict launch_context -> ARCHIVE_COMMIT_REJECTED;
  3. LaunchContext construction contract: final == AND(conditions),
     version pinned, every hash a 64-char digest;
  4. gate/context hash divergence -> ARCHIVE_COMMIT_REJECTED;
  5. refresh(dry_run=False, gate, launch_context=None) ->
     REFRESH_CONTEXT_REQUIRED (the gate check stays first);
  6. director authorization record changed after the gate ->
     ARCHIVE_COMMIT_REJECTED;
  7. critic envelope changed after the gate -> ARCHIVE_COMMIT_REJECTED;
  8. symbolic clip batch different from the bound batch ->
     ARCHIVE_COMMIT_REJECTED;
  9. evaluate_launch_context fail-closed defaults + gate requirement.

SYNTHETIC unit tests only — C.TRAINING_AUTHORIZED stays false, so every
commit that survives all structural checks still meets the package
authorization backstop (ARCHIVE_COMMIT_UNAUTHORIZED). NO training, NO real
LLM, NO rollout.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.archive import ProposalArchive
from d052.bagr_ued.batch_planner import BatchPlanner
from d052.bagr_ued.budget_allocator import BudgetPlan
from d052.bagr_ued.environment_proposer import TaskParamsDescriptor
from d052.bagr_ued.hashing import canonical_sha256
from d052.bagr_ued.launch_gate import (
    CONTEXT_VERSION,
    LaunchContext,
    evaluate_launch_context,
    evaluate_launch_gate,
)

H64 = "0" * 64
BATCH = BatchPlanner().plan(8)


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


def _descriptor(env_id):
    return TaskParamsDescriptor(
        descriptor_id=env_id,
        mock_env_family="threat_distance_family",
        mock_axis_values={"threat_distance_grading": "graded"},
        mock_variant_id=f"var:{env_id}",
        mock_variant_kind="single_axis",
        mutation_axes=["threat_distance_grading"],
        provenance={"source_intervention_ids": []})


def _authorized_gate(board=None, record=None):
    """A structurally perfect gate WITH director authorization (synthetic —
    the package backstop still refuses every commit)."""
    plan = _ok_plan(12)
    selected = [_descriptor(e) for e in plan.ued_slots]
    legal_ids = [d.descriptor_id for d in selected]
    gate = evaluate_launch_gate(plan, BATCH, selected, [],
                                board or _board(),
                                director_training_authorized=True,
                                legal_ids=legal_ids,
                                director_authorization_record=record)
    return plan, selected, legal_ids, gate


def _all_true_context(gate):
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


def _ctx_kwargs(**over):
    kw = dict(structural_batch_ready=True, review_certificate_valid=True,
              provenance_valid=True, guards_passed=True,
              simulator_probe_complete=True, selection_complete=True,
              director_training_authorized=True,
              final_training_launch_authorized=True,
              batch_plan_hash=H64, selected_descriptor_hash=H64,
              legality_report_hash=H64, guard_report_hash=H64,
              critic_report_hash=H64, director_authorization_hash=H64,
              clip_batch_hash=H64, reasons=())
    kw.update(over)
    return kw


# ---------------------------------------------------------------------------
# 1+2. commit without / with a non-LaunchContext context fails closed
# ---------------------------------------------------------------------------

def test_commit_omitting_launch_context_is_type_error():
    _, selected, legal_ids, gate = _authorized_gate()
    with pytest.raises(TypeError):
        # launch_context is keyword-REQUIRED with no default — omitting it is
        # a type-level failure, exactly like omitting the gate (CC3 fix3 §3)
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids)


def test_commit_with_none_or_dict_context_fails_closed():
    _, selected, legal_ids, gate = _authorized_gate()
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 launch_context=None,
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids)
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 launch_context={"final_training_launch_"
                                                 "authorized": True},
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# 3. LaunchContext construction contract
# ---------------------------------------------------------------------------

def test_launch_context_contract_enforced_at_construction():
    # final != AND(conditions) -> refused, whichever way it is forged
    with pytest.raises(ValueError, match="LAUNCH_CONTEXT_CONTRACT_VIOLATED"):
        LaunchContext(**_ctx_kwargs(final_training_launch_authorized=False))
    with pytest.raises(ValueError, match="LAUNCH_CONTEXT_CONTRACT_VIOLATED"):
        LaunchContext(**_ctx_kwargs(simulator_probe_complete=False))
    with pytest.raises(ValueError, match="LAUNCH_CONTEXT_VERSION_MISMATCH"):
        LaunchContext(**_ctx_kwargs(context_version="forged.v0"))
    with pytest.raises(ValueError, match="LAUNCH_CONTEXT_HASH_INVALID"):
        LaunchContext(**_ctx_kwargs(clip_batch_hash="not-a-sha"))
    with pytest.raises(ValueError, match="LAUNCH_CONTEXT_HASH_INVALID"):
        LaunchContext(**_ctx_kwargs(director_authorization_hash="abc123"))


def test_launch_context_final_is_computed_conjunction():
    ctx = LaunchContext(**_ctx_kwargs())
    assert ctx.final_training_launch_authorized is True
    assert ctx.context_version == CONTEXT_VERSION


# ---------------------------------------------------------------------------
# 4. gate/context hash divergence fails closed
# ---------------------------------------------------------------------------

def test_gate_context_hash_divergence_fails_closed():
    _, selected, legal_ids, gate = _authorized_gate()
    forged = dataclasses.replace(_all_true_context(gate),
                                 batch_plan_hash="1" * 64)
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_REJECTED"):
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 launch_context=forged,
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# 5. refresh(dry_run=False) needs the context alongside the gate
# ---------------------------------------------------------------------------

def test_refresh_non_dry_run_requires_context():
    plan, selected, legal_ids, gate = _authorized_gate()
    scores = {d.descriptor_id: 0.5 for d in selected}
    with pytest.raises(AssertionError, match="REFRESH_CONTEXT_REQUIRED"):
        ProposalArchive().refresh(selected, scores, dry_run=False,
                                  launch_gate=gate, launch_context=None)
    # with gate AND context the flow reaches the package backstop —
    # proving the context passed all structural checks (still unauthorized)
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_UNAUTHORIZED"):
        ProposalArchive().refresh(selected, scores, dry_run=False,
                                  launch_gate=gate,
                                  launch_context=_all_true_context(gate))


# ---------------------------------------------------------------------------
# 6. director authorization record changed after the gate fails closed
# ---------------------------------------------------------------------------

def test_director_authorization_record_mismatch_fails_closed():
    record = {"source": "director.authorization.v1"}
    _, selected, legal_ids, gate = _authorized_gate(record=record)
    with pytest.raises(AssertionError,
                       match="ARCHIVE_COMMIT_REJECTED.*director_authorization"):
        ProposalArchive().commit(
            selected, {}, launch_gate=gate,
            launch_context=_all_true_context(gate),
            batch_plan=BATCH, board_out=_board(), legal_ids=legal_ids,
            director_authorization_record={"source": "forged.v2"})


# ---------------------------------------------------------------------------
# 7. critic envelope changed after the gate fails closed
# ---------------------------------------------------------------------------

def test_critic_envelope_mismatch_fails_closed():
    _, selected, legal_ids, gate = _authorized_gate()
    altered = _board(critic_rejects=["itv:added:after:gate"])
    with pytest.raises(AssertionError,
                       match="ARCHIVE_COMMIT_REJECTED.*critic_report_hash"):
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 launch_context=_all_true_context(gate),
                                 batch_plan=BATCH, board_out=altered,
                                 legal_ids=legal_ids)


# ---------------------------------------------------------------------------
# 8. a clip batch different from the bound batch fails closed
# ---------------------------------------------------------------------------

def test_clip_batch_mismatch_fails_closed():
    _, selected, legal_ids, gate = _authorized_gate()
    foreign = [{"clip_payload_sha256": "a" * 64}]
    with pytest.raises(AssertionError,
                       match="ARCHIVE_COMMIT_REJECTED.*clip_batch_hash"):
        ProposalArchive().commit(selected, {}, launch_gate=gate,
                                 launch_context=_all_true_context(gate),
                                 batch_plan=BATCH, board_out=_board(),
                                 legal_ids=legal_ids,
                                 symbolic_payloads=foreign)


# ---------------------------------------------------------------------------
# 9. evaluate_launch_context: fail-closed defaults + gate requirement
# ---------------------------------------------------------------------------

def test_evaluate_launch_context_fail_closed_defaults():
    _, _, _, gate = _authorized_gate()
    ctx = evaluate_launch_context(gate, _board(), symbolic_payloads=())
    assert ctx.structural_batch_ready is True
    assert ctx.director_training_authorized is True
    assert ctx.guards_passed is True
    # every extra window condition DEFAULTS FALSE -> final stays false
    assert ctx.review_certificate_valid is False
    assert ctx.provenance_valid is False
    assert ctx.simulator_probe_complete is False
    assert ctx.selection_complete is False
    assert ctx.final_training_launch_authorized is False
    for needle in ("review_certificate_not_established",
                   "provenance_chain_not_established",
                   "simulator_probe_incomplete",
                   "selection_incomplete"):
        assert any(needle in r for r in ctx.reasons)
    assert ctx.clip_batch_hash == canonical_sha256([])
    # the six hashes are carried over from the gate — one identical binding
    for name in ("batch_plan_hash", "selected_descriptor_hash",
                 "legality_report_hash", "guard_report_hash",
                 "critic_report_hash", "director_authorization_hash"):
        assert getattr(ctx, name) == getattr(gate, name)


def test_evaluate_launch_context_all_conditions_true():
    _, _, _, gate = _authorized_gate()
    ctx = evaluate_launch_context(gate, _board(),
                                  review_certificate_valid=True,
                                  provenance_valid=True,
                                  simulator_probe_complete=True,
                                  selection_complete=True,
                                  symbolic_payloads=())
    assert ctx.final_training_launch_authorized is True
    assert ctx.reasons == ()


def test_evaluate_launch_context_requires_strong_gate():
    with pytest.raises(AssertionError, match="LAUNCH_CONTEXT_REQUIRES_GATE"):
        evaluate_launch_context({"structural_batch_ready": True}, _board())
