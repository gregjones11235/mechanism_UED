"""CC1 audit fix1 §4 — controller shortfall launch-gate regression tests.

At the final batch/launch decision, BATCH_PLAN_READY and
TRAINING_LAUNCH_AUTHORIZED both require ALL structural conditions at once:

    budget_plan.status==OK / selected_ued_slots==12 /
    canonical_anchor_slots==4 (the fixed global anchors) /
    total_envs==16 / rollout_length==128 / transitions_per_update==2048 /
    all selected descriptors legal / no unresolved shortfall /
    no unresolved guard violation.

Cases A-F from the fix task:
  A. 12 UED + 4 anchors            -> BATCH_PLAN_READY=true (still NO training)
  B. 11 UED + 4 anchors            -> both false, shortfall=1
  C. 12 UED + 3 anchors            -> false (schema refuses; gate second layer)
  D. 12 + 4 but one illegal descr. -> false
  E. status=INSUFFICIENT even with an apparent 12 -> false (status governs)
  F. active archive commit with gate false -> fail closed
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.archive import ProposalArchive
from d052.bagr_ued.batch_planner import BatchPlanner
from d052.bagr_ued.budget_allocator import BudgetPlan
from d052.bagr_ued.controller import BAGRUEdController
from d052.bagr_ued.synthetic_traces import build_unsafe_rest_raw_rollout


def _ued_ids(n):
    return [f"CF_ENV_{i:02d}" for i in range(n)]


def _ok_plan(n=12):
    return BudgetPlan(ued_slots=_ued_ids(n),
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="OK")


def _board(passed=True):
    status = "PASS" if passed else "FAIL"
    return SimpleNamespace(supervision_guard_status=status,
                           leakage_guard_status=status)


CTRL = BAGRUEdController()
BATCH = BatchPlanner().plan(8)


# ---------------------------------------------------------------------------
# A. 12 UED + 4 anchors -> ready — but training still unauthorized
# ---------------------------------------------------------------------------

def test_case_a_full_complement_is_ready_but_never_trains():
    gate = CTRL._launch_gate(_ok_plan(12), BATCH, [], _board())
    assert gate["batch_plan_ready"] is True
    assert gate["training_launch_authorized"] is True
    assert gate["launch_block_reasons"] == []
    assert gate["shortfall"] == 0
    # readiness is NOT authorization: the director flag stays false
    assert C.TRAINING_AUTHORIZED is False
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_UNAUTHORIZED"):
        ProposalArchive().commit([], {}, launch_gate=gate)


# ---------------------------------------------------------------------------
# B. 11 UED + 4 anchors -> both false, shortfall=1
# ---------------------------------------------------------------------------

def test_case_b_eleven_ued_slots_blocks_with_shortfall_one():
    plan = BudgetPlan(ued_slots=_ued_ids(11),
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="INSUFFICIENT",
                      shortfall_note="only 11 eligible descriptors for 12 UED "
                                     "slots; NO backfill / NO k-reduction / "
                                     "NO re-LLM")
    gate = CTRL._launch_gate(plan, BATCH, [], _board())
    assert gate["batch_plan_ready"] is False
    assert gate["training_launch_authorized"] is False
    assert gate["shortfall"] == 1
    assert any("budget_plan_status=INSUFFICIENT" in r
               for r in gate["launch_block_reasons"])
    assert any("selected_ued_slots=11" in r
               for r in gate["launch_block_reasons"])
    assert any("unresolved_shortfall" in r
               for r in gate["launch_block_reasons"])


# ---------------------------------------------------------------------------
# C. 3 anchors -> refused at the schema level AND by the gate
# ---------------------------------------------------------------------------

def test_case_c_three_anchors_refused_at_schema_and_gate():
    with pytest.raises(ValueError, match="ANCHOR_COUNT"):
        BudgetPlan(ued_slots=_ued_ids(12),
                   anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS)[:3],
                   status="OK")
    # gate second layer (defense in depth — e.g. a foreign plan object):
    plan_like = SimpleNamespace(ued_slots=_ued_ids(12),
                                anchor_slots=list(
                                    C.GLOBAL_CANONICAL_ANCHOR_IDS)[:3],
                                status="OK", shortfall_note="")
    gate = CTRL._launch_gate(plan_like, BATCH, [], _board())
    assert gate["batch_plan_ready"] is False
    assert gate["training_launch_authorized"] is False
    assert any("canonical_anchor_slots" in r
               for r in gate["launch_block_reasons"])


# ---------------------------------------------------------------------------
# D. full complement but one illegal descriptor -> blocked
# ---------------------------------------------------------------------------

def test_case_d_illegal_descriptor_blocks_the_gate():
    rejected = [dict(descriptor_id="CF_ENV_07",
                     reasons=["UNAUTHORIZED_DESCRIPTOR_FIELD: mob_spawn_rate"])]
    gate = CTRL._launch_gate(_ok_plan(12), BATCH, rejected, _board())
    assert gate["batch_plan_ready"] is False
    assert gate["training_launch_authorized"] is False
    assert any("illegal_descriptors=1" in r
               for r in gate["launch_block_reasons"])


# ---------------------------------------------------------------------------
# E. status=INSUFFICIENT even if the slot list LOOKS like 12 -> blocked
# ---------------------------------------------------------------------------

def test_case_e_status_governs_over_apparent_length():
    # (a duplicated list cannot even exist — BudgetPlan rejects duplicates —
    # so "looks like 12 via duplicates" collapses to the status signal)
    with pytest.raises(ValueError, match="DUPLICATE_UED_SLOT"):
        BudgetPlan(ued_slots=_ued_ids(6) + _ued_ids(6),
                   anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                   status="OK")
    plan = BudgetPlan(ued_slots=_ued_ids(12),
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="INSUFFICIENT",
                      shortfall_note="apparent 12 slots but allocator "
                                     "reported shortfall; status governs")
    gate = CTRL._launch_gate(plan, BATCH, [], _board())
    assert gate["batch_plan_ready"] is False
    assert gate["training_launch_authorized"] is False
    assert any("budget_plan_status=INSUFFICIENT" in r
               for r in gate["launch_block_reasons"])


# ---------------------------------------------------------------------------
# F. active archive commit while the gate is false -> fail closed
# ---------------------------------------------------------------------------

def test_case_f_active_commit_blocked_when_gate_false():
    plan = BudgetPlan(ued_slots=_ued_ids(11),
                      anchor_slots=list(C.GLOBAL_CANONICAL_ANCHOR_IDS),
                      status="INSUFFICIENT", shortfall_note="short 1")
    gate = CTRL._launch_gate(plan, BATCH, [], _board())
    assert gate["batch_plan_ready"] is False
    with pytest.raises(AssertionError,
                       match="ACTIVE_ARCHIVE_COMMIT_BLOCKED"):
        ProposalArchive().commit([], {}, launch_gate=gate)
    # the non-dry-run refresh path is rejected too (authorization backstop)
    with pytest.raises(AssertionError, match="ARCHIVE_COMMIT_UNAUTHORIZED"):
        ProposalArchive().refresh([], {}, dry_run=False)
    # unresolved guard violation also blocks
    gate_guard = CTRL._launch_gate(_ok_plan(12), BATCH, [], _board(False))
    assert gate_guard["batch_plan_ready"] is False
    assert any("unresolved_guard_violation" in r
               for r in gate_guard["launch_block_reasons"])


# ---------------------------------------------------------------------------
# end-to-end: the synthetic window IS structurally ready, still never trains
# ---------------------------------------------------------------------------

def test_full_dry_run_gate_ready_but_training_forbidden():
    result = CTRL.run_dry_run(build_unsafe_rest_raw_rollout())
    gate = result.launch_gate
    assert gate["batch_plan_ready"] is True
    assert gate["training_launch_authorized"] is True
    assert gate["launch_block_reasons"] == []
    cert = result.dry_run_certificate
    assert cert["run_class"] == "ENGINEERING_DRY_RUN"
    assert cert["training_authorized"] is False
    assert cert["training_started"] is False
    assert cert["batch_plan_ready"] is True
    assert cert["launch_block_reasons"] == []
    assert cert["real_llm_calls"] == 0
    assert cert["real_environment_rollouts"] == 0
