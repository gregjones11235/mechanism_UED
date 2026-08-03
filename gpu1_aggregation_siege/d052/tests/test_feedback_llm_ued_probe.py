"""Staged probe funnel (64 -> 24 -> 12 + 4) + runner honesty."""
import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_HOLD,
    DIRECTION_INCREASE,
    LEVEL_NONE,
    ROLE_CONTROL,
    ROLE_TREATMENT,
    AxisDirective,
)
from d052.feedback_llm_ued.environment_generator import (
    FAMILY_AXES,
    generate_candidates_from_directives,
)
from d052.feedback_llm_ued.feedback_contracts import (
    CurriculumPlan,
    FamilyAllocation,
)
from d052.feedback_llm_ued.formal_isolation import ReferenceOutputGuard
from d052.feedback_llm_ued.simulator_probe import (
    CraftaxPreflightProbeRunner,
    DeterministicSymbolicProbeRunner,
    ProbeRunnerBlocked,
    run_staged_funnel,
    static_legality_check,
)

#: expected per-window probe cost: 64 fast probes x (2+1) episodes +
#: 24 full probes x (8+4) episodes, all at ROLLOUT_LENGTH transitions
EXPECTED_TRANSITIONS_PER_WINDOW = (
    C.RAW_CANDIDATES * (C.STAGE1_STUDENT_EPISODES
                        + C.STAGE1_REFERENCE_EPISODES) * C.ROLLOUT_LENGTH
    + C.STAGE1_KEEP * (C.STAGE2_STUDENT_EPISODES_MAX
                       + C.STAGE2_REFERENCE_EPISODES_MAX) * C.ROLLOUT_LENGTH)


def _bootstrap_plan():
    return CurriculumPlan(
        plan_id="plan-test-bootstrap", window=0, mode=C.MODE_NORMAL_FEEDBACK,
        allocations=[FamilyAllocation(
            environment_family=f, slots=3, decision=C.DECISION_MUTATE,
            reason="bootstrap", is_exploration=True)
            for f in C.ENVIRONMENT_FAMILIES[:4]])


def _directive(family, *, window=0, role=ROLE_TREATMENT, old_level=LEVEL_NONE,
               new_level="medium"):
    """One legal AxisDirective for ``family`` (board -> generator contract)."""
    axis = FAMILY_AXES[family][0]
    held = {a: "medium" for a in FAMILY_AXES[family] if a != axis}
    if role == ROLE_CONTROL:
        direction = DIRECTION_HOLD
        if old_level == LEVEL_NONE:
            old_level = "medium"     # a control re-measures an existing level
        new_level = old_level        # CONTROL_DIRECTIVE_MUST_HOLD
    else:
        direction = DIRECTION_INCREASE
    return AxisDirective(
        directive_id=f"dir-w{window:02d}-{family}-{axis}-"
                     f"{'treatment' if role == ROLE_TREATMENT else 'control'}",
        source_window=window,
        environment_family=family,
        axis=axis,
        old_level=old_level,
        new_level=new_level,
        direction=direction,
        experiment_control_role=role,
        held_constant_axes=held,
        expected_next_signature={"student_success_rate": 0.5},
        rationale="test directive")


def _candidates():
    """64 directive-driven candidates: 4 funded families, one directive each."""
    plan = _bootstrap_plan()
    directives = [_directive(f) for f in C.ENVIRONMENT_FAMILIES[:4]]
    hyp_fam = {f: [f"hyp-{i:02d}"]
               for i, f in enumerate(C.ENVIRONMENT_FAMILIES[:4])}
    return generate_candidates_from_directives(
        plan, directives=directives, hypothesis_families=hyp_fam)


class TestStaticLegality:
    def test_legal_candidate_passes(self):
        cand = _candidates()[0]
        ok, err = static_legality_check(cand)
        assert ok is True and err == ""

    def test_missing_legality_hint_rejected(self):
        cand = _candidates()[0].model_copy(update={"legality_hint": "REAL"})
        ok, err = static_legality_check(cand)
        assert ok is False and "legality_hint_missing" in err

    def test_unexpected_adapter_status_rejected(self):
        cand = _candidates()[0].model_copy(
            update={"real_adapter_status": "READY"})
        ok, err = static_legality_check(cand)
        assert ok is False and "unexpected_real_adapter_status" in err


class TestSymbolicRunner:
    def test_honest_identity(self):
        runner = DeterministicSymbolicProbeRunner()
        assert runner.real_simulator is False
        assert runner.status == C.REAL_SIMULATOR_PROBE_STATUS

    def test_probe_is_deterministic_and_clean(self):
        cand = _candidates()[0]
        runner = DeterministicSymbolicProbeRunner()
        m1 = runner.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)
        m2 = runner.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)
        assert m1.model_dump() == m2.model_dump()
        assert m1.probe_source == C.SOURCE_CANDIDATE_PROBE
        assert m1.simulator_transitions == 3 * C.ROLLOUT_LENGTH
        assert isinstance(m1.too_hard, bool) and isinstance(m1.too_easy, bool)
        # Reference/Student payloads carry no action-guidance carriers
        assert ReferenceOutputGuard().scan(m1.model_dump(),
                                           label="test") == []

    def test_illegal_stage_and_episodes(self):
        cand = _candidates()[0]
        runner = DeterministicSymbolicProbeRunner()
        with pytest.raises(ValueError, match="ILLEGAL_PROBE_STAGE"):
            runner.probe(cand, stage="medium", student_episodes=1,
                         reference_episodes=1)
        with pytest.raises(ValueError, match="ILLEGAL_PROBE_EPISODES"):
            runner.probe(cand, stage="fast", student_episodes=0,
                         reference_episodes=1)

    def test_episode_budget_enforced(self):
        cand = _candidates()[0]
        runner = DeterministicSymbolicProbeRunner()
        with pytest.raises(ValueError,
                           match="FAST_PROBE_EPISODE_BUDGET_EXCEEDED"):
            runner.probe(cand, stage="fast", student_episodes=3,
                         reference_episodes=1)
        with pytest.raises(ValueError,
                           match="FAST_PROBE_EPISODE_BUDGET_EXCEEDED"):
            runner.probe(cand, stage="fast", student_episodes=2,
                         reference_episodes=2)
        with pytest.raises(ValueError,
                           match="FULL_PROBE_STUDENT_EPISODES_OUT_OF_RANGE"):
            runner.probe(cand, stage="full", student_episodes=3,
                         reference_episodes=2)
        with pytest.raises(ValueError,
                           match="FULL_PROBE_STUDENT_EPISODES_OUT_OF_RANGE"):
            runner.probe(cand, stage="full", student_episodes=9,
                         reference_episodes=2)
        with pytest.raises(ValueError,
                           match="FULL_PROBE_REFERENCE_EPISODES_OUT_OF_RANGE"):
            runner.probe(cand, stage="full", student_episodes=4,
                         reference_episodes=1)
        with pytest.raises(ValueError,
                           match="FULL_PROBE_REFERENCE_EPISODES_OUT_OF_RANGE"):
            runner.probe(cand, stage="full", student_episodes=4,
                         reference_episodes=5)


class TestRealSeamBlocked:
    def test_craftax_runner_fails_closed(self):
        with pytest.raises(ProbeRunnerBlocked,
                           match="BLOCKED_NO_LOCAL_CRAFTAX"):
            CraftaxPreflightProbeRunner()


class TestStagedFunnel:
    def test_funnel_shape_and_accounting(self):
        cands = _candidates()
        assert len(cands) == C.RAW_CANDIDATES
        runner = DeterministicSymbolicProbeRunner()
        batch = run_staged_funnel(cands, runner, window=0)
        stats = batch.funnel_stats
        assert stats["raw"] == 64
        assert stats["static_rejects"] == 0
        assert stats["duplicates"] == 0
        assert stats["stage1_probed"] == 64
        assert stats["stage1_survivors"] == C.STAGE1_KEEP == 24
        assert stats["stage2_probed"] == 24
        assert stats["stage2_selected"] == C.STAGE2_KEEP == 12
        assert stats["dynamic_selected"] == 12
        assert stats["anchors"] == 4
        assert stats["final_batch"] == C.FINAL_BATCH == 16
        # honest simulator-cost accounting (fast + full probes only)
        assert stats["total_simulator_transitions"] == \
            EXPECTED_TRANSITIONS_PER_WINDOW
        assert batch.final_batch[12:] == list(C.GLOBAL_CANONICAL_ANCHOR_IDS)
        # all 24 full-probed candidates are recorded, 12 flagged selected
        assert len(batch.stage2_results) == 24
        assert sum(1 for r in batch.stage2_results if r["selected"]) == 12

    def test_funnel_deterministic_across_runners(self):
        cands = _candidates()
        b1 = run_staged_funnel(cands, DeterministicSymbolicProbeRunner(),
                               window=0)
        b2 = run_staged_funnel(cands, DeterministicSymbolicProbeRunner(),
                               window=0)
        assert b1.dynamic_selected == b2.dynamic_selected
        assert b1.funnel_stats == b2.funnel_stats
        assert [r["candidate_id"] for r in b1.stage1_results] == \
            [r["candidate_id"] for r in b2.stage1_results]

    def test_raw_cap_enforced(self):
        cands = _candidates()
        with pytest.raises(ValueError, match="RAW_CANDIDATE_CAP_EXCEEDED"):
            run_staged_funnel(cands + [cands[0]],
                              DeterministicSymbolicProbeRunner(), window=0)

    def test_duplicate_hash_dropped(self):
        cands = _candidates()
        batch = run_staged_funnel(cands + [cands[0]],
                                  DeterministicSymbolicProbeRunner(),
                                  window=0, raw_cap=C.RAW_CANDIDATES + 1)
        assert batch.duplicates == [cands[0].candidate_id]
        assert batch.funnel_stats["stage1_probed"] == 64

    def test_static_reject_recorded(self):
        cands = _candidates()
        bad = cands[1].model_copy(update={"legality_hint": "REAL"})
        batch = run_staged_funnel([cands[0], bad],
                                  DeterministicSymbolicProbeRunner(),
                                  window=0, raw_cap=64)
        assert len(batch.static_rejects) == 1
        assert batch.static_rejects[0]["candidate_id"] == bad.candidate_id
        assert "legality_hint_missing" in batch.static_rejects[0]["reason"]
        assert batch.funnel_stats["stage1_probed"] == 1


def _single_family_plan(family, slots=12):
    return CurriculumPlan(
        plan_id="plan-test-single", window=0, mode=C.MODE_NORMAL_FEEDBACK,
        allocations=[FamilyAllocation(
            environment_family=family, slots=slots, decision=C.DECISION_MUTATE,
            reason="single-family", is_exploration=True)])


class TestGenerator:
    def test_candidates_are_legal_and_bound_to_directives(self):
        cands = _candidates()
        assert len(cands) == 64
        hashes = {c.candidate_hash for c in cands}
        assert len(hashes) == 64                     # no hash collisions
        directives = {_directive(f).directive_id
                      for f in C.ENVIRONMENT_FAMILIES[:4]}
        for c in cands:
            assert c.real_adapter_status == C.REAL_SIMULATOR_PROBE_STATUS
            assert c.legality_hint.startswith("MOCK_ONLY")
            assert c.variant_kind == "directive"
            assert c.provenance["source"] == C.SOURCE_CANDIDATE_PROBE
            assert c.provenance["directive_id"] in directives
            assert c.provenance["directive_hash"]          # bound, non-empty
            assert len(c.distinguishes_hypothesis_ids) == 1
            assert len(c.mutation_axes) == 1
        families = {c.environment_family for c in cands}
        assert families == set(C.ENVIRONMENT_FAMILIES[:4])
        # slot-proportional split: 4 families x 3 slots each -> 16 each
        for f in C.ENVIRONMENT_FAMILIES[:4]:
            assert sum(1 for c in cands
                       if c.environment_family == f) == 16

    def test_generation_is_deterministic(self):
        c1 = _candidates()
        c2 = _candidates()
        assert [c.candidate_hash for c in c1] == [c.candidate_hash for c in c2]

    def test_single_directive_axis_config_is_not_rotated(self):
        """Abolished-index-rotation check: every candidate of one directive
        carries the IDENTICAL axis configuration (no i%len / i%3 cycling)."""
        fam = C.ENVIRONMENT_FAMILIES[0]
        axis = FAMILY_AXES[fam][0]
        d = _directive(fam, new_level="high")
        cands = generate_candidates_from_directives(
            _single_family_plan(fam), directives=[d],
            hypothesis_families={fam: ["hyp-00"]})
        assert len(cands) == 64
        for c in cands:
            assert c.axis_values == {axis: "high"}
            assert c.held_constant_axes == d.held_constant_axes
            assert c.provenance["directive_id"] == d.directive_id
            assert c.provenance["directive_hash"] == d.directive_hash

    def test_control_directive_remeasures_old_level(self):
        fam = C.ENVIRONMENT_FAMILIES[0]
        axis = FAMILY_AXES[fam][0]
        d = _directive(fam, role=ROLE_CONTROL, old_level="low")
        cands = generate_candidates_from_directives(
            _single_family_plan(fam), directives=[d],
            hypothesis_families={fam: ["hyp-00"]})
        assert len(cands) == 64
        assert all(c.axis_values == {axis: "low"} for c in cands)

    def test_two_directives_split_within_family(self):
        fam = C.ENVIRONMENT_FAMILIES[0]
        axis = FAMILY_AXES[fam][0]
        treat = _directive(fam, new_level="high")
        ctrl = _directive(fam, role=ROLE_CONTROL, old_level="medium")
        cands = generate_candidates_from_directives(
            _single_family_plan(fam), directives=[ctrl, treat],
            hypothesis_families={fam: ["hyp-00"]})
        assert len(cands) == 64
        by_dir = {}
        for c in cands:
            by_dir.setdefault(c.provenance["directive_id"], []).append(c)
        assert set(by_dir) == {treat.directive_id, ctrl.directive_id}
        assert len(by_dir[treat.directive_id]) == 32
        assert len(by_dir[ctrl.directive_id]) == 32
        assert all(c.axis_values == {axis: "high"}
                   for c in by_dir[treat.directive_id])
        assert all(c.axis_values == {axis: "medium"}
                   for c in by_dir[ctrl.directive_id])
        # variant ids bind candidate -> directive -> within-directive replica
        for c in by_dir[treat.directive_id]:
            assert c.variant_id.startswith(f"var-{fam}-{treat.directive_id}-")

    def test_directives_sorted_by_id_within_family(self):
        """Directive order in the input list must not change the batch."""
        fam = C.ENVIRONMENT_FAMILIES[0]
        treat = _directive(fam, new_level="high")
        ctrl = _directive(fam, role=ROLE_CONTROL, old_level="medium")
        hyp = {fam: ["hyp-00"]}
        c_ab = generate_candidates_from_directives(
            _single_family_plan(fam), directives=[ctrl, treat],
            hypothesis_families=hyp)
        c_ba = generate_candidates_from_directives(
            _single_family_plan(fam), directives=[treat, ctrl],
            hypothesis_families=hyp)
        assert [c.candidate_hash for c in c_ab] == \
            [c.candidate_hash for c in c_ba]

    def test_funded_family_without_directive_fails_closed(self):
        """Probing a funded family without a board specification is an
        uncontrolled mutation and must be refused (no silent fallback)."""
        directives = [_directive(f) for f in C.ENVIRONMENT_FAMILIES[:3]]
        hyp_fam = {f: [f"hyp-{i:02d}"]
                   for i, f in enumerate(C.ENVIRONMENT_FAMILIES[:4])}
        with pytest.raises(ValueError,
                           match="FUNDED_FAMILY_WITHOUT_DIRECTIVE"):
            generate_candidates_from_directives(
                _bootstrap_plan(), directives=directives,
                hypothesis_families=hyp_fam)

    def test_unfunded_plan_fails_closed(self):
        plan = CurriculumPlan(plan_id="p", window=0,
                              mode=C.MODE_NORMAL_FEEDBACK, allocations=[])
        with pytest.raises(ValueError, match="EMPTY_PLAN_BUDGET"):
            generate_candidates_from_directives(
                plan, directives=[], hypothesis_families={})
        # all-zero slots is equally unfunded
        plan = CurriculumPlan(
            plan_id="p", window=0, mode=C.MODE_NORMAL_FEEDBACK,
            allocations=[FamilyAllocation(
                environment_family=C.ENVIRONMENT_FAMILIES[0], slots=0,
                decision=C.DECISION_RETIRE, reason="retired",
                is_exploration=False)])
        with pytest.raises(ValueError, match="EMPTY_PLAN_BUDGET"):
            generate_candidates_from_directives(
                plan, directives=[_directive(C.ENVIRONMENT_FAMILIES[0])],
                hypothesis_families={})
