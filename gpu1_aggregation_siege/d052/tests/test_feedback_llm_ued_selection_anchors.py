"""C12 + C13: multi-criterion Soft-Copeland Stage-2 selection + the shared
frozen anchor-manifest seam.

C12 asserts the Stage-2 cut consumes the SHARED ``d052.bagr_ued.soft_copeland``
implementation (no fork, no hand-written scalar): the eight RAW criteria are
stored separately on every audit record, the ranking hash is exactly the one
the shared module produces for the same bundles, the degenerate
critic_penalty dimension is recorded ``constant=true`` (never silently
dropped), and the family-diverse greedy pick is deterministic.

C13 asserts the four anchor slots are filled ONLY through the shared-manifest
seam: absent / unfrozen manifest -> AnchorManifestBlocked
(BLOCKED_SHARED_ANCHOR_MANIFEST), tampered hash -> fail closed, a valid
frozen manifest's anchors are consumed verbatim, and the controller falls
back to the EXPLICITLY LABELED scaffold placeholder with the budget
unchanged (12 dynamic + 4 anchors).
"""
import pytest

from d052.bagr_ued.soft_copeland import (
    RAW_DIMENSIONS,
    EnvironmentScoreBundle,
    soft_copeland_rank,
)
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.anchor_manifest import (
    SCAFFOLD_PLACEHOLDER_NOT_SHARED,
    SHARED_MANIFEST_BOUND_LABEL,
    AnchorManifestBlocked,
    AnchorManifestSource,
    SharedAnchorManifest,
)
from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_INCREASE,
    LEVEL_NONE,
    ROLE_TREATMENT,
    AxisDirective,
)
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.environment_generator import (
    FAMILY_AXES,
    generate_candidates_from_directives,
)
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    CurriculumPlan,
    FamilyAllocation,
    ProbeMetrics,
)
from d052.feedback_llm_ued.multi_criterion_selection import (
    CRITERION_NAMES,
    copeland_stage2_selection,
    stage2_criteria,
)
from d052.feedback_llm_ued.simulator_probe import (
    DeterministicSymbolicProbeRunner,
    run_staged_funnel,
)

MAX_TRANSITIONS = (C.STAGE2_STUDENT_EPISODES_MAX
                   + C.STAGE2_REFERENCE_EPISODES_MAX) * C.ROLLOUT_LENGTH

#: four anchor ids that are NOT the local constants — a frozen manifest
#: carrying them proves the seam consumes INJECTED ids, not the fallback
SHARED_TEST_ANCHORS = (
    "SHARED_ANCHOR_ALPHA", "SHARED_ANCHOR_BETA",
    "SHARED_ANCHOR_GAMMA", "SHARED_ANCHOR_DELTA",
)


def _metrics(**over) -> ProbeMetrics:
    base = dict(stage="full", student_success_rate=0.5,
                student_behavior_activation=0.5,
                student_front_progress=0.5,
                reference_success_rate=0.6,
                reference_mean_progress=0.6,
                reference_behavior_activation=0.6,
                global_retention=0.8, regret=0.1, learnability=0.7,
                simulator_transitions=MAX_TRANSITIONS)
    base.update(over)
    return ProbeMetrics(**base)


def _cand(cid: str, family: str) -> CandidateEnvironment:
    axis = FAMILY_AXES[family][0]
    return CandidateEnvironment(
        candidate_id=cid, environment_family=family,
        axis_values={axis: "medium"},
        held_constant_axes={a: "medium" for a in FAMILY_AXES[family]
                            if a != axis},
        variant_id=f"var-{cid}", variant_kind="test",
        mutation_axes=[axis], distinguishes_hypothesis_ids=["hyp-00"])


def _pool():
    """A deterministic 6-candidate pool: 3 from each of two families."""
    fam_a, fam_b = C.ENVIRONMENT_FAMILIES[0], C.ENVIRONMENT_FAMILIES[1]
    pool = []
    for i in range(3):
        pool.append((_cand(f"a{i:02d}", fam_a),
                     _metrics(regret=0.1 + i * 0.01)))
    for i in range(3):
        pool.append((_cand(f"b{i:02d}", fam_b),
                     _metrics(regret=0.2 + i * 0.01)))
    return pool


def _directive(family, *, window=0, role=ROLE_TREATMENT):
    """One legal treatment AxisDirective for ``family``."""
    axis = FAMILY_AXES[family][0]
    held = {a: "medium" for a in FAMILY_AXES[family] if a != axis}
    return AxisDirective(
        directive_id=f"dir-sel-w{window:02d}-{family}-{axis}-treatment",
        source_window=window, environment_family=family, axis=axis,
        old_level=LEVEL_NONE, new_level="medium",
        direction=DIRECTION_INCREASE, experiment_control_role=role,
        held_constant_axes=held,
        expected_next_signature={"student_success_rate": 0.5},
        rationale="selection-test directive")


def _funnel_candidates():
    """64 directive-driven candidates over the four bootstrap families."""
    plan = CurriculumPlan(
        plan_id="plan-sel-test", window=0, mode=C.MODE_NORMAL_FEEDBACK,
        allocations=[FamilyAllocation(
            environment_family=f, slots=3, decision=C.DECISION_MUTATE,
            reason="selection test", is_exploration=True)
            for f in C.ENVIRONMENT_FAMILIES[:4]])
    directives = [_directive(f) for f in C.ENVIRONMENT_FAMILIES[:4]]
    hyp_fam = {f: [f"hyp-{i:02d}"]
               for i, f in enumerate(C.ENVIRONMENT_FAMILIES[:4])}
    return generate_candidates_from_directives(
        plan, directives=directives, hypothesis_families=hyp_fam)


class TestStage2Criteria:
    def test_maps_exactly_the_eight_raw_criteria(self):
        m = _metrics(regret=0.25, student_front_progress=0.6,
                     student_behavior_activation=0.4,
                     reference_behavior_activation=0.7,
                     learnability=0.9, global_retention=0.75)
        crit = stage2_criteria(m, family=C.ENVIRONMENT_FAMILIES[0],
                               family_counts={C.ENVIRONMENT_FAMILIES[0]: 2,
                                              C.ENVIRONMENT_FAMILIES[1]: 2},
                               pool_size=4, max_transitions=MAX_TRANSITIONS)
        assert set(crit) == set(RAW_DIMENSIONS) == set(CRITERION_NAMES)
        assert crit["front_regret"] == pytest.approx(0.25)
        assert crit["global_regret"] == pytest.approx(0.25)  # degenerate
        assert crit["behavioral_gap"] == pytest.approx(0.3)
        assert crit["learning_progress"] == pytest.approx(0.6)
        assert crit["learnability"] == pytest.approx(0.9)
        assert crit["global_retention"] == pytest.approx(0.75)
        # pool-relative family rarity: 2 of 4 -> 1 - 1/3
        assert crit["diversity"] == pytest.approx(1.0 - 1.0 / 3.0)
        # identical episode budget -> cost ratio 1.0
        assert crit["critic_penalty"] == pytest.approx(1.0)

    def test_clamps_into_bundle_ranges(self):
        """Extreme-but-schema-legal probe metrics still map inside [0,1]:
        regret is only lower-bounded in ProbeMetrics, and a transition
        overshoot must clamp instead of crashing the bundle schema."""
        m = _metrics(regret=1.0, student_front_progress=1.0,
                     learnability=1.0, global_retention=0.0,
                     simulator_transitions=5 * MAX_TRANSITIONS,
                     student_behavior_activation=0.9,
                     reference_behavior_activation=0.1)
        crit = stage2_criteria(m, family=C.ENVIRONMENT_FAMILIES[0],
                               family_counts={C.ENVIRONMENT_FAMILIES[0]: 1},
                               pool_size=1, max_transitions=MAX_TRANSITIONS)
        for name, value in crit.items():
            assert 0.0 <= value <= 1.0, name
        assert crit["behavioral_gap"] == 0.0       # negative gap floors at 0
        assert crit["critic_penalty"] == 1.0       # cost overshoot clamps
        assert crit["global_retention"] == 0.0

    def test_negative_inputs_fail_closed(self):
        m = _metrics()
        with pytest.raises(ValueError, match="ILLEGAL_STAGE2_POOL_SIZE"):
            stage2_criteria(m, family="f", family_counts={"f": 1},
                            pool_size=0, max_transitions=MAX_TRANSITIONS)
        with pytest.raises(ValueError,
                           match="ILLEGAL_STAGE2_MAX_TRANSITIONS"):
            stage2_criteria(m, family="f", family_counts={"f": 1},
                            pool_size=1, max_transitions=0)
        with pytest.raises(ValueError,
                           match="STAGE2_FAMILY_COUNT_MISSING"):
            stage2_criteria(m, family="f", family_counts={"other": 1},
                            pool_size=1, max_transitions=MAX_TRANSITIONS)


class TestCopelandStage2Selection:
    def test_consumes_the_shared_soft_copeland_verbatim(self):
        """The ranking carried by the selection MUST be byte-identical to a
        direct call of the shared ``soft_copeland_rank`` over the same
        bundles — proof the direction consumes the common implementation
        instead of forking it."""
        pool = _pool()
        picked, audit, ranking = copeland_stage2_selection(
            pool, keep=3, max_transitions=MAX_TRANSITIONS)
        bundles = [EnvironmentScoreBundle(
            environment_id=cand.candidate_id,
            alpha_front=C.ALPHA_FRONT_STAGE2,
            **{k: v for k, v in audit[cand.candidate_id]["criteria"].items()})
            for cand, _m in pool]
        direct = soft_copeland_rank(bundles)
        assert direct.ranking_hash == ranking.ranking_hash
        direct_scores = {e.environment_id: e.copeland_score
                         for e in direct.entries}
        direct_ranks = {e.environment_id: e.rank for e in direct.entries}
        for cid, entry in audit.items():
            assert entry["copeland_score"] == direct_scores[cid]
            assert entry["copeland_rank"] == direct_ranks[cid]
            # the eight criteria stay SEPARATE on every audit record
            assert set(entry["criteria"]) == set(RAW_DIMENSIONS)
        assert len(picked) == 3

    def test_constant_critic_penalty_is_recorded_not_dropped(self):
        pool = _pool()          # identical episode budget across the pool
        _picked, _audit, ranking = copeland_stage2_selection(
            pool, keep=3, max_transitions=MAX_TRANSITIONS)
        prov = ranking.normalization_provenance["critic_penalty"]
        assert prov["constant"] is True

    def test_selection_is_deterministic(self):
        p1, a1, r1 = copeland_stage2_selection(
            _pool(), keep=4, max_transitions=MAX_TRANSITIONS)
        p2, a2, r2 = copeland_stage2_selection(
            _pool(), keep=4, max_transitions=MAX_TRANSITIONS)
        assert [c.candidate_id for c, _m in p1] == \
            [c.candidate_id for c, _m in p2]
        assert a1 == a2
        assert r1.ranking_hash == r2.ranking_hash

    def test_family_penalty_diversifies_the_pick(self):
        """With zero family penalty a strictly dominant family takes every
        slot; with the diversity penalty the greedy pass yields ground to
        the other family. Same pool, same Copeland scores — only the
        documented greedy penalty differs."""
        no_penalty, _, _ = copeland_stage2_selection(
            _pool(), keep=3, max_transitions=MAX_TRANSITIONS,
            family_penalty=0.0)
        fams_no = [c.environment_family for c, _m in no_penalty]
        assert len(set(fams_no)) == 1, "dominant family must sweep keep=3"
        penalized, _, _ = copeland_stage2_selection(
            _pool(), keep=3, max_transitions=MAX_TRANSITIONS,
            family_penalty=0.5)
        fams_pen = [c.environment_family for c, _m in penalized]
        assert len(set(fams_pen)) > 1

    def test_keep_larger_than_pool_takes_everything(self):
        picked, _audit, _ranking = copeland_stage2_selection(
            _pool(), keep=99, max_transitions=MAX_TRANSITIONS)
        assert len(picked) == 6

    def test_negative_inputs_fail_closed(self):
        with pytest.raises(ValueError, match="EMPTY_STAGE2_POOL"):
            copeland_stage2_selection([], keep=3,
                                      max_transitions=MAX_TRANSITIONS)
        with pytest.raises(ValueError, match="ILLEGAL_STAGE2_KEEP"):
            copeland_stage2_selection(_pool(), keep=0,
                                      max_transitions=MAX_TRANSITIONS)
        bad = [(_cand("x00", C.ENVIRONMENT_FAMILIES[0]),
                dict(not_metrics=True))]
        with pytest.raises(ValueError, match="ILLEGAL_STAGE2_POOL_METRICS"):
            copeland_stage2_selection(bad, keep=1,
                                      max_transitions=MAX_TRANSITIONS)


class TestFunnelSelectionAudit:
    """The funnel's stage-2 audit trail after the C12 rewiring."""

    def _batch(self, **kw):
        return run_staged_funnel(_funnel_candidates(),
                                 DeterministicSymbolicProbeRunner(),
                                 window=0, **kw)

    def test_stage2_records_carry_separate_criteria(self):
        batch = self._batch()
        assert len(batch.stage2_results) == 24
        for rec in batch.stage2_results:
            assert set(rec["criteria"]) == set(RAW_DIMENSIONS)
            assert isinstance(rec["copeland_rank"], int)
            assert rec["score"] == round(
                rec["score"], 6)                    # rounded audit score
        assert sum(1 for r in batch.stage2_results if r["selected"]) == 12
        assert batch.copeland_ranking_hash           # hash-bound audit trail

    def test_anchor_ids_are_injected_not_hardcoded(self):
        batch = self._batch(anchor_ids=SHARED_TEST_ANCHORS)
        assert batch.anchor_ids == SHARED_TEST_ANCHORS
        assert batch.final_batch[12:] == list(SHARED_TEST_ANCHORS)
        assert batch.funnel_stats["anchors"] == C.GLOBAL_ANCHOR_SLOTS
        assert batch.funnel_stats["final_batch"] == C.FINAL_BATCH

    def test_anchor_slot_count_is_fail_closed(self):
        cands = _funnel_candidates()
        with pytest.raises(ValueError, match="ILLEGAL_ANCHOR_SLOT_COUNT"):
            run_staged_funnel(cands, DeterministicSymbolicProbeRunner(),
                              window=0, anchor_ids=SHARED_TEST_ANCHORS[:3])
        with pytest.raises(ValueError, match="ILLEGAL_ANCHOR_SLOT_COUNT"):
            run_staged_funnel(cands, DeterministicSymbolicProbeRunner(),
                              window=0,
                              anchor_ids=SHARED_TEST_ANCHORS + ("EXTRA",))
        with pytest.raises(ValueError, match="DUPLICATE_ANCHOR_ID"):
            run_staged_funnel(
                cands, DeterministicSymbolicProbeRunner(), window=0,
                anchor_ids=("A", "B", "C", "A"))


class TestSharedAnchorManifest:
    def test_valid_manifest_computes_its_hash(self):
        m = SharedAnchorManifest(manifest_id="shared.manifest.v1",
                                 anchors=list(SHARED_TEST_ANCHORS),
                                 frozen=True)
        assert m.manifest_hash
        assert m.rehash() == m.manifest_hash

    def test_wrong_slot_count_rejected(self):
        with pytest.raises(ValueError, match="ILLEGAL_ANCHOR_SLOT_COUNT"):
            SharedAnchorManifest(manifest_id="m",
                                 anchors=list(SHARED_TEST_ANCHORS[:3]),
                                 frozen=True)

    def test_duplicate_anchor_rejected(self):
        with pytest.raises(ValueError, match="DUPLICATE_ANCHOR_ID"):
            SharedAnchorManifest(manifest_id="m",
                                 anchors=["A", "B", "C", "A"], frozen=True)

    def test_empty_anchor_id_rejected(self):
        with pytest.raises(ValueError, match="ILLEGAL_ANCHOR_ID"):
            SharedAnchorManifest(manifest_id="m",
                                 anchors=["A", "B", "C", ""], frozen=True)


class TestAnchorManifestSource:
    def test_absent_manifest_fails_closed(self):
        with pytest.raises(AnchorManifestBlocked,
                           match=C.BLOCKED_SHARED_ANCHOR_MANIFEST):
            AnchorManifestSource().resolve()

    def test_unfrozen_manifest_fails_closed(self):
        m = SharedAnchorManifest(manifest_id="m",
                                 anchors=list(SHARED_TEST_ANCHORS),
                                 frozen=False)
        with pytest.raises(AnchorManifestBlocked,
                           match=C.BLOCKED_SHARED_ANCHOR_MANIFEST):
            AnchorManifestSource(manifest=m).resolve()

    def test_tampered_manifest_fails_closed(self):
        m = SharedAnchorManifest(manifest_id="m",
                                 anchors=list(SHARED_TEST_ANCHORS),
                                 frozen=True)
        # simulate storage-level tampering past the model validators
        object.__setattr__(m, "anchors",
                           ["TAMPERED_1", "TAMPERED_2", "TAMPERED_3",
                            "TAMPERED_4"])
        with pytest.raises(ValueError,
                           match="ANCHOR_MANIFEST_HASH_MISMATCH"):
            AnchorManifestSource(manifest=m).resolve()

    def test_frozen_manifest_resolves_to_its_own_anchors(self):
        m = SharedAnchorManifest(manifest_id="m",
                                 anchors=list(SHARED_TEST_ANCHORS),
                                 frozen=True)
        assert AnchorManifestSource(manifest=m).resolve() == \
            SHARED_TEST_ANCHORS

    def test_scaffold_placeholder_is_the_local_constant(self):
        assert AnchorManifestSource().scaffold_placeholder() == \
            C.GLOBAL_CANONICAL_ANCHOR_IDS


class TestControllerAnchorBinding:
    """E2E: the loop's anchor slots come from the seam, honestly labeled,
    with the budget (12 dynamic + 4 anchors) identical either way."""

    def test_default_run_uses_labeled_placeholder(self):
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK)
        assert ctl.anchor_ids == C.GLOBAL_CANONICAL_ANCHOR_IDS
        assert ctl.anchor_binding == SCAFFOLD_PLACEHOLDER_NOT_SHARED
        summary = ctl.run(max_windows=2)
        for w in summary.windows:
            assert w["anchor_binding"] == SCAFFOLD_PLACEHOLDER_NOT_SHARED
            assert w["funnel_stats"]["anchors"] == C.GLOBAL_ANCHOR_SLOTS
            assert w["funnel_stats"]["final_batch"] == C.FINAL_BATCH
            # per-window probe cost invariant (budget unchanged)
            assert w["funnel_stats"]["total_simulator_transitions"] == 61440
        # the round-level honesty flag stays False
        assert C.SHARED_ANCHOR_MANIFEST_BOUND is False

    def test_injected_frozen_manifest_binds_the_slots(self):
        manifest = SharedAnchorManifest(manifest_id="shared.manifest.v1",
                                        anchors=list(SHARED_TEST_ANCHORS),
                                        frozen=True)
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                    anchor_manifest=manifest)
        assert ctl.anchor_binding == SHARED_MANIFEST_BOUND_LABEL
        summary = ctl.run(max_windows=1)
        w = summary.windows[0]
        assert w["anchor_binding"] == SHARED_MANIFEST_BOUND_LABEL
        assert w["funnel_stats"]["anchors"] == C.GLOBAL_ANCHOR_SLOTS
        assert w["funnel_stats"]["final_batch"] == C.FINAL_BATCH

    def test_mapping_manifest_is_accepted(self):
        ctl = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            anchor_manifest=dict(manifest_id="shared.manifest.v1",
                                 anchors=list(SHARED_TEST_ANCHORS),
                                 frozen=True))
        assert ctl.anchor_binding == SHARED_MANIFEST_BOUND_LABEL
        assert ctl.anchor_ids == SHARED_TEST_ANCHORS

    def test_unfrozen_injection_falls_back_to_placeholder(self):
        """Unfrozen manifest is blocked at the seam; the loop does not die —
        it runs on the EXPLICITLY LABELED placeholder, budget unchanged."""
        manifest = SharedAnchorManifest(manifest_id="shared.manifest.v1",
                                        anchors=list(SHARED_TEST_ANCHORS),
                                        frozen=False)
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                    anchor_manifest=manifest)
        assert ctl.anchor_binding == SCAFFOLD_PLACEHOLDER_NOT_SHARED
        assert ctl.anchor_ids == C.GLOBAL_CANONICAL_ANCHOR_IDS
        summary = ctl.run(max_windows=1)
        assert summary.windows[0]["anchor_binding"] == \
            SCAFFOLD_PLACEHOLDER_NOT_SHARED
        assert summary.windows[0]["funnel_stats"]["final_batch"] == \
            C.FINAL_BATCH

    def test_all_three_modes_share_the_anchor_budget(self):
        bindings = {}
        for mode in (C.MODE_STATIC_LLM, C.MODE_NORMAL_FEEDBACK,
                     C.MODE_SHUFFLED_FEEDBACK):
            ctl = FeedbackUEDController(mode)
            summary = ctl.run(max_windows=1)
            w = summary.windows[0]
            bindings[mode] = (w["anchor_binding"],
                              w["funnel_stats"]["anchors"],
                              w["funnel_stats"]["final_batch"],
                              w["funnel_stats"]["total_simulator_transitions"])
        assert len(set(bindings.values())) == 1
        assert bindings[C.MODE_STATIC_LLM][0] == SCAFFOLD_PLACEHOLDER_NOT_SHARED
