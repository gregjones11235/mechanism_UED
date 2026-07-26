"""GATE 2 — canonical schemas reject illegal / unknown / empty targets, forbid
extra fields, and enforce hash integrity + gate consistency.

Builds valid objects with the deterministic content-hash helpers, then asserts
each violation mode raises pydantic ValidationError (fail-closed).
"""
import pytest
from pydantic import ValidationError

from d052.schemas import (
    REQUIRED_GATES,
    AchievementRef,
    Candidate,
    CandidatePool,
    ExecutionMappingCertificate,
    NormalizedEntry,
    NormalizedRoleScores,
    RoleJudgment,
    RunConfig,
    ScoringRole,
    SelectionResult,
    SelectionStatus,
    SelectorConfig,
    SelectorType,
    TaskParams,
    compute_candidate_chash,
    compute_selection_hash,
)

HEX64 = "a" * 64


def mk_params(**over):
    base = dict(passive_spawn_multiplier=1.0, melee_spawn_multiplier=1.0,
                mob_health_multiplier=1.0, mob_damage_multiplier=1.0)
    base.update(over)
    return TaskParams(**base)


def mk_candidate(task_id="t1", names=("collect_wood",), params=None):
    tp = params or mk_params()
    distinct = sorted(set(names))
    chash = compute_candidate_chash(task_id, distinct, tp.model_dump())
    return Candidate(task_id=task_id, chash=chash, task_params=tp,
                     target_achievements=[{"name": n} for n in names])


# --- AchievementRef ---------------------------------------------------------

def test_achievement_ref_fills_ids():
    r = AchievementRef(name="defeat_kobold")
    assert r.canonical_id == 41 and r.goal_vector_index == 41


def test_achievement_ref_alias_resolves():
    r = AchievementRef(name="defeat_orc_soldier")  # correct spelling -> canonical
    assert r.name == "defeat_orc_solider" and r.canonical_id == 38


def test_achievement_ref_unknown_rejected():
    with pytest.raises(ValidationError) as ei:
        AchievementRef(name="defeat_dragon")
    assert "UNKNOWN_ACHIEVEMENT" in str(ei.value)


def test_achievement_ref_forbids_extra():
    with pytest.raises(ValidationError):
        AchievementRef(name="collect_wood", bogus=1)


# --- TaskParams -------------------------------------------------------------

def test_taskparams_valid():
    assert mk_params().mob_damage_multiplier == 1.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_taskparams_non_positive_rejected(bad):
    with pytest.raises(ValidationError) as ei:
        mk_params(passive_spawn_multiplier=bad)
    assert "NON_POSITIVE" in str(ei.value)


def test_taskparams_non_finite_rejected():
    with pytest.raises(ValidationError) as ei:
        mk_params(mob_health_multiplier=float("inf"))
    assert "NON_FINITE" in str(ei.value)


def test_taskparams_forbids_extra():
    with pytest.raises(ValidationError):
        TaskParams(passive_spawn_multiplier=1.0, melee_spawn_multiplier=1.0,
                   mob_health_multiplier=1.0, mob_damage_multiplier=1.0,
                   difficulty_tier=3)  # no such field anywhere


# --- Candidate --------------------------------------------------------------

def test_candidate_valid_and_short_id():
    c = mk_candidate("t1", ["defeat_archer", "collect_wood"])
    assert c.canonical_target_names == ["collect_wood", "defeat_archer"]
    assert len(c.legacy_short_id) == 16


def test_candidate_unknown_target_rejected():
    tp = mk_params()
    chash = compute_candidate_chash("t1", ["collect_wood"], tp.model_dump())
    with pytest.raises(ValidationError) as ei:
        Candidate(task_id="t1", chash=chash, task_params=tp,
                  target_achievements=[{"name": "defeat_dragon"}])
    assert "UNKNOWN_ACHIEVEMENT" in str(ei.value)


def test_candidate_empty_targets_rejected():
    tp = mk_params()
    chash = compute_candidate_chash("t1", [], tp.model_dump())
    with pytest.raises(ValidationError) as ei:
        Candidate(task_id="t1", chash=chash, task_params=tp, target_achievements=[])
    assert "min_length" in str(ei.value) or "at least" in str(ei.value).lower()


def test_candidate_too_many_targets_rejected():
    five = ["collect_wood", "place_table", "eat_cow", "collect_sapling",
            "collect_drink"]
    with pytest.raises(ValidationError) as ei:
        mk_candidate("t1", five)
    assert "MAX_TARGETS_EXCEEDED" in str(ei.value)


def test_candidate_duplicate_targets_rejected():
    with pytest.raises(ValidationError) as ei:
        mk_candidate("t1", ["collect_wood", "collect_wood"])
    assert "DUPLICATE_TARGET" in str(ei.value)


def test_candidate_hash_mismatch_rejected():
    tp = mk_params()
    with pytest.raises(ValidationError) as ei:
        Candidate(task_id="t1", chash=HEX64, task_params=tp,
                  target_achievements=[{"name": "collect_wood"}])
    assert "HASH_MISMATCH" in str(ei.value)


def test_candidate_bad_chash_format_rejected():
    tp = mk_params()
    with pytest.raises(ValidationError) as ei:
        Candidate(task_id="t1", chash="nothex", task_params=tp,
                  target_achievements=[{"name": "collect_wood"}])
    assert "INVALID_HASH" in str(ei.value)


# --- CandidatePool ----------------------------------------------------------

def test_pool_valid():
    cs = [mk_candidate("t1"), mk_candidate("t2", ["defeat_archer"])]
    pool = CandidatePool(pool_id="p", pool_hash=CandidatePool.hash_candidates(cs),
                         candidate_count=2, candidates=cs)
    assert pool.frozen is True


def test_pool_count_mismatch_rejected():
    cs = [mk_candidate("t1")]
    with pytest.raises(ValidationError) as ei:
        CandidatePool(pool_id="p", pool_hash=CandidatePool.hash_candidates(cs),
                      candidate_count=5, candidates=cs)
    assert "COUNT_MISMATCH" in str(ei.value)


def test_pool_duplicate_task_id_rejected():
    cs = [mk_candidate("t1"), mk_candidate("t1", ["defeat_archer"])]
    with pytest.raises(ValidationError) as ei:
        CandidatePool(pool_id="p", pool_hash=CandidatePool.hash_candidates(cs),
                      candidate_count=2, candidates=cs)
    assert "DUPLICATE_TASK_ID" in str(ei.value)


def test_pool_hash_mismatch_rejected():
    cs = [mk_candidate("t1")]
    with pytest.raises(ValidationError) as ei:
        CandidatePool(pool_id="p", pool_hash=HEX64, candidate_count=1, candidates=cs)
    assert "POOL_HASH_MISMATCH" in str(ei.value)


def test_pool_must_be_frozen():
    cs = [mk_candidate("t1")]
    with pytest.raises(ValidationError):
        CandidatePool(pool_id="p", pool_hash=CandidatePool.hash_candidates(cs),
                      candidate_count=1, candidates=cs, frozen=False)


# --- RoleJudgment -----------------------------------------------------------

def test_role_judgment_tutor_valid():
    j = RoleJudgment(role="tutor", candidate_id="t1",
                     scores={"progression_score": 0.7})
    assert j.headline_score == 0.7


def test_role_judgment_missing_headline_rejected():
    with pytest.raises(ValidationError) as ei:
        RoleJudgment(role="tutor", candidate_id="t1", scores={"other": 1.0})
    assert "MISSING_HEADLINE_SCORE" in str(ei.value)


def test_role_judgment_critic_requires_reject():
    with pytest.raises(ValidationError) as ei:
        RoleJudgment(role="critic", candidate_id="t1",
                     scores={"critic_penalty": 0.2})
    assert "MISSING_CRITIC_REJECT" in str(ei.value)
    # with the bit set, valid
    RoleJudgment(role="critic", candidate_id="t1",
                 scores={"critic_penalty": 0.2}, critic_reject=False)


def test_role_judgment_noncritic_reject_rejected():
    with pytest.raises(ValidationError) as ei:
        RoleJudgment(role="explorer", candidate_id="t1",
                     scores={"novelty_score": 0.5}, critic_reject=True)
    assert "UNEXPECTED_CRITIC_REJECT" in str(ei.value)


def test_role_judgment_empty_scores_rejected():
    with pytest.raises(ValidationError) as ei:
        RoleJudgment(role="tutor", candidate_id="t1", scores={})
    assert "EMPTY_SCORES" in str(ei.value)


def test_role_judgment_nonfinite_score_rejected():
    with pytest.raises(ValidationError) as ei:
        RoleJudgment(role="tutor", candidate_id="t1",
                     scores={"progression_score": float("nan")})
    assert "NON_FINITE" in str(ei.value)


# --- NormalizedRoleScores ---------------------------------------------------

def test_normalized_valid():
    e = [NormalizedEntry(candidate_id="t1", raw=3.0, normalized=0.5, rank=0,
                         tie_group=0)]
    NormalizedRoleScores(role="tutor", entries=e)


def test_normalized_out_of_range_rejected():
    with pytest.raises(ValidationError):
        NormalizedEntry(candidate_id="t1", raw=3.0, normalized=1.5, rank=0,
                        tie_group=0)


def test_normalized_wrong_method_rejected():
    e = [NormalizedEntry(candidate_id="t1", raw=1.0, normalized=0.5, rank=0,
                         tie_group=0)]
    with pytest.raises(ValidationError) as ei:
        NormalizedRoleScores(role="tutor", normalization="minmax", entries=e)
    assert "UNKNOWN_NORMALIZATION" in str(ei.value)


# --- SelectorConfig ---------------------------------------------------------

def test_selector_config_valid_soft_copeland():
    SelectorConfig(selector="SOFT_COPELAND", k=8, seed=0,
                   roles=["tutor", "critic", "explorer"])


def test_selector_budgeted_requires_budget():
    with pytest.raises(ValidationError) as ei:
        SelectorConfig(selector="BUDGETED_SOFT_COPELAND", k=8, seed=0,
                       roles=["tutor"])
    assert "MISSING_BUDGET" in str(ei.value)


def test_selector_nonbudgeted_rejects_budget():
    with pytest.raises(ValidationError) as ei:
        SelectorConfig(selector="SOFT_COPELAND", k=8, seed=0, roles=["tutor"],
                       budget=10.0)
    assert "UNEXPECTED_BUDGET" in str(ei.value)


def test_selector_s0_no_roles():
    SelectorConfig(selector="S0_CANONICAL_BASELINE", k=8, seed=0)  # ok, no roles
    with pytest.raises(ValidationError) as ei:
        SelectorConfig(selector="S0_CANONICAL_BASELINE", k=8, seed=0,
                       roles=["tutor"])
    assert "S0_NO_ROLES" in str(ei.value)


# --- SelectionResult --------------------------------------------------------

def _sel_hash(sel, pol, k, seed, ids):
    return compute_selection_hash(sel, pol, k, seed, ids)


def test_selection_result_ok_valid():
    ids = ["t1", "t2"]
    SelectionResult(selector="SOFT_COPELAND", critic_policy="hard_veto",
                    k_requested=2, seed=0, candidate_count_in=10, eligible_count=8,
                    selected_ids=ids, selection_status="OK",
                    selection_hash=_sel_hash("SOFT_COPELAND", "hard_veto", 2, 0, ids))


def test_selection_result_ok_requires_k():
    ids = ["t1"]
    with pytest.raises(ValidationError) as ei:
        SelectionResult(selector="SOFT_COPELAND", critic_policy="hard_veto",
                        k_requested=2, seed=0, candidate_count_in=10,
                        eligible_count=8, selected_ids=ids, selection_status="OK",
                        selection_hash=_sel_hash("SOFT_COPELAND", "hard_veto", 2, 0, ids))
    assert "STATUS_OK_REQUIRES_K" in str(ei.value)


def test_selection_result_insufficient_no_backfill():
    ids = ["t1", "t2"]
    h = _sel_hash("SOFT_COPELAND", "hard_veto", 2, 0, ids)
    # INSUFFICIENT but selected == k -> forbidden
    with pytest.raises(ValidationError) as ei:
        SelectionResult(selector="SOFT_COPELAND", critic_policy="hard_veto",
                        k_requested=2, seed=0, candidate_count_in=10,
                        eligible_count=2, selected_ids=ids,
                        selection_status="INSUFFICIENT_ELIGIBLE_CANDIDATES",
                        selection_hash=h, shortfall_note="only 2 eligible")
    assert "INSUFFICIENT_BUT_FULL" in str(ei.value)


def test_selection_result_insufficient_requires_note():
    ids = ["t1"]
    h = _sel_hash("SOFT_COPELAND", "hard_veto", 2, 0, ids)
    with pytest.raises(ValidationError) as ei:
        SelectionResult(selector="SOFT_COPELAND", critic_policy="hard_veto",
                        k_requested=2, seed=0, candidate_count_in=10,
                        eligible_count=1, selected_ids=ids,
                        selection_status="INSUFFICIENT_ELIGIBLE_CANDIDATES",
                        selection_hash=h, shortfall_note="")
    assert "MISSING_SHORTFALL_NOTE" in str(ei.value)


def test_selection_result_hash_mismatch():
    with pytest.raises(ValidationError) as ei:
        SelectionResult(selector="SOFT_COPELAND", critic_policy="hard_veto",
                        k_requested=1, seed=0, candidate_count_in=10,
                        eligible_count=8, selected_ids=["t1"],
                        selection_status="OK", selection_hash=HEX64)
    assert "SELECTION_HASH_MISMATCH" in str(ei.value)


# --- ExecutionMappingCertificate --------------------------------------------

def _cert(gates=None, executed=True, **over):
    g = {k: True for k in REQUIRED_GATES}
    if gates:
        g.update(gates)
    base = dict(candidate_id="t1", chash=HEX64,
                canonical_names=["collect_wood", "defeat_archer"],
                canonical_ids=[0, 66], goal_vector_indices=[0, 66],
                goal_vector_dim=67, goal_vector_ones=2, student_obs_dim=8335,
                conditioning_type="achievement_multi_hot",
                conditioning_dimension=67, task_spec_hash=HEX64,
                training_task_id="train_t1", gates=g,
                executed_as_intended=executed)
    base.update(over)
    return ExecutionMappingCertificate(**base)


def test_certificate_valid_all_pass():
    c = _cert()
    assert c.executed_as_intended is True


def test_certificate_executed_requires_all_gates():
    with pytest.raises(ValidationError) as ei:
        _cert(gates={"task_compiled": False}, executed=True)
    assert "EXECUTED_AS_INTENDED_REQUIRES_ALL_GATES" in str(ei.value)


def test_certificate_all_pass_but_not_executed_inconsistent():
    with pytest.raises(ValidationError) as ei:
        _cert(executed=False)
    assert "INCONSISTENT_CERTIFICATE" in str(ei.value)


def test_certificate_index_misaligned():
    with pytest.raises(ValidationError) as ei:
        _cert(goal_vector_indices=[0, 65])
    assert "INDEX_MISALIGNED" in str(ei.value)


def test_certificate_wrong_obs_dim():
    with pytest.raises(ValidationError) as ei:
        _cert(student_obs_dim=8300)  # banned legacy value
    assert "STUDENT_OBS_DIM" in str(ei.value)


def test_certificate_wrong_conditioning_dim():
    with pytest.raises(ValidationError) as ei:
        _cert(conditioning_dimension=32)  # banned 32-slot
    assert "CONDITIONING_DIMENSION" in str(ei.value)


# --- RunConfig --------------------------------------------------------------

def _run_cfg(**over):
    base = dict(protocol_version="canonical_v2", run_id="r1", seed=0,
                pool_id="p", pool_hash=HEX64,
                selector={"selector": "S0_CANONICAL_BASELINE", "k": 8, "seed": 0},
                output_dir="outputs/r1")
    base.update(over)
    return RunConfig(**base)


def test_runconfig_valid():
    rc = _run_cfg()
    assert rc.student_obs_dim == 8335
    assert rc.conditioning_dimension == 67


def test_runconfig_requires_protocol_version():
    with pytest.raises(ValidationError) as ei:
        _run_cfg(protocol_version=None)
    assert "protocol_version" in str(ei.value)


def test_runconfig_rejects_wrong_obs_dim():
    with pytest.raises(ValidationError):
        _run_cfg(student_obs_dim=8300)


def test_runconfig_rejects_legacy():
    with pytest.raises(ValidationError):
        _run_cfg(protocol_version="legacy")


def test_runconfig_rejects_allow_legacy_true():
    with pytest.raises(ValidationError):
        _run_cfg(allow_legacy_d052=True)
