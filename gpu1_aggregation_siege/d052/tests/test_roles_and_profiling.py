"""GATE 6 — role protocol (4 roles, reconciled registry, judgment-batch validation)
+ deterministic Student profile + Modeler judgment firewall."""
import pytest

from d052.achievements import AchievementError
from d052.profiling import (
    EvidenceCheck,
    MachineFacts,
    ModelerJudgment,
    build_student_profile,
    is_mastered,
    is_proficient,
    mastery_tier,
)
from d052.roles import (
    ROLE_REGISTRY,
    SCORING_ROLES,
    RoleProtocolError,
    assert_registry_consistency,
    critic_vetoed,
    headline_scores,
    role_definition,
    validate_judgment_batch,
)
from d052.roles.protocol import RoleName
from d052.schemas.roles import RoleJudgment


def _batch(cid="t1", critic_reject=False):
    return [
        RoleJudgment(role="tutor", candidate_id=cid,
                     scores={"progression_score": 0.7}),
        RoleJudgment(role="critic", candidate_id=cid,
                     scores={"critic_penalty": 0.2}, critic_reject=critic_reject),
        RoleJudgment(role="explorer", candidate_id=cid,
                     scores={"novelty_score": 0.5}),
    ]


# --- role registry ----------------------------------------------------------

def test_registry_has_four_roles_three_scoring():
    assert_registry_consistency()
    assert set(ROLE_REGISTRY) == set(RoleName)
    assert len(ROLE_REGISTRY) == 4
    assert set(SCORING_ROLES) == {RoleName.TUTOR, RoleName.CRITIC, RoleName.EXPLORER}
    assert ROLE_REGISTRY[RoleName.MODELER].is_scoring_role is False


def test_role_pins_are_versioned():
    d = role_definition(RoleName.TUTOR)
    assert d.prompt_version and d.exact_model_id and d.provider
    assert d.output_schema == "role_judgment_v2"


# --- judgment-batch validation ---------------------------------------------

def test_valid_batch():
    batch = validate_judgment_batch("t1", _batch())
    assert set(batch) == set(SCORING_ROLES)


def test_duplicate_role_rejected():
    js = _batch()
    js.append(RoleJudgment(role="tutor", candidate_id="t1",
                           scores={"progression_score": 0.1}))
    with pytest.raises(RoleProtocolError) as ei:
        validate_judgment_batch("t1", js)
    assert ei.value.code == RoleProtocolError.DUPLICATE_ROLE


def test_missing_required_role_rejected():
    with pytest.raises(RoleProtocolError) as ei:
        validate_judgment_batch("t1", _batch()[:2])  # drop explorer
    assert ei.value.code == RoleProtocolError.MISSING_ROLE


def test_candidate_mismatch_rejected():
    js = _batch(cid="t2")
    with pytest.raises(RoleProtocolError) as ei:
        validate_judgment_batch("t1", js)
    assert ei.value.code == RoleProtocolError.CANDIDATE_MISMATCH


def test_modeler_not_a_scoring_role_in_batch():
    js = _batch()
    # a modeler judgment cannot be coerced into a per-candidate RoleJudgment
    # (ScoringRole excludes modeler) -> validation error
    with pytest.raises(Exception):
        validate_judgment_batch(
            "t1", js + [{"role": "modeler", "candidate_id": "t1",
                         "scores": {"x": 1.0}}])


def test_critic_veto_and_headlines():
    batch = validate_judgment_batch("t1", _batch(critic_reject=True))
    assert critic_vetoed(batch) is True
    batch_ok = validate_judgment_batch("t1", _batch(critic_reject=False))
    assert critic_vetoed(batch_ok) is False
    h = headline_scores(batch_ok)
    assert h["tutor"] == 0.7 and h["explorer"] == 0.5


# --- deterministic student profile -----------------------------------------

@pytest.mark.parametrize("sr,tier", [(0.0, 0), (0.05, 0), (0.1, 1), (0.49, 1),
                                     (0.5, 2), (0.79, 2), (0.8, 3), (0.94, 3),
                                     (0.95, 4), (1.0, 4)])
def test_mastery_tier_boundaries(sr, tier):
    assert mastery_tier(sr) == tier


def test_proficient_mastered_flags():
    assert is_proficient(0.8) and not is_proficient(0.79)
    assert is_mastered(0.95) and not is_mastered(0.94)


def test_profile_conservative_missing_is_zero():
    prof = build_student_profile({"collect_wood": 1.0})
    # only 1 of 67 measured -> overall ~ 1/67, mastered_count 1
    assert prof.measured_count == 1
    assert prof.mastered_count == 1
    assert prof.overall_mastery == pytest.approx(1.0 / 67)
    # per-tier mastery in [0,1]
    for t, m in prof.per_depth_tier_mastery.items():
        assert 0.0 <= m <= 1.0


def test_profile_alias_resolves_and_unknown_rejected():
    prof = build_student_profile({"defeat_orc_soldier": 0.9})  # alias
    assert "defeat_orc_solider" in prof.per_achievement_sr
    with pytest.raises(AchievementError):
        build_student_profile({"defeat_dragon": 0.5})


def test_profile_sr_bounds_enforced():
    with pytest.raises(Exception):
        build_student_profile({"collect_wood": 1.5})


# --- modeler judgment firewall ---------------------------------------------

def _modeler(foci=("collect_wood", "defeat_archer")):
    facts = MachineFacts(latest_sr={"collect_wood": 0.9}, recent_series=[],
                         forgetting_prefilter=[], num_snapshots=3)
    return ModelerJudgment(machine_facts=facts, student_state="RISING",
                           recommendation="DEPTH", guidance="push chains",
                           siege_foci=list(foci), evidence_check="supported",
                           provider="zhipu", exact_model_id="glm-4.5",
                           prompt_version="canonical_v2.roles.v1")


def test_modeler_judgment_valid_and_foci_canonicalized():
    j = _modeler(foci=["defeat_archer", "collect_wood"])
    assert j.siege_foci == ["collect_wood", "defeat_archer"]  # sorted canonical


def test_modeler_unknown_focus_rejected():
    with pytest.raises(AchievementError):
        _modeler(foci=["defeat_dragon"])


def test_modeler_duplicate_focus_rejected():
    with pytest.raises(Exception):
        _modeler(foci=["collect_wood", "collect_wood"])


def test_machine_facts_unknown_name_rejected():
    with pytest.raises(AchievementError):
        MachineFacts(latest_sr={"defeat_dragon": 0.1}, num_snapshots=1)
