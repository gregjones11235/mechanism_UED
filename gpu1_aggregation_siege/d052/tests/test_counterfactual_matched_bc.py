"""GATE 1 (Phase 2.5) — B/C strict-match counterfactual verifier.

Arms B and C must be IDENTICAL in every field EXCEPT the StudentProfile / Modeler
conditioning. verify_matched_bc passes a correctly-matched pair and fails closed
with a SPECIFIC code on any non-permitted difference. Permitted deltas (arm label,
modeler_enabled, student_profile_hash, modeler_context_hash, selector type S1->S2,
prompt_set_hash) may differ without failing.
"""
import pytest

from d052.counterfactual.ablation import modeler_ablation_arms
from d052.counterfactual.prompts import build_prompt_set
from d052.counterfactual.protocol import (
    CounterfactualArm,
    CounterfactualProtocolError,
    verify_matched_bc,
)
from d052.roles.protocol import RoleName
from d052.schemas.roles import ScoringRole
from d052.schemas.selector import CriticPolicy, SelectorConfig, SelectorType

_ROLE_NAMES = [RoleName.TUTOR, RoleName.CRITIC, RoleName.EXPLORER]
_POOL = "a" * 64
_CACHE = "b" * 64
_SP = "d" * 64
_CTX = "e" * 64


def _matched(k=8, seed=7):
    pb = build_prompt_set("B", _ROLE_NAMES, modeler_enabled=False)
    pc = build_prompt_set("C", _ROLE_NAMES, modeler_enabled=True,
                          student_profile_channel_id=_CTX)
    return modeler_ablation_arms(
        pool_hash=_POOL, judgment_cache_hash=_CACHE, prompt_set_b=pb,
        prompt_set_c=pc, k=k, seed=seed, student_profile_hash=_SP,
        modeler_context_hash=_CTX)


def test_valid_matched_pair_passes():
    b, c = _matched()
    v = verify_matched_bc(b, c)
    assert v.passed is True
    assert v.selector_b == "S1_THREE_ROLE"
    assert v.selector_c == "S2_FOUR_ROLE_MODELER"
    assert "pool_hash" in v.identical_fields
    assert "modeler_enabled" in v.permitted_delta_fields
    assert v.verification_hash  # content-bound


def test_permitted_deltas_actually_differ():
    b, c = _matched()
    # the legitimate counterfactual differences
    assert b.prompt_set_hash != c.prompt_set_hash
    assert b.modeler_enabled is False and c.modeler_enabled is True
    assert b.student_profile_hash is None and c.student_profile_hash == _SP
    assert b.modeler_context_hash is None and c.modeler_context_hash == _CTX
    assert b.selector.selector is SelectorType.S1_THREE_ROLE
    assert c.selector.selector is SelectorType.S2_FOUR_ROLE_MODELER
    # ...while the shared bindings are identical
    assert b.pool_hash == c.pool_hash == _POOL
    assert b.judgment_cache_hash == c.judgment_cache_hash == _CACHE
    assert b.modeler_bonus_weight == c.modeler_bonus_weight
    assert b.selector.k == c.selector.k and b.selector.seed == c.selector.seed


def test_pool_mismatch_fails():
    b, c = _matched()
    b2 = b.model_copy(update={"pool_hash": "f" * 64})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b2, c)
    assert ei.value.code == CounterfactualProtocolError.POOL_MISMATCH


def test_cache_mismatch_fails():
    b, c = _matched()
    c2 = c.model_copy(update={"judgment_cache_hash": "9" * 64})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b, c2)
    assert ei.value.code == CounterfactualProtocolError.CACHE_MISMATCH


def test_weight_mismatch_fails():
    b, c = _matched()
    b2 = b.model_copy(update={"modeler_bonus_weight": 0.5})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b2, c)
    assert ei.value.code == CounterfactualProtocolError.WEIGHT_MISMATCH


def test_k_mismatch_fails():
    b, c = _matched()
    c2 = c.model_copy(update={"selector": SelectorConfig(
        selector=SelectorType.S2_FOUR_ROLE_MODELER, k=9, seed=7,
        roles=[ScoringRole.TUTOR, ScoringRole.CRITIC, ScoringRole.EXPLORER])})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b, c2)
    assert ei.value.code == CounterfactualProtocolError.K_MISMATCH


def test_seed_mismatch_fails():
    b, c = _matched()
    c2 = c.model_copy(update={"selector": SelectorConfig(
        selector=SelectorType.S2_FOUR_ROLE_MODELER, k=8, seed=8,
        roles=[ScoringRole.TUTOR, ScoringRole.CRITIC, ScoringRole.EXPLORER])})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b, c2)
    assert ei.value.code == CounterfactualProtocolError.SEED_MISMATCH


def test_critic_policy_mismatch_fails():
    b, c = _matched()
    c2 = c.model_copy(update={"selector": SelectorConfig(
        selector=SelectorType.S2_FOUR_ROLE_MODELER, k=8, seed=7,
        critic_policy=CriticPolicy.SOFT_PENALTY,
        roles=[ScoringRole.TUTOR, ScoringRole.CRITIC, ScoringRole.EXPLORER])})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b, c2)
    assert ei.value.code == CounterfactualProtocolError.CRITIC_POLICY_MISMATCH


def test_roles_mismatch_fails():
    b, c = _matched()
    c2 = c.model_copy(update={"selector": SelectorConfig(
        selector=SelectorType.S2_FOUR_ROLE_MODELER, k=8, seed=7,
        roles=[ScoringRole.TUTOR, ScoringRole.CRITIC])})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b, c2)
    assert ei.value.code == CounterfactualProtocolError.ROLES_MISMATCH


def test_b_must_be_modeler_off():
    b, c = _matched()
    b2 = b.model_copy(update={"modeler_enabled": True})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b2, c)
    assert ei.value.code == CounterfactualProtocolError.B_NOT_MODELER_OFF


def test_c_must_be_modeler_on():
    b, c = _matched()
    c2 = c.model_copy(update={"modeler_enabled": False})
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(b, c2)
    assert ei.value.code == CounterfactualProtocolError.C_NOT_MODELER_ON


def test_wrong_ablation_selectors_fail():
    _, c = _matched()
    bad_b = CounterfactualArm(
        arm_label="B",
        selector=SelectorConfig(selector=SelectorType.S0_CANONICAL_BASELINE,
                                k=8, seed=7),
        pool_hash=_POOL, judgment_cache_hash=_CACHE, prompt_set_hash="1" * 64,
        modeler_enabled=False)
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(bad_b, c)
    assert ei.value.code == CounterfactualProtocolError.WRONG_ABLATION_SELECTORS


def test_arm_labels_enforced():
    b, c = _matched()
    with pytest.raises(CounterfactualProtocolError) as ei:
        verify_matched_bc(c, b)        # swapped -> first arm is not B
    assert ei.value.code == CounterfactualProtocolError.NOT_ARM_B
