"""GATE 3 (Phase 2.5) — canonical-target firewall regression.

Any salted / hash-modulo / unknown / empty / non-string target entering the
execution-mapping boundary MUST fail, and must fail with a SPECIFIC code so the
banned schemes (Python hash()-with-salt target mapping; hash modulo) are provably
rejected rather than slipping through as a generic "unknown". Legal canonical names
and the single audited alias pass.
"""
import pytest

from d052.achievements import AchievementError
from d052.counterfactual.firewall import (
    TargetFirewallError,
    assert_execution_mapping_rejects,
    assert_target_firewall,
    classify_target,
)
from d052.generation import build_pool

_TP = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
       "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}


# --- classification of each banned scheme ----------------------------------
@pytest.mark.parametrize("target", [
    "collect_wood::salt=abc123",
    "eat_cow#deadbeefcafe",
    "sha256:0123456789abcdef0123456789abcdef",
    "collect_wood 0123456789abcdef0123456789abcdef",
    "mine_coal::9f9f",
])
def test_salted_targets_classified_salted(target):
    assert classify_target(target) == TargetFirewallError.SALTED_TARGET_FORBIDDEN


@pytest.mark.parametrize("target", [
    "38",                       # raw integer id (hash-modulo output)
    "target_5",
    "id_12",
    "ach_66",
    "goal_3",
    "hash(name) % 67",
    "collect_wood_mod67",
    "0x2a",
])
def test_hash_modulo_targets_classified_modulo(target):
    assert classify_target(target) == TargetFirewallError.HASH_MODULO_TARGET_FORBIDDEN


def test_empty_and_nonstring_classified():
    assert classify_target("") == TargetFirewallError.EMPTY_TARGET_FORBIDDEN
    assert classify_target("   ") == TargetFirewallError.EMPTY_TARGET_FORBIDDEN
    assert classify_target(38) == TargetFirewallError.NON_STRING_TARGET_FORBIDDEN
    assert classify_target(None) == TargetFirewallError.NON_STRING_TARGET_FORBIDDEN


def test_unknown_plain_string_classified_unknown():
    assert classify_target("not_a_real_achievement") == \
        TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN


def test_legal_canonical_and_alias_pass():
    assert classify_target("collect_wood") is None
    assert classify_target("defeat_orc_soldier") is None   # audited alias -> id 38


# --- assert_target_firewall raises the specific code -----------------------
def test_firewall_rejects_salted_with_specific_code():
    with pytest.raises(TargetFirewallError) as ei:
        assert_target_firewall(["collect_wood::salt=ff"])
    assert ei.value.code == TargetFirewallError.SALTED_TARGET_FORBIDDEN


def test_firewall_rejects_hash_modulo_with_specific_code():
    with pytest.raises(TargetFirewallError) as ei:
        assert_target_firewall(["target_5"])
    assert ei.value.code == TargetFirewallError.HASH_MODULO_TARGET_FORBIDDEN


def test_firewall_rejects_empty():
    with pytest.raises(TargetFirewallError) as ei:
        assert_target_firewall([])
    assert ei.value.code == TargetFirewallError.EMPTY_TARGET_FORBIDDEN


def test_firewall_rejects_unknown():
    with pytest.raises(TargetFirewallError) as ei:
        assert_target_firewall(["utterly_unknown_goal"])
    assert ei.value.code == TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN


def test_firewall_salted_wins_over_unknown_in_mixed_set():
    # a salted target present alongside a legal one is still named SALTED
    with pytest.raises(TargetFirewallError) as ei:
        assert_target_firewall(["collect_wood", "eat_cow#deadbeef"])
    assert ei.value.code == TargetFirewallError.SALTED_TARGET_FORBIDDEN


def test_firewall_resolves_legal_set_including_alias():
    resolved = assert_target_firewall(["defeat_orc_soldier", "collect_wood"])
    assert resolved == ["collect_wood", "defeat_orc_solider"]  # alias->canonical, sorted


# --- execution-mapping boundary MUST reject each banned class --------------
@pytest.mark.parametrize("targets,expected_code", [
    (["collect_wood::salt=ab"], TargetFirewallError.SALTED_TARGET_FORBIDDEN),
    (["38"], TargetFirewallError.HASH_MODULO_TARGET_FORBIDDEN),
    (["target_9"], TargetFirewallError.HASH_MODULO_TARGET_FORBIDDEN),
    ([], TargetFirewallError.EMPTY_TARGET_FORBIDDEN),
    (["unknown_goal"], TargetFirewallError.UNKNOWN_TARGET_FORBIDDEN),
    ([42], TargetFirewallError.NON_STRING_TARGET_FORBIDDEN),
])
def test_execution_mapping_boundary_rejects(targets, expected_code):
    code = assert_execution_mapping_rejects(targets)
    assert code == expected_code


# --- the shared-frozen pool build also rejects illegal raw targets ---------
@pytest.mark.parametrize("bad", ["collect_wood::salt=ab", "target_5", "nope"])
def test_pool_build_rejects_illegal_target(bad):
    with pytest.raises((TargetFirewallError, AchievementError)):
        build_pool("p", [{"task_id": "x", "task_params": dict(_TP),
                          "target_achievements": [bad]}])


def test_firewall_regression_guard_is_not_silent():
    # the regression helper raises AssertionError (not a quiet pass) if a banned
    # target were ever accepted by the mapping boundary.
    assert assert_execution_mapping_rejects(["collect_wood#cafebabe"]) == \
        TargetFirewallError.SALTED_TARGET_FORBIDDEN
