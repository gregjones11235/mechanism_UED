"""CC1 audit fix1 §5/§6/§7 — TrajectorySupervisionGuard hardening tests.

§5 serialized-string parsing: a string that looks like JSON after trim is
   parsed and the FULL guard re-runs inside, under MAX_SERIALIZED_* limits;
   parse failure falls back to plain-text NL patterns (never a lenient
   skip); limit excess fails closed.
§6 alias hardening: renames of forbidden concepts rejected after
   normalization (casefold + separator stripping), incl. nested scans.
§7 natural-language action-advice: bilingual imperative advice rejected;
   third-person behavior DESCRIPTIONS must NOT false-reject.
"""
from __future__ import annotations

import json

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.formal_evaluation_leakage_guard import (
    FormalEvaluationLeakageGuard,
    FormalLeakageViolation,
)
from d052.bagr_ued.trajectory_supervision_guard import (
    GuardViolation,
    TrajectorySupervisionGuard,
)

GUARD = TrajectorySupervisionGuard()


def _codes(report):
    return {f["code"] for f in report["findings"]}


# ===========================================================================
# §5 — serialized-string parsing
# ===========================================================================

def test_serialized_forbidden_key_is_rejected():
    r = GUARD.scan('{"suggested_actions": ["wait here"]}')
    assert r["passed"] is False
    assert GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN in _codes(r)


def test_serialized_reward_delta_is_rejected():
    r = GUARD.scan('{"reward_delta": 1.0}')
    assert r["passed"] is False
    assert GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN in _codes(r)


def test_serialized_advice_inside_json_value_is_rejected():
    # advice smuggled as a VALUE inside serialized JSON: the parsed walk
    # re-runs the NL patterns inside the structure
    r = GUARD.scan('{"note": "please go left at the fork"}')
    assert r["passed"] is False
    assert GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN in _codes(r)


def test_serialized_nested_under_depth_limit_is_rejected():
    s = '{"a": {"b": {"c": {"reward_shaping": {}}}}}'  # depth 4 <= 12
    r = GUARD.scan(s)
    assert r["passed"] is False
    assert GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN in _codes(r)


def test_double_encoded_serialization_cannot_evade():
    s = json.dumps(json.dumps({"reward_delta": 1.0}))  # string-in-string
    r = GUARD.scan(s)
    assert r["passed"] is False
    assert GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN in _codes(r)


def test_serialized_over_depth_limit_fails_closed():
    inner = '"leaf"'
    for _ in range(C.MAX_SERIALIZED_PARSE_DEPTH + 2):  # 14 levels > 12
        inner = '{"k": ' + inner + "}"
    r = GUARD.scan(inner)
    assert r["passed"] is False
    assert GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED in _codes(r)


def test_serialized_over_length_limit_fails_closed():
    s = '{"pad": "' + "a" * (C.MAX_SERIALIZED_STRING_LENGTH + 1) + '"}'
    r = GUARD.scan(s)
    assert r["passed"] is False
    assert GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED in _codes(r)


def test_serialized_over_container_items_fails_closed():
    items = ",".join(["0"] * (C.MAX_SERIALIZED_CONTAINER_ITEMS + 1))
    r = GUARD.scan("[" + items + "]")
    assert r["passed"] is False
    assert GuardViolation.SERIALIZED_GUARD_LIMIT_EXCEEDED in _codes(r)


def test_invalid_json_is_not_leniently_skipped():
    # looks JSON-ish but fails to parse -> still plain-text pattern-scanned
    assert GUARD.scan("{this is not json at all")["passed"] is True
    r = GUARD.scan("{don't sleep near the monster")
    assert r["passed"] is False
    assert GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN in _codes(r)


def test_clean_serialized_role_output_passes():
    payload = json.dumps({
        "behavior_findings": [{"severity": 0.9, "recurrence": 4,
                               "note": "the student repeatedly rests while a "
                                       "hostile is near"}]})
    assert GUARD.scan(payload)["passed"] is True


# ===========================================================================
# §6 — alias hardening (after normalization)
# ===========================================================================

@pytest.mark.parametrize("alias", sorted(C.FORBIDDEN_SUPERVISION_KEY_ALIASES))
def test_every_alias_key_is_rejected(alias):
    r = GUARD.scan({"context": {alias: "anything"}})
    assert r["passed"] is False, f"alias {alias!r} slipped through"
    assert GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN in _codes(r)


@pytest.mark.parametrize("spelling", [
    "Suggested-Actions", "RECOMMENDED ACTION", "suggested_actions",
    "Navigation_Route", "expert  plan", "State-Payload", "RewardDelta",
])
def test_alias_normalization_catches_case_and_separators(spelling):
    r = GUARD.scan({spelling: 1})
    assert r["passed"] is False, f"spelling {spelling!r} slipped through"


def test_aliases_reach_nested_sequences_and_mappings():
    r = GUARD.scan({"envelopes": [{"parsed_json": {"route": ["go", "here"]}}]})
    assert r["passed"] is False
    assert GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN in _codes(r)


@pytest.mark.parametrize("payload_key", [
    "bank_blob", "formal_state_blob", "formal_state_payload", "state_payload",
])
def test_payload_aliases_rejected_by_both_guards(payload_key):
    assert GUARD.scan({payload_key: "x"})["passed"] is False
    leakage = FormalEvaluationLeakageGuard().scan({payload_key: "x"})
    assert leakage["passed"] is False
    assert leakage["findings"][0]["code"] == \
        FormalLeakageViolation.FORBIDDEN_PROVENANCE_KEY


def test_direct_action_advice_never_reaches_reconciler_or_selector():
    # the board-level contract: a parsed role output carrying an alias key
    # raises before reconciliation/selection can consume it
    role_output = {"hypotheses": [], "recommended_move": "north"}
    with pytest.raises(Exception) as ei:
        GUARD.assert_clean(role_output, label="role_output")
    assert ei.value.code == \
        GuardViolation.TRAJECTORY_SUPERVISION_KEY_FORBIDDEN


# ===========================================================================
# §7 — natural-language action advice: REJECTIONS
# ===========================================================================

@pytest.mark.parametrize("text", [
    "go left",
    "head north",
    "attack the monster",
    "don't sleep",
    "you should flee",
    "move toward the ladder",
    "向左走",
    "往北走",
    "攻击怪物",
    "不要睡觉",
    "你应该逃跑",
    "朝梯子移动",
])
def test_imperative_action_advice_is_rejected(text):
    r = GUARD.scan({"free_text": text})
    assert r["passed"] is False, f"advice {text!r} slipped through"
    assert GuardViolation.DIRECT_ACTION_ADVICE_FORBIDDEN in _codes(r)


# ===========================================================================
# §7 — behavior DESCRIPTIONS must NOT false-reject
# ===========================================================================

@pytest.mark.parametrize("text", [
    "The student repeatedly attacks without effect.",
    "The trace contains a sleep action near a hostile.",
    "智能体重复攻击但没有效果。",
    "Rest frequency should respond to graded threat distance.",
    "The episode shows oscillation between two corridors.",
    "Death occurred after an unsafe rest window.",
    "学生在危险区域附近休息后死亡。",
])
def test_behavior_descriptions_are_allowed(text):
    r = GUARD.scan({"free_text": text})
    assert r["passed"] is True, (
        f"false positive on behavior description {text!r}: "
        f"{r['findings']}")
