"""GATE 5 — candidate validator rejects illegal / unknown / empty / duplicate
targets and unknown raw keys, fail-closed (NO_SILENT_SCHEMA_COERCION)."""
import pytest
from pydantic import ValidationError

from d052.achievements import AchievementError
from d052.generation.validator import (
    CandidateValidationError,
    canonicalize_candidate,
    validate_target_names,
)
from d052.schemas.candidate import compute_candidate_chash


def raw(task_id="t1", names=("collect_wood",), **tp_over):
    tp = dict(passive_spawn_multiplier=1.0, melee_spawn_multiplier=1.0,
              mob_health_multiplier=1.0, mob_damage_multiplier=1.0)
    tp.update(tp_over)
    return {"task_id": task_id, "task_params": tp,
            "target_achievements": list(names)}


def test_valid_raw_canonicalizes_and_hashes():
    c = canonicalize_candidate(raw("t1", ["defeat_archer", "collect_wood"]))
    assert c.canonical_target_names == ["collect_wood", "defeat_archer"]
    expected = compute_candidate_chash(
        "t1", ["collect_wood", "defeat_archer"], c.task_params.model_dump())
    assert c.chash == expected
    assert len(c.legacy_short_id) == 16


def test_alias_target_resolves():
    c = canonicalize_candidate(raw("t1", ["defeat_orc_soldier"]))  # correct spelling
    assert c.canonical_target_names == ["defeat_orc_solider"]


def test_unknown_raw_key_rejected():
    r = raw()
    r["difficulty_tier"] = 3  # no such field
    with pytest.raises(CandidateValidationError) as ei:
        canonicalize_candidate(r)
    assert ei.value.code == CandidateValidationError.UNKNOWN_RAW_KEY


def test_missing_raw_key_rejected():
    r = raw()
    del r["target_achievements"]
    with pytest.raises(CandidateValidationError) as ei:
        canonicalize_candidate(r)
    assert ei.value.code == CandidateValidationError.MISSING_RAW_KEY


def test_non_mapping_rejected():
    with pytest.raises(CandidateValidationError) as ei:
        canonicalize_candidate(["not", "a", "mapping"])
    assert ei.value.code == CandidateValidationError.INVALID_TYPE


def test_empty_task_id_rejected():
    with pytest.raises(CandidateValidationError) as ei:
        canonicalize_candidate(raw(task_id=""))
    assert ei.value.code == CandidateValidationError.INVALID_TYPE


def test_targets_wrong_type_rejected():
    with pytest.raises(CandidateValidationError) as ei:
        validate_target_names("collect_wood")  # not a list
    assert ei.value.code == CandidateValidationError.INVALID_TYPE


def test_empty_targets_rejected():
    with pytest.raises(AchievementError) as ei:
        canonicalize_candidate(raw(names=[]))
    assert ei.value.code == AchievementError.EMPTY_GOAL_SET


def test_unknown_target_rejected():
    with pytest.raises(AchievementError) as ei:
        canonicalize_candidate(raw(names=["defeat_dragon"]))
    assert ei.value.code == AchievementError.UNKNOWN_ACHIEVEMENT


def test_duplicate_targets_rejected_not_silently_deduped():
    with pytest.raises(CandidateValidationError) as ei:
        canonicalize_candidate(raw(names=["collect_wood", "collect_wood"]))
    assert ei.value.code == CandidateValidationError.DUPLICATE_TARGET


def test_alias_and_canonical_same_target_is_duplicate():
    # both resolve to defeat_orc_solider -> duplicate after canonical resolution
    with pytest.raises(CandidateValidationError) as ei:
        canonicalize_candidate(
            raw(names=["defeat_orc_soldier", "defeat_orc_solider"]))
    assert ei.value.code == CandidateValidationError.DUPLICATE_TARGET


def test_too_many_targets_rejected():
    five = ["collect_wood", "place_table", "eat_cow", "collect_sapling",
            "collect_drink"]
    with pytest.raises(ValidationError) as ei:
        canonicalize_candidate(raw(names=five))
    assert "MAX_TARGETS_EXCEEDED" in str(ei.value)


def test_bad_task_params_rejected():
    with pytest.raises(ValidationError) as ei:
        canonicalize_candidate(raw(passive_spawn_multiplier=0.0))
    assert "NON_POSITIVE" in str(ei.value)
