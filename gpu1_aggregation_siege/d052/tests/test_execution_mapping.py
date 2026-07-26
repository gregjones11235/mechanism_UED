"""GATE 10 — candidate -> real-training-goal execution mapping certificate.

A conforming compiled spec yields executed_as_intended=True with the full
canonical chain (ids==indices, dim 67, ones==#targets, obs 8335,
conditioning achievement_multi_hot). ANY deviation (wrong obs_dim like the banned
8300/32-slot, wrong conditioning, wrong goal vector, silent fallback, no compiled
spec) fails the relevant gate and forces executed_as_intended=False — never a
silent pass. The schema itself forbids executed_as_intended=True with a failed gate.
"""
import pytest

from d052.execution import (
    REQUIRED_GATES,
    CompiledTaskSpec,
    ExecutionMappingCertificate,
    ExecutionMappingError,
    build_execution_certificate,
    candidate_goal_vector,
    canonical_compiled_spec,
)
from d052.generation import canonicalize_candidate

_TP = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
       "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}


def _cand(targets=("collect_wood",), tid="t1"):
    return canonicalize_candidate(
        {"task_id": tid, "task_params": dict(_TP),
         "target_achievements": list(targets)})


# --- conforming mapping -----------------------------------------------------

def test_conforming_spec_executed_as_intended():
    c = _cand()
    cert = build_execution_certificate(c, canonical_compiled_spec(c, "train-1"))
    assert cert.executed_as_intended is True
    assert set(cert.gates) == set(REQUIRED_GATES)
    assert all(cert.gates.values())
    assert cert.canonical_ids == cert.goal_vector_indices
    assert cert.goal_vector_dim == 67
    assert cert.conditioning_dimension == 67
    assert cert.goal_vector_ones == 1
    assert cert.student_obs_dim == 8335
    assert cert.conditioning_type == "achievement_multi_hot"
    assert cert.training_task_id == "train-1"


def test_multi_target_chain_is_sorted_and_aligned():
    c = _cand(targets=["eat_cow", "collect_wood"])
    cert = build_execution_certificate(c, canonical_compiled_spec(c, "train-2"))
    assert cert.goal_vector_ones == 2
    assert cert.canonical_ids == sorted(cert.canonical_ids)
    assert cert.canonical_ids == cert.goal_vector_indices
    assert cert.executed_as_intended is True


def test_canonical_compiled_spec_is_deterministic():
    c = _cand()
    s1 = canonical_compiled_spec(c, "train-1")
    s2 = canonical_compiled_spec(c, "train-1")
    assert s1.model_dump() == s2.model_dump()
    cert1 = build_execution_certificate(c, s1)
    cert2 = build_execution_certificate(c, s2)
    assert cert1.model_dump() == cert2.model_dump()


def test_candidate_goal_vector_is_67_multihot():
    c = _cand(targets=["collect_wood", "eat_cow"])
    vec = candidate_goal_vector(c)
    assert len(vec) == 67
    assert sum(vec) == 2
    assert set(vec) <= {0.0, 1.0}


# --- deviations force executed_as_intended=False ---------------------------

def test_banned_obs_dim_8300_fails_gate():
    c = _cand()
    spec = CompiledTaskSpec(
        training_task_id="x", task_spec_hash="a" * 64, student_obs_dim=8300,
        conditioning_type="achievement_multi_hot", goal_vector_dim=67,
        compiled_goal_vector=candidate_goal_vector(c))
    cert = build_execution_certificate(c, spec)
    assert cert.gates["student_obs_dim_8335"] is False
    assert cert.executed_as_intended is False


def test_banned_32_slot_goal_vector_fails_gates():
    c = _cand()
    spec = CompiledTaskSpec(
        training_task_id="x", task_spec_hash="a" * 64, student_obs_dim=8335,
        conditioning_type="achievement_multi_hot", goal_vector_dim=32,
        compiled_goal_vector=[0.0] * 32)
    cert = build_execution_certificate(c, spec)
    assert cert.gates["goal_vector_dim_67"] is False
    assert cert.gates["goal_vector_index_aligned"] is False
    assert cert.gates["no_silent_fallback"] is False
    assert cert.executed_as_intended is False


def test_wrong_goal_vector_indices_fail_alignment():
    c = _cand()
    bad_vec = [0.0] * 67          # all-zero: ones in the wrong (no) places
    spec = CompiledTaskSpec(
        training_task_id="x", task_spec_hash="a" * 64, student_obs_dim=8335,
        conditioning_type="achievement_multi_hot", goal_vector_dim=67,
        compiled_goal_vector=bad_vec)
    cert = build_execution_certificate(c, spec)
    assert cert.gates["goal_vector_index_aligned"] is False
    assert cert.executed_as_intended is False


def test_legacy_one_hot_conditioning_fails_no_silent_fallback():
    c = _cand()
    spec = CompiledTaskSpec(
        training_task_id="x", task_spec_hash="a" * 64, student_obs_dim=8335,
        conditioning_type="one_hot", goal_vector_dim=67,
        compiled_goal_vector=candidate_goal_vector(c))
    cert = build_execution_certificate(c, spec)
    assert cert.gates["no_silent_fallback"] is False
    assert cert.executed_as_intended is False


def test_flagged_silent_fallback_fails_gate():
    c = _cand()
    spec = canonical_compiled_spec(c, "train-1")
    cert = build_execution_certificate(c, spec, silent_fallback_occurred=True)
    assert cert.gates["no_silent_fallback"] is False
    assert cert.executed_as_intended is False


# --- hard failures (no certificate at all) ---------------------------------

def test_no_spec_is_task_not_compiled():
    c = _cand()
    with pytest.raises(ExecutionMappingError) as ei:
        build_execution_certificate(c, None)
    assert ei.value.code == ExecutionMappingError.TASK_NOT_COMPILED


def test_uncompiled_spec_is_task_not_compiled():
    c = _cand()
    spec = canonical_compiled_spec(c, "train-1").model_copy(update={"compiled": False})
    with pytest.raises(ExecutionMappingError) as ei:
        build_execution_certificate(c, spec)
    assert ei.value.code == ExecutionMappingError.TASK_NOT_COMPILED


def test_tampered_candidate_chash_hard_fails():
    c = _cand()
    bad = c.model_copy(update={"chash": "0" * 64})  # bypass revalidation
    spec = canonical_compiled_spec(c, "train-1")
    with pytest.raises(ExecutionMappingError) as ei:
        build_execution_certificate(bad, spec)
    assert ei.value.code == ExecutionMappingError.CHASH_MISMATCH


def test_non_multihot_compiled_vector_rejected():
    with pytest.raises(Exception):
        CompiledTaskSpec(
            training_task_id="x", task_spec_hash="a" * 64, student_obs_dim=8335,
            conditioning_type="achievement_multi_hot", goal_vector_dim=2,
            compiled_goal_vector=[0.5, 0.5])   # not 0/1


# --- schema-level NO_RAW_DATA_NO_STRONG_CLAIM ------------------------------

def test_schema_forbids_success_with_failed_gate():
    c = _cand()
    cert = build_execution_certificate(c, canonical_compiled_spec(c, "train-1"))
    d = cert.model_dump()
    d["gates"]["task_compiled"] = False        # a failed gate...
    d["executed_as_intended"] = True           # ...yet claiming success
    with pytest.raises(Exception):
        ExecutionMappingCertificate(**d)
