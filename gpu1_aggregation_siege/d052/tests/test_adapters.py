"""GATE 13 — authorization-gated training + evaluation adapters (no-op this phase).

Training: a no-training authorization yields 0 timesteps; a training-scope
authorization REFUSES (not implemented) rather than silently running; the frozen
no-training labels hold. Evaluation: a deterministic held-out plan with NO results,
NO_RAW_DATA_NO_STRONG_CLAIM enforced, RESULTS_REUSABILITY=ENGINEERING_ONLY until
raw data is attached.
"""
import pytest

from d052.cells import (
    SCOPE_NO_TRAINING,
    SCOPE_TRAINING,
    CellRecord,
    make_authorization,
)
from d052.cells.spec import CellSpec
from d052.evaluation import (
    EvaluationAdapterError,
    assert_no_strong_claim,
    attach_results,
    build_evaluation_plan,
)
from d052.execution import (
    build_execution_certificate,
    canonical_compiled_spec,
)
from d052.generation import canonicalize_candidate
from d052.schemas.selector import SelectorConfig, SelectorType
from d052.training import (
    TrainingAdapterError,
    assert_no_training_phase,
    canonical_training_runner,
)

_TP = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
       "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}


def _record(scope):
    spec = CellSpec(
        cell_id="c1", protocol_version="canonical_v2", hypothesis="h",
        pool_id="p", pool_hash="a" * 64,
        selector=SelectorConfig(selector=SelectorType.S1_THREE_ROLE, k=2, seed=7,
                                roles=["tutor", "critic", "explorer"]),
        candidate_ids=["t1"], selection_hash="b" * 64,
        intended_total_timesteps=4096, output_dir="runs/c1", created_by="CC3")
    auth = make_authorization("c1", spec.identity_hash(), "human", scope,
                              spec.intended_total_timesteps)
    return CellRecord(spec=spec, authorization=auth)


def _cert():
    c = canonicalize_candidate(
        {"task_id": "t1", "task_params": dict(_TP),
         "target_achievements": ["collect_wood", "eat_cow"]})
    return build_execution_certificate(c, canonical_compiled_spec(c, "train-1"))


# --- training adapter -------------------------------------------------------

def test_no_training_scope_yields_zero_timesteps():
    art = canonical_training_runner(_record(SCOPE_NO_TRAINING))
    assert art["timesteps_run"] == 0
    assert art["trained"] is False


def test_training_scope_refuses_not_implemented():
    with pytest.raises(TrainingAdapterError) as ei:
        canonical_training_runner(_record(SCOPE_TRAINING))
    assert ei.value.code == TrainingAdapterError.NOT_IMPLEMENTED


def test_absent_authorization_is_no_op():
    spec = _record(SCOPE_NO_TRAINING).spec
    art = canonical_training_runner(CellRecord(spec=spec))  # no authorization
    assert art["timesteps_run"] == 0


def test_frozen_no_training_labels():
    labels = assert_no_training_phase()
    assert labels["D052_LONG_TRAINING_RUNS"] == 0
    assert labels["D052_4096_SMOKE_AUTHORIZED"] is False
    assert labels["D052_24576_AUTHORIZED"] is False
    assert labels["D052_98304_AUTHORIZED"] is False
    assert labels["NO_UNAUTHORIZED_TRAINING"] is True


# --- evaluation adapter -----------------------------------------------------

def test_evaluation_plan_is_deterministic_and_result_free():
    p1 = build_evaluation_plan(_cert())
    p2 = build_evaluation_plan(_cert())
    assert p1 == p2
    assert p1["results"] is None
    assert p1["strong_claim_permitted"] is False
    assert p1["RESULTS_REUSABILITY"] == "ENGINEERING_ONLY"
    assert p1["metric"] == "success_rate"
    assert p1["num_achievements"] == 2
    assert p1["canonical_names"] == ["collect_wood", "eat_cow"]


def test_no_strong_claim_without_raw_data():
    plan = build_evaluation_plan(_cert())
    assert_no_strong_claim(plan)  # OK: no claim made
    bad = dict(plan)
    bad["strong_claim_permitted"] = True   # claim without raw data
    with pytest.raises(EvaluationAdapterError) as ei:
        assert_no_strong_claim(bad)
    assert ei.value.code == EvaluationAdapterError.NO_RAW_DATA_NO_STRONG_CLAIM


def test_attach_results_requires_nonempty_raw_data():
    plan = build_evaluation_plan(_cert())
    with pytest.raises(EvaluationAdapterError):
        attach_results(plan, {})   # empty raw data refused
    with_raw = attach_results(plan, {"collect_wood": {"success_rate": 0.8}})
    assert with_raw["strong_claim_permitted"] is True
    assert with_raw["RESULTS_REUSABILITY"] == "SCIENTIFIC_WITH_EVIDENCE"
    assert_no_strong_claim(with_raw)   # now consistent (raw data present)
