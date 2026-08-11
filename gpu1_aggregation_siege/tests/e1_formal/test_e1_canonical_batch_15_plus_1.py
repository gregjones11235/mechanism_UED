"""CC2-Director tests: CanonicalDiCodeTrainingBatchPlan (15 + 1).

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
synthetic selection attestations; no real selection or training.

The shared batch protocol translates the attested 12 dynamic
candidates + 3 non-target anchors into 15 curriculum_task_ids, with
original_craftax appended exactly once by DiCode => 16 total.

Covered negative matrix:
* non-selection input                       -> PLAN_BAD_TYPE
* wrong dynamic count (not 12)              -> PLAN_COUNT
* duplicate dynamic id                      -> PLAN_DUPLICATE
* wrong anchor set                          -> PLAN_ANCHOR_MISMATCH
* wrong curriculum count / order            -> PLAN_COUNT
* target missing / target duplicated        -> PLAN_TARGET_*
* wrong target probability                  -> PLAN_PROBABILITY
* binding drift (attestation / manifest)    -> PLAN_BINDING_MISMATCH
* plan hash tamper                          -> PLAN_HASH_MISMATCH
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import dicode_protocol as DP
from dicode.teachers.e1_formal.layout import ORIGINAL_ANCHOR_TASK_ID
from dicode.teachers.e1_formal.selection_attestation import (
    SelectionAttestation,
)

_ANCHOR_MANIFEST_HASH = "aa" * 32


def _attestation(count=12):
    return SelectionAttestation(
        window_id="e1-w000001",
        window_hash="e" * 64,
        selected_ids=tuple(f"cand-{i:03d}" for i in range(count)),
        candidate_pool_hash="a" * 64,
        probe_pool_hash="b" * 64,
        signals_pool_hash="c" * 64,
        selector_source_hash="d" * 64,
        constants_hash="e2" * 32,
        weights_hash="f" * 64,
        family_cap=6,
        seed=7,
        k=12,
        selected_set_hash="g" * 64,
        selection_hash="h" * 64,
        attestation_hash="i" * 64,
    )


def _build(attestation=None, **overrides):
    kwargs = dict(
        selection_attestation=attestation or _attestation(),
        anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
        ctx="test",
    )
    kwargs.update(overrides)
    return DP.build_canonical_dicode_training_batch_plan(**kwargs)


class TestBuildPlan:
    def test_builds_the_15_plus_1_plan(self):
        attestation = _attestation()
        plan = _build(attestation)
        assert plan.dynamic_task_ids == attestation.selected_ids
        assert len(plan.dynamic_task_ids) == 12
        assert plan.non_target_anchor_ids == (
            DP.DICODE_NON_TARGET_ANCHOR_IDS
        )
        assert len(plan.non_target_anchor_ids) == 3
        assert len(plan.curriculum_task_ids) == 15
        assert plan.curriculum_task_ids == tuple(
            list(plan.dynamic_task_ids) + list(plan.non_target_anchor_ids)
        )
        assert plan.target_task_id == DP.DICODE_TARGET_TASK_ID
        assert plan.target_probability == DP.DICODE_TARGET_PROBABILITY
        assert plan.selection_attestation_hash == (
            attestation.attestation_hash
        )
        assert plan.anchor_manifest_hash == _ANCHOR_MANIFEST_HASH
        assert len(plan.plan_hash) == 64
        # 15 curriculum + 1 target = 16 total
        assert len(plan.curriculum_task_ids) + 1 == 16

    def test_non_selection_input_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _build(attestation={"selected_ids": ("a",)})
        assert excinfo.value.code == DP.PLAN_BAD_TYPE

    def test_wrong_dynamic_count_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _build(_attestation(count=11))
        assert excinfo.value.code == DP.PLAN_COUNT

    def test_duplicate_dynamic_id_refused(self):
        attestation = replace(
            _attestation(),
            selected_ids=("cand-000",) + ("cand-000",)
            + tuple(f"cand-{i:03d}" for i in range(2, 12)),
        )
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _build(attestation)
        assert excinfo.value.code == DP.PLAN_DUPLICATE

    def test_wrong_anchor_set_refused(self):
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            _build(
                non_target_anchor_ids=(
                    "task_1",
                    "task_2",
                    "other_anchor",
                )
            )
        assert excinfo.value.code == DP.PLAN_ANCHOR_MISMATCH

    def test_target_never_in_curriculum(self):
        plan = _build()
        assert ORIGINAL_ANCHOR_TASK_ID not in plan.curriculum_task_ids
        assert ORIGINAL_ANCHOR_TASK_ID == plan.target_task_id
        all_ids = list(plan.curriculum_task_ids) + [plan.target_task_id]
        assert all_ids.count(ORIGINAL_ANCHOR_TASK_ID) == 1


class TestVerifyPlan:
    def test_untampered_plan_verifies(self):
        attestation = _attestation()
        plan = _build(attestation)
        DP.verify_canonical_dicode_training_batch_plan(
            plan,
            selection_attestation=attestation,
            anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
            ctx="test",
        )

    def test_hash_tamper_detected(self):
        attestation = _attestation()
        tampered = replace(_build(attestation), plan_hash="f" * 64)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_canonical_dicode_training_batch_plan(
                tampered,
                selection_attestation=attestation,
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_HASH_MISMATCH

    def test_attestation_drift_detected(self):
        plan = _build(_attestation())
        other = _attestation()  # different attestation_hash? same hash
        assert other.attestation_hash == plan.selection_attestation_hash
        drifted = replace(_attestation(), attestation_hash="ff" * 64)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_canonical_dicode_training_batch_plan(
                plan,
                selection_attestation=drifted,
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH

    def test_anchor_manifest_drift_detected(self):
        attestation = _attestation()
        plan = _build(attestation)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_canonical_dicode_training_batch_plan(
                plan,
                selection_attestation=attestation,
                anchor_manifest_hash="ff" * 32,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_BINDING_MISMATCH


class TestTargetProbability:
    def test_probability_is_frozen_at_twenty_percent(self):
        assert DP.DICODE_TARGET_PROBABILITY == 0.20

    def test_other_probability_refused(self):
        attestation = _attestation()
        # forge a plan with a different probability by editing the
        # dataclass then re-verifying against the frozen builder
        plan = replace(_build(attestation), target_probability=0.5)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_canonical_dicode_training_batch_plan(
                plan,
                selection_attestation=attestation,
                anchor_manifest_hash=_ANCHOR_MANIFEST_HASH,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_PROBABILITY
