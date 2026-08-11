"""CC2-Director tests: the OriginalTask appears EXACTLY ONCE.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

DiCode's ``run_session_training`` appends the OriginalTask internally;
方向一 must pass only the 15 curriculum ids as ``sampled_task_ids`` —
original_craftax is never a sampled/curriculum id and never appears
twice in the session.
"""
import pytest

from dicode.teachers.e1_formal import dicode_protocol as DP
from dicode.teachers.e1_formal.layout import ORIGINAL_ANCHOR_TASK_ID
from dicode.teachers.e1_formal.selection_attestation import (
    SelectionAttestation,
)


def _attestation():
    return SelectionAttestation(
        window_id="e1-w000001",
        window_hash="e" * 64,
        selected_ids=tuple(f"cand-{i:03d}" for i in range(12)),
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


def _plan():
    return DP.build_canonical_dicode_training_batch_plan(
        selection_attestation=_attestation(),
        anchor_manifest_hash="aa" * 32,
        ctx="test",
    )


class TestOriginalTaskAppearsOnce:
    def test_original_not_in_curriculum(self):
        plan = _plan()
        assert ORIGINAL_ANCHOR_TASK_ID not in plan.curriculum_task_ids
        assert ORIGINAL_ANCHOR_TASK_ID not in plan.dynamic_task_ids
        assert ORIGINAL_ANCHOR_TASK_ID not in plan.non_target_anchor_ids

    def test_original_appears_exactly_once_overall(self):
        plan = _plan()
        all_ids = list(plan.curriculum_task_ids) + [plan.target_task_id]
        assert all_ids.count(ORIGINAL_ANCHOR_TASK_ID) == 1
        assert plan.target_task_id == ORIGINAL_ANCHOR_TASK_ID

    def test_original_never_in_sampled_task_ids(self):
        # the 15 curriculum ids are exactly what is passed to
        # run_session_training(sampled_task_ids=...)
        sampled = list(_plan().curriculum_task_ids)
        assert len(sampled) == 15
        assert ORIGINAL_ANCHOR_TASK_ID not in sampled
        # ...and the session then has exactly 15 + 1 = 16 tasks
        assert len(sampled) + 1 == 16

    def test_duplicate_target_in_curriculum_fails_closed(self):
        # the builder can never produce a duplicated target (the
        # invariant holds by construction); a forged plan that smuggles
        # the target into the 15-id curriculum violates the shape and
        # is refused before it could ever reach DiCode
        plan = _plan()
        smuggled = tuple(
            list(plan.curriculum_task_ids[:14]) + [ORIGINAL_ANCHOR_TASK_ID]
        )
        assert ORIGINAL_ANCHOR_TASK_ID in smuggled
        assert len(smuggled) == 15
        with pytest.raises(DP.DiCodePlanError):
            DP._validate_shape(
                dynamic_task_ids=plan.dynamic_task_ids,
                non_target_anchor_ids=plan.non_target_anchor_ids,
                curriculum_task_ids=smuggled,
                target_task_id=plan.target_task_id,
                target_probability=plan.target_probability,
                ctx="test",
            )
