"""CC2-Director tests: target probability stays 0.20.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE.

``conf/dicode_manager/default.yaml`` freezes ``original_task_proportion:
0.2``. The shared plan must carry EXACTLY 0.20 for the OriginalTask
(the remaining 0.80 is shared by the 15 curriculum ids); any other
probability fails closed.
"""
import pytest

from dicode.teachers.e1_formal import dicode_protocol as DP
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


class TestTargetProbabilityPreserved:
    def test_frozen_value_matches_the_dicode_config(self):
        # conf/dicode_manager/default.yaml: original_task_proportion: 0.2
        assert DP.DICODE_TARGET_PROBABILITY == 0.20

    def test_plan_carries_exactly_twenty_percent(self):
        plan = DP.build_canonical_dicode_training_batch_plan(
            selection_attestation=_attestation(),
            anchor_manifest_hash="aa" * 32,
            ctx="test",
        )
        assert plan.target_probability == 0.20
        # the remaining 0.80 is shared by the 15 curriculum ids
        assert plan.target_probability + 0.80 == 1.0

    def test_probability_is_committed_to_the_plan_payload(self):
        from dataclasses import replace

        attestation = _attestation()
        plan = DP.build_canonical_dicode_training_batch_plan(
            selection_attestation=attestation,
            anchor_manifest_hash="aa" * 32,
            ctx="test",
        )
        # a forged probability is refused on re-verification even
        # though the plan_hash field itself is unchanged (replace does
        # not recompute) — the shape gate catches the drift
        forged = replace(plan, target_probability=0.5)
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_canonical_dicode_training_batch_plan(
                forged,
                selection_attestation=attestation,
                anchor_manifest_hash="aa" * 32,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_PROBABILITY

    def test_non_numeric_probability_refused(self):
        from dataclasses import replace

        attestation = _attestation()
        plan = DP.build_canonical_dicode_training_batch_plan(
            selection_attestation=attestation,
            anchor_manifest_hash="aa" * 32,
            ctx="test",
        )
        forged = replace(plan, target_probability="0.20")
        with pytest.raises(DP.DiCodePlanError) as excinfo:
            DP.verify_canonical_dicode_training_batch_plan(
                forged,
                selection_attestation=attestation,
                anchor_manifest_hash="aa" * 32,
                ctx="test",
            )
        assert excinfo.value.code == DP.PLAN_PROBABILITY
