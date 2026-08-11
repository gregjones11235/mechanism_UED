"""§八 (director smoke handoff): the Canonical DiCode 15+1 batch plan.

Contract under test:

* the window final selection (12 dynamic + 4 anchors) converts to 12
  dynamic + 3 non-target anchors = 15 curriculum task ids;
* ``batch_candidate_ids`` is EXACTLY the 15 curriculum ids — the
  OriginalTask NEVER enters it and is appended ONCE internally by DiCode;
* total = 16, original_task_proportion = 0.20, the remaining 15 share
  0.80;
* the plan is immutable + content-hashed (deterministic), and every
  structural violation fails closed (dynamic/anchor count, original-in-
  batch, original-duplicated, total count, proportion).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.dicode_batch_plan import (
    CanonicalDiCodeTrainingBatchPlan,
    build_dicode_batch_plan,
)

DYNAMIC = [f"dyn-{i:02d}" for i in range(12)]
ANCHORS = list(C.GLOBAL_CANONICAL_ANCHOR_IDS)
NON_TARGET = ANCHORS[:3]
ORIGINAL = "DICODE_ORIGINAL_TASK_V1"
RUNTIME = "TEST_ONLY_DICODE_ONE_UPDATE_RUNTIME"
FINAL_BATCH = DYNAMIC + ANCHORS


def plan(**over):
    kwargs = dict(window=1, final_batch_ids=FINAL_BATCH, anchor_ids=ANCHORS,
                  non_target_anchor_ids=NON_TARGET,
                  original_task_id=ORIGINAL,
                  original_appended_by=RUNTIME)
    kwargs.update(over)
    return build_dicode_batch_plan(**kwargs)


class TestFifteenPlusOneStructure:
    def test_structure(self):
        p = plan()
        assert len(p.dynamic_task_ids) == 12
        assert len(p.non_target_anchor_ids) == 3
        assert len(p.curriculum_task_ids) == 15
        assert p.total_task_count == 16
        assert p.original_task_id == ORIGINAL
        assert abs(p.original_task_proportion - 0.20) < 1e-9
        assert p.original_appended_by == RUNTIME

    def test_batch_candidate_ids_are_the_15_curriculum_only(self):
        p = plan()
        assert p.batch_candidate_ids == p.curriculum_task_ids
        assert len(p.batch_candidate_ids) == 15
        assert ORIGINAL not in p.batch_candidate_ids
        assert ORIGINAL not in p.dynamic_task_ids
        assert ORIGINAL not in p.non_target_anchor_ids

    def test_all_15_curriculum_tasks_are_unique(self):
        p = plan()
        assert len(set(p.curriculum_task_ids)) == 15
        assert len(set(p.batch_candidate_ids)) == 15

    def test_plan_is_content_hashed_and_deterministic(self):
        assert len(plan().plan_hash) == 64
        assert plan().plan_hash == plan().plan_hash

    def test_plan_is_immutable(self):
        p = plan()
        with pytest.raises(ValidationError, match="frozen"):
            p.window = 2


class TestFailClosedLadders:
    def test_wrong_dynamic_count_rejected(self):
        with pytest.raises(ValueError,
                           match="DICODE_DYNAMIC_COUNT_MISMATCH"):
            plan(final_batch_ids=DYNAMIC[:11] + ANCHORS)

    def test_non_target_anchor_not_in_manifest_rejected(self):
        with pytest.raises(ValueError,
                           match="DICODE_NON_TARGET_ANCHOR_NOT_IN_MANIFEST"):
            plan(non_target_anchor_ids=["GLOBAL_ANCHOR_FAKE"] + NON_TARGET)

    def test_original_in_batch_rejected(self):
        #: direct construction with the OriginalTask inside the curriculum
        #: (counts still 12+3=15) is refused (DICODE_ORIGINAL_IN_BATCH)
        payload = dict(window=1,
                       dynamic_task_ids=list(DYNAMIC[:11]) + [ORIGINAL],
                       non_target_anchor_ids=list(NON_TARGET),
                       curriculum_task_ids=(list(DYNAMIC[:11]) + [ORIGINAL]
                                            + list(NON_TARGET)),
                       original_task_id=ORIGINAL,
                       original_task_proportion=0.20, total_task_count=16,
                       original_appended_by=RUNTIME, plan_hash="x" * 64)
        with pytest.raises(ValueError, match="DICODE_ORIGINAL_IN_BATCH"):
            CanonicalDiCodeTrainingBatchPlan(**payload)

    def test_original_duplicated_rejected(self):
        #: the OriginalTask duplicated inside dynamic (while the curriculum
        #: keeps 12 regular + 3 anchors, so the in-batch check passes) is
        #: refused (DICODE_ORIGINAL_DUPLICATED)
        payload = dict(window=1,
                       dynamic_task_ids=list(DYNAMIC[:11]) + [ORIGINAL],
                       non_target_anchor_ids=list(NON_TARGET),
                       curriculum_task_ids=list(DYNAMIC) + list(NON_TARGET),
                       original_task_id=ORIGINAL,
                       original_task_proportion=0.20, total_task_count=16,
                       original_appended_by=RUNTIME, plan_hash="x" * 64)
        with pytest.raises(ValueError, match="DICODE_ORIGINAL_DUPLICATED"):
            CanonicalDiCodeTrainingBatchPlan(**payload)

    def test_wrong_total_count_rejected(self):
        payload = dict(window=1, dynamic_task_ids=list(DYNAMIC),
                       non_target_anchor_ids=list(NON_TARGET),
                       curriculum_task_ids=list(DYNAMIC) + list(NON_TARGET),
                       original_task_id=ORIGINAL,
                       original_task_proportion=0.20, total_task_count=17,
                       original_appended_by=RUNTIME, plan_hash="x" * 64)
        with pytest.raises(ValueError, match="DICODE_TOTAL_COUNT_MISMATCH"):
            CanonicalDiCodeTrainingBatchPlan(**payload)

    def test_wrong_proportion_rejected(self):
        payload = dict(window=1, dynamic_task_ids=list(DYNAMIC),
                       non_target_anchor_ids=list(NON_TARGET),
                       curriculum_task_ids=list(DYNAMIC) + list(NON_TARGET),
                       original_task_id=ORIGINAL,
                       original_task_proportion=0.5, total_task_count=16,
                       original_appended_by=RUNTIME, plan_hash="x" * 64)
        with pytest.raises(ValueError,
                           match="DICODE_ORIGINAL_PROPORTION_MISMATCH"):
            CanonicalDiCodeTrainingBatchPlan(**payload)

    def test_duplicate_curriculum_task_rejected(self):
        payload = dict(window=1, dynamic_task_ids=list(DYNAMIC),
                       non_target_anchor_ids=list(NON_TARGET),
                       curriculum_task_ids=(list(DYNAMIC[:11])
                                            + [DYNAMIC[0]]
                                            + list(NON_TARGET)),
                       original_task_id=ORIGINAL,
                       original_task_proportion=0.20, total_task_count=16,
                       original_appended_by=RUNTIME, plan_hash="x" * 64)
        with pytest.raises(ValueError,
                           match="DICODE_DUPLICATE_CURRICULUM_TASK"):
            CanonicalDiCodeTrainingBatchPlan(**payload)


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
