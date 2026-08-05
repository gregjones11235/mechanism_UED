"""§八 (director smoke handoff): the OriginalTask is appended ONCE — never
duplicated, never in batch_candidate_ids.

Contract under test:

* the director bundle declares the OriginalTask separately from the
  curriculum anchors; the plan never lists it among the 15 curriculum
  ids;
* across repeated windows (a plan per window) the OriginalTask appears
  exactly once per plan (as ``original_task_id``) and never in any
  dynamic / non-target-anchor / curriculum list;
* the model and the bundle both refuse an OriginalTask that collides with
  a curriculum task.

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

from e2_test_sign_helpers import (
    valid_director_bundle,
    valid_director_bundle_payload,
    sign_director_runtime_bundle,
)

DYNAMIC = [f"dyn-{i:02d}" for i in range(12)]
ANCHORS = list(C.GLOBAL_CANONICAL_ANCHOR_IDS)
NON_TARGET = ANCHORS[:3]
ORIGINAL = "DICODE_ORIGINAL_TASK_V1"
RUNTIME = "TEST_ONLY_DICODE_ONE_UPDATE_RUNTIME"


def _plan(window):
    return build_dicode_batch_plan(
        window=window, final_batch_ids=DYNAMIC + ANCHORS, anchor_ids=ANCHORS,
        non_target_anchor_ids=NON_TARGET, original_task_id=ORIGINAL,
        original_appended_by=RUNTIME)


class TestOriginalTaskUniqueness:
    def test_original_never_in_any_curriculum_list(self):
        for window in range(3):
            p = _plan(window)
            assert p.original_task_id == ORIGINAL
            assert ORIGINAL not in p.batch_candidate_ids
            assert ORIGINAL not in p.dynamic_task_ids
            assert ORIGINAL not in p.non_target_anchor_ids
            assert ORIGINAL not in p.curriculum_task_ids
            #: exactly one OriginalTask per plan (declared, not listed)
            assert p.curriculum_task_ids.count(ORIGINAL) == 0

    def test_total_is_15_plus_1(self):
        for window in range(3):
            p = _plan(window)
            assert len(p.curriculum_task_ids) + 1 == p.total_task_count == 16

    def test_bundle_declares_original_apart_from_anchors(self):
        manifest = valid_director_bundle()
        bb = manifest.batch_binding
        assert bb.original_task_id not in bb.non_target_anchor_ids
        assert len(set(bb.non_target_anchor_ids)) \
            == len(bb.non_target_anchor_ids)

    def test_bundle_with_original_as_anchor_refused(self):
        #: the director bundle refuses an OriginalTask that is one of the
        #: curriculum anchors (it must be appended once, internally)
        payload = valid_director_bundle_payload(
            original_task_id=C.GLOBAL_CANONICAL_ANCHOR_IDS[0])
        with pytest.raises(ValidationError,
                           match="DICODE_ORIGINAL_IS_A_CURRICULUM_TASK"):
            sign_director_runtime_bundle(payload)


class TestPosture:
    def test_no_capability_flag_flips(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
