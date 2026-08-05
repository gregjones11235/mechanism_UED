# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-6): Student and Reference evidence is NEVER mixed into one
success rate.  Each source keeps its own Wilson estimate; the frontier
classification consumes ONLY the Student branches.  These tests pin the
per-source feasibility contract.
"""

import pytest

from dicode.simulator_frontier.branch_search_runner import (
    SEARCH_SOURCE_REFERENCE_POLICY,
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
)
from dicode.simulator_frontier.search_statistics import (
    BranchOutcome,
    estimate_feasibility,
    estimate_feasibility_by_source,
)

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _outcome(branch_id: str, source: str, success: bool, progress: float) -> BranchOutcome:
    return BranchOutcome(
        branch_id=branch_id, state_id="s", search_source=source,
        rng_seed=0, horizon=8, transitions_used=4, success=success,
        progress=progress, terminal_event=None,
        failure_category=None if success else "HORIZON_EXHAUSTED",
        memory_mode="SAVED_POLICY_MEMORY", outcome_hash="a" * 64)


class TestPerSourceFeasibility:
    def test_student_and_reference_never_mixed_into_one_rate(self):
        outcomes = (
            _outcome("s0", SEARCH_SOURCE_STUDENT_DETERMINISTIC, True, 0.8),
            _outcome("s1", SEARCH_SOURCE_STUDENT_STOCHASTIC, False, 0.2),
            _outcome("s2", SEARCH_SOURCE_REFERENCE_POLICY, True, 0.9),
        )
        by_source = estimate_feasibility_by_source(outcomes)
        # Two distinct sources each carry their OWN estimate — Student and
        # Reference are never pooled into a single success rate.
        assert SEARCH_SOURCE_STUDENT_DETERMINISTIC in by_source
        assert SEARCH_SOURCE_REFERENCE_POLICY in by_source
        student_est = by_source[SEARCH_SOURCE_STUDENT_DETERMINISTIC]
        reference_est = by_source[SEARCH_SOURCE_REFERENCE_POLICY]
        assert int(student_est.actual_branches) == 1
        assert float(student_est.success_rate) == 1.0
        assert int(reference_est.actual_branches) == 1
        assert float(reference_est.success_rate) == 1.0

    def test_student_only_aggregate_for_frontier_classification(self):
        student_outcomes = (
            _outcome("s0", SEARCH_SOURCE_STUDENT_DETERMINISTIC, True, 0.8),
            _outcome("s1", SEARCH_SOURCE_STUDENT_STOCHASTIC, False, 0.2),
            _outcome("s2", SEARCH_SOURCE_STUDENT_DETERMINISTIC, False, 0.1),
        )
        estimate = estimate_feasibility(student_outcomes)
        # The aggregate is built ONLY from the Student branches.
        assert int(estimate.actual_branches_by_source[SEARCH_SOURCE_STUDENT_DETERMINISTIC]) == 2
        assert int(estimate.actual_branches_by_source[SEARCH_SOURCE_STUDENT_STOCHASTIC]) == 1
        assert SEARCH_SOURCE_REFERENCE_POLICY not in estimate.actual_branches_by_source
        assert int(estimate.successes) == 1

    def test_empty_source_group_returns_honest_zero_estimate(self):
        estimate = estimate_feasibility([])
        assert int(estimate.total_actual_branches) == 0
        assert int(estimate.successes) == 0
        assert float(estimate.success_rate) == 0.0

    def test_duplicate_branch_raises(self):
        outcomes = (
            _outcome("dup", SEARCH_SOURCE_STUDENT_DETERMINISTIC, True, 0.8),
            _outcome("dup", SEARCH_SOURCE_STUDENT_DETERMINISTIC, True, 0.8),
        )
        with pytest.raises(ValueError):
            estimate_feasibility_by_source(outcomes)
