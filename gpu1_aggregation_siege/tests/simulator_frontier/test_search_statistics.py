import pytest

from dicode.simulator_frontier.search_statistics import BranchOutcome, estimate_feasibility


def _row(i, success=False):
    return BranchOutcome(str(i), "state", "rollout", 0, 8, 8, success, float(i), None, None, "ZERO_MEMORY", f"h{i}")


def test_actual_n_statistics_and_wilson():
    estimate = estimate_feasibility([_row(0), _row(1, True), _row(2, True)])
    assert estimate.total_actual_branches == 3
    assert estimate.successes == 2
    assert estimate.success_rate == 2 / 3
    assert estimate.confidence_interval[0] >= 0 and estimate.confidence_interval[1] <= 1


def test_duplicate_branch_and_nonfinite_fail_closed():
    with pytest.raises(ValueError):
        estimate_feasibility([_row(0), _row(0)])
    with pytest.raises(ValueError):
        BranchOutcome("x", "s", "r", 0, 1, 1, False, float("nan"), None, None, "ZERO_MEMORY", "h")
