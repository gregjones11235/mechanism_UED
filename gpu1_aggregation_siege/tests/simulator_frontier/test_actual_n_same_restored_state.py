# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-2/P0-4): actual-N branch search runs from the EXACT
restored state and can never report a partial run as complete.  The gates
that are reachable without the controller-signed bundle are pinned here:
per-branch seeds are canonical and deterministic, a run without a minted
VerifiedRestoreContext fails closed BEFORE any branch executes, and the
summary reflects exactly the number of executed branches.
"""

import pytest

from dicode.simulator_frontier.branch_search_runner import (
    BranchSearchBlockedError,
    BranchSearchRunner,
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
    actual_n_summary,
    derive_branch_seeds,
    require_production_restore_context,
)
from dicode.simulator_frontier.search_statistics import BranchOutcome
from dicode.student_adapters.fake import FakeStudentAdapter

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _runner() -> BranchSearchRunner:
    student = FakeStudentAdapter(candidate_id="FAKE_SEARCH_CONTRACT_ONLY")
    return BranchSearchRunner(
        student=student,
        student_params=student._params,
        step_fn=lambda *a: (None, None, 0.0, False, {}),
        env_params={},
        template={"x": 0},
        observe_fn=lambda state: None,
        capture_student_id="FAKE_CAPTURE_CONTRACT_ONLY",
        search_student_id="FAKE_SEARCH_CONTRACT_ONLY",
        train_student_id="FAKE_TRAIN_CONTRACT_ONLY",
    )


class TestDeriveBranchSeeds:
    def test_seeds_are_deterministic_and_canonical(self):
        a = derive_branch_seeds(seed_base=7, state_id="s1",
                                source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                branch_index=0)
        b = derive_branch_seeds(seed_base=7, state_id="s1",
                                source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                branch_index=0)
        assert a == b

    def test_seeds_vary_per_branch_index_and_source(self):
        s0 = derive_branch_seeds(seed_base=7, state_id="s1",
                                 source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                 branch_index=0)
        s1 = derive_branch_seeds(seed_base=7, state_id="s1",
                                 source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                 branch_index=1)
        t0 = derive_branch_seeds(seed_base=7, state_id="s1",
                                 source=SEARCH_SOURCE_STUDENT_STOCHASTIC,
                                 branch_index=0)
        assert s0 != s1 and s0 != t0

    def test_seeds_vary_per_state_and_seed_base(self):
        a = derive_branch_seeds(seed_base=7, state_id="s1",
                                source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                branch_index=0)
        b = derive_branch_seeds(seed_base=8, state_id="s1",
                                source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                branch_index=0)
        c = derive_branch_seeds(seed_base=7, state_id="s2",
                                source=SEARCH_SOURCE_STUDENT_DETERMINISTIC,
                                branch_index=0)
        assert a != b and a != c


class TestRequireProductionRestoreContext:
    def test_none_context_fails_closed(self):
        with pytest.raises(BranchSearchBlockedError):
            require_production_restore_context(None)

    def test_mapping_context_rejected_as_self_reported(self):
        with pytest.raises(BranchSearchBlockedError):
            require_production_restore_context({"production_joint_pass": True})

    def test_foreign_context_rejected(self):
        with pytest.raises(BranchSearchBlockedError):
            require_production_restore_context("context")


class TestNoPartialActualN:
    def test_run_without_verified_context_never_executes_branches(self):
        # The context gate runs BEFORE the first branch: without a minted
        # VerifiedRestoreContext the run raises instead of executing a
        # partial set of branches.  actual_N can therefore never be smaller
        # than requested_N on this path.
        runner = _runner()
        with pytest.raises(BranchSearchBlockedError):
            runner.run_actual_n(None, config=None, seed_base=7,
                                restore_context=None)

    def test_mapping_context_refused_at_run_entry(self):
        runner = _runner()
        with pytest.raises(BranchSearchBlockedError):
            runner.run_actual_n(None, config=None, seed_base=7,
                                restore_context={"production_joint_pass": True})


class TestActualNSummaryHonesty:
    def test_summary_counts_exactly_the_executed_branches(self):
        outcomes = (
            BranchOutcome(branch_id="b0", state_id="s", search_source="STUDENT_DETERMINISTIC",
                          rng_seed=0, horizon=8, transitions_used=3, success=False,
                          progress=0.25, terminal_event=None,
                          failure_category="HORIZON_EXHAUSTED", memory_mode="SAVED_POLICY_MEMORY",
                          outcome_hash="a" * 64),
            BranchOutcome(branch_id="b1", state_id="s", search_source="STUDENT_STOCHASTIC",
                          rng_seed=1, horizon=8, transitions_used=4, success=True,
                          progress=0.75, terminal_event=None,
                          failure_category=None, memory_mode="SAVED_POLICY_MEMORY",
                          outcome_hash="b" * 64),
        )
        summary = actual_n_summary(outcomes)
        assert int(summary["actual_n"]) == len(outcomes) == 2
