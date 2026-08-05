"""CC2 follow-up P0-15 tests: the 98304 budget semantics.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
no longrun ever starts here. The 98304 total is never applied
implicitly; the budget must come from a director decision with
explicit semantics.

Covered negative matrix:
* absent / incomplete director decision    -> BUDGET_UNDECIDED
  (BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION)
* unknown semantics                        -> BUDGET_UNKNOWN_SEMANTICS
* inconsistent arithmetic                  -> BUDGET_INCONSISTENT
* zero additional / zero final             -> BUDGET_ZERO
* bool / negative steps                    -> BUDGET_BAD_TYPE
* TOTAL_FROM_COMMON_INITIALIZATION with
  nonzero initial steps                    -> BUDGET_INCONSISTENT
* longrun refuses without a decision       -> REFUSED + blocker
* decided budget prepares the manifest     -> PREPARED (fields frozen)
"""
import json
import os
import sys

import pytest

from dicode.teachers.e1_formal import budget_semantics as BS

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import run_e1_longrun as LONG  # noqa: E402


def _block(**overrides):
    kwargs = dict(
        semantics=BS.ADDITIONAL_FROM_PRETRAINED_CHECKPOINT,
        initial_checkpoint_env_steps=4096,
        additional_training_env_steps=94208,
        final_total_env_steps=98304,
    )
    kwargs.update(overrides)
    return kwargs


class TestBudgetResolution:
    def test_additional_from_pretrained_resolves(self):
        budget = BS.resolve_training_budget(_block(), "test")
        assert budget.semantics == (
            BS.ADDITIONAL_FROM_PRETRAINED_CHECKPOINT
        )
        assert budget.initial_checkpoint_env_steps == 4096
        assert budget.additional_training_env_steps == 94208
        assert budget.final_total_env_steps == 98304
        assert len(budget.budget_hash) == 64
        BS.require_budget_decided(budget, "test")

    def test_total_from_common_init_resolves(self):
        budget = BS.resolve_training_budget(
            _block(
                semantics=BS.TOTAL_FROM_COMMON_INITIALIZATION,
                initial_checkpoint_env_steps=0,
                additional_training_env_steps=98304,
            ),
            "test",
        )
        assert budget.semantics == BS.TOTAL_FROM_COMMON_INITIALIZATION
        assert budget.final_total_env_steps == 98304

    def test_absent_decision_is_undecided(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(None, "test")
        assert excinfo.value.code == BS.BUDGET_UNDECIDED
        assert BS.BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION in str(
            excinfo.value
        )

    def test_incomplete_decision_is_undecided(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                {"semantics": BS.TOTAL_FROM_COMMON_INITIALIZATION},
                "test",
            )
        assert excinfo.value.code == BS.BUDGET_UNDECIDED

    def test_unknown_semantics_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                _block(semantics="compute-matched-98304"), "test"
            )
        assert excinfo.value.code == BS.BUDGET_UNKNOWN_SEMANTICS

    def test_inconsistent_arithmetic_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                _block(final_total_env_steps=90000), "test"
            )
        assert excinfo.value.code == BS.BUDGET_INCONSISTENT

    def test_zero_additional_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                _block(
                    additional_training_env_steps=0,
                    final_total_env_steps=4096,
                ),
                "test",
            )
        assert excinfo.value.code == BS.BUDGET_ZERO

    def test_bool_and_negative_steps_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                _block(initial_checkpoint_env_steps=True), "test"
            )
        assert excinfo.value.code == BS.BUDGET_BAD_TYPE
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                _block(additional_training_env_steps=-5), "test"
            )
        assert excinfo.value.code == BS.BUDGET_BAD_TYPE

    def test_common_init_with_nonzero_initial_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.resolve_training_budget(
                _block(
                    semantics=BS.TOTAL_FROM_COMMON_INITIALIZATION,
                    initial_checkpoint_env_steps=4096,
                    additional_training_env_steps=94208,
                ),
                "test",
            )
        assert excinfo.value.code == BS.BUDGET_INCONSISTENT

    def test_undecided_budget_never_starts(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            BS.require_budget_decided(None, "test")
        assert excinfo.value.code == BS.BUDGET_UNDECIDED


class TestLongrunIntegration:
    def test_longrun_refuses_without_a_budget_decision(self):
        manifest = LONG.build_frozen_manifest(
            LONG.RT.TEACHER_CONFIG_PATH, budget_block=None
        )
        codes = [b["code"] for b in manifest["blockers"]]
        assert BS.BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION in codes
        fields = manifest["fields"]
        assert fields["training_budget_semantics"]["frozen"] is False
        assert fields["final_total_env_steps"]["frozen"] is False
        assert fields["total_env_steps"]["frozen"] is False
        # never applied implicitly as a compute-matched 98304
        assert fields["total_env_steps"]["value"] is None

    def test_longrun_prepares_with_a_decided_budget(self):
        manifest = LONG.build_frozen_manifest(
            LONG.RT.TEACHER_CONFIG_PATH, budget_block=_block()
        )
        fields = manifest["fields"]
        assert fields["training_budget_semantics"]["frozen"] is True
        assert fields["final_total_env_steps"]["value"] == 98304
        assert fields["total_env_steps"]["value"] == 98304
        budget_codes = [
            b["code"]
            for b in manifest["blockers"]
            if b["field"] == "training_budget"
        ]
        assert budget_codes == []
