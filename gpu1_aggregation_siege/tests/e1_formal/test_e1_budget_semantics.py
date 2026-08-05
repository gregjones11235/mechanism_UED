"""CC2-Director tests: the training budget on the DiCode timeline.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
no formal experiment ever starts here. 98304 is NOT a formal budget;
the only timeline is the frozen DiCode config's total_timesteps.

Covered negative matrix:
* absent / incomplete director decision    -> BUDGET_UNDECIDED
* unknown semantics                        -> BUDGET_UNKNOWN_SEMANTICS
* total_timesteps != frozen DiCode value   -> BUDGET_TIMELINE_MISMATCH
* inconsistent arithmetic                  -> BUDGET_INCONSISTENT
* zero additional / zero final             -> BUDGET_ZERO
* bool / negative fields                   -> BUDGET_BAD_TYPE
* longrun refuses without a decision       -> REFUSED + blocker
* decided budget prepares the manifest     -> PREPARED (fields frozen)
"""
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

#: the frozen DiCode resolved config value (conf/training/default.yaml)
_TOTAL = 2_005_401_600
_ADD = _TOTAL - 4096


def _block(**overrides):
    kwargs = dict(
        semantics=BS.ADDITIONAL_FROM_PRETRAINED_CHECKPOINT,
        total_timesteps=_TOTAL,
        initial_checkpoint_timesteps=4096,
        additional_training_timesteps=_ADD,
        final_total_timesteps=_TOTAL,
    )
    kwargs.update(overrides)
    return kwargs


def _resolve(block, **overrides):
    return BS.resolve_training_budget(
        block,
        frozen_total_timesteps=_TOTAL,
        ctx="test",
        **overrides,
    )


class TestBudgetResolution:
    def test_additional_from_pretrained_resolves(self):
        budget = _resolve(_block())
        assert budget.semantics == (
            BS.ADDITIONAL_FROM_PRETRAINED_CHECKPOINT
        )
        assert budget.initial_checkpoint_timesteps == 4096
        assert budget.additional_training_timesteps == _ADD
        assert budget.final_total_timesteps == _TOTAL
        assert len(budget.budget_hash) == 64
        BS.require_budget_decided(budget, "test")

    def test_total_from_common_init_resolves(self):
        budget = _resolve(
            _block(
                semantics=BS.TOTAL_FROM_COMMON_INITIALIZATION,
                initial_checkpoint_timesteps=0,
                additional_training_timesteps=_TOTAL,
                final_total_timesteps=_TOTAL,
            )
        )
        assert budget.semantics == BS.TOTAL_FROM_COMMON_INITIALIZATION
        assert budget.final_total_timesteps == _TOTAL

    def test_absent_decision_is_undecided(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(None)
        assert excinfo.value.code == BS.BUDGET_UNDECIDED
        assert BS.BLOCKED_WAITING_DIRECTOR_BUDGET_DECISION in str(
            excinfo.value
        )

    def test_incomplete_decision_is_undecided(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve({"semantics": BS.TOTAL_FROM_COMMON_INITIALIZATION})
        assert excinfo.value.code == BS.BUDGET_UNDECIDED

    def test_unknown_semantics_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(_block(semantics="compute-matched-98304"))
        assert excinfo.value.code == BS.BUDGET_UNKNOWN_SEMANTICS

    def test_timeline_mismatch_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(_block(total_timesteps=98304))
        assert excinfo.value.code == BS.BUDGET_TIMELINE_MISMATCH

    def test_inconsistent_arithmetic_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(_block(final_total_timesteps=90000))
        assert excinfo.value.code == BS.BUDGET_INCONSISTENT

    def test_zero_additional_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(
                _block(
                    additional_training_timesteps=0,
                    final_total_timesteps=4096,
                )
            )
        assert excinfo.value.code == BS.BUDGET_ZERO

    def test_bool_and_negative_fields_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(_block(initial_checkpoint_timesteps=True))
        assert excinfo.value.code == BS.BUDGET_BAD_TYPE
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(_block(additional_training_timesteps=-5))
        assert excinfo.value.code == BS.BUDGET_BAD_TYPE

    def test_common_init_with_nonzero_initial_refused(self):
        with pytest.raises(BS.BudgetError) as excinfo:
            _resolve(
                _block(
                    semantics=BS.TOTAL_FROM_COMMON_INITIALIZATION,
                    initial_checkpoint_timesteps=4096,
                    additional_training_timesteps=_TOTAL - 4096,
                )
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
        assert fields["final_total_timesteps"]["frozen"] is False
        assert fields["total_timesteps"]["frozen"] is False
        assert fields["total_timesteps"]["value"] is None

    def test_longrun_prepares_with_a_decided_budget(self):
        manifest = LONG.build_frozen_manifest(
            LONG.RT.TEACHER_CONFIG_PATH, budget_block=_block()
        )
        fields = manifest["fields"]
        assert fields["training_budget_semantics"]["frozen"] is True
        assert fields["total_timesteps"]["value"] == _TOTAL
        assert fields["final_total_timesteps"]["value"] == _TOTAL
        budget_codes = [
            b["code"]
            for b in manifest["blockers"]
            if b["field"] == "training_budget"
        ]
        assert budget_codes == []
