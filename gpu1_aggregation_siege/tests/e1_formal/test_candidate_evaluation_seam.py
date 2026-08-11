"""C11 tests: the candidate evaluation seam (pipeline stage 6).

Loading strategy: ``dicode.evaluation.__init__`` imports
``online_evaluation``, which imports craftax/minicraftax — absent from
the audit venv. The seam module itself is pure gate logic (no craftax,
verified by AST in test_wiring_sources), so this suite loads the file
directly. In the full environment the package import chain also works;
the module's internal imports resolve to the same cached e1_formal
modules either way.
"""
import importlib.util
import os

import pytest

from dicode.teachers.e1_formal.flags import E1Flags
from dicode.teachers.e1_formal.reference_contract import (
    consume_reference_identity_contract,
)
from test_reference_contract import _block  # frozen-block fixture

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SEAM_PATH = os.path.join(
    REPO_ROOT, "src", "dicode", "evaluation", "candidate_evaluation.py"
)


def _load_seam():
    spec = importlib.util.spec_from_file_location(
        "e1_candidate_evaluation_under_test", _SEAM_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CE = _load_seam()


def _frozen_contract():
    return consume_reference_identity_contract(_block(), "seam-test")


def _evaluate(contract, *, states=True, flags=None, enabled=False):
    student = object() if states else None
    reference = object() if states else None
    config = {"candidate_eval": {"enabled": enabled}}
    return CE.evaluate_candidates_with_reference(
        config,
        None,  # rng: consumed only by the (absent) real path
        ["dyn_a", "dyn_b"],
        None,  # archive
        None,  # embedding_model
        student,
        reference,
        flags,
        contract,
    )


class TestGate1ReferenceContractFirst:
    def test_no_contract_blocks_with_g1_code(self):
        result = _evaluate(None)
        assert result["status"] == CE.EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN
        assert result["evaluated"] is False
        assert result["results"] == ()
        assert result["gates_checked"] == ["G1_reference_contract"]
        assert result["candidate_task_ids"] == ("dyn_a", "dyn_b")

    def test_blocked_result_carries_no_provenance_stamp(self):
        result = _evaluate(None)
        assert "provenance" not in result

    def test_non_contract_object_fails_closed(self):
        result = _evaluate(object())
        assert result["status"] == CE.EVAL_BLOCKED_REFERENCE_CONTRACT_BAD_TYPE
        assert result["evaluated"] is False

    def test_g1_runs_before_the_adapter_gate(self):
        # even with NO states at all, an unfrozen contract reports G1,
        # not the adapter code (gate order is fixed)
        result = _evaluate(None, states=False)
        assert result["status"] == CE.EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN


class TestGate2StudentAdapter:
    def test_no_states_skips_with_adapter_code(self):
        result = _evaluate(_frozen_contract(), states=False)
        assert result["status"] == CE.EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER
        assert result["gates_checked"] == [
            "G1_reference_contract",
            "student_adapter_state",
        ]

    def test_missing_flags_contract_skips(self):
        result = _evaluate(_frozen_contract(), flags=None)
        assert result["status"] == CE.EVAL_SEAM_SKIPPED_NO_STUDENT_ADAPTER


class TestGate3ConfigKnob:
    def test_disabled_knob_blocks_even_with_states(self):
        result = _evaluate(_frozen_contract(), flags=E1Flags(), enabled=False)
        assert result["status"] == CE.EVAL_DISABLED_BY_CONFIG

    def test_knob_must_be_literal_true(self):
        config = {"candidate_eval": {"enabled": "yes"}}
        result = CE.evaluate_candidates_with_reference(
            config, None, ["dyn_a"], None, None, object(), object(),
            E1Flags(), _frozen_contract(),
        )
        assert result["status"] == CE.EVAL_DISABLED_BY_CONFIG

    def test_missing_knob_fails_closed(self):
        result = CE.evaluate_candidates_with_reference(
            {}, None, ["dyn_a"], None, None, object(), object(),
            E1Flags(), _frozen_contract(),
        )
        assert result["status"] == CE.EVAL_DISABLED_BY_CONFIG


class TestRealProbePathIsNeverStubbed:
    def test_all_gates_pass_raises_notimplemented_with_cc4_pointer(self):
        with pytest.raises(NotImplementedError) as excinfo:
            _evaluate(_frozen_contract(), flags=E1Flags(), enabled=True)
        message = str(excinfo.value)
        assert "CC4" in message
        assert CE.CANDIDATE_EVALUATION_PROVENANCE in message


class TestInputValidationFailClosed:
    def test_non_sequence_candidates_rejected(self):
        with pytest.raises(CE.CandidateEvaluationError) as excinfo:
            CE.evaluate_candidates_with_reference(
                {}, None, "dyn_a", None, None, None, None, None, None
            )
        assert excinfo.value.code == CE.EVAL_BAD_CANDIDATE_SET

    def test_empty_or_nonstr_id_rejected(self):
        with pytest.raises(CE.CandidateEvaluationError) as excinfo:
            CE.evaluate_candidates_with_reference(
                {}, None, ["dyn_a", "  "], None, None, None, None, None, None
            )
        assert excinfo.value.code == CE.EVAL_BAD_CANDIDATE_SET

    def test_duplicate_id_rejected(self):
        with pytest.raises(CE.CandidateEvaluationError) as excinfo:
            CE.evaluate_candidates_with_reference(
                {}, None, ["dyn_a", "dyn_a"], None, None, None, None,
                None, None,
            )
        assert excinfo.value.code == CE.EVAL_BAD_CANDIDATE_SET

    def test_input_validation_runs_before_gate_1(self):
        # even with a valid contract, bad candidate ids fail closed
        with pytest.raises(CE.CandidateEvaluationError):
            CE.evaluate_candidates_with_reference(
                {}, None, [42], None, None, object(), object(),
                E1Flags(), _frozen_contract(),
            )

    def test_empty_candidate_set_still_gates_honestly(self):
        result = CE.evaluate_candidates_with_reference(
            {}, None, [], None, None, None, None, None, None
        )
        assert result["status"] == CE.EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN
        assert result["candidate_task_ids"] == ()
