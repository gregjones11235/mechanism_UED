"""C7: independent EnvCoder + compile/reset/step gates."""
import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_HOLD,
    DIRECTION_INCREASE,
    ROLE_CONTROL,
    ROLE_TREATMENT,
    AxisDirective,
)
from d052.feedback_llm_ued.env_coder import (
    CODE_SYMBOL_PREFIX,
    RealEnvCoderBlocked,
    RealEnvCoderSeam,
    run_env_coder,
)
from d052.feedback_llm_ued.env_coder_gate import (
    EnvCoderGate,
    EnvCoderGateBlocked,
)
from d052.feedback_llm_ued.llm_backend import DeterministicMockFeedbackBackend


def make_directives(window=1):
    treatment = AxisDirective(
        directive_id="dir-t1", source_window=window,
        environment_family="threat_distance_family",
        axis="threat_distance_grading", old_level="low", new_level="high",
        direction=DIRECTION_INCREASE,
        experiment_control_role=ROLE_TREATMENT,
        held_constant_axes={"threat_count": "medium"},
        expected_next_signature={"student_success_rate": 0.35},
        rationale="treatment")
    control = AxisDirective(
        directive_id="dir-c1", source_window=window,
        environment_family="threat_distance_family",
        axis="threat_distance_grading", old_level="medium",
        new_level="medium", direction=DIRECTION_HOLD,
        experiment_control_role=ROLE_CONTROL,
        held_constant_axes={"threat_count": "medium"},
        expected_next_signature={"student_success_rate": 0.5},
        rationale="control")
    return [treatment, control]


def code(window=1, sequence=6):
    backend = DeterministicMockFeedbackBackend()
    directives = make_directives(window)
    output, envelope = run_env_coder(window=window, directives=directives,
                                     backend=backend, sequence=sequence)
    return output, envelope, directives, backend


class TestEnvCoderCall:
    def test_env_coder_is_the_seventh_window_call(self):
        output, envelope, _, backend = code()
        assert backend.usage.mock_calls == 1
        assert envelope.role == C.ROLE_ENV_CODER
        assert envelope.window == 1
        assert envelope.sequence == 6
        assert C.LLM_CALLS_PER_WINDOW == C.BOARD_CALLS_PER_WINDOW + 1 == 7

    def test_coded_batch_covers_directives_one_to_one(self):
        output, _, directives, _ = code()
        # 1:1 coverage (the coder emits deterministic sorted order)
        assert sorted(c.directive_id for c in output.coded) == \
            sorted(d.directive_id for d in directives)
        for c in output.coded:
            assert c.code_symbol.startswith(CODE_SYMBOL_PREFIX)
            assert c.directive_hash  # binds the source directive

    def test_coding_is_deterministic(self):
        a = code()[0].model_dump()
        b = code()[0].model_dump()
        assert a == b
        assert a["directive_batch_hash"] == b["directive_batch_hash"]

    def test_code_symbol_depends_on_directive_content_not_id(self):
        output, _, directives, _ = code()
        symbols = {c.directive_id: c.code_symbol for c in output.coded}
        assert symbols["dir-t1"] != symbols["dir-c1"]
        # re-coding the SAME directives reproduces the same symbols
        again = code()[0]
        for c in again.coded:
            assert symbols[c.directive_id] == c.code_symbol

    def test_real_env_coder_seam_is_blocked_this_round(self):
        assert C.REAL_ENVCODER_USED is False
        with pytest.raises(RealEnvCoderBlocked, match="REAL_ENVCODER_BLOCKED"):
            RealEnvCoderSeam(authorized=C.REAL_LLM_CALLS_AUTHORIZED)


class TestGates:
    def test_honest_batch_passes_all_three_gates(self):
        output, _, directives, _ = code()
        report = EnvCoderGate().evaluate(window=1, directives=directives,
                                         output=output)
        assert report.passed
        assert report.compile_passed and report.reset_passed \
            and report.step_passed
        assert report.blockers == []
        assert len(report.gate_hash) == 64
        EnvCoderGate().assert_passed(report)           # no raise

    def test_missing_directive_blocks_compile(self):
        output, _, directives, _ = code()
        report = EnvCoderGate().evaluate(window=1,
                                         directives=directives,
                                         output=_drop(output, "dir-c1"))
        assert not report.compile_passed
        assert any("DIRECTIVE_NOT_CODED" in b for b in report.blockers)

    def test_unauthorized_coded_entry_blocks_compile(self):
        output, _, directives, _ = code()
        report = EnvCoderGate().evaluate(window=1,
                                         directives=directives[:1],
                                         output=output)     # coded has both
        assert not report.compile_passed
        assert any("UNAUTHORIZED_CODED_DIRECTIVE" in b
                   for b in report.blockers)

    def test_directive_hash_mismatch_blocks_compile(self):
        output, _, directives, _ = code()
        dump = output.model_dump()
        dump["coded"][0]["directive_hash"] = "0" * 64
        tampered = _rebuild(dump)
        report = EnvCoderGate().evaluate(window=1, directives=directives,
                                         output=tampered)
        assert not report.compile_passed
        assert any("DIRECTIVE_HASH_MISMATCH" in b for b in report.blockers)

    def test_malformed_code_symbol_blocks_compile(self):
        output, _, directives, _ = code()
        dump = output.model_dump()
        dump["coded"][0]["code_symbol"] = "definitely-not-code"
        report = EnvCoderGate().evaluate(window=1, directives=directives,
                                         output=_rebuild(dump))
        assert not report.compile_passed
        assert any("EMPTY_CODE_SYMBOL" in b for b in report.blockers)

    def test_missing_reset_contract_blocks_reset_gate(self):
        output, _, directives, _ = code()
        dump = output.model_dump()
        dump["coded"][0]["reset_contract"] = "no-contract-declared"
        report = EnvCoderGate().evaluate(window=1, directives=directives,
                                         output=_rebuild(dump))
        assert report.compile_passed                    # compile unaffected
        assert not report.reset_passed
        assert any("RESET_CONTRACT_MISSING" in b for b in report.blockers)

    def test_missing_step_contract_blocks_step_gate(self):
        output, _, directives, _ = code()
        dump = output.model_dump()
        dump["coded"][1]["step_contract"] = "no-contract-declared"
        report = EnvCoderGate().evaluate(window=1, directives=directives,
                                         output=_rebuild(dump))
        assert not report.step_passed
        assert any("STEP_CONTRACT_MISSING" in b for b in report.blockers)

    def test_assert_passed_raises_with_all_blockers(self):
        output, _, directives, _ = code()
        dump = output.model_dump()
        dump["coded"][0]["reset_contract"] = "broken"
        dump["coded"][0]["step_contract"] = "broken"
        report = EnvCoderGate().evaluate(window=1, directives=directives,
                                         output=_rebuild(dump))
        with pytest.raises(EnvCoderGateBlocked,
                           match="RESET_CONTRACT_MISSING") as exc:
            EnvCoderGate().assert_passed(report)
        assert "STEP_CONTRACT_MISSING" in str(exc.value)

    def test_empty_directive_batch_passes_trivially(self):
        backend = DeterministicMockFeedbackBackend()
        output, _ = run_env_coder(window=0, directives=[], backend=backend,
                                  sequence=6)
        assert output.coded == []
        report = EnvCoderGate().evaluate(window=0, directives=[],
                                         output=output)
        assert report.passed


def _drop(output, directive_id):
    dump = output.model_dump()
    dump["coded"] = [c for c in dump["coded"]
                     if c["directive_id"] != directive_id]
    return _rebuild(dump)


def _rebuild(dump):
    from d052.feedback_llm_ued.env_coder import EnvCoderOutput
    return EnvCoderOutput(**dump)
