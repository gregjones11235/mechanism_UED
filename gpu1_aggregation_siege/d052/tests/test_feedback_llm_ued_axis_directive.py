"""C5 / P1-1: AxisDirective — the board -> EnvCoder controlled spec."""
import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_DECREASE,
    DIRECTION_HOLD,
    DIRECTION_INCREASE,
    LEVEL_NONE,
    ROLE_CONTROL,
    ROLE_TREATMENT,
    AxisDirective,
    assert_directive_batch_legal,
    candidate_axis_config,
)


def make_directive(**overrides):
    base = dict(directive_id="dir-001",
                source_window=1,
                environment_family="threat_distance_family",
                axis="threat_distance_grading",
                old_level="low",
                new_level="high",
                direction=DIRECTION_INCREASE,
                experiment_control_role=ROLE_TREATMENT,
                held_constant_axes={"threat_count": "medium"},
                expected_next_signature={"student_success_rate": 0.35},
                rationale="probe the distance-graded threat regime")
    base.update(overrides)
    return AxisDirective(**base)


class TestLegalDirectives:
    def test_treatment_directive_hashes_deterministically(self):
        d1 = make_directive()
        d2 = make_directive()
        assert len(d1.directive_hash) == 64
        assert d1.directive_hash == d2.directive_hash
        assert d1.rehash() == d1.directive_hash

    def test_hash_participates_in_content(self):
        assert make_directive().directive_hash != \
            make_directive(new_level="medium",
                           directive_id="dir-002").directive_hash

    def test_first_measurement_from_none_level(self):
        d = make_directive(old_level=LEVEL_NONE, new_level="medium",
                           direction=DIRECTION_INCREASE)
        assert d.old_level == LEVEL_NONE

    def test_control_directive_holds_the_setting(self):
        d = make_directive(old_level="medium", new_level="medium",
                           direction=DIRECTION_HOLD,
                           experiment_control_role=ROLE_CONTROL)
        axis_values, held = candidate_axis_config(d)
        assert axis_values == {"threat_distance_grading": "medium"}
        assert held == {"threat_count": "medium"}


class TestFailClosedLadder:
    def test_direction_level_mismatch(self):
        with pytest.raises(ValidationError, match="DIRECTION_LEVEL_MISMATCH"):
            make_directive(old_level="low", new_level="high",
                           direction=DIRECTION_DECREASE)

    def test_no_op_treatment_rejected(self):
        with pytest.raises(ValidationError, match="NO_OP_TREATMENT_DIRECTIVE"):
            make_directive(old_level="high", new_level="high",
                           direction=DIRECTION_INCREASE)

    def test_control_must_hold(self):
        with pytest.raises(ValidationError, match="CONTROL_DIRECTIVE_MUST_HOLD"):
            make_directive(experiment_control_role=ROLE_CONTROL,
                           old_level="low", new_level="high",
                           direction=DIRECTION_INCREASE)
        with pytest.raises(ValidationError, match="CONTROL_DIRECTIVE_MUST_HOLD"):
            make_directive(experiment_control_role=ROLE_CONTROL,
                           direction=DIRECTION_HOLD,
                           old_level="low", new_level="high")

    def test_treatment_needs_increase_or_decrease(self):
        with pytest.raises(ValidationError,
                           match="TREATMENT_DIRECTIVE_NEEDS_DIRECTION"):
            make_directive(direction=DIRECTION_HOLD)

    def test_axis_must_belong_to_family(self):
        with pytest.raises(ValidationError, match="AXIS_NOT_IN_FAMILY"):
            make_directive(axis="resource_pressure")

    def test_axis_must_be_a_legal_mutation_axis(self):
        with pytest.raises(ValidationError, match="ILLEGAL_DIRECTIVE_AXIS"):
            make_directive(axis="reward_scale")

    def test_held_axes_constrained(self):
        with pytest.raises(ValidationError, match="ILLEGAL_HELD_AXIS"):
            make_directive(
                held_constant_axes={"threat_distance_grading": "medium"})
        with pytest.raises(ValidationError, match="ILLEGAL_HELD_LEVEL"):
            make_directive(held_constant_axes={"threat_count": "extreme"})

    def test_levels_and_role_vocab(self):
        with pytest.raises(ValidationError, match="ILLEGAL_OLD_LEVEL"):
            make_directive(old_level="extreme")
        with pytest.raises(ValidationError, match="ILLEGAL_NEW_LEVEL"):
            make_directive(new_level=LEVEL_NONE)
        with pytest.raises(ValidationError, match="ILLEGAL_DIRECTION"):
            make_directive(direction="sideways")
        with pytest.raises(ValidationError, match="ILLEGAL_EXPERIMENT_ROLE"):
            make_directive(experiment_control_role="placebo")

    def test_expected_signature_required_and_finite(self):
        with pytest.raises(ValidationError, match="EMPTY_EXPECTED_SIGNATURE"):
            make_directive(expected_next_signature={})
        with pytest.raises(ValidationError, match="NON_FINITE_EXPECTATION"):
            make_directive(expected_next_signature={
                "student_success_rate": float("inf")})


class TestCandidateAxisConfig:
    def test_treatment_applies_new_level(self):
        d = make_directive()
        axis_values, held = candidate_axis_config(d)
        assert axis_values == {"threat_distance_grading": "high"}
        assert held == {"threat_count": "medium"}

    def test_config_derives_only_from_directive(self):
        # the same directive always yields the same configuration — no index
        # rotation, no window-order coupling
        d = make_directive()
        assert candidate_axis_config(d) == candidate_axis_config(d)


class TestBatchLegality:
    def test_duplicate_directive_ids_rejected(self):
        d1 = make_directive()
        d2 = make_directive(axis="threat_count", old_level="low",
                            new_level="medium",
                            held_constant_axes={
                                "threat_distance_grading": "medium"})
        with pytest.raises(ValueError, match="DUPLICATE_DIRECTIVE_ID"):
            assert_directive_batch_legal([d1, d2])

    def test_one_treatment_per_window_family_axis(self):
        d1 = make_directive()
        d2 = make_directive(directive_id="dir-002")
        with pytest.raises(ValueError, match="DUPLICATE_AXIS_DIRECTIVE"):
            assert_directive_batch_legal([d1, d2])

    def test_treatment_plus_control_on_same_axis_is_legal(self):
        treatment = make_directive()
        control = make_directive(directive_id="dir-002",
                                 old_level="low", new_level="low",
                                 direction=DIRECTION_HOLD,
                                 experiment_control_role=ROLE_CONTROL)
        assert_directive_batch_legal([treatment, control])  # no raise

    def test_different_windows_do_not_collide(self):
        d1 = make_directive()
        d2 = make_directive(directive_id="dir-002", source_window=2)
        assert_directive_batch_legal([d1, d2])              # no raise
