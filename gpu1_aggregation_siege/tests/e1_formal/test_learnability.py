"""C9 tests: probe-based learnability metrics (G2 gate).

Every probe used here is an explicitly labeled FIXTURE — no real
Student/Reference probe exists this round, and nothing in this file
pretends otherwise.
"""
import os

import pytest

from dicode.teachers.e1_formal import metrics as MT
from dicode.teachers.e1_formal import selector as SEL

PROBE_PROVENANCE = "CANDIDATE_EVALUATION"

#: frozen-config threshold fixture (G2: no defaults exist in code)
THRESHOLDS = {
    "tau_saturated": 0.9,
    "tau_reachable": 0.6,
    "tau_unreachable": 0.2,
    "delta_min": 0.15,
    "min_episodes": 20,
    "ci_level": 0.95,
}


def _thresholds():
    return MT.consume_learnability_thresholds(dict(THRESHOLDS), "test")


def _probe(candidate_id, side, successes, episodes):
    """FIXTURE probe record (never a real measurement this round)."""
    return {
        "candidate_id": candidate_id,
        "side": side,
        "successes": successes,
        "episodes": episodes,
        "provenance": PROBE_PROVENANCE,
    }


def _consume(candidate_id, s_success, s_episodes, r_success, r_episodes):
    stu = MT.consume_probe_result(
        _probe(candidate_id, MT.SIDE_STUDENT, s_success, s_episodes), "test"
    )
    ref = MT.consume_probe_result(
        _probe(candidate_id, MT.SIDE_REFERENCE, r_success, r_episodes), "test"
    )
    return stu, ref


class TestWilsonInterval:
    def test_bounds_within_unit_interval(self):
        for successes, episodes in [(0, 1), (1, 1), (3, 10), (10, 10), (7, 25)]:
            lo, hi = MT.wilson_interval(successes, episodes, 1.9599639845400540)
            assert 0.0 <= lo <= hi <= 1.0

    def test_point_estimate_inside_interval(self):
        lo, hi = MT.wilson_interval(30, 100, 1.9599639845400540)
        assert lo < 0.30 < hi

    def test_more_episodes_shrink_interval(self):
        z = 1.9599639845400540
        lo_a, hi_a = MT.wilson_interval(3, 10, z)
        lo_b, hi_b = MT.wilson_interval(30, 100, z)
        assert (hi_a - lo_a) > (hi_b - lo_b)

    @pytest.mark.parametrize(
        "successes,episodes", [(0, 0), (-1, 5), (6, 5), (True, 5), (3, 2.0)]
    )
    def test_bad_inputs_fail_closed(self, successes, episodes):
        with pytest.raises(MT.MetricsError):
            MT.wilson_interval(successes, episodes, 1.96)


class TestZTable:
    def test_supported_levels(self):
        assert MT.z_for_ci_level(0.95, "t") == pytest.approx(1.9599639845400540)
        assert MT.z_for_ci_level(0.90, "t") == pytest.approx(1.6448536269514722)
        assert MT.z_for_ci_level(0.99, "t") == pytest.approx(2.5758293035489004)

    @pytest.mark.parametrize("bad", [0.5, 1.0, True, "0.95", None])
    def test_unsupported_or_bad_level_fails_closed(self, bad):
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.z_for_ci_level(bad, "t")
        assert excinfo.value.code in (
            MT.CI_LEVEL_UNSUPPORTED,
            MT.METRICS_BAD_TYPE,
        )


class TestProbeConsumption:
    def test_happy_path(self):
        probe = MT.consume_probe_result(
            _probe("cand_1", MT.SIDE_STUDENT, 7, 20), "test"
        )
        assert probe.candidate_id == "cand_1"
        assert probe.side == MT.SIDE_STUDENT
        assert probe.success_rate == pytest.approx(7 / 20)
        assert probe.provenance == PROBE_PROVENANCE

    @pytest.mark.parametrize("drop", ["candidate_id", "side", "successes", "episodes", "provenance"])
    def test_missing_field_fails_closed(self, drop):
        record = _probe("cand_1", MT.SIDE_STUDENT, 1, 5)
        del record[drop]
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_probe_result(record, "test")
        assert excinfo.value.code == MT.METRICS_MISSING_FIELD

    def test_unknown_field_fails_closed(self):
        record = dict(_probe("cand_1", MT.SIDE_STUDENT, 1, 5), extra=1)
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_probe_result(record, "test")
        assert excinfo.value.code == MT.METRICS_UNKNOWN_FIELD

    @pytest.mark.parametrize(
        "field,value",
        [
            ("side", "both"),
            ("successes", 6),        # > episodes
            ("episodes", -1),
            ("successes", True),
            ("candidate_id", "  "),
        ],
    )
    def test_out_of_range_or_bad_type_fails_closed(self, field, value):
        record = _probe("cand_1", MT.SIDE_STUDENT, 1, 5)
        record[field] = value
        with pytest.raises(MT.MetricsError):
            MT.consume_probe_result(record, "test")

    def test_formal_provenance_rejected_at_probe_layer(self):
        record = _probe("cand_1", MT.SIDE_STUDENT, 1, 5)
        record["provenance"] = "FORMAL_FRONT"
        with pytest.raises(Exception) as excinfo:
            MT.consume_probe_result(record, "test")
        assert getattr(excinfo.value, "code", "") != ""


class TestThresholdConsumption:
    def test_happy_path(self):
        thresholds = _thresholds()
        assert thresholds.tau_saturated == pytest.approx(0.9)
        assert thresholds.min_episodes == 20

    @pytest.mark.parametrize("drop", sorted(THRESHOLDS))
    def test_every_field_is_mandatory(self, drop):
        block = dict(THRESHOLDS)
        del block[drop]
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_learnability_thresholds(block, "test")
        assert excinfo.value.code == MT.LEARNABILITY_THRESHOLD_MISSING

    @pytest.mark.parametrize("field", sorted(THRESHOLDS))
    def test_none_is_missing_not_defaulted(self, field):
        block = dict(THRESHOLDS)
        block[field] = None
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_learnability_thresholds(block, "test")
        assert excinfo.value.code == MT.LEARNABILITY_THRESHOLD_MISSING

    def test_unknown_field_fails_closed(self):
        block = dict(THRESHOLDS, extra=1)
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_learnability_thresholds(block, "test")
        assert excinfo.value.code == MT.METRICS_UNKNOWN_FIELD

    def test_bad_ordering_fails_closed(self):
        block = dict(THRESHOLDS, tau_unreachable=0.7, tau_reachable=0.6)
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_learnability_thresholds(block, "test")
        assert excinfo.value.code == MT.METRICS_OUT_OF_RANGE

    @pytest.mark.parametrize(
        "field,value",
        [
            ("delta_min", 0.0),
            ("tau_saturated", 1.5),
            ("min_episodes", 0),
            ("min_episodes", True),
            ("ci_level", 0.5),
        ],
    )
    def test_bad_values_fail_closed(self, field, value):
        block = dict(THRESHOLDS)
        block[field] = value
        with pytest.raises(MT.MetricsError):
            MT.consume_learnability_thresholds(block, "test")


class TestClassificationStates:
    """Labeled FIXTURE probes hitting each G2 state boundary."""

    def _classify(self, s_success, s_episodes, r_success, r_episodes, **kw):
        stu, ref = _consume(
            "cand_1", s_success, s_episodes, r_success, r_episodes
        )
        return MT.build_learnability_verdict(
            candidate_id="cand_1",
            student_probe=stu,
            reference_probe=ref,
            thresholds=_thresholds(),
            ctx="test",
            **kw,
        )

    def test_learnable_when_gap_ci_lower_exceeds_delta_min(self):
        # FIXTURE: student 4/20, reference 18/20 -> conservative gap
        # lower bound comfortably above delta_min and reference reachable.
        verdict = self._classify(4, 20, 18, 20)
        assert verdict.state == MT.LEARNABLE
        assert verdict.gap_ci_lower >= 0.15
        assert verdict.regret == pytest.approx(18 / 20 - 4 / 20)
        assert verdict.behavioral_gap == pytest.approx(abs(18 / 20 - 4 / 20))
        assert verdict.episodes_student == 20

    def test_saturated_when_student_ci_lower_above_tau_sat(self):
        # FIXTURE: both sides near-perfect -> no learning headroom.
        verdict = self._classify(99, 100, 99, 100)
        assert verdict.state == MT.SATURATED
        assert verdict.student_ci[0] >= 0.9

    def test_both_unreachable_when_reference_ci_upper_below_tau_unreach(self):
        # FIXTURE: reference barely succeeds (2/40) -> nobody can reach it.
        verdict = self._classify(0, 40, 2, 40)
        assert verdict.state == MT.BOTH_UNREACHABLE
        assert verdict.reference_ci[1] < 0.2

    def test_insufficient_when_episodes_below_floor(self):
        verdict = self._classify(1, 2, 2, 2)
        assert verdict.state == MT.INSUFFICIENT_EVIDENCE
        assert "min_episodes" in verdict.note

    def test_insufficient_when_ci_straddles_thresholds(self):
        # FIXTURE: 12/40 vs 16/40 -> conservative gap bound is negative;
        # no verdict is fabricated.
        verdict = self._classify(12, 40, 16, 40)
        assert verdict.state == MT.INSUFFICIENT_EVIDENCE
        assert verdict.gap_ci_lower < 0.15

    def test_unreachable_beats_saturated_precedence(self):
        # both sides at zero -> BOTH_UNREACHABLE, not SATURATED
        verdict = self._classify(0, 40, 0, 40)
        assert verdict.state == MT.BOTH_UNREACHABLE

    def test_regret_is_zero_when_student_ahead(self):
        # FIXTURE: student 18/20 ahead of reference 4/20.
        verdict = self._classify(18, 20, 4, 20)
        assert verdict.regret == 0.0
        assert verdict.behavioral_gap == pytest.approx(abs(4 / 20 - 18 / 20))


class TestMissingProbesAndPrior:
    def test_missing_probe_is_unavailable_never_guessed(self):
        thresholds = _thresholds()
        for missing in ("student", "reference"):
            stu, ref = _consume("cand_1", 5, 20, 15, 20)
            verdict = MT.build_learnability_verdict(
                candidate_id="cand_1",
                student_probe=None if missing == "student" else stu,
                reference_probe=None if missing == "reference" else ref,
                thresholds=thresholds,
                ctx="test",
            )
            assert verdict.state == MT.LEARNABILITY_UNAVAILABLE
            assert verdict.regret is None
            assert verdict.gap is None

    def test_lp_prior_is_recorded_but_changes_nothing(self):
        # the archive LP prior is inert: two verdicts differing ONLY in
        # the prior must be identical in state and every real number.
        stu, ref = _consume("cand_1", 4, 20, 18, 20)
        v1 = MT.build_learnability_verdict(
            candidate_id="cand_1", student_probe=stu, reference_probe=ref,
            thresholds=_thresholds(), learnability_prior_lp=0.9, ctx="test",
        )
        v2 = MT.build_learnability_verdict(
            candidate_id="cand_1", student_probe=stu, reference_probe=ref,
            thresholds=_thresholds(), learnability_prior_lp=0.1, ctx="test",
        )
        assert v1.learnability_prior_lp == pytest.approx(0.9)
        assert v2.learnability_prior_lp == pytest.approx(0.1)
        assert v1.state == v2.state
        assert v1.gap == v2.gap
        assert v1.gap_ci_lower == v2.gap_ci_lower
        assert v1.regret == v2.regret

    def test_lp_prior_validation(self):
        stu, ref = _consume("cand_1", 4, 20, 18, 20)
        with pytest.raises(MT.MetricsError):
            MT.build_learnability_verdict(
                candidate_id="cand_1", student_probe=stu, reference_probe=ref,
                thresholds=_thresholds(), learnability_prior_lp=1.5, ctx="t",
            )

    def test_probe_candidate_mismatch_fails_closed(self):
        stu = MT.consume_probe_result(
            _probe("cand_1", MT.SIDE_STUDENT, 4, 20), "test"
        )
        ref = MT.consume_probe_result(
            _probe("cand_OTHER", MT.SIDE_REFERENCE, 18, 20), "test"
        )
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.build_learnability_verdict(
                candidate_id="cand_1", student_probe=stu,
                reference_probe=ref, thresholds=_thresholds(), ctx="test",
            )
        assert excinfo.value.code == MT.METRICS_CANDIDATE_MISMATCH

    def test_swapped_sides_fail_closed(self):
        stu = MT.consume_probe_result(
            _probe("cand_1", MT.SIDE_REFERENCE, 4, 20), "test"
        )
        ref = MT.consume_probe_result(
            _probe("cand_1", MT.SIDE_STUDENT, 18, 20), "test"
        )
        with pytest.raises(MT.MetricsError):
            MT.build_learnability_verdict(
                candidate_id="cand_1", student_probe=stu,
                reference_probe=ref, thresholds=_thresholds(), ctx="test",
            )


class TestSelectionDegradation:
    """no probe -> LEARNABILITY_UNAVAILABLE -> SELECTION_BLOCKED..."""

    def test_no_signals_blocks_selection(self):
        with pytest.raises(SEL.SelectorError) as excinfo:
            SEL.select_dynamic_batch((), k=12, seed=7)
        assert excinfo.value.code == "SELECTION_BLOCKED_NO_REAL_EVIDENCE"

    def test_signal_without_real_probe_blocks_selection(self):
        signals = SEL.consume_candidate_signals(
            [
                {
                    "candidate_id": "cand_1",
                    "role_scores": {"eval": 0.5},
                    "provenance": PROBE_PROVENANCE,
                    "has_real_probe": False,  # archive prior only
                }
            ],
            "test",
        )
        with pytest.raises(SEL.SelectorError) as excinfo:
            SEL.select_dynamic_batch(signals, k=12, seed=7)
        assert excinfo.value.code == "SELECTION_BLOCKED_NO_REAL_EVIDENCE"


class TestNoSubstituteAudit:
    def test_metrics_module_contains_no_fixed_learnability_substitute(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "dicode", "teachers", "e1_formal", "metrics.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "0.25" not in source, (
            "metrics.py must not contain any fixed learnability "
            "substitute constant"
        )

    def test_selector_never_reads_lp_prior(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "dicode", "teachers", "e1_formal", "selector.py",
        )
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        assert "learnability_prior_lp" not in source
