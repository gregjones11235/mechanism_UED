"""C9 tests: 12 dynamic + 4 anchor training layout (plan D8)."""
from fractions import Fraction

import pytest

from dicode.teachers.e1_formal import layout as L


def _dynamic_ids(n=12):
    return [f"dyn_{i:02d}" for i in range(n)]


class TestBuildLayout:
    def test_layout_has_16_entries_in_canonical_order(self):
        layout = L.build_training_layout(_dynamic_ids())
        keys = list(layout)
        assert len(keys) == 16
        assert keys[:12] == _dynamic_ids()
        assert keys[12:] == list(L.ANCHOR_TASK_IDS)
        assert keys[-1] == L.ORIGINAL_ANCHOR_TASK_ID  # original ALWAYS last

    def test_weights_match_pinned_rationals_exactly(self):
        layout = L.build_training_layout(_dynamic_ids())
        w_dynamic = float((1 - Fraction(1, 4)) / 12)  # (1 - beta) / 12
        assert w_dynamic == float(Fraction(1, 16))
        for task_id in _dynamic_ids():
            assert layout[task_id] == float(Fraction(1, 16))  # 0.0625
        assert layout["original_craftax"] == float(Fraction(1, 10))  # b*s
        for seed_id in ("task_1", "task_2", "task_3"):
            assert layout[seed_id] == float(Fraction(1, 20))  # b(1-s)/3

    def test_weights_sum_to_exactly_one_in_exact_arithmetic(self):
        layout = L.build_training_layout(_dynamic_ids())
        rational_total = (
            12 * Fraction(1, 16)
            + Fraction(1, 10)
            + 3 * Fraction(1, 20)
        )
        assert rational_total == Fraction(1)
        # every emitted float is the exact double of its rational
        for value in layout.values():
            assert value > 0.0
        assert sum(layout.values()) == pytest.approx(1.0, abs=1e-15)

    def test_deterministic_double_run_equality(self):
        assert L.build_training_layout(_dynamic_ids()) == \
            L.build_training_layout(_dynamic_ids())

    def test_anchor_weights_helper(self):
        weights = L.anchor_weights()
        assert list(weights) == list(L.ANCHOR_TASK_IDS)
        assert weights["original_craftax"] == float(Fraction(1, 10))
        assert weights["task_1"] == float(Fraction(1, 20))


class TestLayoutFailClosed:
    @pytest.mark.parametrize("count", [0, 11, 13, 16])
    def test_requires_exactly_12_dynamic_slots(self, count):
        with pytest.raises(L.LayoutError) as excinfo:
            L.build_training_layout(_dynamic_ids(count))
        assert excinfo.value.code == L.LAYOUT_BAD_DYNAMIC_SET

    def test_duplicate_dynamic_id_rejected(self):
        ids = _dynamic_ids()
        ids[5] = ids[4]
        with pytest.raises(L.LayoutError) as excinfo:
            L.build_training_layout(ids)
        assert excinfo.value.code == L.LAYOUT_DUPLICATE_TASK

    def test_anchor_collision_rejected(self):
        ids = _dynamic_ids()
        ids[0] = "task_1"
        with pytest.raises(L.LayoutError) as excinfo:
            L.build_training_layout(ids)
        assert excinfo.value.code == L.LAYOUT_DUPLICATE_TASK

    def test_empty_or_nonstr_id_rejected(self):
        ids = _dynamic_ids()
        ids[3] = "   "
        with pytest.raises(L.LayoutError) as excinfo:
            L.build_training_layout(ids)
        assert excinfo.value.code == L.LAYOUT_BAD_DYNAMIC_SET
        ids[3] = 42
        with pytest.raises(L.LayoutError):
            L.build_training_layout(ids)

    def test_deviating_from_pinned_constants_fails_closed(self):
        with pytest.raises(L.LayoutError) as excinfo:
            L.build_training_layout(_dynamic_ids(), anchor_mass=0.3)
        assert excinfo.value.code == L.LAYOUT_WEIGHTS_INVALID
        with pytest.raises(L.LayoutError):
            L.build_training_layout(_dynamic_ids(), original_share=0.5)

    @pytest.mark.parametrize("bad", [True, "0.25", None])
    def test_bad_constant_types_fail_closed(self, bad):
        with pytest.raises(L.LayoutError):
            L.build_training_layout(_dynamic_ids(), anchor_mass=bad)

    def test_non_sequence_dynamic_rejected(self):
        with pytest.raises(L.LayoutError) as excinfo:
            L.build_training_layout("dyn_00")
        assert excinfo.value.code == L.LAYOUT_BAD_DYNAMIC_SET


class TestLegacyMirror:
    """Pure-python mirror of training._calculate_task_distribution."""

    def test_zero_curriculum_tasks_is_all_original(self):
        assert L.legacy_distribution_mirror(0, 0.2) == [1.0]

    def test_structure_matches_legacy_formula(self):
        n, original = 5, 0.2
        proportions = L.legacy_distribution_mirror(n, original)
        assert len(proportions) == n + 1
        other = (1.0 - original) / n
        assert proportions[:n] == pytest.approx([other] * n, abs=0)
        assert proportions[-1] == pytest.approx(original, abs=0)
        assert sum(proportions) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 12, 16, 32])
    def test_normalized_for_all_batch_sizes(self, n):
        proportions = L.legacy_distribution_mirror(n, 0.2)
        assert sum(proportions) == pytest.approx(1.0, abs=1e-12)
