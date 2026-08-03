"""C6 tests: deterministic invocation gate (8 triggers, else REUSE)."""
import pytest

from dicode.teachers.e1_formal import invocation_gate as G


def _raw(**over):
    base = {
        "session_idx": 5,
        "is_first_window": False,
        "capability_shift": False,
        "new_failure_pattern": False,
        "interventions_exhausted": False,
        "stagnation": False,
        "forgetting_regression": False,
        "exploration_slot_available": False,
        "curriculum_drift": False,
    }
    base.update(over)
    return base


_FIELD_BY_CODE = {
    G.FIRST_WINDOW: "is_first_window",
    G.CAPABILITY_SHIFT: "capability_shift",
    G.NEW_FAILURE_PATTERN: "new_failure_pattern",
    G.INTERVENTIONS_EXHAUSTED: "interventions_exhausted",
    G.STAGNATION: "stagnation",
    G.FORGETTING_REGRESSION: "forgetting_regression",
    G.EXPLORATION_SLOT_AVAILABLE: "exploration_slot_available",
    G.CURRICULUM_DRIFT: "curriculum_drift",
}


class TestGateEvaluation:
    @pytest.mark.parametrize("code", G.GATE_TRIGGER_ORDER)
    def test_each_condition_alone_triggers_its_code(self, code):
        state = G.build_gate_state(_raw(**{_FIELD_BY_CODE[code]: True}), "t")
        decision = G.evaluate_invocation_gate(state)
        assert decision.triggered is True
        assert decision.code == code
        assert decision.session_idx == 5

    def test_no_condition_means_reuse_with_zero_triggers(self):
        state = G.build_gate_state(_raw(), "t")
        decision = G.evaluate_invocation_gate(state)
        assert decision.triggered is False
        assert decision.code == G.REUSE

    def test_priority_first_true_wins(self):
        # everything true -> FIRST_WINDOW (top of the fixed order)
        all_true = _raw(**{f: True for f in _FIELD_BY_CODE.values()})
        state = G.build_gate_state(all_true, "t")
        assert G.evaluate_invocation_gate(state).code == G.FIRST_WINDOW
        # stagnation + curriculum_drift -> STAGNATION (higher priority)
        state = G.build_gate_state(
            _raw(stagnation=True, curriculum_drift=True), "t"
        )
        assert G.evaluate_invocation_gate(state).code == G.STAGNATION

    def test_evaluation_is_deterministic(self):
        state = G.build_gate_state(_raw(new_failure_pattern=True), "t")
        assert G.evaluate_invocation_gate(state) == G.evaluate_invocation_gate(state)

    def test_non_gate_state_rejected(self):
        with pytest.raises(G.InvocationGateError) as excinfo:
            G.evaluate_invocation_gate(_raw())
        assert excinfo.value.code == "INVOCATION_GATE_BAD_TYPE"


class TestGateStateConsumption:
    def test_all_fields_required(self):
        for field in _FIELD_BY_CODE.values():
            raw = _raw()
            del raw[field]
            with pytest.raises(G.InvocationGateError) as excinfo:
                G.build_gate_state(raw, "t")
            assert excinfo.value.code == "INVOCATION_GATE_MISSING_FIELD"

    def test_unknown_field_rejected(self):
        with pytest.raises(G.InvocationGateError) as excinfo:
            G.build_gate_state(_raw(extra="x"), "t")
        assert excinfo.value.code == "INVOCATION_GATE_UNKNOWN_FIELD"

    @pytest.mark.parametrize("bad", [1, 0, "true", None])
    def test_bool_only_no_coercion(self, bad):
        with pytest.raises(G.InvocationGateError) as excinfo:
            G.build_gate_state(_raw(stagnation=bad), "t")
        assert excinfo.value.code == "INVOCATION_GATE_BAD_TYPE"

    @pytest.mark.parametrize("bad", [True, -1, "3", None])
    def test_session_idx_strict(self, bad):
        with pytest.raises(G.InvocationGateError) as excinfo:
            G.build_gate_state(_raw(session_idx=bad), "t")
        assert excinfo.value.code == "INVOCATION_GATE_BAD_STEP"

    def test_bad_container_rejected(self):
        with pytest.raises(G.InvocationGateError) as excinfo:
            G.build_gate_state(["not", "a", "mapping"], "t")
        assert excinfo.value.code == "INVOCATION_GATE_BAD_TYPE"
