import pytest

from dicode.simulator_frontier.terminal_events import TerminalEventAdapter


def test_autoreset_preserves_terminal_state():
    t = TerminalEventAdapter().adapt(previous_state={"ep": 1}, action_metadata={}, reward=1, done=True,
                                     terminal_state={"achievement": "defeat_kobold"},
                                     returned_state={"ep": 2}, reset_state={"ep": 2})
    assert t.terminal_state["achievement"] == "defeat_kobold"
    assert t.returned_state["ep"] == 2
    assert TerminalEventAdapter().goal_state(t)["achievement"] == "defeat_kobold"


def test_done_autoreset_without_terminal_evidence_fails_closed():
    with pytest.raises(ValueError):
        TerminalEventAdapter().adapt(previous_state={}, action_metadata={}, reward=0, done=True,
                                     returned_state={}, reset_state={})
