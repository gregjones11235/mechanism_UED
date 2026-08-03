from dicode.simulator_frontier.goals import (
    AchievementGoal, CompositeGoal, GateProgressGoal, GoalStatus, StateFact,
    StateFactsGoal, TerminalEventGoal, evaluate_goal,
)


def test_achievement_is_not_gate_string_guess():
    result = evaluate_goal(AchievementGoal("defeat_kobold"), terminal_state={"achievements": {"defeat_kobold": True}})
    assert result.status is GoalStatus.SATISFIED


def test_state_facts_all_any_and_unknown():
    state = {"inventory": {"iron": 2}, "floor": 2}
    assert evaluate_goal(StateFactsGoal((StateFact("inventory.iron", "ge", 1),), mode="all"), terminal_state=state).satisfied
    any_goal = StateFactsGoal((StateFact("inventory.iron", "lt", 1), StateFact("floor", "eq", 2)), mode="any")
    assert evaluate_goal(any_goal, terminal_state=state).satisfied
    assert evaluate_goal(GateProgressGoal("missing", 1), terminal_state=state).status is GoalStatus.UNKNOWN


def test_terminal_event_and_composite_hash_are_deterministic():
    goal = CompositeGoal((AchievementGoal("defeat_kobold"), TerminalEventGoal("achievement_unlocked", "defeat_kobold")), mode="any")
    result = evaluate_goal(goal, events=[{"type": "achievement_unlocked", "value": "defeat_kobold"}])
    assert result.satisfied and len(result.goal_hash) == 64
