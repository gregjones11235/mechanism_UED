"""Explicit, typed goal specifications and fail-closed evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "simulator_frontier.goal/v1"


class Comparison(str, Enum):
    EQ = "eq"
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"


class GoalStatus(str, Enum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    UNKNOWN = "UNKNOWN"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"


def _ensure_schema(version: str) -> None:
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported goal schema_version: {version!r}")


def _path_get(obj: Any, path: str) -> tuple[bool, Any]:
    if not path or path.startswith(".") or ".." in path:
        return False, None
    cur = obj
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif hasattr(cur, part) and not part.startswith("_"):
            cur = getattr(cur, part)
        else:
            return False, None
    return True, cur


def _compare(actual: Any, comparison: Comparison, expected: Any) -> bool:
    if comparison is Comparison.EQ:
        return actual == expected
    if comparison is Comparison.GE:
        return actual >= expected
    if comparison is Comparison.GT:
        return actual > expected
    if comparison is Comparison.LE:
        return actual <= expected
    if comparison is Comparison.LT:
        return actual < expected
    raise AssertionError(comparison)


@dataclass(frozen=True)
class StateFact:
    path: str
    comparison: Comparison | str
    value: Any

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith(".") or ".." in self.path:
            raise ValueError("StateFact.path must be a simple dotted field path")
        object.__setattr__(self, "comparison", Comparison(self.comparison))


class GoalSpec:
    schema_version: str


@dataclass(frozen=True)
class AchievementGoal(GoalSpec):
    achievement: str
    required: bool = True
    schema_version: str = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_schema(self.schema_version)
        if not self.achievement or not isinstance(self.achievement, str):
            raise ValueError("achievement must be a non-empty string")


@dataclass(frozen=True)
class GateProgressGoal(GoalSpec):
    gate_name: str
    threshold: float
    comparison: Comparison | str = Comparison.GE
    schema_version: str = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_schema(self.schema_version)
        if not self.gate_name:
            raise ValueError("gate_name must be non-empty")
        object.__setattr__(self, "comparison", Comparison(self.comparison))


@dataclass(frozen=True)
class StateFactsGoal(GoalSpec):
    facts: tuple[StateFact, ...]
    mode: str = "all"
    schema_version: str = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_schema(self.schema_version)
        facts = tuple(self.facts)
        if not facts or any(not isinstance(f, StateFact) for f in facts):
            raise ValueError("StateFactsGoal.facts must contain StateFact values")
        if self.mode not in {"all", "any"}:
            raise ValueError("StateFactsGoal.mode must be 'all' or 'any'")
        object.__setattr__(self, "facts", facts)


@dataclass(frozen=True)
class TerminalEventGoal(GoalSpec):
    event_type: str
    event_value: Any = None
    schema_version: str = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_schema(self.schema_version)
        if not self.event_type:
            raise ValueError("event_type must be non-empty")


@dataclass(frozen=True)
class CompositeGoal(GoalSpec):
    goals: tuple[GoalSpec, ...]
    mode: str = "all"
    schema_version: str = field(default=SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        _ensure_schema(self.schema_version)
        goals = tuple(self.goals)
        if not goals or any(not isinstance(g, GoalSpec) for g in goals):
            raise ValueError("CompositeGoal.goals must contain GoalSpec values")
        if self.mode not in {"all", "any"}:
            raise ValueError("CompositeGoal.mode must be 'all' or 'any'")
        object.__setattr__(self, "goals", goals)


@dataclass(frozen=True)
class GoalEvaluation:
    satisfied: bool
    status: GoalStatus
    matched_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    goal_hash: str
    evaluator_version: str = SCHEMA_VERSION


def _canonical(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, GoalSpec):
        if isinstance(value, StateFactsGoal):
            return {"type": type(value).__name__, "schema_version": value.schema_version,
                    "facts": [_canonical(f) for f in value.facts], "mode": value.mode}
        if isinstance(value, CompositeGoal):
            return {"type": type(value).__name__, "schema_version": value.schema_version,
                    "goals": [_canonical(g) for g in value.goals], "mode": value.mode}
        return {"type": type(value).__name__, **{k: _canonical(v) for k, v in vars(value).items()}}
    if isinstance(value, StateFact):
        return {"path": value.path, "comparison": value.comparison.value, "value": _canonical(value.value)}
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def goal_hash(goal: GoalSpec) -> str:
    blob = json.dumps(_canonical(goal), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _event_match(goal: TerminalEventGoal, events: Sequence[Any]) -> bool | None:
    found = False
    for event in events:
        if isinstance(event, Mapping):
            kind = event.get("event_type", event.get("type"))
            value = event.get("event_value", event.get("value"))
        else:
            kind = getattr(event, "event_type", getattr(event, "type", None))
            value = getattr(event, "event_value", getattr(event, "value", None))
        if kind == goal.event_type:
            found = True
            if goal.event_value is None or value == goal.event_value:
                return True
    return False if found else None


def _evaluate(goal: GoalSpec, *, previous_state: Any, terminal_state: Any,
              returned_state: Any, events: Sequence[Any], info: Mapping[str, Any]) -> tuple[GoalStatus, list[str], list[str]]:
    state = terminal_state if terminal_state is not None else returned_state
    if isinstance(goal, AchievementGoal):
        ok, value = _path_get(state, f"achievements.{goal.achievement}")
        if not ok:
            ok, value = _path_get(info, f"achievements.{goal.achievement}")
        if not ok:
            return GoalStatus.UNKNOWN, [], [f"achievements.{goal.achievement}"]
        return (GoalStatus.SATISFIED if bool(value) == goal.required else GoalStatus.NOT_SATISFIED,
                [goal.achievement] if bool(value) == goal.required else [], [] if bool(value) == goal.required else [goal.achievement])
    if isinstance(goal, GateProgressGoal):
        ok, value = _path_get(state, goal.gate_name)
        if not ok:
            ok, value = _path_get(info, goal.gate_name)
        if not ok:
            return GoalStatus.UNKNOWN, [], [goal.gate_name]
        try:
            hit = _compare(value, goal.comparison, goal.threshold)
        except (TypeError, ValueError):
            return GoalStatus.INVALID_EVIDENCE, [], [goal.gate_name]
        return (GoalStatus.SATISFIED if hit else GoalStatus.NOT_SATISFIED,
                [goal.gate_name] if hit else [], [] if hit else [goal.gate_name])
    if isinstance(goal, StateFactsGoal):
        statuses = []
        matched, missing = [], []
        for fact in goal.facts:
            ok, value = _path_get(state, fact.path)
            if not ok:
                missing.append(fact.path)
                statuses.append(None)
                continue
            try:
                hit = bool(_compare(value, fact.comparison, fact.value))
            except (TypeError, ValueError):
                return GoalStatus.INVALID_EVIDENCE, matched, missing + [fact.path]
            statuses.append(hit)
            (matched if hit else missing).append(fact.path)
        if any(v is None for v in statuses):
            return GoalStatus.UNKNOWN, matched, missing
        satisfied = all(statuses) if goal.mode == "all" else any(statuses)
        return (GoalStatus.SATISFIED if satisfied else GoalStatus.NOT_SATISFIED, matched if satisfied else [], missing if not satisfied else [])
    if isinstance(goal, TerminalEventGoal):
        hit = _event_match(goal, events)
        if hit is None:
            return GoalStatus.UNKNOWN, [], [goal.event_type]
        return (GoalStatus.SATISFIED if hit else GoalStatus.NOT_SATISFIED,
                [goal.event_type] if hit else [], [] if hit else [goal.event_type])
    if isinstance(goal, CompositeGoal):
        results = [_evaluate(g, previous_state=previous_state, terminal_state=terminal_state,
                             returned_state=returned_state, events=events, info=info) for g in goal.goals]
        statuses = [r[0] for r in results]
        if any(s is GoalStatus.INVALID_EVIDENCE for s in statuses):
            status = GoalStatus.INVALID_EVIDENCE
        elif goal.mode == "all" and any(s is GoalStatus.UNKNOWN for s in statuses):
            status = GoalStatus.UNKNOWN
        elif goal.mode == "any" and any(s is GoalStatus.SATISFIED for s in statuses):
            status = GoalStatus.SATISFIED
        elif goal.mode == "all" and all(s is GoalStatus.SATISFIED for s in statuses):
            status = GoalStatus.SATISFIED
        elif goal.mode == "any" and all(s is GoalStatus.NOT_SATISFIED for s in statuses):
            status = GoalStatus.NOT_SATISFIED
        else:
            status = GoalStatus.NOT_SATISFIED
        matched = [x for r in results for x in r[1]] if status is GoalStatus.SATISFIED else []
        missing = [x for r in results for x in r[2]]
        return status, matched, missing
    raise TypeError(f"unsupported GoalSpec: {type(goal).__name__}")


def evaluate_goal(goal: GoalSpec, *, previous_state: Any = None, terminal_state: Any = None,
                  returned_state: Any = None, events: Sequence[Any] = (), info: Mapping[str, Any] | None = None) -> GoalEvaluation:
    if not isinstance(goal, GoalSpec):
        raise TypeError("goal must be an explicit GoalSpec instance")
    status, matched, missing = _evaluate(goal, previous_state=previous_state, terminal_state=terminal_state,
                                         returned_state=returned_state, events=events, info=info or {})
    return GoalEvaluation(status is GoalStatus.SATISFIED, status, tuple(matched), tuple(missing), goal_hash(goal))
