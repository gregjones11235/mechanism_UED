"""Typed, provenance-safe primitives for simulator-centric frontier work.

This package is intentionally a foundation layer.  It does not start PPO,
call an LLM, execute branch rollouts, or claim training/evaluation results.
"""

from .goals import (
    AchievementGoal,
    CompositeGoal,
    GateProgressGoal,
    GoalEvaluation,
    GoalSpec,
    StateFact,
    StateFactsGoal,
    TerminalEventGoal,
    Comparison,
    GoalStatus,
    goal_hash,
    evaluate_goal,
)
from .terminal_events import TerminalEventAdapter, TerminalTransition
from .state_codec import EncodedState, StateBundle, StateCodec
from .archive_schema import FrontierArchiveEntry
from .frontier_archive import FrontierArchive
from .search_statistics import BranchOutcome, FeasibilityEstimate, estimate_feasibility
from .memory_modes import (
    MemoryCompatibilityReport,
    MemoryRestoreMode,
    MemoryRestoreRequest,
    MemoryRestoreResult,
    validate_memory_request,
)
from .provenance import (
    DataSource,
    FormalDataLeakageGuard,
    SearchActionLeakageGuard,
)

__all__ = [
    "AchievementGoal", "CompositeGoal", "GateProgressGoal", "GoalEvaluation",
    "GoalSpec", "StateFact", "StateFactsGoal", "TerminalEventGoal", "Comparison", "GoalStatus",
    "goal_hash", "evaluate_goal",
    "TerminalEventAdapter", "TerminalTransition", "EncodedState", "StateBundle",
    "StateCodec", "FrontierArchiveEntry", "FrontierArchive", "BranchOutcome",
    "FeasibilityEstimate", "estimate_feasibility", "MemoryCompatibilityReport",
    "MemoryRestoreMode", "MemoryRestoreRequest", "MemoryRestoreResult", "validate_memory_request", "DataSource",
    "FormalDataLeakageGuard", "SearchActionLeakageGuard",
]
