"""FeedbackInvocationGate — when the (精简) LLM may be invoked (task §3/§5).

Default: 2 LLM calls per TRIGGERED window (FeedbackDiagnostician +
AdaptiveEnvironmentDesigner). If NONE of the eight must-invoke conditions
fires, the previous diagnosis and plan are REUSED unchanged and the generator
continues in the neighborhood of existing interventions — no LLM call.

The gate is pure and deterministic: it consumes a ``GateInput`` snapshot of
the window state and returns which conditions fired. It never calls the LLM
itself; the controller does, only when ``invoke_llm`` is true.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import Field

from d052.feedback_llm_ued import constants as C
from d052.schemas.common import CanonicalModel

#: threshold floors for the numeric conditions (documented here only)
CORE_BEHAVIOR_RATE_SHIFT_FLOOR = 0.25        # >= 25 pct change in a core rate
GLOBAL_RETENTION_REGRESSION_DELTA = -0.05    # retention dropped by >= 5 pct
FRONT_STALLED_WINDOW_COUNT = 2               # front flat for two windows
CACHED_PLAN_AGE_LIMIT = 4                    # cached plan older than 4 windows


class GateInput(CanonicalModel):
    """Window-state snapshot the gate judges on."""

    window: int = Field(ge=0)
    has_prior_diagnosis: bool = False
    new_detector_types: List[str] = Field(default_factory=list)
    #: max absolute rate change across core behaviors since previous window
    core_behavior_rate_change: float = Field(default=0.0, ge=0.0)
    #: consecutive windows the front metric has been flat
    front_stalled_windows: int = Field(default=0, ge=0)
    #: global_retention(this window) - global_retention(previous window)
    global_retention_delta: float = 0.0
    previous_plan_exhausted: bool = False
    valid_candidate_count: int = Field(default=0, ge=0)
    required_candidate_count: int = Field(default=C.DYNAMIC_UED_SLOTS, ge=0)
    cached_plan_age_windows: int = Field(default=0, ge=0)


def evaluate_gate(inp: GateInput) -> Dict[str, object]:
    """Return ``dict(invoke_llm=..., conditions=(...), reason=...)``.

    Condition order follows ``GATE_MUST_INVOKE_CONDITIONS`` so equal states
    always produce identical reports.
    """
    fired: List[Tuple[str, str]] = []
    if inp.window == 0 or not inp.has_prior_diagnosis:
        fired.append((C.GATE_FIRST_WINDOW,
                      "first window or no prior diagnosis on record"))
    if inp.new_detector_types:
        fired.append((C.GATE_NEW_DETECTOR_TYPE,
                      f"new detector types observed: "
                      f"{sorted(set(inp.new_detector_types))}"))
    if inp.core_behavior_rate_change >= CORE_BEHAVIOR_RATE_SHIFT_FLOOR:
        fired.append((C.GATE_CORE_BEHAVIOR_RATE_SHIFT,
                      f"core behavior rate change "
                      f"{inp.core_behavior_rate_change:.3f} >= "
                      f"{CORE_BEHAVIOR_RATE_SHIFT_FLOOR:.2f}"))
    if inp.front_stalled_windows >= FRONT_STALLED_WINDOW_COUNT:
        fired.append((C.GATE_FRONT_STALLED_TWO_WINDOWS,
                      f"front stalled for {inp.front_stalled_windows} "
                      f"windows (>= {FRONT_STALLED_WINDOW_COUNT})"))
    if inp.global_retention_delta <= GLOBAL_RETENTION_REGRESSION_DELTA:
        fired.append((C.GATE_GLOBAL_RETENTION_REGRESSION,
                      f"global retention delta "
                      f"{inp.global_retention_delta:.3f} <= "
                      f"{GLOBAL_RETENTION_REGRESSION_DELTA:.2f}"))
    if inp.previous_plan_exhausted:
        fired.append((C.GATE_PREVIOUS_PLAN_EXHAUSTED,
                      "previous plan's candidates are exhausted"))
    if inp.valid_candidate_count < inp.required_candidate_count:
        fired.append((C.GATE_INSUFFICIENT_VALID_CANDIDATES,
                      f"valid candidates {inp.valid_candidate_count} < "
                      f"required {inp.required_candidate_count}"))
    if inp.cached_plan_age_windows >= CACHED_PLAN_AGE_LIMIT:
        fired.append((C.GATE_CACHED_PLAN_AGE,
                      f"cached plan age {inp.cached_plan_age_windows} >= "
                      f"{CACHED_PLAN_AGE_LIMIT} windows"))
    conditions = tuple(name for name, _ in fired)
    if fired:
        reason = "must-invoke conditions fired: " + "; ".join(
            f"{name} ({detail})" for name, detail in fired)
    else:
        reason = ("no must-invoke condition fired; reuse previous diagnosis "
                  "and plan, generator continues in the neighborhood of "
                  "existing interventions")
    return dict(invoke_llm=bool(fired), conditions=conditions, reason=reason)
