"""Stage 2a-1: real invocation-gate signal computation (round-3 P0-3).

Computes the eight review-window trigger signals from REAL facts —
training-window evidence, the previous review window and teacher
session counters — replacing the former literal-False constants of the
edge teacher. Honesty rules:

* a signal whose data producer does not exist in this worktree is
  COMPUTED False with reason ``SIGNAL_NO_PRODUCER`` (runtime state,
  never a hardcoded constant pretending to have looked);
* a signal whose available facts are insufficient is COMPUTED False
  with reason ``SIGNAL_INSUFFICIENT_FACTS``;
* while the invocation thresholds are not frozen by the supervisor,
  every threshold-driven signal is False with reason
  ``INVOCATION_THRESHOLD_MISSING`` (thresholds have NO defaults);
* ``is_first_window`` is the only signal derivable from the teacher's
  own cycle counter alone;
* the full report is bound to its inputs (session, cycle counter,
  evidence hash, previous window hash, reuse counter, threshold
  version, every signal value and every reason) via ``binding_hash``,
  which enters the ``GateState`` as ``signals_binding_hash``.

Fact access: the evidence snapshot keeps only facts hashes by design;
the caller therefore passes the SAME raw items it built the snapshot
from, and this module re-binds them fail-closed against the
snapshot's per-item ``facts_sha`` before reading anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .canonical import canonical_sha256
from .evidence import EvidenceSnapshot
from .schemas import E1SchemaError

# per-signal reason codes (runtime states, greppable)
SIGNAL_OK = "SIGNAL_OK"
SIGNAL_NO_PRODUCER = "SIGNAL_NO_PRODUCER"
SIGNAL_INSUFFICIENT_FACTS = "SIGNAL_INSUFFICIENT_FACTS"
INVOCATION_THRESHOLD_MISSING = "INVOCATION_THRESHOLD_MISSING"

#: gate-state boolean field names produced by this module (exactly the
#: invocation_gate trigger fields; kept in the gate's priority order)
SIGNAL_FIELD_ORDER = (
    "is_first_window",
    "capability_shift",
    "new_failure_pattern",
    "interventions_exhausted",
    "stagnation",
    "forgetting_regression",
    "exploration_slot_available",
    "curriculum_drift",
)

#: threshold fields — every one REQUIRED, no defaults (fail-closed)
_THRESHOLD_FIELDS = (
    "capability_shift_delta",
    "stagnation_max_delta",
    "stagnation_min_sessions",
    "forgetting_regression_drop",
    "intervention_exhaustion_max_reuses",
    "exploration_slot_period",
)


class GateSignalError(E1SchemaError):
    """Fail-closed gate-signal violation; ``code`` is greppable."""


class _GSCode:
    BAD_TYPE = "GATE_SIGNALS_BAD_TYPE"
    MISSING_FIELD = "GATE_SIGNALS_MISSING_FIELD"
    UNKNOWN_FIELD = "GATE_SIGNALS_UNKNOWN_FIELD"
    OUT_OF_RANGE = "GATE_SIGNALS_OUT_OF_RANGE"
    FACTS_BINDING_MISMATCH = "GATE_SIGNALS_FACTS_BINDING_MISMATCH"


@dataclass(frozen=True)
class InvocationThresholds:
    """Supervisor-frozen gate thresholds (no defaults anywhere)."""

    capability_shift_delta: float
    stagnation_max_delta: float
    stagnation_min_sessions: int
    forgetting_regression_drop: float
    intervention_exhaustion_max_reuses: int
    exploration_slot_period: int


@dataclass(frozen=True)
class GateSignalReport:
    """The computed signals plus per-signal reasons and the binding."""

    session_idx: int
    signals: Tuple[Tuple[str, bool], ...]  # SIGNAL_FIELD_ORDER
    reasons: Tuple[Dict[str, str], ...]  # {signal, status, detail}
    binding_hash: str

    def signal(self, field: str) -> bool:
        for name, value in self.signals:
            if name == field:
                return value
        raise GateSignalError(
            _GSCode.MISSING_FIELD, f"unknown gate signal field {field!r}"
        )


def _require_number(value: Any, ctx: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateSignalError(
            _GSCode.BAD_TYPE,
            f"{ctx}: threshold {field!r} must be a number, got {value!r} "
            "(no defaults)",
        )
    return float(value)


def _require_count(value: Any, ctx: str, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateSignalError(
            _GSCode.BAD_TYPE,
            f"{ctx}: threshold {field!r} must be an int, got {value!r} "
            "(no defaults)",
        )
    if value < minimum:
        raise GateSignalError(
            _GSCode.OUT_OF_RANGE,
            f"{ctx}: threshold {field!r} must be >= {minimum}, got {value}",
        )
    return value


def consume_gate_thresholds(
    block: Any, ctx: str
) -> InvocationThresholds:
    """Consume the threshold block fail-closed; null/missing => blocked.

    Raises ``GateSignalError(INVOCATION_THRESHOLD_MISSING)`` naming the
    first absent field — the caller records that code as an honest
    block reason instead of guessing any threshold value.
    """
    if not isinstance(block, Mapping):
        raise GateSignalError(
            _GSCode.BAD_TYPE,
            f"{ctx}: thresholds must be a mapping, got "
            f"{type(block).__name__}",
        )
    unknown = sorted(k for k in block if k not in _THRESHOLD_FIELDS)
    if unknown:
        raise GateSignalError(
            _GSCode.UNKNOWN_FIELD,
            f"{ctx}: unknown threshold field(s) {unknown}",
        )
    for field in _THRESHOLD_FIELDS:
        if field not in block or block[field] is None:
            raise GateSignalError(
                INVOCATION_THRESHOLD_MISSING,
                f"{ctx}: threshold {field!r} is not frozen (no defaults; "
                "gate signals depending on it stay False with an explicit "
                "reason until the supervisor freezes the value)",
            )
    capability_shift_delta = _require_number(
        block["capability_shift_delta"], ctx, "capability_shift_delta"
    )
    if not (0.0 < capability_shift_delta <= 1.0):
        raise GateSignalError(
            _GSCode.OUT_OF_RANGE,
            f"{ctx}: capability_shift_delta must be in (0, 1], got "
            f"{capability_shift_delta!r}",
        )
    stagnation_max_delta = _require_number(
        block["stagnation_max_delta"], ctx, "stagnation_max_delta"
    )
    if not (0.0 <= stagnation_max_delta <= 1.0):
        raise GateSignalError(
            _GSCode.OUT_OF_RANGE,
            f"{ctx}: stagnation_max_delta must be in [0, 1], got "
            f"{stagnation_max_delta!r}",
        )
    forgetting_regression_drop = _require_number(
        block["forgetting_regression_drop"], ctx, "forgetting_regression_drop"
    )
    if not (0.0 < forgetting_regression_drop <= 1.0):
        raise GateSignalError(
            _GSCode.OUT_OF_RANGE,
            f"{ctx}: forgetting_regression_drop must be in (0, 1], got "
            f"{forgetting_regression_drop!r}",
        )
    return InvocationThresholds(
        capability_shift_delta=capability_shift_delta,
        stagnation_max_delta=stagnation_max_delta,
        stagnation_min_sessions=_require_count(
            block["stagnation_min_sessions"], ctx,
            "stagnation_min_sessions", 2,
        ),
        forgetting_regression_drop=forgetting_regression_drop,
        intervention_exhaustion_max_reuses=_require_count(
            block["intervention_exhaustion_max_reuses"], ctx,
            "intervention_exhaustion_max_reuses", 1,
        ),
        exploration_slot_period=_require_count(
            block["exploration_slot_period"], ctx,
            "exploration_slot_period", 1,
        ),
    )


# ---------------------------------------------------------------------------
# fact access: re-bind the caller's raw items against the snapshot's
# per-item facts hashes (fail-closed), then read facts ONLY from there
# ---------------------------------------------------------------------------
def _bind_facts(
    evidence: EvidenceSnapshot, raw_items: Sequence[Mapping[str, Any]], ctx: str
) -> Tuple[Mapping[str, Any], ...]:
    if len(raw_items) != len(evidence.items):
        raise GateSignalError(
            _GSCode.FACTS_BINDING_MISMATCH,
            f"{ctx}: {len(raw_items)} raw items != "
            f"{len(evidence.items)} snapshot items",
        )
    facts_list = []
    for i, (item, raw) in enumerate(zip(evidence.items, raw_items)):
        facts = raw.get("facts") if isinstance(raw, Mapping) else None
        if not isinstance(facts, Mapping):
            raise GateSignalError(
                _GSCode.FACTS_BINDING_MISMATCH,
                f"{ctx}: raw item [{i}] carries no facts mapping",
            )
        try:
            recomputed = canonical_sha256(dict(facts))
        except E1SchemaError as e:
            raise GateSignalError(
                _GSCode.FACTS_BINDING_MISMATCH,
                f"{ctx}: raw item [{i}] facts not canonical-encodable "
                f"({e.code})",
            ) from e
        if recomputed != item.facts_sha:
            raise GateSignalError(
                _GSCode.FACTS_BINDING_MISMATCH,
                f"{ctx}: raw item [{i}] facts hash != snapshot facts_sha "
                "(the facts offered to the gate are not the facts that "
                "were admitted into the evidence snapshot)",
            )
        facts_list.append(facts)
    return tuple(facts_list)


def _is_rate(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


def _success_rate_series(
    evidence: EvidenceSnapshot, facts_list: Sequence[Mapping[str, Any]]
) -> List[Tuple[int, float]]:
    """(session_idx, success_rate) pairs from all admissible items."""
    series: List[Tuple[int, float]] = []
    for item, facts in zip(evidence.items, facts_list):
        if "history" in facts and isinstance(facts["history"], (list, tuple)):
            for entry in facts["history"]:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and not isinstance(entry[0], bool)
                    and isinstance(entry[0], int)
                    and _is_rate(entry[1])
                ):
                    series.append((entry[0], float(entry[1])))
        if "success_rate" in facts and _is_rate(facts["success_rate"]):
            series.append((item.session_idx, float(facts["success_rate"])))
    series.sort(key=lambda pair: pair[0])
    return series


def _skill_series(
    evidence: EvidenceSnapshot, facts_list: Sequence[Mapping[str, Any]]
) -> Dict[str, List[Tuple[int, float]]]:
    """Per-skill (session_idx, rate) series from ``skill_*`` fact keys."""
    skills: Dict[str, List[Tuple[int, float]]] = {}
    for item, facts in zip(evidence.items, facts_list):
        for key, value in facts.items():
            if key.startswith("skill_") and _is_rate(value):
                skills.setdefault(key, []).append(
                    (item.session_idx, float(value))
                )
    for entries in skills.values():
        entries.sort(key=lambda pair: pair[0])
    return skills


def compute_gate_signals(
    *,
    session_idx: int,
    cycles_run: int,
    evidence: EvidenceSnapshot,
    raw_items: Sequence[Mapping[str, Any]],
    prev_window: Any,
    thresholds: Optional[InvocationThresholds],
    threshold_version: str,
    consecutive_reuses: int,
) -> GateSignalReport:
    """Compute all eight signals honestly; never fabricate a trigger.

    ``raw_items`` must be EXACTLY the sequence the evidence snapshot
    was built from (re-bound fail-closed against ``facts_sha``).
    ``prev_window`` is the previous ``ReviewWindow`` (or None);
    ``consecutive_reuses`` counts review cycles that produced no
    usable window since the last COMPLETE one.
    """
    ctx = "gate_signals"
    if isinstance(session_idx, bool) or not isinstance(session_idx, int):
        raise GateSignalError(
            _GSCode.BAD_TYPE, f"{ctx}: session_idx must be int, got "
            f"{session_idx!r}"
        )
    if isinstance(cycles_run, bool) or not isinstance(cycles_run, int):
        raise GateSignalError(
            _GSCode.BAD_TYPE, f"{ctx}: cycles_run must be int, got "
            f"{cycles_run!r}"
        )
    if isinstance(consecutive_reuses, bool) or not isinstance(
        consecutive_reuses, int
    ):
        raise GateSignalError(
            _GSCode.BAD_TYPE, f"{ctx}: consecutive_reuses must be int, got "
            f"{consecutive_reuses!r}"
        )
    facts_list = _bind_facts(evidence, raw_items, ctx)
    signals: Dict[str, bool] = {name: False for name in SIGNAL_FIELD_ORDER}
    reasons: Dict[str, Dict[str, str]] = {}

    def _set(field: str, value: bool, status: str, detail: str) -> None:
        signals[field] = value
        reasons[field] = {
            "signal": field,
            "status": status,
            "detail": detail,
        }

    # 1. FIRST_WINDOW — teacher cycle counter (always computable)
    _set(
        "is_first_window",
        cycles_run == 0,
        SIGNAL_OK,
        f"teacher cycle counter: cycles_run={cycles_run}",
    )

    # facts shared by the threshold-driven signals
    sr_series = _success_rate_series(evidence, facts_list)

    # 2. CAPABILITY_SHIFT — SR movement between earliest and latest
    if thresholds is None:
        _set("capability_shift", False, INVOCATION_THRESHOLD_MISSING,
             "capability_shift_delta not frozen")
    elif len(sr_series) < 2:
        _set("capability_shift", False, SIGNAL_INSUFFICIENT_FACTS,
             f"only {len(sr_series)} success-rate fact(s) < 2")
    else:
        shift = abs(sr_series[-1][1] - sr_series[0][1])
        _set(
            "capability_shift",
            shift >= thresholds.capability_shift_delta,
            SIGNAL_OK,
            f"|sr_last - sr_first| = {shift:.6f} vs delta "
            f"{thresholds.capability_shift_delta:.6f} over "
            f"{len(sr_series)} facts",
        )

    # 3. NEW_FAILURE_PATTERN — no fingerprint producer exists yet
    _set(
        "new_failure_pattern",
        False,
        SIGNAL_NO_PRODUCER,
        "no failure-pattern fingerprint producer in this worktree",
    )

    # 4. INTERVENTIONS_EXHAUSTED — consecutive windowless review cycles
    if thresholds is None:
        _set("interventions_exhausted", False, INVOCATION_THRESHOLD_MISSING,
             "intervention_exhaustion_max_reuses not frozen")
    else:
        _set(
            "interventions_exhausted",
            consecutive_reuses
            >= thresholds.intervention_exhaustion_max_reuses,
            SIGNAL_OK,
            f"consecutive_reuses={consecutive_reuses} vs max "
            f"{thresholds.intervention_exhaustion_max_reuses}",
        )

    # 5. STAGNATION — flat SR across the last N sessions
    if thresholds is None:
        _set("stagnation", False, INVOCATION_THRESHOLD_MISSING,
             "stagnation thresholds not frozen")
    elif len(sr_series) < thresholds.stagnation_min_sessions:
        _set(
            "stagnation",
            False,
            SIGNAL_INSUFFICIENT_FACTS,
            f"{len(sr_series)} success-rate fact(s) < "
            f"stagnation_min_sessions={thresholds.stagnation_min_sessions}",
        )
    else:
        window = [
            sr for _, sr in sr_series[-thresholds.stagnation_min_sessions:]
        ]
        span = max(window) - min(window)
        _set(
            "stagnation",
            span <= thresholds.stagnation_max_delta,
            SIGNAL_OK,
            f"SR span {span:.6f} over last {len(window)} facts vs max "
            f"delta {thresholds.stagnation_max_delta:.6f}",
        )

    # 6. FORGETTING_REGRESSION — a skill dropped below its earlier peak
    if thresholds is None:
        _set("forgetting_regression", False, INVOCATION_THRESHOLD_MISSING,
             "forgetting_regression_drop not frozen")
    else:
        skill_series = _skill_series(evidence, facts_list)
        usable = {
            name: entries
            for name, entries in skill_series.items()
            if len(entries) >= 2
        }
        if not usable:
            _set("forgetting_regression", False, SIGNAL_INSUFFICIENT_FACTS,
                 "no skill_* series with >= 2 facts")
        else:
            regressed = []
            for name, entries in sorted(usable.items()):
                peak_before_last = max(sr for _, sr in entries[:-1])
                drop = peak_before_last - entries[-1][1]
                if drop >= thresholds.forgetting_regression_drop:
                    regressed.append(name)
            _set(
                "forgetting_regression",
                len(regressed) > 0,
                SIGNAL_OK,
                (
                    f"regressed skills: {regressed}"
                    if regressed
                    else f"no skill dropped >= "
                    f"{thresholds.forgetting_regression_drop:.6f} below "
                    f"its earlier peak ({len(usable)} skill series)"
                ),
            )

    # 7. EXPLORATION_SLOT_AVAILABLE — periodic slot
    if thresholds is None:
        _set("exploration_slot_available", False,
             INVOCATION_THRESHOLD_MISSING,
             "exploration_slot_period not frozen")
    else:
        due = (
            session_idx > 0
            and session_idx % thresholds.exploration_slot_period == 0
        )
        _set(
            "exploration_slot_available",
            due,
            SIGNAL_OK,
            f"session_idx={session_idx} mod period "
            f"{thresholds.exploration_slot_period}",
        )

    # 8. CURRICULUM_DRIFT — no batch-composition history producer yet
    _set(
        "curriculum_drift",
        False,
        SIGNAL_NO_PRODUCER,
        "no batch-composition history producer in this worktree",
    )

    prev_window_hash = getattr(prev_window, "window_hash", "") or ""
    binding_hash = canonical_sha256(
        {
            "session_idx": session_idx,
            "cycles_run": cycles_run,
            "evidence_hash": evidence.evidence_hash,
            "prev_window_hash": prev_window_hash,
            "consecutive_reuses": consecutive_reuses,
            "threshold_version": threshold_version,
            "signals": {name: signals[name] for name in SIGNAL_FIELD_ORDER},
            "reasons": [reasons[name] for name in SIGNAL_FIELD_ORDER],
        }
    )
    return GateSignalReport(
        session_idx=session_idx,
        signals=tuple((name, signals[name]) for name in SIGNAL_FIELD_ORDER),
        reasons=tuple(reasons[name] for name in SIGNAL_FIELD_ORDER),
        binding_hash=binding_hash,
    )
