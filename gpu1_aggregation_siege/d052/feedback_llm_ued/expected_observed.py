"""ExpectedObservedComparator — hypothesis-vs-probe scoring (task §2.4).

Each hypothesis carries a PREDICTED signature: the coarse episode-level
metrics we expect a discriminating probe to produce IF the hypothesis is
true (and optionally a counter-signature for the REFUTED case). After the
simulator probe runs, this comparator scores expected-vs-observed per metric
and emits a direction verdict:

    agree    — observation is consistent with the prediction
    opposite — observation is closer to the counter-prediction than to the
               prediction (probe ran against the hypothesis)
    neutral  — too close to call

Rules (deterministic, documented here and nowhere else):

* Metrics are compared only where BOTH sides have a value; unknown metric
  names are skipped but logged in the detail block (never silently coerced).
* Relative gap uses ``max(|expected|, floor)`` in the denominator so a
  predicted zero does not divide.
* Without a counter-signature: agree if rel_gap <= tol, neutral if
  rel_gap <= 2*tol, opposite beyond that.
* With a counter-signature: whichever side (predicted / counter) the
  observation is closer to wins, but a winner is only declared when the
  margin over the loser exceeds ``tol`` in relative terms; otherwise neutral.
* Overall direction: MAJORITY over the compared metrics — agree when more
  metrics agree than oppose, opposite when more oppose than agree, neutral on
  a tie. A verdict with zero comparable metrics is neutral + ``no_overlap=True``.

The comparator itself performs NO mutation; the caller binds the verdict to
the feedback record through ``SimulatorFeedbackStore.bind_match``.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import ProbeMetrics
from d052.feedback_llm_ued.simulator_feedback_store import SimulatorFeedbackStore

_DIVISION_FLOOR = 1e-6


def relative_gap(expected: float, observed: float) -> float:
    return abs(observed - expected) / max(abs(expected), _DIVISION_FLOOR)


def observable_metrics(metrics: ProbeMetrics) -> Dict[str, float]:
    """The episode-level observables a predicted signature may address."""
    return {
        "student_success_rate": metrics.student_success_rate,
        "student_behavior_activation": metrics.student_behavior_activation,
        "student_front_progress": metrics.student_front_progress,
        "reference_success_rate": metrics.reference_success_rate,
        "reference_mean_progress": metrics.reference_mean_progress,
        "reference_behavior_activation": metrics.reference_behavior_activation,
        "global_retention": metrics.global_retention,
        "regret": metrics.regret,
        "learnability": metrics.learnability,
    }


class ExpectedObservedComparator:
    """Pure scoring of predicted-signature vs observed probe metrics."""

    def __init__(self, tol: float = C.COMPARATOR_RELATIVE_TOLERANCE) -> None:
        if tol <= 0:
            raise ValueError(f"ILLEGAL_COMPARATOR_TOLERANCE: {tol!r}")
        self.tol = tol

    def compare(self, expected: Mapping[str, float],
                observed: Mapping[str, float], *,
                counter: Optional[Mapping[str, float]] = None) -> dict:
        """Return a detail block: per-metric verdicts + overall direction."""
        per_metric: Dict[str, dict] = {}
        skipped: List[str] = []
        for name, exp in sorted(expected.items()):
            if name not in observed:
                skipped.append(name)
                continue
            cnt = counter.get(name) if counter is not None else None
            per_metric[name] = dict(
                expected=float(exp),
                observed=float(observed[name]),
                counter=None if cnt is None else float(cnt),
                direction=self._one(float(exp), float(observed[name]), cnt),
            )
        directions = [v["direction"] for v in per_metric.values()]
        n_agree = sum(d == C.MATCH_DIRECTION_AGREE for d in directions)
        n_opposite = sum(d == C.MATCH_DIRECTION_OPPOSITE for d in directions)
        if not directions or n_agree == n_opposite:
            overall = C.MATCH_DIRECTION_NEUTRAL
        elif n_agree > n_opposite:
            overall = C.MATCH_DIRECTION_AGREE
        else:
            overall = C.MATCH_DIRECTION_OPPOSITE
        return dict(
            overall=overall,
            no_overlap=not directions,
            n_compared=len(directions),
            n_agree=sum(d == C.MATCH_DIRECTION_AGREE for d in directions),
            n_opposite=sum(d == C.MATCH_DIRECTION_OPPOSITE for d in directions),
            n_neutral=sum(d == C.MATCH_DIRECTION_NEUTRAL for d in directions),
            per_metric=per_metric,
            skipped_expected_metrics=skipped,
            tolerance=self.tol,
        )

    def grade_record(self, store: SimulatorFeedbackStore, feedback_id: str, *,
                     counter: Optional[Mapping[str, float]] = None) -> dict:
        """Score one stored feedback record and bind the verdict (hash-bound).

        The observation comes from the FULL probe when present, otherwise the
        fast probe. The prediction comes from the record's expected_signature
        unless overridden. A record with no probe metrics or no comparable
        prediction is graded NEUTRAL with an explicit reason, never dropped
        silently.
        """
        record = store.get(feedback_id)
        metrics = record.stage2_metrics or record.stage1_metrics
        if metrics is None:
            detail = dict(overall=C.MATCH_DIRECTION_NEUTRAL, no_overlap=True,
                          reason="NO_PROBE_METRICS")
            store.bind_match(feedback_id, direction=C.MATCH_DIRECTION_NEUTRAL,
                             detail=detail)
            return detail
        observed = observable_metrics(metrics)
        detail = self.compare(record.expected_signature, observed,
                              counter=counter)
        store.bind_match(feedback_id, direction=detail["overall"],
                         detail=detail)
        return detail

    def _one(self, expected: float, observed: float,
             counter: Optional[float]) -> str:
        if counter is not None:
            d_pred = abs(observed - expected)
            d_counter = abs(observed - counter)
            if d_pred == d_counter:
                return C.MATCH_DIRECTION_NEUTRAL
            margin = (abs(d_counter - d_pred)
                      / max(abs(expected - counter), _DIVISION_FLOOR))
            if margin <= self.tol:
                return C.MATCH_DIRECTION_NEUTRAL
            return (C.MATCH_DIRECTION_AGREE if d_pred < d_counter
                    else C.MATCH_DIRECTION_OPPOSITE)
        gap = relative_gap(expected, observed)
        if gap <= self.tol:
            return C.MATCH_DIRECTION_AGREE
        if gap <= 2.0 * self.tol:
            return C.MATCH_DIRECTION_NEUTRAL
        return C.MATCH_DIRECTION_OPPOSITE
