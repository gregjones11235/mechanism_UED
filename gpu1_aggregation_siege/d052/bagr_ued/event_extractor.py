"""Deterministic event extraction + plugin detectors (task section 4).

Pure deterministic extraction — NO LLM, NO randomness, NO hardcoded Craftax
action integers or state leaf indices (detectors read only symbolic action
CLASSES and symbolic state summary fields, resolved upstream by the external
symbolic adapter).

Every detector implements the EventDetector Protocol:

    detector_id: str
    version: str
    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]

Every AnomalyCandidate carries the full provenance the task requires:
anomaly_id, episode_id, evidence_span, severity, recurrence,
supporting_events, counter_evidence, detector_version, detector_source_sha256
(sha256 of the detector class source — behaviour cannot drift from identity).

Minimum detector set (all present, each in its own class):
  unsafe_rest_near_hostile / repeated_no_effect_action / oscillation_loop /
  combat_freeze / resource_neglect / threat_approach_without_preparation /
  progress_regression / premature_terminal_behavior
"""
from __future__ import annotations

from typing import Dict, List, Protocol, runtime_checkable

from pydantic import Field, model_validator

from d052.bagr_ued import behavior_taxonomy as T
from d052.bagr_ued.hashing import canonical_sha256, source_sha256
from d052.bagr_ued.trajectory_evidence import (
    EpisodeEvidence,
    EvidenceSpan,
    StepRecord,
    TrajectoryEvidenceBundle,
)
from d052.schemas.common import CanonicalModel, validate_finite


class EventRecord(CanonicalModel):
    """One atomic event supporting an anomaly (symbolic, step-anchored)."""

    event_type: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    fields: Dict[str, str] = Field(default_factory=dict)


class AnomalyCandidate(CanonicalModel):
    """A detector-raised anomaly with full provenance (task section 4)."""

    anomaly_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    behavior_pattern: str = Field(min_length=1)
    evidence_span: EvidenceSpan
    severity: float = Field(ge=0.0, le=1.0)
    recurrence: int = Field(ge=1)
    supporting_events: List[EventRecord] = Field(default_factory=list)
    counter_evidence: List[str] = Field(default_factory=list)
    detector_version: str = Field(min_length=1)
    detector_source_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _finite(self) -> "AnomalyCandidate":
        validate_finite(self.severity, "severity")
        return self

    @property
    def span_hash(self) -> str:
        return canonical_sha256(self.evidence_span.span_hash_payload)


@runtime_checkable
class EventDetector(Protocol):
    detector_id: str
    version: str

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]: ...


# ---------------------------------------------------------------------------
# shared helpers (symbolic fields only — never raw ints/leaf indices)
# ---------------------------------------------------------------------------

def _steps(ep: EpisodeEvidence) -> List[StepRecord]:
    return sorted(ep.steps, key=lambda s: s.step_index)


def _has_class(step: StepRecord, cls: str) -> bool:
    return cls in step.action_semantic_classes


def _hostile_band(step: StepRecord) -> str:
    return str(step.state_summary.get("hostile_distance_band", "none"))


def _env_safe(step: StepRecord) -> bool:
    return bool(step.state_summary.get("env_confirmed_safe", False))


def _later_events(steps: List[StepRecord], start_idx: int, horizon: int,
                  wanted: frozenset) -> List[EventRecord]:
    found: List[EventRecord] = []
    for s in steps:
        if s.step_index < start_idx or s.step_index > start_idx + horizon:
            continue
        for ev in s.env_events:
            if ev in wanted:
                found.append(EventRecord(event_type=ev, step_index=s.step_index,
                                         fields={"action": s.symbolic_action}))
    return found


class _DetectorBase:
    detector_id = "base"
    version = "v1"
    behavior_pattern = "base"

    def _candidate(self, ep: EpisodeEvidence, span: EvidenceSpan, *,
                   severity: float, recurrence: int,
                   supporting: List[EventRecord],
                   counter: List[str]) -> AnomalyCandidate:
        aid = (f"{self.detector_id}:{ep.episode_id}:{span.start_step}-"
               f"{span.end_step}:{canonical_sha256([e.model_dump() for e in supporting])[:8]}")
        return AnomalyCandidate(
            anomaly_id=aid,
            episode_id=ep.episode_id,
            behavior_pattern=self.behavior_pattern,
            evidence_span=span,
            severity=round(float(severity), 6),
            recurrence=int(recurrence),
            supporting_events=supporting,
            counter_evidence=counter,
            detector_version=self.version,
            detector_source_sha256=source_sha256(type(self)),
        )


# ---------------------------------------------------------------------------
# 1. unsafe_rest_near_hostile (the required synthetic-test detector)
# ---------------------------------------------------------------------------

class UnsafeRestNearHostileDetector(_DetectorBase):
    """REST/SLEEP-class action + hostile nearby + env NOT confirmed safe,
    followed (within horizon) by damage/chase/death."""

    detector_id = "unsafe_rest_near_hostile"
    version = "v1"
    behavior_pattern = "unsafe_rest_near_hostile"
    HORIZON = 8
    OUTCOME_SEVERITY = (("died", 1.0), ("damage_taken", 0.8), ("chased", 0.6))

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        incidents: List[EventRecord] = []
        rest_steps: List[StepRecord] = []
        counter: List[str] = []
        worst = 0.0
        for s in steps:
            if not _has_class(s, "rest_class"):
                continue
            if _hostile_band(s) not in ("adjacent", "near"):
                counter.append(
                    f"step {s.step_index}: rest with no nearby hostile "
                    f"(band={_hostile_band(s)}) — not counted")
                continue
            if _env_safe(s):
                counter.append(
                    f"step {s.step_index}: rest while env confirmed safe — "
                    f"not counted")
                continue
            follow = _later_events(steps, s.step_index, self.HORIZON,
                                   frozenset({"died", "damage_taken", "chased"}))
            if not follow:
                counter.append(
                    f"step {s.step_index}: risky rest but no subsequent "
                    f"damage/chase/death within {self.HORIZON} steps")
                continue
            rest_steps.append(s)
            incidents.append(EventRecord(
                event_type="rest_near_hostile", step_index=s.step_index,
                fields={"hostile_band": _hostile_band(s),
                        "action": s.symbolic_action,
                        "env_confirmed_safe": str(_env_safe(s))}))
            incidents.extend(follow)
            for ev, sev in self.OUTCOME_SEVERITY:
                if any(e.event_type == ev for e in follow):
                    worst = max(worst, sev)
                    break
        if not rest_steps:
            return []
        span = EvidenceSpan(episode_id=episode.episode_id,
                            start_step=rest_steps[0].step_index,
                            end_step=max(e.step_index for e in incidents))
        return [self._candidate(episode, span, severity=worst,
                                recurrence=len(rest_steps),
                                supporting=incidents, counter=counter)]


# ---------------------------------------------------------------------------
# 2. repeated_no_effect_action
# ---------------------------------------------------------------------------

class RepeatedNoEffectActionDetector(_DetectorBase):
    """Runs of >=3 consecutive no_effect events (tool misuse / semantics gap)."""

    detector_id = "repeated_no_effect_action"
    version = "v1"
    behavior_pattern = "repeated_no_effect"
    MIN_RUN = 3

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        out: List[AnomalyCandidate] = []
        run: List[StepRecord] = []
        for s in steps + [None]:
            if s is not None and "no_effect" in s.env_events:
                run.append(s)
                continue
            if len(run) >= self.MIN_RUN:
                evs = [EventRecord(event_type="no_effect", step_index=r.step_index,
                                   fields={"action": r.symbolic_action})
                       for r in run]
                sev = min(1.0, 0.3 + 0.1 * len(run))
                span = EvidenceSpan(episode_id=episode.episode_id,
                                    start_step=run[0].step_index,
                                    end_step=run[-1].step_index)
                out.append(self._candidate(
                    episode, span, severity=sev, recurrence=len(run),
                    supporting=evs,
                    counter=["no counter-evidence: each listed action produced "
                             "no observable state change"]))
            run = []
        return out


# ---------------------------------------------------------------------------
# 3. oscillation_loop
# ---------------------------------------------------------------------------

class OscillationLoopDetector(_DetectorBase):
    """Persistent A/B/A/B alternation (>=6 steps) without progress."""

    detector_id = "oscillation_loop"
    version = "v1"
    behavior_pattern = "oscillation_loop"
    MIN_LENGTH = 6

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        out: List[AnomalyCandidate] = []
        i = 0
        n = len(steps)
        while i < n - 1:
            a, b = steps[i].symbolic_action, steps[i + 1].symbolic_action
            if a == b:
                i += 1
                continue
            j = i
            ok = True
            while j < n:
                expected = a if (j - i) % 2 == 0 else b
                if steps[j].symbolic_action != expected:
                    break
                j += 1
            length = j - i
            if length >= self.MIN_LENGTH:
                prog = [s for s in steps[i:j]
                        if any(ev.startswith("achievement_") for ev in s.env_events)]
                out.append(self._candidate(
                    episode,
                    EvidenceSpan(episode_id=episode.episode_id,
                                 start_step=steps[i].step_index,
                                 end_step=steps[j - 1].step_index),
                    severity=min(1.0, 0.4 + 0.05 * length),
                    recurrence=length // 2,
                    supporting=[EventRecord(event_type="oscillation_pair",
                                            step_index=steps[i].step_index,
                                            fields={"a": a, "b": b,
                                                    "length": str(length)})],
                    counter=(["progress events inside loop window: "
                             + ", ".join(str(p.step_index) for p in prog)]
                             if prog else
                             ["no progress event inside loop window"])))
                i = j
                ok = False
            else:
                i += 1
            _ = ok
        return out


# ---------------------------------------------------------------------------
# 4. combat_freeze
# ---------------------------------------------------------------------------

class CombatFreezeDetector(_DetectorBase):
    """Hostile adjacent while a NON-combat action repeats >=4 steps under
    ongoing damage."""

    detector_id = "combat_freeze"
    version = "v1"
    behavior_pattern = "combat_freeze"
    MIN_REPEAT = 4

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        out: List[AnomalyCandidate] = []
        run: List[StepRecord] = []
        for s in steps + [None]:
            cond = (s is not None and _hostile_band(s) == "adjacent"
                    and not _has_class(s, "combat_class"))
            if cond and (not run or run[-1].symbolic_action == s.symbolic_action):
                run.append(s)
                continue
            if len(run) >= self.MIN_REPEAT:
                damaged = [r for r in run if "damage_taken" in r.env_events]
                out.append(self._candidate(
                    episode,
                    EvidenceSpan(episode_id=episode.episode_id,
                                 start_step=run[0].step_index,
                                 end_step=run[-1].step_index),
                    severity=min(1.0, 0.5 + 0.1 * len(damaged)),
                    recurrence=len(run),
                    supporting=[EventRecord(event_type="non_combat_repeat",
                                            step_index=r.step_index,
                                            fields={"action": r.symbolic_action,
                                                    "damage": str("damage_taken"
                                                                  in r.env_events)})
                                for r in run],
                    counter=[f"damage taken during {len(damaged)}/{len(run)} "
                             f"repeated non-combat steps"]))
            run = [s] if cond else []
        return out


# ---------------------------------------------------------------------------
# 5. resource_neglect
# ---------------------------------------------------------------------------

class ResourceNeglectDetector(_DetectorBase):
    """Critical resource need persists >=N steps with no resource-class action."""

    detector_id = "resource_neglect"
    version = "v1"
    behavior_pattern = "resource_neglect"
    MIN_STEPS = 6

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        out: List[AnomalyCandidate] = []
        run: List[StepRecord] = []
        for s in steps + [None]:
            cond = (s is not None
                    and str(s.state_summary.get("resource_band")) in
                    ("critical", "depleted")
                    and not _has_class(s, "resource_class"))
            if cond:
                run.append(s)
                continue
            if len(run) >= self.MIN_STEPS:
                out.append(self._candidate(
                    episode,
                    EvidenceSpan(episode_id=episode.episode_id,
                                 start_step=run[0].step_index,
                                 end_step=run[-1].step_index),
                    severity=min(1.0, 0.3 + 0.05 * len(run)),
                    recurrence=len(run),
                    supporting=[EventRecord(event_type="need_unmet",
                                            step_index=r.step_index,
                                            fields={"resource_band": str(
                                                r.state_summary.get(
                                                    "resource_band")),
                                                "action": r.symbolic_action})
                                for r in run],
                    counter=["no resource-class action inside window"]))
            run = [s] if cond else []
        return out


# ---------------------------------------------------------------------------
# 6. threat_approach_without_preparation
# ---------------------------------------------------------------------------

class ThreatApproachWithoutPreparationDetector(_DetectorBase):
    """Hostile distance closes far/none -> near/adjacent within K steps with no
    preparation (equipment/resource-class) action in the approach window."""

    detector_id = "threat_approach_without_preparation"
    version = "v1"
    behavior_pattern = "unprepared_threat_approach"
    WINDOW = 6

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        out: List[AnomalyCandidate] = []
        for i, s in enumerate(steps):
            if _hostile_band(s) not in ("near", "adjacent"):
                continue
            prev = steps[i - 1] if i > 0 else None
            if prev is None or _hostile_band(prev) not in ("far", "none"):
                continue
            window = steps[max(0, i - self.WINDOW):i]
            prep = [w for w in window
                    if _has_class(w, "equipment_class") or _has_class(w, "resource_class")]
            if prep:
                continue
            out.append(self._candidate(
                episode,
                EvidenceSpan(episode_id=episode.episode_id,
                             start_step=window[0].step_index if window
                             else s.step_index,
                             end_step=s.step_index),
                severity=0.7 if _hostile_band(s) == "adjacent" else 0.5,
                recurrence=1,
                supporting=[EventRecord(
                    event_type="unprepared_approach", step_index=s.step_index,
                    fields={"from_band": _hostile_band(prev),
                            "to_band": _hostile_band(s)})],
                counter=[f"no preparation action in last {len(window)} steps"]))
        return out


# ---------------------------------------------------------------------------
# 7. progress_regression
# ---------------------------------------------------------------------------

class ProgressRegressionDetector(_DetectorBase):
    """Progress ordinal drops by >=1 and stays down across a window."""

    detector_id = "progress_regression"
    version = "v1"
    behavior_pattern = "progress_regression"
    HOLD = 4

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        steps = _steps(episode)
        out: List[AnomalyCandidate] = []
        progs = [(s.step_index, int(s.state_summary.get("progress_ordinal", 0)))
                 for s in steps if "progress_ordinal" in s.state_summary]
        i = 0
        while i < len(progs) - 1:
            idx0, p0 = progs[i]
            idx1, p1 = progs[i + 1]
            if p1 >= p0:
                i += 1
                continue
            # hold check: next HOLD samples stay <= p1
            tail = progs[i + 2: i + 2 + self.HOLD]
            holds = all(p <= p1 for _, p in tail) and len(tail) >= 1
            if not holds:
                i += 1
                continue
            last_idx = tail[-1][0] if tail else idx1
            out.append(self._candidate(
                episode,
                EvidenceSpan(episode_id=episode.episode_id,
                             start_step=idx0, end_step=last_idx),
                severity=min(1.0, 0.4 + 0.15 * (p0 - p1)),
                recurrence=1,
                supporting=[EventRecord(event_type="progress_drop",
                                        step_index=idx1,
                                        fields={"from": str(p0), "to": str(p1)})],
                counter=[f"progress stayed <= {p1} for {len(tail)} further "
                         f"samples (sustained, not transient)"]))
            i += 2 + len(tail)
        return out


# ---------------------------------------------------------------------------
# 8. premature_terminal_behavior
# ---------------------------------------------------------------------------

class PrematureTerminalBehaviorDetector(_DetectorBase):
    """Episode ends in death with no recent progress gain and a terminal
    high-risk behavior (risky rest or walking adjacent to a hostile)."""

    detector_id = "premature_terminal_behavior"
    version = "v1"
    behavior_pattern = "premature_terminal"
    TAIL = 8

    def detect(self, episode: EpisodeEvidence) -> List[AnomalyCandidate]:
        if episode.outcome != "death":
            return []
        steps = _steps(episode)
        if not steps:
            return []
        tail = steps[-self.TAIL:]
        gained = any(any(ev.startswith("achievement_") or ev == "progress_gain"
                         for ev in s.env_events) for s in tail)
        if gained:
            return []
        risky = [s for s in tail
                 if (_has_class(s, "rest_class")
                     and _hostile_band(s) in ("adjacent", "near")
                     and not _env_safe(s))
                 or _hostile_band(s) == "adjacent"]
        if not risky:
            return []
        return [self._candidate(
            episode,
            EvidenceSpan(episode_id=episode.episode_id,
                         start_step=tail[0].step_index,
                         end_step=tail[-1].step_index),
            severity=0.8,
            recurrence=len(risky),
            supporting=[EventRecord(event_type="terminal_risk",
                                    step_index=s.step_index,
                                    fields={"action": s.symbolic_action,
                                            "hostile_band": _hostile_band(s)})
                        for s in risky],
            counter=["no progress/achievement event in terminal window"])]


DEFAULT_DETECTORS: tuple = (
    UnsafeRestNearHostileDetector(),
    RepeatedNoEffectActionDetector(),
    OscillationLoopDetector(),
    CombatFreezeDetector(),
    ResourceNeglectDetector(),
    ThreatApproachWithoutPreparationDetector(),
    ProgressRegressionDetector(),
    PrematureTerminalBehaviorDetector(),
)


class DeterministicEventExtractor:
    """Runs registered plugin detectors over every episode of a bundle.

    Deterministic: same bundle -> same anomalies (anomaly ids are content
    hashes; detector provenance is source-hashed).
    """

    def __init__(self, detectors: tuple = DEFAULT_DETECTORS) -> None:
        for d in detectors:
            assert isinstance(d, EventDetector), f"BAD_DETECTOR:{d!r}"
            pats = T.patterns_for_detector(d.detector_id)
            assert pats, f"UNREGISTERED_DETECTOR_ID:{d.detector_id}"
        self.detectors = tuple(detectors)

    def extract(self, bundle: TrajectoryEvidenceBundle) -> List[AnomalyCandidate]:
        anomalies: List[AnomalyCandidate] = []
        for ep in sorted(bundle.episodes, key=lambda e: e.episode_id):
            for det in self.detectors:
                found = det.detect(ep)
                for a in found:
                    assert a.behavior_pattern in T.BEHAVIOR_PATTERNS, \
                        f"UNKNOWN_BEHAVIOR_PATTERN:{a.behavior_pattern}"
                anomalies.extend(found)
        anomalies.sort(key=lambda a: (a.episode_id, a.evidence_span.start_step,
                                      a.anomaly_id))
        return anomalies

    def detector_manifest(self) -> List[dict]:
        return [dict(detector_id=d.detector_id, version=d.version,
                     detector_source_sha256=source_sha256(type(d)))
                for d in self.detectors]
