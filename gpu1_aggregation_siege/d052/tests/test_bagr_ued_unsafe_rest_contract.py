"""CC1 audit fix1 §2 — UnsafeRestNearHostileDetector contract regression tests.

The finding contract is CONDITION-BASED:

    finding exists  <=>  rest_or_sleep=true AND hostile_nearby=true AND
                         environment_confirmed_safe=false

Subsequent harm (damage_taken / chased / died) may ONLY enhance severity /
confidence / supporting_events and set realized_harm=true — it MUST NEVER
decide whether the finding exists.

Cases A-F from the fix task:
  A. REST+HOSTILE+NOT_SAFE+NO_HARM      -> 1 finding, realized_harm=false
  B. A + subsequent damage              -> 1 finding, realized_harm=true,
                                           severity/confidence >= no-harm case
  C. REST+NO_HOSTILE                    -> 0 findings
  D. REST+HOSTILE+CONFIRMED_SAFE        -> 0 findings
  E. NON_REST+HOSTILE+NOT_SAFE          -> 0 findings
  F. multiple unsafe rests, no harm     -> recurrence accumulates, no miss
"""
from __future__ import annotations

import pytest

from d052.bagr_ued.event_extractor import (
    DeterministicEventExtractor,
    UnsafeRestNearHostileDetector,
)
from d052.bagr_ued.trajectory_evidence import (
    EpisodeEvidence,
    EvidenceSource,
    StepRecord,
    TrajectoryEvidenceBundle,
)

DETECTOR = UnsafeRestNearHostileDetector()


def _step(i, action="REST", classes=("rest_class",), band="near",
          safe=False, events=()):
    return StepRecord(
        step_index=i,
        symbolic_action=action,
        action_semantic_classes=list(classes),
        state_summary={"hostile_distance_band": band,
                       "env_confirmed_safe": safe},
        env_events=list(events))


def _episode(steps, episode_id="ep_contract", outcome=None):
    return EpisodeEvidence(episode_id=episode_id,
                           source=EvidenceSource.SYNTHETIC_TEST_TRACE,
                           steps=steps, outcome=outcome)


# ---------------------------------------------------------------------------
# A. unsafe condition WITHOUT realized harm -> finding still exists
# ---------------------------------------------------------------------------

def test_case_a_unsafe_rest_without_harm_is_still_a_finding():
    ep = _episode([_step(0), _step(1), _step(2)])  # rest near hostile, no harm
    found = DETECTOR.detect(ep)
    assert len(found) == 1, "NO_HARM must not suppress the finding"
    f = found[0]
    assert f.anomaly_id.startswith("unsafe_rest_near_hostile:")
    assert f.behavior_pattern == "unsafe_rest_near_hostile"
    assert f.unsafe_condition_observed is True
    assert f.realized_harm is False
    assert f.severity == DETECTOR.BASE_SEVERITY
    assert f.confidence == DETECTOR.BASE_CONFIDENCE
    assert f.detector_version == "v2"
    # harm absence is documented as counter-evidence, not as a veto
    assert any("finding stands on the unsafe CONDITION alone" in c
               for c in f.counter_evidence)


# ---------------------------------------------------------------------------
# B. same condition WITH subsequent harm -> finding enhanced, not duplicated
# ---------------------------------------------------------------------------

def test_case_b_harm_only_enhances_severity_and_confidence():
    ep_noharm = _episode([_step(0), _step(1), _step(2)])
    ep_harm = _episode([_step(0), _step(1),
                        _step(2, events=("damage_taken",))])
    no_harm = DETECTOR.detect(ep_noharm)[0]
    harm = DETECTOR.detect(ep_harm)[0]

    assert harm.realized_harm is True
    assert harm.unsafe_condition_observed is True
    assert harm.severity >= no_harm.severity, \
        "harm may only raise severity"
    assert harm.confidence >= no_harm.confidence, \
        "harm may only raise confidence"
    assert harm.severity == DETECTOR.HARM_SEVERITY_LEVELS[1][1]  # damage_taken
    assert harm.confidence == pytest.approx(
        DETECTOR.BASE_CONFIDENCE + DETECTOR.HARM_CONFIDENCE_BONUS)
    # harm events SUPPLEMENT supporting events; condition events remain
    types = {e.event_type for e in harm.supporting_events}
    assert "rest_near_hostile" in types
    assert "damage_taken" in types


# ---------------------------------------------------------------------------
# C / D / E — the three ways the condition does NOT hold -> 0 findings
# ---------------------------------------------------------------------------

def test_case_c_rest_without_hostile_is_not_a_finding():
    ep = _episode([_step(0, band="none"), _step(1, band="far")])
    assert DETECTOR.detect(ep) == []


def test_case_d_rest_while_env_confirmed_safe_is_not_a_finding():
    ep = _episode([_step(0, safe=True), _step(1, safe=True)])
    assert DETECTOR.detect(ep) == []


def test_case_e_non_rest_action_near_hostile_is_not_a_finding():
    ep = _episode([_step(0, action="SCOUT", classes=("movement_class",)),
                   _step(1, action="SCOUT", classes=("movement_class",))])
    assert DETECTOR.detect(ep) == []


# ---------------------------------------------------------------------------
# F. multiple unsafe rests without harm -> recurrence accumulates, no miss
# ---------------------------------------------------------------------------

def test_case_f_repeated_unsafe_rest_accumulates_recurrence_without_harm():
    # three unsafe rest episodes, none followed by harm, interleaved with
    # non-rest filler steps
    steps = []
    for base in (0, 5, 10):
        steps.append(_step(base))
        steps.append(_step(base + 1, action="SCOUT",
                           classes=("movement_class",)))
    ep = _episode(steps)
    found = DETECTOR.detect(ep)
    assert len(found) == 1
    assert found[0].recurrence == 3, \
        "every unsafe rest must count toward recurrence even without harm"
    assert found[0].realized_harm is False
    cond_events = [e for e in found[0].supporting_events
                   if e.event_type == "rest_near_hostile"]
    assert len(cond_events) == 3
    assert all(e.fields["unsafe_condition_observed"] == "True"
               for e in cond_events)


# ---------------------------------------------------------------------------
# full-extractor sanity: the contract holds through the extractor pipeline
# (only unsafe_rest fires on these minimal traces)
# ---------------------------------------------------------------------------

def test_case_a_through_full_extractor_single_finding():
    bundle = TrajectoryEvidenceBundle(
        bundle_id="contract_check",
        source=EvidenceSource.SYNTHETIC_TEST_TRACE,
        symbolic_adapter_version="mock_symbolic_adapter.v1",
        episodes=[_episode([_step(0), _step(1), _step(2)])])
    anomalies = DeterministicEventExtractor().extract(bundle)
    assert len(anomalies) == 1
    assert anomalies[0].behavior_pattern == "unsafe_rest_near_hostile"
    assert anomalies[0].realized_harm is False
