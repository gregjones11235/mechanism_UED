"""Deterministic event extraction tests (section 4).

The required synthetic trace: unsafe_rest_near_hostile MUST be detected with
the full provenance field set.
"""
from d052.bagr_ued import constants as C
from d052.bagr_ued.event_extractor import (
    DEFAULT_DETECTORS,
    DeterministicEventExtractor,
    UnsafeRestNearHostileDetector,
)
from d052.bagr_ued.synthetic_traces import (
    TEST_VOCABULARY,
    build_unsafe_rest_raw_rollout,
)
from d052.bagr_ued.training_trace_adapter import TrainingTrajectoryEvidenceAdapter
from d052.bagr_ued.trajectory_evidence import (
    EvidenceSource,
    MockSymbolicAdapter,
)

REQUIRED_ANOMALY_FIELDS = (
    "anomaly_id", "episode_id", "evidence_span", "severity", "recurrence",
    "supporting_events", "counter_evidence", "detector_version",
    "detector_source_sha256")


def _bundle():
    adapter = TrainingTrajectoryEvidenceAdapter(MockSymbolicAdapter(TEST_VOCABULARY))
    return adapter.adapt(build_unsafe_rest_raw_rollout(),
                         bundle_id="t", source=EvidenceSource.SYNTHETIC_TEST_TRACE)


def test_minimum_detector_set_present():
    ids = sorted(d.detector_id for d in DEFAULT_DETECTORS)
    assert ids == sorted([
        "unsafe_rest_near_hostile", "repeated_no_effect_action",
        "oscillation_loop", "combat_freeze", "resource_neglect",
        "threat_approach_without_preparation", "progress_regression",
        "premature_terminal_behavior"])


def test_unsafe_rest_detected_with_full_provenance():
    anomalies = DeterministicEventExtractor().extract(_bundle())
    unsafe = [a for a in anomalies
              if a.behavior_pattern == "unsafe_rest_near_hostile"]
    assert unsafe, "unsafe_rest_near_hostile must fire on the synthetic trace"
    a = unsafe[0]
    for field in REQUIRED_ANOMALY_FIELDS:
        assert field in a.model_dump(), f"missing {field}"
    assert a.episode_id == "ep_unsafe_rest_01"
    assert a.severity > 0.0 and a.recurrence >= 1
    assert len(a.detector_source_sha256) == 64
    types = {e.event_type for e in a.supporting_events}
    assert "rest_near_hostile" in types
    assert types & {"damage_taken", "died", "chased"}
    assert a.evidence_span.start_step <= a.evidence_span.end_step


def test_global_patterns_beyond_threat_axis_detected():
    patterns = {a.behavior_pattern
                for a in DeterministicEventExtractor().extract(_bundle())}
    # GLOBAL coverage: resource waste + action waste + terminal behavior, not
    # just the threat axis
    assert {"unsafe_rest_near_hostile", "repeated_no_effect",
            "resource_neglect", "premature_terminal"} <= patterns


def test_extraction_is_deterministic():
    ex = DeterministicEventExtractor()
    run1 = [a.anomaly_id for a in ex.extract(_bundle())]
    run2 = [a.anomaly_id for a in ex.extract(_bundle())]
    assert run1 == run2


def test_unsafe_rest_counter_evidence_path():
    """A safe rest / far-hostile rest is NOT flagged (counter-evidence path)."""
    from d052.bagr_ued.trajectory_evidence import (
        EpisodeEvidence, StepRecord)
    steps = [
        StepRecord(step_index=0, symbolic_action="REST",
                   action_semantic_classes=["rest_class"],
                   state_summary=dict(hostile_distance_band="far",
                                      env_confirmed_safe=False,
                                      health_band="high"),
                   env_events=[]),
        StepRecord(step_index=1, symbolic_action="REST",
                   action_semantic_classes=["rest_class"],
                   state_summary=dict(hostile_distance_band="none",
                                      env_confirmed_safe=True,
                                      health_band="high"),
                   env_events=[]),
    ]
    ep = EpisodeEvidence(episode_id="calm",
                         source=EvidenceSource.SYNTHETIC_TEST_TRACE,
                         steps=steps, outcome="timeout")
    assert UnsafeRestNearHostileDetector().detect(ep) == []


def test_no_hardcoded_action_integers_in_detectors():
    import inspect
    from d052.bagr_ued import event_extractor
    src = inspect.getsource(event_extractor)
    # detectors key off semantic classes + symbolic bands, never raw ints
    assert "rest_class" in src and "hostile_distance_band" in src
    assert "action_int" not in src
