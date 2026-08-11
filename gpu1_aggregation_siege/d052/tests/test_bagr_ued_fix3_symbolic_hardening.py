"""CC3 audit fix3 §4/§5/§6/§7/§8/§10 — symbolic clip hardening coverage.

Contract coverage added by fix3 on top of the fix2 symbolic clip suite:

  §4  a DICT symbolic clip payload gets its payload SHA RECOMPUTED from the
      dict content (a tampered dict fails closed; a dict with a missing /
      empty hash fails closed as CLIP_PAYLOAD_HASH_MISSING);
  §5  a TRUNCATED clip never fabricates terminal timing — the terminal
      category rides only on the episode's true last step, and a truncated
      window carries none;
  §6  safety_status is exactly one of safe / unsafe / unknown — anything
      else is refused at the schema; the builder maps env-confirmed signals
      and defaults missing evidence to "unknown";
  §7  the per-episode and per-review-window clip caps are enforced with an
      explicit, surfaced drop count (no silent truncation);
  §8  every provenance field is a LOWER-CASE full-64 sha256 hex digest —
      upper-case / truncated values are refused at the schema layer AND
      re-checked over serialized dicts;
  §10 the board consumes the injected payload batch verbatim — a batch of
      the wrong size or with tampered content fails closed.

SYNTHETIC unit tests only — no training, no real LLM, no rollout.
"""
from __future__ import annotations

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.behavior_clip_selector import BehaviorClip, BehaviorClipSelector
from d052.bagr_ued.event_extractor import AnomalyCandidate
from d052.bagr_ued.symbolic_behavior_clip import (
    SAFETY_STATUS_VOCABULARY,
    SymbolicBehaviorClipError,
    SymbolicBehaviorClipPayload,
    SymbolicBehaviorStep,
    build_symbolic_clip_payload,
    dict_clip_payload_hash,
    validate_symbolic_clip_payload,
)
from d052.bagr_ued.synthetic_traces import (
    TEST_VOCABULARY,
    build_unsafe_rest_raw_rollout,
)
from d052.bagr_ued.training_trace_adapter import TrainingTrajectoryEvidenceAdapter
from d052.bagr_ued.trajectory_evidence import (
    EpisodeEvidence,
    EvidenceSource,
    EvidenceSpan,
    MockSymbolicAdapter,
    StepRecord,
    TrajectoryEvidenceBundle,
)

SOURCE = EvidenceSource.SYNTHETIC_TEST_TRACE


def _evidence():
    adapter = TrainingTrajectoryEvidenceAdapter(
        MockSymbolicAdapter(TEST_VOCABULARY))
    bundle = adapter.adapt(build_unsafe_rest_raw_rollout(), bundle_id="t",
                           source=SOURCE)
    from d052.bagr_ued.event_extractor import DeterministicEventExtractor
    anomalies = DeterministicEventExtractor().extract(bundle)
    clips, dropped = BehaviorClipSelector().select(bundle, anomalies)
    return bundle, anomalies, clips, dropped


def _bundle_with_episode(n_steps, outcome, episode_id="ep_fix3"):
    steps = [StepRecord(step_index=i, symbolic_action="noop_symbolic",
                        action_semantic_classes=[],
                        state_summary={"health_band": "high"},
                        env_events=[])
             for i in range(n_steps)]
    ep = EpisodeEvidence(episode_id=episode_id, source=SOURCE, steps=steps,
                         outcome=outcome)
    return TrajectoryEvidenceBundle(
        bundle_id="fix3", source=SOURCE,
        symbolic_adapter_version="mock.symbolic.adapter.v1", episodes=[ep])


def _clip(episode_id, start, end, clip_id="clip:fix3"):
    return BehaviorClip(clip_id=clip_id, episode_id=episode_id,
                        span=EvidenceSpan(episode_id=episode_id,
                                          start_step=start, end_step=end),
                        reason_anomaly_ids=["an:fix3:1"])


def _anomaly(anomaly_id, episode_id, step):
    return AnomalyCandidate(
        anomaly_id=anomaly_id, episode_id=episode_id,
        behavior_pattern="unsafe_rest",
        evidence_span=EvidenceSpan(episode_id=episode_id,
                                   start_step=step, end_step=step),
        severity=0.5, recurrence=1,
        detector_version="fix3.test.v1",
        detector_source_sha256="a" * 64)


# ===========================================================================
# §4 — dict payloads get their SHA recomputed (never trusted)
# ===========================================================================

def test_dict_payload_sha_is_recomputed():
    bundle, _, clips, _ = _evidence()
    payload = build_symbolic_clip_payload(bundle, clips[0])
    dump = payload.model_dump()

    # an UNTOUCHED dict passes: recomputed SHA == recorded SHA
    assert dict_clip_payload_hash(dump) == payload.clip_payload_sha256
    assert validate_symbolic_clip_payload(dump)["passed"] is True

    # a tampered dict keeping the OLD recorded hash fails closed
    tampered = payload.model_dump()
    tampered["episode_id"] = "ep_tampered"
    report = validate_symbolic_clip_payload(tampered)
    assert report["passed"] is False
    assert any(f["code"] ==
               SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISMATCH
               for f in report["findings"])

    # a dict with a MISSING or EMPTY recorded hash fails closed — the SHA
    # must be recomputed and bound, never omitted
    missing = payload.model_dump()
    missing.pop("clip_payload_sha256")
    report = validate_symbolic_clip_payload(missing)
    assert report["passed"] is False
    assert any(f["code"] == SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISSING
               for f in report["findings"])
    empty = payload.model_dump()
    empty["clip_payload_sha256"] = ""
    report = validate_symbolic_clip_payload(empty)
    assert report["passed"] is False
    assert any(f["code"] == SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISSING
               for f in report["findings"])


# ===========================================================================
# §5 — truncated clips never fabricate terminal timing
# ===========================================================================

def test_truncated_clip_carries_no_terminal():
    n = C.MAX_CLIP_STEPS + 36        # guaranteed oversize window
    bundle = _bundle_with_episode(n, "death")
    clip = _clip("ep_fix3", 0, n - 1)
    payload = build_symbolic_clip_payload(bundle, clip)
    assert payload.truncation_applied is True
    assert len(payload.steps) == C.MAX_CLIP_STEPS
    # NO step may claim a terminal category — the true last step (death) is
    # outside the truncated window; fabricating it is forbidden
    assert all(s.terminal_category == "none" for s in payload.steps)
    assert validate_symbolic_clip_payload(payload)["passed"] is True


def test_untruncated_clip_carries_true_terminal_only_on_last_step():
    n = C.MAX_CLIP_STEPS - 4         # fits, no truncation
    bundle = _bundle_with_episode(n, "timeout")
    clip = _clip("ep_fix3", 0, n - 1)
    payload = build_symbolic_clip_payload(bundle, clip)
    assert payload.truncation_applied is False
    assert payload.steps[-1].terminal_category == "timeout"
    assert all(s.terminal_category == "none" for s in payload.steps[:-1])


# ===========================================================================
# §6 — safety_status vocabulary is exactly safe / unsafe / unknown
# ===========================================================================

def test_safety_vocabulary_refused_outside_schema():
    assert SAFETY_STATUS_VOCABULARY == frozenset({"safe", "unsafe", "unknown"})
    with pytest.raises(SymbolicBehaviorClipError,
                       match="CLIP_SAFETY_STATUS_INVALID"):
        SymbolicBehaviorStep(step_offset=0, safety_status="cautious")
    for ok in ("safe", "unsafe", "unknown"):
        SymbolicBehaviorStep(step_offset=0, safety_status=ok)


def test_builder_safety_mapping_is_evidence_bound():
    # env-confirmed signals decide; missing evidence -> "unknown"
    bundle = _bundle_with_episode(4, None)
    ep = bundle.episodes[0]
    ep.steps[0].state_summary = {"env_confirmed_safe": True}
    ep.steps[1].state_summary = {"env_confirmed_safe": False}
    ep.steps[2].state_summary = {"env_confirmed_unsafe": True}
    ep.steps[3].state_summary = {}                    # no evidence at all
    payload = build_symbolic_clip_payload(bundle, _clip("ep_fix3", 0, 3))
    assert [s.safety_status for s in payload.steps] == \
        ["safe", "unsafe", "unsafe", "unknown"]


def test_builder_never_emits_out_of_vocabulary_safety():
    bundle, _, clips, _ = _evidence()
    for c in clips:
        payload = build_symbolic_clip_payload(bundle, c)
        assert all(s.safety_status in SAFETY_STATUS_VOCABULARY
                   for s in payload.steps)


# ===========================================================================
# §8 — provenance must be lower-case full-64 sha256 hex (schema + dict)
# ===========================================================================

def test_provenance_format_refused_at_schema():
    bundle, _, clips, _ = _evidence()
    dump = build_symbolic_clip_payload(bundle, clips[0]).model_dump()

    upper = dict(dump)
    upper["student_checkpoint_sha256"] = \
        dump["student_checkpoint_sha256"].upper()
    with pytest.raises(SymbolicBehaviorClipError,
                       match="CLIP_PROVENANCE_FORMAT_INVALID"):
        SymbolicBehaviorClipPayload.model_validate(upper)

    truncated = dict(dump)
    truncated["taskparams_hash"] = "0" * 63
    with pytest.raises(SymbolicBehaviorClipError,
                       match="CLIP_PROVENANCE_FORMAT_INVALID"):
        SymbolicBehaviorClipPayload.model_validate(truncated)


def test_provenance_format_rechecked_over_serialized_dicts():
    bundle, _, clips, _ = _evidence()
    dump = build_symbolic_clip_payload(bundle, clips[0]).model_dump()

    # upper-case provenance in a dict that bypassed model construction
    forged = dict(dump)
    forged["environment_lock_sha256"] = \
        dump["environment_lock_sha256"].upper()
    report = validate_symbolic_clip_payload(forged)
    assert report["passed"] is False
    assert any(f["code"] ==
               SymbolicBehaviorClipError.CLIP_PROVENANCE_FORMAT_INVALID
               for f in report["findings"])

    # a missing provenance field fails the same re-check
    gone = dict(dump)
    gone.pop("rollout_runner_sha256")
    report = validate_symbolic_clip_payload(gone)
    assert report["passed"] is False
    assert any(f["code"] ==
               SymbolicBehaviorClipError.CLIP_PROVENANCE_FORMAT_INVALID
               for f in report["findings"])


# ===========================================================================
# §7 — per-episode and per-window clip caps, explicit drop count
# ===========================================================================

def _cap_bundle():
    ep1 = EpisodeEvidence(
        episode_id="ep_cap_a", source=SOURCE,
        steps=[StepRecord(step_index=i, symbolic_action="noop_symbolic",
                          state_summary={}) for i in range(60)],
        outcome=None)
    ep2 = EpisodeEvidence(
        episode_id="ep_cap_b", source=SOURCE,
        steps=[StepRecord(step_index=i, symbolic_action="noop_symbolic",
                          state_summary={}) for i in range(60)],
        outcome=None)
    return TrajectoryEvidenceBundle(
        bundle_id="caps", source=SOURCE,
        symbolic_adapter_version="mock.symbolic.adapter.v1",
        episodes=[ep1, ep2])


def test_per_episode_cap_enforced_with_drop_count():
    bundle = _cap_bundle()
    # 6 well-separated anomalies in ep_cap_a -> 6 merged windows
    anomalies = [_anomaly(f"an:a:{i}", "ep_cap_a", 5 + i * 10)
                 for i in range(6)]
    clips, dropped = BehaviorClipSelector().select(bundle, anomalies)
    assert len(clips) == C.MAX_CLIPS_PER_EPISODE
    assert dropped == 6 - C.MAX_CLIPS_PER_EPISODE
    # deterministic earliest-first: the kept clips are the earliest windows
    starts = [c.span.start_step for c in clips]
    assert starts == sorted(starts)
    assert starts[0] <= 5


def test_per_window_cap_enforced_over_all_episodes():
    bundle = _cap_bundle()
    anomalies = ([_anomaly(f"an:a:{i}", "ep_cap_a", 5 + i * 10)
                  for i in range(6)] +
                 [_anomaly(f"an:b:{i}", "ep_cap_b", 5 + i * 10)
                  for i in range(3)])
    sel = BehaviorClipSelector(max_clips_per_episode=2,
                               max_clips_per_window=3)
    clips, dropped = sel.select(bundle, anomalies)
    # ep_a keeps 2 (drops 4), ep_b keeps 2 (drops 1); 4 > window cap 3
    # -> one more dropped at the window level; nothing is silent
    assert len(clips) == 3
    assert dropped == 4 + 1 + 1
    for c in clips:
        assert c.episode_id in {"ep_cap_a", "ep_cap_b"}


def test_controller_dry_run_respects_caps_and_shares_batch():
    from d052.bagr_ued.controller import BAGRUEdController
    result = BAGRUEdController().run_dry_run(build_unsafe_rest_raw_rollout())
    payloads = result.symbolic_behavior_clips
    assert len(payloads) <= C.MAX_CLIPS_PER_REVIEW_WINDOW
    per_episode = {}
    for p in payloads:
        per_episode[p["episode_id"]] = per_episode.get(p["episode_id"], 0) + 1
    assert all(count <= C.MAX_CLIPS_PER_EPISODE
               for count in per_episode.values())
    assert result.clips_dropped == \
        result.dry_run_certificate["clips_dropped_by_caps"]
    assert result.dry_run_certificate[
        "board_and_certificate_share_clip_batch"] is True


# ===========================================================================
# §10 — the board consumes the injected batch verbatim, re-validated
# ===========================================================================

def test_board_injected_batch_size_must_match():
    from d052.bagr_ued.event_extractor import DeterministicEventExtractor
    from d052.bagr_ued.review_board import build_base_context
    bundle, anomalies, clips, _ = _evidence()
    manifest = DeterministicEventExtractor().detector_manifest()
    assert clips, "the synthetic trace yields clips"
    with pytest.raises(ValueError, match="SYMBOLIC_CLIP_BATCH_MISMATCH"):
        build_base_context(bundle, anomalies, clips, manifest,
                           symbolic_payload_dumps=[])   # wrong size


def test_board_injected_batch_is_revalidated_fail_closed():
    from d052.bagr_ued.event_extractor import DeterministicEventExtractor
    from d052.bagr_ued.review_board import build_base_context
    bundle, anomalies, clips, _ = _evidence()
    manifest = DeterministicEventExtractor().detector_manifest()
    dumps = [build_symbolic_clip_payload(bundle, c).model_dump()
             for c in clips]
    # the exact batch is accepted verbatim (shared-batch path)
    ctx = build_base_context(bundle, anomalies, clips, manifest,
                             symbolic_payload_dumps=dumps)
    assert ctx["symbolic_behavior_clips"] == dumps

    # a tampered injected payload (old hash over changed content) fails
    # closed at board intake — trust nothing
    bad = [dict(d) for d in dumps]
    bad[0] = dict(bad[0])
    bad[0]["episode_id"] = "ep_injected_tamper"
    with pytest.raises(SymbolicBehaviorClipError,
                       match="CLIP_PAYLOAD_HASH_MISMATCH"):
        build_base_context(bundle, anomalies, clips, manifest,
                           symbolic_payload_dumps=bad)
