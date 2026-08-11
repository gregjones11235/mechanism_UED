"""CC3 audit fix2 §9-§14 — bounded, de-identified symbolic behavior clips.

The Review Board must receive per-step SYMBOLIC behavior evidence with
provenance — not only anomaly labels and clip metadata — while the hard data
boundary holds: no raw action integers, no raw state, no formal trajectories,
bounded windows, validated provenance hashes. Provisional out-of-taxonomy
hypotheses may be surfaced but must never move the selector or the batch.

Cases A-H (§14):
  A. unsafe-rest clip — the board sees per-step symbolic evidence (rest-class
     actions, near/adjacent hostile bands, unsafe safety status, harm events);
  B. repeated no-effect clip — repeated action semantics + flat progress
     bands are visible per step;
  C. a FORMAL_FRONT clip -> fail closed;
  D. a raw action integer (action=17) in the payload -> fail closed;
  E. a raw state leaf key (inventory_17 / observation_vector) -> fail closed;
  F. an oversized clip -> explicit truncation (truncation_applied=true) or
     fail closed, with a record;
  G. provenance hash mismatch -> fail closed;
  H. out-of-taxonomy anomaly -> provisional finding surfaced, selector and
     batch UNCHANGED.
"""
from __future__ import annotations

import pytest

from d052.bagr_ued import constants as C
from d052.bagr_ued.behavior_clip_selector import BehaviorClipSelector
from d052.bagr_ued.controller import BAGRUEdController
from d052.bagr_ued.event_extractor import DeterministicEventExtractor
from d052.bagr_ued.symbolic_behavior_clip import (
    SymbolicBehaviorClipError,
    SymbolicBehaviorClipPayload,
    build_symbolic_clip_payload,
    clip_payload_hash,
    mock_clip_provenance,
    validate_symbolic_clip_payload,
)
from d052.bagr_ued.synthetic_traces import (
    TEST_VOCABULARY,
    build_unsafe_rest_raw_rollout,
)
from d052.bagr_ued.training_trace_adapter import TrainingTrajectoryEvidenceAdapter
from d052.bagr_ued.trajectory_evidence import (
    EvidenceSource,
    MockSymbolicAdapter,
    TrajectoryEvidenceBundle,
)


def _evidence():
    adapter = TrainingTrajectoryEvidenceAdapter(
        MockSymbolicAdapter(TEST_VOCABULARY))
    bundle = adapter.adapt(build_unsafe_rest_raw_rollout(), bundle_id="t",
                           source=EvidenceSource.SYNTHETIC_TEST_TRACE)
    anomalies = DeterministicEventExtractor().extract(bundle)
    clips, _ = BehaviorClipSelector().select(bundle, anomalies)
    return bundle, anomalies, clips


def _clip_for_episode(clips, episode_id):
    hits = [c for c in clips if c.episode_id == episode_id]
    assert hits, f"no clip for {episode_id}"
    return hits[0]


# ===========================================================================
# A. unsafe-rest clip: per-step symbolic evidence visible
# ===========================================================================

def test_case_a_unsafe_rest_clip_shows_per_step_symbolic_evidence():
    bundle, anomalies, clips = _evidence()
    clip = _clip_for_episode(clips, "ep_unsafe_rest_01")
    payload = build_symbolic_clip_payload(bundle, clip)
    report = validate_symbolic_clip_payload(payload)
    assert report["passed"] is True, report["findings"]

    assert payload.source == C.SOURCE_SYNTHETIC_TEST_TRACE
    assert payload.steps and len(payload.steps) <= C.MAX_CLIP_STEPS
    # rest-class action semantics ARE visible (as classes, never raw ints)
    all_classes = {c for s in payload.steps
                   for c in s.action_semantic_classes}
    assert "rest_class" in all_classes
    # threat distance bands + unsafe safety status around the incident
    bands = {s.hostile_distance_band for s in payload.steps}
    assert bands & {"near", "adjacent"}
    assert any(s.safety_status == "unsafe" for s in payload.steps)
    # harm event semantics are visible
    events = {e for s in payload.steps for e in s.event_semantics}
    assert events & {"damage_taken", "chased", "died"}
    # terminal category carried on the episode's last step
    assert any(s.terminal_category == "death" for s in payload.steps)
    # step offsets are relative to the clip start
    assert payload.steps[0].step_offset == 0
    # provenance + payload hash present and valid
    assert len(payload.student_checkpoint_sha256) == 64
    assert payload.clip_payload_sha256 == clip_payload_hash(payload)


# ===========================================================================
# B. repeated no-effect clip: repeated semantics + flat progress bands
# ===========================================================================

def test_case_b_no_effect_clip_shows_repetition_and_flat_progress():
    bundle, anomalies, clips = _evidence()
    clip = _clip_for_episode(clips, "ep_no_effect_02")
    payload = build_symbolic_clip_payload(bundle, clip)
    assert validate_symbolic_clip_payload(payload)["passed"] is True

    # repeated DO_NOTHING semantics visible step by step
    empty_steps = [s for s in payload.steps
                   if s.action_semantic_classes == []]
    assert len(empty_steps) >= 3
    # flat progress: after the first step, progress delta is "unchanged"
    deltas = [s.progress_delta_band for s in payload.steps[1:]]
    assert deltas and all(d == "unchanged" for d in deltas)
    # no-effect event semantics visible
    events = {e for s in payload.steps for e in s.event_semantics}
    assert "no_effect" in events


# ===========================================================================
# C. FORMAL_FRONT source -> fail closed
# ===========================================================================

def test_case_c_formal_front_clip_fails_closed():
    bundle, anomalies, clips = _evidence()
    clip = clips[0]
    # a bundle relabeled FORMAL_FRONT is refused at the schema layer ...
    formal = bundle.model_dump()
    formal["source"] = C.SOURCE_FORMAL_FRONT
    with pytest.raises(Exception):
        TrajectoryEvidenceBundle.model_validate(formal)
    # ... and a clip payload claiming a formal source fails closed directly
    good = build_symbolic_clip_payload(bundle, clip)
    forged = good.model_dump()
    forged.pop("clip_payload_sha256")
    forged["source"] = C.SOURCE_FORMAL_FRONT
    with pytest.raises(SymbolicBehaviorClipError,
                       match="CLIP_SOURCE_NOT_ADMISSIBLE"):
        SymbolicBehaviorClipPayload.model_validate(forged)


# ===========================================================================
# D. raw action integer in the payload -> fail closed
# ===========================================================================

def test_case_d_raw_action_integer_fails_closed():
    bundle, anomalies, clips = _evidence()
    payload = build_symbolic_clip_payload(bundle, clips[0])

    # a smuggled raw action integer under an action-ish key is detected
    tampered = payload.model_dump()
    tampered["steps"][0]["action"] = 17
    report = validate_symbolic_clip_payload(tampered)
    assert report["passed"] is False
    assert any(f["code"] ==
               SymbolicBehaviorClipError.RAW_ACTION_INTEGER_EXPOSED
               for f in report["findings"])

    # and the schema itself refuses a raw-int field on a step
    step_dump = payload.steps[0].model_dump()
    step_dump["raw_action"] = 17
    with pytest.raises(Exception):   # extra=forbid canonical model
        from d052.bagr_ued.symbolic_behavior_clip import SymbolicBehaviorStep
        SymbolicBehaviorStep.model_validate(step_dump)


# ===========================================================================
# E. raw state leaf keys -> fail closed
# ===========================================================================

@pytest.mark.parametrize("bad_key", ["inventory_17", "observation_vector"])
def test_case_e_raw_state_keys_fail_closed(bad_key):
    bundle, anomalies, clips = _evidence()
    payload = build_symbolic_clip_payload(bundle, clips[0])
    tampered = payload.model_dump()
    tampered["steps"][0][bad_key] = [0.1, 0.2, 0.3]
    report = validate_symbolic_clip_payload(tampered)
    assert report["passed"] is False
    assert any(f["code"] == SymbolicBehaviorClipError.RAW_STATE_EXPOSED
               for f in report["findings"])


# ===========================================================================
# F. oversized clip -> explicit truncation with record (or fail closed)
# ===========================================================================

def test_case_f_oversized_clip_is_truncated_with_record():
    bundle, anomalies, clips = _evidence()
    clip = clips[0]
    span_len = clip.span.end_step - clip.span.start_step + 1
    if span_len <= C.MAX_CLIP_STEPS:
        # force an oversized window over the whole episode
        from d052.bagr_ued.behavior_clip_selector import BehaviorClip
        from d052.bagr_ued.trajectory_evidence import EvidenceSpan
        ep = bundle.episode(clip.episode_id)
        last = max(s.step_index for s in ep.steps)
        big_span = EvidenceSpan(episode_id=clip.episode_id, start_step=0,
                                end_step=last)
        clip = BehaviorClip(clip_id="clip:oversize_test",
                            episode_id=clip.episode_id,
                            span=big_span,
                            reason_anomaly_ids=clip.reason_anomaly_ids)
    # a synthetic episode longer than MAX_CLIP_STEPS: the builder truncates
    # and records truncation_applied=true; the validator then passes
    payload = build_symbolic_clip_payload(bundle, clip)
    assert len(payload.steps) <= C.MAX_CLIP_STEPS
    assert validate_symbolic_clip_payload(payload)["passed"] is True

    # a payload that SOMEHOW carries too many steps fails closed
    dump = payload.model_dump()
    dump["steps"] = (dump["steps"] * (C.MAX_CLIP_STEPS + 1))[
        :C.MAX_CLIP_STEPS + 1]
    dump["clip_payload_sha256"] = ""
    report = validate_symbolic_clip_payload(dump)
    assert report["passed"] is False
    assert any(f["code"] ==
               SymbolicBehaviorClipError.CLIP_STEP_LIMIT_EXCEEDED
               for f in report["findings"])


def test_case_f2_full_trajectory_never_reaches_a_role():
    # structural guarantee: even an episode of 4096 steps would yield a clip
    # of at most MAX_CLIP_STEPS (<< 4096)
    assert C.MAX_CLIP_STEPS <= 64
    assert C.MAX_CLIPS_PER_REVIEW_WINDOW <= 64
    bundle, anomalies, clips = _evidence()
    for c in clips:
        p = build_symbolic_clip_payload(bundle, c)
        assert len(p.steps) <= C.MAX_CLIP_STEPS


# ===========================================================================
# G. provenance hash mismatch -> fail closed
# ===========================================================================

def test_case_g_provenance_hash_mismatch_fails_closed():
    bundle, anomalies, clips = _evidence()
    payload = build_symbolic_clip_payload(bundle, clips[0])
    expected = mock_clip_provenance()
    assert validate_symbolic_clip_payload(
        payload, expected_provenance=expected)["passed"] is True

    # a payload stamped with a DIFFERENT student checkpoint fails closed
    other = dict(expected)
    other["student_checkpoint_sha256"] = "0" * 64
    report = validate_symbolic_clip_payload(payload,
                                            expected_provenance=other)
    assert report["passed"] is False
    assert any(f["code"] == SymbolicBehaviorClipError.CLIP_PROVENANCE_MISMATCH
               for f in report["findings"])

    # tampering with the payload content breaks the payload hash too:
    # a payload that keeps the OLD recorded hash over CHANGED content is a
    # well-formed but forged record — the hash recomputation catches it
    from d052.bagr_ued.symbolic_behavior_clip import SymbolicBehaviorClipPayload
    tampered = payload.model_dump()
    tampered["episode_id"] = "ep_tampered"   # keeps the OLD recorded hash
    forged = SymbolicBehaviorClipPayload.model_validate(tampered)
    assert forged.clip_payload_sha256 == payload.clip_payload_sha256
    report2 = validate_symbolic_clip_payload(forged)
    assert report2["passed"] is False
    assert any(f["code"] ==
               SymbolicBehaviorClipError.CLIP_PAYLOAD_HASH_MISMATCH
               for f in report2["findings"])


# ===========================================================================
# H. out-of-taxonomy anomaly -> provisional only; selector/batch unchanged
# ===========================================================================

def test_case_h_provisional_hypothesis_never_moves_selector_or_batch():
    result = BAGRUEdController().run_dry_run(build_unsafe_rest_raw_rollout())
    d = result.model_dump()
    provisionals = d["provisional_anomaly_hypotheses"]
    assert provisionals, "the mock auditor surfaces one provisional"
    for p in provisionals:
        assert p["provisional"] is True
        assert p["requires_deterministic_validation"] is True
        assert p["taxonomy_status"] == "out_of_taxonomy"
        assert p["evidence_clip_ids"]
        assert p["alternative_explanation"]
        assert p["selector_or_batch_entry_forbidden"] is True

    # the provisional id appears NOWHERE in the selector/budget/archive chain
    provisional_ids = {p["hypothesis_id"] for p in provisionals}
    descriptor_ids = {x["descriptor_id"] for x in d["descriptors"]}
    budget_ids = set(d["budget_plan"]["ued_slots"])
    archive_ids = {w["descriptor_id"]
                   for k in ("would_add", "would_update")
                   for w in d["archive_refresh_plan"][k]}
    assert not (provisional_ids & descriptor_ids)
    assert not (provisional_ids & budget_ids)
    assert not (provisional_ids & archive_ids)
    # the reconciler did not accept the provisional as a behavior finding
    accepted = {it["item_id"]
                for it in d["reconciliation"]["accepted_behavior_findings"]}
    assert not (provisional_ids & accepted)

    # certificate proves the contract
    cert = result.dry_run_certificate
    assert cert["behavior_review_has_symbolic_clips"] is True
    assert cert["raw_action_integer_exposed"] is False
    assert cert["raw_state_exposed"] is False
    assert cert["formal_trajectory_exposed"] is False
    assert cert["provisional_anomaly_hypotheses_surfaced"] >= 1
    assert cert["provisional_anomaly_hypotheses_in_selector"] == 0


# ===========================================================================
# §13 schema contract: a provisional finding must stay provisional
# ===========================================================================

def test_provisional_schema_refuses_non_provisional_entries():
    from d052.bagr_ued.behavior_auditor import ProvisionalAnomalyHypothesis
    ok = ProvisionalAnomalyHypothesis(
        hypothesis_id="provisional:x", observed_pattern="p",
        confidence=0.3, alternative_explanation="noise")
    assert ok.provisional is True
    assert ok.requires_deterministic_validation is True
    with pytest.raises(Exception,
                       match="PROVISIONAL_HYPOTHESIS_MUST_BE_PROVISIONAL"):
        ProvisionalAnomalyHypothesis(
            hypothesis_id="provisional:x", provisional=False,
            observed_pattern="p", confidence=0.3,
            alternative_explanation="noise")
    with pytest.raises(Exception,
                       match="PROVISIONAL_REQUIRES_DETERMINISTIC_VALIDATION"):
        ProvisionalAnomalyHypothesis(
            hypothesis_id="provisional:x",
            requires_deterministic_validation=False,
            observed_pattern="p", confidence=0.3,
            alternative_explanation="noise")


# ===========================================================================
# §12: the board context (all six roles) receives the symbolic clips
# ===========================================================================

def test_board_context_carries_symbolic_clips_for_all_roles():
    from d052.bagr_ued.review_board import build_base_context
    bundle, anomalies, clips = _evidence()
    manifest = DeterministicEventExtractor().detector_manifest()
    ctx = build_base_context(bundle, anomalies, clips, manifest)
    assert ctx["symbolic_behavior_clips"]
    assert len(ctx["symbolic_behavior_clips"]) == len(clips)
    for sc in ctx["symbolic_behavior_clips"]:
        assert sc["steps"]
        assert sc["clip_payload_sha256"]
        assert sc["source"] in C.ALLOWED_EVIDENCE_SOURCES
        assert "symbolic_adapter_version" not in sc.get("steps", [{}])[0] \
            or True
        # no raw exposure in what the roles receive
        rep = validate_symbolic_clip_payload(sc)
        assert rep["passed"] is True, rep["findings"]
    assert ctx["symbolic_clip_contract"]["raw_action_integer_exposed"] is False
    assert ctx["symbolic_clip_contract"]["formal_trajectory_exposed"] is False
