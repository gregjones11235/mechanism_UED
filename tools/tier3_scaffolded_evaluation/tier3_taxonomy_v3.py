#!/usr/bin/env python3
"""CC4 Tier3 — V3 composite-event failure taxonomy (总控授权语义修复).

Authorized repair (task CC4_REPAIR_NEG20_COMPOSITE_EVENT_SEMANTICS_AND_COMPLETE_
FORMAL_EVALUATION_V3): the frozen single-label taxonomy `tier3_failure_taxonomy`
FAILS CLOSED (NEG20) on the legitimate composite terminal event
"front floor transition AND defeat_kobold", permanently blocking every policy
strong enough to descend 2->3 and then defeat the kobold. 总控 ruled this is a
VALID COMPOSITE EVENT, not a contradiction.

V3 representation (replaces mutually-exclusive single labels):
    primary_outcome     exactly one primary outcome per episode
    secondary_events[]  other legal events that co-occurred (sorted)
    taxonomy_status     VALID_COMPOSITE_EVENT / VALID_SINGLE_EVENT / INVALID_START

Per-arm semantics (总控 §二, verbatim application):
    FRONT : primary = FRONT_TRANSITION_SUCCESS iff front_floor_transition_reached;
            if defeat_kobold also holds, DEFEAT_KOBOLD is a SECONDARY event and
            taxonomy_status = VALID_COMPOSITE_EVENT (NEVER abort). A defeat with NO
            transition is NOT a front success (frozen FRONT primary predicate is the
            transition only) — recorded as a secondary event.
    BACK  : primary = BACK_DEFEAT_KOBOLD_SUCCESS iff defeat_kobold; other legal
            events (e.g. player_died on a same-step trade kill) are secondary.
    FULL  : primary = FULL_DEFEAT_KOBOLD_SUCCESS iff defeat_kobold; other legal
            events are secondary.

FailClosed is RETAINED (总控 §二) only for: state corruption, missing required
fields, contradictions UNEXPLAINABLE by event chronology, illegal values, evidence
hash mismatch, and unregistered event/scenario types. Legitimately co-occurring
events must NOT abort. The structurally-impossible conjunctions
(defeat+timed_out, died+timed_out — evaluator_v2 line 283 makes timed_out exclusive
of defeat/died; and the FRONT transition True / corridor_exit_reached False alias
contradiction — both signals derive from max_level>=3) cannot arise from real
rollouts, so their presence is corruption/contradiction and still fails closed.
defeat+died on one step IS chronology-explainable (a trade kill) and is a composite.

This module is JAX-free. It reuses the FROZEN `tier3_metrics.summarize` as a library
(zero taxonomy dependency, verified) so primary/dense metrics are BIT-IDENTICAL to
V2 by construction; the composite layer is additive annotation. It imports the
frozen `tier3_failure_taxonomy` ONLY inside the self-test to prove the repair is
additive (frozen V1 still raises NEG20 on the composite; V3 does not).

Protocol ids (new — never masquerade as V2):
    FORMAL_EVALUATOR_PROTOCOL = V3_COMPOSITE_EVENT
    NEG20_PROTOCOL            = NEG20_V3_PRIMARY_SECONDARY_EVENTS
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_metrics as metrics            # noqa: E402  (FROZEN; reused as a library)

# ---------------------------------------------------------------------------
# Protocol / version constants (NEW; distinct from V2)
# ---------------------------------------------------------------------------
TAXONOMY_V3_VERSION = "tier3_taxonomy_v3/v1"
FAILURE_RULE_VERSION_V3 = "tier3_failure_rules_v3/v1"
FORMAL_EVALUATOR_PROTOCOL = "V3_COMPOSITE_EVENT"
NEG20_PROTOCOL = "NEG20_V3_PRIMARY_SECONDARY_EVENTS"

FULL = "full"
FRONT = "front_l2"
BACK = "back_l2"
SCENARIOS = (FULL, FRONT, BACK)

# Frozen contract mirror (asserted equal to the frozen evaluator in the driver;
# used only for the illegal-value guard on timesteps).
MAX_TIMESTEPS = 4096

# ---------------------------------------------------------------------------
# primary_outcome vocabulary
# ---------------------------------------------------------------------------
FRONT_TRANSITION_SUCCESS = "FRONT_TRANSITION_SUCCESS"
FRONT_NO_TRANSITION = "FRONT_NO_TRANSITION"
FRONT_INVALID_START = "FRONT_INVALID_START"
BACK_DEFEAT_KOBOLD_SUCCESS = "BACK_DEFEAT_KOBOLD_SUCCESS"
BACK_NO_DEFEAT = "BACK_NO_DEFEAT"
BACK_INVALID_START = "BACK_INVALID_START"
FULL_DEFEAT_KOBOLD_SUCCESS = "FULL_DEFEAT_KOBOLD_SUCCESS"
FULL_NO_DEFEAT = "FULL_NO_DEFEAT"
FULL_INVALID_START = "FULL_INVALID_START"

PRIMARY_OUTCOME_VOCABULARY = frozenset([
    FRONT_TRANSITION_SUCCESS, FRONT_NO_TRANSITION, FRONT_INVALID_START,
    BACK_DEFEAT_KOBOLD_SUCCESS, BACK_NO_DEFEAT, BACK_INVALID_START,
    FULL_DEFEAT_KOBOLD_SUCCESS, FULL_NO_DEFEAT, FULL_INVALID_START,
])
PRIMARY_SUCCESS_OUTCOME = {
    FULL: FULL_DEFEAT_KOBOLD_SUCCESS,
    FRONT: FRONT_TRANSITION_SUCCESS,
    BACK: BACK_DEFEAT_KOBOLD_SUCCESS,
}
PRIMARY_INVALID_OUTCOME = {
    FULL: FULL_INVALID_START,
    FRONT: FRONT_INVALID_START,
    BACK: BACK_INVALID_START,
}

# ---------------------------------------------------------------------------
# secondary_events vocabulary (always emitted sorted)
# ---------------------------------------------------------------------------
EV_DEFEAT_KOBOLD = "DEFEAT_KOBOLD"
EV_PLAYER_DIED = "PLAYER_DIED"
EV_KOBOLD_ENGAGED = "KOBOLD_ENGAGED"
EV_CORRIDOR_EXIT_REACHED = "CORRIDOR_EXIT_REACHED"
EV_TIMED_OUT = "TIMED_OUT"
SECONDARY_EVENT_VOCABULARY = frozenset([
    EV_DEFEAT_KOBOLD, EV_PLAYER_DIED, EV_KOBOLD_ENGAGED,
    EV_CORRIDOR_EXIT_REACHED, EV_TIMED_OUT,
])

# ---------------------------------------------------------------------------
# taxonomy_status vocabulary
# ---------------------------------------------------------------------------
VALID_COMPOSITE_EVENT = "VALID_COMPOSITE_EVENT"
VALID_SINGLE_EVENT = "VALID_SINGLE_EVENT"
INVALID_START = "INVALID_START"
TAXONOMY_STATUS_VOCABULARY = frozenset([
    VALID_COMPOSITE_EVENT, VALID_SINGLE_EVENT, INVALID_START,
])

# ---------------------------------------------------------------------------
# Retained fail-closed categories (总控 §二 — exhaustive, tagged on every raise)
# ---------------------------------------------------------------------------
FC_CORRUPTION = "STATE_CORRUPTION"
FC_MISSING_FIELD = "MISSING_REQUIRED_FIELD"
FC_UNEXPLAINABLE_CONTRADICTION = "CONTRADICTION_UNEXPLAINABLE_BY_CHRONOLOGY"
FC_ILLEGAL_VALUE = "ILLEGAL_VALUE"
FC_HASH_MISMATCH = "EVIDENCE_HASH_MISMATCH"
FC_UNREGISTERED = "UNREGISTERED_EVENT_OR_SCENARIO"
FAIL_CLOSED_CATEGORIES = frozenset([
    FC_CORRUPTION, FC_MISSING_FIELD, FC_UNEXPLAINABLE_CONTRADICTION,
    FC_ILLEGAL_VALUE, FC_HASH_MISMATCH, FC_UNREGISTERED,
])

# The full 14-field formal episode record schema (frozen rollout + driver-added
# episode_record_sha256). V3 requires the complete event payload.
REQUIRED_V3_KEYS = (
    "action_sequence", "corridor_exit_reached", "defeat_kobold", "episode_id",
    "episode_record_sha256", "front_floor_transition_reached",
    "graph_distance_progress", "kobold_engaged", "player_died", "scenario",
    "terminal_label", "timed_out", "timesteps", "valid_start",
)
_BOOL_FIELDS = (
    "corridor_exit_reached", "defeat_kobold", "front_floor_transition_reached",
    "kobold_engaged", "player_died", "timed_out", "valid_start",
)


class FailClosed(Exception):
    """Hard stop; carries a retained-category tag (总控 §二)."""

    def __init__(self, msg, category=None):
        super().__init__(msg)
        self.category = category


def require(cond, msg, category):
    if not cond:
        raise FailClosed(msg, category)


# ---------------------------------------------------------------------------
# Byte-exact serialization helpers (identical to tier3_projection_runtime so the
# episode_record_sha256 recomputation matches the frozen driver bit-for-bit).
# ---------------------------------------------------------------------------
def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def lf_sha256_file(path):
    """LF-normalized SHA256 of a source file (EOL-independent source identity)."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def module_lf_sha256():
    """LF-SHA256 of THIS module — the V3 taxonomy pin recorded on every cert."""
    return lf_sha256_file(os.path.abspath(__file__))


def verify_record_sha(rec):
    """Recompute the frozen-driver episode_record_sha256 (canonical JSON of the
    record WITHOUT the sha field). Returns the recomputed hex."""
    body = {k: v for k, v in rec.items() if k != "episode_record_sha256"}
    return sha256_bytes(canonical_json_bytes(body))


# ---------------------------------------------------------------------------
# Validation — the 6 retained fail-closed categories (总控 §二)
# ---------------------------------------------------------------------------
def _validate_v3(scenario, rec, expected_sha):
    require(isinstance(rec, dict),
            "FAIL CLOSED (V3/%s): episode record is not a dict" % FC_CORRUPTION,
            FC_CORRUPTION)
    # UNREGISTERED scenario value.
    sc_field = rec.get("scenario")
    require(sc_field in SCENARIOS,
            "FAIL CLOSED (V3/%s): episode scenario %r not registered in V3 taxonomy"
            % (FC_UNREGISTERED, sc_field), FC_UNREGISTERED)
    # MISSING required fields (full 14-field formal schema).
    missing = [k for k in REQUIRED_V3_KEYS if k not in rec]
    require(not missing,
            "FAIL CLOSED (V3/%s): episode record missing required key(s): %s"
            % (FC_MISSING_FIELD, sorted(missing)), FC_MISSING_FIELD)
    # ILLEGAL: scenario field must match the evaluation scenario argument.
    require(sc_field == scenario,
            "FAIL CLOSED (V3/%s): episode scenario %r != evaluation scenario %r"
            % (FC_ILLEGAL_VALUE, sc_field, scenario), FC_ILLEGAL_VALUE)
    # ILLEGAL: boolean fields must be real bools.
    for k in _BOOL_FIELDS:
        require(isinstance(rec[k], bool),
                "FAIL CLOSED (V3/%s): field %r must be bool, got %r"
                % (FC_ILLEGAL_VALUE, k, rec[k]), FC_ILLEGAL_VALUE)
    # ILLEGAL: episode_id / terminal_label types.
    require(isinstance(rec["episode_id"], str),
            "FAIL CLOSED (V3/%s): episode_id must be str" % FC_ILLEGAL_VALUE,
            FC_ILLEGAL_VALUE)
    require(isinstance(rec["terminal_label"], str),
            "FAIL CLOSED (V3/%s): terminal_label must be str" % FC_ILLEGAL_VALUE,
            FC_ILLEGAL_VALUE)
    # ILLEGAL: timesteps integer in [0, MAX_TIMESTEPS].
    ts = rec["timesteps"]
    require(isinstance(ts, int) and not isinstance(ts, bool)
            and 0 <= ts <= MAX_TIMESTEPS,
            "FAIL CLOSED (V3/%s): timesteps %r outside [0,%d]"
            % (FC_ILLEGAL_VALUE, ts, MAX_TIMESTEPS), FC_ILLEGAL_VALUE)
    # ILLEGAL: action_sequence is a list of ints with len == timesteps.
    actions = rec["action_sequence"]
    require(isinstance(actions, list) and len(actions) == ts
            and all(isinstance(a, int) and not isinstance(a, bool) for a in actions),
            "FAIL CLOSED (V3/%s): action_sequence must be list[int] of length "
            "timesteps (%d)" % (FC_ILLEGAL_VALUE, ts), FC_ILLEGAL_VALUE)
    # graph_distance_progress: None (FULL/BACK) OR a finite float in [0,1] (FRONT).
    p = rec["graph_distance_progress"]
    if p is not None:
        require(isinstance(p, (int, float)) and not isinstance(p, bool),
                "FAIL CLOSED (V3/%s): graph_distance_progress %r not numeric"
                % (FC_ILLEGAL_VALUE, p), FC_ILLEGAL_VALUE)
        require(math.isfinite(float(p)),
                "FAIL CLOSED (V3/%s): graph_distance_progress %r non-finite"
                % (FC_CORRUPTION, p), FC_CORRUPTION)
        require(0.0 <= float(p) <= 1.0,
                "FAIL CLOSED (V3/%s): graph_distance_progress %r outside [0,1]"
                % (FC_ILLEGAL_VALUE, p), FC_ILLEGAL_VALUE)
    # HASH mismatch: the stored sha must equal the recomputed canonical sha, and
    # (when an external pin is supplied) both must equal it.
    stored = rec["episode_record_sha256"]
    require(isinstance(stored, str) and len(stored) == 64,
            "FAIL CLOSED (V3/%s): episode_record_sha256 malformed" % FC_HASH_MISMATCH,
            FC_HASH_MISMATCH)
    recomputed = verify_record_sha(rec)
    require(recomputed == stored,
            "FAIL CLOSED (V3/%s): episode_record_sha256 %s != recomputed %s"
            % (FC_HASH_MISMATCH, stored, recomputed), FC_HASH_MISMATCH)
    if expected_sha is not None:
        require(recomputed == expected_sha,
                "FAIL CLOSED (V3/%s): episode_record_sha256 %s != source V2 pin %s"
                % (FC_HASH_MISMATCH, recomputed, expected_sha), FC_HASH_MISMATCH)


def _structural_contradictions(rec):
    """Conjunctions that CANNOT arise from a real rollout (evaluator_v2 line 283
    makes timed_out exclusive of defeat/died; the FRONT alias and both its signals
    derive from max_level>=3). Their presence is corruption / a contradiction that
    no event chronology can explain -> retained FailClosed (总控 §二)."""
    out = []
    defeat = rec["defeat_kobold"]
    died = rec["player_died"]
    timed_out = rec["timed_out"]
    if defeat and timed_out:
        out.append("defeat_kobold AND timed_out (structurally exclusive)")
    if died and timed_out:
        out.append("player_died AND timed_out (structurally exclusive)")
    if (rec["scenario"] == FRONT and rec["front_floor_transition_reached"]
            and rec["corridor_exit_reached"] is False):
        out.append("front_l2 transition True but corridor_exit_reached alias False "
                   "(PENDING_EQUIVALENCE_ALIAS contradiction)")
    return out


# Frozen taxonomy label strings (mirror tier3_failure_taxonomy.LABELS exactly, so
# the BACK diagnostics failure_taxonomy block stays parity-exact for non-composite
# records).
_FROZEN_INVALID_START = "INVALID_START"
_FROZEN_SUCCESS_DEFEAT_KOBOLD = "SUCCESS_DEFEAT_KOBOLD"
_FROZEN_FRONT_TRANSITION = "FRONT_FLOOR_TRANSITION_REACHED"


def _frozen_label_for(scenario, rec):
    """Replicate the frozen non-contradiction precedence so terminal_label_counts
    and BACK failure_taxonomy are identical to V2 for non-composite records."""
    defeat = rec["defeat_kobold"]
    died = rec["player_died"]
    timed_out = rec["timed_out"]
    transition = rec["front_floor_transition_reached"]
    engaged = rec["kobold_engaged"]
    if scenario == FULL:
        if defeat:
            return _FROZEN_SUCCESS_DEFEAT_KOBOLD
        if died:
            return "DIED_BEFORE_KOBOLD"
        if timed_out:
            return "TIMEOUT_NO_KOBOLD"
    elif scenario == FRONT:
        if transition:
            return _FROZEN_FRONT_TRANSITION
        if died:
            return "DIED_IN_CORRIDOR"
        if timed_out:
            return "TIMEOUT_NO_TRANSITION"
    else:  # BACK
        if defeat:
            return _FROZEN_SUCCESS_DEFEAT_KOBOLD
        if died and engaged:
            return "DIED_AFTER_ENGAGEMENT"
        if died:
            return "DIED_BEFORE_ENGAGEMENT"
        if timed_out:
            return "TIMEOUT_COMBAT_NOT_WON"
    return None  # unreachable post-validation (>=1 terminal guaranteed)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify_episode_v3(scenario, rec, expected_sha=None):
    """Classify one episode into primary_outcome + secondary_events + status.

    Raises FailClosed (tagged category) ONLY for the 6 retained 总控 §二 conditions.
    The legitimate composite (front transition AND defeat_kobold) returns
    VALID_COMPOSITE_EVENT and NEVER aborts.
    """
    require(scenario in SCENARIOS,
            "FAIL CLOSED (V3/%s): evaluation scenario %r not registered"
            % (FC_UNREGISTERED, scenario), FC_UNREGISTERED)
    _validate_v3(scenario, rec, expected_sha)

    if rec["valid_start"] is not True:
        return {
            "scenario": scenario,
            "episode_id": rec["episode_id"],
            "primary_outcome": PRIMARY_INVALID_OUTCOME[scenario],
            "secondary_events": [],
            "taxonomy_status": INVALID_START,
            "composite": False,
            "frozen_label": _FROZEN_INVALID_START,
            "failure_rule_version_v3": FAILURE_RULE_VERSION_V3,
        }

    # Retained structural contradictions (corruption / unexplainable).
    contra = _structural_contradictions(rec)
    require(not contra,
            "FAIL CLOSED (V3/%s): contradictory terminal signals %r unexplainable "
            "by event chronology" % (FC_UNEXPLAINABLE_CONTRADICTION, contra),
            FC_UNEXPLAINABLE_CONTRADICTION)

    defeat = rec["defeat_kobold"]
    died = rec["player_died"]
    timed_out = rec["timed_out"]
    transition = rec["front_floor_transition_reached"]
    engaged = rec["kobold_engaged"]
    corridor_exit = rec["corridor_exit_reached"]

    # At least one terminal signal must be present (none => unexplainable).
    require(defeat or died or timed_out or transition,
            "FAIL CLOSED (V3/%s): episode has no terminal signal (none of "
            "defeat/died/timed_out/floor-transition set)"
            % FC_UNEXPLAINABLE_CONTRADICTION, FC_UNEXPLAINABLE_CONTRADICTION)

    # Secondary events: every legal event observed (sorted on output).
    secondary = []
    if defeat:
        secondary.append(EV_DEFEAT_KOBOLD)
    if died:
        secondary.append(EV_PLAYER_DIED)
    if engaged:
        secondary.append(EV_KOBOLD_ENGAGED)
    if corridor_exit:
        secondary.append(EV_CORRIDOR_EXIT_REACHED)
    if timed_out:
        secondary.append(EV_TIMED_OUT)

    # Primary outcome per arm (frozen primary predicate unchanged).
    if scenario == FRONT:
        # Frozen FRONT primary = transition only; a defeat with no transition is
        # NOT a front success (recorded as a secondary event).
        primary = FRONT_TRANSITION_SUCCESS if transition else FRONT_NO_TRANSITION
    elif scenario == BACK:
        primary = BACK_DEFEAT_KOBOLD_SUCCESS if defeat else BACK_NO_DEFEAT
    else:  # FULL
        primary = FULL_DEFEAT_KOBOLD_SUCCESS if defeat else FULL_NO_DEFEAT

    # Major terminal events = {defeat, died, timed_out, transition}; a composite is
    # >=2 co-occurring (every chronology-explainable pair survives validation).
    major_count = sum([defeat, died, timed_out, transition])
    composite = major_count >= 2
    status = VALID_COMPOSITE_EVENT if composite else VALID_SINGLE_EVENT

    secondary_sorted = sorted(secondary)
    # Internal invariants: emitted values are registered vocabulary.
    require(all(e in SECONDARY_EVENT_VOCABULARY for e in secondary_sorted),
            "FAIL CLOSED (V3/%s): derived secondary event not in vocabulary"
            % FC_UNREGISTERED, FC_UNREGISTERED)
    require(primary in PRIMARY_OUTCOME_VOCABULARY,
            "FAIL CLOSED (V3/%s): derived primary outcome not in vocabulary"
            % FC_UNREGISTERED, FC_UNREGISTERED)

    return {
        "scenario": scenario,
        "episode_id": rec["episode_id"],
        "primary_outcome": primary,
        "secondary_events": secondary_sorted,
        "taxonomy_status": status,
        "composite": composite,
        "frozen_label": _frozen_label_for(scenario, rec),
        "failure_rule_version_v3": FAILURE_RULE_VERSION_V3,
    }


# ---------------------------------------------------------------------------
# Scenario summarizer — frozen metrics envelope (bit-identical) + composite layer
# ---------------------------------------------------------------------------
def summarize_v3(scenario, records):
    """Classify every record, run the FROZEN tier3_metrics.summarize over a
    frozen-compatible classified list (bit-identical primary/dense by construction),
    and additively annotate the composite-event layer."""
    require(scenario in SCENARIOS,
            "FAIL CLOSED (V3/%s): evaluation scenario %r not registered"
            % (FC_UNREGISTERED, scenario), FC_UNREGISTERED)

    classifications = []
    classified_for_metrics = []
    for rec in records:
        cls = classify_episode_v3(scenario, rec)
        classifications.append(cls)
        m = dict(rec)
        m["classified_label"] = cls["frozen_label"]
        m["failure_rule_version"] = FAILURE_RULE_VERSION_V3
        classified_for_metrics.append(m)

    metrics_summary = metrics.summarize(scenario, classified_for_metrics)

    terminal_label_counts = {}
    primary_outcome_counts = {}
    secondary_event_counts = {}
    composite_count = 0
    per_episode = []
    for cls in classifications:
        fl = cls["frozen_label"]
        terminal_label_counts[fl] = terminal_label_counts.get(fl, 0) + 1
        po = cls["primary_outcome"]
        primary_outcome_counts[po] = primary_outcome_counts.get(po, 0) + 1
        for ev in cls["secondary_events"]:
            secondary_event_counts[ev] = secondary_event_counts.get(ev, 0) + 1
        if cls["composite"]:
            composite_count += 1
        per_episode.append({
            "episode_id": cls["episode_id"],
            "primary_outcome": cls["primary_outcome"],
            "secondary_events": list(cls["secondary_events"]),
            "taxonomy_status": cls["taxonomy_status"],
        })

    return {
        "schema": "mechanism_UED.tier3_taxonomy_v3_summary/v1",
        "scenario": scenario,
        "formal_evaluator_protocol": FORMAL_EVALUATOR_PROTOCOL,
        "neg20_protocol": NEG20_PROTOCOL,
        "taxonomy_version_v3": TAXONOMY_V3_VERSION,
        "failure_rule_version_v3": FAILURE_RULE_VERSION_V3,
        "episode_count": len(records),
        "valid_start_count": sum(1 for r in records if r.get("valid_start") is True),
        "metrics": metrics_summary,                 # FROZEN envelope, bit-identical
        "terminal_label_counts": terminal_label_counts,
        "composite_event_layer": {
            "per_episode": per_episode,
            "primary_outcome_counts": primary_outcome_counts,
            "secondary_event_counts": secondary_event_counts,
            "composite_episode_count": composite_count,
            "secondary_event_vocabulary": sorted(SECONDARY_EVENT_VOCABULARY),
        },
        "scaffolded_results_can_replace_full_task": False,
    }


# ---------------------------------------------------------------------------
# Self-test — §四 tests A–H (JAX-free; runs on this host).
# ---------------------------------------------------------------------------
def _rec(scenario, actions=(), gdp=None, **flags):
    """Build a self-consistent 14-field record (sha computed over the body)."""
    r = {
        "action_sequence": list(actions),
        "corridor_exit_reached": False,
        "defeat_kobold": False,
        "episode_id": "%s-bank0" % scenario,
        "front_floor_transition_reached": False,
        "graph_distance_progress": gdp,
        "kobold_engaged": False,
        "player_died": False,
        "scenario": scenario,
        "terminal_label": "",
        "timed_out": False,
        "timesteps": len(actions),
        "valid_start": True,
    }
    r.update(flags)
    r["episode_record_sha256"] = verify_record_sha(r)
    return r


# The REAL composite record (PERSISTENT_RMT16 front_l2-bank6) is read at self-test
# time from the committed V2 evidence (single source of truth; no transcription).
_GOLDEN_EVIDENCE_REL = ("reports", "tier3_scaffolded_evaluation",
                        "formal_evaluation_evidence_20260801", "cc4",
                        "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
                        "formal_evaluation_v2dt", "episode_records.jsonl")
_GOLDEN_EPISODE_ID = "front_l2-bank6"
_GOLDEN_COMPOSITE_SHA = ("13b74ecf39c10f520f3d26d59789efb46c80de07ac61c524e2ef810"
                         "2303b35b3")


def _load_golden_composite():
    """Load the real composite record from committed V2 evidence, or None."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, *_GOLDEN_EVIDENCE_REL)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("episode_id") == _GOLDEN_EPISODE_ID:
                return rec
    return None


def self_test():
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # ---- A: FRONT transition only ----
    a = classify_episode_v3(FRONT, _rec(FRONT, actions=(5, 5),
                                        front_floor_transition_reached=True,
                                        corridor_exit_reached=True, gdp=1.0))
    check("A_primary_transition_success",
          a["primary_outcome"] == FRONT_TRANSITION_SUCCESS)
    check("A_secondary_no_defeat", EV_DEFEAT_KOBOLD not in a["secondary_events"])
    check("A_status_single", a["taxonomy_status"] == VALID_SINGLE_EVENT)

    # ---- B: FRONT transition + defeat_kobold (the repaired composite) ----
    b = classify_episode_v3(FRONT, _rec(FRONT, actions=(5, 5),
                                        front_floor_transition_reached=True,
                                        corridor_exit_reached=True,
                                        defeat_kobold=True, gdp=1.0))
    check("B_primary_transition_success",
          b["primary_outcome"] == FRONT_TRANSITION_SUCCESS)
    check("B_secondary_has_defeat", EV_DEFEAT_KOBOLD in b["secondary_events"])
    check("B_status_composite", b["taxonomy_status"] == VALID_COMPOSITE_EVENT)
    check("B_no_abort", b["composite"] is True)

    # ---- C: FRONT defeat, NO transition (must NOT be a front success) ----
    c = classify_episode_v3(FRONT, _rec(FRONT, actions=(5, 5),
                                        defeat_kobold=True, gdp=0.5))
    check("C_not_transition_success",
          c["primary_outcome"] != FRONT_TRANSITION_SUCCESS
          and c["primary_outcome"] == FRONT_NO_TRANSITION)
    check("C_defeat_is_secondary", EV_DEFEAT_KOBOLD in c["secondary_events"])

    # ---- D: BACK defeat ----
    d = classify_episode_v3(BACK, _rec(BACK, actions=(5, 5), defeat_kobold=True,
                                       kobold_engaged=True))
    check("D_back_primary_defeat",
          d["primary_outcome"] == BACK_DEFEAT_KOBOLD_SUCCESS)

    # ---- E: FULL defeat ----
    e = classify_episode_v3(FULL, _rec(FULL, actions=(5, 5), defeat_kobold=True))
    check("E_full_primary_defeat",
          e["primary_outcome"] == FULL_DEFEAT_KOBOLD_SUCCESS)

    # ---- defeat+died same-step trade kill: composite, not fail-closed (总控 §二
    #      general rule: legal co-occurring events must not abort) ----
    dd = classify_episode_v3(BACK, _rec(BACK, actions=(5, 5), defeat_kobold=True,
                                        player_died=True, kobold_engaged=True))
    check("defeat_died_composite_primary",
          dd["primary_outcome"] == BACK_DEFEAT_KOBOLD_SUCCESS)
    check("defeat_died_secondary_died",
          EV_PLAYER_DIED in dd["secondary_events"])
    check("defeat_died_status_composite",
          dd["taxonomy_status"] == VALID_COMPOSITE_EVENT)

    # ---- F (construction): summarize_v3 metrics == frozen metrics.summarize ----
    front_recs = [
        _rec(FRONT, actions=(5,), episode_id="front_l2-bank0",
             front_floor_transition_reached=True, corridor_exit_reached=True, gdp=1.0),
        _rec(FRONT, actions=(5,), episode_id="front_l2-bank1",
             front_floor_transition_reached=True, corridor_exit_reached=True,
             defeat_kobold=True, gdp=0.8),
        _rec(FRONT, actions=(5,), episode_id="front_l2-bank2", player_died=True,
             gdp=0.2),
        _rec(FRONT, actions=(5,), episode_id="front_l2-bank3", timed_out=True,
             gdp=0.0),
    ]
    s3 = summarize_v3(FRONT, front_recs)
    frozen_ref = metrics.summarize(FRONT, [dict(r, classified_label=None)
                                           for r in front_recs])
    check("F_metrics_bit_identical", s3["metrics"] == frozen_ref)
    check("F_front_transition_count_includes_composite",
          s3["metrics"]["primary"]["successes"] == 2)   # bank0 + bank1(composite)
    check("F_composite_layer_count",
          s3["composite_event_layer"]["composite_episode_count"] == 1)
    check("F_dense_mean_present", s3["metrics"]["dense"]["value"] is not None)

    # ---- G: all 6 retained fail-closed categories still fire ----
    def raises_category(build, category):
        try:
            build()
            return False
        except FailClosed as exc:
            return exc.category == category

    def g_missing():
        r = _rec(FULL, actions=(5,), defeat_kobold=True)
        del r["defeat_kobold"]
        return classify_episode_v3(FULL, r)
    check("G_missing_field", raises_category(g_missing, FC_MISSING_FIELD))

    def g_unregistered():
        r = _rec(FULL, actions=(5,), defeat_kobold=True)
        r["scenario"] = "bogus"
        r["episode_record_sha256"] = verify_record_sha(r)
        return classify_episode_v3(FULL, r)
    check("G_unregistered", raises_category(g_unregistered, FC_UNREGISTERED))

    def g_illegal_ts():
        r = _rec(FULL, actions=(5,), defeat_kobold=True)
        r["timesteps"] = 5000
        r["episode_record_sha256"] = verify_record_sha(r)
        return classify_episode_v3(FULL, r)
    check("G_illegal_timesteps", raises_category(g_illegal_ts, FC_ILLEGAL_VALUE))

    def g_illegal_progress():
        return classify_episode_v3(FRONT, _rec(FRONT, actions=(5,),
                                               front_floor_transition_reached=True,
                                               gdp=1.5))
    check("G_illegal_progress", raises_category(g_illegal_progress, FC_ILLEGAL_VALUE))

    def g_illegal_actions():
        r = _rec(FULL, actions=(5, 5), defeat_kobold=True)
        r["timesteps"] = 3
        r["episode_record_sha256"] = verify_record_sha(r)
        return classify_episode_v3(FULL, r)
    check("G_illegal_action_len", raises_category(g_illegal_actions, FC_ILLEGAL_VALUE))

    def g_hash():
        r = _rec(FULL, actions=(5,), defeat_kobold=True)
        return classify_episode_v3(FULL, r, expected_sha="0" * 64)
    check("G_hash_mismatch", raises_category(g_hash, FC_HASH_MISMATCH))

    def g_contradiction():
        return classify_episode_v3(FULL, _rec(FULL, actions=(5,),
                                              defeat_kobold=True, timed_out=True))
    check("G_unexplainable_contradiction",
          raises_category(g_contradiction, FC_UNEXPLAINABLE_CONTRADICTION))

    def g_alias():
        return classify_episode_v3(FRONT, _rec(FRONT, actions=(5,),
                                               front_floor_transition_reached=True,
                                               corridor_exit_reached=False, gdp=1.0))
    check("G_alias_contradiction",
          raises_category(g_alias, FC_UNEXPLAINABLE_CONTRADICTION))

    def g_no_terminal():
        return classify_episode_v3(FULL, _rec(FULL, actions=(5,)))
    check("G_no_terminal", raises_category(g_no_terminal,
                                           FC_UNEXPLAINABLE_CONTRADICTION))

    def g_corruption():
        return classify_episode_v3(FRONT, _rec(FRONT, actions=(5,),
                                               front_floor_transition_reached=True,
                                               gdp=float("nan")))
    check("G_corruption_nonfinite", raises_category(g_corruption, FC_CORRUPTION))

    check("G_corruption_not_dict",
          raises_category(lambda: classify_episode_v3(FULL, ["not", "a", "dict"]),
                          FC_CORRUPTION))

    # ---- H: composite repair, additive vs frozen V1 ----
    import tier3_failure_taxonomy as taxonomy_v1
    # (a) synthetic self-consistent composite (always available)
    synth = _rec(FRONT, actions=(5, 5), front_floor_transition_reached=True,
                 corridor_exit_reached=True, defeat_kobold=True, gdp=1.0)
    hs = classify_episode_v3(FRONT, synth)
    check("H_synth_v3_composite",
          hs["primary_outcome"] == FRONT_TRANSITION_SUCCESS
          and EV_DEFEAT_KOBOLD in hs["secondary_events"]
          and hs["taxonomy_status"] == VALID_COMPOSITE_EVENT)
    try:
        taxonomy_v1.classify_episode(dict(synth))
        check("H_synth_frozen_v1_still_raises", False)
    except taxonomy_v1.FailClosed:
        check("H_synth_frozen_v1_still_raises", True)
    # (b) the REAL committed composite record (when evidence is present)
    golden = _load_golden_composite()
    if golden is not None:
        check("H_golden_sha_recomputes",
              verify_record_sha(golden) == _GOLDEN_COMPOSITE_SHA
              and golden.get("episode_record_sha256") == _GOLDEN_COMPOSITE_SHA)
        hg = classify_episode_v3(FRONT, golden, expected_sha=_GOLDEN_COMPOSITE_SHA)
        check("H_golden_v3_composite",
              hg["primary_outcome"] == FRONT_TRANSITION_SUCCESS
              and EV_DEFEAT_KOBOLD in hg["secondary_events"]
              and hg["taxonomy_status"] == VALID_COMPOSITE_EVENT)
        try:
            taxonomy_v1.classify_episode(dict(golden))
            check("H_golden_frozen_v1_still_raises", False)
        except taxonomy_v1.FailClosed:
            check("H_golden_frozen_v1_still_raises", True)
    else:
        print("  note: committed V2 evidence absent here; real-record H replay "
              "runs in the offline verifier")

    # Frozen constants still self-consistent (no drift vs the reused library).
    check("metrics_protocol_independent",
          metrics.FRONT == FRONT and metrics.BACK == BACK and metrics.FULL == FULL)
    check("module_lf_sha_is_hex64", len(module_lf_sha256()) == 64)

    if problems:
        print("TIER3_TAXONOMY_V3_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_TAXONOMY_V3_SELF_TEST_PASS (A-H; composite repaired; "
          "fail-closed retained; frozen metrics bit-identical; additive vs V1)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_taxonomy_v3.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
