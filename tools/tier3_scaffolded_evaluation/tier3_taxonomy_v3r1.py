#!/usr/bin/env python3
"""CC4 Tier3 — V3R1 secondary-event non-redundancy reclassification wrapper.

Implements the 总控 re-close ruling CC4_RECLOSE_FORMAL_GLOBAL_RANKING_WITH_
FRONT_FIRST_AND_TOP_TIE_ONLY §六/§七: the frozen V3 taxonomy emits the primary
terminal event AGAIN as a secondary event for BACK/FULL defeat episodes
(primary = BACK_DEFEAT_KOBOLD_SUCCESS / FULL_DEFEAT_KOBOLD_SUCCESS while
secondary_events also contains DEFEAT_KOBOLD). V3R1 removes that redundancy:

    NEG20_V3R1_PRIMARY_NONREDUNDANT_SECONDARY
      FRONT : primary = FRONT_TRANSITION_SUCCESS / FRONT_NO_TRANSITION. The FRONT
              primary predicate is the floor transition only, so a co-occurring
              defeat_kobold MAY legitimately appear in secondary_events and is
              KEPT (transition+defeat remains a VALID_COMPOSITE_EVENT).
      BACK  : defeat_kobold=true -> primary = BACK_DEFEAT_KOBOLD_SUCCESS and
              secondary_events MUST NOT also contain DEFEAT_KOBOLD.
      FULL  : defeat_kobold=true -> primary = FULL_DEFEAT_KOBOLD_SUCCESS and
              secondary_events MUST NOT also contain DEFEAT_KOBOLD.

All other legal secondary events (PLAYER_DIED, KOBOLD_ENGAGED,
CORRIDOR_EXIT_REACHED, TIMED_OUT) are untouched, and so are INVALID_START
records (empty secondary list either way).

Hard parity guarantee (§七): this wrapper calls the FROZEN
`tier3_taxonomy_v3.classify_episode_v3` and ONLY post-processes
`secondary_events`. primary_outcome / taxonomy_status / composite /
frozen_label are carried through verbatim and re-asserted inside the wrapper,
so the four frozen primary metrics (front transition_count, front mean
graph_distance_progress, full success_count, back defeat_count) are unchanged
BY CONSTRUCTION. Any deviation fails closed and V3R1 must not be published.
This is an OFFLINE reclassification of existing episode records — no
environment rerun happens here or anywhere in V3R1.

This module is JAX-free. Protocol ids are new and never masquerade as V3/V2.

Usage:
  python tools/tier3_scaffolded_evaluation/tier3_taxonomy_v3r1.py --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_taxonomy_v3 as t3            # noqa: E402  (FROZEN; reused as a library)

# ---------------------------------------------------------------------------
# Protocol / version constants (NEW — V3R1 ranking re-close; distinct from V3)
# ---------------------------------------------------------------------------
TAXONOMY_V3R1_VERSION = "tier3_taxonomy_v3r1/v1"
FAILURE_RULE_VERSION_V3R1 = "tier3_failure_rules_v3r1/v1"
NEG20_PROTOCOL = "NEG20_V3R1_PRIMARY_NONREDUNDANT_SECONDARY"
SECONDARY_DEDUP_RULE = "PRIMARY_EVENT_NOT_REPEATED_IN_SECONDARY"

# Frozen V3 vocabulary re-exports (single source of truth stays in taxonomy_v3).
FULL, FRONT, BACK = t3.FULL, t3.FRONT, t3.BACK
SCENARIOS = t3.SCENARIOS
EV_DEFEAT_KOBOLD = t3.EV_DEFEAT_KOBOLD
SECONDARY_EVENT_VOCABULARY = t3.SECONDARY_EVENT_VOCABULARY
BACK_DEFEAT_KOBOLD_SUCCESS = t3.BACK_DEFEAT_KOBOLD_SUCCESS
FULL_DEFEAT_KOBOLD_SUCCESS = t3.FULL_DEFEAT_KOBOLD_SUCCESS
PRIMARY_SUCCESS_OUTCOME = t3.PRIMARY_SUCCESS_OUTCOME
FailClosed = t3.FailClosed


def module_lf_sha256():
    """LF-SHA256 of THIS module — the V3R1 taxonomy pin recorded on the new
    certificates / audit artifacts."""
    return t3.lf_sha256_file(os.path.abspath(__file__))


# Primary outcomes that ALREADY encode the defeat event as the primary result;
# for exactly these, DEFEAT_KOBOLD is redundant inside secondary_events (§六).
_DEFEAT_PRIMARY_OUTCOMES = frozenset([
    BACK_DEFEAT_KOBOLD_SUCCESS,
    FULL_DEFEAT_KOBOLD_SUCCESS,
])


def classify_episode_v3r1(scenario, rec, expected_sha=None):
    """Offline V3R1 reclassification of ONE existing episode record.

    Delegates validation + classification to the frozen V3 taxonomy, then
    removes the primary-event duplication from secondary_events (BACK/FULL
    defeat only). Raises the frozen FailClosed (tagged category) on any invalid
    input, and a corruption-tagged FailClosed if any parity invariant breaks.
    """
    base = t3.classify_episode_v3(scenario, rec, expected_sha=expected_sha)

    # ---- parity invariants BEFORE any post-processing (guard the wrapper) ----
    t3.require(base["scenario"] == scenario,
               "FAIL CLOSED (V3R1/%s): wrapper scenario mismatch %r != %r"
               % (t3.FC_CORRUPTION, base["scenario"], scenario), t3.FC_CORRUPTION)
    t3.require(base["primary_outcome"] in t3.PRIMARY_OUTCOME_VOCABULARY,
               "FAIL CLOSED (V3R1/%s): base primary outcome unregistered"
               % t3.FC_UNREGISTERED, t3.FC_UNREGISTERED)

    # ---- §六 dedup: primary defeat event must not repeat in secondary ----
    rule_evaluated = (scenario in (BACK, FULL)
                      and base["primary_outcome"] in _DEFEAT_PRIMARY_OUTCOMES)
    base_secondary = list(base["secondary_events"])
    if rule_evaluated:
        deduped = [ev for ev in base_secondary if ev != EV_DEFEAT_KOBOLD]
        removed = [ev for ev in base_secondary if ev == EV_DEFEAT_KOBOLD]
    else:
        deduped = list(base_secondary)
        removed = []

    # ---- fail-closed parity re-assertions (§七) ----
    t3.require(len(removed) <= 1,
               "FAIL CLOSED (V3R1/%s): DEFEAT_KOBOLD appeared %d times in "
               "secondary_events (frozen V3 emits it at most once)"
               % (t3.FC_CORRUPTION, len(removed)), t3.FC_CORRUPTION)
    t3.require(set(deduped) <= set(base_secondary),
               "FAIL CLOSED (V3R1/%s): dedup invented secondary events"
               % t3.FC_CORRUPTION, t3.FC_CORRUPTION)
    t3.require(all(ev in SECONDARY_EVENT_VOCABULARY for ev in deduped),
               "FAIL CLOSED (V3R1/%s): secondary event outside vocabulary"
               % t3.FC_UNREGISTERED, t3.FC_UNREGISTERED)
    # FRONT is NEVER deduped: its primary predicate is the transition, so a
    # co-occurring defeat remains a genuine secondary event (§六).
    t3.require(scenario != FRONT or deduped == base_secondary,
               "FAIL CLOSED (V3R1/%s): FRONT secondary_events were modified"
               % t3.FC_CORRUPTION, t3.FC_CORRUPTION)
    t3.require(not (base["primary_outcome"] in (t3.FRONT_TRANSITION_SUCCESS,
                                                t3.FRONT_NO_TRANSITION)
                    and removed),
               "FAIL CLOSED (V3R1/%s): removed DEFEAT_KOBOLD although the "
               "primary outcome is not a defeat success" % t3.FC_CORRUPTION,
               t3.FC_CORRUPTION)

    out = dict(base)
    out["secondary_events"] = deduped
    out["neg20_protocol"] = NEG20_PROTOCOL
    out["taxonomy_version_v3r1"] = TAXONOMY_V3R1_VERSION
    out["failure_rule_version_v3r1"] = FAILURE_RULE_VERSION_V3R1
    out["secondary_dedup"] = {
        "rule": SECONDARY_DEDUP_RULE,
        "evaluated": rule_evaluated,
        "applied": bool(removed),
        "removed_redundant_secondary_events": removed,
        "environment_rerun": False,
        "classification_only": True,
    }
    # Parity fields carried verbatim from the frozen classifier (re-asserted):
    t3.require(out["primary_outcome"] == base["primary_outcome"]
               and out["taxonomy_status"] == base["taxonomy_status"]
               and out["composite"] == base["composite"]
               and out["frozen_label"] == base["frozen_label"],
               "FAIL CLOSED (V3R1/%s): primary classification drifted from "
               "frozen V3" % t3.FC_CORRUPTION, t3.FC_CORRUPTION)
    return out


# ---------------------------------------------------------------------------
# Self-test — §六/§七 semantics + parity (JAX-free; runs on this host)
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
    r["episode_record_sha256"] = t3.verify_record_sha(r)
    return r


def run_self_test():
    checks = 0

    def ok(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            raise FailClosed("FAIL CLOSED (V3R1/SELF_TEST): %s" % msg)

    # ---- 1. BACK defeat: primary kept, DEFEAT_KOBOLD removed from secondary ----
    v3 = t3.classify_episode_v3(BACK, _rec(BACK, actions=(5, 5),
                                           defeat_kobold=True,
                                           kobold_engaged=True))
    r = classify_episode_v3r1(BACK, _rec(BACK, actions=(5, 5),
                                         defeat_kobold=True,
                                         kobold_engaged=True))
    ok(EV_DEFEAT_KOBOLD in v3["secondary_events"],
       "1a frozen V3 BACK defeat DOES duplicate DEFEAT_KOBOLD in secondary")
    ok(r["primary_outcome"] == BACK_DEFEAT_KOBOLD_SUCCESS == v3["primary_outcome"],
       "1b BACK primary identical to V3")
    ok(EV_DEFEAT_KOBOLD not in r["secondary_events"],
       "1c V3R1 removed DEFEAT_KOBOLD from BACK secondary")
    ok("KOBOLD_ENGAGED" in r["secondary_events"],
       "1d other legal secondary events kept")
    ok(r["secondary_dedup"]["applied"] is True
       and r["secondary_dedup"]["evaluated"] is True
       and r["secondary_dedup"]["removed_redundant_secondary_events"]
       == [EV_DEFEAT_KOBOLD], "1e dedup bookkeeping")

    # ---- 2. FULL defeat: same non-redundancy rule ----
    r = classify_episode_v3r1(FULL, _rec(FULL, actions=(5, 5),
                                         defeat_kobold=True))
    ok(r["primary_outcome"] == FULL_DEFEAT_KOBOLD_SUCCESS,
       "2a FULL primary kept")
    ok(EV_DEFEAT_KOBOLD not in r["secondary_events"]
       and r["secondary_dedup"]["applied"] is True,
       "2b FULL secondary deduplicated")

    # ---- 3. FRONT transition+defeat composite: defeat STAYS secondary ----
    r = classify_episode_v3r1(FRONT, _rec(FRONT, actions=(5, 5),
                                          front_floor_transition_reached=True,
                                          corridor_exit_reached=True,
                                          defeat_kobold=True, gdp=1.0))
    ok(r["primary_outcome"] == t3.FRONT_TRANSITION_SUCCESS,
       "3a FRONT composite primary kept")
    ok(EV_DEFEAT_KOBOLD in r["secondary_events"],
       "3b FRONT keeps DEFEAT_KOBOLD as secondary (primary is transition)")
    ok(r["taxonomy_status"] == t3.VALID_COMPOSITE_EVENT
       and r["composite"] is True, "3c composite status kept")
    ok(r["secondary_dedup"]["applied"] is False
       and r["secondary_dedup"]["evaluated"] is False,
       "3d dedup never applied to FRONT")

    # ---- 4. FRONT defeat WITHOUT transition: still not a front success ----
    r = classify_episode_v3r1(FRONT, _rec(FRONT, actions=(5, 5),
                                          defeat_kobold=True, gdp=0.5))
    ok(r["primary_outcome"] == t3.FRONT_NO_TRANSITION,
       "4a defeat-only FRONT is NOT a transition success")
    ok(EV_DEFEAT_KOBOLD in r["secondary_events"],
       "4b FRONT defeat-only keeps DEFEAT_KOBOLD secondary")

    # ---- 5. BACK defeat+died trade kill: only PLAYER_DIED remains secondary ----
    r = classify_episode_v3r1(BACK, _rec(BACK, actions=(5, 5),
                                         defeat_kobold=True, player_died=True,
                                         kobold_engaged=True))
    ok(r["primary_outcome"] == BACK_DEFEAT_KOBOLD_SUCCESS, "5a primary kept")
    ok(r["secondary_events"] == ["KOBOLD_ENGAGED", "PLAYER_DIED"],
       "5b secondary == engaged+died only (DEFEAT_KOBOLD removed), sorted")
    ok(r["taxonomy_status"] == t3.VALID_COMPOSITE_EVENT, "5c composite kept")

    # ---- 6. BACK without defeat: untouched ----
    r = classify_episode_v3r1(BACK, _rec(BACK, actions=(5, 5),
                                         player_died=True, kobold_engaged=True))
    ok(r["primary_outcome"] == t3.BACK_NO_DEFEAT, "6a no-defeat primary")
    ok(r["secondary_events"] == ["KOBOLD_ENGAGED", "PLAYER_DIED"]
       and r["secondary_dedup"]["applied"] is False
       and r["secondary_dedup"]["evaluated"] is False,
       "6b no dedup on non-defeat BACK")

    # ---- 7. FULL timeout: TIMED_OUT untouched ----
    r = classify_episode_v3r1(FULL, _rec(FULL, actions=(5, 5), timed_out=True))
    ok(r["primary_outcome"] == t3.FULL_NO_DEFEAT
       and r["secondary_events"] == ["TIMED_OUT"], "7a timeout untouched")

    # ---- 8. INVALID_START: untouched on every arm ----
    for sc in SCENARIOS:
        r = classify_episode_v3r1(sc, _rec(sc, actions=(5, 5), valid_start=False,
                                           defeat_kobold=(sc != FRONT)))
        ok(r["primary_outcome"] == t3.PRIMARY_INVALID_OUTCOME[sc]
           and r["secondary_events"] == []
           and r["taxonomy_status"] == t3.INVALID_START
           and r["secondary_dedup"]["applied"] is False,
           "8 INVALID_START untouched (%s)" % sc)

    # ---- 9. fail-closed propagation (corruption reaches through the wrapper) ----
    bad = _rec(BACK, actions=(5, 5), defeat_kobold=True)
    bad["timesteps"] = 999999          # > MAX_TIMESTEPS
    try:
        classify_episode_v3r1(BACK, bad)
        ok(False, "9 accepted corrupted record")
    except t3.FailClosed:
        ok(True, "9 fail-closed propagates")

    # ---- 10. parity sweep over the synthetic battery ----
    battery = []
    for sc in SCENARIOS:
        battery.append(_rec(sc, actions=(5,), defeat_kobold=True,
                            kobold_engaged=True, gdp=(0.25 if sc == FRONT else None),
                            front_floor_transition_reached=(sc == FRONT),
                            corridor_exit_reached=(sc == FRONT),
                            episode_id="%s-bank1" % sc))
        battery.append(_rec(sc, actions=(5,), timed_out=True,
                            gdp=(0.1 if sc == FRONT else None),
                            episode_id="%s-bank2" % sc))
        battery.append(_rec(sc, actions=(5,), player_died=True,
                            gdp=(0.0 if sc == FRONT else None),
                            episode_id="%s-bank3" % sc))
        battery.append(_rec(sc, actions=(5,), valid_start=False,
                            episode_id="%s-bank4" % sc))
    for rec in battery:
        v3c = t3.classify_episode_v3(rec["scenario"], rec)
        r = classify_episode_v3r1(rec["scenario"], rec)
        ok(r["primary_outcome"] == v3c["primary_outcome"]
           and r["taxonomy_status"] == v3c["taxonomy_status"]
           and r["composite"] == v3c["composite"]
           and r["frozen_label"] == v3c["frozen_label"]
           and r["episode_id"] == v3c["episode_id"],
           "10 parity fields identical (%s/%s)" % (rec["scenario"],
                                                   rec["episode_id"]))
        ok(set(r["secondary_events"]) <= set(v3c["secondary_events"]),
           "10 secondary subset (%s/%s)" % (rec["scenario"], rec["episode_id"]))
        diff = set(v3c["secondary_events"]) - set(r["secondary_events"])
        ok(diff <= {EV_DEFEAT_KOBOLD},
           "10 only DEFEAT_KOBOLD may be removed (%s/%s)"
           % (rec["scenario"], rec["episode_id"]))
        # primary-success count proxy: identical before/after dedup
        ok((v3c["primary_outcome"] == PRIMARY_SUCCESS_OUTCOME[rec["scenario"]])
           == (r["primary_outcome"] == PRIMARY_SUCCESS_OUTCOME[rec["scenario"]]),
           "10 primary-success bit identical (%s/%s)" % (rec["scenario"],
                                                         rec["episode_id"]))

    # ---- 11. golden replay: real V3 evidence (skipped if not on disk) ----
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ev_root = os.path.join(root, "reports", "tier3_scaffolded_evaluation",
                           "formal_evaluation_evidence_v3_20260801", "cc4")
    replayed = 0
    dedup_hits = 0
    if os.path.isdir(ev_root):
        for cand in sorted(os.listdir(ev_root)):
            p = os.path.join(ev_root, cand, "formal_evaluation_v3",
                             "episode_records.jsonl")
            if not os.path.isfile(p):
                continue
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    v3c = t3.classify_episode_v3(rec["scenario"], rec)
                    r = classify_episode_v3r1(rec["scenario"], rec)
                    ok(r["primary_outcome"] == v3c["primary_outcome"]
                       and r["taxonomy_status"] == v3c["taxonomy_status"]
                       and r["composite"] == v3c["composite"],
                       "11 real-record parity (%s/%s)"
                       % (cand, rec["episode_id"]))
                    if r["secondary_dedup"]["applied"]:
                        dedup_hits += 1
                        ok(EV_DEFEAT_KOBOLD not in r["secondary_events"],
                           "11 dedup effective (%s/%s)" % (cand,
                                                           rec["episode_id"]))
                    if rec["scenario"] == FRONT:
                        ok(r["secondary_events"] == list(v3c["secondary_events"]),
                           "11 FRONT secondary untouched (%s/%s)"
                           % (cand, rec["episode_id"]))
                    replayed += 1
    ok(replayed == 0 or dedup_hits > 0,
       "11 golden replay found at least one BACK/FULL defeat to dedup")

    print("TAXONOMY_V3R1_SELF_TEST_PASS checks=%d golden_records=%d "
          "golden_dedup_hits=%d module_lf_sha256=%s"
          % (checks, replayed, dedup_hits, module_lf_sha256()))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    ap.error("this module is an offline reclassification wrapper; use --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
