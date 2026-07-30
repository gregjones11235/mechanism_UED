#!/usr/bin/env python3
"""CC4 Tier3 — frozen metric semantics (pure functions over episode records).

Metrics are FROZEN and identical across every arm / scenario. Scaffold metrics are
CONDITIONAL on a valid scaffold start and are for MECHANISM DIAGNOSIS ONLY — they can
never replace the full-task DEFEAT_KOBOLD_SR (scaffolded_results_can_replace_full_task
= false). Dense progress is not a success substitute.

Frozen primary / dense metrics (收口 fast-track):
  FULL   : DEFEAT_KOBOLD_SR                                        = P(defeat | valid_full_start)
  FRONT  : P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START      = P(player level 2->3 | valid_front_start)
           dense GRAPH_DISTANCE_PROGRESS in [0,1]                  (graph-distance; see predicates)
           (corridor_exit_reached is reported only as PENDING_EQUIVALENCE_ALIAS, never primary)
  BACK   : identity = BOSS_COMBAT_SCAFFOLDED
           P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START                  = P(defeat | valid_back_start)
           boss_area_reached / time_to_boss_area / BACK_BOSS_NOT_FOUND are N/A (search not exercised)

All estimators are pure ratios over validated episode records; an episode without a
valid_start flag is rejected upstream by the evaluator (NEG19).
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402

SCHEMA = "mechanism_UED.tier3_metrics/v1"

FULL = "full"
FRONT = "front_l2"
BACK = "back_l2"

FULL_PRIMARY_METRIC = "DEFEAT_KOBOLD_SR"
FRONT_PRIMARY_METRIC = "P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START"
FRONT_DENSE_METRIC = "GRAPH_DISTANCE_PROGRESS"
BACK_PRIMARY_METRIC = "P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START"
BACK_IDENTITY = "BOSS_COMBAT_SCAFFOLDED"
BACK_NA_METRICS = ["boss_area_reached", "time_to_boss_area", "BACK_BOSS_NOT_FOUND"]

# The frozen public metric schema document (closing contract §2/§7) that mirrors
# these constants byte-for-byte and binds this source file by LF-SHA. The metrics
# self-test fails closed if the document disagrees with ANY constant below.
METRIC_SCHEMA_PATH_REL = ("schemas", "tier3_metric_schema_v1.json")
METRIC_SCHEMA_ID = "mechanism_UED.tier3_metric_schema/v1"

PRIMARY_METRIC = {
    FULL: FULL_PRIMARY_METRIC,
    FRONT: FRONT_PRIMARY_METRIC,
    BACK: BACK_PRIMARY_METRIC,
}


class FailClosed(Exception):
    """Hard stop on invalid metric inputs."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def assert_progress_in_range(progress):
    """NEG17 guard reused at the metric layer: dense progress must lie in [0,1]."""
    require(isinstance(progress, (int, float)) and not isinstance(progress, bool)
            and 0.0 <= float(progress) <= 1.0,
            "FAIL CLOSED (NEG17): normalized_corridor_progress %r outside [0,1]" % (progress,))
    return float(progress)


def _valid_episodes(scenario, episodes):
    return [e for e in episodes if e.get("scenario") == scenario and e.get("valid_start") is True]


def _ratio(num, den):
    if den == 0:
        return None     # undefined (no valid starts) -> reported as such, never faked
    return float(num) / float(den)


def _median(vals):
    """Plain median over a list of floats (None when empty); pure, no statistics dep."""
    if not vals:
        return None
    s = sorted(float(v) for v in vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0


def compute_primary_metric(scenario, episodes):
    """Conditional success probability over valid starts for the scenario."""
    valid = _valid_episodes(scenario, episodes)
    n = len(valid)
    if scenario == FULL:
        successes = sum(1 for e in valid if e.get("defeat_kobold") is True)
    elif scenario == FRONT:
        # Episode-level primary event: player level transitioned 2 -> 3.
        successes = sum(1 for e in valid if e.get("front_floor_transition_reached") is True)
    elif scenario == BACK:
        successes = sum(1 for e in valid if e.get("defeat_kobold") is True)
    else:
        raise FailClosed("FAIL CLOSED: unknown scenario %r" % scenario)
    return {
        "metric": PRIMARY_METRIC[scenario],
        "scenario": scenario,
        "valid_starts": n,
        "successes": successes,
        "value": _ratio(successes, n),
        "conditional_on": "valid_start",
        "diagnostic_only": scenario != FULL,
    }


def compute_dense_progress(scenario, episodes):
    """Mean + median GRAPH_DISTANCE_PROGRESS (max over the episode) over valid FRONT
    starts, plus the per-state paired progress (task §八: one progress value per
    frozen scaffold state; under greedy a state is never repeated to fake samples)."""
    if scenario != FRONT:
        return {"metric": FRONT_DENSE_METRIC, "scenario": scenario,
                "value": None, "median": None, "per_state_progress": [],
                "note": "dense progress defined for front_l2 only"}
    valid = _valid_episodes(FRONT, episodes)
    vals = []
    per_state = []
    for e in valid:
        p = e.get("graph_distance_progress")
        if p is not None:
            p = assert_progress_in_range(p)
            vals.append(p)
            per_state.append({"state_entry_id": e.get("episode_id"),
                              "graph_distance_progress": p})
    return {
        "metric": FRONT_DENSE_METRIC,
        "scenario": FRONT,
        "valid_starts": len(valid),
        "scored": len(vals),
        "value": (sum(vals) / len(vals)) if vals else None,
        "median": _median(vals),
        "per_state_progress": per_state,
        "range": [0, 1],
        "monotonicity_guaranteed": False,
        "is_success_substitute": False,
    }


def compute_back_diagnostics(episodes):
    """BACK combat diagnostics (task §八) computed ONLY from fields that already
    exist in the episode record schema — no scenario-semantics expansion:
    kobold_engaged, player_died, defeat_kobold, timesteps, classified_label.
    time_to_first_engagement / time_to_kill / accumulated damage are NOT episode
    fields and are honestly reported as null with a schema note."""
    valid = _valid_episodes(BACK, episodes)
    steps = [int(e.get("timesteps") or 0) for e in valid]
    label_counts = {}
    for e in valid:
        lbl = e.get("classified_label")
        if lbl is not None:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    return {
        "valid_starts": len(valid),
        "kobold_engaged_count": sum(1 for e in valid if e.get("kobold_engaged") is True),
        "time_to_first_engagement": None,
        "time_to_kill": None,
        "damage": None,
        "schema_note": ("time_to_first_engagement / time_to_kill / damage are not "
                        "fields of the frozen episode record schema "
                        "(mechanism_UED.tier3_evaluation_result/v1); reported as null "
                        "rather than expanding the schema"),
        "survival": {
            "died_count": sum(1 for e in valid if e.get("player_died") is True),
            "defeat_count": sum(1 for e in valid if e.get("defeat_kobold") is True),
            "mean_timesteps": (sum(steps) / len(steps)) if steps else None,
            "max_timesteps_observed": max(steps) if steps else None,
        },
        "failure_taxonomy": label_counts,
    }


def summarize(scenario, episodes):
    primary = compute_primary_metric(scenario, episodes)
    dense = compute_dense_progress(scenario, episodes)
    out = {
        "schema": SCHEMA,
        "scenario": scenario,
        "primary": primary,
        "dense": dense,
        "scaffolded_results_can_replace_full_task": False,
    }
    if scenario == BACK:
        out["identity_class"] = BACK_IDENTITY
        out["na_metrics"] = list(BACK_NA_METRICS)
        out["na_reason"] = ("BACK start is already on floor 3 next to a live Kobold; "
                            "boss-area search is not exercised by this scaffold")
        out["diagnostics"] = compute_back_diagnostics(episodes)
    return out


# ---------------------------------------------------------------------------
# Self-test (synthetic episodes; runs on this host).
# ---------------------------------------------------------------------------
def _ep(scenario, valid_start, **flags):
    e = {"scenario": scenario, "valid_start": valid_start,
         "front_floor_transition_reached": False, "corridor_exit_reached": False,
         "defeat_kobold": False, "graph_distance_progress": None}
    e.update(flags)
    return e


def _sha256_lf_file(path: str) -> str:
    """LF-normalized SHA256 of a source file (EOL-independent source identity)."""
    import hashlib
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # FULL: 1/2 defeat among valid starts; invalid starts excluded.
    full = [_ep(FULL, True, defeat_kobold=True), _ep(FULL, True, defeat_kobold=False),
            _ep(FULL, False, defeat_kobold=True)]
    fp = compute_primary_metric(FULL, full)
    check("full_primary_1_of_2", fp["value"] == 0.5 and fp["valid_starts"] == 2)
    check("full_metric_name", fp["metric"] == FULL_PRIMARY_METRIC)

    # FRONT: 2/4 floor transitions; dense progress mean over scored valid starts.
    front = [_ep(FRONT, True, front_floor_transition_reached=True, graph_distance_progress=1.0),
             _ep(FRONT, True, front_floor_transition_reached=True, graph_distance_progress=0.8),
             _ep(FRONT, True, front_floor_transition_reached=False, graph_distance_progress=0.2),
             _ep(FRONT, True, front_floor_transition_reached=False, graph_distance_progress=0.0),
             _ep(FRONT, False, front_floor_transition_reached=True)]
    frp = compute_primary_metric(FRONT, front)
    check("front_primary_2_of_4", frp["value"] == 0.5 and frp["valid_starts"] == 4)
    check("front_metric_name", frp["metric"] == FRONT_PRIMARY_METRIC)
    frd = compute_dense_progress(FRONT, front)
    check("front_dense_mean", abs(frd["value"] - 0.5) < 1e-9)
    check("front_dense_median", abs(frd["median"] - 0.5) < 1e-9)
    check("front_dense_per_state_paired",
          len(frd["per_state_progress"]) == 4
          and all(0.0 <= x["graph_distance_progress"] <= 1.0
                  for x in frd["per_state_progress"]))
    check("front_dense_metric_name", frd["metric"] == FRONT_DENSE_METRIC)
    check("front_dense_not_success_substitute", frd["is_success_substitute"] is False)

    # BACK: 1/3 defeat; summary carries BOSS_COMBAT_SCAFFOLDED identity + N/A metrics.
    back = [_ep(BACK, True, defeat_kobold=True, kobold_engaged=True, timesteps=900),
            _ep(BACK, True, defeat_kobold=False, player_died=True, kobold_engaged=True,
                timesteps=300),
            _ep(BACK, True, defeat_kobold=False, timed_out=True, timesteps=4096)]
    bp = compute_primary_metric(BACK, back)
    check("back_primary_1_of_3", abs(bp["value"] - 1 / 3) < 1e-9 and bp["valid_starts"] == 3)
    check("back_metric_name", bp["metric"] == BACK_PRIMARY_METRIC)
    bs = summarize(BACK, back)
    check("back_identity_boss_combat_scaffolded", bs.get("identity_class") == BACK_IDENTITY)
    check("back_na_metrics", bs.get("na_metrics") == BACK_NA_METRICS)
    diag = bs.get("diagnostics") or {}
    check("back_diagnostics_engagement",
          diag.get("kobold_engaged_count") == 2
          and diag.get("time_to_first_engagement") is None
          and diag.get("time_to_kill") is None
          and diag.get("damage") is None
          and "schema_note" in diag)
    check("back_diagnostics_survival",
          diag.get("survival", {}).get("died_count") == 1
          and diag.get("survival", {}).get("defeat_count") == 1
          and abs(diag.get("survival", {}).get("mean_timesteps") - (900 + 300 + 4096) / 3)
          < 1e-9)

    # No valid starts -> value undefined (None), never faked.
    check("empty_undefined", compute_primary_metric(FRONT, [])["value"] is None)

    # NEG17 guard: out-of-range progress rejected.
    check("progress_in_range_ok", assert_progress_in_range(0.5) == 0.5)
    try:
        assert_progress_in_range(1.5)
        check("NEG17_metric_range_rejected", False)
    except FailClosed:
        check("NEG17_metric_range_rejected", True)

    # The frozen public metric schema document (closing contract §2/§7) must mirror
    # every constant above AND bind this source file by LF-SHA — a drift on either
    # side fails closed.
    import json as _json
    schema_path = audit.repo_root().joinpath(*METRIC_SCHEMA_PATH_REL)
    check("metric_schema_doc_present", schema_path.is_file())
    if schema_path.is_file():
        doc = _json.loads(schema_path.read_text(encoding="utf-8"))
        check("metric_schema_id", doc.get("schema") == METRIC_SCHEMA_ID)
        sc = doc.get("scenarios", {})
        check("metric_schema_full_primary",
              sc.get(FULL, {}).get("primary_metric") == FULL_PRIMARY_METRIC)
        check("metric_schema_front_primary",
              sc.get(FRONT, {}).get("primary_metric") == FRONT_PRIMARY_METRIC)
        check("metric_schema_front_dense",
              sc.get(FRONT, {}).get("dense_metric") == FRONT_DENSE_METRIC)
        check("metric_schema_back_primary",
              sc.get(BACK, {}).get("primary_metric") == BACK_PRIMARY_METRIC)
        check("metric_schema_back_identity",
              sc.get(BACK, {}).get("identity_class") == BACK_IDENTITY)
        check("metric_schema_back_na",
              sc.get(BACK, {}).get("na_metrics") == BACK_NA_METRICS)
        check("metric_schema_no_boss_search",
              sc.get(BACK, {}).get("boss_search_claimed") is False)
        check("metric_schema_source_bound",
              doc.get("metrics_source_sha256")
              == _sha256_lf_file(str(schema_path.parent.parent
                                     / "tools" / "tier3_scaffolded_evaluation"
                                     / "tier3_metrics.py")))
        ba = doc.get("bit_agreement_policy", {})
        check("metric_schema_bit_exact_required",
              ba.get("canonical_fields_bit_exact") is True
              and "action_sequence" in ba.get("bit_exact_fields", [])
              and "episode_record_sha256" in ba.get("bit_exact_fields", [])
              and "terminal_label" in ba.get("bit_exact_fields", []))

    if problems:
        print("TIER3_METRICS_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_METRICS_SELF_TEST_PASS (metrics frozen; diagnostic_only enforced)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_metrics.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
