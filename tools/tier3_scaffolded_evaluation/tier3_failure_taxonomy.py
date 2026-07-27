#!/usr/bin/env python3
"""CC4 Tier3 — failure taxonomy (deterministic, unambiguous episode classification).

Every episode is assigned EXACTLY ONE terminal label by an explicit, versioned rule
set (failure_rule_version is recorded on every classification). If the terminal
signals are CONTRADICTORY or genuinely AMBIGUOUS (more than one mutually-exclusive
terminal condition holds with no defined precedence), the classifier FAILS CLOSED
(NEG20) rather than silently picking one label. Silent mislabelling would corrupt the
mechanism diagnosis, so ambiguity is an error, not a default.

The taxonomy is scenario-aware:
  FULL  : SUCCESS_DEFEAT_KOBOLD / DIED_BEFORE_KOBOLD / TIMEOUT_NO_KOBOLD / INVALID_START
  FRONT : EXIT_REACHED / DIED_IN_CORRIDOR / TIMEOUT_EXIT_NOT_FOUND / INVALID_START
  BACK  : SUCCESS_DEFEAT_KOBOLD / DIED_AFTER_ENGAGEMENT / DIED_BEFORE_ENGAGEMENT /
          TIMEOUT_IN_BOSS_AREA / TIMEOUT_KOBOLD_NOT_FOUND / INVALID_START
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAXONOMY_VERSION = "tier3_failure_taxonomy/v1"
FAILURE_RULE_VERSION = "tier3_failure_rules/v1"

FULL = "full"
FRONT = "front_l2"
BACK = "back_l2"

INVALID_START = "INVALID_START"
SUCCESS_DEFEAT_KOBOLD = "SUCCESS_DEFEAT_KOBOLD"

LABELS = {
    FULL: [SUCCESS_DEFEAT_KOBOLD, "DIED_BEFORE_KOBOLD", "TIMEOUT_NO_KOBOLD", INVALID_START],
    FRONT: ["EXIT_REACHED", "DIED_IN_CORRIDOR", "TIMEOUT_EXIT_NOT_FOUND", INVALID_START],
    BACK: [SUCCESS_DEFEAT_KOBOLD, "DIED_AFTER_ENGAGEMENT", "DIED_BEFORE_ENGAGEMENT",
           "TIMEOUT_IN_BOSS_AREA", "TIMEOUT_KOBOLD_NOT_FOUND", INVALID_START],
}


class FailClosed(Exception):
    """Hard stop on ambiguous / contradictory classification."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _contradictions(ep) -> list:
    """Detect mutually-exclusive terminal signals that cannot be silently reconciled."""
    out = []
    defeat = ep.get("defeat_kobold") is True
    died = ep.get("player_died") is True
    timed_out = ep.get("timed_out") is True
    # A single episode cannot both defeat the kobold AND die / time out as its terminal.
    if defeat and died:
        out.append("defeat_kobold AND player_died both terminal")
    if defeat and timed_out:
        out.append("defeat_kobold AND timed_out both terminal")
    if died and timed_out:
        out.append("player_died AND timed_out both terminal")
    # FRONT terminates at the corridor exit; a defeat signal there is contradictory.
    if ep.get("scenario") == FRONT and ep.get("corridor_exit_reached") is True and defeat:
        out.append("front_l2 reached exit AND defeat_kobold (front ends at exit)")
    return out


def classify_episode(ep: dict) -> dict:
    """Classify one episode into exactly one label; FAIL CLOSED on ambiguity (NEG20)."""
    scenario = ep.get("scenario")
    require(scenario in LABELS,
            "FAIL CLOSED: episode scenario %r not in taxonomy" % scenario)

    # Invalid start is terminal and exclusive.
    if ep.get("valid_start") is not True:
        return {"label": INVALID_START, "scenario": scenario,
                "failure_rule_version": FAILURE_RULE_VERSION, "ambiguous": False}

    ambiguities = _contradictions(ep)
    require(not ambiguities,
            "FAIL CLOSED (NEG20): ambiguous/contradictory terminal signals %r; refusing to "
            "silently assign a single label" % ambiguities)

    defeat = ep.get("defeat_kobold") is True
    died = ep.get("player_died") is True
    timed_out = ep.get("timed_out") is True
    exit_reached = ep.get("corridor_exit_reached") is True
    engaged = ep.get("kobold_engaged") is True
    boss_area = ep.get("boss_area_reached") is True

    if scenario == FULL:
        if defeat:
            label = SUCCESS_DEFEAT_KOBOLD
        elif died:
            label = "DIED_BEFORE_KOBOLD"
        elif timed_out:
            label = "TIMEOUT_NO_KOBOLD"
        else:
            label = None
    elif scenario == FRONT:
        if exit_reached:
            label = "EXIT_REACHED"
        elif died:
            label = "DIED_IN_CORRIDOR"
        elif timed_out:
            label = "TIMEOUT_EXIT_NOT_FOUND"
        else:
            label = None
    else:  # BACK
        if defeat:
            label = SUCCESS_DEFEAT_KOBOLD
        elif died and engaged:
            label = "DIED_AFTER_ENGAGEMENT"
        elif died:
            label = "DIED_BEFORE_ENGAGEMENT"
        elif timed_out and boss_area:
            label = "TIMEOUT_IN_BOSS_AREA"
        elif timed_out:
            label = "TIMEOUT_KOBOLD_NOT_FOUND"
        else:
            label = None

    # Exactly one terminal condition must hold; none => ambiguous (no terminal signal).
    require(label is not None,
            "FAIL CLOSED (NEG20): episode has no terminal signal (none of defeat/died/timed_out/"
            "exit set); cannot assign a label silently")
    require(label in LABELS[scenario],
            "FAIL CLOSED: derived label %r not in scenario taxonomy %r" % (label, scenario))
    return {"label": label, "scenario": scenario,
            "failure_rule_version": FAILURE_RULE_VERSION, "ambiguous": False}


# ---------------------------------------------------------------------------
# Self-test (synthetic; runs on this host).
# ---------------------------------------------------------------------------
def _ep(scenario, **flags):
    e = {"scenario": scenario, "valid_start": True, "defeat_kobold": False,
         "player_died": False, "timed_out": False, "corridor_exit_reached": False,
         "kobold_engaged": False, "boss_area_reached": False}
    e.update(flags)
    return e


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    check("full_success", classify_episode(_ep(FULL, defeat_kobold=True))["label"] == SUCCESS_DEFEAT_KOBOLD)
    check("full_died", classify_episode(_ep(FULL, player_died=True))["label"] == "DIED_BEFORE_KOBOLD")
    check("full_timeout", classify_episode(_ep(FULL, timed_out=True))["label"] == "TIMEOUT_NO_KOBOLD")
    check("front_exit", classify_episode(_ep(FRONT, corridor_exit_reached=True))["label"] == "EXIT_REACHED")
    check("front_died", classify_episode(_ep(FRONT, player_died=True))["label"] == "DIED_IN_CORRIDOR")
    check("back_success", classify_episode(_ep(BACK, defeat_kobold=True))["label"] == "SUCCESS_DEFEAT_KOBOLD")
    check("back_died_after_engage",
          classify_episode(_ep(BACK, player_died=True, kobold_engaged=True))["label"] == "DIED_AFTER_ENGAGEMENT")
    check("back_died_before_engage",
          classify_episode(_ep(BACK, player_died=True))["label"] == "DIED_BEFORE_ENGAGEMENT")
    check("back_timeout_boss_area",
          classify_episode(_ep(BACK, timed_out=True, boss_area_reached=True))["label"] == "TIMEOUT_IN_BOSS_AREA")
    check("invalid_start", classify_episode(_ep(FULL, valid_start=False))["label"] == INVALID_START)
    check("rule_version_recorded",
          classify_episode(_ep(FULL, defeat_kobold=True))["failure_rule_version"] == FAILURE_RULE_VERSION)

    # NEG20: contradictory terminal signals -> fail closed (not silently labelled).
    for bad in [_ep(FULL, defeat_kobold=True, player_died=True),
                _ep(FULL, defeat_kobold=True, timed_out=True),
                _ep(FULL, player_died=True, timed_out=True),
                _ep(FRONT, corridor_exit_reached=True, defeat_kobold=True),
                _ep(FULL)]:                                  # no terminal signal at all
        try:
            classify_episode(bad)
            check("NEG20_ambiguous_rejected", False)
            break
        except FailClosed:
            check("NEG20_ambiguous_rejected", True)

    if problems:
        print("TIER3_FAILURE_TAXONOMY_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_FAILURE_TAXONOMY_SELF_TEST_PASS (labels unambiguous; NEG20 fail-closed)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_failure_taxonomy.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
