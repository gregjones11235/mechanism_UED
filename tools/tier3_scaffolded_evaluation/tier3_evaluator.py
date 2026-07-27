#!/usr/bin/env python3
"""CC4 Tier3 — decomposed evaluator (deterministic; inference-only).

Runs ONE frozen evaluation contract across the three scenarios
(FULL_END_TO_END / TIER3_FRONT_HALF_SCAFFOLDED_L2 / TIER3_BACK_HALF_SCAFFOLDED_L2):
    action_mode = greedy_argmax   observation_schema = canonical_craftax_symbolic
    action_space = canonical_craftax_action_set   max_timesteps = 4096
identically for every arm. It validates each episode record (NEG19: an episode without
a valid_start flag is rejected), classifies the terminal label via the failure taxonomy
(NEG20 ambiguity fails closed), computes the frozen metrics, and asserts the Student
checkpoint params are UNCHANGED across the batch (NEG23, with the checkpoint adapter).

ROLLING OUT REAL EPISODES (env.reset/env.step with a real Student) requires a
JAX + craftax host and is BLOCKED_ENVIRONMENT here; the evaluator consumes
already-produced episode records (synthetic in tests). It NEVER produces a scaffold
result that claims full-task success (enforced downstream by the certificate, NEG25).
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit            # noqa: E402
import tier3_event_predicates as pred         # noqa: E402
import tier3_state_serializer as ser          # noqa: E402
import tier3_metrics as metrics               # noqa: E402
import tier3_failure_taxonomy as taxonomy     # noqa: E402
import tier3_checkpoint_adapter as ckpt       # noqa: E402

SCHEMA = "mechanism_UED.tier3_evaluation_result/v1"
RESULT_VERSION = "tier3_evaluation_result/v1"

FULL = metrics.FULL
FRONT = metrics.FRONT
BACK = metrics.BACK

ACTION_MODE = "greedy_argmax"
MAX_TIMESTEPS = 4096

REQUIRED_EPISODE_KEYS = [
    "episode_id", "scenario", "valid_start", "terminal_label",
    "corridor_exit_reached", "defeat_kobold", "timesteps",
]


class FailClosed(Exception):
    """Hard stop on any evaluation-contract violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Episode record validation (NEG19)
# ---------------------------------------------------------------------------
def validate_episode_record(ep: dict):
    """An episode MUST carry an explicit valid_start flag and required keys (NEG19)."""
    require(isinstance(ep, dict),
            "FAIL CLOSED (NEG19): episode is not a dict")
    missing = [k for k in REQUIRED_EPISODE_KEYS if k not in ep]
    require(not missing,
            "FAIL CLOSED (NEG19): episode record missing required key(s): %s" % sorted(missing))
    require("valid_start" in ep,
            "FAIL CLOSED (NEG19): episode record has no valid_start flag")
    require(isinstance(ep["valid_start"], bool),
            "FAIL CLOSED (NEG19): valid_start must be a bool")
    require(ep["scenario"] in (FULL, FRONT, BACK),
            "FAIL CLOSED: episode scenario %r unknown" % ep["scenario"])
    return True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def frozen_contract():
    return {
        "action_mode": ACTION_MODE,
        "observation_schema": "canonical_craftax_symbolic",
        "action_space": "canonical_craftax_action_set",
        "max_timesteps": MAX_TIMESTEPS,
        "identical_for_all_arms": True,
    }


def evaluate(scenario: str, episodes: list, checkpoint_record: dict = None,
             checkpoint_record_after: dict = None):
    """Validate + classify + measure one scenario's episodes under the frozen contract.

    If a checkpoint record is supplied, its params SHA must be unchanged after the run
    (NEG23). Real rollouts are BLOCKED_ENVIRONMENT here; this operates on episode
    records.
    """
    require(scenario in (FULL, FRONT, BACK),
            "FAIL CLOSED: unknown scenario %r" % scenario)
    # NEG23: params must not be updated by the evaluation.
    if checkpoint_record is not None:
        after = checkpoint_record_after if checkpoint_record_after is not None else checkpoint_record
        ckpt.assert_evaluation_does_not_update_params(checkpoint_record, after)

    classified = []
    for ep in episodes:
        validate_episode_record(ep)                       # NEG19
        require(ep["scenario"] == scenario,
                "FAIL CLOSED: episode %r scenario %r != evaluation scenario %r"
                % (ep.get("episode_id"), ep.get("scenario"), scenario))
        cls = taxonomy.classify_episode(ep)               # NEG20 ambiguity fails closed
        rec = dict(ep)
        rec["classified_label"] = cls["label"]
        rec["failure_rule_version"] = cls["failure_rule_version"]
        classified.append(rec)

    summary = metrics.summarize(scenario, classified)
    label_counts = {}
    for rec in classified:
        label_counts[rec["classified_label"]] = label_counts.get(rec["classified_label"], 0) + 1

    return {
        "schema": SCHEMA,
        "result_version": RESULT_VERSION,
        "scenario": scenario,
        "contract": frozen_contract(),
        "episode_count": len(classified),
        "valid_start_count": sum(1 for r in classified if r["valid_start"]),
        "terminal_label_counts": label_counts,
        "metrics": summary,
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "checkpoint_params_sha256": (checkpoint_record or {}).get("params_sha256"),
        "materialization_status": ser.environment_status(),
        "rollout_status": "BLOCKED_ENVIRONMENT" if not ser.have_jax_craftax() else "REAL",
        "scaffolded_results_can_replace_full_task": False,
    }


# ---------------------------------------------------------------------------
# Self-test (synthetic episodes; runs on this host).
# ---------------------------------------------------------------------------
def _ep(scenario, eid, valid_start, **flags):
    e = {"episode_id": eid, "scenario": scenario, "valid_start": valid_start,
         "terminal_label": "", "corridor_exit_reached": False, "defeat_kobold": False,
         "player_died": False, "timed_out": False, "timesteps": 10,
         "kobold_engaged": False, "boss_area_reached": False,
         "normalized_corridor_progress": None}
    e.update(flags)
    return e


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # FULL evaluation over synthetic episodes.
    full_eps = [
        _ep(FULL, "f0", True, defeat_kobold=True, timesteps=900),
        _ep(FULL, "f1", True, player_died=True, timesteps=300),
        _ep(FULL, "f2", True, timed_out=True, timesteps=MAX_TIMESTEPS),
        _ep(FULL, "f3", False, timed_out=True, timesteps=MAX_TIMESTEPS),  # invalid start
    ]
    res = evaluate(FULL, full_eps)
    check("full_episode_count", res["episode_count"] == 4)
    check("full_valid_start_count", res["valid_start_count"] == 3)
    check("full_primary_value", abs(res["metrics"]["primary"]["value"] - 1 / 3) < 1e-9)
    check("full_labels_classified",
          res["terminal_label_counts"].get("SUCCESS_DEFEAT_KOBOLD") == 1
          and res["terminal_label_counts"].get("INVALID_START") == 1)

    # FRONT evaluation with dense progress.
    front_eps = [
        _ep(FRONT, "r0", True, corridor_exit_reached=True, timesteps=500,
            normalized_corridor_progress=1.0),
        _ep(FRONT, "r1", True, player_died=True, timesteps=200,
            normalized_corridor_progress=0.4),
    ]
    fres = evaluate(FRONT, front_eps)
    check("front_primary", fres["metrics"]["primary"]["value"] == 0.5)

    # NEG19: episode missing valid_start rejected.
    bad = _ep(FULL, "x", True)
    del bad["valid_start"]
    try:
        evaluate(FULL, [bad])
        check("NEG19_missing_valid_start_rejected", False)
    except (FailClosed, taxonomy.FailClosed, metrics.FailClosed):
        check("NEG19_missing_valid_start_rejected", True)

    # NEG19: episode missing a required key rejected.
    bad2 = _ep(FULL, "y", True, defeat_kobold=True)
    del bad2["episode_id"]
    try:
        validate_episode_record(bad2)
        check("NEG19_missing_required_key_rejected", False)
    except FailClosed:
        check("NEG19_missing_required_key_rejected", True)

    # NEG23: evaluation with unchanged checkpoint accepted; changed rejected.
    rec = ckpt.make_checkpoint_record({"w": [1, 2, 3]}, (67, 7, 7), "canonical_craftax_action_set")
    check("NEG23_unchanged_params_ok",
          evaluate(FULL, full_eps, checkpoint_record=rec)["checkpoint_params_sha256"]
          == rec["params_sha256"])
    mutated = dict(rec)
    mutated["params_sha256"] = "0" * 64
    try:
        evaluate(FULL, full_eps, checkpoint_record=rec, checkpoint_record_after=mutated)
        check("NEG23_changed_params_rejected", False)
    except (ckpt.FailClosed, FailClosed):
        check("NEG23_changed_params_rejected", True)

    if problems:
        print("TIER3_EVALUATOR_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_EVALUATOR_SELF_TEST_PASS (contract frozen; NEG19/NEG23 guards live; rollout=%s)"
          % ("BLOCKED_ENVIRONMENT" if not ser.have_jax_craftax() else "REAL"))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_evaluator.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
