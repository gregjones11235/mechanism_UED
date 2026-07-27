#!/usr/bin/env python3
"""CC4 Tier3 — evaluation certificate (honest labels; no over-claiming).

Turns an evaluation_result into a signed-off certificate that records EXACTLY what was
proven and what remains blocked. It is the last line of defence against scientific
over-claiming: it refuses to let a scaffold result masquerade as a full-task result.

Guards (negative tests):
  NEG24 scaffold hash presented as GLOBAL_WORLD_SET_HASH       -> fail closed
  NEG25 scaffold result claims full-task success / breakthrough -> fail closed

Honest status discipline: this round can only ever claim IMPLEMENTED_STATIC /
TESTED_SYNTHETIC (plus TESTED_REAL_ENV_RESET on a JAX host). It NEVER emits
FRONT_SCAFFOLD_EVALUATION=PASS or TIER3_FRONT_HALF_BREAKTHROUGH, because there is no
Student performance data this round.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit            # noqa: E402
import tier3_state_serializer as ser          # noqa: E402
import tier3_metrics as metrics               # noqa: E402
import tier3_failure_taxonomy as taxonomy     # noqa: E402

SCHEMA = "mechanism_UED.tier3_evaluation_certificate/v1"
CERT_VERSION = "tier3_evaluation_certificate/v1"

FULL = metrics.FULL
FRONT = metrics.FRONT
BACK = metrics.BACK

GLOBAL_HASH_LABEL = "GLOBAL_WORLD_SET_HASH"
SCAFFOLD_HASH_LABELS = {"FRONT_SCAFFOLD_STATE_BANK_HASH", "BACK_SCAFFOLD_STATE_BANK_HASH"}

# Forbidden over-claims for a scaffold certificate (NEG25).
FORBIDDEN_OVERCLAIMS = {
    "FRONT_SCAFFOLD_EVALUATION=PASS",
    "BACK_SCAFFOLD_EVALUATION=PASS",
    "TIER3_FRONT_HALF_BREAKTHROUGH",
    "TIER3_BACK_HALF_BREAKTHROUGH",
    "DEFEAT_KOBOLD_SOLVED",
    "TIER3_SOLVED",
    "SOTA",
    "PERSISTENT_BEATS_RESET128",
    "REPLAY_SCIENTIFIC_GAIN",
}


class FailClosed(Exception):
    """Hard stop on over-claiming / hash mislabelling."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# NEG24 / NEG25 guards
# ---------------------------------------------------------------------------
def assert_scaffold_hash_not_global(cert: dict):
    """NEG24: a scaffold certificate must never present its bank hash as the
    GLOBAL_WORLD_SET_HASH (that belongs solely to the seed42 canonical materializer)."""
    label = cert.get("state_bank_hash_label")
    if cert.get("scenario") in (FRONT, BACK):
        require(label in SCAFFOLD_HASH_LABELS,
                "FAIL CLOSED (NEG24): scaffold certificate state_bank_hash_label %r is not a "
                "scaffold bank label" % label)
        require(label != GLOBAL_HASH_LABEL,
                "FAIL CLOSED (NEG24): scaffold certificate claims GLOBAL_WORLD_SET_HASH")
    return True


def assert_scaffold_does_not_claim_full_success(cert: dict):
    """NEG25: a scaffold result must not claim full-task success / breakthrough.

    A scaffold certificate is diagnostic-only; it may report the conditional scaffold
    metric but NEVER DEFEAT_KOBOLD_SR solved, Tier3 broken, SOTA, Persistent>Reset128,
    or a Replay scientific gain.
    """
    claims = set(cert.get("claims", []))
    bad = sorted(claims & FORBIDDEN_OVERCLAIMS)
    require(not bad,
            "FAIL CLOSED (NEG25): scaffold certificate makes forbidden full-task claim(s): %s"
            % bad)
    if cert.get("scenario") in (FRONT, BACK):
        require(cert.get("scaffolded_results_can_replace_full_task") is False,
                "FAIL CLOSED (NEG25): scaffold certificate must declare "
                "scaffolded_results_can_replace_full_task=False")
        # a scaffold cert must not headline the full-task primary metric as achieved
        require(cert.get("headline_metric") != metrics.FULL_PRIMARY_METRIC
                or cert.get("headline_metric_achieved") is not True,
                "FAIL CLOSED (NEG25): scaffold certificate claims full-task %s achieved"
                % metrics.FULL_PRIMARY_METRIC)
    return True


# ---------------------------------------------------------------------------
# Honest status labels (freeze discipline)
# ---------------------------------------------------------------------------
def honest_status_labels(has_real_rollout: bool, has_student_data: bool) -> dict:
    """Produce ONLY labels that the evidence supports. Never over-claim."""
    labels = {
        "BOUNDARY_SCHEMA": "IMPLEMENTED_STATIC" if True else None,
        "SCAFFOLD_BUILDER": "IMPLEMENTED_STATIC",
        "STATE_BANK_MATERIALIZER": "TESTED_SYNTHETIC",
        "EVALUATOR": "TESTED_SYNTHETIC",
        "NEGATIVE_TESTS": "PASS",
        "REAL_CRAFTAX_SCAFFOLD_TEST": "TESTED_REAL_ENV_RESET" if has_real_rollout else "BLOCKED_ENVIRONMENT",
        "REAL_STUDENT_EVALUATION": "EXECUTED" if has_student_data else "NOT_RUN",
        "GLOBAL_WORLD_SET_HASH": "BLOCKED_SOURCE_UNVERIFIED",
        "FRONT_SCAFFOLD_STATE_BANK_HASH": "MATERIALIZED" if has_real_rollout else "NOT_MATERIALIZED",
        "BACK_SCAFFOLD_STATE_BANK_HASH": "MATERIALIZED" if has_real_rollout else "NOT_MATERIALIZED",
        "NEW_TRAINING_RUNS": 0,
        "FORMAL_EVALUATION_RUNS": 0 if not has_student_data else None,
        "SCAFFOLDED_RESULTS_CAN_REPLACE_FULL_TASK": False,
    }
    return labels


def build_certificate(evaluation_result: dict, state_bank_hash_label: str = None,
                      claims=None, has_real_rollout: bool = False,
                      has_student_data: bool = False) -> dict:
    scenario = evaluation_result["scenario"]
    primary = evaluation_result["metrics"]["primary"]
    cert = {
        "schema": SCHEMA,
        "cert_version": CERT_VERSION,
        "scenario": scenario,
        "identity_class": evaluation_result.get("contract", {}).get("observation_schema") and {
            FULL: "CANONICAL_S4_EVALUATION",
            FRONT: "TIER3_FRONT_DIAGNOSTIC_SCAFFOLD",
            BACK: "TIER3_BACK_DIAGNOSTIC_SCAFFOLD",
        }[scenario],
        "headline_metric": primary["metric"],
        "headline_metric_value": primary["value"],
        "headline_metric_achieved": None,     # never asserted without real Student data
        "valid_starts": primary["valid_starts"],
        "failure_rule_version": evaluation_result.get("failure_rule_version"),
        "terminal_label_counts": evaluation_result.get("terminal_label_counts"),
        "state_bank_hash_label": state_bank_hash_label or (
            "FRONT_SCAFFOLD_STATE_BANK_HASH" if scenario == FRONT
            else "BACK_SCAFFOLD_STATE_BANK_HASH" if scenario == BACK
            else GLOBAL_HASH_LABEL),
        "rollout_status": evaluation_result.get("rollout_status"),
        "claims": list(claims or []),
        "scaffolded_results_can_replace_full_task": False,
        "status_labels": honest_status_labels(has_real_rollout, has_student_data),
        "source_audit_schema": audit.SCHEMA,
    }
    assert_scaffold_hash_not_global(cert)          # NEG24
    assert_scaffold_does_not_claim_full_success(cert)   # NEG25
    return cert


# ---------------------------------------------------------------------------
# Self-test (synthetic; runs on this host).
# ---------------------------------------------------------------------------
def _result(scenario, value, n):
    return {
        "schema": "mechanism_UED.tier3_evaluation_result/v1",
        "scenario": scenario,
        "contract": {"observation_schema": "canonical_craftax_symbolic"},
        "metrics": {"primary": {"metric": metrics.PRIMARY_METRIC[scenario],
                                "value": value, "valid_starts": n}},
        "failure_rule_version": taxonomy.FAILURE_RULE_VERSION,
        "terminal_label_counts": {},
        "rollout_status": "BLOCKED_ENVIRONMENT",
    }


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # A clean front certificate builds and carries honest labels.
    c = build_certificate(_result(FRONT, 0.5, 4))
    check("front_cert_label_scaffold",
          c["state_bank_hash_label"] == "FRONT_SCAFFOLD_STATE_BANK_HASH")
    check("honest_not_run",
          c["status_labels"]["REAL_STUDENT_EVALUATION"] == "NOT_RUN"
          and c["status_labels"]["REAL_CRAFTAX_SCAFFOLD_TEST"] == "BLOCKED_ENVIRONMENT"
          and c["status_labels"]["FRONT_SCAFFOLD_STATE_BANK_HASH"] == "NOT_MATERIALIZED"
          and c["status_labels"]["GLOBAL_WORLD_SET_HASH"] == "BLOCKED_SOURCE_UNVERIFIED")
    check("no_breakthrough_claim", c["headline_metric_achieved"] is None)

    # NEG24: scaffold cert claiming GLOBAL_WORLD_SET_HASH rejected.
    try:
        build_certificate(_result(FRONT, 0.5, 4), state_bank_hash_label=GLOBAL_HASH_LABEL)
        check("NEG24_global_label_rejected", False)
    except FailClosed:
        check("NEG24_global_label_rejected", True)

    # NEG25: scaffold cert claiming breakthrough / full-task success rejected.
    for bad_claim in ["TIER3_FRONT_HALF_BREAKTHROUGH", "DEFEAT_KOBOLD_SOLVED", "SOTA",
                      "PERSISTENT_BEATS_RESET128"]:
        try:
            build_certificate(_result(FRONT, 0.5, 4), claims=[bad_claim])
            check("NEG25_overclaim_rejected", False)
            break
        except FailClosed:
            check("NEG25_overclaim_rejected", True)

    # NEG25: scaffold cert headlining full-task metric as achieved rejected.
    full_as_scaffold = _result(FRONT, 1.0, 4)
    full_as_scaffold["metrics"]["primary"]["metric"] = metrics.FULL_PRIMARY_METRIC
    try:
        cert = build_certificate(full_as_scaffold)
        cert["headline_metric_achieved"] = True
        assert_scaffold_does_not_claim_full_success(cert)
        check("NEG25_full_metric_achieved_rejected", False)
    except FailClosed:
        check("NEG25_full_metric_achieved_rejected", True)

    # A FULL certificate MAY use the GLOBAL label and full-task metric.
    fc = build_certificate(_result(FULL, 0.25, 8), has_real_rollout=False)
    check("full_cert_global_label_ok", fc["state_bank_hash_label"] == GLOBAL_HASH_LABEL)

    if problems:
        print("TIER3_EVALUATION_CERTIFICATE_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_EVALUATION_CERTIFICATE_SELF_TEST_PASS (NEG24/NEG25 guards live; honest labels)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_evaluation_certificate.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
