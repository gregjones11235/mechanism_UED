#!/usr/bin/env python3
"""CC4 Tier3 — evaluation certificate (honest labels; no over-claiming).

Turns an evaluation_result into a signed-off certificate that records EXACTLY what was
proven and what remains blocked. It is the last line of defence against scientific
over-claiming: it refuses to let a scaffold result masquerade as a full-task result.

Guards (negative tests):
  NEG24 scaffold hash presented as GLOBAL_WORLD_SET_HASH       -> fail closed
  NEG25 scaffold result claims full-task success / breakthrough -> fail closed
  NEG29 certificate provenance missing/invalid (pid/argv/times/exit code/driver SHA)
                                                                 -> fail closed

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

# REAL certificate bindings (task §五): a certificate that records an evaluation
# must bind ACTUAL VALUES for every field below — a hash LABEL or an omitted value
# fails closed (NEG27). These are the interface + provenance facts that make the
# certificate auditable against the frozen Tier3 contract.
EVAL_BINDING_REQUIRED_FIELDS = (
    "state_bank_hash", "state_payload_hashes", "checkpoint_file_sha256",
    "cc2_params_sha256", "checkpoint_step", "carry_mode", "run_class",
    "episode_records_sha256", "cc2_policy_source_sha256", "evaluator_source_sha256",
    "predicate_code_sha256", "observation_shape", "action_dim", "params_unchanged",
    "performance_claim_authorized",
    # REAL process provenance (task §五 / NEG29): the certificate must bind the ACTUAL
    # evaluator process that produced it and the SHA-bound driver source the network
    # hyperparameters were recovered from.
    "driver_source_sha256", "process_pid", "process_argv",
    "run_start_utc", "run_end_utc", "run_exit_code",
)
EVAL_BINDING_SHA_FIELDS = (
    "state_bank_hash", "checkpoint_file_sha256", "cc2_params_sha256",
    "episode_records_sha256", "cc2_policy_source_sha256", "evaluator_source_sha256",
    "predicate_code_sha256", "driver_source_sha256",
)
FROZEN_OBSERVATION_SHAPE = [8335]      # canonical S4 symbolic obs (unchanged)
FROZEN_ACTION_DIM = 43                 # canonical craftax action set (unchanged)
RUN_CLASSES = ("INTERFACE_SMOKE", "FORMAL_EVALUATION")


class FailClosed(Exception):
    """Hard stop on over-claiming / hash mislabelling / incomplete binding."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _is_sha256_hex(v) -> bool:
    return (isinstance(v, str) and len(v) == 64
            and all(c in "0123456789abcdef" for c in v))


def _require_iso_utc(value, field):
    """NEG29: a provenance timestamp must be a non-empty, parseable ISO-8601 string."""
    require(isinstance(value, str) and value,
            "FAIL CLOSED (NEG29): eval_binding.%s %r is not a non-empty ISO-8601 string "
            "(the actual run start/end time must be bound)" % (field, value))
    import datetime as _dt
    try:
        _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise FailClosed(
            "FAIL CLOSED (NEG29): eval_binding.%s %r is not a parseable ISO-8601 "
            "timestamp" % (field, value))


# ---------------------------------------------------------------------------
# NEG27: REAL certificate value binding (never just hash labels)
# ---------------------------------------------------------------------------
def assert_eval_binding_complete(cert: dict) -> dict:
    """NEG27: a certificate carrying an evaluation binding must bind ACTUAL VALUES
    for every EVAL_BINDING_REQUIRED_FIELDS field. Missing / empty values, hash
    LABELS in place of 64-hex SHAs, a non-frozen observation/action interface,
    params_unchanged != True, or an unknown run_class all fail closed.

    Extra enforcement: run_class=INTERFACE_SMOKE can NEVER authorize a performance
    claim (performance_claim_authorized must be False).
    """
    binding = cert.get("eval_binding")
    require(isinstance(binding, dict),
            "FAIL CLOSED (NEG27): certificate has no eval_binding dict — a real "
            "evaluation certificate must bind actual values, not labels")
    missing = []
    for f in EVAL_BINDING_REQUIRED_FIELDS:
        v = binding.get(f)
        if v is None or v == "" or v == [] or v == {}:
            missing.append(f)
    require(not missing,
            "FAIL CLOSED (NEG27): eval_binding missing / empty field(s) %s — actual "
            "values required, hash labels are not enough" % missing)
    for f in EVAL_BINDING_SHA_FIELDS:
        require(_is_sha256_hex(binding[f]),
                "FAIL CLOSED (NEG27): eval_binding.%s = %r is not a 64-hex sha256 VALUE "
                "(a label or truncated hash is forbidden)" % (f, binding[f]))
    require(isinstance(binding["state_payload_hashes"], list)
            and all(_is_sha256_hex(h) for h in binding["state_payload_hashes"]),
            "FAIL CLOSED (NEG27): state_payload_hashes must be an ordered list of "
            "64-hex sha256 values")
    require(list(binding["observation_shape"]) == FROZEN_OBSERVATION_SHAPE,
            "FAIL CLOSED (NEG27): eval_binding observation_shape %s != frozen %s "
            "(observation interface changed)"
            % (binding["observation_shape"], FROZEN_OBSERVATION_SHAPE))
    require(int(binding["action_dim"]) == FROZEN_ACTION_DIM,
            "FAIL CLOSED (NEG27): eval_binding action_dim %r != frozen %d "
            "(action interface changed)" % (binding["action_dim"], FROZEN_ACTION_DIM))
    require(binding["params_unchanged"] is True,
            "FAIL CLOSED (NEG27/NEG23): eval_binding params_unchanged must be exactly "
            "True (params SHA identical before/after evaluation)")
    require(binding["carry_mode"] in ("persistent", "reset128"),
            "FAIL CLOSED (NEG27): eval_binding carry_mode %r not in (persistent, reset128)"
            % binding["carry_mode"])
    require(binding["run_class"] in RUN_CLASSES,
            "FAIL CLOSED (NEG27): eval_binding run_class %r not in %s"
            % (binding["run_class"], RUN_CLASSES))
    require(binding["performance_claim_authorized"] is False
            or binding["run_class"] == "FORMAL_EVALUATION",
            "FAIL CLOSED (NEG27): run_class=INTERFACE_SMOKE can never authorize a "
            "performance claim")
    # ---- NEG29: REAL process provenance (actual pid / argv / times / exit code) ----
    require(isinstance(binding["process_pid"], int)
            and not isinstance(binding["process_pid"], bool)
            and binding["process_pid"] > 0,
            "FAIL CLOSED (NEG29): eval_binding process_pid %r is not a positive int "
            "(the certificate must bind the ACTUAL evaluator child PID)"
            % binding["process_pid"])
    require(isinstance(binding["process_argv"], list) and binding["process_argv"]
            and all(isinstance(a, str) and a for a in binding["process_argv"]),
            "FAIL CLOSED (NEG29): eval_binding process_argv %r is not a non-empty list "
            "of non-empty strings (the ACTUAL evaluator argv must be bound)"
            % binding["process_argv"])
    _require_iso_utc(binding["run_start_utc"], "run_start_utc")
    _require_iso_utc(binding["run_end_utc"], "run_end_utc")
    require(isinstance(binding["run_exit_code"], int)
            and not isinstance(binding["run_exit_code"], bool)
            and binding["run_exit_code"] == 0,
            "FAIL CLOSED (NEG29): eval_binding run_exit_code %r != 0 (a certificate is "
            "only ever emitted on a successful evaluator exit)"
            % binding["run_exit_code"])
    return binding


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
                      has_student_data: bool = False, eval_binding: dict = None) -> dict:
    scenario = evaluation_result["scenario"]
    primary = evaluation_result["metrics"]["primary"]
    cert = {
        "schema": SCHEMA,
        "cert_version": CERT_VERSION,
        "scenario": scenario,
        "identity_class": evaluation_result.get("contract", {}).get("observation_schema") and {
            FULL: "CANONICAL_S4_EVALUATION",
            FRONT: "TIER3_FRONT_DIAGNOSTIC_SCAFFOLD",
            BACK: "BOSS_COMBAT_SCAFFOLDED",   # 收口: combat only; boss-area search is N/A
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
    if eval_binding is not None:
        cert["eval_binding"] = eval_binding
    assert_scaffold_hash_not_global(cert)          # NEG24
    assert_scaffold_does_not_claim_full_success(cert)   # NEG25
    if eval_binding is not None:
        assert_eval_binding_complete(cert)         # NEG27 (real value binding)
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

    # ---- NEG27: REAL value binding (actual SHAs, never labels) ----
    def _binding(**over):
        b = {
            "state_bank_hash": "2" + "a" * 63,
            "state_payload_hashes": ["b" * 64, "c" * 64],
            "checkpoint_file_sha256": "d" * 64,
            "cc2_params_sha256": "e" * 64,
            "checkpoint_step": 4096,
            "carry_mode": "persistent",
            "run_class": "INTERFACE_SMOKE",
            "episode_records_sha256": "f" * 64,
            "cc2_policy_source_sha256": "0" * 64,
            "evaluator_source_sha256": "1" * 64,
            "predicate_code_sha256": "a4fba86b054d20412fc1df2c79e7000d66b0525decb1801f"
                                     "a474ee7fb0d25b4c",
            "observation_shape": [8335],
            "action_dim": 43,
            "params_unchanged": True,
            "performance_claim_authorized": False,
            "driver_source_sha256": "9" * 64,
            "process_pid": 12345,
            "process_argv": ["python", "tier3_evaluator.py", "--interface-smoke"],
            "run_start_utc": "2026-07-30T00:00:00+00:00",
            "run_end_utc": "2026-07-30T00:05:00+00:00",
            "run_exit_code": 0,
        }
        b.update(over)
        return b

    cb = build_certificate(_result(FRONT, 0.5, 4), eval_binding=_binding())
    check("NEG27_complete_binding_accepted", cb["eval_binding"]["params_unchanged"] is True)
    for bad_over, tag in (
            ({"state_bank_hash": "FRONT_SCAFFOLD_STATE_BANK_HASH"}, "label_not_sha"),
            ({"checkpoint_file_sha256": None}, "missing_value"),
            ({"state_payload_hashes": []}, "empty_payload_hashes"),
            ({"observation_shape": [67, 7, 7]}, "wrong_obs_shape"),
            ({"action_dim": 42}, "wrong_action_dim"),
            ({"params_unchanged": False}, "params_changed"),
            ({"run_class": "SMOKE_BUT_PERFORMANCE"}, "bad_run_class"),
            ({"performance_claim_authorized": True}, "smoke_claims_performance"),
            ({"carry_mode": "sideways"}, "bad_carry_mode"),
            # ---- NEG29: process provenance must be real and valid ----
            ({"process_pid": -1}, "NEG29_bad_pid"),
            ({"process_pid": None}, "NEG29_missing_pid"),
            ({"process_argv": []}, "NEG29_empty_argv"),
            ({"process_argv": ["python", ""]}, "NEG29_empty_argv_element"),
            ({"run_start_utc": "not-a-time"}, "NEG29_bad_start_time"),
            ({"run_end_utc": ""}, "NEG29_empty_end_time"),
            ({"run_exit_code": 1}, "NEG29_nonzero_exit_code"),
            ({"run_exit_code": None}, "NEG29_missing_exit_code"),
            ({"driver_source_sha256": "xyz"}, "NEG29_bad_driver_sha")):
        try:
            build_certificate(_result(FRONT, 0.5, 4), eval_binding=_binding(**bad_over))
            check("NEG27_rejects_%s" % tag, False)
        except FailClosed:
            check("NEG27_rejects_%s" % tag, True)
    # A FORMAL_EVALUATION run class MAY (only if separately authorized) carry
    # performance_claim_authorized=True — the completeness gate still passes.
    formal = build_certificate(_result(FRONT, 0.5, 4),
                               eval_binding=_binding(run_class="FORMAL_EVALUATION",
                                                     performance_claim_authorized=True))
    check("NEG27_formal_class_binding_ok",
          formal["eval_binding"]["run_class"] == "FORMAL_EVALUATION")

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
