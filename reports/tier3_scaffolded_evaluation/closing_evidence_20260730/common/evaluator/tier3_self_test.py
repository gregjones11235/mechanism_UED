#!/usr/bin/env python3
"""CC4 Tier3 — aggregate self-test (single entry point; §二十三).

Runs every Tier3 module's self-test IN-PROCESS (boundary schema, predicates, source
audit, serializer, builder, state-bank materializer, checkpoint adapter, metrics,
failure taxonomy, evaluator, certificate) plus the full negative-test suite, and
returns a single process exit code: 0 only if ALL pass. This is the command the final
test battery invokes.

It also prints an honest, machine-checkable STATUS block (the freeze labels): nothing
here can over-claim — real JAX/craftax runs and Student evaluation stay
BLOCKED_ENVIRONMENT / NOT_RUN on this host.
"""
from __future__ import annotations

import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit            # noqa: E402
import tier3_event_predicates as pred         # noqa: E402
import tier3_boundary_schema as boundary      # noqa: E402
import tier3_state_serializer as ser          # noqa: E402
import tier3_scaffold_builder as builder      # noqa: E402
import tier3_state_bank_materializer as mat   # noqa: E402
import tier3_checkpoint_adapter as ckpt       # noqa: E402
import tier3_metrics as metrics               # noqa: E402
import tier3_failure_taxonomy as taxonomy     # noqa: E402
import tier3_evaluator as evaluator           # noqa: E402
import tier3_evaluation_certificate as cert   # noqa: E402
import tier3_negative_tests as neg            # noqa: E402

MODULE_SELF_TESTS = [
    ("source_audit", audit._self_test),
    ("event_predicates", pred.self_test),
    ("boundary_schema", boundary.self_test),
    ("state_serializer", ser.self_test),
    ("scaffold_builder", builder.self_test),
    ("state_bank_materializer", mat.self_test),
    ("checkpoint_adapter", ckpt.self_test),
    ("metrics", metrics.self_test),
    ("failure_taxonomy", taxonomy.self_test),
    ("evaluator", evaluator.self_test),
    ("evaluation_certificate", cert.self_test),
]


def honest_labels() -> dict:
    has_real = ser.have_jax_craftax()
    return {
        "SCHEMA": "mechanism_UED.tier3_self_test_status/v1",
        "BOUNDARY_SCHEMA": "IMPLEMENTED_STATIC",
        "SCAFFOLD_BUILDER": "IMPLEMENTED_STATIC",
        "STATE_BANK_MATERIALIZER": "TESTED_SYNTHETIC",
        "EVALUATOR": "TESTED_SYNTHETIC",
        "REAL_CRAFTAX_SCAFFOLD_TEST": "TESTED_REAL_ENV_RESET" if has_real else "BLOCKED_ENVIRONMENT",
        "REAL_STUDENT_EVALUATION": "NOT_RUN",
        "GLOBAL_WORLD_SET_HASH": "BLOCKED_SOURCE_UNVERIFIED",
        "FRONT_SCAFFOLD_STATE_BANK_HASH": "MATERIALIZED" if has_real else "NOT_MATERIALIZED",
        "BACK_SCAFFOLD_STATE_BANK_HASH": "MATERIALIZED" if has_real else "NOT_MATERIALIZED",
        "NEW_TRAINING_RUNS": 0,
        "FORMAL_EVALUATION_RUNS": 0,
        "CC2_CC3_FILES_TOUCHED": False,
        "HENRY_BRANCH_TOUCHED": False,
        "PUSH_PERFORMED": False,
        "SCAFFOLDED_RESULTS_CAN_REPLACE_FULL_TASK": False,
        "JAX_AVAILABLE": ser.have_jax(),
        "CRAFTAX_AVAILABLE": ser.have_craftax(),
    }


def run_all() -> int:
    failures = []
    print("=" * 70)
    print("CC4 Tier3 aggregate self-test")
    print("=" * 70)
    for name, fn in MODULE_SELF_TESTS:
        print("--- module: %s ---" % name)
        try:
            rc = fn()
        except Exception as exc:
            rc = 1
            print("  unexpected exception: %r" % exc)
        if rc != 0:
            failures.append(name)
    print("--- negative_tests ---")
    neg_rc = neg.self_test()
    if neg_rc != 0:
        failures.append("negative_tests")

    print("=" * 70)
    labels = honest_labels()
    print("HONEST STATUS LABELS:")
    for k in sorted(labels):
        print("  %-42s = %s" % (k, labels[k]))
    print("=" * 70)

    if failures:
        print("TIER3_AGGREGATE_SELF_TEST_FAIL (failed: %s)" % ", ".join(failures))
        return 1
    print("TIER3_AGGREGATE_SELF_TEST_PASS (modules=%d, negative_tests=FAIL0)"
          % len(MODULE_SELF_TESTS))
    return 0


def main(argv=None) -> int:
    return run_all()


if __name__ == "__main__":
    raise SystemExit(main())
