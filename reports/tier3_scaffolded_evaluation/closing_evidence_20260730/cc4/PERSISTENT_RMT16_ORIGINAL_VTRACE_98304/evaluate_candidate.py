#!/usr/bin/env python3
"""CANDIDATE EVALUATION ENTRY SHIM — PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 (closing contract S6).

Delegates to the COMMON evaluator (verified by full SHA256). This round
authorizes INTERFACE_SMOKE binding ONLY: --performance-evaluation and
--round1-screening are REFUSED here (performance_claim_authorized=false; the
formal pool ranking is dispatched by the unified scheduler, never by a
candidate shim).
"""
import hashlib
import os
import subprocess
import sys

CANDIDATE_ID = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"
ARM = "persistent"
COMMON_ROOT = "/home/oseasy/student_pool_v1/common"
COMMON_EVALUATOR_SHA256 = "a47ff97f9dc745c4f0cf015966b777f90c6dd6c7fe934b9b552a542df188a344"
CHECKPOINT_CONTRACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoint_contract.json")
PROHIBITED_THIS_ROUND = ("--performance-evaluation", "--round1-screening")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag in PROHIBITED_THIS_ROUND:
        if flag in argv:
            print("FAIL CLOSED: %s is NOT authorized this round "
                  "(performance_claim_authorized=false)" % flag)
            return 2
    if "--interface-smoke" not in argv:
        argv = ["--interface-smoke"] + argv
    if "--checkpoint-contract" not in argv:
        argv += ["--checkpoint-contract", CHECKPOINT_CONTRACT_PATH]
    if "--arm" not in argv:
        argv += ["--arm", ARM]
    if "--frozen-bank-artifacts" not in argv:
        argv += ["--frozen-bank-artifacts",
                 os.path.join(COMMON_ROOT, "frozen_bank_artifacts")]
    if "--max-steps" not in argv:
        argv += ["--max-steps", "32"]
    evaluator = os.path.join(COMMON_ROOT, "common_evaluator.py")
    if not os.path.isfile(evaluator):
        print("FAIL CLOSED: common evaluator missing at %s" % evaluator)
        return 2
    got = _sha256_file(evaluator)
    if got != COMMON_EVALUATOR_SHA256:
        print("FAIL CLOSED: common evaluator SHA drift %s != frozen %s"
              % (got, COMMON_EVALUATOR_SHA256))
        return 2
    return subprocess.call([sys.executable, evaluator] + argv)


if __name__ == "__main__":
    raise SystemExit(main())
