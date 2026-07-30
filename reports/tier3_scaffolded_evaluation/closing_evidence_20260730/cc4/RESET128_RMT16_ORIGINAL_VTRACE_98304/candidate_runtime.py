#!/usr/bin/env python3
"""CANDIDATE RUNTIME SHIM — RESET128_RMT16_ORIGINAL_VTRACE_98304 (closing contract S5/S6).

Thin real binding of ONE candidate to the COMMON runtime ABI. It defines NO
evaluation semantics and NO scientific predicates
(scientific_predicates_defined_here=false): family registration, memory
semantics, checkpoint verification and action selection all live in the common
runner engine, verified by full SHA256 below (tampering fails closed).
"""
import hashlib
import importlib.util
import os

CANDIDATE_ID = "RESET128_RMT16_ORIGINAL_VTRACE_98304"
RUNTIME_FAMILY = "rmt16_gtrxl_cc2"
ARM = "reset128"
COMMON_ROOT = "/home/oseasy/student_pool_v1/common"
COMMON_RUNNER_SHA256 = "135332d3b30c60cb7b29c620dc931da852e99b2ca256c7a77dbf365dfc94075b"
CHECKPOINT_CONTRACT_SHA256 = "7dda2bc7517342b189a1f1ba949d620eb4d1c978e252b74f4e2bdeb61363f2e5"
CHECKPOINT_FILE_SHA256 = "de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638"
PARAMS_SHA256 = "78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2"
BASE_CHECKPOINT_PARAMS_SHA256 = "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5"
SCIENTIFIC_PREDICATES_DEFINED_HERE = False
TRAINABLE = False
IMMUTABLE = True
CAPSULE_CONTRACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "checkpoint_contract.json")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _common_runner():
    p = os.path.join(COMMON_ROOT, "common_runner.py")
    if not os.path.isfile(p):
        raise SystemExit("FAIL CLOSED: common runner missing at %s" % p)
    got = _sha256_file(p)
    if got != COMMON_RUNNER_SHA256:
        raise SystemExit("FAIL CLOSED: common runner SHA drift %s != frozen %s"
                         % (got, COMMON_RUNNER_SHA256))
    spec = importlib.util.spec_from_file_location("common_runner_bound", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def checkpoint_contract(checkpoint_path):
    return {
        "runtime_family": RUNTIME_FAMILY,
        "arm": ARM,
        "checkpoint_path": checkpoint_path,
        "checkpoint_contract_path": CAPSULE_CONTRACT_PATH,
    }


def load(checkpoint_path):
    """ABI load through the common runner (family dispatch + verification inside)."""
    return _common_runner().load_candidate(checkpoint_contract(checkpoint_path))


def frozen_identities():
    return {
        "candidate_id": CANDIDATE_ID,
        "runtime_family": RUNTIME_FAMILY,
        "arm": ARM,
        "checkpoint_contract_sha256": CHECKPOINT_CONTRACT_SHA256,
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
        "params_sha256": PARAMS_SHA256,
        "base_checkpoint_params_sha256": BASE_CHECKPOINT_PARAMS_SHA256,
        "scientific_predicates_defined_here": SCIENTIFIC_PREDICATES_DEFINED_HERE,
        "trainable": TRAINABLE,
        "immutable": IMMUTABLE,
    }
