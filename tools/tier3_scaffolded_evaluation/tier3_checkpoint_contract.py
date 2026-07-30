#!/usr/bin/env python3
"""CC4 Tier3 — frozen final-98304 checkpoint contract (machine-readable, verified).

The REAL direct-98304 CC2 final checkpoints (both arms) are pinned by
``configs/tier3_cc2_final98304_checkpoint_contract_v1.json``: per-arm checkpoint file
SHA / params SHA / step / arm / carry_mode plus the common identity fields
(replay_mode / seed / run_class / sequence_length / segment_len / crosses_boundary /
base_checkpoint_params_sha256 / driver_source_sha256 / cc2_policy_source_sha256).

``checkpoint_contract_sha256`` is the SHA256 of the canonical JSON bytes of the
contract with that very field removed — the file is self-verifying, and any tamper
with ANY field (or the recorded checksum) fails closed.

``verify_checkpoint_against_contract`` then checks the LOADED real checkpoint —
the actual file SHA, the recomputed params SHA (NEG21 upstream) and every manifest
field — against the declared arm of the contract. A mismatch on ANY field raises
FailClosed with the stable ID ``FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH``. The
evaluation CLI must bind the contract + arm BEFORE any rollout; copying contract
fields into a certificate without verification is not what this module does — every
comparison here is against the actually-loaded bytes / manifest.
"""
from __future__ import annotations

import json
import os
import sys

# Runnable-as-script AND importable-as-package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402

SCHEMA = "mechanism_UED.tier3_final98304_checkpoint_contract/v1"
CONTRACT_VERSION = "tier3_final98304_checkpoint_contract/v1"

# Frozen repo-relative location of the contract (task §一).
DEFAULT_CONTRACT_PATH = os.path.join(
    str(audit.repo_root()), "configs", "tier3_cc2_final98304_checkpoint_contract_v1.json")

ARMS = ("persistent", "reset128")

# Frozen values the contract itself must carry (defense in depth: a contract file
# with the right self-checksum but wrong frozen content is still rejected).
FROZEN_CHECKPOINT_STEP = 98304
FROZEN_ARM_NAME = {"persistent": "RMT16-Persistent-OrigVtrace",
                   "reset128": "RMT16-Reset128-OrigVtrace"}
FROZEN_CARRY_MODE = {"persistent": "persistent", "reset128": "reset128"}
FROZEN_CHECKPOINT_FILE_SHA256 = {
    "persistent": "2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723",
    "reset128": "de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638",
}
FROZEN_PARAMS_SHA256 = {
    "persistent": "aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d",
    "reset128": "78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2",
}
FROZEN_BASE_CHECKPOINT_PARAMS_SHA256 = ("d4e85af58b7f87d689fadea12eec70c852fa098a09f5"
                                        "ea8907448684b3bf60f5")
FROZEN_DRIVER_SOURCE_SHA256 = ("453bd1ecc8d9671c741c4462214bd7699c74611a52ec157ff30c"
                               "d68653b4bafc")
FROZEN_CC2_POLICY_SOURCE_SHA256 = ("31c1092c037577c56ba0eba9d51ea40cc6a97210bbcbc98f"
                                   "e047762daed2f46f")

REQUIRED_COMMON_FIELDS = ("replay_mode", "seed", "run_class", "sequence_length",
                          "segment_len", "crosses_boundary",
                          "base_checkpoint_params_sha256", "driver_source_sha256",
                          "cc2_policy_source_sha256")
REQUIRED_ARM_FIELDS = ("checkpoint_file_sha256", "params_sha256", "checkpoint_step",
                       "arm", "carry_mode")


class FailClosed(Exception):
    """Hard stop on any checkpoint-contract violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _is_sha256_hex(v) -> bool:
    return (isinstance(v, str) and len(v) == 64
            and all(c in "0123456789abcdef" for c in v))


def contract_sha256(contract: dict) -> str:
    """SHA256 of the canonical JSON bytes of the contract with the
    ``checkpoint_contract_sha256`` field removed (sorted keys, compact separators)."""
    import hashlib
    body = {k: v for k, v in contract.items() if k != "checkpoint_contract_sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canon).hexdigest()


def load_contract(path: str = DEFAULT_CONTRACT_PATH) -> dict:
    """Load + structurally validate + self-checksum-verify the contract file.

    Any structural problem, a wrong recorded ``checkpoint_contract_sha256``, or a
    frozen-value mismatch fails closed."""
    require(os.path.isfile(path),
            "FAIL CLOSED: checkpoint contract not found at %r" % path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    require(isinstance(doc, dict), "FAIL CLOSED: contract is not a JSON object")
    require(doc.get("schema") == SCHEMA,
            "FAIL CLOSED: contract schema %r != %r" % (doc.get("schema"), SCHEMA))
    require(doc.get("contract_version") == CONTRACT_VERSION,
            "FAIL CLOSED: contract_version %r != %r"
            % (doc.get("contract_version"), CONTRACT_VERSION))
    # Self-checksum: the recorded value must reproduce from the remaining bytes.
    recorded = doc.get("checkpoint_contract_sha256")
    require(_is_sha256_hex(recorded),
            "FAIL CLOSED: contract checkpoint_contract_sha256 %r is not a 64-hex value"
            % (recorded,))
    recomputed = contract_sha256(doc)
    require(recomputed == recorded,
            "FAIL CLOSED: contract checkpoint_contract_sha256 %s does not reproduce "
            "(recomputed %s) — the contract file was tampered with"
            % (recorded[:16], recomputed[:16]))
    arms = doc.get("arms")
    require(isinstance(arms, dict) and set(arms) == set(ARMS),
            "FAIL CLOSED: contract arms %r != %r" % (sorted(arms or {}), ARMS))
    for arm in ARMS:
        a = arms[arm]
        require(isinstance(a, dict), "FAIL CLOSED: contract arm %r is not an object" % arm)
        missing = [f for f in REQUIRED_ARM_FIELDS if f not in a]
        require(not missing, "FAIL CLOSED: contract arm %r missing field(s) %s" % (arm, missing))
        require(_is_sha256_hex(a["checkpoint_file_sha256"]),
                "FAIL CLOSED: contract arm %r checkpoint_file_sha256 not 64-hex" % arm)
        require(_is_sha256_hex(a["params_sha256"]),
                "FAIL CLOSED: contract arm %r params_sha256 not 64-hex" % arm)
        # Frozen-value bindings (even a self-consistent contract with wrong content
        # is rejected).
        require(a["checkpoint_step"] == FROZEN_CHECKPOINT_STEP,
                "FAIL CLOSED: contract arm %r checkpoint_step %r != frozen %d"
                % (arm, a["checkpoint_step"], FROZEN_CHECKPOINT_STEP))
        require(a["arm"] == FROZEN_ARM_NAME[arm],
                "FAIL CLOSED: contract arm %r arm %r != frozen %r"
                % (arm, a["arm"], FROZEN_ARM_NAME[arm]))
        require(a["carry_mode"] == FROZEN_CARRY_MODE[arm],
                "FAIL CLOSED: contract arm %r carry_mode %r != frozen %r"
                % (arm, a["carry_mode"], FROZEN_CARRY_MODE[arm]))
        require(a["checkpoint_file_sha256"] == FROZEN_CHECKPOINT_FILE_SHA256[arm],
                "FAIL CLOSED: contract arm %r checkpoint_file_sha256 %s != frozen %s"
                % (arm, a["checkpoint_file_sha256"][:16],
                   FROZEN_CHECKPOINT_FILE_SHA256[arm][:16]))
        require(a["params_sha256"] == FROZEN_PARAMS_SHA256[arm],
                "FAIL CLOSED: contract arm %r params_sha256 %s != frozen %s"
                % (arm, a["params_sha256"][:16], FROZEN_PARAMS_SHA256[arm][:16]))
    common = doc.get("common")
    require(isinstance(common, dict), "FAIL CLOSED: contract has no common object")
    missing = [f for f in REQUIRED_COMMON_FIELDS if f not in common]
    require(not missing, "FAIL CLOSED: contract common missing field(s) %s" % missing)
    require(common["replay_mode"] == "original_vtrace",
            "FAIL CLOSED: contract common replay_mode %r != original_vtrace"
            % common["replay_mode"])
    require(common["seed"] == 42,
            "FAIL CLOSED: contract common seed %r != 42" % common["seed"])
    require(common["run_class"] == "long_run_98304",
            "FAIL CLOSED: contract common run_class %r != long_run_98304"
            % common["run_class"])
    require(common["sequence_length"] == 129,
            "FAIL CLOSED: contract common sequence_length %r != 129"
            % common["sequence_length"])
    require(common["segment_len"] == 128,
            "FAIL CLOSED: contract common segment_len %r != 128" % common["segment_len"])
    require(common["crosses_boundary"] is True,
            "FAIL CLOSED: contract common crosses_boundary %r != true"
            % common["crosses_boundary"])
    require(common["base_checkpoint_params_sha256"] == FROZEN_BASE_CHECKPOINT_PARAMS_SHA256,
            "FAIL CLOSED: contract common base_checkpoint_params_sha256 %s != frozen %s"
            % (str(common["base_checkpoint_params_sha256"])[:16],
               FROZEN_BASE_CHECKPOINT_PARAMS_SHA256[:16]))
    require(common["driver_source_sha256"] == FROZEN_DRIVER_SOURCE_SHA256,
            "FAIL CLOSED: contract common driver_source_sha256 %s != frozen %s"
            % (str(common["driver_source_sha256"])[:16], FROZEN_DRIVER_SOURCE_SHA256[:16]))
    require(common["cc2_policy_source_sha256"] == FROZEN_CC2_POLICY_SOURCE_SHA256,
            "FAIL CLOSED: contract common cc2_policy_source_sha256 %s != frozen %s"
            % (str(common["cc2_policy_source_sha256"])[:16],
               FROZEN_CC2_POLICY_SOURCE_SHA256[:16]))
    return doc


def verify_checkpoint_against_contract(arm_key: str, checkpoint_file_sha256: str,
                                       params_sha256: str, manifest: dict,
                                       driver_source_sha256: str,
                                       cc2_policy_source_sha256: str,
                                       contract: dict = None) -> dict:
    """Verify a LOADED real checkpoint against the declared arm of the contract.

    Every comparison is against the actually-loaded bytes / manifest (the file SHA
    and the recomputed params SHA come from tier3_checkpoint_adapter.load_full_params_readonly;
    driver_source_sha256 / cc2_policy_source_sha256 from the loaded sources). ANY
    mismatch raises FailClosed carrying the stable ID
    FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH.
    """
    MID = "FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH"
    require(arm_key in ARMS,
            "FAIL CLOSED (%s): --arm %r not in %r" % (MID, arm_key, ARMS))
    if contract is None:
        contract = load_contract()
    arm = contract["arms"][arm_key]
    common = contract["common"]

    def mismatch(field, got, want):
        raise FailClosed(
            "FAIL CLOSED (%s): arm=%s field %s: loaded=%r contract=%r"
            % (MID, arm_key, field, got, want))

    if checkpoint_file_sha256 != arm["checkpoint_file_sha256"]:
        mismatch("checkpoint_file_sha256", checkpoint_file_sha256,
                 arm["checkpoint_file_sha256"])
    if params_sha256 != arm["params_sha256"]:
        mismatch("params_sha256", params_sha256, arm["params_sha256"])
    require(isinstance(manifest, dict),
            "FAIL CLOSED (%s): loaded checkpoint has no manifest dict" % MID)
    if manifest.get("step") != arm["checkpoint_step"]:
        mismatch("manifest.step (checkpoint_step)", manifest.get("step"),
                 arm["checkpoint_step"])
    if manifest.get("arm") != arm["arm"]:
        mismatch("manifest.arm", manifest.get("arm"), arm["arm"])
    if manifest.get("carry_mode") != arm["carry_mode"]:
        mismatch("manifest.carry_mode", manifest.get("carry_mode"), arm["carry_mode"])
    if manifest.get("replay_mode") != common["replay_mode"]:
        mismatch("manifest.replay_mode", manifest.get("replay_mode"),
                 common["replay_mode"])
    if manifest.get("seed") != common["seed"]:
        mismatch("manifest.seed", manifest.get("seed"), common["seed"])
    p4 = manifest.get("phase4a_v2")
    require(isinstance(p4, dict),
            "FAIL CLOSED (%s): loaded manifest has no phase4a_v2 dict" % MID)
    if p4.get("run_class") != common["run_class"]:
        mismatch("manifest.phase4a_v2.run_class", p4.get("run_class"),
                 common["run_class"])
    if p4.get("sequence_length") != common["sequence_length"]:
        mismatch("manifest.phase4a_v2.sequence_length", p4.get("sequence_length"),
                 common["sequence_length"])
    if p4.get("segment_len") != common["segment_len"]:
        mismatch("manifest.phase4a_v2.segment_len", p4.get("segment_len"),
                 common["segment_len"])
    if p4.get("crosses_boundary") is not common["crosses_boundary"]:
        mismatch("manifest.phase4a_v2.crosses_boundary", p4.get("crosses_boundary"),
                 common["crosses_boundary"])
    if p4.get("base_checkpoint_params_sha256") != common["base_checkpoint_params_sha256"]:
        mismatch("manifest.phase4a_v2.base_checkpoint_params_sha256",
                 p4.get("base_checkpoint_params_sha256"),
                 common["base_checkpoint_params_sha256"])
    if driver_source_sha256 != common["driver_source_sha256"]:
        mismatch("driver_source_sha256 (loaded source)", driver_source_sha256,
                 common["driver_source_sha256"])
    if cc2_policy_source_sha256 != common["cc2_policy_source_sha256"]:
        mismatch("cc2_policy_source_sha256 (loaded source)", cc2_policy_source_sha256,
                 common["cc2_policy_source_sha256"])
    return {
        "verified": True,
        "arm": arm_key,
        "checkpoint_contract_sha256": contract["checkpoint_contract_sha256"],
        "checkpoint_step": arm["checkpoint_step"],
        "arm_manifest_name": arm["arm"],
        "carry_mode": arm["carry_mode"],
    }


# ---------------------------------------------------------------------------
# Self-test (pure stdlib; runs on ANY host).
# ---------------------------------------------------------------------------
def _synthetic_manifest(**over):
    m = {
        "params_sha256": FROZEN_PARAMS_SHA256["persistent"],
        "step": 98304,
        "arm": FROZEN_ARM_NAME["persistent"],
        "carry_mode": "persistent",
        "replay_mode": "original_vtrace",
        "seed": 42,
        "config": {},
        "phase4a_v2": {
            "run_class": "long_run_98304",
            "sequence_length": 129,
            "segment_len": 128,
            "crosses_boundary": True,
            "base_checkpoint_params_sha256": FROZEN_BASE_CHECKPOINT_PARAMS_SHA256,
        },
        "tag": "save",
    }
    m.update(over)
    return m


def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # The committed contract file loads + self-checksum-verifies + frozen-binds.
    doc = load_contract()
    check("contract_self_checksum",
          contract_sha256(doc) == doc["checkpoint_contract_sha256"])

    # A fully-matching loaded checkpoint verifies against the persistent arm.
    ok = verify_checkpoint_against_contract(
        "persistent", FROZEN_CHECKPOINT_FILE_SHA256["persistent"],
        FROZEN_PARAMS_SHA256["persistent"], _synthetic_manifest(),
        FROZEN_DRIVER_SOURCE_SHA256, FROZEN_CC2_POLICY_SOURCE_SHA256, doc)
    check("contract_match_accepted",
          ok["verified"] is True
          and ok["checkpoint_contract_sha256"] == doc["checkpoint_contract_sha256"])
    ok2 = verify_checkpoint_against_contract(
        "reset128", FROZEN_CHECKPOINT_FILE_SHA256["reset128"],
        FROZEN_PARAMS_SHA256["reset128"],
        _synthetic_manifest(arm=FROZEN_ARM_NAME["reset128"], carry_mode="reset128",
                            params_sha256=FROZEN_PARAMS_SHA256["reset128"]),
        FROZEN_DRIVER_SOURCE_SHA256, FROZEN_CC2_POLICY_SOURCE_SHA256, doc)
    check("contract_match_accepted_reset128", ok2["verified"] is True)

    # Tamper matrix: every mismatch path must raise the stable MISMATCH id.
    def mismatches(fn):
        try:
            fn()
            return False
        except FailClosed as exc:
            return "FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH" in str(exc)

    base = dict(file_sha=FROZEN_CHECKPOINT_FILE_SHA256["persistent"],
                params_sha=FROZEN_PARAMS_SHA256["persistent"],
                manifest=_synthetic_manifest(),
                driver_sha=FROZEN_DRIVER_SOURCE_SHA256,
                policy_sha=FROZEN_CC2_POLICY_SOURCE_SHA256)

    def run(arm="persistent", **kw):
        b = dict(base)
        b.update(kw)
        return verify_checkpoint_against_contract(
            arm, b["file_sha"], b["params_sha"], b["manifest"],
            b["driver_sha"], b["policy_sha"], doc)

    check("NEG30_wrong_file_sha_rejected",
          mismatches(lambda: run(file_sha="0" * 64)))
    check("NEG31_wrong_params_sha_rejected",
          mismatches(lambda: run(params_sha="1" * 64)))
    check("NEG32_step8192_impersonating_final_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(step=8192))))
    check("NEG33_wrong_arm_name_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(arm="RMT16-Evil-Arm"))))
    check("NEG33_wrong_carry_mode_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(carry_mode="reset128"))))
    check("NEG34_wrong_replay_mode_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(replay_mode="replay"))))
    check("NEG35_wrong_seed_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(seed=43))))
    check("NEG35_wrong_run_class_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(
              phase4a_v2=dict(_synthetic_manifest()["phase4a_v2"], run_class="smoke")))))
    check("NEG36_wrong_base_checkpoint_sha_rejected",
          mismatches(lambda: run(manifest=_synthetic_manifest(
              phase4a_v2=dict(_synthetic_manifest()["phase4a_v2"],
                              base_checkpoint_params_sha256="2" * 64)))))
    check("wrong_driver_source_sha_rejected",
          mismatches(lambda: run(driver_sha="3" * 64)))
    check("wrong_policy_source_sha_rejected",
          mismatches(lambda: run(policy_sha="4" * 64)))
    check("arm_key_outside_contract_rejected",
          mismatches(lambda: verify_checkpoint_against_contract(
              "sideways", base["file_sha"], base["params_sha"], base["manifest"],
              base["driver_sha"], base["policy_sha"], doc)))
    # The reset128 file SHA must NOT verify against the persistent arm entry.
    check("cross_arm_file_sha_rejected",
          mismatches(lambda: run(file_sha=FROZEN_CHECKPOINT_FILE_SHA256["reset128"])))

    # A tampered contract file (checksum intact but content mutated) is rejected by
    # load_contract's frozen-value bindings even though it is structurally valid.
    tampered = json.loads(json.dumps(doc))
    tampered["arms"]["persistent"]["checkpoint_step"] = 8192
    tampered["checkpoint_contract_sha256"] = contract_sha256(tampered)   # re-seal
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "tampered_contract.json")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(tampered, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        try:
            load_contract(p)
            check("resealed_tampered_contract_rejected", False)
        except FailClosed:
            check("resealed_tampered_contract_rejected", True)
        # A contract with a broken self-checksum is rejected outright.
        broken = json.loads(json.dumps(doc))
        broken["common"]["seed"] = 43
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(broken, fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
        try:
            load_contract(p)
            check("broken_self_checksum_rejected", False)
        except FailClosed:
            check("broken_self_checksum_rejected", True)

    if problems:
        print("TIER3_CHECKPOINT_CONTRACT_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_CHECKPOINT_CONTRACT_SELF_TEST_PASS (contract_sha256=%s; NEG30-36 guards live)"
          % doc["checkpoint_contract_sha256"][:16])
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--print-sha" in argv:
        print(load_contract()["checkpoint_contract_sha256"])
        return 0
    print("usage: tier3_checkpoint_contract.py --self-test | --print-sha")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
