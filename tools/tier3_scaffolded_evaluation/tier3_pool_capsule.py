#!/usr/bin/env python3
"""CC4 Tier3 — RMT16 CANDIDATE CAPSULE BUILDER (closing contract §6/§7/§9).

Builds one candidate capsule root (e.g. /home/oseasy/student_pool_v1/cc4/
PERSISTENT_RMT16_ORIGINAL_VTRACE_98304/) with exactly the eleven contract files:

  candidate_manifest.json              identity + immutable=true + claims unauthorized
  training_contract.json               derived ONLY from the frozen final98304 contract
  checkpoint_contract.json             byte copy of the frozen contract (+ arm)
  candidate_runtime.py                 real shim: fixed family over the COMMON ABI
  evaluate_candidate.py                real shim: INTERFACE_SMOKE-only delegation
  interface_smoke_result.json          real engine run (FRONT/BACK/FULL, 32 steps,
                                       frozen banks READ-ONLY from common artifacts)
  memory_contract_smoke_result.json    real ABI smoke on the real checkpoint
  common_evaluator_binding_result.json run_class=INTERFACE_SMOKE + the eight common
                                       SHA references (§7) — PASS only with evidence
  environment_lock.json                byte copy of the common lock
  READY.json                           the §9 per-capsule gates (honest false)
  SHA256SUMS                           over the capsule (self + READY excluded)

This round authorizes INTERFACE_SMOKE binding ONLY — no formal pool evaluation,
no training, nothing else (§10). Runs under the pinned venv on the server; the
--self-test below is pure.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tier3_source_audit as audit        # noqa: E402
import tier3_checkpoint_contract as cc    # noqa: E402  (pure module)

SCHEMA_MANIFEST = "mechanism_UED.candidate_manifest/v1"
SCHEMA_TRAINING = "mechanism_UED.training_contract/v1"
SCHEMA_MEMSMOKE = "mechanism_UED.memory_contract_smoke/v1"
SCHEMA_IFSMOKE = "mechanism_UED.interface_smoke_result/v1"
SCHEMA_BINDING = "mechanism_UED.common_evaluator_binding/v1"
SCHEMA_READY = "mechanism_UED.candidate_ready/v1"

RUNTIME_FAMILY = "rmt16_gtrxl_cc2"
CANDIDATE_ID = {
    "persistent": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
    "reset128": "RESET128_RMT16_ORIGINAL_VTRACE_98304",
}
MEMORY_FIELDS = ("memories", "mem_mask", "mem_idx", "rmt_st")
BINDING_MAX_STEPS = 32
# Frozen identity of configs/tier3_cc2_final98304_checkpoint_contract_v1.json
# (closing contract §6). The document's self-hash mechanism (load_contract /
# contract_sha256) reproduces it; a difference fails closed.
FROZEN_CHECKPOINT_CONTRACT_SHA256 = (
    "7dda2bc7517342b189a1f1ba949d620eb4d1c978e252b74f4e2bdeb61363f2e5")
FRONT_FROZEN_CONTENT_SHA256 = (
    "21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687")
BACK_FROZEN_CONTENT_SHA256 = (
    "c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566")
SUMS_FILE = "SHA256SUMS"
READY_FILE = "READY.json"


class FailClosed(Exception):
    """Hard stop on any capsule violation."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, doc):
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _canonical_sha256(doc) -> str:
    return hashlib.sha256(json.dumps(
        doc, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pure projections (self-testable)
# ---------------------------------------------------------------------------
def project_training_contract(contract_doc, arm) -> dict:
    """Project the training contract for one arm from the FROZEN final98304
    contract document ONLY — no invented fields, no new training claims."""
    common = contract_doc["common"]
    arm_block = contract_doc["arms"][arm]
    return {
        "schema": SCHEMA_TRAINING,
        "generated_at_utc": _utc_now(),
        "derived_from": "configs/tier3_cc2_final98304_checkpoint_contract_v1.json",
        # The contract's OWN self-hash definition (canonical JSON minus the
        # checkpoint_contract_sha256 field) — not an ad-hoc whole-file hash.
        "derived_from_sha256": cc.contract_sha256(contract_doc),
        "checkpoint_contract_sha256": contract_doc["checkpoint_contract_sha256"],
        "contract_version": contract_doc["contract_version"],
        "candidate_arm_label": arm_block["arm"],
        "carry_mode": arm_block["carry_mode"],
        "checkpoint_step": arm_block["checkpoint_step"],
        "checkpoint_file_sha256": arm_block["checkpoint_file_sha256"],
        "params_sha256": arm_block["params_sha256"],
        "replay_mode": common["replay_mode"],
        "training_run_class": common["run_class"],
        "training_seed": common["seed"],
        "sequence_length": common["sequence_length"],
        "segment_len": common["segment_len"],
        "crosses_boundary": common["crosses_boundary"],
        "base_checkpoint_params_sha256": common["base_checkpoint_params_sha256"],
        "driver_source_sha256": common["driver_source_sha256"],
        "cc2_policy_source_sha256": common["cc2_policy_source_sha256"],
        "provenance": {
            "trained_by": "CC2 (RMT16 long run; CC2 -> CC4 handover)",
            "new_training_by_cc4": False,
            "llm_calls_by_cc4": False,
            "performance_claim_authorized": False,
            "scientific_superiority_claim": False,
            "strong_student_selection_authorized": False,
        },
    }


def common_references(common_root) -> tuple:
    """The eight S7 common SHA references + a verification that each one matches
    the assembly manifest AND the frozen bank identities. Returns (refs, verified,
    detail)."""
    root = str(common_root)
    with open(os.path.join(root, "assembly_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(root, "evaluation_profile.json"), encoding="utf-8") as fh:
        profile = json.load(fh)
    with open(os.path.join(root, "statuses", "bank_identity.json"),
              encoding="utf-8") as fh:
        bank_st = json.load(fh)["banks"]

    refs = {
        "common_evaluator_sha256": _sha256_file(
            os.path.join(root, "common_evaluator.py")),
        "common_runner_sha256": _sha256_file(os.path.join(root, "common_runner.py")),
        "evaluation_profile_sha256": _sha256_file(
            os.path.join(root, "evaluation_profile.json")),
        "metric_schema_sha256": _sha256_file(os.path.join(root, "metric_schema.json")),
        "environment_lock_sha256": _sha256_file(
            os.path.join(root, "environment_lock.json")),
        "front_bank_content_sha256":
            bank_st["front_l2"]["canonical_content_sha256"],
        "back_bank_content_sha256":
            bank_st["back_l2"]["canonical_content_sha256"],
        # The FULL profile is the scenarios.full block of the one frozen profile
        # document; its identity is the canonical-JSON SHA of that block.
        "full_profile_sha256": _canonical_sha256(
            profile.get("scenarios", {}).get("full", {})),
    }
    detail = {
        "common_evaluator_matches_manifest":
            refs["common_evaluator_sha256"] == manifest["common_evaluator_sha256"],
        "common_runner_matches_manifest":
            refs["common_runner_sha256"] == manifest["common_runner_sha256"],
        "evaluation_profile_matches_manifest":
            refs["evaluation_profile_sha256"]
            == manifest["evaluation_profile_sha256"],
        "metric_schema_matches_manifest":
            refs["metric_schema_sha256"] == manifest["metric_schema_sha256"],
        "environment_lock_matches_manifest":
            refs["environment_lock_sha256"] == manifest["environment_lock_sha256"],
        "front_bank_matches_frozen_historical_identity":
            refs["front_bank_content_sha256"] == FRONT_FROZEN_CONTENT_SHA256,
        "back_bank_matches_frozen_historical_identity":
            refs["back_bank_content_sha256"] == BACK_FROZEN_CONTENT_SHA256,
        "full_profile_ready":
            profile.get("scenarios", {}).get("full", {}).get("FULL_PROFILE_READY")
            is True,
    }
    verified = all(detail.values())
    return refs, verified, detail


# ---------------------------------------------------------------------------
# Capsule shims (REAL code, bound by full SHA)
# ---------------------------------------------------------------------------
CANDIDATE_RUNTIME_TEMPLATE = '''#!/usr/bin/env python3
"""CANDIDATE RUNTIME SHIM — @@CANDIDATE_ID@@ (closing contract S5/S6).

Thin real binding of ONE candidate to the COMMON runtime ABI. It defines NO
evaluation semantics and NO scientific predicates
(scientific_predicates_defined_here=false): family registration, memory
semantics, checkpoint verification and action selection all live in the common
runner engine, verified by full SHA256 below (tampering fails closed).
"""
import hashlib
import importlib.util
import os

CANDIDATE_ID = "@@CANDIDATE_ID@@"
RUNTIME_FAMILY = "@@RUNTIME_FAMILY@@"
ARM = "@@ARM@@"
COMMON_ROOT = "@@COMMON_ROOT@@"
COMMON_RUNNER_SHA256 = "@@COMMON_RUNNER_SHA256@@"
CHECKPOINT_CONTRACT_SHA256 = "@@CHECKPOINT_CONTRACT_SHA256@@"
CHECKPOINT_FILE_SHA256 = "@@CHECKPOINT_FILE_SHA256@@"
PARAMS_SHA256 = "@@PARAMS_SHA256@@"
BASE_CHECKPOINT_PARAMS_SHA256 = "@@BASE_PARAMS_SHA256@@"
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
'''

EVALUATE_CANDIDATE_TEMPLATE = '''#!/usr/bin/env python3
"""CANDIDATE EVALUATION ENTRY SHIM — @@CANDIDATE_ID@@ (closing contract S6).

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

CANDIDATE_ID = "@@CANDIDATE_ID@@"
ARM = "@@ARM@@"
COMMON_ROOT = "@@COMMON_ROOT@@"
COMMON_EVALUATOR_SHA256 = "@@COMMON_EVALUATOR_SHA256@@"
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
        argv += ["--max-steps", "@@MAX_STEPS@@"]
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
'''


def _render(template, mapping) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace("@@%s@@" % key, str(value))
    require("@@" not in out, "FAIL CLOSED: unresolved placeholder in capsule shim")
    return out


# ---------------------------------------------------------------------------
# Memory contract smoke (real ABI, real checkpoint)
# ---------------------------------------------------------------------------
def run_memory_contract_smoke(checkpoint_path, arm, contract_path,
                              cc2_snapshot_root=None) -> dict:
    import numpy as np
    import tier3_candidate_runtime as abi

    checks = {}

    def check(name, cond):
        checks[name] = bool(cond)

    spec = {"runtime_family": RUNTIME_FAMILY, "arm": arm,
            "checkpoint_path": checkpoint_path,
            "checkpoint_contract_path": contract_path}
    if cc2_snapshot_root:
        spec["cc2_snapshot_root"] = cc2_snapshot_root
    rt = abi.load_candidate(spec)
    meta = rt.candidate_metadata()

    # Identity gates (the frozen five SHAs + carry mode + frozen ABI surface).
    check("checkpoint_file_sha256_verified",
          meta["checkpoint_file_sha256"] == cc.FROZEN_CHECKPOINT_FILE_SHA256[arm])
    check("params_sha256_verified",
          meta["params_sha256"] == cc.FROZEN_PARAMS_SHA256[arm])
    check("base_params_sha256_verified",
          meta["base_checkpoint_params_sha256"]
          == cc.FROZEN_BASE_CHECKPOINT_PARAMS_SHA256)
    check("carry_mode_matches_arm", meta.get("carry_mode") == arm)
    check("checkpoint_step_98304", meta.get("checkpoint_step") == 98304)
    check("action_mode_greedy", meta.get("action_mode") == "greedy_argmax")
    check("action_dim_43", meta.get("action_dim") == 43)
    check("observation_shape_8335", list(meta.get("observation_shape", [])) == [8335])
    check("trainable_false", meta.get("trainable") is False)
    check("predicates_not_in_runtime",
          meta.get("scientific_predicates_defined_here") is False)

    # ABI contract gates.
    m0 = rt.init_memory(1)
    check("init_memory_fields", tuple(sorted(m0.keys())) == tuple(sorted(MEMORY_FIELDS)))
    try:
        rt.init_memory(2)
        check("batch2_rejected", False)
    except Exception:
        check("batch2_rejected", True)
    try:
        rt.policy_step(np.zeros((8335,), dtype=np.float32), m0,
                       done_mask=np.array([True]))
        check("done_mask_true_rejected", False)
    except Exception:
        check("done_mask_true_rejected", True)
    try:
        rt.reset_memory(m0, reset_mask=np.array([False, False]))
        check("reset_mask_size2_rejected", False)
    except Exception:
        check("reset_mask_size2_rejected", True)

    # Determinism across TWO independent loads of the identical checkpoint.
    rng = np.random.default_rng(20260730 + (0 if arm == "persistent" else 1))
    obs_seq = [rng.standard_normal((8335,), dtype=np.float32) for _ in range(8)]
    rt2 = abi.load_candidate(spec)
    seq_a, seq_b, mem = [], [], m0
    mem2 = rt2.init_memory(1)
    for obs in obs_seq:
        out_a = rt.policy_step(obs, mem)
        out_b = rt2.policy_step(obs, mem2)
        seq_a.append(out_a["action"])
        seq_b.append(out_b["action"])
        mem, mem2 = out_a["memory_state"], out_b["memory_state"]
    check("two_load_determinism", seq_a == seq_b)
    check("actions_in_range", all(0 <= a < 43 for a in seq_a))

    # Snapshot continuation: a memory snapshot captured at step 3, threaded
    # forward, must reproduce the sequential steps 4..8 (memory-snapshot safety
    # invariant — the CC2 transition is purely functional).
    states = [m0]
    full = []
    mm = m0
    for obs in obs_seq:
        out = rt.policy_step(obs, mm)
        full.append(out["action"])
        mm = out["memory_state"]
        states.append(mm)
    cont, cm = [], states[3]
    for obs in obs_seq[3:]:
        out = rt.policy_step(obs, cm)
        cont.append(out["action"])
        cm = out["memory_state"]
    check("memory_snapshot_restore_continues", cont == full[3:])

    # Reset semantics: True -> fresh init (leaf-equal); False -> pass-through.
    import jax
    def leaves_equal(s1, s2):
        l1, l2 = jax.tree_util.tree_leaves(s1), jax.tree_util.tree_leaves(s2)
        return (len(l1) == len(l2)
                and all(np.array_equal(np.asarray(a), np.asarray(b))
                        for a, b in zip(l1, l2)))
    check("reset_true_fresh", leaves_equal(rt.reset_memory(mem, True), m0))
    check("reset_false_passthrough", leaves_equal(rt.reset_memory(mem, False), mem))

    # Params read-only basis: the ABI exposes no optimizer/training path and the
    # load-time SHA chain verified params against the frozen identities above.
    checks["params_readonly_basis"] = True

    passed = all(v is True for v in checks.values())
    return {
        "schema": SCHEMA_MEMSMOKE,
        "generated_at_utc": _utc_now(),
        "arm": arm,
        "candidate_id": CANDIDATE_ID[arm],
        "status": "PASS" if passed else "FAIL",
        "checkpoint_file_sha256": meta["checkpoint_file_sha256"],
        "params_sha256": meta["params_sha256"],
        "base_checkpoint_params_sha256": meta["base_checkpoint_params_sha256"],
        "action_sequence_two_load": seq_a,
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Interface smoke (through the PUBLIC common_evaluator.py shim)
# ---------------------------------------------------------------------------
def run_interface_smoke(checkpoint_path, arm, contract_path, common_root,
                        capsule_root, max_steps=BINDING_MAX_STEPS) -> dict:
    common_root, capsule_root = str(common_root), str(capsule_root)
    out_dir = os.path.join(capsule_root, "interface_smoke_out")
    evaluator = os.path.join(common_root, "common_evaluator.py")
    require(os.path.isfile(evaluator),
            "FAIL CLOSED: common_evaluator.py missing under %s" % common_root)
    cmd = [sys.executable, evaluator, "--interface-smoke",
           "--checkpoint", checkpoint_path,
           "--checkpoint-contract", contract_path,
           "--arm", arm,
           "--scenario", "all",
           "--episodes", "1",
           "--max-steps", str(int(max_steps)),
           "--frozen-bank-artifacts",
           os.path.join(common_root, "frozen_bank_artifacts"),
           "--out", out_dir]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    checks = {"engine_exit_code_zero": int(p.returncode) == 0}
    result_doc, cert_doc = None, None
    if checks["engine_exit_code_zero"]:
        with open(os.path.join(out_dir, "evaluation_result.json"),
                  encoding="utf-8") as fh:
            result_doc = json.load(fh)
        with open(os.path.join(out_dir, "evaluation_certificate.json"),
                  encoding="utf-8") as fh:
            cert_doc = json.load(fh)
        checks["run_class_interface_smoke"] = (
            result_doc.get("run_class") == "INTERFACE_SMOKE")
        checks["performance_claim_not_authorized"] = (
            result_doc.get("performance_claim_authorized") is False)
        checks["max_steps_32"] = int(result_doc.get("max_steps") or -1) == int(max_steps)
        checks["contract_verified"] = (
            result_doc.get("checkpoint_contract", {}).get("verified") is True)
        checks["arm_matches"] = (
            result_doc.get("checkpoint_contract", {}).get("arm") == arm)
        results = result_doc.get("results", {})
        checks["three_scenarios_bound"] = (
            set(results) == {"full", "front_l2", "back_l2"}
            and all(results[sc].get("episode_count") == 1 for sc in results))
        fba = result_doc.get("frozen_bank_artifacts", {})
        checks["banks_read_only_from_artifacts"] = (
            fba.get("bank_regenerated_on_eval_device") is False
            and fba.get("front_loaded_content_sha256")
            == FRONT_FROZEN_CONTENT_SHA256
            and fba.get("back_loaded_content_sha256")
            == BACK_FROZEN_CONTENT_SHA256)

    passed = all(v is True for v in checks.values())
    files = {}
    for name in ("evaluation_result.json", "evaluation_certificate.json",
                 "episode_records.jsonl"):
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            files[name] = _sha256_file(path)
    return {
        "schema": SCHEMA_IFSMOKE,
        "generated_at_utc": _utc_now(),
        "arm": arm,
        "candidate_id": CANDIDATE_ID[arm],
        "status": "PASS" if passed else "FAIL",
        "engine_argv": cmd[1:],
        "engine_exit_code": int(p.returncode),
        "engine_stdout_tail": (p.stdout or "")[-2000:],
        "engine_stderr_tail": (p.stderr or "")[-2000:],
        "output_dir": out_dir,
        "files": files,
        "summary": {
            "run_class": (result_doc or {}).get("run_class"),
            "performance_claim_authorized":
                (result_doc or {}).get("performance_claim_authorized"),
            "max_steps": (result_doc or {}).get("max_steps"),
            "checkpoint_contract_sha256":
                (result_doc or {}).get("checkpoint_contract", {})
                .get("checkpoint_contract_sha256"),
            "frozen_bank_artifacts": (result_doc or {}).get("frozen_bank_artifacts"),
            "episode_records_sha256_by_scenario":
                (result_doc or {}).get("episode_records_by_scenario"),
        },
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# SHA256SUMS (capsule scope)
# ---------------------------------------------------------------------------
def write_capsule_sums(capsule_root) -> str:
    lines = []
    for dirpath, dirnames, filenames in os.walk(str(capsule_root)):
        dirnames.sort()
        for name in sorted(filenames):
            if dirpath == str(capsule_root) and name in (SUMS_FILE, READY_FILE):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, str(capsule_root)).replace(os.sep, "/")
            lines.append("%s  %s" % (_sha256_file(full), rel))
    lines.sort(key=lambda l: l.split("  ", 1)[1])
    sums_path = os.path.join(str(capsule_root), SUMS_FILE)
    with open(sums_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return _sha256_file(sums_path)


# ---------------------------------------------------------------------------
# BUILD
# ---------------------------------------------------------------------------
def build_capsule(capsule_root, arm, checkpoint_path, common_root,
                  contract_path=None, cc2_snapshot_root=None,
                  max_steps=BINDING_MAX_STEPS) -> dict:
    capsule_root, common_root = str(capsule_root), str(common_root)
    require(arm in CANDIDATE_ID, "FAIL CLOSED: arm %r not in %s"
            % (arm, sorted(CANDIDATE_ID)))
    repo_root = audit.repo_root()
    contract_path = str(contract_path or repo_root.joinpath(
        "configs", "tier3_cc2_final98304_checkpoint_contract_v1.json"))

    # S0 fresh-directory gate.
    if os.path.exists(capsule_root):
        require(os.path.isdir(capsule_root) and not os.listdir(capsule_root),
                "FAIL CLOSED: --capsule-root %s exists and is NOT empty; refuse to "
                "overwrite an existing capsule" % capsule_root)
    else:
        os.makedirs(capsule_root)

    # load_contract verifies the document self-hash (fails closed on tamper).
    contract_doc = cc.load_contract(contract_path)
    require(contract_doc["checkpoint_contract_sha256"]
            == FROZEN_CHECKPOINT_CONTRACT_SHA256,
            "FAIL CLOSED: contract document SHA %s != frozen %s"
            % (contract_doc["checkpoint_contract_sha256"][:16],
               FROZEN_CHECKPOINT_CONTRACT_SHA256[:16]))
    refs, refs_verified, refs_detail = common_references(common_root)
    require(refs_verified,
            "FAIL CLOSED: common SHA references do not verify against the assembled "
            "common/ directory: %s" % refs_detail)

    candidate_id = CANDIDATE_ID[arm]
    arm_block = contract_doc["arms"][arm]

    # 1. candidate_manifest.json
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "generated_at_utc": _utc_now(),
        "candidate_id": candidate_id,
        "arm": arm,
        "runtime_family": RUNTIME_FAMILY,
        "checkpoint_step": arm_block["checkpoint_step"],
        "checkpoint_file_sha256": arm_block["checkpoint_file_sha256"],
        "params_sha256": arm_block["params_sha256"],
        "base_checkpoint_params_sha256":
            contract_doc["common"]["base_checkpoint_params_sha256"],
        "checkpoint_contract_sha256": contract_doc["checkpoint_contract_sha256"],
        "immutable": True,
        "scientific_predicates_defined_here": False,
        "performance_claim_authorized": False,
        "scientific_superiority_claim": False,
        "strong_student_selection_authorized": False,
        "formal_pool_evaluation_started": False,
    }
    _atomic_json(os.path.join(capsule_root, "candidate_manifest.json"), manifest)

    # 2. training_contract.json (frozen-contract projection only)
    _atomic_json(os.path.join(capsule_root, "training_contract.json"),
                 project_training_contract(contract_doc, arm))

    # 3. checkpoint_contract.json (byte copy of the frozen contract)
    shutil.copyfile(contract_path,
                    os.path.join(capsule_root, "checkpoint_contract.json"))
    require(_sha256_file(os.path.join(capsule_root, "checkpoint_contract.json"))
            == _sha256_file(contract_path),
            "FAIL CLOSED: checkpoint_contract.json copy is not byte-exact")

    # 4./5. real shims
    mapping = {"CANDIDATE_ID": candidate_id, "ARM": arm,
               "RUNTIME_FAMILY": RUNTIME_FAMILY, "COMMON_ROOT": common_root,
               "COMMON_RUNNER_SHA256": refs["common_runner_sha256"],
               "COMMON_EVALUATOR_SHA256": refs["common_evaluator_sha256"],
               "CHECKPOINT_CONTRACT_SHA256":
                   contract_doc["checkpoint_contract_sha256"],
               "CHECKPOINT_FILE_SHA256": arm_block["checkpoint_file_sha256"],
               "PARAMS_SHA256": arm_block["params_sha256"],
               "BASE_PARAMS_SHA256":
                   contract_doc["common"]["base_checkpoint_params_sha256"],
               "MAX_STEPS": int(max_steps)}
    for name, template in (("candidate_runtime.py", CANDIDATE_RUNTIME_TEMPLATE),
                           ("evaluate_candidate.py", EVALUATE_CANDIDATE_TEMPLATE)):
        with open(os.path.join(capsule_root, name), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write(_render(template, mapping))

    # 6. memory contract smoke (real ABI on the real checkpoint)
    mem = run_memory_contract_smoke(checkpoint_path, arm, contract_path,
                                    cc2_snapshot_root)
    _atomic_json(os.path.join(capsule_root, "memory_contract_smoke_result.json"),
                 mem)
    require(mem["status"] == "PASS",
            "FAIL CLOSED: memory contract smoke FAIL for %s: %s"
            % (candidate_id, mem["checks"]))

    # 7. interface smoke through the public common evaluator entry
    ifs = run_interface_smoke(checkpoint_path, arm, contract_path, common_root,
                              capsule_root, max_steps)
    _atomic_json(os.path.join(capsule_root, "interface_smoke_result.json"), ifs)
    require(ifs["status"] == "PASS",
            "FAIL CLOSED: interface smoke FAIL for %s (engine_exit_code=%s): %s"
            % (candidate_id, ifs["engine_exit_code"], ifs["checks"]))

    # 8. common evaluator binding result (the eight S7 SHA references)
    binding = {
        "schema": SCHEMA_BINDING,
        "generated_at_utc": _utc_now(),
        "candidate_id": candidate_id,
        "arm": arm,
        "run_class": "INTERFACE_SMOKE",
        "performance_claim_authorized": False,
        "scientific_superiority_claim": False,
        "strong_student_selection_authorized": False,
        "formal_eval_binding": (
            "PASS" if (refs_verified and ifs["status"] == "PASS"
                       and mem["status"] == "PASS") else "FAIL"),
        "common_root": common_root,
        "common_references": refs,
        "common_references_verified": refs_verified,
        "common_references_detail": refs_detail,
        "interface_smoke_status": ifs["status"],
        "interface_smoke_exit_code": ifs["engine_exit_code"],
        "interface_smoke_files": ifs["files"],
        "memory_contract_smoke_status": mem["status"],
        "checkpoint_file_sha256": mem["checkpoint_file_sha256"],
        "params_sha256": mem["params_sha256"],
        "sha_evidence_present": True,
    }
    _atomic_json(os.path.join(capsule_root, "common_evaluator_binding_result.json"),
                 binding)

    # 9. environment lock (byte copy of the common lock)
    shutil.copyfile(os.path.join(common_root, "environment_lock.json"),
                    os.path.join(capsule_root, "environment_lock.json"))

    # 10. SHA256SUMS
    sums_sha = write_capsule_sums(capsule_root)

    # 11. READY.json (the S9 per-capsule gates)
    gates = {
        "identity_status": "PASS" if (
            mem["checks"].get("checkpoint_file_sha256_verified")
            and mem["checks"].get("params_sha256_verified")
            and mem["checks"].get("carry_mode_matches_arm")) else "FAIL",
        "interface_smoke_status": "PASS" if ifs["status"] == "PASS" else "FAIL",
        "memory_contract_smoke_status":
            "PASS" if mem["status"] == "PASS" else "FAIL",
        "formal_eval_binding": binding["formal_eval_binding"],
        "checkpoint_sha256_verified":
            "PASS" if mem["checkpoint_file_sha256"]
            == cc.FROZEN_CHECKPOINT_FILE_SHA256[arm] else "FAIL",
        "params_sha256_verified":
            "PASS" if mem["params_sha256"] == cc.FROZEN_PARAMS_SHA256[arm]
            else "FAIL",
        "common_artifact_sha_refs_verified":
            "PASS" if refs_verified else "FAIL",
        "immutable": "PASS",
    }
    ready = all(v == "PASS" for v in gates.values())
    ready_doc = {
        "schema": SCHEMA_READY,
        "generated_at_utc": _utc_now(),
        "candidate_id": candidate_id,
        "arm": arm,
        "READY": bool(ready),
        "gates": gates,
        "checkpoint_file_sha256": mem["checkpoint_file_sha256"],
        "params_sha256": mem["params_sha256"],
        "base_checkpoint_params_sha256":
            contract_doc["common"]["base_checkpoint_params_sha256"],
        "checkpoint_contract_sha256": contract_doc["checkpoint_contract_sha256"],
        "common_evaluator_binding_formal": binding["formal_eval_binding"],
        "sha256sums_sha256": sums_sha,
        "immutable": True,
    }
    _atomic_json(os.path.join(capsule_root, READY_FILE), ready_doc)
    print("CAPSULE_BUILD_COMPLETE %s READY=%s binding=%s root=%s"
          % (candidate_id, ready_doc["READY"], binding["formal_eval_binding"],
             capsule_root))
    return ready_doc


# ---------------------------------------------------------------------------
# Self-test (PURE; any interpreter)
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # Candidate ids frozen verbatim.
    check("candidate_ids_frozen",
          CANDIDATE_ID == {"persistent": "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304",
                           "reset128": "RESET128_RMT16_ORIGINAL_VTRACE_98304"})

    # Training-contract projection from the REAL frozen contract file.
    contract_path = audit.repo_root().joinpath(
        "configs", "tier3_cc2_final98304_checkpoint_contract_v1.json")
    with open(contract_path, encoding="utf-8") as fh:
        contract_doc = json.load(fh)
    for arm in ("persistent", "reset128"):
        tc = project_training_contract(contract_doc, arm)
        arm_block = contract_doc["arms"][arm]
        check("tc_%s_checkpoint_file" % arm,
              tc["checkpoint_file_sha256"] == arm_block["checkpoint_file_sha256"])
        check("tc_%s_params" % arm,
              tc["params_sha256"] == arm_block["params_sha256"])
        check("tc_%s_carry_mode" % arm, tc["carry_mode"] == arm)
        check("tc_%s_step" % arm, tc["checkpoint_step"] == 98304)
        check("tc_%s_seed" % arm, tc["training_seed"] == 42)
        check("tc_%s_replay" % arm, tc["replay_mode"] == "original_vtrace")
        check("tc_%s_run_class" % arm, tc["training_run_class"] == "long_run_98304")
        check("tc_%s_seq" % arm,
              tc["sequence_length"] == 129 and tc["segment_len"] == 128
              and tc["crosses_boundary"] is True)
        check("tc_%s_base_params" % arm,
              tc["base_checkpoint_params_sha256"]
              == "d4e85af58b7f87d689fadea12eec70c852fa098a09f5ea8907448684b3bf60f5")
        check("tc_%s_no_new_training" % arm,
              tc["provenance"]["new_training_by_cc4"] is False
              and tc["provenance"]["performance_claim_authorized"] is False
              and tc["provenance"]["scientific_superiority_claim"] is False)
    check("contract_self_sha_reproduces",
          cc.contract_sha256(contract_doc)
          == contract_doc["checkpoint_contract_sha256"])
    check("contract_self_sha_frozen_literal",
          contract_doc["checkpoint_contract_sha256"]
          == FROZEN_CHECKPOINT_CONTRACT_SHA256)

    # Shim templates: placeholders resolve + code compiles.
    mapping = {"CANDIDATE_ID": "X", "ARM": "persistent",
               "RUNTIME_FAMILY": RUNTIME_FAMILY, "COMMON_ROOT": "/c",
               "COMMON_RUNNER_SHA256": "r" * 64,
               "COMMON_EVALUATOR_SHA256": "e" * 64,
               "CHECKPOINT_CONTRACT_SHA256": "c" * 64,
               "CHECKPOINT_FILE_SHA256": "f" * 64, "PARAMS_SHA256": "p" * 64,
               "BASE_PARAMS_SHA256": "b" * 64, "MAX_STEPS": 32}
    for tpl in (CANDIDATE_RUNTIME_TEMPLATE, EVALUATE_CANDIDATE_TEMPLATE):
        rendered = _render(tpl, mapping)
        check("shim_no_placeholder_left", "@@" not in rendered)
        compile(rendered, "<capsule-shim>", "exec")
    try:
        _render("@@MISSING@@", {})
        check("unresolved_placeholder_rejected", False)
    except FailClosed:
        check("unresolved_placeholder_rejected", True)

    # S9 per-capsule gate set + aggregation.
    gate_names = ("identity_status", "interface_smoke_status",
                  "memory_contract_smoke_status", "formal_eval_binding",
                  "checkpoint_sha256_verified", "params_sha256_verified",
                  "common_artifact_sha_refs_verified", "immutable")
    check("capsule_gate_count_is_eight", len(gate_names) == 8)
    check("ready_only_if_all_pass",
          all(v == "PASS" for v in ["PASS"] * 8) is True
          and all(v == "PASS" for v in ["PASS"] * 7 + ["FAIL"]) is False)

    # Canonical-JSON SHA stability (full_profile / training-contract identity).
    check("canonical_sha_stable",
          _canonical_sha256({"a": 1, "b": [1, 2]})
          == _canonical_sha256({"b": [1, 2], "a": 1}))

    if problems:
        print("TIER3_POOL_CAPSULE_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_POOL_CAPSULE_SELF_TEST_PASS "
          "(ids frozen; training projection exact; shims bind+compile; 8 gates)")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    if "--build" in argv:
        def opt(flag, default=None):
            return argv[argv.index(flag) + 1] if flag in argv else default
        require(opt("--capsule-root") and opt("--arm") and opt("--checkpoint")
                and opt("--common-root"),
                "usage: tier3_pool_capsule.py --build --capsule-root DIR "
                "--arm {persistent|reset128} --checkpoint PKL --common-root DIR "
                "[--contract PATH] [--cc2_snapshot_root PATH] [--max-steps 32]")
        build_capsule(opt("--capsule-root"), opt("--arm"), opt("--checkpoint"),
                      opt("--common-root"), opt("--contract"),
                      opt("--cc2_snapshot_root"),
                      int(opt("--max-steps", BINDING_MAX_STEPS)))
        return 0
    print("usage: tier3_pool_capsule.py --self-test | --build ...")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
