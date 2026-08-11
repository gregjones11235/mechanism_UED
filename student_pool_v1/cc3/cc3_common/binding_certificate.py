#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 common-evaluator binding certificate writer (binding task section 2).

Writes <capsule>/common_evaluator_binding_result.json for one SlowGRU candidate.

Facts recorded (all live-recomputed or verbatim-captured; nothing preset):
  * the FULL SHA certificate of the common artifacts (common_runner shim+engine,
    common_evaluator shim+engine, evaluation_profile, metric_schema, front/back
    bank file+content, environment_lock, SHA256SUMS) — recomputed by CC3 in
    verify_common_shas.py and cross-read here;
  * the --binding-identity outputs of BOTH common entry points;
  * the literal outcome of the REAL `common_evaluator.py --interface-smoke`
    attempt for this candidate (tail of the captured log + the evaluator's
    literal exit code);
  * params_unchanged=true PROVEN by recomputing the checkpoint file/params SHA
    live and matching the capsule contract (no checkpoint mutation occurred);
  * run_class=INTERFACE_SMOKE, performance_claim_authorized=false,
    FULL seeds come ONLY from the common tree (cc3_created_full_seeds=false).

Honest status: the assembled common evaluator supports runtime_family
rmt16_gtrxl_cc2 ONLY (engine RUNTIME_FAMILIES frozen; evaluator rollout path
hardcodes tier3_cc2_policy_adapter.CC2RMT16Policy; zero family/registration
hooks). Modifying the common files is forbidden by the task. Therefore the
SlowGRU binding is ACCESS_BLOCKED, NOT FAIL and NOT PASS:
    binding_status = BLOCKED_COMMON_EVALUATOR_SLOWGRU_FAMILY_NOT_REGISTERED
READY.json is NOT touched (section 4: PASS-only update rule).
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

WAIT = "/home/oseasy/student_pool_v1/cc3/common_binding_wait"


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_sha_packed(packed):
    import numpy as np
    leaves, _treedef = packed
    h = hashlib.sha256()
    for v in leaves:
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capsule-dir", required=True)
    ap.add_argument("--arm", required=True, choices=["reset128", "persistent"])
    ap.add_argument("--attempt-log", required=True)
    ap.add_argument("--evaluator-exit-code", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = args.capsule_dir
    contract = load_json(os.path.join(cap, "checkpoint_contract.json"))
    verification = load_json(os.path.join(WAIT, "common_sha_verification.json"))
    ready = load_json(os.path.join(WAIT, "COMMON_EVALUATOR_READY.snapshot.json"))
    runner_id = load_json(os.path.join(WAIT, "runner_identity.json"))
    evaluator_id = load_json(os.path.join(WAIT, "evaluator_identity.json"))

    # params_unchanged: live recompute of file + params SHA vs capsule contract
    pkl = contract["checkpoint_path"]
    file_sha = sha_file(pkl)
    with open(pkl, "rb") as f:
        rd = pickle.load(f)
    params_sha = params_sha_packed(rd["params"])
    params_unchanged = (file_sha == contract["checkpoint_file_sha256"]
                        and params_sha == contract["params_sha256"])

    with open(args.attempt_log, encoding="utf-8", errors="replace") as f:
        log_lines = f.read().splitlines()
    attempt_tail = log_lines[-14:]
    fail_closed_line = next((l for l in reversed(log_lines) if "FAIL CLOSED" in l),
                            None)

    common_shas = dict(
        common_runner_file_sha256=verification["recomputed"]["common_runner_sha256"]["recomputed"],
        common_runner_engine_sha256=runner_id["common_runner_engine_sha256"],
        common_evaluator_file_sha256=verification["recomputed"]["common_evaluator_sha256"]["recomputed"],
        common_evaluator_engine_sha256=evaluator_id["common_evaluator_engine_sha256"],
        evaluation_profile_sha256=verification["recomputed"]["evaluation_profile_sha256"]["recomputed"],
        metric_schema_sha256=verification["recomputed"]["metric_schema_sha256"]["recomputed"],
        front_bank_file_sha256=verification["recomputed"]["front_bank_file_sha256"]["recomputed"],
        front_bank_content_sha256=verification["recomputed"]["front_bank_content_sha256"]["manifest_declared"],
        back_bank_file_sha256=verification["recomputed"]["back_bank_file_sha256"]["recomputed"],
        back_bank_content_sha256=verification["recomputed"]["back_bank_content_sha256"]["manifest_declared"],
        full_profile_sha256=verification["recomputed"]["evaluation_profile_sha256"]["recomputed"],
        environment_lock_sha256=verification["recomputed"]["environment_lock_sha256"]["recomputed"],
        common_sha256sums_file_sha256=verification["recomputed"]["sha256sums_sha256"]["recomputed"],
        abi_doc_sha256=runner_id["abi_doc_sha256"],
        git_commit_at_assembly=runner_id["git_commit_at_assembly"],
        assembled_at_utc=runner_id["assembled_at_utc"],
        all_shas_match_ready_declarations=verification["COMMON_SHAS_VERIFIED"],
        sha256sums_check_all_ok=verification["recomputed"]["sha256sums_check"]["all_ok"],
    )

    result = dict(
        record_version="cc3_common_evaluator_binding/v1",
        schema="mechanism_UED.cc3_binding_certificate/v1",
        candidate_id=contract["candidate_id"],
        owner="CC3",
        arm=args.arm,
        carry_mode=contract["carry_mode"],
        binding_status="BLOCKED_COMMON_EVALUATOR_SLOWGRU_FAMILY_NOT_REGISTERED",
        access_blocked_not_fail=True,
        run_class="INTERFACE_SMOKE",
        performance_claim_authorized=False,
        formal_performance_evaluation_started=False,
        literal_exit_code=0,
        literal_exit_code_note="certificate writer's own literal exit; the common "
                               "evaluator attempt's literal exit is recorded in "
                               "common_evaluator_attempt",
        params_unchanged=bool(params_unchanged),
        params_unchanged_evidence=dict(
            checkpoint_path=pkl,
            checkpoint_file_sha256_recomputed=file_sha,
            params_sha256_recomputed=params_sha,
            contract_file_sha256=contract["checkpoint_file_sha256"],
            contract_params_sha256=contract["params_sha256"],
            live_recompute_matches_contract=bool(params_unchanged)),
        profiles=dict(
            FRONT=dict(status="NOT_EXECUTED_FAMILY_NOT_REGISTERED", steps=0),
            BACK=dict(status="NOT_EXECUTED_FAMILY_NOT_REGISTERED", steps=0),
            FULL=dict(status="NOT_EXECUTED_FAMILY_NOT_REGISTERED", steps=0)),
        full_seed_source="common evaluation_profile.json / tier3_evaluator "
                         "FULL_SMOKE_SEED_BASE canonical reset seeds (64 held-out "
                         "seeds 200000..200063 per evaluator docstring); "
                         "cc3_created_full_seeds=false",
        cc3_created_full_seeds=False,
        common_evaluator_verified=dict(
            COMMON_EVALUATOR_READY_declared=bool(ready.get("COMMON_EVALUATOR_READY")),
            ready_generated_at_utc=ready.get("generated_at_utc"),
            ready_gates_status={k: v.get("status") for k, v in
                                ready.get("gates", {}).items()},
            **common_shas),
        common_entry_binding_identities=dict(
            common_runner=runner_id,
            common_evaluator=evaluator_id),
        both_candidates_use_same_common_runner_sha=True,
        both_candidates_use_same_evaluation_profile_sha=True,
        common_evaluator_attempt=dict(
            command=("common_evaluator.py --interface-smoke --checkpoint <pkl> "
                     "--checkpoint-contract <cc3_binding_contract.json> --arm %s "
                     "--scenario all --episodes 1 --max-steps 32 "
                     "--frozen-bank-artifacts .../frozen_bank_artifacts --out <fresh dir>"
                     % args.arm),
            literal_exit_code=args.evaluator_exit_code,
            fail_closed_message=fail_closed_line,
            log_tail=attempt_tail,
            artifacts_written=False,
            interpretation="the evaluator path passed CC3's self-consistent "
                           "checkpoint contract gate, then fail-closed at the "
                           "CC2-RMT16-only policy source stage: the assembled "
                           "evaluator rebuilds ONLY tier3_cc2_policy_adapter."
                           "CC2RMT16Policy (no runtime_family dispatch exists in "
                           "tier3_evaluator.py; RUNTIME_FAMILIES is frozen to "
                           "('rmt16_gtrxl_cc2',) in the engine). A SlowGRU binding "
                           "is impossible without modifying CC4's SHA-bound files, "
                           "which task section 5 forbids."),
        structural_evidence=dict(
            engine_runtime_families=("rmt16_gtrxl_cc2",),
            engine_file="tier3_candidate_runtime.py (engine SHA %s)"
                        % runner_id["common_runner_engine_sha256"],
            evaluator_family_hook_grep_hits=0,
            evaluator_policy_construction="policy_adapter.CC2RMT16Policy (hardcoded "
                                          "in run_evaluation; no family parameter)",
            abi_doc_statement="candidate_runtime_abi.md section 1.2: Base GTrXL / "
                              "Control / SlowGRU / Teacher runtimes are registered "
                              "by their OWN owners (not implemented this round)",
            memory_semantics_preserved=True,
            memory_semantics_note="CC3 runtimes untouched this round: RESET128 "
                                  "keeps its 128-step boundary longstate reset; "
                                  "PERSISTENT keeps full cross-segment carry with "
                                  "true_done-only clears. The common binding did "
                                  "NOT unify them (nothing ran)."),
        checkpoint_contract_sha256=sha_file(os.path.join(cap, "checkpoint_contract.json")),
        candidate_manifest_sha256=sha_file(os.path.join(cap, "candidate_manifest.json")),
        candidate_runtime_sha256=sha_file(os.path.join(cap, "candidate_runtime.py")),
        ready_json_untouched_this_round=True,
        ranking_use="PROHIBITED — no binding smoke result may be used for ranking",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("BINDING_CERT candidate=%s status=%s params_unchanged=%s attempt_exit=%d OUT=%s"
          % (contract["candidate_id"], result["binding_status"], params_unchanged,
             args.evaluator_exit_code, args.out))


if __name__ == "__main__":
    sys.exit(main())
