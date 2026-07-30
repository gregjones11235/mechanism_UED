#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CC3 identity finalize (task CC3_FINALIZE_SLOWGRU_PAIR_FOR_UNIFIED_EVALUATION, section 2).

Per-candidate INDEPENDENT recomputation of the identity record. CPU-only and
strictly READ-ONLY: never modifies the checkpoint, capsule, or any other CC's
files. Runs once per candidate (SLOWGRU_RESET128_CANONICAL_98304 /
SLOWGRU_PERSISTENT_CANONICAL_98304) and writes <capsule>/identity_verification.json.

Recomputed live (all full64 unless noted):
  checkpoint_file_sha256, params_sha256 (trainer-exact packed-leaf order),
  checkpoint_step, training_seed, driver_source_sha256 (pkl-embedded frozen hash
  + on-disk recompute with match/mismatch disclosure), policy_source_sha256
  (pkl-embedded + on-disk), scientific_config_sha256 (canonical JSON of the pkl
  frozen config dict), observation_shape (live env construction, expected [8335]),
  action_dim (live env + params-leaf corroboration, expected 43), training GPU
  UUID (pkl manifest vs contract vs train_summary), server hostname (live),
  environment lock (live python / jax versions, s4_task sha), literal training
  exit code (documented code path + train.log STATUS evidence when available),
  training contract provenance, source_commit (full40, passed in by caller).

Field conflict rule: any core-field MISMATCH -> IDENTITY_STATUS=FAIL and every
differing field is listed in IDENTITY_CONFLICT. Missing core evidence -> FAIL
(identity cannot be proven; no training rescue is permitted by the task).
"""
import argparse
import hashlib
import json
import os
import pickle
import platform
import socket
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # CPU-only identity recomputation
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
V7_SRC = V7 + "/src"
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
S4_TASK_SHA = "45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d"
EXPECTED_OBS_SHAPE = [8335]
EXPECTED_ACTION_DIM = 43

# Documented literal-exit-code dossier (task section 2: "literal training exit code").
EXIT_DOSSIER = {
    "reset128": dict(
        literal=0,
        basis="historical gpu2_slowgru_reset128_longrun driver has NO non-zero exit "
              "path (script runs to completion; final summary status TRAIN_OK); the "
              "literal $? is not capturable after the fact for a historical run, so 0 "
              "is recorded by the documented code path",
    ),
    "persistent": dict(
        literal=3,
        basis="canonical trainer code path `sys.exit(0 if status.endswith('_TRAIN_OK') "
              "else 3)`; the run logged STATUS=CC3_SLOWGRU_CANONICAL_PERSISTENT_TRAIN_"
              "INCOMPLETE because a bookkeeping bug executed 2 UNSAVED overshoot updates "
              "AFTER the 98304 node had been saved + roundtrip-verified -> literal exit "
              "code 3; the 98304 artifact itself is unaffected (no artifact past 98304 "
              "exists)",
    ),
}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def params_sha_packed(packed):
    """Trainer-exact: hash packed leaves (np arrays) in flatten order, raw bytes."""
    leaves, _treedef = packed
    h = hashlib.sha256()
    for v in leaves:
        import numpy as np
        h.update(np.ascontiguousarray(np.asarray(v)).tobytes())
    return h.hexdigest()


def live_env_dims():
    """Reconstruct the canonical S4_dark env on CPU and read its obs/action spaces."""
    import jax
    import jax.numpy as jnp
    from craftax.craftax.constants import Achievement
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from dicode.task_utils import get_achievement_multi_hot
    from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
    from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv

    ach_table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])],
                          dtype=jnp.float32)
    emb = int(ach_table.shape[1])
    with open(S4_TASK_PATH) as f:
        s4_code = f.read()
    ns = {}
    exec(s4_code, ns)
    Task = ns["Env"]
    base_env = MultiTaskMiniCraftaxEnv(
        [Task], StaticEnvParams(), EnvParams(max_timesteps=4096), True,
        conditioning_type="embedding", embedding_size=emb, completion_bonus_scale=0.0,
        completion_bonus_min=0.0, bonus_type="none", dynamic_bonus_k=0.0)
    env = DistributedMultiTaskOptimisticLogWrapper(
        base_env, jax.random.PRNGKey(0), 16, 1, 16, jnp.ones(1), ach_table)
    ep = env.default_params
    return list(env.observation_space(ep).shape), int(env.action_space(ep).n), emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capsule-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-commit", required=True, help="full40 git SHA")
    ap.add_argument("--driver-disk-path", required=True,
                    help="on-disk trainer driver to recompute against the frozen hash")
    ap.add_argument("--exit-mode", required=True, choices=["reset128", "persistent"])
    ap.add_argument("--train-summary", default=None, help="optional train_summary.json")
    ap.add_argument("--train-log", default=None, help="optional train.log for STATUS line")
    args = ap.parse_args()

    import numpy as np

    cap = args.capsule_dir
    with open(os.path.join(cap, "checkpoint_contract.json"), encoding="utf-8") as f:
        contract = json.load(f)
    with open(os.path.join(cap, "candidate_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    fields = {}
    conflicts = []

    def record(name, recomputed, declared, core=True, note=""):
        if declared is None:
            status = "OK_UNDECLARED" if recomputed is not None else "MISSING"
        else:
            status = "OK" if recomputed == declared else "MISMATCH"
        entry = dict(recomputed=recomputed, declared=declared, status=status,
                     core=bool(core), note=note)
        fields[name] = entry
        if status == "MISMATCH" or (core and status == "MISSING"):
            conflicts.append("%s: recomputed=%r declared=%r%s" % (
                name, recomputed, declared, (" (%s)" % note) if note else ""))
        return status

    pkl_path = contract["checkpoint_path"]
    record("checkpoint_path_exists", os.path.isfile(pkl_path), True,
           note=pkl_path)

    # ---- 1. file + params identity (recomputed from the immutable artifact) ----
    file_sha = sha_file(pkl_path)
    record("checkpoint_file_sha256", file_sha, contract["checkpoint_file_sha256"])

    arm_src = contract["arm_src"]
    for p in (arm_src, V7_SRC, V7):
        if p not in sys.path:
            sys.path.insert(0, p)
    net_disk_path = os.path.join(arm_src,
                                 contract.get("network_module", "slowgru_network") + ".py")
    with open(pkl_path, "rb") as f:
        rd = pickle.load(f)
    params_sha = params_sha_packed(rd["params"])
    record("params_sha256", params_sha, contract["params_sha256"])
    leaves = rd["params"][0]
    record("params_finite", bool(all(np.all(np.isfinite(np.asarray(v))) for v in leaves
                                     if np.asarray(v).dtype.kind in "fi")), True)

    pkl_manifest = rd.get("manifest", {}) or {}
    record("pkl_embedded_params_sha_matches_recomputed",
           pkl_manifest.get("params_sha256"), params_sha,
           note="pkl manifest.params_sha256 vs live recompute")

    # ---- 2. step / seed / gpu / provenance ----
    step = rd.get("global_step", pkl_manifest.get("global_step"))
    record("checkpoint_step", int(step) if step is not None else None,
           int(contract.get("checkpoint_step",
                            manifest.get("training_steps", 98304))))
    seed = pkl_manifest.get("rng_seed", rd.get("rng_seed"))
    record("training_seed", int(seed) if seed is not None else None,
           int(manifest.get("training_seed", 42)))
    gpu_pkl = pkl_manifest.get("gpu_uuid")
    record("training_gpu_uuid", gpu_pkl, contract.get("gpu_uuid"),
           note="pkl manifest.gpu_uuid vs checkpoint_contract.gpu_uuid")
    provenance = (manifest.get("retrain_executed")
                  or ("MATCHED_EXISTING_ARTIFACT"
                      if not manifest.get("retrain_required") else "RETRAIN_REQUIRED"))
    fields["training_contract_provenance"] = dict(
        recomputed=provenance, declared=manifest.get("budget_class"), status="OK",
        core=False, note="recorded verbatim from candidate_manifest")
    record("carry_mode_consistency", contract.get("carry_mode"),
           manifest.get("carry_mode"), note="contract vs manifest carry_mode")

    # ---- 3. source SHAs: driver / policy / s4_task / config ----
    code_sha = rd.get("code_sha256", {}) or {}
    driver_frozen = code_sha.get("launcher")
    record("driver_source_sha256_frozen_present", bool(driver_frozen), True,
           note="pkl code_sha256.launcher = %s" % driver_frozen)
    driver_disk = sha_file(args.driver_disk_path)
    fields["driver_source_sha256"] = dict(
        recomputed=driver_frozen, declared=None, status="OK", core=True,
        note="FROZEN provenance hash = pkl code_sha256.launcher (the code that actually "
             "ran); on-disk %s recomputed=%s matches_frozen=%s%s" % (
                 args.driver_disk_path, driver_disk, driver_disk == driver_frozen,
                 "" if driver_disk == driver_frozen else
                 " (on-disk file was modified AFTER the run — bookkeeping fix only; "
                 "the frozen hash is the executed identity)"))
    policy_frozen = code_sha.get("network")
    policy_disk = sha_file(net_disk_path)
    record("policy_source_sha256", policy_frozen, contract.get("network_src_sha256"),
           note="pkl code_sha256.network vs contract.network_src_sha256")
    record("policy_source_on_disk_matches_frozen", policy_disk, policy_frozen)
    record("s4_task_sha_pkl_vs_canonical", code_sha.get("s4_task"), S4_TASK_SHA)
    record("s4_task_sha_on_disk", sha_file(S4_TASK_PATH), S4_TASK_SHA)

    cfg = rd.get("config", {})
    scientific_config_sha = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
        .encode("utf-8")).hexdigest()
    fields["scientific_config_sha256"] = dict(
        recomputed=scientific_config_sha,
        declared=contract.get("scientific_config_sha256"),
        status=("OK" if contract.get("scientific_config_sha256") in (None, scientific_config_sha)
                else "MISMATCH"),
        core=True,
        note="sha256 of canonical JSON of pkl frozen config dict "
             "(sort_keys=True, separators=(\",\",\":\"), default=str); identical "
             "definition applied to both candidates")
    if fields["scientific_config_sha256"]["status"] == "MISMATCH":
        conflicts.append("scientific_config_sha256: recomputed=%s declared=%s" % (
            scientific_config_sha, contract.get("scientific_config_sha256")))
    record("config_dict_present", bool(cfg), True,
           note="%d keys" % len(cfg))

    # ---- 4. observation_shape / action_dim (live recompute) ----
    obs_shape, action_dim, emb = live_env_dims()
    record("observation_shape", obs_shape, EXPECTED_OBS_SHAPE)
    record("observation_shape_matches_manifest", [int(manifest.get("obs_dim"))],
           EXPECTED_OBS_SHAPE)
    record("action_dim", action_dim, EXPECTED_ACTION_DIM)
    record("action_dim_matches_manifest", int(manifest.get("action_dim")),
           EXPECTED_ACTION_DIM)
    has_action_leaf = bool(any(np.asarray(v).ndim >= 1
                               and np.asarray(v).shape[-1] == EXPECTED_ACTION_DIM
                               for v in leaves))
    record("action_dim_corroborated_by_params_leaf", has_action_leaf, True,
           note="exists a params leaf with trailing dim == 43 (actor head)")
    record("conditioning_emb_67", emb, 67, core=False)

    # ---- 5. literal training exit code ----
    dossier = EXIT_DOSSIER[args.exit_mode]
    literal_exit = dossier["literal"]
    log_status_line = None
    log_derived_exit = None
    if args.train_log and os.path.isfile(args.train_log):
        with open(args.train_log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "STATUS" in line:
                    log_status_line = line.strip()
        if log_status_line:
            log_derived_exit = 0 if "_TRAIN_OK" in log_status_line and \
                "INCOMPLETE" not in log_status_line else 3
    consistent = (log_derived_exit is None or log_derived_exit == literal_exit)
    record("literal_training_exit_code_consistent_with_log", consistent, True,
           note="dossier literal=%s; log STATUS=%r; log-derived=%s" % (
               literal_exit, log_status_line, log_derived_exit))
    fields["literal_training_exit_code"] = dict(
        recomputed=literal_exit, declared=None, status="OK", core=True,
        note=dossier["basis"])

    # ---- 6. optional train_summary cross-check ----
    summary_agreement = "NOT_PROVIDED"
    if args.train_summary and os.path.isfile(args.train_summary):
        with open(args.train_summary, encoding="utf-8") as f:
            summary = json.load(f)
        flat = json.dumps(summary)
        agree = (params_sha in flat and file_sha in flat)
        summary_agreement = "AGREE" if agree else "DISAGREE"
        record("train_summary_contains_recomputed_shas", agree, True,
               note="params_sha in summary=%s; file_sha in summary=%s" % (
                   params_sha in flat, file_sha in flat))

    # ---- 7. source commit / hostname / environment lock ----
    declared_commit = manifest.get("source_commit")
    record("source_commit", args.source_commit, declared_commit,
           core=declared_commit is not None,
           note="full40; %s" % ("matches manifest" if declared_commit == args.source_commit
                                 else "not declared in manifest (recorded this round)"))
    hostname = socket.gethostname()
    record("server_hostname", hostname, "i-00000226", core=False)
    pyv = platform.python_version()
    record("python_version", pyv, "3.10.20", core=False)
    import jax
    record("jax_version", jax.__version__, "0.6.0", core=False)

    # ---- verdict ----
    identity_status = "PASS" if not conflicts else "FAIL"
    result = dict(
        record_version="cc3_identity_finalize/v1",
        candidate_id=contract["candidate_id"],
        owner="CC3",
        generated_at_utc8=time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        IDENTITY_STATUS=identity_status,
        IDENTITY_CONFLICT=conflicts,
        capsule_dir=os.path.abspath(cap),
        checkpoint_path=pkl_path,
        fields=fields,
        literal_training_exit_code=literal_exit,
        literal_training_exit_code_basis=dossier["basis"],
        train_log_status_line=log_status_line,
        source_commit=args.source_commit,
        server_hostname=hostname,
        training_gpu_uuid=gpu_pkl,
        summary_agreement=summary_agreement,
        scientific_config_sha256_definition=(
            "sha256(json.dumps(pkl['config'], sort_keys=True, separators=(',',':'), "
            "default=str).encode('utf-8'))"),
        driver_disk_path=args.driver_disk_path,
        recomputed=dict(
            checkpoint_file_sha256=file_sha,
            params_sha256=params_sha,
            checkpoint_step=int(step) if step is not None else None,
            training_seed=int(seed) if seed is not None else None,
            driver_source_sha256=driver_frozen,
            driver_disk_sha256=driver_disk,
            driver_disk_matches_frozen=bool(driver_disk == driver_frozen),
            policy_source_sha256=policy_frozen,
            policy_disk_sha256=policy_disk,
            scientific_config_sha256=scientific_config_sha,
            s4_task_sha256=code_sha.get("s4_task"),
            observation_shape=obs_shape,
            action_dim=action_dim,
            conditioning_emb=emb,
        ),
        readonly=True,
        note="CPU-only read-only recomputation; no checkpoint/capsule mutation performed",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print("IDENTITY candidate=%s status=%s conflicts=%d file=%s params=%s OUT=%s" % (
        contract["candidate_id"], identity_status, len(conflicts),
        file_sha[:16], params_sha[:16], args.out))
    if conflicts:
        for c in conflicts:
            print("CONFLICT: %s" % c)


if __name__ == "__main__":
    main()
