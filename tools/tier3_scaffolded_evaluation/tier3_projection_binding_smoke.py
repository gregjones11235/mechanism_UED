#!/usr/bin/env python3
"""CC4 Tier3 — NON-RMT projection binding smoke DRIVER (contract
NON_RMT_RUNTIME_ABI_BINDING_CLOSURE §二–§六).

Runs ONE candidate's INTERFACE_SMOKE through the FROZEN CC4 engine (imported as
a library; LF-SHA-verified against assembly_manifest) with a CC4 projection
policy adapter over the owner's OWN SHA-bound runtime, then writes the v2
binding evidence:

  <out>/episode_records.jsonl
  <out>/projection_record.json
  <out>/common_evaluator_binding_result_v2.json   (contract §六 full field set)
  <out>/SHA256SUMS_V2                              (over the three evidence files)
  <out>/READY_V2.json                              (honest gates; sums-excluded)

Gate order (each fail closed, each BEFORE any binding claim):
  0. anti-pollution env gate + candidate registry membership + output freshness
  1. frozen engine + common/ identity: assembly_manifest byte SHA + 24 engine
     module LF-SHAs; common SHA256SUMS byte SHA + live 57/57 re-verify +
     unlisted-file scan; individual common artifact byte SHAs; evaluation
     profile invariants + the FROZEN episode counts 8/8/64 EXTRACTED from the
     profile (contract §六); READY marker cross-check
  2. GPU discipline: visible device UUIDs ⊆ CC4 allowlist (GPU2/GPU3), never
     GPU0/GPU1
  3. frozen bank artifact load (FRONT/BACK) with content-hash gate
  4. owner runtime load through the projection registry (owner fail-closed
     gates) + capsule file triple-SHA verification + CC4-side recompute of
     params_sha256 / checkpoint_file_sha256 PER THE OWNER PROTOCOL vs the
     owner-declared full64 (mismatch => fail closed, never faked)
  5. canonical env (8335 / 43) + policy adapter + (slowgru) boundary unit check
  6. smoke rollouts: FULL canonical-reset seeds 42+i, FRONT/BACK frozen bank
     state prefix; engine rollout_episode; canonical episode-record SHAs;
     engine evaluate() aggregates recorded AS SMOKE-ONLY
  7. NEG23 analog: params byte-unchanged via the OWNER hash protocol
  8. evidence writes + provenance

This driver performs NO formal performance evaluation, NO ranking, and makes NO
performance claim (run_class=INTERFACE_SMOKE, performance_claim_authorized=false).

Usage (server, locked CC4 venv, GPU2 or GPU3):
  CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
  python tier3_projection_binding_smoke.py \
      --candidate-id BASE_GTRXL_ORIGINAL_VTRACE_98304 \
      [--episodes 2] [--max-steps 32] \
      [--common-dir /home/oseasy/student_pool_v1/common] \
      [--frozen-bank-artifacts /home/oseasy/student_pool_v1/common/frozen_bank_artifacts] \
      [--out /home/oseasy/student_pool_v1/cc4/<ID>/projection_binding_v2]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import tier3_projection_runtime as proj        # noqa: E402

BINDING_SCHEMA = "mechanism_UED.common_evaluator_binding/v2"
PROJECTION_RECORD_SCHEMA = "mechanism_UED.projection_binding_record/v1"
READY_V2_SCHEMA = "mechanism_UED.projection_ready/v2"

# Files inside common/ that are deliberately NOT covered by SHA256SUMS (the
# READY marker is sums-excluded by construction — 57/57 must stay untouched).
COMMON_SUMS_EXCLUDED = {"COMMON_EVALUATOR_READY.json"}
SKIP_DIR_NAMES = {"__pycache__"}


def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stage 1 — frozen engine + common identity
# ---------------------------------------------------------------------------
def verify_engine_and_common(common_dir, tools_dir):
    ev = {}
    # 1a. assembly_manifest byte SHA + engine module LF-SHAs (all 24 entries)
    manifest_path = os.path.join(common_dir, "assembly_manifest.json")
    am_sha = proj.sha256_file(manifest_path)
    proj.require(am_sha == proj.FROZEN_ASSEMBLY_MANIFEST_SHA256,
                 "FAIL CLOSED (ASSEMBLY_MANIFEST_SHA_MISMATCH): live %s != frozen %s"
                 % (am_sha, proj.FROZEN_ASSEMBLY_MANIFEST_SHA256))
    manifest = proj.read_json(manifest_path)
    engine_shas = manifest.get("engine_module_sha256")
    proj.require(isinstance(engine_shas, dict) and engine_shas,
                 "FAIL CLOSED: assembly_manifest.engine_module_sha256 missing")
    engine_verified = {}
    for fname, want in sorted(engine_shas.items()):
        got = proj.lf_sha256_file(os.path.join(tools_dir, fname))
        proj.require(got == want,
                     "FAIL CLOSED (ENGINE_MODULE_LF_SHA_MISMATCH): %s live %s != "
                     "manifest %s" % (fname, got, want))
        engine_verified[fname] = got
    for fname, want in proj.FROZEN_ENGINE_LF_SHA256.items():
        proj.require(engine_verified.get(fname) == want,
                     "FAIL CLOSED: engine constant for %s not reproduced" % fname)
    abi_sha = manifest.get("abi_doc_sha256")
    proj.require(abi_sha == proj.FROZEN_ABI_DOC_SHA256,
                 "FAIL CLOSED (ABI_DOC_SHA_MISMATCH): manifest %r != frozen %s"
                 % (abi_sha, proj.FROZEN_ABI_DOC_SHA256))
    ev["assembly_manifest_sha256"] = am_sha
    ev["engine_modules_lf_sha_verified"] = len(engine_verified)
    ev["engine_modules"] = engine_verified

    # 1b. common SHA256SUMS byte SHA + live 57/57 + unlisted scan
    sums_path = os.path.join(common_dir, "SHA256SUMS")
    sums_sha = proj.sha256_file(sums_path)
    proj.require(sums_sha == proj.FROZEN_SHA256SUMS_SHA256,
                 "FAIL CLOSED (SHA256SUMS_SHA_MISMATCH): live %s != frozen %s"
                 % (sums_sha, proj.FROZEN_SHA256SUMS_SHA256))
    sums = proj.parse_sha256sums(sums_path)
    proj.require(len(sums) == proj.FROZEN_COMMON_SUMS_ENTRY_COUNT,
                 "FAIL CLOSED: SHA256SUMS lists %d entries, frozen %d"
                 % (len(sums), proj.FROZEN_COMMON_SUMS_ENTRY_COUNT))
    mismatches, missing = [], []
    for rel, want in sorted(sums.items()):
        p = os.path.join(common_dir, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        if proj.sha256_file(p) != want:
            mismatches.append(rel)
    proj.require(not missing,
                 "FAIL CLOSED (COMMON_SUMS_MISSING_FILES): %s" % missing)
    proj.require(not mismatches,
                 "FAIL CLOSED (COMMON_SUMS_MISMATCH): %s" % mismatches)
    # Unlisted scan: every regular file under common/ must be sums-listed or
    # explicitly sums-excluded (skip __pycache__).
    listed_abs = {os.path.normpath(os.path.join(common_dir, r)) for r in sums}
    unlisted = []
    for dp, dns, fns in os.walk(common_dir):
        dns[:] = [d for d in dns if d not in SKIP_DIR_NAMES]
        for fn in fns:
            full = os.path.normpath(os.path.join(dp, fn))
            if full in listed_abs:
                continue
            if os.path.basename(full) in COMMON_SUMS_EXCLUDED:
                continue
            unlisted.append(os.path.relpath(full, common_dir))
    proj.require(not unlisted,
                 "FAIL CLOSED (COMMON_UNLISTED_FILES): %s" % sorted(unlisted))
    ev["sha256sums_sha256"] = sums_sha
    ev["common_sha256sums_self_check"] = "PASS (%d/%d)" % (
        len(sums), proj.FROZEN_COMMON_SUMS_ENTRY_COUNT)
    ev["common_unlisted_files"] = []

    # 1c. Individual common artifact byte SHAs
    named = {
        "common_runner.py": proj.FROZEN_COMMON_RUNNER_SHA256,
        "common_evaluator.py": proj.FROZEN_COMMON_EVALUATOR_SHA256,
        "evaluation_profile.json": proj.FROZEN_EVALUATION_PROFILE_SHA256,
        "metric_schema.json": proj.FROZEN_METRIC_SCHEMA_SHA256,
        "environment_lock.json": proj.FROZEN_ENVIRONMENT_LOCK_SHA256,
    }
    for fn, want in named.items():
        got = proj.sha256_file(os.path.join(common_dir, fn))
        proj.require(got == want,
                     "FAIL CLOSED (COMMON_ARTIFACT_SHA_MISMATCH): %s live %s != "
                     "frozen %s" % (fn, got, want))
        ev[fn + "_sha256"] = got

    # 1d. evaluation_profile invariants + FROZEN episode counts EXTRACTED (contract §六)
    profile = proj.read_json(os.path.join(common_dir, "evaluation_profile.json"))
    inv = profile.get("common_evaluation_invariants", {})
    proj.require(inv.get("max_timesteps") == proj.FROZEN_MAX_TIMESTEPS,
                 "FAIL CLOSED: profile max_timesteps %r" % inv.get("max_timesteps"))
    proj.require(inv.get("action_mode") == proj.FROZEN_ACTION_MODE,
                 "FAIL CLOSED: profile action_mode %r" % inv.get("action_mode"))
    proj.require(tuple(inv.get("observation_shape") or ()) == proj.FROZEN_OBSERVATION_SHAPE,
                 "FAIL CLOSED: profile observation_shape %r" % inv.get("observation_shape"))
    proj.require(inv.get("action_dim") == proj.FROZEN_ACTION_DIM,
                 "FAIL CLOSED: profile action_dim %r" % inv.get("action_dim"))
    sc = profile.get("scenarios", {})
    front_n = sc.get("front_l2", {}).get("n")
    back_n = sc.get("back_l2", {}).get("n")
    full_n = (sc.get("full", {}).get("world_seed_set") or {}).get("count")
    proj.require(front_n == proj.FROZEN_FRONT_EPISODE_COUNT
                 and back_n == proj.FROZEN_BACK_EPISODE_COUNT
                 and full_n == proj.FROZEN_FULL_EPISODE_COUNT,
                 "FAIL CLOSED: profile episode counts %r/%r/%r != frozen 8/8/64"
                 % (front_n, back_n, full_n))
    proj.require(sc.get("front_l2", {}).get("bank_content_sha256")
                 == proj.FROZEN_FRONT_BANK_CONTENT_SHA256,
                 "FAIL CLOSED: profile front bank_content_sha256 drift")
    proj.require(sc.get("back_l2", {}).get("bank_content_sha256")
                 == proj.FROZEN_BACK_BANK_CONTENT_SHA256,
                 "FAIL CLOSED: profile back bank_content_sha256 drift")
    ev["profile_episode_counts"] = {
        "front_episode_count": front_n, "back_episode_count": back_n,
        "full_episode_count": full_n,
        "source": "evaluation_profile.json scenarios.{front_l2.n,back_l2.n,"
                  "full.world_seed_set.count}"}

    # 1e. READY marker cross-check (marker is sums-excluded; presence tolerant
    # for the pre-supplement field set, strict where fields exist)
    ready_path = os.path.join(common_dir, "COMMON_EVALUATOR_READY.json")
    if os.path.isfile(ready_path):
        ready = proj.read_json(ready_path)
        proj.require(ready.get("COMMON_EVALUATOR_READY") is True,
                     "FAIL CLOSED: COMMON_EVALUATOR_READY != true")
        for key, want in (("common_runner_sha256", proj.FROZEN_COMMON_RUNNER_SHA256),
                          ("common_evaluator_sha256", proj.FROZEN_COMMON_EVALUATOR_SHA256),
                          ("evaluation_profile_sha256", proj.FROZEN_EVALUATION_PROFILE_SHA256),
                          ("metric_schema_sha256", proj.FROZEN_METRIC_SCHEMA_SHA256),
                          ("environment_lock_sha256", proj.FROZEN_ENVIRONMENT_LOCK_SHA256),
                          ("front_bank_content_sha256", proj.FROZEN_FRONT_BANK_CONTENT_SHA256),
                          ("back_bank_content_sha256", proj.FROZEN_BACK_BANK_CONTENT_SHA256),
                          ("sha256sums_sha256", proj.FROZEN_SHA256SUMS_SHA256)):
            if key in ready:
                proj.require(ready[key] == want,
                             "FAIL CLOSED: READY marker %s %r != frozen %s"
                             % (key, ready[key], want))
        if "full_profile_sha256" in ready:
            proj.require(ready["full_profile_sha256"] == proj.FROZEN_FULL_PROFILE_SHA256,
                         "FAIL CLOSED: READY full_profile_sha256 drift")
        if "FULL_PROFILE_STATUS" in ready:
            proj.require(ready["FULL_PROFILE_STATUS"] == "FROZEN",
                         "FAIL CLOSED: READY FULL_PROFILE_STATUS != FROZEN")
        if "FORMAL_RANKING_AUTHORIZED" in ready:
            proj.require(ready["FORMAL_RANKING_AUTHORIZED"] is False,
                         "FAIL CLOSED: READY FORMAL_RANKING_AUTHORIZED must be false "
                         "before this closure round")
        ev["common_ready_marker_present"] = True
    else:
        ev["common_ready_marker_present"] = False
    return ev


# ---------------------------------------------------------------------------
# Stage 2 — GPU discipline
# ---------------------------------------------------------------------------
def verify_gpu_allowed():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    proj.require(cvd.strip(),
                 "FAIL CLOSED (GPU_DISCIPLINE): CUDA_VISIBLE_DEVICES must be set "
                 "explicitly (GPU2 or GPU3 UUID)")
    try:
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                              text=True, timeout=30)
    except Exception as exc:
        raise proj.FailClosed("FAIL CLOSED: nvidia-smi -L failed: %r" % exc)
    proj.require(proc.returncode == 0,
                 "FAIL CLOSED: nvidia-smi -L rc=%d: %s"
                 % (proc.returncode, (proc.stderr or "").strip()[:200]))
    by_index = {}
    for line in (proc.stdout or "").splitlines():
        # "GPU 0: NVIDIA ... (UUID: GPU-xxxx)"
        if "(UUID:" not in line:
            continue
        head, uuid_part = line.split("(UUID:", 1)
        uuid = uuid_part.strip().rstrip(")").strip()
        toks = head.strip().split()
        idx = toks[1].rstrip(":") if len(toks) > 1 else None
        if idx is not None and idx.isdigit():
            by_index[int(idx)] = uuid
    entries = [e.strip() for e in cvd.split(",") if e.strip()]
    visible = []
    for e in entries:
        if e.startswith("GPU-"):
            visible.append(e)
        elif e.isdigit():
            proj.require(int(e) in by_index,
                         "FAIL CLOSED: CUDA_VISIBLE_DEVICES index %s not found in "
                         "nvidia-smi -L" % e)
            visible.append(by_index[int(e)])
        else:
            raise proj.FailClosed(
                "FAIL CLOSED: unsupported CUDA_VISIBLE_DEVICES entry %r" % e)
    for u in visible:
        proj.require(u in proj.CC4_GPU_ALLOWED_UUIDS,
                     "FAIL CLOSED (GPU_DISCIPLINE): visible GPU %s is NOT in the "
                     "CC4 allowlist %s" % (u, list(proj.CC4_GPU_ALLOWED_UUIDS)))
        for pre in proj.CC4_GPU_BANNED_UUID_PREFIXES:
            proj.require(not u.startswith(pre),
                         "FAIL CLOSED (GPU_DISCIPLINE): visible GPU %s is BANNED "
                         "(prefix %s)" % (u, pre))
    return {"cuda_visible_devices": cvd, "visible_gpu_uuids": visible,
            "nvidia_smi_L": (proc.stdout or "").strip().splitlines(),
            "gpu_allowlist_cc4": list(proj.CC4_GPU_ALLOWED_UUIDS)}


# ---------------------------------------------------------------------------
# Stage 8 — evidence writers
# ---------------------------------------------------------------------------
def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--episodes", type=int, default=2,
                    help="smoke episodes per scenario [1, 8]; default 2")
    ap.add_argument("--max-steps", type=int, default=32,
                    help="smoke step cap [1, 4096]; default 32")
    ap.add_argument("--common-dir", default="/home/oseasy/student_pool_v1/common")
    ap.add_argument("--frozen-bank-artifacts",
                    default="/home/oseasy/student_pool_v1/common/frozen_bank_artifacts")
    ap.add_argument("--out", default=None,
                    help="output dir; default <pool>/cc4/<ID>/projection_binding_v2")
    args = ap.parse_args(argv)

    # --- Stage 0: anti-pollution + registry + freshness ---------------------
    hook = os.environ.get("RMT16_POSTJAX_BINDING_SELFTEST", "")
    proj.require(hook.strip() in ("", "0"),
                 "FAIL CLOSED (anti-pollution): RMT16_POSTJAX_BINDING_SELFTEST=%r "
                 "is set (false-success hook). Unset before any binding." % hook)
    spec = proj.get_spec(args.candidate_id)
    episodes = int(args.episodes)
    max_steps = int(args.max_steps)
    proj.require(1 <= episodes <= proj.FROZEN_BACK_EPISODE_COUNT,
                 "FAIL CLOSED: --episodes %d outside [1, %d]"
                 % (episodes, proj.FROZEN_BACK_EPISODE_COUNT))
    proj.require(1 <= max_steps <= proj.FROZEN_MAX_TIMESTEPS,
                 "FAIL CLOSED: --max-steps %d outside [1, %d]"
                 % (max_steps, proj.FROZEN_MAX_TIMESTEPS))
    out_dir = args.out
    if out_dir is None:
        pool_root = os.path.dirname(os.path.normpath(args.common_dir))
        out_dir = os.path.join(pool_root, "cc4", args.candidate_id,
                               "projection_binding_v2")
    tools_dir = HERE
    repo_root = os.path.dirname(os.path.dirname(tools_dir))

    print("[stage0] candidate=%s family=%s class=%s out=%s"
          % (args.candidate_id, spec["runtime_family"], spec["candidate_class"],
             out_dir), flush=True)

    # --- engine import (guarded) -------------------------------------------
    import tier3_evaluator as ev
    import tier3_state_bank_materializer as mat
    import tier3_state_serializer as ser
    proj.require(ser.have_jax_craftax(),
                 "FAIL CLOSED (BLOCKED_ENVIRONMENT): JAX+craftax required "
                 "(jax=%s, craftax=%s)" % (ser.have_jax(), ser.have_craftax()))
    ev.assert_output_dir_fresh(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- Stage 1: engine + common identity ----------------------------------
    print("[stage1] verifying frozen engine + common/ identity ...", flush=True)
    common_ev = verify_engine_and_common(args.common_dir, tools_dir)
    print("[stage1] %s; engine modules LF-SHA %d/%d; profile counts %s"
          % (common_ev["common_sha256sums_self_check"],
             common_ev["engine_modules_lf_sha_verified"],
             common_ev["engine_modules_lf_sha_verified"],
             common_ev["profile_episode_counts"]), flush=True)

    # --- Stage 2: GPU discipline --------------------------------------------
    gpu_ev = verify_gpu_allowed()
    print("[stage2] visible GPUs %s (allowlist enforced)"
          % gpu_ev["visible_gpu_uuids"], flush=True)

    # --- Stage 3: frozen bank artifacts --------------------------------------
    import jax
    import jax.numpy as jnp
    import tier3_frozen_bank_artifacts as art
    print("[stage3] loading frozen bank artifacts (read-only) ...", flush=True)
    bindings = {}
    bank_ev = {}
    for sc in (ev.FRONT, ev.BACK):
        b = art.load_bank(sc, args.frozen_bank_artifacts)
        want = (proj.FROZEN_FRONT_BANK_CONTENT_SHA256 if sc == ev.FRONT
                else proj.FROZEN_BACK_BANK_CONTENT_SHA256)
        proj.require(b.get("state_bank_hash") == want,
                     "FAIL CLOSED (FROZEN_BANK_CONTENT_MISMATCH): %s state_bank_hash "
                     "%r != frozen %s" % (sc, b.get("state_bank_hash"), want))
        bindings[sc] = b
        bank_ev[sc] = {"state_bank_hash": b.get("state_bank_hash"),
                       "bank_source": b.get("bank_source"),
                       "artifact_file_sha256": b.get("artifact_file_sha256"),
                       "loaded_content_sha256": b.get("loaded_content_sha256"),
                       "device_provenance": b.get("device_provenance"),
                       "n_states": len(b.get("states", []))}
        proj.require(bank_ev[sc]["n_states"] >= episodes,
                     "FAIL CLOSED: bank %s has %d states < episodes %d"
                     % (sc, bank_ev[sc]["n_states"], episodes))

    # --- Stage 4: capsule verification + owner runtime load ------------------
    print("[stage4] verifying capsule files + loading owner runtime via "
          "projection registry ...", flush=True)
    capsule_ev = proj.verify_capsule_files(spec)
    dicode_ev = proj.pin_dicode_resolution(repo_root)
    ctx = proj.load_owner_runtime(spec)

    params_before = proj.recompute_params_sha_owner(ctx)
    declared_params = spec["declared_params_sha256"]["value"]
    proj.require(params_before == declared_params,
                 "FAIL CLOSED (PARAMS_SHA_CC4_RECOMPUTE_MISMATCH): owner-protocol "
                 "recompute %s != owner-declared %s [%s]"
                 % (params_before, declared_params,
                    spec["declared_params_sha256"]["declaration_source"]))
    file_sha = proj.recompute_checkpoint_file_sha_owner(spec, ctx)
    declared_file = spec["declared_checkpoint_file_sha256"]["value"]
    proj.require(file_sha == declared_file,
                 "FAIL CLOSED (CHECKPOINT_FILE_SHA_CC4_RECOMPUTE_MISMATCH): "
                 "owner-protocol recompute %s != owner-declared %s [%s]"
                 % (file_sha, declared_file,
                    spec["declared_checkpoint_file_sha256"]["declaration_source"]))
    print("[stage4] params_sha256(owner protocol)==declared MATCH; "
          "checkpoint_file_sha256(owner protocol)==declared MATCH", flush=True)

    # --- Stage 5: canonical env + policy adapter + boundary check ------------
    print("[stage5] building canonical env + policy adapter ...", flush=True)
    entry = ev.make_canonical_env()
    proj.require(tuple(entry["observation_shape"]) == proj.FROZEN_OBSERVATION_SHAPE,
                 "FAIL CLOSED: observation shape %s != frozen %s"
                 % (entry["observation_shape"], proj.FROZEN_OBSERVATION_SHAPE))
    proj.require(int(entry["action_count"]) == proj.FROZEN_ACTION_DIM,
                 "FAIL CLOSED: action count %d != frozen %d"
                 % (entry["action_count"], proj.FROZEN_ACTION_DIM))
    policy = proj.build_policy(spec, ctx)
    boundary_ev = None
    if spec["loader_kind"] == "cc3_slowgru":
        boundary_ev = proj.slowgru_boundary_unit_check(ctx["module"],
                                                       spec["carry_mode"])
        print("[stage5] slowgru boundary unit check: carry_mode=%s info=%s"
              % (boundary_ev["carry_mode"], boundary_ev["boundary_info"]),
              flush=True)

    # --- Stage 6: smoke rollouts (engine library path) ------------------------
    print("[stage6] interface smoke: %d episodes/scenario, max_steps=%d ..."
          % (episodes, max_steps), flush=True)
    scenarios = [ev.FULL, ev.FRONT, ev.BACK]
    schedule = {}
    for sc in scenarios:
        seeds = ([ev.FULL_SMOKE_SEED_BASE + i for i in range(episodes)]
                 if sc == ev.FULL
                 else mat.fixed_seed_schedule(sc, mat.FROZEN_BANK_N,
                                              mat.FROZEN_SEED_BASE,
                                              mat.FROZEN_SEED_STRIDE)[:episodes])
        schedule[sc] = {
            "kind": ("canonical_reset_seeds_smoke" if sc == ev.FULL
                     else "frozen_bank_state_smoke"),
            "count": len(seeds), "seeds": [int(s) for s in seeds]}
    reset_fn = ev._jit_reset(entry)
    records_by_scenario, results_by_scenario = {}, {}
    for sc in scenarios:
        seeds = [int(s) for s in schedule[sc]["seeds"]]
        entry_ids = ev.state_entry_ids_for(sc, seeds)
        eps = []
        for i, seed in enumerate(seeds):
            policy.reset()
            if sc == ev.FULL:
                _obs0, start_state = reset_fn(jax.random.PRNGKey(int(seed)))
            else:
                start_state = jax.tree.map(jnp.asarray,
                                           bindings[sc]["states"][i])
            rec = ev.rollout_episode(entry, start_state, sc, policy,
                                     entry_ids[i], int(seed), max_steps=max_steps)
            rec["episode_record_sha256"] = proj.sha256_bytes(
                proj.canonical_json_bytes(rec))
            eps.append(rec)
            print("  [%s %d/%d %s seed=%d] steps=%d defeat=%s died=%s "
                  "transition=%s engaged=%s"
                  % (sc, i + 1, len(seeds), entry_ids[i], seed, rec["timesteps"],
                     rec["defeat_kobold"], rec["player_died"],
                     rec["front_floor_transition_reached"], rec["kobold_engaged"]),
                  flush=True)
        lines = [proj.canonical_json_bytes(r).decode("utf-8") for r in eps]
        records_by_scenario[sc] = {
            "seeds": seeds,
            "entry_ids": entry_ids,
            "episode_records": eps,
            "episode_records_sha256": proj.sha256_bytes(
                ("\n".join(lines) + "\n").encode("utf-8"))}
        results_by_scenario[sc] = ev.evaluate(sc, eps)

    # --- Stage 7: NEG23 analog (owner-protocol params unchanged) --------------
    params_after = proj.recompute_params_sha_owner(ctx)
    proj.require(params_after == params_before,
                 "FAIL CLOSED (PARAMS_CHANGED_BY_EVALUATION): before %s != after %s"
                 % (params_before, params_after))
    params_unchanged = True

    # --- provenance -----------------------------------------------------------
    try:
        device_identity = ev._eval_device_identity()
    except Exception as exc:                       # never mask the smoke result
        device_identity = {"error": repr(exc)}
    try:
        git_head = ev._git_commit_head()
    except Exception as exc:
        git_head = "unavailable: %r" % exc
    xla_flags = {k: os.environ.get(k) for k in sorted(os.environ)
                 if k.startswith(("XLA_", "JAX_"))}
    provenance = {
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python_executable": sys.executable,
        "generated_at_utc": utc_now_iso(),
        "git_commit_head": git_head,
        "device_identity": device_identity,
        "gpu": gpu_ev,
        "xla_jax_env": xla_flags,
        "projection_module_sha256": proj.sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_sha256": proj.sha256_file(os.path.abspath(__file__)),
        "projection_module_lf_sha256": proj.lf_sha256_file(
            os.path.join(tools_dir, "tier3_projection_runtime.py")),
        "driver_module_lf_sha256": proj.lf_sha256_file(os.path.abspath(__file__)),
    }

    # --- Stage 8: evidence writes ---------------------------------------------
    # 8a. episode_records.jsonl (all scenarios, canonical order)
    jsonl_lines = []
    for sc in scenarios:
        for r in records_by_scenario[sc]["episode_records"]:
            jsonl_lines.append(proj.canonical_json_bytes(r).decode("utf-8"))
    jsonl_path = os.path.join(out_dir, "episode_records.jsonl")
    with open(jsonl_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(jsonl_lines) + "\n")
    episode_records_jsonl_sha256 = proj.sha256_file(jsonl_path)

    # 8b. projection_record.json
    policy_state = None
    if isinstance(policy, proj.SlowGRUProjectionPolicy):
        policy_state = {"boundary_invocations": policy.boundary_invocations,
                        "boundary_info_log": policy.boundary_info_log,
                        "segment_boundary_steps": policy.segment_boundary_steps}
    projection_record = {
        "schema": PROJECTION_RECORD_SCHEMA,
        "candidate_id": args.candidate_id,
        "runtime_family": spec["runtime_family"],
        "owner": spec["owner"],
        "candidate_class": spec["candidate_class"],
        "run_class": "INTERFACE_SMOKE",
        "registration_authority": {
            "NON_RMT_RUNTIME_REGISTRATION_AUTHORITY": "CC4_CAN_REGISTER_PROJECTIONS",
            "qualification": "CONDITIONAL",
            "conditions": ["C1_FROZEN_COMMON_UNTOUCHED", "C2_ZERO_REIMPLEMENTATION",
                           "C3_PROJECTION_ADDENDUM_DOCUMENTED", "C4_HONEST_LABELS",
                           "C5_OWNER_ARTIFACTS_READONLY"],
            "audit_record": "reports/tier3_scaffolded_evaluation/"
                            "non_rmt_abi_binding_closure_20260731/"
                            "NON_RMT_RUNTIME_REGISTRATION_AUTHORITY_AUDIT.md",
            "owner_action_required": False,
        },
        "capsule_verification": capsule_ev,
        "source_capsule_root": spec["source_capsule_root"],
        "owner_runtime_load": {
            "kind": ctx["kind"],
            "checkpoint_path": ctx["checkpoint_path"],
            "wandb_stub": ctx.get("wandb_stub"),
            "frozen_modules_live_sha256": ctx.get("frozen_modules_live_sha256"),
            "shared_runtime": ctx.get("shared_runtime"),
            "slowgru_runtime_path": ctx.get("slowgru_runtime_path"),
            "slowgru_runtime_sha256": ctx.get("slowgru_runtime_sha256"),
            "slowgru_network_sha256": ctx.get("slowgru_network_sha256"),
            "arm_src": ctx.get("arm_src"),
            "obs_dim": ctx.get("obs_dim"),
            "action_dim": ctx.get("action_dim"),
            "carry_mode": (ctx["handle"].get("carry_mode")
                           if ctx["kind"] == "cc3_slowgru" else None),
        },
        "dicode_resolution": dicode_ev,
        "sha_recomputation": {
            "params_sha256_owner_protocol": params_before,
            "params_sha256_owner_declared": declared_params,
            "params_sha256_declaration_source":
                spec["declared_params_sha256"]["declaration_source"],
            "params_hash_protocol": spec["params_hash_protocol"],
            "checkpoint_file_sha256_owner_protocol": file_sha,
            "checkpoint_file_sha256_owner_declared": declared_file,
            "checkpoint_file_sha256_declaration_source":
                spec["declared_checkpoint_file_sha256"]["declaration_source"],
            "checkpoint_file_hash_protocol": spec["checkpoint_file_hash_protocol"],
            "params_unchanged_after_rollouts": params_unchanged,
            "params_sha256_after": params_after,
        },
        "canonical_env": {"observation_shape": list(entry["observation_shape"]),
                          "action_count": int(entry["action_count"])},
        "boundary_unit_check": boundary_ev,
        "policy_adapter": {
            "class": type(policy).__name__,
            "greedy_readout": (
                "owner argmax(logits) inside Candidate.policy_step"
                if ctx["kind"] == "cc2_base_gtrxl"
                else "owner pi.mode() (greedy=True)"
                if ctx["kind"] == "cc1_gtrxl128"
                else "argmax(extras['logits']) — memory update is action-independent; "
                     "faithful greedy readout of the owner's identical forward"),
            "state": policy_state,
        },
        "smoke_schedule": schedule,
        "smoke_episodes": episodes,
        "smoke_max_steps": max_steps,
        "frozen_banks": bank_ev,
        "records_by_scenario": {
            sc: {"seeds": records_by_scenario[sc]["seeds"],
                 "entry_ids": records_by_scenario[sc]["entry_ids"],
                 "episode_records_sha256":
                     records_by_scenario[sc]["episode_records_sha256"],
                 "episode_records": records_by_scenario[sc]["episode_records"]}
            for sc in scenarios},
        "results_by_scenario_smoke_only": results_by_scenario,
        "episode_records_jsonl_sha256": episode_records_jsonl_sha256,
        "common_verification": common_ev,
        "provenance": provenance,
        "performance_claim_authorized": False,
    }
    record_path = os.path.join(out_dir, "projection_record.json")
    write_json(record_path, projection_record)

    # 8c. common_evaluator_binding_result_v2.json (contract §六 full field set)
    is_teacher = spec["candidate_class"] == "TEACHER_REFERENCE"
    binding = {
        "schema": BINDING_SCHEMA,
        "supersedes": "cc4/%s/common_evaluator_binding_result.json (v1 PENDING / "
                      "MISSING_EVIDENCE, pool-readiness round 2026-07-30; retained "
                      "as history, NOT modified)" % args.candidate_id,
        "generated_at_utc": utc_now_iso(),
        "candidate_id": args.candidate_id,
        "owner": spec["owner"],
        "candidate_class": spec["candidate_class"],
        "runtime_family": spec["runtime_family"],
        "runtime_registered": True,
        "registration_authority": "CC4_CAN_REGISTER_PROJECTIONS (conditional C1-C5; "
            "audit: non_rmt_abi_binding_closure_20260731)",
        "source_capsule_root": spec["source_capsule_root"],
        "cc4_projection_path": out_dir,
        # frozen contract (formal scale — NOT the smoke scale)
        "action_mode": proj.FROZEN_ACTION_MODE,
        "action_mode_source": "frozen_evaluation_profile",
        "max_steps": proj.FROZEN_MAX_TIMESTEPS,
        "observation_shape": list(proj.FROZEN_OBSERVATION_SHAPE),
        "action_dim": proj.FROZEN_ACTION_DIM,
        "front_episode_count": proj.FROZEN_FRONT_EPISODE_COUNT,
        "back_episode_count": proj.FROZEN_BACK_EPISODE_COUNT,
        "full_episode_count": proj.FROZEN_FULL_EPISODE_COUNT,
        "episode_count_source": "evaluation_profile.json (live byte-SHA verified) "
            "scenarios.{front_l2.n,back_l2.n,full.world_seed_set.count}",
        "episode_count_status": "PASS (frozen 8/8/64 from profile; smoke executed "
            "%d/scenario at max_steps=%d)" % (episodes, max_steps),
        # smoke scale (executed; NOT a performance evaluation)
        "run_class": "INTERFACE_SMOKE",
        "smoke_episodes_per_scenario": episodes,
        "smoke_max_steps": max_steps,
        "smoke_schedule": schedule,
        "performance_claim_authorized": False,
        "strong_student_selection_authorized": False,
        "evaluation_certificate_status": "PENDING_FORMAL_EVALUATION",
        "evaluation_certificate_file": None,
        # common SHAs (live-reverified this run)
        "common_root": args.common_dir,
        "common_ready_at_binding_time": True,
        "common_runner_sha256": proj.FROZEN_COMMON_RUNNER_SHA256,
        "common_evaluator_sha256": proj.FROZEN_COMMON_EVALUATOR_SHA256,
        "evaluation_profile_sha256": proj.FROZEN_EVALUATION_PROFILE_SHA256,
        "metric_schema_sha256": proj.FROZEN_METRIC_SCHEMA_SHA256,
        "environment_lock_sha256": proj.FROZEN_ENVIRONMENT_LOCK_SHA256,
        "full_profile_sha256": proj.FROZEN_FULL_PROFILE_SHA256,
        "full_profile_status": "FROZEN",
        "front_bank_content_sha256": proj.FROZEN_FRONT_BANK_CONTENT_SHA256,
        "back_bank_content_sha256": proj.FROZEN_BACK_BANK_CONTENT_SHA256,
        "sha256sums_sha256": proj.FROZEN_SHA256SUMS_SHA256,
        "common_sha256sums_self_check": common_ev["common_sha256sums_self_check"],
        "assembly_manifest_sha256": proj.FROZEN_ASSEMBLY_MANIFEST_SHA256,
        "common_sha_match_status": "PASS (live 57/57 + per-artifact byte SHAs + "
            "profile invariants + bank content hashes)",
        "engine_module_lf_sha_verified": common_ev["engine_modules_lf_sha_verified"],
        # owner capsule artifact SHAs (live == registry == capsule sums)
        "candidate_runtime_sha256":
            spec["capsule_file_sha256"]["candidate_runtime.py"],
        "evaluate_candidate_sha256":
            spec["capsule_file_sha256"]["evaluate_candidate.py"],
        "candidate_manifest_sha256":
            spec["capsule_file_sha256"]["candidate_manifest.json"],
        "checkpoint_contract_sha256":
            spec["capsule_file_sha256"]["checkpoint_contract.json"],
        "capsule_file_verification": capsule_ev,
        "source_capsule_missing_files": [],
        # params / checkpoint (owner protocol, recomputed this run)
        "params_sha256": params_before,
        "params_sha256_cc4_recomputation": "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL",
        "params_hash_protocol": spec["params_hash_protocol"],
        "params_sha256_declaration_source":
            spec["declared_params_sha256"]["declaration_source"],
        "params_unchanged": params_unchanged,
        "checkpoint_file_sha256": file_sha,
        "checkpoint_file_sha256_cc4_recomputed": file_sha,
        "checkpoint_file_sha256_status": "CC4_RECOMPUTED_MATCH_VIA_OWNER_PROTOCOL",
        "checkpoint_file_sha256_verified_by_cc4": True,
        "checkpoint_file_hash_protocol": spec["checkpoint_file_hash_protocol"],
        # projection identity
        "projection_module_sha256": provenance["projection_module_sha256"],
        "driver_module_sha256": provenance["driver_module_sha256"],
        "bound_owner_runtime_sha256": spec["bound_owner_runtime_sha256"],
        "interface_smoke_status": "PASS",
        "binding_status": "PASS",
        "formal_eval_binding": "INTERFACE_SMOKE_PASS_FORMAL_PENDING",
        "remaining_blocker": None,
        "episode_records_jsonl_sha256": episode_records_jsonl_sha256,
        "records_sha256_by_scenario": {
            sc: records_by_scenario[sc]["episode_records_sha256"]
            for sc in scenarios},
        "results_by_scenario_smoke_only": results_by_scenario,
        "wandb_stub": ctx.get("wandb_stub"),
        "dicode_resolution": dicode_ev,
        "gpu": gpu_ev,
        "device_identity": provenance["device_identity"],
        "git_commit_head": provenance["git_commit_head"],
        "provenance_pid": provenance["pid"],
        # eligibility flags
        "formal_student_ranking_eligible": spec["formal_student_ranking_eligible"],
        "strong_student_selection_eligible":
            spec["strong_student_selection_eligible"],
        "reference_only": spec["reference_only"],
        "student_rank": spec["student_rank"],
        "budget_class": spec["budget_class"],
        "training_steps": spec["training_steps"],
        "training_seed": spec["training_seed"],
        "teacher_included_in_student_ranking": False,
        "counts_toward_student_binding_count": (
            False if is_teacher else True),
    }
    if spec["loader_kind"] == "cc3_slowgru":
        binding["carry_mode"] = spec["carry_mode"]
        binding["segment_boundary_steps"] = spec["segment_boundary_steps"]
        binding["boundary_semantics"] = spec["boundary_semantics"]
        binding["boundary_unit_check"] = boundary_ev
    binding_path = os.path.join(out_dir, "common_evaluator_binding_result_v2.json")
    write_json(binding_path, binding)

    # 8d. SHA256SUMS_V2 over the three evidence files
    sums_entries = []
    for fn in ("episode_records.jsonl", "projection_record.json",
               "common_evaluator_binding_result_v2.json"):
        sums_entries.append("%s  %s" % (proj.sha256_file(os.path.join(out_dir, fn)), fn))
    sums_v2_path = os.path.join(out_dir, "SHA256SUMS_V2")
    with open(sums_v2_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(sums_entries) + "\n")

    # 8e. READY_V2.json (honest gates; sums-excluded like READY.json)
    gates = {
        "G1_CAPSULE_FILE_SHA_MATCH": all(
            capsule_ev[fn]["match"] for fn in proj.CAPSULE_FILES),
        "G2_CHECKPOINT_FILE_SHA_OWNER_RECOMPUTE_MATCH": file_sha == declared_file,
        "G3_PARAMS_SHA_OWNER_RECOMPUTE_MATCH": params_before == declared_params,
        "G4_INTERFACE_SMOKE_EXECUTED": all(
            len(records_by_scenario[sc]["episode_records"]) == episodes
            for sc in scenarios),
        "G5_PARAMS_UNCHANGED": params_unchanged,
        "G6_COMMON_SUMS_57_57": common_ev["common_sha256sums_self_check"]
            == "PASS (%d/%d)" % (proj.FROZEN_COMMON_SUMS_ENTRY_COUNT,
                                 proj.FROZEN_COMMON_SUMS_ENTRY_COUNT),
        "G7_EPISODE_COUNTS_FROM_PROFILE":
            common_ev["profile_episode_counts"]["front_episode_count"] == 8
            and common_ev["profile_episode_counts"]["back_episode_count"] == 8
            and common_ev["profile_episode_counts"]["full_episode_count"] == 64,
        "G8_GPU_ALLOWED": gpu_ev["visible_gpu_uuids"] and all(
            u in proj.CC4_GPU_ALLOWED_UUIDS
            for u in gpu_ev["visible_gpu_uuids"]),
    }
    if spec["loader_kind"] == "cc3_slowgru":
        gates["G9_BOUNDARY_SEMANTICS_UNIT_CHECK"] = bool(boundary_ev)
    ready_v2 = {
        "schema": READY_V2_SCHEMA,
        "candidate_id": args.candidate_id,
        "runtime_family": spec["runtime_family"],
        "READY_V2": all(gates.values()),
        "binding_status": "PASS",
        "candidate_class": spec["candidate_class"],
        "counts_toward_student_binding_count": (False if is_teacher else True),
        "teacher_reference_binding": ("PASS" if is_teacher else None),
        "run_class": "INTERFACE_SMOKE",
        "performance_claim_authorized": False,
        "formal_ranking_authorized_by_this_file": False,
        "gates": gates,
        "generated_at_utc": utc_now_iso(),
        "evidence_files": ["episode_records.jsonl", "projection_record.json",
                           "common_evaluator_binding_result_v2.json",
                           "SHA256SUMS_V2"],
        "honest_false_discipline": "ANY failed gate keeps READY_V2 false; no gate "
            "is ever faked to PASS; smoke is not performance",
    }
    write_json(os.path.join(out_dir, "READY_V2.json"), ready_v2)

    print("[done] %s binding_status=PASS READY_V2=%s gates=%s"
          % (args.candidate_id, ready_v2["READY_V2"],
             {k: v for k, v in gates.items() if not v} or "all PASS"), flush=True)
    print("[done] out=%s" % out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
