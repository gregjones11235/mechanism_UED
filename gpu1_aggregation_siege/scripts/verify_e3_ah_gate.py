#!/usr/bin/env python3
"""E3 verification gate A-H checker.

Reads the evidence produced by run_e3_formal_longrun.py for ONE candidate arm
and verifies gates:

  A  actual-N same-state determinism (positive + negative) — needs a live GPU
     mount, run via --do-a (once per arm).
  B  session index strict 1 -> 2 -> 3 (each +1, no double increment).
  C/D RMT16 / SlowGRU continuity: params carry-forward, optimizer step
     continuity, RNG advance, architecture memory restored across sessions.
  F/G real 1x100 session (num_updates_in_session == 100, env_steps == 13,107,200).
  H  two-arm 2x100 + independent fresh-process restore (per-session
     fresh_process_restore_equivalent + FINAL_STATUS).

Usage:
  python verify_e3_ah_gate.py --run-dir=<RUN_DIR> --candidate=<ID>
      [--do-a] [--out=<VERIFY_JSON>]
"""

import hashlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIEGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(SIEGE_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCRIPT_DIR)

PASS, FAIL, BLOCKED = 0, 4, 5
NUMBER_ENVS = 1024
NUMBER_STEPS = 128
ENV_STEPS_PER_UPDATE = NUMBER_ENVS * NUMBER_STEPS   # 131072
UPDATES_PER_SESSION = 100


def _params_hash(params) -> str:
    import jax
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256()
    for leaf in leaves:
        arr = jax.numpy.asarray(leaf)
        digest.update(arr.astype(jax.numpy.float32).tobytes())
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _restore_runstate(ckpt_path):
    from dicode.simulator_frontier.runstate_codec import RunStateCheckpointManager
    mgr = RunStateCheckpointManager()
    return mgr.restore(ckpt_path)["run_state"]


def _verify_a(candidate_id, run_id_prefix="e3-verify-ah"):
    """A: real capsule same-state determinism (positive + negative)."""
    import jax
    import run_e3_real_smoke as prod
    import e3_capsule_actualn as capsule_mod

    mount = prod.mount_student(candidate_id)
    out = {"candidate": candidate_id, "schema": "e3_verification_gate_A/v1"}
    # Positive: same reset_seed -> same state_id; same seed_base -> same branches.
    caps1 = capsule_mod.capture_frontier_capsule(
        student=mount["adapter"], student_params=mount["params"],
        run_id=f"{run_id_prefix}-p1", reset_seed=42, capture_at_step=16,
        max_timesteps=24, success_threshold=0.50,
        memory_mode="SAVED_POLICY_MEMORY")
    caps2 = capsule_mod.capture_frontier_capsule(
        student=mount["adapter"], student_params=mount["params"],
        run_id=f"{run_id_prefix}-p2", reset_seed=42, capture_at_step=16,
        max_timesteps=24, success_threshold=0.50,
        memory_mode="SAVED_POLICY_MEMORY")
    out["same_seed_same_state_id"] = (caps1["state_id"] == caps2["state_id"])
    out["state_id"] = caps1["state_id"]

    an1 = capsule_mod.run_same_state_actual_n(
        capsule=caps1, n=4, horizon=16, seed_base=42,
        memory_mode="SAVED_POLICY_MEMORY")
    an2 = capsule_mod.run_same_state_actual_n(
        capsule=caps2, n=4, horizon=16, seed_base=42,
        memory_mode="SAVED_POLICY_MEMORY")

    def _branch_signature(an):
        # Per-branch outcome signature: rng_seed (derives from seed_base) and
        # outcome_hash (success/progress/terminal).  Varies with seed_base.
        return [(o.rng_seed, o.outcome_hash) for o in an["outcomes"]]

    sig1, sig2 = _branch_signature(an1), _branch_signature(an2)
    out["same_capsule_same_branch_seeds"] = (sig1 == sig2)
    out["same_capsule_same_successes"] = (
        int(an1["estimate"].successes) == int(an2["estimate"].successes))

    # Negative 1: different reset_seed -> different state_id.
    caps3 = capsule_mod.capture_frontier_capsule(
        student=mount["adapter"], student_params=mount["params"],
        run_id=f"{run_id_prefix}-n1", reset_seed=43, capture_at_step=16,
        max_timesteps=24, success_threshold=0.50,
        memory_mode="SAVED_POLICY_MEMORY")
    out["diff_seed_diff_state_id"] = (caps1["state_id"] != caps3["state_id"])

    # Negative 2: different seed_base -> different per-branch outcomes.
    an3 = capsule_mod.run_same_state_actual_n(
        capsule=caps1, n=4, horizon=16, seed_base=7,
        memory_mode="SAVED_POLICY_MEMORY")
    sig3 = _branch_signature(an3)
    out["diff_seedbase_diff_branches"] = (sig1 != sig3)
    out["positive_pass"] = bool(
        out["same_seed_same_state_id"] and out["same_capsule_same_branch_seeds"]
        and out["same_capsule_same_successes"])
    out["negative_pass"] = bool(
        out["diff_seed_diff_state_id"] and out["diff_seedbase_diff_branches"])
    out["verdict"] = "PASS" if (out["positive_pass"] and out["negative_pass"]) else "FAIL"
    return out


def _verify_evidence(run_dir, candidate_id):
    """B/C/D/F/G/H evidence analysis from a completed controller run_dir."""
    ev_dir = Path(run_dir) / "evidence"
    ckpt_dir = Path(run_dir) / "runstate"
    reports = {}
    for p in sorted(ev_dir.glob("session_*.json")):
        r = _load_json(p)
        reports[int(r["session_idx"])] = r
    idxs = sorted(reports)
    if not idxs:
        return {"verdict": "FAIL", "error": "no session reports"}

    # ---- B: session index strict 1->2->3 ----
    b_pass = (idxs == list(range(1, idxs[-1] + 1)))
    b_deltas = [reports[i]["current_session_idx"] - i for i in idxs]
    b_pass = b_pass and all(d == 0 for d in b_deltas)
    for i in idxs:
        r = reports[i]
        if i > 1:
            expect = (i - 1) * UPDATES_PER_SESSION
            if int(r["start_global_update"]) != expect:
                b_pass = False
        if int(r["global_update_step"]) != i * UPDATES_PER_SESSION:
            b_pass = False
        if int(r["num_updates_in_session"]) != UPDATES_PER_SESSION:
            b_pass = False

    # ---- F/G: real 1x100 session ----
    fg_pass = all(
        int(r["num_updates_in_session"]) == UPDATES_PER_SESSION
        and int(r["global_update_step"] - r["start_global_update"]) == UPDATES_PER_SESSION
        and (int(r["global_env_steps"] - r["start_global_env_steps"])
             == UPDATES_PER_SESSION * ENV_STEPS_PER_UPDATE)
        for r in reports.values())
    fg_envs = [(i, int(r["global_env_steps"] - r["start_global_env_steps"]))
               for i, r in reports.items()]

    # ---- C/D: continuity across sessions ----
    params_hashes = {}     # session -> checkpoint params sha
    opt_steps = {}         # session -> train_step from runstate
    rng_tokens = {}        # session -> training_rng
    arch_mem = {}          # session -> architecture_memory present
    for i in idxs:
        ckpt = ckpt_dir / f"e3_canonical_runstate_s{i:03d}"
        if not (ckpt_dir / f"e3_canonical_runstate_s{i:03d}.state.pkl").is_file():
            return {"verdict": "FAIL", "error": f"missing checkpoint {ckpt}"}
        rs = _restore_runstate(str(ckpt))
        params_hashes[i] = _params_hash(rs["params"])
        opt_steps[i] = int(rs["train_step"])
        rng_tokens[i] = _rng_token(rs["training_rng"])
        arch_mem[i] = "architecture_memory" in rs and bool(rs["architecture_memory"])

    c_pass = True
    for i in idxs:
        if i == 1:
            # session 2 report params_sha == session 1 checkpoint final params
            if 2 in reports:
                if reports[2]["params_sha256"] != params_hashes[1]:
                    c_pass = False
        else:
            if reports[i]["params_sha256"] != params_hashes[i - 1]:
                c_pass = False
        # optimizer step == i completed sessions x 100 updates x 32
        # (num_minibatches 8 x update_epochs 4 per outer update).
        expect_step = i * UPDATES_PER_SESSION * 32
        if opt_steps[i] != expect_step:
            c_pass = False
        # RNG advances each session (never identical)
        if i > 1 and rng_tokens[i] == rng_tokens[i - 1]:
            c_pass = False
        # architecture memory present in every session's runstate
        if not arch_mem[i]:
            c_pass = False

    # ---- H: fresh-process restore per session ----
    h_pass = all(bool(r.get("fresh_process_restore_equivalent"))
                 for r in reports.values())
    final = _load_json(Path(run_dir) / "FINAL_STATUS.json") \
        if (Path(run_dir) / "FINAL_STATUS.json").is_file() else {}
    h_pass = h_pass and bool(final.get("all_fresh_restore_ok"))

    b_verdict = "PASS" if b_pass else "FAIL"
    c_verdict = "PASS" if c_pass else "FAIL"
    fg_verdict = "PASS" if fg_pass else "FAIL"
    h_verdict = "PASS" if h_pass else "FAIL"
    return {
        "candidate": candidate_id,
        "architecture_family": reports[1]["architecture_family"],
        "sessions_run": idxs,
        "gate_B_session_index": b_verdict,
        "session_index_sequence": [reports[i]["current_session_idx"] for i in idxs],
        "start_global_updates": [reports[i]["start_global_update"] for i in idxs],
        "gate_CD_continuity": c_verdict,
        "params_carry_ok": c_pass,
        "optimizer_steps": opt_steps,
        "rng_advanced": {i: rng_tokens[i] for i in idxs},
        "arch_memory_present": arch_mem,
        "gate_FG_real_100": fg_verdict,
        "env_steps_per_session": fg_envs,
        "gate_H_fresh_restore": h_verdict,
        "fresh_restore_per_session": {i: bool(reports[i]["fresh_process_restore_equivalent"])
                                      for i in idxs},
        "final_all_fresh_restore_ok": final.get("all_fresh_restore_ok"),
    }


def _rng_token(rng):
    import jax.numpy as jnp
    return jnp.asarray(rng).tobytes().hex()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    run_dir = None
    candidate_id = None
    do_a = False
    out_path = None
    for arg in argv:
        if arg.startswith("--run-dir="):
            run_dir = arg.split("=", 1)[1]
        elif arg.startswith("--candidate="):
            candidate_id = arg.split("=", 1)[1]
        elif arg == "--do-a":
            do_a = True
        elif arg.startswith("--out="):
            out_path = arg.split("=", 1)[1]
    if not run_dir or not candidate_id:
        print("usage: verify_e3_ah_gate.py --run-dir=<DIR> --candidate=<ID> [--do-a]")
        return FAIL

    report = {}
    if do_a:
        a = _verify_a(candidate_id)
        report["gate_A"] = a
        print(f"[verify-ah] gate_A verdict={a['verdict']} "
              f"state_id={a['state_id'][:12]}")
    ev = _verify_evidence(run_dir, candidate_id)
    report["evidence"] = ev
    print(f"[verify-ah] {candidate_id}: "
          f"B={ev['gate_B_session_index']} C/D={ev['gate_CD_continuity']} "
          f"F/G={ev['gate_FG_real_100']} H={ev['gate_H_fresh_restore']}")
    all_pass = True
    for gate_key in ("gate_B_session_index", "gate_CD_continuity",
                     "gate_FG_real_100", "gate_H_fresh_restore"):
        if ev.get(gate_key) != "PASS":
            all_pass = False
    if do_a:
        all_pass = all_pass and a["verdict"] == "PASS"
    report["overall"] = "PASS" if all_pass else "FAIL"
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
        print(f"[verify-ah] wrote {out_path}")
    return PASS if all_pass else FAIL


if __name__ == "__main__":
    raise SystemExit(main())
