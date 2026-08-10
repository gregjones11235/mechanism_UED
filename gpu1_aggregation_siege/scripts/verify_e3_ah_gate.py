#!/usr/bin/env python3
"""E3 verification gate A-H checker — audit-hardened (sole-controller 2026-08-10).

The verifier does NOT trust the run reports' booleans.  It:
  * independently re-loads each RunState checkpoint and recomputes the params
    / optimizer / RNG / memory facts from the actual bytes;
  * independently verifies the controller-signed authorization manifest
    (Ed25519 signature, registry hash recomputation, per-asset SHA);
  * for H, spawns an INDEPENDENT subprocess to restore the checkpoint and
    records child PID / argv / exit code / output hash;
  * C/D: per-leaf path/shape/dtype/hash/value equality across sessions
    (params + optimizer all leaves + training RNG + architecture memory).

Usage:
  python verify_e3_ah_gate.py --run-dir=<RUN_DIR> --candidate=<ID>
      [--auth-manifest=<signed.json>] [--registry=<registry.json>]
      [--do-a] [--out=<VERIFY_JSON>]
"""

import hashlib
import dataclasses
from collections.abc import Mapping
import json
import os
import pickle
import subprocess
import sys
import time
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
OPT_STEPS_PER_UPDATE = 32  # num_minibatches 8 x update_epochs 4
AUTH_PUBLIC_KEY = os.path.join(SIEGE_ROOT, "auth", "e3_controller_public_key.bin")
AUTH_REGISTRY = os.path.join(SIEGE_ROOT, "auth", "formal_asset_registry.json")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _params_hash(params) -> str:
    import jax
    leaves = jax.tree_util.tree_leaves(params)
    digest = hashlib.sha256()
    for leaf in leaves:
        arr = jax.numpy.asarray(leaf)
        digest.update(arr.astype(jax.numpy.float32).tobytes())
    return digest.hexdigest()


def _leaf_report(path: str, leaf) -> dict:
    import numpy as np
    arr = np.asarray(leaf)
    return {
        "path": path,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        # dtype is checked separately; hash the authoritative bytes without
        # lossy float32/bool conversion.
        "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        "value": None,
    }


def _per_leaf_equality(a, b) -> dict:
    """Per-leaf path/shape/dtype/hash equality between two pytrees."""
    la = _tree_leaves_with_path(a)
    lb = _tree_leaves_with_path(b)
    if len(la) != len(lb):
        return {"equal": False,
                "reason": f"leaf count mismatch {len(la)} vs {len(lb)}"}
    mismatches = []
    checked = 0
    for (pa, xa), (pb, xb) in zip(la, lb):
        if pa != pb:
            mismatches.append(f"path {pa} != {pb}")
            continue
        ra, rb = _leaf_report(pa, xa), _leaf_report(pb, xb)
        if ra["shape"] != rb["shape"] or ra["dtype"] != rb["dtype"]:
            mismatches.append(
                f"{ra['path']}: shape/dtype {ra['shape']}/{ra['dtype']} != "
                f"{rb['shape']}/{rb['dtype']}")
            continue
        if ra["sha256"] != rb["sha256"]:
            mismatches.append(f"{ra['path']}: hash {ra['sha256'][:8]} != "
                              f"{rb['sha256'][:8]}")
            continue
        checked += 1
    return {"equal": not mismatches, "checked_leaves": checked,
            "mismatches": mismatches[:10]}


def _tree_leaves_with_path(tree, prefix: str = "") -> list[tuple[str, object]]:
    """Dependency-free deterministic pytree walk for checkpoint evidence.

    Covers mappings, dataclasses, namedtuples and sequence containers used by
    Flax/Optax state.  It intentionally does not import JAX, so the verifier's
    structural primitive can also run before a GPU runtime is available.
    """
    if isinstance(tree, Mapping):
        out = []
        for key in sorted(tree, key=lambda k: str(k)):
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_tree_leaves_with_path(tree[key], path))
        return out
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        out = []
        for field in dataclasses.fields(tree):
            path = f"{prefix}.{field.name}" if prefix else field.name
            out.extend(_tree_leaves_with_path(getattr(tree, field.name), path))
        return out
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        out = []
        for name in tree._fields:
            path = f"{prefix}.{name}" if prefix else str(name)
            out.extend(_tree_leaves_with_path(getattr(tree, name), path))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for idx, value in enumerate(tree):
            path = f"{prefix}.{idx}" if prefix else str(idx)
            out.extend(_tree_leaves_with_path(value, path))
        return out
    return [(prefix or "<root>", tree)]


def _path_str(keypath) -> str:
    parts = []
    for p in keypath:
        from jax.tree_util import GetAttrKey, SequenceKey, DictKey
        if isinstance(p, GetAttrKey):
            parts.append(str(p.name))
        elif isinstance(p, SequenceKey):
            parts.append(str(p.idx))
        elif isinstance(p, DictKey):
            parts.append(str(p.key))
        else:
            parts.append(str(p))
    return ".".join(parts)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=SIEGE_ROOT,
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(
            f"cannot obtain verifier worktree HEAD: {proc.stderr[-300:]}")
    head = (proc.stdout or "").strip()
    if len(head) != 40:
        raise RuntimeError("verifier worktree HEAD is not a full commit SHA")
    return head


def _load_initial_boundary(run_dir: str, session_idx: int) -> dict:
    """Load and independently hash-check a runner-written input boundary."""
    stem = Path(run_dir) / "boundaries" / (
        "session_%03d_initial" % session_idx)
    state_path = Path(str(stem) + ".state.pkl")
    meta_path = Path(str(stem) + ".meta.json")
    if not state_path.is_file() or not meta_path.is_file():
        raise RuntimeError(
            f"missing initial boundary evidence for session {session_idx}")
    meta = _load_json(meta_path)
    got = _sha256_file(str(state_path))
    if got != meta.get("state_file_sha256"):
        raise RuntimeError(
            f"initial boundary session {session_idx} SHA mismatch")
    with open(state_path, "rb") as fh:
        state = pickle.load(fh)
    required = {
        "schema", "session_idx", "source_commit", "params", "opt_state",
        "train_step", "training_rng", "global_update_step",
        "global_env_steps", "session_boundary_semantics",
        "environment_restore_input", "architecture_memory_restore_input",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise RuntimeError(
            f"initial boundary session {session_idx} missing {missing}")
    if int(state["session_idx"]) != session_idx:
        raise RuntimeError("initial boundary session index mismatch")
    return {"state": state, "metadata": meta, "sha256": got}


def _restore_runstate(ckpt_base_path: str) -> dict:
    from dicode.simulator_frontier.runstate_codec import RunStateCheckpointManager
    mgr = RunStateCheckpointManager()
    return mgr.restore(ckpt_base_path)["run_state"]


def _independent_subprocess_restore(ckpt_base_path: str) -> dict:
    """H: restore the RunState in an INDEPENDENT python subprocess.

    Records the child PID, argv, exit code, and the sha256 of the child's
    stdout (the content hash the child prints).  Never trusts the parent's
    in-process restore.
    """
    code = (
        "import sys, hashlib; sys.path.insert(0, %r); "
        "from dicode.simulator_frontier.runstate_codec import "
        "RunStateCheckpointManager, runstate_content_hash; "
        "rs = RunStateCheckpointManager().restore(%r)[\"run_state\"]; "
        "print(runstate_content_hash(rs))" % (SRC_DIR, ckpt_base_path)
    )
    started = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=300)
    out = (proc.stdout or "").strip()
    child = {
        "pid": proc.pid,
        "argv": [sys.executable, "-c", "<restore-script>"],
        "exit_code": proc.returncode,
        "stdout_sha256": hashlib.sha256(
            (proc.stdout or "").encode("utf-8")).hexdigest(),
        "restored_content_hash": out if proc.returncode == 0 else "",
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-3:],
        "elapsed_s": round(time.time() - started, 2),
    }
    return child


def _verify_a(candidate_id, run_dir) -> dict:
    """A: independently recompute the task-based success predicate and its
    applicability (positive + negative example), plus a real branch
    non-degeneracy probe from an actual-N run."""
    import jax
    import run_e3_real_smoke as prod
    import e3_capsule_actualn as capsule_mod

    mount = prod.mount_student(candidate_id)
    out = {"candidate": candidate_id, "schema": "e3_verification_gate_A/v2"}
    # Build + validate the task predicate (fail closed on applicability).
    setup = capsule_mod._build_multitask_setup(
        max_timesteps=24, reset_seed=42)
    task = setup["task"]
    task_name = type(task).__name__
    pred, meta = capsule_mod.build_task_success_predicate(task)
    prog = capsule_mod.build_task_progress_fn(task)
    indices = meta["achievement_indices"]
    out["predicate_meta"] = meta
    # Positive / negative example from a real state.
    state0 = setup["state0"]
    out["applicability"] = capsule_mod.verify_predicate_applicability(
        state0, pred, indices, task_name)
    # Real branch non-degeneracy probe: run actual-N and check transitions +
    # distinct seeds.
    capsule = capsule_mod.capture_frontier_capsule(
        student=mount["adapter"], student_params=mount["params"],
        run_id="e3-verify-ah-a", reset_seed=42, capture_at_step=16,
        max_timesteps=24, success_threshold=0.50,
        memory_mode="SAVED_POLICY_MEMORY")
    capsule["student"] = mount["adapter"]
    capsule["student_params"] = mount["params"]
    an = capsule_mod.run_same_state_actual_n(
        capsule=capsule, n=4, horizon=16, seed_base=42,
        memory_mode="SAVED_POLICY_MEMORY")
    branch_seeds = an.get("branch_seeds")
    out["branch_seeds_distinct"] = (
        branch_seeds is not None and len(set(branch_seeds)) == len(branch_seeds))
    out["branches_transitioned"] = all(
        o.transitions_used >= 1 for o in an["outcomes"])
    out["progress_finite"] = all(
        0.0 <= float(o.progress) <= 1.0 for o in an["outcomes"])
    out["success_basis_recorded"] = all(
        getattr(o, "outcome_hash", "") != "" for o in an["outcomes"])
    out["real_branches_non_degenerate"] = bool(
        out["branch_seeds_distinct"] and out["branches_transitioned"]
        and out["progress_finite"])
    out["verdict"] = "PASS" if (
        out["applicability"]["applicable"]
        and out["real_branches_non_degenerate"]
        and out["success_basis_recorded"]) else "FAIL"
    return out


def _verify_evidence(run_dir, candidate_id, auth_manifest=None) -> dict:
    """B/C/D/F/G/H — independent recomputation from checkpoints + reports."""
    import numpy as np
    metadata_path = Path(run_dir) / "RUN_METADATA.json"
    git_binding_path = Path(run_dir) / "GIT_BINDING.json"
    if not metadata_path.is_file() or not git_binding_path.is_file():
        return {"verdict": "FAIL",
                "error": "missing RUN_METADATA.json or GIT_BINDING.json"}
    run_metadata = _load_json(metadata_path)
    git_binding = _load_json(git_binding_path)
    try:
        verifier_head = _git_head()
    except Exception as exc:
        return {"verdict": "FAIL", "error": str(exc)}
    runtime_head = str(run_metadata.get("source_commit", ""))
    source_binding_pass = (
        len(runtime_head) == 40
        and runtime_head == str(git_binding.get("head", ""))
        and runtime_head == verifier_head
        and candidate_id == str(run_metadata.get("candidate_id", "")))
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
    b_pass = b_pass and all(reports[i]["current_session_idx"] == i for i in idxs)
    for i in idxs:
        r = reports[i]
        if i > 1 and int(r["start_global_update"]) != (i - 1) * UPDATES_PER_SESSION:
            b_pass = False
        if int(r["global_update_step"]) != i * UPDATES_PER_SESSION:
            b_pass = False
        if int(r["num_updates_in_session"]) != UPDATES_PER_SESSION:
            b_pass = False

    # ---- F/G: real 1x100 ----
    fg_pass = all(
        int(r["num_updates_in_session"]) == UPDATES_PER_SESSION
        and int(r["global_update_step"] - r["start_global_update"]) == UPDATES_PER_SESSION
        and int(r["global_env_steps"] - r["start_global_env_steps"])
        == UPDATES_PER_SESSION * ENV_STEPS_PER_UPDATE
        for r in reports.values())

    # ---- C/D: per-leaf continuity across sessions (INDEPENDENT) ----
    # Load each checkpoint, recompute facts from bytes; compare consecutive.
    ckpt_facts = {}   # session -> final checkpoint facts
    boundaries = {}   # session -> checkpoint-authoritative initial state
    for i in idxs:
        ckpt = ckpt_dir / ("e3_canonical_runstate_s%03d" % i)
        if not (ckpt_dir / ("e3_canonical_runstate_s%03d.state.pkl" % i)).is_file():
            return {"verdict": "FAIL", "error": "missing checkpoint %s" % ckpt}
        rs = _restore_runstate(str(ckpt))
        if str(rs.get("source_commit", "")) != runtime_head:
            source_binding_pass = False
        ckpt_facts[i] = {
            "params_hash": _params_hash(rs["params"]),
            "opt_step": int(rs["train_step"]),
            "training_rng": _rng_token(rs["training_rng"]),
            "mem_present": "architecture_memory" in rs and bool(rs["architecture_memory"]),
            "rs": rs,
        }
        try:
            boundaries[i] = _load_initial_boundary(run_dir, i)
        except Exception as exc:
            return {"verdict": "FAIL", "error": str(exc)}

    c_pass = True
    continuity = {"params": {}, "optimizer": {}, "rng": {}, "memory": {},
                  "boundary_semantics": {}}
    for i in idxs:
        rs = ckpt_facts[i]["rs"]
        initial = boundaries[i]["state"]
        boundary_ok = (
            initial["session_boundary_semantics"] ==
                "B_NEW_SESSION_ENV_AND_MEMORY_RESET"
            and initial["environment_restore_input"] is None
            and initial["architecture_memory_restore_input"] is None)
        if not boundary_ok:
            c_pass = False
        # expected optimizer step after i sessions of 100 updates
        exp_step = i * UPDATES_PER_SESSION * OPT_STEPS_PER_UPDATE
        continuity["optimizer"][str(i)] = {
            "actual_step": ckpt_facts[i]["opt_step"], "expected": exp_step}
        if ckpt_facts[i]["opt_step"] != exp_step:
            c_pass = False
        continuity["rng"][str(i)] = ckpt_facts[i]["training_rng"][:12]
        if i > 1 and ckpt_facts[i]["training_rng"] == ckpt_facts[i - 1]["training_rng"]:
            c_pass = False
        continuity["memory"][str(i)] = ckpt_facts[i]["mem_present"]
        if not ckpt_facts[i]["mem_present"]:
            c_pass = False
        continuity["boundary_semantics"][str(i)] = reports[i].get(
            "session_boundary_semantics")
        if reports[i].get("session_boundary_semantics") != \
                "B_NEW_SESSION_ENV_AND_MEMORY_RESET":
            c_pass = False
        # params carry: session i+1's mounted params == session i's final
        if i + 1 in reports:
            mounted_next = str(reports[i + 1].get("params_sha256", ""))
            continuity["params"][str(i)] = {
                "next_mounted": mounted_next[:16],
                "this_final": ckpt_facts[i]["params_hash"][:16],
            }
            if mounted_next != ckpt_facts[i]["params_hash"]:
                c_pass = False
        # per-leaf equality: session i+1's INITIAL params == session i's FINAL.
        # The controller resumes with the previous runstate's params; we verify
        # the report's recorded initial hash against the previous final hash
        # (independent recompute) — this is the per-leaf continuity evidence.
        if i > 1:
            ini = str(reports[i].get("initial_trainstate_params_sha256", ""))
            if ini != ckpt_facts[i - 1]["params_hash"]:
                c_pass = False
            previous = ckpt_facts[i - 1]["rs"]
            params_eq = _per_leaf_equality(previous["params"], initial["params"])
            opt_eq = _per_leaf_equality(previous["opt_state"], initial["opt_state"])
            rng_eq = _per_leaf_equality(
                previous["training_rng"], initial["training_rng"])
            counters_equal = (
                int(previous["train_step"]) == int(initial["train_step"])
                and int(previous["global_update_step"]) ==
                    int(initial["global_update_step"])
                and int(previous["global_env_steps"]) ==
                    int(initial["global_env_steps"]))
            source_equal = (
                str(previous["source_commit"]) == str(initial["source_commit"]))
            continuity["params"][str(i)] = params_eq
            continuity["optimizer"][str(i)]["final_to_initial"] = opt_eq
            continuity["optimizer"][str(i)]["counter_equality"] = counters_equal
            continuity["rng"][str(i)] = {
                "previous_final_to_initial": rng_eq}
            continuity["boundary_semantics"][str(i)] = {
                "semantic": initial["session_boundary_semantics"],
                "environment_restore_input_is_none":
                    initial["environment_restore_input"] is None,
                "architecture_memory_restore_input_is_none":
                    initial["architecture_memory_restore_input"] is None,
                "source_equal": source_equal,
                "boundary_sha256": boundaries[i]["sha256"],
            }
            if not (params_eq["equal"] and opt_eq["equal"] and rng_eq["equal"]
                    and counters_equal and source_equal):
                c_pass = False

    # ---- H: independent subprocess restore for the LAST session ----
    last = idxs[-1]
    last_ckpt = str(ckpt_dir / ("e3_canonical_runstate_s%03d" % last))
    child = _independent_subprocess_restore(last_ckpt)
    # independent recompute of the content hash in-process for comparison
    local_hash = ""
    try:
        rs_last = _restore_runstate(last_ckpt)
        from dicode.simulator_frontier.runstate_codec import runstate_content_hash
        local_hash = runstate_content_hash(rs_last)
    except Exception as exc:
        local_hash = f"ERR:{exc!r}"
    h_pass = (child["exit_code"] == 0
              and bool(child["restored_content_hash"])
              and child["restored_content_hash"] == local_hash)

    # ---- Authorization (independent, not trusting reports) ----
    auth_report = {"verified": False,
                   "error": "authorization manifest required"}
    if auth_manifest:
        import e3_authorization as am
        try:
            auth = am.load_authorization(
                auth_manifest,
                public_key_path=AUTH_PUBLIC_KEY,
                registry_path=AUTH_REGISTRY,
            )
            runner_sha = _sha256_file(
                os.path.join(SIEGE_ROOT, "scripts", "run_e3_formal_longrun.py"))
            static_assets = am.resolve_candidate_static_assets(
                AUTH_REGISTRY, candidate_id)
            if str(run_metadata.get("runner_sha256", "")) != runner_sha:
                raise ValueError(
                    "run metadata runner SHA != verifier worktree runner SHA")
            if str(run_metadata.get("authorization_manifest_hash", "")) != \
                    auth.manifest_hash:
                raise ValueError(
                    "run metadata authorization manifest hash mismatch")
            am.verify_runtime_authorization(
                auth, runtime_head, candidate_id,
                runner_sha, static_assets["checkpoint_sha256"],
                str(run_metadata.get("mounted_params_sha256", "")),
                AUTH_REGISTRY,
                task_asset_manifest_sha256=
                    static_assets["task_asset_manifest_sha256"])
            if not source_binding_pass:
                raise ValueError(
                    "runtime source commit is not consistently bound across "
                    "RUN_METADATA/GIT_BINDING/checkpoints/current worktree")
            auth_report = {
                "verified": True, "authorization_id": auth.authorization_id,
                "source_commit": runtime_head,
                "verifier_head": verifier_head,
                "scope": auth.scope,
            }
        except (ValueError, OSError, KeyError) as exc:
            auth_report = {"verified": False, "error": str(exc)}

    # FINAL_STATUS booleans are informational only; H is decided solely by
    # the independent child restore and independently recomputed content hash.

    return {
        "candidate": candidate_id,
        "architecture_family": reports[1].get("architecture_family"),
        "sessions_run": idxs,
        "gate_B_session_index": "PASS" if b_pass else "FAIL",
        "gate_CD_continuity": "PASS" if c_pass else "FAIL",
        "continuity": continuity,
        "gate_FG_real_100": "PASS" if fg_pass else "FAIL",
        "env_steps_per_session": [
            (i, int(r["global_env_steps"] - r["start_global_env_steps"]))
            for i, r in reports.items()],
        "gate_H_independent_restore": "PASS" if h_pass else "FAIL",
        "independent_restore": child,
        "local_content_hash": local_hash,
        "authorization": auth_report,
        "gate_AUTH_source_binding": (
            "PASS" if auth_report.get("verified") and source_binding_pass
            else "FAIL"),
        "source_binding": {
            "runtime_head": runtime_head,
            "git_binding_head": str(git_binding.get("head", "")),
            "verifier_head": verifier_head,
            "consistent": source_binding_pass,
        },
    }


def _rng_token(rng):
    import jax.numpy as jnp
    return jnp.asarray(rng).tobytes().hex()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_dir = None
    candidate_id = None
    do_a = False
    out_path = None
    auth_manifest = None
    for arg in argv:
        if arg.startswith("--run-dir="):
            run_dir = arg.split("=", 1)[1]
        elif arg.startswith("--candidate="):
            candidate_id = arg.split("=", 1)[1]
        elif arg == "--do-a":
            do_a = True
        elif arg.startswith("--out="):
            out_path = arg.split("=", 1)[1]
        elif arg.startswith("--auth-manifest="):
            auth_manifest = arg.split("=", 1)[1]
    if not run_dir or not candidate_id:
        print("usage: verify_e3_ah_gate.py --run-dir=<DIR> --candidate=<ID> "
              "[--auth-manifest=...] [--do-a] [--out=...]")
        return FAIL

    report = {}
    if do_a:
        a = _verify_a(candidate_id, run_dir)
        report["gate_A"] = a
        print(f"[verify-ah] gate_A verdict={a['verdict']} "
              f"applicable={a.get('applicability', {}).get('applicable')} "
              f"branches_nd={a.get('real_branches_non_degenerate')}")
    ev = _verify_evidence(run_dir, candidate_id, auth_manifest=auth_manifest)
    report["evidence"] = ev
    print(f"[verify-ah] {candidate_id}: "
          f"B={ev['gate_B_session_index']} C/D={ev['gate_CD_continuity']} "
          f"F/G={ev['gate_FG_real_100']} H={ev['gate_H_independent_restore']}"
          + (f" AUTH={'OK' if ev['authorization'] and ev['authorization']['verified'] else 'BLOCKED'}"
             if ev.get("authorization") else ""))
    all_pass = True
    for gate_key in ("gate_B_session_index", "gate_CD_continuity",
                     "gate_FG_real_100", "gate_H_independent_restore",
                     "gate_AUTH_source_binding"):
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
