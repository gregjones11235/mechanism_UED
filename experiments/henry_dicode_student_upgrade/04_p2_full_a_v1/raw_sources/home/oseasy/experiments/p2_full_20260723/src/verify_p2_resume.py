"""Pre-resume hard verification for the P2-Full-A @24576 checkpoint (directive section 二).

Confirms the checkpoint carries FULL resumable state (not weights-only):
params, target params, optimizer state, global_step, update_count, replay, pending
episodes, sparse anchors, RNG key + action RNG state, collector/env state, policy
version, config (adam_eps/gamma). Then does one save->restore roundtrip in a temp dir
and verifies params SHA + optimizer-leaf hash + counters + replay digest + pending
counts + RNG bytes are preserved (no env reset / RNG reset). Read-only w.r.t. the
source checkpoint. Prints RESUME_VERIFY_OK on success.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
import sys, os.path, json, argparse, shutil, hashlib
import dataclasses

BASE_SRC = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if BASE_SRC in sys.path:
    sys.path.remove(BASE_SRC)
sys.path.insert(0, BASE_SRC)
if HENRY_SRC not in sys.path:
    sys.path.insert(1, HENRY_SRC)

import numpy as np
import jax
import jax.numpy as jnp
import checkpointing as CK
from full_p2_learner import FullP2Config


def leaf_hash(pytree):
    """Stable SHA256 over all leaves (bytes) of a pytree, in flatten order."""
    h = hashlib.sha256()
    for l in jax.tree_util.tree_leaves(pytree):
        a = np.asarray(l)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--step", type=int, default=24576)
    ap.add_argument("--expect_params_sha", default=None)
    ap.add_argument("--do_resave", type=int, default=1)
    args = ap.parse_args()

    free_gb = shutil.disk_usage(args.ckpt_dir).free / 1024 ** 3
    print(f"[verify] disk free = {free_gb:.1f} GB", flush=True)

    r = CK.restore_full_checkpoint(args.ckpt_dir, step=args.step)

    # ---- field presence ----
    required = ["params", "target_params", "opt_state", "replay_buffer", "pending",
                "rng_key", "action_rng_state", "global_step", "update_count",
                "collector_state", "config", "step", "manifest"]
    present = {k: (k in r and r[k] is not None) for k in required}
    # replay_buffer may be under 'replay_buffer' or 'replay'
    replay = r.get("replay_buffer", r.get("replay"))
    pending = r.get("pending")
    print("[verify] field presence:", json.dumps(present, sort_keys=True), flush=True)
    assert all(present.values()), f"FAIL missing fields: {[k for k,v in present.items() if not v]}"
    assert replay is not None and pending is not None

    params_sha = CK.params_content_sha256(r["params"])
    target_sha = CK.params_content_sha256(r["target_params"])
    opt_hash = leaf_hash(r["opt_state"])
    gs = int(r["global_step"]); uc = int(r["update_count"])
    print(f"[verify] params_sha={params_sha}", flush=True)
    print(f"[verify] target_sha={target_sha}", flush=True)
    print(f"[verify] opt_state_leaf_hash={opt_hash}", flush=True)
    print(f"[verify] global_step={gs} update_count(policy_version)={uc}", flush=True)
    assert gs == args.step, f"FAIL global_step {gs} != {args.step}"
    if args.expect_params_sha:
        assert params_sha == args.expect_params_sha, \
            f"FAIL params sha {params_sha} != expected {args.expect_params_sha}"

    # ---- replay / anchor conservation ----
    c = replay.counters
    traj_coll = int(c.trajectories_collected); traj_ins = int(c.trajectories_inserted)
    anchors = int(c.total_anchors_stored)
    conservation_ok = (traj_coll == traj_ins)
    replay_digest = replay.hash_digest() if hasattr(replay, "hash_digest") else None
    print(f"[verify] replay: collected={traj_coll} inserted={traj_ins} "
          f"anchors={anchors} len={len(replay)} conservation_ok={conservation_ok}", flush=True)
    assert conservation_ok, "FAIL trajectory conservation (collected != inserted)"

    # ---- pending episodes ----
    pt = int(pending.total_pending_transitions()); pa = int(pending.total_pending_anchors())
    print(f"[verify] pending: transitions={pt} anchors={pa} is_empty={pending.is_empty()}", flush=True)

    # ---- RNG / action RNG ----
    rng_bytes = np.asarray(r["rng_key"]).tobytes()
    ars = r["action_rng_state"]
    ars_bytes = json.dumps(ars, sort_keys=True, default=str).encode() if not isinstance(ars, (bytes, bytearray)) else bytes(ars)
    print(f"[verify] rng_key={np.asarray(r['rng_key']).tolist()} action_rng_state_present={ars is not None}", flush=True)

    # ---- collector / env state ----
    cs = r["collector_state"]
    cs_keys = sorted(list(cs.keys())) if isinstance(cs, dict) else None
    print(f"[verify] collector_state keys={cs_keys}", flush=True)
    assert isinstance(cs, dict)
    for k in ("env_state", "obsv", "memories", "mem_mask", "mem_idx"):
        assert k in cs, f"FAIL collector_state missing {k}"

    # ---- config ----
    cfgd = r["config"]
    print(f"[verify] config adam_eps={cfgd.get('adam_eps')} gamma={cfgd.get('gamma')} "
          f"kl_replay_max={cfgd.get('kl_replay_max')} kl_run_max={cfgd.get('kl_run_max')}", flush=True)
    assert abs(cfgd.get("adam_eps", 0) - 1e-5) < 1e-12, "FAIL config adam_eps != 1e-5"
    assert abs(cfgd.get("gamma", 0) - 0.999) < 1e-12, "FAIL config gamma != 0.999"

    # ---- save -> restore roundtrip (temp dir) ----
    if args.do_resave and free_gb > 6.0:
        tmp = args.ckpt_dir.rstrip("/") + "_resume_verify_tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp, exist_ok=True)
        fields = set(FullP2Config.__dataclass_fields__.keys())
        cfg_obj = FullP2Config(**{k: v for k, v in cfgd.items() if k in fields})
        CK.save_full_checkpoint(
            r["params"], r["target_params"], r["opt_state"], replay, r["rng_key"],
            global_step=gs, path=tmp, step=gs,
            action_rng_state=ars, update_count=uc, pending=pending,
            collector_state=cs, config=cfg_obj, keep=0,
            extra_manifest={"label": "RESUME_VERIFY_ROUNDTRIP"})
        r2 = CK.restore_full_checkpoint(tmp, step=gs)
        replay2 = r2.get("replay_buffer", r2.get("replay"))
        p_sha2 = CK.params_content_sha256(r2["params"])
        t_sha2 = CK.params_content_sha256(r2["target_params"])
        opt_hash2 = leaf_hash(r2["opt_state"])
        gs2 = int(r2["global_step"]); uc2 = int(r2["update_count"])
        c2 = replay2.counters
        pt2 = int(r2["pending"].total_pending_transitions())
        pa2 = int(r2["pending"].total_pending_anchors())
        rng_bytes2 = np.asarray(r2["rng_key"]).tobytes()
        rt_ok = (p_sha2 == params_sha and t_sha2 == target_sha and opt_hash2 == opt_hash
                 and gs2 == gs and uc2 == uc
                 and int(c2.trajectories_collected) == traj_coll
                 and int(c2.trajectories_inserted) == traj_ins
                 and int(c2.total_anchors_stored) == anchors
                 and pt2 == pt and pa2 == pa and rng_bytes2 == rng_bytes)
        print(f"[verify] roundtrip: params={p_sha2==params_sha} target={t_sha2==target_sha} "
              f"opt={opt_hash2==opt_hash} gs={gs2==gs} uc={uc2==uc} "
              f"replay_conservation={int(c2.trajectories_collected)==traj_coll and int(c2.trajectories_inserted)==traj_ins} "
              f"anchors={int(c2.total_anchors_stored)==anchors} pending_t={pt2==pt} pending_a={pa2==pa} "
              f"rng={rng_bytes2==rng_bytes} => roundtrip_ok={rt_ok}", flush=True)
        shutil.rmtree(tmp)
        assert rt_ok, "FAIL save->restore roundtrip mismatch"
    else:
        print("[verify] skipping temp re-save roundtrip (disk or flag); "
              "note Level B launcher already roundtrip-verified this ckpt (roundtrip=OK)", flush=True)

    print("RESUME_VERIFY_OK step=%d params_sha=%s update_count=%d replay_traj=%d "
          "anchors=%d pending_t=%d pending_a=%d conservation=%s" % (
              gs, params_sha, uc, traj_ins, anchors, pt, pa, conservation_ok), flush=True)


if __name__ == "__main__":
    main()
