#!/usr/bin/env python3
"""POSTHOC_REPLAY_ATTRIBUTION_AND_HENRY_GAP_AUDIT — offline, READ-ONLY attribution.

Processes ONE P2-Full-A-v1 full-state checkpoint and emits a JSON with §5-§8 metrics.
NO training, NO optimizer commit, NO env rollout, NO re-collection, NO algorithm change.
GPU0 only (UUID bound). All candidate updates are computed in-memory and DISCARDED; a
read-only SHA bundle (params/target/opt/replay/pending/rng) is asserted identical before
and after the whole analysis to prove the checkpoint state is untouched.

TWO deterministic replay batches are analysed independently for §5/§6/§7:
  * batch_fixed  — fail[:half]+succ[:half] by ASCENDING trajectory_id (the oldest traj).
                   Primary, backward-compatible with earlier runs.
  * batch_recent — fail[:half]+succ[:half] by ASCENDING policy-lag (the NEWEST traj).
                   Required because the hindsight/AWR path is gated by
                   `lag <= max_policy_lag` (full_p2_learner.compute_loss L186-187):
                   at later checkpoints the OLDEST trajectories (batch_fixed) exceed the
                   lag gate, so the AWR/hindsight gradient (C/D) is masked to zero on
                   batch_fixed even though it is alive in training on fresh samples.
                   batch_recent makes the AWR path estimable at every checkpoint. It is
                   reported INDEPENDENTLY and never replaces batch_fixed.

Sections:
  §5 gradient attribution: per-component (A vtrace-actor, B vtrace-value, C awr-actor,
     D hindsight-value, E full) loss / global+module grad norms / nonzero count / finite;
     grad cosines.
  §6 candidate one-step-update drift: 7 read-only candidates -> one optimizer step (SAME
     opt_state) -> forward KL, entropy/logit/value change, per-module param-delta; discard.
  §7 off-policy & trajectory quality: IS ratio percentiles + ESS; policy-lag; per-sample
     actor-grad; success vs failure split; hindsight relabel-goal distribution; AWR weights;
     FIFO eviction accounting.
  §8 long-context gap audit: A/B/C/D memory ablation on DEEP windows (start>=128) in LONG
     episodes. A=real anchor+full re-burn-in, B=zero memory, C=last-128 from scratch,
     D=384-step burn-in (window caps at 128). All four share the IDENTICAL scan_fn forward;
     only the entering (memory,mask,idx) differs.

Run 4x (steps 24576/49152/73728/98304). Synthesis + §9 ruling + §10 reports done by the CC.
"""
import os
GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID            # MUST precede jax import
import sys, argparse, hashlib, json, copy
import numpy as np

BASE_SRC = "/home/oseasy/experiments/p2_full_20260723/src"
HENRY_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
if BASE_SRC in sys.path:
    sys.path.remove(BASE_SRC)
sys.path.insert(0, BASE_SRC)
if HENRY_SRC not in sys.path:
    sys.path.insert(1, HENRY_SRC)

import jax
import jax.numpy as jnp
import optax

import compat_init as CI
import hindsight as H
import memory_anchor as MA
import awr as A
import vtrace as V
import full_p2_learner as FL
import checkpointing as CK
from full_p2_learner import FullP2Config, build_optimizer

L_SEQ = 129                      # training loss-window length (run_p2_full_levelB.L_SEQ)
TRAIN_LR = 2e-5                  # training lr (run_p2_full_levelB.LEVELB_LR)

# ----------------------------- module partition -----------------------------
VALUE_TOKENS = ("critic_ln1", "critic_ln2", "critic_out")
ACTOR_TOKENS = ("actor_ln1", "actor_ln2", "actor_out")


def module_of(path_str):
    if any(t in path_str for t in VALUE_TOKENS):
        return "value_head"
    if any(t in path_str for t in ACTOR_TOKENS):
        return "actor_head"
    if "encoder" in path_str:
        return "encoder"
    return "trunk"


def build_module_index(params):
    pl = jax.tree_util.tree_leaves_with_path(params)
    path_strs, modules, sizes = [], [], []
    for kp, leaf in pl:
        ps = FL._path_str(kp)
        path_strs.append(ps)
        modules.append(module_of(ps))
        sizes.append(int(np.asarray(leaf).size))
    return path_strs, modules, sizes


def module_select(modules, want):
    if want == "all":
        return [True] * len(modules)
    if want == "shared_trunk":
        return [m in ("encoder", "trunk") for m in modules]
    return [m == want for m in modules]


def grad_module_norms(g, modules):
    leaves = jax.tree_util.tree_leaves(g)
    out = {}
    for name in ("encoder", "trunk", "actor_head", "value_head", "shared_trunk", "all"):
        sel = module_select(modules, name)
        ss = 0.0
        for lv, keep in zip(leaves, sel):
            if keep:
                ss += float(np.sum(np.square(np.asarray(lv, dtype=np.float64))))
        out[name] = float(np.sqrt(ss))
    return out


def grad_nonzero(g):
    leaves = jax.tree_util.tree_leaves(g)
    nz_leaves = 0
    nz_elems = 0
    finite = True
    for lv in leaves:
        a = np.asarray(lv)
        if not np.all(np.isfinite(a)):
            finite = False
        n = int(np.count_nonzero(a))
        if n > 0:
            nz_leaves += 1
        nz_elems += n
    return nz_leaves, nz_elems, finite


def flat_vec(g, sel):
    leaves = jax.tree_util.tree_leaves(g)
    parts = [np.asarray(lv, dtype=np.float64).ravel() for lv, keep in zip(leaves, sel) if keep]
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def cosine(g1, g2, sel):
    a = flat_vec(g1, sel)
    b = flat_vec(g2, sel)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def tree_add_scaled(pairs):
    acc = None
    for s, pt in pairs:
        acc = (jax.tree_util.tree_map(lambda x: s * x, pt) if acc is None
               else jax.tree_util.tree_map(lambda a, x: a + s * x, acc, pt))
    return acc


def delta_module_norms(p1, p2, modules):
    l1 = jax.tree_util.tree_leaves(p1)
    l2 = jax.tree_util.tree_leaves(p2)
    out = {}
    for name in ("encoder", "trunk", "actor_head", "value_head", "shared_trunk", "all"):
        sel = module_select(modules, name)
        ss = 0.0
        for a, b, keep in zip(l1, l2, sel):
            if keep:
                d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
                ss += float(np.sum(np.square(d)))
        out[name] = float(np.sqrt(ss))
    return out


# ----------------------------- read-only SHA bundle -----------------------------

def _tree_hash(tree):
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        a = np.asarray(leaf)
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _bytes_hash(b):
    return hashlib.sha256(b).hexdigest()


def sha_bundle(params, target, opt_state, buf, pending, rng_key, action_rng_state):
    pend_repr = ""
    if pending is not None:
        sd = pending.state_dict()
        pend_repr = json.dumps({
            "num_envs": sd["num_envs"], "next_episode_id": sd["next_episode_id"],
            "slot_lens": [len(s["obs"]) for s in sd["slots"]],
            "anchor_lens": [len(s["anchor_mem"]) for s in sd["slots"]],
            "episode_id": sd["episode_id"], "policy_version": sd["policy_version"],
        }, sort_keys=True, default=str)
    bundle = {
        "params": CK.params_content_sha256(params),
        "target": CK.params_content_sha256(target),
        "opt_state": _tree_hash(opt_state),
        "replay": buf.hash_digest(),
        "pending": _bytes_hash(pend_repr.encode()),
        "rng_key": _bytes_hash(np.asarray(rng_key).tobytes()),
        "action_rng_state": _bytes_hash(json.dumps(action_rng_state, sort_keys=True, default=str).encode()),
    }
    combined = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
    return combined, bundle


def pct(arr, qs=(50, 90, 95, 99)):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        d = {f"p{q}": float("nan") for q in qs}; d.update({"max": float("nan"), "n": 0})
        return d
    out = {f"p{q}": float(np.percentile(arr, q)) for q in qs}
    out["max"] = float(np.max(arr)); out["n"] = int(arr.size)
    return out


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--expected_sha", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_target", type=int, default=6)
    args = ap.parse_args()

    print(f"[posthoc] step={args.step} ckpt_root={args.ckpt_root}", flush=True)
    cfg = FullP2Config()
    assert cfg.window_mem == 128 and cfg.num_heads == 8 and cfg.num_layers == 2
    wm, heads, layers, emb = cfg.window_mem, cfg.num_heads, cfg.num_layers, cfg.embed

    network = CI.build_network()
    a_rec, a_raw, scan_fn = CI.build_callables(network)
    print("[posthoc] network + callables built (window_mem=%d)" % cfg.window_mem, flush=True)

    rd = CK.restore_full_checkpoint(args.ckpt_root, step=args.step)
    params = rd["params"]; target = rd["target_params"]; opt_state = rd["opt_state"]
    buf = rd["replay_buffer"]; pending = rd["pending"]
    rng_key = rd["rng_key"]; action_rng_state = rd["action_rng_state"]
    update_count = int(rd["update_count"]); manifest = rd["manifest"]
    src_sha = CK.params_content_sha256(params)
    print(f"[posthoc] restored params_sha={src_sha[:16]} update_count={update_count} "
          f"replay_size={len(buf)} inserted={buf.counters.trajectories_inserted}", flush=True)
    assert src_sha == args.expected_sha, f"HARD STOP source-mismatch {src_sha} != {args.expected_sha}"
    assert manifest.get("params_sha256", src_sha) == src_sha, "manifest params_sha mismatch"

    opt = build_optimizer(TRAIN_LR, cfg)
    n_params = int(sum(np.asarray(l).size for l in jax.tree_util.tree_leaves(params)))
    assert n_params == 4906028, n_params

    path_strs, modules, sizes = build_module_index(params)
    mod_counts = {}
    for m, s in zip(modules, sizes):
        mod_counts[m] = mod_counts.get(m, 0) + s
    print(f"[posthoc] module param counts: {mod_counts} (total={n_params})", flush=True)

    # ---- buffer inventory + DK index ----
    trajs = buf._buffer
    assert len(trajs) > 0, "empty replay buffer"
    print("[posthoc] first trajectory_ids:",
          [(t.trajectory_id, t.length) for t in trajs[:5]], flush=True)
    n_ach = int(trajs[0].target_achievements.shape[0])
    DK_idx = int(np.argmax(trajs[0].target_achievements))
    inv = []
    for i, t in enumerate(trajs):
        ag = t.achieved_goals()
        tid = t.trajectory_id if t.trajectory_id is not None else i
        inv.append(dict(
            buf_index=i, trajectory_id=int(tid), length=int(t.length),
            collected_update_count=int(t.collected_update_count),
            lag=int(update_count - t.collected_update_count),
            n_anchors=int(t.n_anchors),
            DK_success=bool(ag[DK_idx] > 0),
            n_achievements=int((ag > 0).sum()),
            achieved_indices=[int(x) for x in np.where(ag > 0)[0]],
            reward_sum=float(np.sum(t.rewards)),
        ))
    inv_by_id = {x["trajectory_id"]: x["buf_index"] for x in inv}
    n_success = sum(1 for x in inv if x["DK_success"])
    n_failure = len(inv) - n_success
    lengths = np.array([x["length"] for x in inv], dtype=np.float64)
    print(f"[posthoc] buffer n_traj={len(inv)} DK_success={n_success} failure={n_failure} "
          f"n_ach={n_ach} DK_idx={DK_idx} longest={int(lengths.max())}", flush=True)

    counters_snapshot = copy.deepcopy(buf.counters)

    # ================= BATCH SELECTION (two deterministic batches) =================
    def build_batch(chosen):
        """Turn chosen inventory records into relabelled (orig, rel) sample windows."""
        so, sr, bm, sk = [], [], [], []
        for c in chosen:
            t = trajs[c["buf_index"]]
            if t.length < L_SEQ:
                sk.append(dict(trajectory_id=c["trajectory_id"], status="TOO_SHORT",
                               length=c["length"], DK_success=c["DK_success"]))
                continue
            ach_per_step = t.achievements.max(axis=1) > 0
            first_ach = int(np.argmax(ach_per_step)) if ach_per_step.any() else 0
            max_start = t.length - L_SEQ
            start0 = int(np.clip(first_ach - 64, 0, max_start)) if max_start > 0 else 0
            s = rel = None
            tried = []
            for cand_start in (start0, 0, max_start):
                cs = int(np.clip(cand_start, 0, max_start))
                tried.append(cs)
                try:
                    ss = buf.sample(sequence_length=L_SEQ, trajectory_id=t.trajectory_id, start_step=cs)
                    rr = H.relabel_sample(ss)
                    s, rel, start0 = ss, rr, cs
                    break
                except (ValueError, RuntimeError):
                    continue
            if s is None:
                sk.append(dict(trajectory_id=c["trajectory_id"], status="UNRELABEABLE",
                               tried_starts=tried, DK_success=c["DK_success"]))
                continue
            rel_goal = int(np.argmax(rel.target_achievements))
            so.append(s); sr.append(rel)
            bm.append(dict(
                trajectory_id=int(t.trajectory_id), status="OK", start_step=int(start0),
                length=int(s.length), burn_in_length=int(s.burn_in_length),
                pre_anchor_step=int(s.pre_anchor_step),
                collected_update_count=int(s.collected_update_count),
                lag=int(update_count - s.collected_update_count),
                DK_success=c["DK_success"], source_length=c["length"],
                in_window_achieved=[int(x) for x in np.where(s.achievements.max(axis=0) > 0)[0]],
                relabel_goal_index=rel_goal, relabel_goal_is_DK=bool(rel_goal == DK_idx),
                reward_sum_orig=float(np.sum(s.rewards)), reward_sum_rel=float(np.sum(rel.rewards)),
                tried_starts=tried,
            ))
        return so, sr, bm, sk

    half = max(1, args.k_target // 2)
    # batch_fixed: oldest trajectories (ascending trajectory_id)
    succ_id = sorted([x for x in inv if x["DK_success"]], key=lambda x: x["trajectory_id"])
    fail_id = sorted([x for x in inv if not x["DK_success"]], key=lambda x: x["trajectory_id"])
    chosen_fixed = fail_id[:half] + succ_id[:half]
    if len(chosen_fixed) < args.k_target:
        have = {c["trajectory_id"] for c in chosen_fixed}
        for x in sorted(inv, key=lambda x: x["trajectory_id"]):
            if x["trajectory_id"] not in have and len(chosen_fixed) < args.k_target:
                chosen_fixed.append(x); have.add(x["trajectory_id"])
    chosen_fixed = sorted(chosen_fixed, key=lambda x: x["trajectory_id"])
    so_f, sr_f, bm_f, sk_f = build_batch(chosen_fixed)
    print(f"[posthoc] batch_fixed B={len(so_f)} (chosen={len(chosen_fixed)}, skipped={len(sk_f)}) "
          f"lags={[m['lag'] for m in bm_f]}", flush=True)

    # batch_recent: NEWEST trajectories (ascending policy-lag) -> AWR path estimable (lag<=16)
    succ_lag = sorted([x for x in inv if x["DK_success"]], key=lambda x: (x["lag"], x["trajectory_id"]))
    fail_lag = sorted([x for x in inv if not x["DK_success"]], key=lambda x: (x["lag"], x["trajectory_id"]))
    chosen_recent = fail_lag[:half] + succ_lag[:half]
    if len(chosen_recent) < args.k_target:
        have = {c["trajectory_id"] for c in chosen_recent}
        for x in sorted(inv, key=lambda x: (x["lag"], x["trajectory_id"])):
            if x["trajectory_id"] not in have and len(chosen_recent) < args.k_target:
                chosen_recent.append(x); have.add(x["trajectory_id"])
    chosen_recent = sorted(chosen_recent, key=lambda x: x["trajectory_id"])
    so_r, sr_r, bm_r, sk_r = build_batch(chosen_recent)
    print(f"[posthoc] batch_recent B={len(so_r)} (chosen={len(chosen_recent)}, skipped={len(sk_r)}) "
          f"lags={[m['lag'] for m in bm_r]}", flush=True)

    # ---- §8 dedicated windows: DEEP windows in LONG trajectories (no relabel needed) ----
    samples8, manifest8 = [], []
    long_candidates = sorted([x for x in inv if x["length"] >= 300], key=lambda x: -x["length"])
    def add_window8(tid, start, src):
        s = buf.sample(sequence_length=L_SEQ, trajectory_id=tid, start_step=int(start))
        samples8.append(s)
        manifest8.append(dict(trajectory_id=int(tid), start_step=int(start),
                              source_length=int(src["length"]), DK_success=bool(src["DK_success"]),
                              burn_in_length=int(s.burn_in_length),
                              pre_anchor_step=int(s.pre_anchor_step)))
    for c in long_candidates:
        if len(samples8) >= 6:
            break
        t = trajs[c["buf_index"]]; max_start = t.length - L_SEQ
        if max_start < 128:
            continue
        deep = max(128, min(max_start, 384))
        add_window8(t.trajectory_id, deep, c)
    if len(samples8) < 2 and long_candidates:
        c = long_candidates[0]; t = trajs[c["buf_index"]]; max_start = t.length - L_SEQ
        have_starts = {m["start_step"] for m in manifest8}
        for d in (768, 640, 512, 384, 256, 128, max_start):
            dd = int(min(max_start, d))
            if dd >= 128 and dd not in have_starts:
                add_window8(t.trajectory_id, dd, c); break
    B8 = len(samples8)
    print(f"[posthoc] §8 deep windows B8={B8} starts={[m['start_step'] for m in manifest8]} "
          f"lens={[m['source_length'] for m in manifest8]}", flush=True)
    assert B8 >= 2, "need >=2 deep windows for §8 (lax.scan B>=2)"

    buf.counters = counters_snapshot   # all sample() calls done; reset for clean SHA

    # ================= BUFFER-LEVEL §7 STATS (computed once, shared) =================
    buf_relabel_goals = []
    for t in trajs:
        ach_idx = set(np.where(t.achievements.max(axis=0) > 0)[0].tolist())
        if ach_idx:
            buf_relabel_goals.append(int(min(ach_idx)))
    buf_lags = np.array([x["lag"] for x in inv], dtype=np.float64)
    n_inserted = int(buf.counters.trajectories_inserted)
    n_current = len(buf); n_evicted = max(0, n_inserted - n_current)
    inv_sorted = sorted(inv, key=lambda x: x["trajectory_id"])
    half_buf = max(1, n_current // 2)
    def grp_stats(grp):
        if not grp:
            return dict(n=0)
        ls = np.array([x["length"] for x in grp], dtype=np.float64)
        lags = np.array([x["lag"] for x in grp], dtype=np.float64)
        return dict(n=len(grp),
                    DK_success_frac=float(np.mean([x["DK_success"] for x in grp])),
                    mean_length=float(ls.mean()),
                    mean_n_achievements=float(np.mean([x["n_achievements"] for x in grp])),
                    mean_lag=float(lags.mean()),
                    traj_ids=[x["trajectory_id"] for x in grp])
    sec7_buffer = dict(
        DK_index=DK_idx, n_achievements_dim=n_ach,
        policy_lag_buffer_percentiles=pct(buf_lags),
        policy_lag_buffer_min=int(buf_lags.min()), policy_lag_buffer_max=int(buf_lags.max()),
        max_policy_lag_cfg=int(cfg.max_policy_lag),
        n_success_trajectories=int(n_success), n_failure_trajectories=int(n_failure),
        success_failure_ratio_in_replay=float(n_success / max(1, len(inv))),
        relabel_goal_distribution_buffer={int(g): buf_relabel_goals.count(g) for g in set(buf_relabel_goals)},
        relabel_goal_is_DK_count_buffer=int(sum(1 for g in buf_relabel_goals if g == DK_idx)),
        fifo=dict(trajectories_inserted=n_inserted, current_size=n_current, n_evicted=n_evicted,
                  capacity=int(buf.capacity),
                  oldest_group=grp_stats(inv_sorted[:half_buf]),
                  newest_group=grp_stats(inv_sorted[-half_buf:])),
    )

    # ================= READ-ONLY SHA BEFORE =================
    sha_before, bundle_before = sha_bundle(params, target, opt_state, buf, pending, rng_key, action_rng_state)
    print(f"[posthoc] SHA_BEFORE={sha_before[:16]}", flush=True)

    # ================= §5/§6/§7 ANALYSIS (per batch) =================
    def analyze_batch(samples_orig, samples_rel, batch_manifest, skipped, tag):
        B = len(samples_orig)
        assert B >= 2, f"{tag}: need >=2 relabelable windows (lax.scan B>=2)"
        po = FL.pack_batch(samples_orig)
        pr = FL.pack_batch(samples_rel)
        obs_o_ext = FL._ext_obs(po)
        obs_r_ext = FL._ext_obs(pr)
        L = po["observations"].shape[1]
        assert L == L_SEQ, (L, L_SEQ)

        recon_o = FL.reconstruct_batch(a_rec, params, samples_orig, cfg)
        recon_r = FL.reconstruct_batch(a_rec, params, samples_rel, cfg)
        recon_o_t = FL.reconstruct_batch(a_rec, target, samples_orig, cfg)
        recon_r_t = FL.reconstruct_batch(a_rec, target, samples_rel, cfg)
        target_vals_o = FL._target_scan(scan_fn, target, recon_o_t, obs_o_ext)
        target_vals_r = FL._target_scan(scan_fn, target, recon_r_t, obs_r_ext)

        def compute(p):
            return FL.compute_loss(p, a_raw, po, pr, obs_o_ext, obs_r_ext,
                                   target_vals_o, target_vals_r, recon_o, recon_r, cfg, update_count)

        loss_E, mvals = compute(params)
        comp_loss = {"A": float(mvals["vtrace_actor"]), "B": float(mvals["vtrace_value"]),
                     "C": float(mvals["awr_actor"]), "D": float(mvals["awr_value"]), "E": float(loss_E)}
        print(f"[posthoc] [{tag}] component losses: {comp_loss}", flush=True)

        def scalar_for(p, key):
            loss, m = compute(p)
            return {"A": m["vtrace_actor"], "B": m["vtrace_value"], "C": m["awr_actor"],
                    "D": m["awr_value"], "E": loss}[key]
        grads = {}
        for key in ("A", "B", "C", "D", "E"):
            gf = jax.jit(jax.grad(lambda p, k=key: scalar_for(p, k)))
            grads[key] = gf(params)
            jax.block_until_ready(jax.tree_util.tree_leaves(grads[key])[0])

        sec5 = {"component_losses": comp_loss,
                "metrics": {k: (float(v) if np.ndim(v) == 0 else v) for k, v in mvals.items()},
                "components": {}}
        for key in ("A", "B", "C", "D", "E"):
            g = grads[key]
            nz_l, nz_e, finite = grad_nonzero(g)
            sec5["components"][key] = dict(
                loss=comp_loss[key],
                global_grad_norm=float(np.asarray(optax.global_norm(g))),
                module_grad_norms=grad_module_norms(g, modules),
                nonzero_grad_leaves=int(nz_l), nonzero_grad_elements=int(nz_e),
                n_leaves=len(jax.tree_util.tree_leaves(g)), finite=bool(finite))
        sel_all = module_select(modules, "all")
        sel_st = module_select(modules, "shared_trunk")
        actor_total = tree_add_scaled([(0.5, grads["A"]), (0.5, grads["C"])])
        value_total = tree_add_scaled([(0.25, grads["B"]), (0.25, grads["D"])])
        sec5["grad_cosines"] = dict(
            vtraceActor_vs_awrActor_all=cosine(grads["A"], grads["C"], sel_all),
            vtraceActor_vs_vtraceValue_sharedTrunk=cosine(grads["A"], grads["B"], sel_st),
            awrActor_vs_hindsightValue_sharedTrunk=cosine(grads["C"], grads["D"], sel_st),
            replayActorTotal_vs_replayValueTotal_all=cosine(actor_total, value_total, sel_all),
            vtraceActor_vs_full_all=cosine(grads["A"], grads["E"], sel_all),
            vtraceValue_vs_full_all=cosine(grads["B"], grads["E"], sel_all),
            awrActor_vs_full_all=cosine(grads["C"], grads["E"], sel_all),
            hindsightValue_vs_full_all=cosine(grads["D"], grads["E"], sel_all),
            actorTotal_vs_full_all=cosine(actor_total, grads["E"], sel_all),
            valueTotal_vs_full_all=cosine(value_total, grads["E"], sel_all))
        sec5["module_param_counts"] = mod_counts
        if tag == "fixed":
            sec5["module_leaf_paths"] = {m: [p for p, mm in zip(path_strs, modules) if mm == m]
                                         for m in ("encoder", "trunk", "actor_head", "value_head")}
        print(f"[posthoc] [{tag}] §5 done cosines={sec5['grad_cosines']}", flush=True)

        # ---- §6 candidate one-step-update drift ----
        lg_cur, val_cur, _, _, _ = scan_fn(params, *recon_o, obs_o_ext)
        lg_cur_w = jnp.transpose(lg_cur[:L], (1, 0, 2))
        val_cur_w = jnp.transpose(val_cur[:L], (1, 0))
        lp_cur = FL._log_softmax(lg_cur_w)
        p_cur = jax.nn.softmax(lg_cur_w, axis=-1)
        actions_t = po["actions"].T
        _, ent_cur = FL._log_pi_and_entropy(lg_cur[:L], actions_t)
        ent_cur_w = jnp.transpose(ent_cur, (1, 0))

        cand_specs = [("1_vtrace_actor_only", grads["A"]), ("2_awr_actor_only", grads["C"]),
                      ("3_vtrace_value_only", grads["B"]), ("4_hindsight_value_only", grads["D"]),
                      ("5_actors_merged", actor_total), ("6_values_merged", value_total),
                      ("7_full_combined", grads["E"])]
        sec6 = {"candidates": {}, "probe_window": {"B": int(B), "L": int(L), "source": "original-goal"}}
        for name, g in cand_specs:
            updates, _new_opt = opt.update(g, opt_state, params)
            cand = optax.apply_updates(params, updates)
            kl = float(np.asarray(FL._policy_kl_window(scan_fn, cand, params, recon_o, obs_o_ext, L)))
            lg_c, val_c, _, _, _ = scan_fn(cand, *recon_o, obs_o_ext)
            lg_c_w = jnp.transpose(lg_c[:L], (1, 0, 2)); val_c_w = jnp.transpose(val_c[:L], (1, 0))
            _, ent_c = FL._log_pi_and_entropy(lg_c[:L], actions_t)
            ent_c_w = jnp.transpose(ent_c, (1, 0))
            logit_d = np.asarray(lg_c_w) - np.asarray(lg_cur_w)
            value_d = np.asarray(val_c_w) - np.asarray(val_cur_w)
            sec6["candidates"][name] = dict(
                grad_global_norm=float(np.asarray(optax.global_norm(g))),
                forward_kl=kl,
                entropy_cur=float(np.asarray(ent_cur_w.mean())),
                entropy_cand=float(np.asarray(ent_c_w.mean())),
                entropy_change=float(np.asarray((ent_c_w - ent_cur_w).mean())),
                logit_change_mean_abs=float(np.abs(logit_d).mean()),
                logit_change_l2=float(np.linalg.norm(logit_d)),
                value_change_mean_abs=float(np.abs(value_d).mean()),
                value_change_l2=float(np.linalg.norm(value_d)),
                param_delta_norms=delta_module_norms(cand, params, modules))
            del cand, updates, _new_opt, lg_c, val_c
        print(f"[posthoc] [{tag}] §6 done candidate KLs="
              f"{ {k: round(v['forward_kl'],5) for k,v in sec6['candidates'].items()} }", flush=True)

        # ---- §7 off-policy & trajectory quality ----
        vt_diag = FL.diagnose_vtrace(params, target, scan_fn, a_rec, samples_orig, cfg)
        ratio = np.asarray(vt_diag["ratio"])
        log_pi = np.asarray(vt_diag["log_pi"]); log_mu = np.asarray(vt_diag["log_mu"])
        lag_per_sample = np.array([update_count - s.collected_update_count for s in samples_orig])
        mean_ratio_per_sample = ratio.mean(axis=1)
        ess_frac = float(vt_diag["ess"])

        def persample_actor_grads(p, b_idx):
            valid = jnp.zeros((B, L)).at[b_idx].set(1.0)
            actions_tt = po["actions"].T
            lg_o, val_o, _, _, _ = FL._scan_lax(a_raw, p, *recon_o, obs_o_ext, cfg)
            v_online = val_o[:L].T
            log_pi_o, ent_o = FL._log_pi_and_entropy(lg_o[:L], actions_tt)
            log_pi_o, ent_o = log_pi_o.T, ent_o.T
            v_target_tp1 = target_vals_o[:, 1:]
            boot_o = jnp.where(po["terminal"] > 0.5, 0.0, target_vals_o[:, L])
            vt_cfg = V.VtraceConfig(cfg.gamma, cfg.rho_bar, cfg.c_bar, cfg.vt_clip_min, cfg.vt_clip_max)
            vt = V.vtrace_targets(log_pi_o, po["log_probs"], v_online, v_target_tp1,
                                  po["rewards"], po["dones"], boot_o, vt_cfg)
            vt_aloss = V.vtrace_actor_loss(vt, log_pi_o, ent_o, valid, cfg.ent_coef)
            lg_r, val_r, _, _, _ = FL._scan_lax(a_raw, p, *recon_r, obs_r_ext, cfg)
            logits_rel = jnp.transpose(lg_r[:L], (1, 0, 2))
            v_rel_online = val_r[:L].T
            logits_before = jax.lax.stop_gradient(logits_rel)
            target_rel = target_vals_r[:, :L]
            boot_r = jnp.where(pr["terminal"] > 0.5, 0.0, target_vals_r[:, L])
            lag = jnp.maximum(update_count - pr["lag"], 0)
            awr_valid = valid * (lag <= cfg.max_policy_lag).astype(jnp.float32)[:, None]
            awr_cfg = A.AWRConfig(cfg.gamma, cfg.beta, cfg.w_max, cfg.lambda_kl,
                                  cfg.vt_clip_min, cfg.vt_clip_max)
            awr = A.awr_losses(logits_rel, logits_before, po["actions"], v_rel_online,
                               target_rel, pr["rewards"], pr["dones"], boot_r, awr_valid, awr_cfg)
            return vt_aloss, awr.actor_loss, awr.weights

        per_sample = []
        awr_weights_all = []
        gf_v = jax.jit(jax.grad(lambda p, bi: persample_actor_grads(p, bi)[0]))
        gf_a = jax.jit(jax.grad(lambda p, bi: persample_actor_grads(p, bi)[1]))
        for b in range(B):
            gv = gf_v(params, b); ga = gf_a(params, b)
            _, _, w_all = persample_actor_grads(params, b)
            w_b = np.asarray(w_all)[b]
            awr_weights_all.append(w_b)
            per_sample.append(dict(
                sample_idx=int(b), trajectory_id=int(samples_orig[b].source_trajectory_id),
                DK_success=bool(batch_manifest[b]["DK_success"]),
                lag=int(lag_per_sample[b]),
                mean_ratio=float(mean_ratio_per_sample[b]), max_ratio=float(ratio[b].max()),
                vtrace_actor_grad_global_norm=float(np.asarray(optax.global_norm(gv))),
                awr_actor_grad_global_norm=float(np.asarray(optax.global_norm(ga))),
                awr_weight_mean=float(np.mean(w_b)), awr_weight_max=float(np.max(w_b))))
        awr_weights_all = np.concatenate(awr_weights_all)

        upd7, _ = opt.update(grads["E"], opt_state, params)
        cand7 = optax.apply_updates(params, upd7)
        lg7, _, _, _, _ = scan_fn(cand7, *recon_o, obs_o_ext)
        lg7_w = jnp.transpose(lg7[:L], (1, 0, 2))
        lp7 = FL._log_softmax(lg7_w); p7 = jax.nn.softmax(lg7_w, axis=-1)
        persample_kl7 = np.asarray((p7 * (lp7 - lp_cur)).sum(axis=-1).mean(axis=-1))
        del cand7, upd7

        def split_by_success(vals):
            sv = [v for v, m in zip(vals, batch_manifest) if m["DK_success"]]
            fv = [v for v, m in zip(vals, batch_manifest) if not m["DK_success"]]
            return (float(np.sum(sv)) if sv else 0.0, float(np.sum(fv)) if fv else 0.0,
                    len(sv), len(fv))
        vt_g_succ, vt_g_fail, ns, nf = split_by_success([ps["vtrace_actor_grad_global_norm"] for ps in per_sample])
        awr_g_succ, awr_g_fail, _, _ = split_by_success([ps["awr_actor_grad_global_norm"] for ps in per_sample])
        kl7_succ, kl7_fail, _, _ = split_by_success(list(persample_kl7))
        awrw_succ_sum, awrw_fail_sum, _, _ = split_by_success([ps["awr_weight_mean"] for ps in per_sample])
        relabel_goals_batch = [m["relabel_goal_index"] for m in batch_manifest]

        sec7 = dict(
            batch_tag=tag,
            importance_ratio_percentiles=pct(ratio), rho_bar_percentiles=pct(np.minimum(1.0, ratio)),
            ess_fraction=ess_frac, mean_log_pi=float(np.mean(log_pi)), mean_log_mu=float(np.mean(log_mu)),
            policy_lag_batch=lag_per_sample.tolist(),
            per_sample=per_sample,
            n_success_samples=ns, n_failure_samples=nf,
            vtrace_actor_grad_sum_success=vt_g_succ, vtrace_actor_grad_sum_failure=vt_g_fail,
            awr_actor_grad_sum_success=awr_g_succ, awr_actor_grad_sum_failure=awr_g_fail,
            candidate7_kl_sum_success=kl7_succ, candidate7_kl_sum_failure=kl7_fail,
            awr_weight_percentiles=pct(awr_weights_all),
            awr_weight_mean_success_trajs=(awrw_succ_sum / ns if ns else float("nan")),
            awr_weight_mean_failure_trajs=(awrw_fail_sum / nf if nf else float("nan")),
            relabel_goal_distribution_batch={int(g): relabel_goals_batch.count(g) for g in set(relabel_goals_batch)},
            relabel_goal_is_DK_count_batch=int(sum(1 for g in relabel_goals_batch if g == DK_idx)),
            skipped_batch_records=skipped,
        )
        sec7.update(sec7_buffer)   # merge buffer-level fields (identical across batches)
        print(f"[posthoc] [{tag}] §7 done ratio_p95={sec7['importance_ratio_percentiles']['p95']:.3f} "
              f"ess={ess_frac:.3f} success_samples={ns}/{B} evicted={n_evicted}", flush=True)
        return sec5, sec6, sec7

    sec5, sec6, sec7 = analyze_batch(so_f, sr_f, bm_f, sk_f, "fixed")
    sec5_recent, sec6_recent, sec7_recent = analyze_batch(so_r, sr_r, bm_r, sk_r, "recent")

    # ================= §8 LONG-CONTEXT GAP AUDIT (deep windows) =================
    burn_lens = np.array([s.burn_in_length for s in samples8], dtype=np.float64)
    ep_len_succ = np.array([x["length"] for x in inv if x["DK_success"]], dtype=np.float64)
    ep_len_fail = np.array([x["length"] for x in inv if not x["DK_success"]], dtype=np.float64)
    def frac_over(arr, thr):
        arr = np.asarray(arr, dtype=np.float64)
        return float(np.mean(arr > thr)) if arr.size else float("nan")

    po8 = FL.pack_batch(samples8)
    obs8_ext = FL._ext_obs(po8)
    recon_o_8 = FL.reconstruct_batch(a_rec, params, samples8, cfg)
    lgA, valA, _, _, _ = scan_fn(params, *recon_o_8, obs8_ext)
    lgA_w = jnp.transpose(lgA[:L_SEQ], (1, 0, 2))
    valA_w = jnp.transpose(valA[:L_SEQ], (1, 0))

    def build_zero8(b, s):
        mem = jnp.zeros((1, wm, layers, emb), jnp.float32)
        mask = jnp.zeros((1, heads, 1, wm + 1), jnp.bool_)
        idx = jnp.full((1,), wm, jnp.int32)
        return mem, mask, idx
    def build_burnin8(n_back):
        def _b(b, s):
            mem, mask, idx = build_zero8(b, s)
            start = int(manifest8[b]["start_step"])
            t = trajs[inv_by_id[s.source_trajectory_id]]
            a = max(0, start - n_back)
            seg = np.asarray(t.observations[a:start], dtype=np.float32)
            if seg.shape[0] > 0:
                mem, mask, idx = MA.reconstruct_state(
                    a_rec, params, mem, mask, idx, jnp.asarray(seg)[:, None, :], wm, heads)
            return mem, mask, idx
        return _b
    def batched_entering8(build):
        mems, masks, idxs = [], [], []
        for b, s in enumerate(samples8):
            m, mk, ix = build(b, s)
            mems.append(m[0]); masks.append(mk[0]); idxs.append(ix[0])
        return jnp.stack(mems), jnp.stack(masks), jnp.stack(idxs)

    reconB = batched_entering8(build_zero8)
    reconC = batched_entering8(build_burnin8(128))
    reconD = batched_entering8(build_burnin8(384))
    lgB, vB, _, _, _ = scan_fn(params, *reconB, obs8_ext)
    lgC, vC, _, _, _ = scan_fn(params, *reconC, obs8_ext)
    lgD, vD, _, _, _ = scan_fn(params, *reconD, obs8_ext)

    def metrics_vs8(lgX, vX, b):
        a_slice = lgA_w[b]
        x_slice = jnp.transpose(lgX[:L_SEQ], (1, 0, 2))[b]
        lpA = FL._log_softmax(a_slice); pA = jax.nn.softmax(a_slice, axis=-1)
        lpX = FL._log_softmax(x_slice); pX = jax.nn.softmax(x_slice, axis=-1)
        kl = float(np.asarray((pX * (lpX - lpA)).sum(-1).mean()))
        flip = float(np.asarray((jnp.argmax(x_slice, -1) != jnp.argmax(a_slice, -1)).astype(jnp.float32).mean()))
        va = valA_w[b]; vx = jnp.transpose(vX[:L_SEQ], (1, 0))[b]
        vdiff = float(np.asarray(jnp.abs(vx - va).mean()))
        entA = -(pA * lpA).sum(-1); entX = -(pX * lpX).sum(-1)
        entdiff = float(np.asarray(jnp.abs(entX - entA).mean()))
        return dict(kl_vs_A=kl, top_action_flip_rate=flip, value_abs_diff=vdiff, entropy_abs_diff=entdiff)

    ablation = []
    for b, s in enumerate(samples8):
        m = manifest8[b]
        ablation.append(dict(
            trajectory_id=int(s.source_trajectory_id), start_step=int(m["start_step"]),
            source_length=int(m["source_length"]), burn_in_length=int(s.burn_in_length),
            pre_anchor_step=int(s.pre_anchor_step), DK_success=bool(m["DK_success"]),
            B_zero_memory=metrics_vs8(lgB, vB, b),
            C_last128_from_scratch=metrics_vs8(lgC, vC, b),
            D_longer_burnin384=metrics_vs8(lgD, vD, b)))
        a = ablation[-1]
        print(f"[posthoc] §8 b={b} traj={s.source_trajectory_id} start={m['start_step']} "
              f"B_kl={a['B_zero_memory']['kl_vs_A']:.4f} B_flip={a['B_zero_memory']['top_action_flip_rate']:.3f} "
              f"C_kl={a['C_last128_from_scratch']['kl_vs_A']:.4f} D_kl={a['D_longer_burnin384']['kl_vs_A']:.4f}",
              flush=True)
    def ablation_mean(cond):
        keys = ("kl_vs_A", "top_action_flip_rate", "value_abs_diff", "entropy_abs_diff")
        return {k: float(np.mean([a[cond][k] for a in ablation])) for k in keys} if ablation else {}

    sec8 = dict(
        window_mem=int(cfg.window_mem), window_grad_evalcfg=64,
        replay_sequence_length=int(L_SEQ), anchor_interval=128,
        n_deep_windows=B8, deep_window_manifest=manifest8,
        burn_in_length_percentiles=pct(burn_lens),
        episode_length_percentiles=pct(lengths),
        episode_length_success_percentiles=pct(ep_len_succ),
        episode_length_failure_percentiles=pct(ep_len_fail),
        frac_episodes_over_128=float(np.mean(lengths > 128)),
        frac_episodes_over_256=float(np.mean(lengths > 256)),
        frac_episodes_over_512=float(np.mean(lengths > 512)),
        frac_failure_episodes_over_128=frac_over(ep_len_fail, 128),
        frac_failure_episodes_over_256=frac_over(ep_len_fail, 256),
        frac_failure_episodes_over_512=frac_over(ep_len_fail, 512),
        ablation_per_sample=ablation,
        ablation_mean_B_zero_memory=ablation_mean("B_zero_memory"),
        ablation_mean_C_last128_from_scratch=ablation_mean("C_last128_from_scratch"),
        ablation_mean_D_longer_burnin384=ablation_mean("D_longer_burnin384"),
        note=("A=real anchor+full re-burn-in (training-exact forward); B=memory zeroed at window "
              "start; C=last-128 obs recovered from scratch; D=384-step burn-in from scratch "
              "(window caps at 128). Windows start DEEP (>=128) in long episodes so A carries a "
              "full memory window. All conditions share the IDENTICAL scan_fn loss-region forward; "
              "only the entering (memory,mask,idx) differs. EXPECTED if window_mem=128 caps "
              "history: A~C~D (all retain only last-128) and B differs (no history) -- proving the "
              "policy can use the 128-window but has NO mechanism for >128-step history (Henry's "
              "long-context structure unsolved). If A~B~C~D all ~0, the model is not using memory "
              "at all at this checkpoint."),
    )
    print(f"[posthoc] §8 done meanB_kl={sec8['ablation_mean_B_zero_memory'].get('kl_vs_A',float('nan')):.4f} "
          f"meanC_kl={sec8['ablation_mean_C_last128_from_scratch'].get('kl_vs_A',float('nan')):.4f} "
          f"meanD_kl={sec8['ablation_mean_D_longer_burnin384'].get('kl_vs_A',float('nan')):.4f}", flush=True)

    # ================= READ-ONLY SHA AFTER + VERIFY =================
    buf.counters = counters_snapshot
    sha_after, bundle_after = sha_bundle(params, target, opt_state, buf, pending, rng_key, action_rng_state)
    readonly_ok = (sha_before == sha_after)
    print(f"[posthoc] SHA_AFTER={sha_after[:16]} READONLY_OK={readonly_ok}", flush=True)
    assert readonly_ok, f"HARD STOP read-only violation before={sha_before} after={sha_after}"
    assert CK.params_content_sha256(params) == args.expected_sha, "params changed"

    result = dict(
        step=int(args.step), ckpt_root=args.ckpt_root,
        params_sha256=src_sha, expected_sha=args.expected_sha, update_count=update_count,
        n_params=n_params, train_lr=TRAIN_LR,
        config=dict(window_mem=cfg.window_mem, num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                    gamma=cfg.gamma, rho_bar=cfg.rho_bar, c_bar=cfg.c_bar, beta=cfg.beta,
                    w_max=cfg.w_max, w_vtrace=cfg.w_vtrace, w_awr=cfg.w_awr, vf_coef=cfg.vf_coef,
                    ent_coef=cfg.ent_coef, kl_replay_max=cfg.kl_replay_max,
                    max_policy_lag=cfg.max_policy_lag, grad_clip=cfg.grad_clip,
                    adam_eps=cfg.adam_eps, ema_tau=cfg.ema_tau),
        module_param_counts=mod_counts,
        buffer_inventory=inv,
        batch_manifest=bm_f, batch_manifest_recent=bm_r,
        batch_note=("batch_fixed=oldest trajectories (ascending trajectory_id); "
                    "batch_recent=newest trajectories (ascending policy-lag) so the hindsight/AWR "
                    "path (gated by lag<=max_policy_lag=%d) is estimable at every checkpoint. "
                    "sec5/sec6/sec7 use batch_fixed; sec5_recent/sec6_recent/sec7_recent use "
                    "batch_recent. Both are deterministic and read-only." % cfg.max_policy_lag),
        readonly_sha_before=sha_before, readonly_sha_after=sha_after, readonly_ok=readonly_ok,
        readonly_bundle_before=bundle_before, readonly_bundle_after=bundle_after,
        sec5_gradient_attribution=sec5, sec6_candidate_drift=sec6, sec7_offpolicy_quality=sec7,
        sec5_recent=sec5_recent, sec6_recent=sec6_recent, sec7_recent=sec7_recent,
        sec8_longcontext_audit=sec8,
    )
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    print(f"[posthoc] WROTE {args.out} POSTHOC_CKPT_OK step={args.step}", flush=True)


if __name__ == "__main__":
    main()
