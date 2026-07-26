#!/usr/bin/env python3
"""P2-v1 Level3 (24576-step) one-shot hard-gate verification (GPU0).

Read-only w.r.t. the real checkpoint (checkpoints/24576 restored + round-tripped
via a tmp re-save; the original is never modified). Verifies, in ONE pass:
  - exit/global_step=24576, params finite, opt_step/update_count consistency
  - conservation: completed(replay) + pending == 24576
  - 方案B presence + no cross-env concat + boundary-not-faked
  - round-trip bit-exact (params/opt/rng/global_step/update_count/replay/pending/
    collector/action_rng stream)
  - full training_log analysis: 12 PPO updates finite, replay-aux count, hindsight
    eligible/accepted, policy_lag/IS/ESS, critic aux loss+grad, episode length,
    NaN/Inf across ALL records
  - isolation re-confirm: manifest learner SHA == frozen 6689426b AND every aux
    record has actor_enabled=false & actor_loss=0
Writes a detailed report JSON (raw data) and prints a terse PASS/FAIL summary.
"""
import os, sys, json, glob, tempfile, shutil
import numpy as np

GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID

import jax, jax.numpy as jnp

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path:
        sys.path.insert(0, p)

import stage4_continue_launcher as L
from checkpointing import save_full_checkpoint
from pending_episodes import PendingEpisodeBuffers
from rng_utils import restore_action_rng, sample_actions

STEP = 24576
COLLECTED = 24576            # 16 envs x 128 rollout x 12 updates
NUM_ENVS = 16
OBS_DIM = 8335
ACTION_DIM = 43
FROZEN_LEARNER_SHA = "6689426b77bb030c8ce3a3a3c97ddab7bd0248d2eaa1d477146dff44ccf1c386"

def leaves_finite(tree):
    return bool(all(np.all(np.isfinite(np.asarray(v)))
                    for v in jax.tree_util.tree_leaves(tree)))
def trees_equal(a, b):
    la = jax.tree_util.tree_leaves(a); lb = jax.tree_util.tree_leaves(b)
    return (len(la) == len(lb) and
            all(bool(jnp.array_equal(jnp.asarray(x), jnp.asarray(y)))
                for x, y in zip(la, lb)))
def arr_equal(a, b):
    return bool(jnp.array_equal(jnp.asarray(a), jnp.asarray(b)))

gates = {}
report = {"directive": "P2-v1 方案2 Level3 (24576) one-shot verification (#46)",
          "gpu_uuid": GPU_UUID, "step": STEP, "collected": COLLECTED}

# ── build network + restore ──────────────────────────────────────────
cfg = L.Cfg()
network = L.ActorCriticTransformer(
    action_dim=ACTION_DIM, activation=cfg.activation, hidden_layers=cfg.hidden_layers,
    encoder_size=cfg.embed_size, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)

print("[restore] checkpoints/24576 ...", flush=True)
r = L.restore_p2_v1_checkpoint(L.CKPT_ROOT, STEP, network, cfg, OBS_DIM)
gates["restore_succeeds"] = isinstance(r, dict) and "train_state" in r
r_ts = r["train_state"]; r_params = r_ts.params
gs = int(r["global_step"]); uc = int(r["update_count"])
replay = r["replay_buffer"]; r_replay_len = len(replay)
opt_step = L._optimizer_step_count(r_ts)
pending_state = r["pending_state"]; collector_state = r["collector_state"]
gates["global_step_24576"] = (gs == STEP)
gates["params_finite"] = leaves_finite(r_params)
gates["action_rng_present"] = (r["action_rng_state"] is not None)
gates["pending_present"] = (pending_state is not None)
gates["collector_present"] = (collector_state is not None)
print(f"  global_step={gs} update_count={uc} replay_len={r_replay_len} opt_step={opt_step}", flush=True)

# ── conservation + 方案B structure ───────────────────────────────────
completed = pending_tot = max_pending_len = -1
if pending_state is not None:
    pending = PendingEpisodeBuffers.from_state_dict(pending_state)
    completed = sum(int(t.length) for t in replay._buffer)
    pending_tot = pending.total_pending_transitions()
    max_pending_len = max(pending.slot_lengths())
    gates["conservation_completed_plus_pending"] = (completed + pending_tot == COLLECTED)
    # no cross-env concat: each completed traj exactly one done at final step
    done_ok = all((np.asarray(t.dones).astype(bool).sum() == 1 and
                   bool(np.asarray(t.dones).astype(bool)[-1])) for t in replay._buffer)
    pending_no_done = all(not any(bool(x) for x in s["don"]) for s in pending.slots)
    gates["no_cross_env_concat"] = bool(done_ok and pending_no_done)
    gates["boundary_not_faked_done"] = bool(pending_no_done and max_pending_len > 0)
    gates["episode_ids_distinct"] = (len(set(pending.episode_id)) == NUM_ENVS)
    gates["next_episode_id_consistent"] = (pending.next_episode_id == NUM_ENVS + r_replay_len)
    print(f"  completed={completed} pending={pending_tot} sum={completed+pending_tot} max_pending_len={max_pending_len}", flush=True)

# ── round-trip bit-exact ─────────────────────────────────────────────
tmp = tempfile.mkdtemp(prefix="p2v1_level3_rt_")
try:
    save_full_checkpoint(r_ts, replay, r["rng"], gs, tmp, step=gs,
        action_rng_state=r["action_rng_state"], update_count=uc,
        pending_state=pending_state, collector_state=collector_state,
        aux_opt_state=None)
    r2 = L.restore_p2_v1_checkpoint(tmp, gs, network, cfg, OBS_DIM)
    gates["rt_params_bitexact"] = trees_equal(r_params, r2["train_state"].params)
    gates["rt_opt_bitexact"] = trees_equal(r_ts.opt_state, r2["train_state"].opt_state)
    gates["rt_jax_rng_equal"] = bool(jnp.all(r["rng"] == r2["rng"]))
    gates["rt_global_step"] = (int(r2["global_step"]) == gs)
    gates["rt_update_count"] = (int(r2["update_count"]) == uc)
    rb2 = r2["replay_buffer"]
    rt_replay = (len(rb2) == r_replay_len)
    if rt_replay:
        for ta, tb in zip(replay._buffer, rb2._buffer):
            if not (arr_equal(ta.observations, tb.observations) and arr_equal(ta.actions, tb.actions)
                    and arr_equal(ta.rewards, tb.rewards) and arr_equal(ta.dones, tb.dones)):
                rt_replay = False
    gates["rt_replay_bitexact"] = bool(rt_replay)
    if pending_state is not None:
        p2 = PendingEpisodeBuffers.from_state_dict(r2["pending_state"])
        pend_ok = (p2.slot_lengths() == pending.slot_lengths() and
                   p2.next_episode_id == pending.next_episode_id and
                   list(p2.episode_id) == list(pending.episode_id) and
                   list(p2.policy_version) == list(pending.policy_version))
        if pend_ok:
            for sa, sb in zip(pending.slots, p2.slots):
                for k in ("obs","act","rew","don","val","lp","next_obs","mem_pre","mask_pre","ach"):
                    for xa, xb in zip(sa[k], sb[k]):
                        if not arr_equal(xa, xb): pend_ok = False
                if not arr_equal(sa["init_mem"], sb["init_mem"]): pend_ok = False
        gates["rt_pending_bitexact"] = bool(pend_ok)
        c2 = r2["collector_state"]
        gates["rt_collector_bitexact"] = bool(
            arr_equal(collector_state["obsv"], c2["obsv"]) and arr_equal(collector_state["memories"], c2["memories"])
            and arr_equal(collector_state["mem_mask"], c2["mem_mask"]) and arr_equal(collector_state["mem_idx"], c2["mem_idx"])
            and trees_equal(collector_state["env_state"], c2["env_state"]))
    probes = np.random.default_rng(7).standard_normal((4, 43))
    sm = np.exp(probes - probes.max(axis=-1, keepdims=True)); pr = sm / sm.sum(-1, keepdims=True)
    g_a = restore_action_rng(r["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
    g_b = restore_action_rng(r2["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
    seq_a = [int(sample_actions(g_a, pr)[0]) for _ in range(16)]
    seq_b = [int(sample_actions(g_b, pr)[0]) for _ in range(16)]
    gates["rt_action_rng_stream"] = (seq_a == seq_b)
    print(f"  round-trip: params={gates['rt_params_bitexact']} opt={gates['rt_opt_bitexact']} "
          f"replay={gates['rt_replay_bitexact']} pending={gates.get('rt_pending_bitexact')} "
          f"collector={gates.get('rt_collector_bitexact')} action_rng={gates['rt_action_rng_stream']}", flush=True)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ── training_log full analysis ───────────────────────────────────────
man = json.load(open(os.path.join(L.CKPT_ROOT, str(STEP), "stage4_manifest.json")))
sess_dir = man.get("session_output_dir")
recs = [json.loads(l) for l in open(os.path.join(sess_dir, "training_log.jsonl")) if l.strip()]
report["learner_sha_in_manifest"] = man["source_hashes"]["long_context_learner.py"]
report["launcher_sha_in_manifest"] = man["source_hashes"]["stage4_continue_launcher.py"]

ppo_recs = [d for d in recs if "ppo_total_loss" in d]
aux_recs = [d for d in recs if d.get("replay_aux_update") is True]
hs_recs = [d for d in recs if d.get("hindsight_relabelled") is True]
n_ppo = len(ppo_recs); n_aux = len(aux_recs); n_hs = len(hs_recs)

ppo_fields = ["ppo_total_loss","ppo_policy_loss","ppo_value_loss","ppo_entropy","ppo_grad_norm"]
ppo_all_finite = all(np.isfinite(d[k]) for d in ppo_recs for k in ppo_fields if k in d)
ppo_grad_nonzero = all(d.get("ppo_grad_norm", 0.0) > 0.0 for d in ppo_recs)
aux_fields = ["replay_aux_value_loss","replay_aux_total_loss","replay_aux_grad_norm"]
aux_all_finite = all(np.isfinite(d[k]) for d in aux_recs for k in aux_fields if k in d)
aux_actor_off = all((d.get("replay_aux_actor_enabled") is False and float(d.get("replay_aux_actor_loss", 0.0)) == 0.0)
                    for d in aux_recs)
# NaN/Inf across ALL numeric fields in ALL records
def allnums(recs):
    for d in recs:
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                yield k, v
naninf = [(i, k, v) for i, d in enumerate(recs) for k, v in allnums([d]) if not np.isfinite(v)]
no_nan_inf = (len(naninf) == 0)

gates["ppo_12_updates"] = (n_ppo == 12)
gates["ppo_metrics_finite"] = bool(ppo_all_finite)
gates["ppo_grad_nonzero"] = bool(ppo_grad_nonzero)
gates["aux_metrics_finite"] = bool(aux_all_finite) if n_aux else True
gates["aux_actor_off_all"] = bool(aux_actor_off) if n_aux else True
gates["no_nan_inf_anywhere"] = bool(no_nan_inf)
gates["isolation_code_sha"] = (man["source_hashes"]["long_context_learner.py"] == FROZEN_LEARNER_SHA)
# update_count consistency: learner logical counter = 12 PPO + n_aux
gates["update_count_consistent"] = (uc == 12 + n_aux)
# MAIN PPO opt_state must advance EXACTLY 12 updates x 2 minibatches = 24, and must
# NOT include the n_aux replay-aux steps: the aux updates use a SEPARATE masked
# optimizer (方案2 isolation). opt_step==24 (not 24+n_aux) is therefore POSITIVE
# proof that replay-aux did not advance the main PPO optimizer state.
gates["main_opt_step_isolated"] = (opt_step == 12 * 2)

# isolation aggregate evidence
report["isolation"] = {
    "learner_sha_matches_frozen": gates["isolation_code_sha"],
    "n_aux_updates": n_aux,
    "aux_actor_enabled_all_false": bool(aux_actor_off) if n_aux else None,
    "aux_actor_loss_all_zero": bool(all(float(d.get("replay_aux_actor_loss",-1))==0.0 for d in aux_recs)) if n_aux else None,
    "note": "bit-exact trunk/actor freeze proven by CPU test B + probe on identical SHA 6689426b"}

# per-aux diagnostics (policy_lag/IS/ESS/critic loss+grad)
report["aux_per_update"] = [{
    "value_loss": d.get("replay_aux_value_loss"), "grad_norm": d.get("replay_aux_grad_norm"),
    "policy_lag": d.get("replay_aux_policy_lag"), "importance_ratio_mean": d.get("replay_aux_importance_ratio_mean"),
    "ess_fraction": d.get("replay_aux_ess_fraction"), "seq_len": d.get("replay_aux_seq_len"),
    "bootstrap_value": d.get("replay_aux_bootstrap_value"), "episode_done": d.get("replay_aux_episode_done"),
    "hindsight_goal_index": d.get("replay_aux_hindsight_goal_index")} for d in aux_recs]
report["ppo_per_update"] = [{k: d.get(k) for k in ppo_fields} for d in ppo_recs]

# hindsight eligible/accepted
elig_keys = [k for k in (recs[0] if recs else {}) if "eligib" in k.lower()]
n_eligible = sum(int(d.get(elig_keys[0])) for d in recs) if elig_keys else None
report["hindsight"] = {"eligible_key": elig_keys, "n_eligible": n_eligible, "n_accepted": n_hs,
    "accepted_goal_transitions": [{"from": d.get("hindsight_from_goal_idx"), "to": d.get("hindsight_to_goal_idx")} for d in hs_recs]}

# episode length distribution (scan likely keys)
ep_keys = sorted({k for d in recs for k in d if "ep_len" in k.lower() or "episode_length" in k.lower() or "mean_ep" in k.lower()})
report["episode_length_keys"] = ep_keys
report["episode_length_last"] = {k: recs[-1].get(k) for k in ep_keys} if recs else {}

# counters + replay stats from manifest
report["manifest_counters"] = man.get("counters")
report["replay_buffer_size"] = man.get("replay_buffer_size")
report["replay_longest_trajectory"] = man.get("replay_longest_trajectory")
report["aux_opt_state_saved"] = man.get("aux_opt_state_saved")
report["naninf_violations"] = naninf[:10]
report["training_log_records"] = len(recs)
report["gates"] = gates

all_pass = all(bool(v) for v in gates.values())
report["all_pass"] = bool(all_pass)
ev = "/home/oseasy/experiments/single_director_20260722/evidence"
os.makedirs(ev, exist_ok=True)
with open(os.path.join(ev, "p2_v1_level3_verification.json"), "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str); f.write("\n")

print("\n" + "=" * 60, flush=True)
for k in sorted(gates):
    print(f"  {'PASS' if gates[k] else 'FAIL'}  {k}", flush=True)
print("=" * 60, flush=True)
print(f"n_ppo={n_ppo} n_aux={n_aux} n_hindsight_accepted={n_hs} update_count={uc} opt_step={opt_step}", flush=True)
print(f"LEVEL3 VERIFY: {'ALL PASS' if all_pass else 'FAIL'}", flush=True)
sys.exit(0 if all_pass else 1)
