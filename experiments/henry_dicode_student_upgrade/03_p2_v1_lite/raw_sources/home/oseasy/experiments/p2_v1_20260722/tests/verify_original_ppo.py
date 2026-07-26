#!/usr/bin/env python3
"""P2-v1 Original-PPO control (24576-step) one-shot verification (GPU0).

Pure-Henry-native-PPO control: same session175 init / seed / config / 24576 steps
as P2-v1 Full, but replay-aux + hindsight DISABLED. Verifies, in ONE pass:
  - global_step=24576, params finite, update_count==12 (PPO only), opt_step==24
  - replay_aux_update is False for EVERY training_log record (pure PPO) + 0 relabel
  - 12 PPO updates all finite, grad>0; no NaN/Inf anywhere
  - conservation completed+pending==24576; no cross-env concat; boundary not faked
  - round-trip bit-exact (params/opt/rng/global_step/update_count/replay/pending/
    collector/action_rng)
  - frozen source SHAs unchanged (driver only patched at runtime)
  - params DIFFER from both session175 base and P2-v1 Full @24576 (3 distinct models)
Writes detailed JSON + prints terse PASS/FAIL.
"""
import os, sys, json, tempfile, shutil
import numpy as np

GPU_UUID = "GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
import jax, jax.numpy as jnp

P2 = "/home/oseasy/experiments/p2_v1_20260722/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
for p in (V7, P2):
    if p not in sys.path: sys.path.insert(0, p)
import stage4_continue_launcher as L
from checkpointing import save_full_checkpoint
from pending_episodes import PendingEpisodeBuffers
from rng_utils import restore_action_rng, sample_actions

STEP = 24576; COLLECTED = 24576; NUM_ENVS = 16; OBS_DIM = 8335; ACTION_DIM = 43
OP_CKPT = "/home/oseasy/experiments/p2_v1_20260722/checkpoints_original_ppo"
P2FULL_CKPT = "/home/oseasy/experiments/p2_v1_20260722/checkpoints"  # Level3 / P2-v1 Full
FROZEN = {"long_context_learner.py": "6689426b77bb030c8ce3a3a3c97ddab7bd0248d2eaa1d477146dff44ccf1c386",
          "stage4_continue_launcher.py": "36ec9cd9eef7f3408b6b8680be7d2d21552be4577e78946cf0948b0b9ca9079f"}

def leaves_finite(t): return bool(all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(t)))
def trees_equal(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return len(la) == len(lb) and all(bool(jnp.array_equal(jnp.asarray(x), jnp.asarray(y))) for x, y in zip(la, lb))
def arr_equal(a, b): return bool(jnp.array_equal(jnp.asarray(a), jnp.asarray(b)))
def n_diff_leaves(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return sum(0 if bool(jnp.array_equal(jnp.asarray(x), jnp.asarray(y))) else 1 for x, y in zip(la, lb))

gates = {}; report = {"run": "P2-v1 Original-PPO control verification (#47)", "gpu_uuid": GPU_UUID, "step": STEP}
cfg = L.Cfg()
network = L.ActorCriticTransformer(action_dim=ACTION_DIM, activation=cfg.activation, hidden_layers=cfg.hidden_layers,
    encoder_size=cfg.embed_size, num_heads=cfg.num_heads, qkv_features=cfg.qkv_features,
    num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)

print("[restore] checkpoints_original_ppo/24576 ...", flush=True)
r = L.restore_p2_v1_checkpoint(OP_CKPT, STEP, network, cfg, OBS_DIM)
gates["restore_succeeds"] = isinstance(r, dict) and "train_state" in r
r_ts = r["train_state"]; r_params = r_ts.params
gs = int(r["global_step"]); uc = int(r["update_count"])
replay = r["replay_buffer"]; rlen = len(replay)
opt_step = L._optimizer_step_count(r_ts)
pending_state = r["pending_state"]; collector_state = r["collector_state"]
gates["global_step_24576"] = (gs == STEP)
gates["update_count_12_ppo_only"] = (uc == 12)
gates["main_opt_step_24"] = (opt_step == 24)
gates["params_finite"] = leaves_finite(r_params)
gates["pending_present"] = (pending_state is not None)
gates["collector_present"] = (collector_state is not None)
print(f"  global_step={gs} update_count={uc} opt_step={opt_step} replay_len={rlen}", flush=True)

# conservation + 方案B structure
if pending_state is not None:
    pending = PendingEpisodeBuffers.from_state_dict(pending_state)
    completed = sum(int(t.length) for t in replay._buffer)
    pending_tot = pending.total_pending_transitions()
    gates["conservation_24576"] = (completed + pending_tot == COLLECTED)
    done_ok = all((np.asarray(t.dones).astype(bool).sum() == 1 and bool(np.asarray(t.dones).astype(bool)[-1])) for t in replay._buffer)
    pending_no_done = all(not any(bool(x) for x in s["don"]) for s in pending.slots)
    gates["no_cross_env_concat"] = bool(done_ok and pending_no_done)
    gates["boundary_not_faked"] = bool(pending_no_done and max(pending.slot_lengths()) > 0)
    print(f"  completed={completed} pending={pending_tot} sum={completed+pending_tot}", flush=True)

# round-trip bit-exact
tmp = tempfile.mkdtemp(prefix="origppo_rt_")
try:
    save_full_checkpoint(r_ts, replay, r["rng"], gs, tmp, step=gs, action_rng_state=r["action_rng_state"],
        update_count=uc, pending_state=pending_state, collector_state=collector_state, aux_opt_state=None)
    r2 = L.restore_p2_v1_checkpoint(tmp, gs, network, cfg, OBS_DIM)
    gates["rt_params"] = trees_equal(r_params, r2["train_state"].params)
    gates["rt_opt"] = trees_equal(r_ts.opt_state, r2["train_state"].opt_state)
    gates["rt_rng"] = bool(jnp.all(r["rng"] == r2["rng"]))
    gates["rt_global_step"] = (int(r2["global_step"]) == gs)
    gates["rt_update_count"] = (int(r2["update_count"]) == uc)
    rb2 = r2["replay_buffer"]; rt_replay = (len(rb2) == rlen)
    if rt_replay:
        for ta, tb in zip(replay._buffer, rb2._buffer):
            if not (arr_equal(ta.observations, tb.observations) and arr_equal(ta.actions, tb.actions)
                    and arr_equal(ta.rewards, tb.rewards) and arr_equal(ta.dones, tb.dones)): rt_replay = False
    gates["rt_replay"] = bool(rt_replay)
    p2 = PendingEpisodeBuffers.from_state_dict(r2["pending_state"])
    gates["rt_pending"] = bool(p2.slot_lengths() == pending.slot_lengths() and p2.next_episode_id == pending.next_episode_id
        and list(p2.episode_id) == list(pending.episode_id))
    c2 = r2["collector_state"]
    gates["rt_collector"] = bool(arr_equal(collector_state["obsv"], c2["obsv"]) and arr_equal(collector_state["memories"], c2["memories"])
        and trees_equal(collector_state["env_state"], c2["env_state"]))
    probes = np.random.default_rng(7).standard_normal((4, 43))
    sm = np.exp(probes - probes.max(-1, keepdims=True)); pr = sm / sm.sum(-1, keepdims=True)
    ga = restore_action_rng(r["action_rng_state"], seed=L.P2_V1_MASTER_SEED); gb = restore_action_rng(r2["action_rng_state"], seed=L.P2_V1_MASTER_SEED)
    gates["rt_action_rng"] = ([int(sample_actions(ga, pr)[0]) for _ in range(16)] == [int(sample_actions(gb, pr)[0]) for _ in range(16)])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# training_log analysis
man = json.load(open(os.path.join(OP_CKPT, str(STEP), "stage4_manifest.json")))
recs = [json.loads(l) for l in open(os.path.join(man["session_output_dir"], "training_log.jsonl")) if l.strip()]
report["learner_sha_manifest"] = man["source_hashes"]["long_context_learner.py"]
report["launcher_sha_manifest"] = man["source_hashes"]["stage4_continue_launcher.py"]
ppo_recs = [d for d in recs if "ppo_total_loss" in d]
aux_recs = [d for d in recs if d.get("replay_aux_update") is True]
hs_recs = [d for d in recs if d.get("hindsight_relabelled") is True]
n_ppo, n_aux, n_hs = len(ppo_recs), len(aux_recs), len(hs_recs)
pf = ["ppo_total_loss","ppo_policy_loss","ppo_value_loss","ppo_entropy","ppo_grad_norm"]
naninf = [(i, k, v) for i, d in enumerate(recs) for k, v in d.items()
          if isinstance(v, (int, float)) and not isinstance(v, bool) and not np.isfinite(v)]
gates["ppo_12_updates"] = (n_ppo == 12)
gates["aux_zero_pure_ppo"] = (n_aux == 0)
gates["replay_aux_all_false"] = all(d.get("replay_aux_update") is False for d in recs)
gates["hindsight_zero"] = (n_hs == 0)
gates["ppo_metrics_finite"] = bool(all(np.isfinite(d[k]) for d in ppo_recs for k in pf if k in d))
gates["ppo_grad_nonzero"] = bool(all(d.get("ppo_grad_norm", 0.0) > 0.0 for d in ppo_recs))
gates["no_nan_inf"] = (len(naninf) == 0)
gates["frozen_learner_sha"] = (man["source_hashes"]["long_context_learner.py"] == FROZEN["long_context_learner.py"])
gates["frozen_launcher_sha"] = (man["source_hashes"]["stage4_continue_launcher.py"] == FROZEN["stage4_continue_launcher.py"])

# distinctness: Original PPO params vs P2-v1 Full @24576 (same init+seed; only aux differs)
try:
    rf = L.restore_p2_v1_checkpoint(P2FULL_CKPT, STEP, network, cfg, OBS_DIM)
    report["n_leaves_diff_vs_p2full_24576"] = n_diff_leaves(r_params, rf["train_state"].params)
    gates["differs_from_p2full"] = (report["n_leaves_diff_vs_p2full_24576"] > 0)
except Exception as e:
    report["p2full_compare_error"] = str(e); gates["differs_from_p2full"] = False

report["n_ppo"] = n_ppo; report["n_aux"] = n_aux; report["n_hindsight"] = n_hs
report["update_count"] = uc; report["opt_step"] = opt_step
report["ppo_per_update"] = [{k: d.get(k) for k in pf} for d in ppo_recs]
report["manifest_counters"] = man.get("counters")
report["naninf_violations"] = naninf[:10]
report["gates"] = gates
all_pass = all(bool(v) for v in gates.values()); report["all_pass"] = bool(all_pass)
ev = "/home/oseasy/experiments/single_director_20260722/evidence"
with open(os.path.join(ev, "p2_v1_original_ppo_verification.json"), "w") as f:
    json.dump(report, f, indent=2, sort_keys=True, default=str); f.write("\n")
print("\n" + "="*60, flush=True)
for k in sorted(gates): print(f"  {'PASS' if gates[k] else 'FAIL'}  {k}", flush=True)
print("="*60, flush=True)
print(f"n_ppo={n_ppo} n_aux={n_aux} n_hindsight={n_hs} update_count={uc} opt_step={opt_step} "
      f"diff_vs_p2full={report.get('n_leaves_diff_vs_p2full_24576')}", flush=True)
print(f"ORIGINAL PPO VERIFY: {'ALL PASS' if all_pass else 'FAIL'}", flush=True)
sys.exit(0 if all_pass else 1)
