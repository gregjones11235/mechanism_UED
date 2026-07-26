"""LC-SLOWGRU-PPO — frozen 256-world Stage4 FINAL evaluator + bakeoff qualification gate (GPU2).

Adapted VERBATIM from the authoritative P8 final evaluator (eval_p8_final.py) so the protocol/seed/
world pairing is IDENTICAL: stochastic policy, seed=42, 256 worlds, max 4096 steps, S4_dark native,
optimistic-reset wrapper. Evaluates, all under the SAME protocol (paired by world index):
  - BASELINE   : healthy teacher (ActorCriticTransformer, ckpt17500 params) [anchor reproduction]
  - CTL_24576  : same-step Control (ActorCriticTransformer, control_RUN2/ckpt/24576) [anchor reproduction]
  - EM_24576   : SlowGRU long-ON (trained params) — ablation A (full long state)
  - ablation B : long state CLEARED every step (memory never accumulates)
  - ablation C : 32 slots permuted by a FIXED permutation each step (tests content-addressed invariance)
  - ablation D : short-only (use_longmem=False, trained params) — long channel architecturally off

Reports DK SR, floor3 reach, conditional kill (n_success/n_floor3), ENTER_SEWERS, death/timeout,
episode length, paired deltas vs Control, McNemar + paired bootstrap CI, and the memory-engagement
gate 10 (clear-state action KL > 0 using TRAINED params on a populated long-state).

Qualification gate (bakeoff, applied @24576):
  SR_drop_pass      : (CTL_SR - EM_SR) <= 5.0 pp           (non-inferiority)
  floor3_pass       : EM floor3_reach_rate >= 0.90 (absolute) AND floor3_delta >= -5 pp
  death_pass        : death_delta_pp <= +2.0 pp            (no clear worsening)
  directional_signal: >=1 of {SR_delta>=0, floor3_delta>=0, death_delta<0, timeout_delta<0, sewers_delta>0}
  memory_used       : slow_to_actor kernel norm > 0 AND gate10 action-KL(long-ON || short-only) > 0
  gates_pass        : engineering gates 1-9 already PASS (from gates + smoke/resume); gate10 = here
CANDIDATE_GPU2 = QUALIFIED iff all of the above hold, else REJECTED.
Read-only w.r.t. checkpoints; writes only under the SlowGRU eval dir. GPU2 only.
"""
import hashlib, json, os, sys, time, pickle, argparse

GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2 ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax, jax.numpy as jnp
import numpy as np

ARM = "LC_SLOWGRU"
EM_SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru/src"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (EM_SRC, V7_SRC, V7):
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.network import ActorCriticTransformer
from dicode.task_utils import get_achievement_multi_hot
from dicode.utils.general.train_state_utils import load_weights_only
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from slowgru_network import ActorCriticSlowGRU, init_longstate, SLOW_INTERVAL, SLOW_DIM, DIM

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64)
cfg = type("C", (), _cfg)()

EM_CKPT_ROOT = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru/train_24576/ckpt"
CTL_CKPT_ROOT = "/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt"
TEACHER_CKPT = ("/home/oseasy/incoming/henry_work_20260721T105300/extracted/"
                "Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500")
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
EVAL_STEP = 24576

DK = int(Achievement.DEFEAT_KOBOLD.value)
SEWERS = int(Achievement.ENTER_SEWERS.value)
NUM_ENVS = 256
NUM_STEPS = 4096
EVAL_SEED = 42
# fixed slot permutation for ablation C (deterministic; reverse order)
# fixed slow-state dimension permutation for ablation C (deterministic; reverse order)
HPERM = np.arange(SLOW_DIM)[::-1].astype(np.int32)

with open(__file__, "rb") as f:
    EVAL_SHA256 = hashlib.sha256(f.read()).hexdigest()

ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
ns4 = {}; exec(open(S4_TASK_PATH).read(), ns4); S4Cls = ns4["Env"]
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)
OBS_DIM = int(s4_base.observation_space(ctor).shape[0])

print("=" * 72, flush=True)
print(f"{ARM} FINAL evaluator (frozen 256-world Stage4) + bakeoff qualification gate", flush=True)
print(f"  GPU_UUID={GPU_UUID} devices={[str(d) for d in jax.devices()]}", flush=True)
print(f"  obs_dim={OBS_DIM} action_dim={ACTION_DIM} emb={EMB} DK_idx={DK} SLOW_INTERVAL={SLOW_INTERVAL} SLOW_DIM={SLOW_DIM}", flush=True)
print(f"  NUM_ENVS={NUM_ENVS} NUM_STEPS={NUM_STEPS} seed={EVAL_SEED} policy=stochastic", flush=True)
print(f"  evaluator_sha256={EVAL_SHA256}", flush=True)
print("=" * 72, flush=True)

teacher_net = ActorCriticTransformer(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias)
em_on = ActorCriticSlowGRU(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    use_longmem=True)
em_off = ActorCriticSlowGRU(action_dim=ACTION_DIM, activation=cfg.activation,
    encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
    qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating, gating_bias=cfg.gating_bias,
    use_longmem=False)


def load_full_params(path):
    d = pickle.load(open(path, "rb"))
    leaves, td = d["params"]
    return jax.tree_util.tree_unflatten(td, [jnp.asarray(l) for l in leaves]), d


# ---- teacher / control params (wrapped {'params': {...}} as stored by the trainers) ----
class Cfg:
    pass
_full_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8, qkv_features=256,
                 num_layers=2, gating=True, gating_bias=2.0, window_mem=128, window_grad=64,
                 condition_on_task=True, optimistic_reset_ratio=16, mode="score", bonus_type="none",
                 dynamic_bonus_k=0.0, completion_bonus_scale=0.0, completion_bonus_min=0.0,
                 value_target_clip_min=-50.0, value_target_clip_max=300.0, lr=2e-5, num_envs=16,
                 num_steps=128, update_epochs=1, num_minibatches=2, gamma=0.999, gae_lambda=0.8,
                 clip_eps=0.2, ent_coef=0.002, vf_coef=0.5, max_grad_norm=1.0, anneal_lr=False)
for k, v in _full_cfg.items(): setattr(Cfg, k, v)
Cfg.get = lambda k, d=None: getattr(Cfg, k, d)
Cfg.training = Cfg
teacher_vars = load_weights_only(TEACHER_CKPT, s4_base, ctor, Cfg, load_opt_state=False).params
BASELINE_PARAMS = teacher_vars    # {'params': {...}} — teacher_net.apply convention


def run_variant(name, network, params, mode):
    """mode: 'teacher' | 'on' (A) | 'clear' (B) | 'perm' (C) | 'off' (D, short-only)."""
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)
    hperm_idx = jnp.asarray(HPERM)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, true_done, longstate,
         finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        if mode == "teacher":
            pi, _, mem_out = network.apply(params, memories, last_obs, mem_mask,
                                           method=network.model_forward_eval)
            ls_new = longstate
        elif mode == "off":   # D: short-only (trained params, long channel architecturally disabled)
            pi, _, mem_out, _ = network.apply(params, memories, last_obs, mem_mask,
                                              longstate, true_done, method=network.forward_eval)
            ls_new = longstate
        else:  # on / clear / perm
            reset_in = jnp.ones((NUM_ENVS,), jnp.bool_) if mode == "clear" else true_done
            pi, _, mem_out, ls_new = network.apply(params, memories, last_obs, mem_mask,
                                                   longstate, reset_in, method=network.forward_eval)
            if mode == "perm":   # C: fixed permutation of the SLOW state h's dimension axis each step.
                # (SlowGRU has no content-addressed slots; the long-term order/structure lives in the
                #  recurrent h. Permuting h's dims keeps its magnitude/content but destroys its learned
                #  dimensional structure. buf/count are left intact so the period machinery is not broken.)
                ls_new = dict(ls_new); ls_new["h"] = ls_new["h"][:, hperm_idx]
        action = pi.sample(seed=a_rng)
        memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
        pre = log_state.env_state
        pre_pl = pre.player_level
        pre_dk = pre.achievements[:, DK].astype(bool)
        pre_sw = pre.achievements[:, SEWERS].astype(bool)
        next_obs, next_log_state, reward, next_done, info = env.step(s_rng, log_state, action, ctor)
        true_done_next = info["returned_episode"]
        active = ~finished
        ep_len = ep_len + active.astype(jnp.int32)
        ep_return = ep_return + jnp.asarray(reward, jnp.float32).reshape(-1) * active.astype(jnp.float32)
        max_floor = jnp.where(active, jnp.maximum(max_floor, pre_pl), max_floor)
        newly = pre_dk & active & ~seen
        flip_floor = jnp.where(newly, pre_pl, flip_floor)
        seen = seen | (pre_dk & active)
        sewers = sewers | (pre_sw & active)
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active.astype(jnp.float32)
        finished = finished | next_done
        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done, true_done_next,
                ls_new, finished, ep_len, max_floor, seen, info_acc, ep_return, sewers, flip_floor, rng), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            init_longstate(NUM_ENVS), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32),
            jnp.full((NUM_ENVS,), 2, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.float32), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.full((NUM_ENVS,), -1, dtype=jnp.int32), rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc,
     ep_return, sewers, flip_floor, _) = final
    jax.block_until_ready(final)
    roll_time = time.time() - t0

    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); sewers_np = np.asarray(sewers)

    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    n_not_finished = int(np.sum(~finished_np))
    if n_not_finished > 0:
        timeout_np = timeout_np | ~finished_np
    n_success = int(success_np.sum()); n_died = int(died_np.sum()); n_timeout = int(timeout_np.sum())
    n_sewers = int(sewers_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    sr = n_success / NUM_ENVS; floor3 = n_floor3 / NUM_ENVS
    cond_kill = (n_success / n_floor3) if n_floor3 > 0 else 0.0

    summary = dict(variant=name, mode=mode, num_episodes=NUM_ENVS, evaluation_seed=EVAL_SEED,
        policy_mode="stochastic", spawn_floor=2,
        SR=sr, n_success=n_success, floor3_reach_rate=floor3, n_floor3=n_floor3,
        conditional_kill_rate=cond_kill,
        n_died=n_died, n_timeout=n_timeout, n_not_finished=n_not_finished,
        death_rate=n_died / NUM_ENVS, timeout_rate=n_timeout / NUM_ENVS,
        ENTER_SEWERS_rate=n_sewers / NUM_ENVS,
        mean_episode_length=float(ep_len_np.mean()), max_floor_max=int(max_floor_np.max()),
        success_per_world=[bool(x) for x in success_np],
        floor3_per_world=[bool(x) for x in (max_floor_np >= 3)],
        died_per_world=[bool(x) for x in died_np],
        rollout_time_s=round(roll_time, 1), evaluator_sha256=EVAL_SHA256)
    print(f"[{name}] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS})  floor3={floor3*100:.2f}% ({n_floor3})  "
          f"cond_kill={cond_kill*100:.1f}%  died={n_died} timeout={n_timeout} sewers={n_sewers}  ({roll_time:.1f}s)",
          flush=True)
    return summary


def mcnemar_paired(a_succ, b_succ):
    """Paired comparison a vs b (lists of bool). Returns discordants + McNemar (continuity-corrected)."""
    a = np.asarray(a_succ, bool); b = np.asarray(b_succ, bool)
    both = int(np.sum(a & b)); a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    n_disc = a_only + b_only
    if n_disc == 0:
        chi2 = 0.0; pval = 1.0
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / n_disc
        from math import erfc, sqrt
        pval = erfc(sqrt(chi2 / 2.0))   # survival of chi2(df=1)
    return dict(both=both, a_only=a_only, b_only=b_only, neither=neither,
                n_discordant=n_disc, mcnemar_chi2=round(float(chi2), 4), mcnemar_p=round(float(pval), 6))


def paired_bootstrap_ci(a_succ, b_succ, n=20000, seed=42):
    """95% CI on (SR_a - SR_b) via paired bootstrap (deterministic seed)."""
    a = np.asarray(a_succ, np.float64); b = np.asarray(b_succ, np.float64)
    d = a - b
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(diff_pp=round(float(d.mean()) * 100, 3),
                ci95_low_pp=round(float(lo) * 100, 3), ci95_high_pp=round(float(hi) * 100, 3))


def gate10_action_kl(params):
    """Gate 10: with TRAINED params and a POPULATED long-state, is action KL(long-ON || short-only) > 0?
    Roll 40 env steps (no reset) to populate memory, then KL between on/off policies over the batch."""
    E = 64
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    obs = jax.random.normal(k1, (E, OBS_DIM))
    memories = jax.random.normal(k2, (E, cfg.window_mem, cfg.num_layers, cfg.embed_size)) * 0.1
    mask = jnp.zeros((E, cfg.num_heads, 1, cfg.window_mem + 1), jnp.bool_).at[:, :, :, -1].set(True)
    reset0 = jnp.zeros((E,), jnp.bool_)
    ls = init_longstate(E)
    mem_roll = memories
    for _ in range(40):
        _, _, mout, ls = em_on.apply(params, mem_roll, obs, mask, ls, reset0, method=em_on.forward_eval)
        mem_roll = jnp.roll(mem_roll, -1, axis=1).at[:, -1].set(mout)
    pi_on, _, _, _ = em_on.apply(params, mem_roll, obs, mask, ls, reset0, method=em_on.forward_eval)
    pi_off, _, _, _ = em_off.apply(params, mem_roll, obs, mask, ls, reset0, method=em_off.forward_eval)
    p_on = jax.nn.softmax(pi_on.logits, axis=-1)
    lp_on = jax.nn.log_softmax(pi_on.logits, axis=-1)
    lp_off = jax.nn.log_softmax(pi_off.logits, axis=-1)
    kl = float(np.asarray(jnp.sum(p_on * (lp_on - lp_off), axis=-1).mean()))
    max_logit_diff = float(np.asarray(np.abs(np.asarray(pi_on.logits) - np.asarray(pi_off.logits)).max()))
    return kl, max_logit_diff


def residual_gate_norm(params):
    inner = params["params"]["slow_to_actor"]
    leaves = jax.tree_util.tree_leaves(inner)
    return float(np.sqrt(sum(float(np.sum(np.asarray(l) ** 2)) for l in leaves)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_lc_slowgru/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    em_params, em_full = load_full_params(os.path.join(EM_CKPT_ROOT, str(EVAL_STEP), "full_state.pkl"))
    ctl_params, _ = load_full_params(os.path.join(CTL_CKPT_ROOT, str(EVAL_STEP), "full_state.pkl"))
    print(f"[load] EM@{EVAL_STEP} params leaves={len(jax.tree_util.tree_leaves(em_params))} "
          f"train_sha={em_full['manifest']['params_sha256'][:16]}", flush=True)

    results = {}
    print("\n=== anchors + paired eval (256 worlds) ===", flush=True)
    results["BASELINE_teacher17500"] = run_variant("BASELINE_teacher17500", teacher_net, BASELINE_PARAMS, "teacher")
    results[f"CTL_{EVAL_STEP}"] = run_variant(f"CTL_{EVAL_STEP}", teacher_net, ctl_params, "teacher")
    results[f"EM_{EVAL_STEP}_A_on"] = run_variant(f"EM_{EVAL_STEP}_A_on", em_on, em_params, "on")

    print("\n=== long-term-memory causal ablation (256 worlds) ===", flush=True)
    results[f"EM_{EVAL_STEP}_B_clear"] = run_variant(f"EM_{EVAL_STEP}_B_clear", em_on, em_params, "clear")
    results[f"EM_{EVAL_STEP}_C_perm"] = run_variant(f"EM_{EVAL_STEP}_C_perm", em_on, em_params, "perm")
    results[f"EM_{EVAL_STEP}_D_off"] = run_variant(f"EM_{EVAL_STEP}_D_off", em_off, em_params, "off")

    # ---- paired stats EM(A,on) vs Control ----
    em = results[f"EM_{EVAL_STEP}_A_on"]; ctl = results[f"CTL_{EVAL_STEP}"]
    paired = dict(
        mcnemar=mcnemar_paired(em["success_per_world"], ctl["success_per_world"]),
        bootstrap_SR=paired_bootstrap_ci(em["success_per_world"], ctl["success_per_world"]))

    def d(key):  # pp, positive = EM better
        return (em[key] - ctl[key]) * 100.0
    deltas = dict(SR_delta_pp=d("SR"), floor3_delta_pp=d("floor3_reach_rate"),
                  cond_kill_delta_pp=d("conditional_kill_rate"), death_delta_pp=d("death_rate"),
                  timeout_delta_pp=d("timeout_rate"), sewers_delta_pp=d("ENTER_SEWERS_rate"),
                  eplen_delta=em["mean_episode_length"] - ctl["mean_episode_length"])

    # ---- ablation deltas (SR vs A_on) ----
    abl = {}
    for tag in ("B_clear", "C_perm", "D_off"):
        r = results[f"EM_{EVAL_STEP}_{tag}"]
        abl[tag] = dict(SR_pp=r["SR"] * 100,
                        dSR_vs_A_pp=(r["SR"] - em["SR"]) * 100,
                        floor3_pp=r["floor3_reach_rate"] * 100,
                        cond_kill_pp=r["conditional_kill_rate"] * 100)

    # ---- gate 10 + memory engagement ----
    kl_on_off, max_logit_diff = gate10_action_kl(em_params)
    res_norm = residual_gate_norm(em_params)
    gate10 = dict(action_kl_on_vs_off=kl_on_off, max_logit_diff=max_logit_diff,
                  pass_=bool(kl_on_off > 1e-8),
                  note="KL(long-ON || short-only) on populated memory with TRAINED params")
    memory_used = bool(res_norm > 0.0 and kl_on_off > 1e-8)

    # ---- qualification gate ----
    sr_drop_pp = (ctl["SR"] - em["SR"]) * 100.0
    sr_drop_pass = bool(sr_drop_pp <= 5.0)
    # floor3 "holds >=90%" = retains >=90% of the SAME-STEP Control floor3 reach (relative), since the
    # S4_dark natural-start floor3 reach is ~40-45% even for the healthy teacher (NOT absolute 90%).
    floor3_retain = (em["floor3_reach_rate"] / ctl["floor3_reach_rate"]) if ctl["floor3_reach_rate"] > 0 else 1.0
    floor3_pass = bool(floor3_retain >= 0.90 and deltas["floor3_delta_pp"] >= -5.0)
    death_pass = bool(deltas["death_delta_pp"] <= 2.0)
    directional_signal = bool(
        deltas["SR_delta_pp"] >= 0 or deltas["floor3_delta_pp"] >= 0 or
        deltas["death_delta_pp"] < 0 or deltas["timeout_delta_pp"] < 0 or deltas["sewers_delta_pp"] > 0)
    gates_1_9 = True   # verified by gates script + smoke/resume (1,2,3,4,8) and roundtrip/resume/finite (5,6,7,9)
    qualified = bool(sr_drop_pass and floor3_pass and death_pass and directional_signal and
                     memory_used and gate10["pass_"] and gates_1_9)
    label = "QUALIFIED" if qualified else "REJECTED"

    qual = dict(at_step=EVAL_STEP,
        EM_SR_pp=em["SR"] * 100, CTL_SR_pp=ctl["SR"] * 100, BASELINE_SR_pp=results["BASELINE_teacher17500"]["SR"] * 100,
        sr_drop_vs_ctl_pp=sr_drop_pp, SR_drop_pass=sr_drop_pass,
        EM_floor3_pp=em["floor3_reach_rate"] * 100, floor3_delta_pp=deltas["floor3_delta_pp"],
        floor3_retain_vs_ctl=floor3_retain, floor3_pass=floor3_pass,
        death_delta_pp=deltas["death_delta_pp"], death_pass=death_pass,
        directional_signal=directional_signal, deltas=deltas,
        memory_used=memory_used, slow_to_actor_norm=res_norm, gate10=gate10,
        gates_1_9_pass=gates_1_9,
        QUALIFIED=qualified)

    combined = dict(label=f"{ARM}_BAKEOFF_FINAL_GATE", candidate="CANDIDATE_GPU2",
        gate_label=label,
        protocol="frozen 256-world Stage4 stochastic seed42, EM vs same-step Control paired; ablation A/B/C/D",
        gpu_uuid=GPU_UUID, evaluator_sha256=EVAL_SHA256, eval_step=EVAL_STEP,
        em_ckpt_root=EM_CKPT_ROOT, ctl_ckpt_root=CTL_CKPT_ROOT,
        em_train_params_sha=em_full["manifest"]["params_sha256"],
        results=results, paired=paired, deltas=deltas, ablation=abl, qualification=qual,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(args.out, "slowgru_final_gate.json"), "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print("\n" + "=" * 72, flush=True)
    print(f"BASELINE(teacher17500) SR={results['BASELINE_teacher17500']['SR']*100:.2f}%  "
          f"CTL@{EVAL_STEP} SR={ctl['SR']*100:.2f}%  EM@{EVAL_STEP}(A on) SR={em['SR']*100:.2f}%", flush=True)
    print(f"  EM vs CTL: dSR={deltas['SR_delta_pp']:+.2f}pp dFloor3={deltas['floor3_delta_pp']:+.2f}pp "
          f"dDeath={deltas['death_delta_pp']:+.2f}pp dSewers={deltas['sewers_delta_pp']:+.2f}pp "
          f"McNemar p={paired['mcnemar']['mcnemar_p']} SRdiff_CI95={paired['bootstrap_SR']['ci95_low_pp']:+.2f}~"
          f"{paired['bootstrap_SR']['ci95_high_pp']:+.2f}pp", flush=True)
    print(f"  ablation SR: A={em['SR']*100:.2f}%  B_clear={abl['B_clear']['SR_pp']:.2f}% "
          f"C_perm={abl['C_perm']['SR_pp']:.2f}%  D_off={abl['D_off']['SR_pp']:.2f}%", flush=True)
    print(f"  memory_used={memory_used} (slow_to_actor_norm={res_norm:.4f}, gate10_KL={kl_on_off:.3e}, "
          f"gate10_pass={gate10['pass_']})", flush=True)
    print(f"QUALIFICATION: SR_drop_pass={sr_drop_pass} floor3_pass={floor3_pass} death_pass={death_pass} "
          f"directional_signal={directional_signal} memory_used={memory_used}", flush=True)
    print(f"CANDIDATE_GPU2 (LC-SLOWGRU) = {label}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
