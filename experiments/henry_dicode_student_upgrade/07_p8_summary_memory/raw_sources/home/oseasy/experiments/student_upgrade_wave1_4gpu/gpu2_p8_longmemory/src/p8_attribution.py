"""P8-LONGMEM-SUMMARY — minimal READ-ONLY failure attribution (GPU2). No training, no code change
to the trained artifacts. Two questions:

  (Q1) ENGAGEMENT: did the long-term readout actually turn on over training? The ONLY path by which
       the long summaries can affect behavior is `summary_to_actor` (zero-initialised). We track the
       Fro-norm of its kernel/bias and the summary-attention param norms at every saved node.

  (Q2) CAUSALITY: is the engaged long path WHAT hurts? For each node we run the frozen 256-world
       Stage4 eval with the long contribution DISABLED at eval time (use_longmem=False) but using
       P8's TRAINED weights, and compare against long-ON (from eval_p8_final) and same-step Control.
       If long-OFF recovers toward Control while long-ON collapses, the long path is the cause.

Read-only w.r.t. checkpoints. GPU2 only. Deterministic ops.
"""
import hashlib, json, os, sys, time, pickle

GPU_UUID = "GPU-8df11537-ab79-722d-606f-411966196c4c"   # GPU2 ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax, jax.numpy as jnp
import numpy as np

P8_SRC = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/src"
V7_SRC = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src"
V7 = "/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB"
for p in (P8_SRC, V7_SRC, V7):
    if p not in sys.path:
        sys.path.insert(0, p)

from craftax.craftax.constants import Achievement
from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
from dicode.task_utils import get_achievement_multi_hot
from dicode.wrappers_cl import DistributedMultiTaskOptimisticLogWrapper
from minicraftax.envs.multitask import MultiTaskMiniCraftaxEnv
from p8_network import ActorCriticLongMem, init_longstate

_cfg = dict(activation="relu", embed_size=256, hidden_layers=256, num_heads=8,
            qkv_features=256, num_layers=2, gating=True, gating_bias=2.0,
            window_mem=128, window_grad=64)
cfg = type("C", (), _cfg)()

P8_CKPT_ROOT = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/train_98304/ckpt"
CTL_CKPT_ROOT = "/home/oseasy/experiments/exploratory_delayed_onset_20260724/control_RUN2/ckpt"
GATE_JSON = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/eval/p8_final_gate.json"
NORM_STEPS = [4096, 24576, 49152, 73728, 98304]   # engagement curve (param norms)
EVAL_STEPS = [24576, 49152, 73728, 98304]          # causal long-on/off/control (gate json has these)

DK = int(Achievement.DEFEAT_KOBOLD.value)
SEWERS = int(Achievement.ENTER_SEWERS.value)
NUM_ENVS = 256
NUM_STEPS = 4096
EVAL_SEED = 42

ctor = EnvParams(max_timesteps=NUM_STEPS)
table = jnp.array([get_achievement_multi_hot([Achievement.DEFEAT_KOBOLD])], dtype=jnp.float32)
EMB = int(table.shape[1])
S4_TASK_PATH = "/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py"
ns4 = {}; exec(open(S4_TASK_PATH).read(), ns4); S4Cls = ns4["Env"]
s4_base = MultiTaskMiniCraftaxEnv([S4Cls], StaticEnvParams(), ctor, True,
                                  conditioning_type="embedding", embedding_size=EMB)
ACTION_DIM = int(s4_base.action_space(ctor).n)


def load_params(path):
    d = pickle.load(open(path, "rb"))
    leaves, td = d["params"]
    return jax.tree_util.tree_unflatten(td, [jnp.asarray(l) for l in leaves]), d


def build_net(use_longmem):
    return ActorCriticLongMem(action_dim=ACTION_DIM, activation=cfg.activation,
        encoder_size=cfg.embed_size, hidden_layers=cfg.hidden_layers, num_heads=cfg.num_heads,
        qkv_features=cfg.qkv_features, num_layers=cfg.num_layers, gating=cfg.gating,
        gating_bias=cfg.gating_bias, use_longmem=use_longmem)


# ------------------------------------------------------------------ Q1: engagement
def param_norms(params):
    flat = {}
    def _rec(path, v):
        flat[jax.tree_util.keystr(path)] = float(np.linalg.norm(np.asarray(v)))
        return v
    jax.tree_util.tree_map_with_path(_rec, params)
    keys = [k for k in flat if any(s in k for s in
            ("summary_to_actor", "summary_proj", "summ_q", "summ_k", "summ_v", "summ_o"))]
    return {k: round(flat[k], 6) for k in sorted(keys)}


# ------------------------------------------------------------------ Q2: long-off eval
def run_longoff(params):
    """256-world natural Stage4 eval using P8 TRAINED weights but use_longmem=False (long contribution
    disabled at eval). Same protocol/seed as eval_p8_final -> paired with long-ON and Control."""
    net = build_net(use_longmem=False)
    env = DistributedMultiTaskOptimisticLogWrapper(s4_base, jax.random.PRNGKey(0),
            NUM_ENVS, 1, 16, jnp.array([1.0]), table)
    memories = jnp.zeros((NUM_ENVS, cfg.window_mem, cfg.num_layers, cfg.embed_size))
    mem_mask = jnp.zeros((NUM_ENVS, cfg.num_heads, 1, cfg.window_mem + 1), dtype=jnp.bool_)
    mem_idx = jnp.zeros((NUM_ENVS,), dtype=jnp.int32) + (cfg.window_mem + 1)

    def _step(carry, _):
        (log_state, memories, mem_mask, mem_idx, last_obs, done, true_done, longstate,
         finished, ep_len, max_floor, seen, info_acc, sewers, rng) = carry
        mem_idx = jnp.where(done, cfg.window_mem, jnp.clip(mem_idx - 1, 0, cfg.window_mem))
        mem_mask = jnp.where(done[:, None, None, None], jnp.zeros_like(mem_mask), mem_mask)
        ohot = jax.nn.one_hot(mem_idx, cfg.window_mem + 1)[:, None, None, :].repeat(cfg.num_heads, 1)
        mem_mask = jnp.logical_or(mem_mask, ohot)
        rng, a_rng, s_rng = jax.random.split(rng, 3)
        pi, _, mem_out, ls_new = net.apply(params, memories, last_obs, mem_mask, longstate, true_done,
                                           method=net.forward_eval)
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
        max_floor = jnp.where(active, jnp.maximum(max_floor, pre_pl), max_floor)
        seen = seen | (pre_dk & active)
        sewers = sewers | (pre_sw & active)
        keys = [k for k in info if "Achievements" in k and "kobold" in k.lower()]
        if keys:
            info_acc = info_acc + jnp.asarray(info[keys[0]], jnp.float32).reshape(-1) * active.astype(jnp.float32)
        finished = finished | next_done
        return (next_log_state, memories, mem_mask, mem_idx, next_obs, next_done, true_done_next,
                ls_new, finished, ep_len, max_floor, seen, info_acc, sewers, rng), None

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, reset_rng = jax.random.split(rng)
    obsv, log_state = env.reset(reset_rng, ctor)
    init = (log_state, memories, mem_mask, mem_idx, obsv,
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            init_longstate(NUM_ENVS), jnp.zeros((NUM_ENVS,), dtype=jnp.bool_),
            jnp.zeros((NUM_ENVS,), dtype=jnp.int32), jnp.full((NUM_ENVS,), 2, dtype=jnp.int32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), jnp.zeros((NUM_ENVS,), dtype=jnp.float32),
            jnp.zeros((NUM_ENVS,), dtype=jnp.bool_), rng)
    t0 = time.time()
    final, _ = jax.lax.scan(_step, init, None, NUM_STEPS)
    (_, _, _, _, _, _, _, _, finished, ep_len, max_floor, seen, info_acc, sewers, _) = final
    jax.block_until_ready(final)
    rt = time.time() - t0
    finished_np = np.asarray(finished); ep_len_np = np.asarray(ep_len)
    max_floor_np = np.asarray(max_floor); seen_np = np.asarray(seen)
    info_acc_np = np.asarray(info_acc); sewers_np = np.asarray(sewers)
    success_np = seen_np | (info_acc_np > 0)
    timeout_np = finished_np & (ep_len_np >= NUM_STEPS) & ~success_np
    died_np = finished_np & ~success_np & ~timeout_np
    if int(np.sum(~finished_np)) > 0:
        timeout_np = timeout_np | ~finished_np
    n_success = int(success_np.sum()); n_floor3 = int((max_floor_np >= 3).sum())
    sr = n_success / NUM_ENVS; floor3 = n_floor3 / NUM_ENVS
    print(f"[long-off] SR={sr*100:.2f}% ({n_success}/{NUM_ENVS}) floor3={floor3*100:.2f}% "
          f"died={int(died_np.sum())} timeout={int(timeout_np.sum())} sewers={int(sewers_np.sum())} "
          f"eplen={ep_len_np.mean():.1f} ({rt:.1f}s)", flush=True)
    return dict(SR=sr, n_success=n_success, floor3=floor3, n_floor3=n_floor3,
                death_rate=int(died_np.sum())/NUM_ENVS, timeout_rate=int(timeout_np.sum())/NUM_ENVS,
                sewers_rate=int(sewers_np.sum())/NUM_ENVS, mean_episode_length=float(ep_len_np.mean()),
                success_per_world=[bool(x) for x in success_np])


def main():
    gate = json.load(open(GATE_JSON))
    res = gate["results"]
    out = dict(label="P8_ATTRIBUTION_READ_ONLY", norm_steps=NORM_STEPS, eval_steps=EVAL_STEPS,
               engagement={}, nodes={})
    print("=" * 78, flush=True)
    print("P8 read-only attribution: engagement (param norms) + causality (long-off eval)", flush=True)
    print("=" * 78, flush=True)

    # ---- Q1: engagement curve (all 5 nodes) ----
    print("\n=== Q1 ENGAGEMENT: long-term readout param norms over training ===", flush=True)
    for st in NORM_STEPS:
        p8_params, _ = load_params(os.path.join(P8_CKPT_ROOT, str(st), "full_state.pkl"))
        norms = param_norms(p8_params)
        out["engagement"][st] = norms
        s2a = [v for k, v in norms.items() if "summary_to_actor" in k and "kernel" in k]
        print(f"  step {st:6d}: summary_to_actor kernel norm = {s2a[0] if s2a else float('nan'):.6f}", flush=True)
        print(f"              {json.dumps(norms)}", flush=True)

    # ---- Q2: causality (long-on from gate, control from gate, long-off run here) ----
    print("\n=== Q2 CAUSALITY: long-ON vs long-OFF vs Control (256 worlds) ===", flush=True)
    for st in EVAL_STEPS:
        p8_params, _ = load_params(os.path.join(P8_CKPT_ROOT, str(st), "full_state.pkl"))
        longon_sr = res[f"P8_{st}"]["SR"] * 100
        ctl_sr = res[f"CTL_{st}"]["SR"] * 100
        longon_floor3 = res[f"P8_{st}"]["floor3_reach_rate"] * 100
        ctl_floor3 = res[f"CTL_{st}"]["floor3_reach_rate"] * 100
        print(f"\n--- step {st} ---", flush=True)
        print(f"  long-ON  SR={longon_sr:.2f}% floor3={longon_floor3:.2f}%  | Control SR={ctl_sr:.2f}% floor3={ctl_floor3:.2f}%", flush=True)
        lo = run_longoff(p8_params)
        out["nodes"][st] = dict(
            long_on_SR=longon_sr, long_on_floor3=longon_floor3,
            control_SR=ctl_sr, control_floor3=ctl_floor3,
            long_off_SR=lo["SR"]*100, long_off_floor3=lo["floor3"]*100,
            long_off_death=lo["death_rate"]*100, long_off_sewers=lo["sewers_rate"]*100,
            long_off_eplen=lo["mean_episode_length"],
            dSR_on_vs_ctl=longon_sr-ctl_sr, dSR_off_vs_ctl=lo["SR"]*100-ctl_sr,
            dSR_off_vs_on=lo["SR"]*100-longon_sr,
            long_off_success_per_world=lo["success_per_world"],
            long_on_success_per_world=res[f"P8_{st}"]["success_per_world"])
        print(f"  long-OFF SR={lo['SR']*100:.2f}%  (dSR off-vs-ctl={lo['SR']*100-ctl_sr:+.2f}pp, "
              f"off-vs-on={lo['SR']*100-longon_sr:+.2f}pp)", flush=True)

    out["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    outdir = "/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu2_p8_longmemory/eval"
    json.dump(out, open(os.path.join(outdir, "p8_attribution.json"), "w"), indent=2, default=str)
    print("\n" + "=" * 78, flush=True)
    print("step    longON_SR  longOFF_SR  Control_SR  | dSR(on-ctl) dSR(off-ctl) dSR(off-on)", flush=True)
    for st in EVAL_STEPS:
        n = out["nodes"][st]
        print(f"{st:6d}   {n['long_on_SR']:6.2f}    {n['long_off_SR']:6.2f}     {n['control_SR']:6.2f}    "
              f"|  {n['dSR_on_vs_ctl']:+6.2f}     {n['dSR_off_vs_ctl']:+6.2f}      {n['dSR_off_vs_on']:+6.2f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
