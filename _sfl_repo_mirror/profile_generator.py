#!/usr/bin/env python
"""Profile generator 瓶颈，定位 8h 黑洞。三块分开计时(block_until_ready 强制同步)：
  (1) gen_rollout_batch  : 363步自回归 scan 造 level
  (2) compute_terminal_reward : 每张 level 跑冻结 student rollout 算 reward
  (3) gen_ppo_loss×epochs : generator PPO 梯度(那个 batch=23232 的 cuDNN 慢路径)
  (4) gen_train_iter 整体 : 三块合起来一次完整迭代

并对 (1) 做 batch 缩放(64/256/1024)看 batch-bound vs serial-bound。
student 用随机 params(评估开销与 params 值无关，只与形状/步数有关)。
"""
import time, sys
sys.path.insert(0, "sfl/train")
sys.path.insert(0, ".")
import jax, jax.numpy as jnp
print("JAX devices:", jax.devices())

from hydra import compose, initialize
from omegaconf import OmegaConf
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav  # noqa
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from pcgrl_generator import (
    make_pcgrl_env, make_generator_network, make_generator_state,
    gen_rollout_batch, gen_train_iter, compute_terminal_reward,
    gen_batch_to_env_instances,
)


def timed(fn, *a, repeat=3, tag="", **k):
    print(f"    [{tag}] 首次调用(含编译)...", flush=True)
    t0 = time.time(); jax.block_until_ready(fn(*a, **k)); first = time.time() - t0
    print(f"    [{tag}] 首次完成 {first:.2f}s, 复跑×{repeat}...", flush=True)
    runs = []
    for i in range(repeat):
        t = time.time(); jax.block_until_ready(fn(*a, **k)); dt = time.time() - t
        runs.append(dt); print(f"    [{tag}] 复跑{i} {dt:.3f}s", flush=True)
    runs.sort()
    return first, runs[len(runs)//2]


def main():
    with initialize(config_path="sfl/train/config", version_base=None):
        cfg = compose(config_name="jaxnav-sfl")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    ep = dict(cfg["env"]["env_params"])
    hidden = cfg["learning"]["HIDDEN_SIZE"]
    roll_steps = cfg["learning"].get("ROLLOUT_STEPS", cfg.get("ROLLOUT_STEPS", 1000))

    rng = jax.random.PRNGKey(0)
    # student env + network（随机 params）
    student_env = JaxNav(num_agents=cfg["env"]["num_agents"], **ep)
    snet = ActorCriticRNN(student_env.agent_action_space().shape[0], config=cfg["learning"])
    rng, ri = jax.random.split(rng)
    init_x = (jnp.zeros((1, 1, student_env.lidar_num_beams + 5)), jnp.zeros((1, 1)))
    ih = ScannedRNN.initialize_carry(1, hidden)
    sparams = snet.init(ri, ih, init_x)

    # generator
    gen_env, _ = make_pcgrl_env(map_size=11, max_board_scans=3.0)
    gnet, _ = make_generator_network(gen_env)
    rng, r = jax.random.split(rng)
    gstate, gopt = make_generator_state(r, gen_env, gnet, ["difficulty"])
    gp, gos = gstate["params"][0], gstate["opt_state"][0]

    # place_env: valid_path_check=True
    ep2 = {**ep, "map_params": {**dict(ep["map_params"]), "valid_path_check": True}}
    place_env = JaxNav(num_agents=cfg["env"]["num_agents"], **ep2)

    print(f"\n>>> setup done. student rollout_steps={roll_steps} hidden={hidden}", flush=True)

    # --- (1) 纯 rollout batch 缩放 ---
    print("\n=== (1) gen_rollout_batch (363步scan) batch缩放 ===", flush=True)
    print(f"{'n':>6} {'首次s':>8} {'复跑s':>8} {'每level_ms':>11} {'相对64x':>9}", flush=True)
    base = None
    for n in [64, 256, 1024]:
        rng, rr = jax.random.split(rng)
        f = jax.jit(lambda key, nn=n: gen_rollout_batch(gen_env, gnet, gp, key, nn))
        fi, md = timed(f, rr, tag=f"rollout-n{n}")
        if base is None: base = md
        print(f"{n:>6} {fi:>8.2f} {md:>8.3f} {md/n*1000:>11.3f} {md/base:>9.2f}", flush=True)

    # --- (2)(3)(4) 完整 gen_train_iter (含 student 评估 + PPO)，n=64 ---
    print("\n=== (4) gen_train_iter 整体 (n=64, 含 rollout+student评估+PPO 4epoch) ===", flush=True)
    def one_iter(key):
        return gen_train_iter(
            gen_env, gnet, gp, gos, gopt, key,
            student_env, snet, sparams, hidden, roll_steps, 64,
            estimator_id="difficulty", place_env=place_env, ppo_epochs=4)
    fjit = jax.jit(one_iter)
    rng, rk = jax.random.split(rng)
    fi, md = timed(fjit, rk, repeat=3, tag="train_iter")
    print(f"  首次(含编译)={fi:.2f}s  复跑={md:.3f}s", flush=True)
    print(f"  → 每50update触发一次×N=3 generator → 单次eval_freq的generator开销≈{md*3:.2f}s(复跑)")
    print(f"  → 整训练40个eval_freq → generator总开销≈{md*3*40/60:.1f}min(纯运行,不含编译/student PPO)")

    # --- 单独 student 评估 (compute_terminal_reward) ---
    print("\n=== (2) compute_terminal_reward (student 评估 64 levels) 单独计时 ===", flush=True)
    rng, ra, rb = jax.random.split(rng, 3)
    emaps = gen_rollout_batch(gen_env, gnet, gp, ra, 64)
    insts, _ = gen_batch_to_env_instances(emaps, place_env, rb)
    def cterm(key):
        return compute_terminal_reward("difficulty", student_env, snet, sparams, insts, key,
                                       hidden, roll_steps, 64)
    cjit = jax.jit(cterm)
    rng, rc = jax.random.split(rng)
    fi, md = timed(cjit, rc, repeat=3, tag="student_eval")
    print(f"  首次={fi:.2f}s 复跑={md:.3f}s  (student rollout {roll_steps}步 × 64 level)", flush=True)
    print("\n>>> PROFILE DONE", flush=True)


if __name__ == "__main__":
    main()
