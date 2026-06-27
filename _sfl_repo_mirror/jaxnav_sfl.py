"""
Run SFL on JaxNav, both single and multi-agent variations.
"""

# BLAS single-thread (fix#1: root-cause for PROBE_ORTHOGONALITY probe futex deadlock
# inside the JAX process). fit_visitation_gmm sklearn silhouette_score uses multi-thread
# OpenBLAS; after JAX fork the BLAS threadpool lock state is undefined, the main thread
# blocks on futex_wait_queue forever -> GPU 0%% stall. Single-thread BLAS cuts that off
# (SBATCH is CPUs=1 anyway). Must be set BEFORE importing numpy/jax.
# See mechanism_UED memory: stage4-phase2-probe-futex-stall.
import os as _os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import os
from functools import partial
import pickle
import time 
from PIL import Image
import wandb
import matplotlib.pyplot as plt

from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav, EnvInstance, NUM_REWARD_COMPONENTS, REWARD_COMPONENT_DENSE, REWARD_COMPONENT_SPARSE, listify_reward

from sfl.runners import EvalSingletonsRunner, EvalSampledRunner
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from sfl.train.train_utils import save_params

# ── V1 正交性探针（边训边验四信号正交性，host 侧）。estimators/probe/train_probe
#    三件套与 jaxnav_sfl.py 同级或 PYTHONPATH 可达。见 INTEGRATION.md。
from train_probe import log_orthogonality_step, summarize_trace
# ── auction 海选打分：N estimator → bid → auction → 混合 score（§5/§6）。
#    PVL 真信号复用官方 jaxued；mix_scores 复用 auction_bid（16/16 单测过）。
from sfl.util.jaxued.jaxued_utils import positive_value_loss
from auction_bid import mix_scores as auction_mix_scores
# ── CENIE 第 3 维：官方 GMM 反密度（cenie_density.py 自包含副本）。jit 内求值 cenie_neg_logp、
#    jit 外每 eval epoch host 重拟 fit_visitation_gmm，参数 gmm_params 显式穿过 jit 边界（不进
#    flax TrainState，避免改 checkpoint/optimizer 结构）。冷启 valid=False → CENIE 返回 0。
from cenie_density import init_gmm_params, fit_visitation_gmm, cenie_neg_logp
# ── 方案 B 注入层：N 个 PCGRL generator 各被异质 estimator 信号驱动造 level 注入
#    固定 buffer，替代随机海选（GENERATOR_INJECTION 开关，默认 false 零回归）。
#    交替训练阶段 G = get_generator_set（冻 student 更新 generator + 产 instances）。
#    见 _sfl_repo_mirror/GENERATOR_DESIGN.md §0/§5/§6。pcgrl_generator 60/60 单测过。
from pcgrl_generator import (
    make_pcgrl_env, make_generator_network, make_generator_state, get_generator_set,
    GEN_ESTIMATOR_DIFFICULTY,
)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mask: jnp.ndarray
    info: jnp.ndarray

class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    targets: jnp.ndarray
    advantages: jnp.ndarray
    # carry: jnp.ndarray
    mask: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
        

@hydra.main(version_base=None, config_path="config", config_name="jaxnav-sfl")
def main(config):

    config = OmegaConf.to_container(config)
    # [A0 提速·纯工程零数值影响] 持久化编译+autotune 缓存：把 XLA 编译产物(含 autotune
    #   算法搜索结果)存盘，10-seed sweep 第 2 个 seed 起跳过重复编译/搜索。generator-on
    #   路径首次编译爆炸(单 generator 45s,N=3+student+auction 叠起来小时级)正是被此缓存摊掉。
    #   min_compile_time_secs=0 → 连小函数也缓存(我们要缓存的就是那个巨型 train_and_eval_step)。
    if config.get("COMPILE_CACHE", True):
        _cache_dir = config.get("COMPILE_CACHE_DIR", os.path.expanduser("~/jax_compile_cache"))
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        print(f"[A0] JAX 持久化编译缓存已开: {_cache_dir}", flush=True)
    run = wandb.init(
        group=config["GROUP_NAME"],
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "DR", f"ts: {config['env']['test_set']}"],
        config=config,
        mode=config["WANDB_MODE"],
    )
        
    rng = jax.random.PRNGKey(config["SEED"])
    
    assert (config["learning"]["NUM_ENVS_FROM_SAMPLED"] +  config["learning"]["NUM_ENVS_TO_GENERATE"]) == config["learning"]["NUM_ENVS"]
    
    env = JaxNav(num_agents=config["env"]["num_agents"],
                        **config["env"]["env_params"])  # use old config for env params to try reduce errors
    print('num agents', env.num_agents)
    t_config = config["learning"]
        
    t_config["NUM_ACTORS"] = env.num_agents * t_config["NUM_ENVS"]
    t_config["NUM_UPDATES"] = (
        t_config["TOTAL_TIMESTEPS"] // t_config["NUM_STEPS"] // t_config["NUM_ENVS"]
    )
    t_config["MINIBATCH_SIZE"] = (
        t_config["NUM_ACTORS"] * t_config["NUM_STEPS"] // t_config["NUM_MINIBATCHES"]
    )
    t_config["CLIP_EPS"] = (
        t_config["CLIP_EPS"] / env.num_agents
        if t_config["SCALE_CLIP_EPS"]
        else t_config["CLIP_EPS"]
    )
        
    network = ActorCriticRNN(env.agent_action_space().shape[0],
                             config=t_config)

    eval_singleton_runner = EvalSingletonsRunner(
        config["env"]["test_set"],
        network,
        init_carry=ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        env_kwargs=config["env"]["env_params"]
    )
    
    with open(config["EVAL_SAMPLED_SET_PATH"], "rb") as f:
      eval_env_instances = pickle.load(f)
    _, eval_init_states = jax.vmap(env.set_env_instance, in_axes=(0))(eval_env_instances)
    
    eval_sampled_runner = EvalSampledRunner(
        None,
        env,
        network,
        ScannedRNN.initialize_carry,
        hidden_size=t_config["HIDDEN_SIZE"],
        greedy=False,
        env_init_states=eval_init_states,
        n_episodes=10,
    )
    
    def linear_schedule(count):
        count = count // (t_config["NUM_MINIBATCHES"] * t_config["UPDATE_EPOCHS"])
        frac = (
            1.0 - count / t_config["NUM_UPDATES"]
        )
        return t_config["LR"] * frac
    
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, t_config["NUM_ENVS"], env.lidar_num_beams+5)  # NOTE hardcoded
        ),
        jnp.zeros((1, t_config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ENVS"], t_config["HIDDEN_SIZE"])
    network_params = network.init(_rng, init_hstate, init_x)
    if t_config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(t_config["MAX_GRAD_NORM"]),
            optax.adam(t_config["LR"], eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    rng, _rng = jax.random.split(rng)
    initial_singleton_test_metrics = eval_singleton_runner.run(_rng, train_state.params)  
    initial_sampled_test_metrics = eval_sampled_runner.run(_rng, train_state.params)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, t_config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
    
    
    if t_config["LAMBDA_SCHEDULE"]:
        raise NotImplementedError("Lambda schedule not implemented for finetuning")
        rng, lambda_rng = jax.random.split(rng)
        env_state = env_state.replace(
            rew_lambda = sample_lambda_set(lambda_rng, 0),
        )
    start_state = env_state
    init_hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
    
    
    
    @jax.jit
    def get_learnability_set(rng, network_params, gmm_params=None):
        # gmm_params: CENIE 的 GMM 参数（GMMParams pytree），由 eval-loop 每 epoch host 重拟后传入。
        # None（默认/CENIE 关）时 CENIE 不参与；valid=False 的冷启参数会让 cenie_neg_logp 返回 0。
        #
        # ── 诊断: AUCTION_STAGE 逐阶段加 estimator 计算，二分卡死元凶（编译/运行时）──
        #   0 = 只 difficulty（无 GAE reverse-scan、emit dummy hidden）
        #   1 = +PVL（加 GAE reverse-scan over ROLLOUT，仍 emit dummy hidden）
        #   2 = +CENIE（加真 hstate emit + GMM 求值）
        #   未开 AUCTION_SCORING 时 stage 无效（走原 p(1-p)）。一个 config 改值即可二分。
        _auc_stage = int(config.get("AUCTION_STAGE", 2)) if config.get("AUCTION_SCORING", False) else -1
        _emit_hidden = (_auc_stage >= 2)

        BATCH_ACTORS = config["BATCH_SIZE"] * env.num_agents


        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                hstate, pi, value, _ = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["BATCH_SIZE"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0)
                )(rng_step, env_state, env_act, start_state)
                if env.do_sep_reward:
                    reward = listify_reward(reward, do_batchify=True)
                else:
                    reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
                # train_mask = batchify(info["terminated"], env.agents, BATCH_ACTORS).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward,
                    log_prob.squeeze(),
                    obs_batch,
                    train_mask,
                    info,
                )
                runner_state = (env_state, start_state, obsv, done_batch, hstate, rng)
                # emit hstate（进入该步前的 GRU hidden，与 value/action 对齐）作 CENIE 特征。
                # 诊断: 只在 stage>=2(CENIE) emit 真 hidden；否则 emit 标量占位（scan 不物化巨型 hidden）。
                if _emit_hidden:
                    return runner_state, (transition, hstate)
                else:
                    return runner_state, (transition, jnp.zeros((), dtype=jnp.float32))
            
            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps: int, dones, returns, info):
                idxs = jnp.arange(max_steps)
                
                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx): 
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    success = jnp.sum(info["GoalR"] * mask)
                    collision = jnp.sum((info["MapC"] + info["AgentC"]) * mask)
                    timeo = jnp.sum(info["TimeO"] * mask)
                    l = end_idx - start_idx
                    return r, success, collision, timeo, l
                
                done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
                mask_done = jnp.where(done_idxs == max_steps, 0, 1)
                ep_return, success, collision, timeo, length = __ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)        
                        
                return {"ep_return": ep_return.mean(where=mask_done),
                        "num_episodes": mask_done.sum(),
                        "success_rate": success.mean(where=mask_done),
                        "collision_rate": collision.mean(where=mask_done),
                        "timeout_rate": timeo.mean(where=mask_done),
                        "ep_len": length.mean(where=mask_done),
                        }
            
            # sample envs
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            env_instances = EnvInstance(
                agent_pos=env_state.pos,
                agent_theta=env_state.theta,
                goal_pos=env_state.goal,
                map_data=env_state.map_data,
                rew_lambda=env_state.rew_lambda,
            )
            
            init_hstate = ScannedRNN.initialize_carry(BATCH_ACTORS, t_config["HIDDEN_SIZE"])
            
            runner_state = (env_state, env_state, obsv, jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, (traj_batch, hstates) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_STEPS"]
            )
            # hstates: (ROLLOUT_STEPS, BATCH_ACTORS, HIDDEN_SIZE)，CENIE 特征
            print('traj batch done', traj_batch.done.shape)
            print('traj batch info', traj_batch.info["NumC"].shape)
            done_by_env = traj_batch.done.reshape((-1, env.num_agents, config["BATCH_SIZE"]))
            reward_by_env = traj_batch.reward.reshape((-1, env.num_agents, config["BATCH_SIZE"]))
            info_by_actor = jax.tree_map(lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            print('done_by_env', done_by_env.shape)
            print('reward_by_env', reward_by_env.shape)
            print('info_by_actor', info_by_actor)
            o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done, traj_batch.reward, info_by_actor)
            print('o', o)
            success_by_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"]))
            learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
            print('learnability_by_env', learnability_by_env)

            # ── auction 海选额外信号（按 _auc_stage 分级算，诊断卡死元凶）──
            # difficulty-match = -(p-0.5)²（stage>=0，纯 outcome，无额外 scan）。
            p_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"])).mean(axis=0)
            nep_env = o["num_episodes"].reshape((env.num_agents, config["BATCH_SIZE"])).sum(axis=0)
            complete_env = nep_env > 0
            difficulty_by_env = jnp.where(complete_env, -((p_env - 0.5) ** 2), -jnp.inf)

            # PVL（stage>=1）：GAE reverse-scan over ROLLOUT。stage 0 时跳过（占位 0）。
            if _auc_stage >= 1:
                last_val = traj_batch.value[-1]
                def _get_adv(gae_next, transition):
                    gae, next_value = gae_next
                    done, value, reward = transition.global_done, transition.value, transition.reward
                    delta = reward + t_config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + t_config["GAMMA"] * t_config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae
                _, advantages = jax.lax.scan(
                    _get_adv, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                pvl_by_env = positive_value_loss(traj_batch.done, advantages)
                pvl_by_env = pvl_by_env.reshape((env.num_agents, config["BATCH_SIZE"])).mean(axis=0)
            else:
                pvl_by_env = jnp.zeros_like(difficulty_by_env)

            # CENIE（stage>=2）：真 hstates emit + GMM 求值（subsample 防 OOM）。
            if _auc_stage >= 2 and gmm_params is not None:
                H = hstates.shape[-1]
                Kc = int(config.get("PROBE_HIDDEN_STEPS", 16))
                t_idx = jnp.linspace(0, config["ROLLOUT_STEPS"] - 1, Kc).astype(jnp.int32)
                hs_sub = hstates[t_idx]                                          # (Kc, BATCH_ACTORS, H)
                nlp = cenie_neg_logp(gmm_params, hs_sub.reshape(-1, H))
                nlp = nlp.reshape(Kc, BATCH_ACTORS).mean(axis=0)
                cenie_by_env = nlp.reshape((env.num_agents, config["BATCH_SIZE"])).mean(axis=0)
                cenie_by_env = jnp.where(complete_env, cenie_by_env, -jnp.inf)
            else:
                cenie_by_env = jnp.full_like(difficulty_by_env, -jnp.inf)        # 占位，外层 N<3 不用
            # [护栏仪表盘 §injp] 透出 per-level 真实 p（海选 pool），与 generator 路径同口径 gen_p_*。
            #   incomplete(nep=0)→ p 置 nan，下游算占比时排除。
            p_env_out = jnp.where(nep_env > 0, p_env, jnp.nan)
            return None, (learnability_by_env, env_instances,
                          difficulty_by_env, pvl_by_env, cenie_by_env, p_env_out)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (learnability, env_instances, difficulty, pvl, cenie, p_pool) = jax.lax.scan(
            _batch_step, None, rngs, config["NUM_BATCHES"])

        flat_env_instances = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        learnability = learnability.flatten()
        difficulty = difficulty.flatten()                                       # (M,)
        pvl = pvl.flatten()                                                      # (M,)
        cenie = cenie.flatten()                                                  # (M,)
        p_pool = p_pool.flatten()                                                # (M,) 海选 pool 真实 p

        # ── 打分：auction 混合 estimator 替代单一 p(1-p)，开关 AUCTION_SCORING ──
        # 向后兼容：默认关→走原 learnability top-K（基线对照不变）；开→auction 混合分 top-K。
        # estimator 集：gmm_params 传入(CENIE 开)→[difficulty,PVL,CENIE] N=3；否则 [difficulty,PVL] N=2。
        # auction_lambda 跑必要性消融：inf=argmax(single-winner)、有限大=fractional/uniform。
        if config.get("AUCTION_SCORING", False):
            # 按 _auc_stage 选 estimator（诊断分级 = 最终 N）：0→[difficulty]、1→[difficulty,PVL]、
            # 2→[difficulty,PVL,CENIE]（CENIE 还需 gmm_params 非 None）。
            ests = [difficulty]
            if _auc_stage >= 1:
                ests.append(pvl)
            if _auc_stage >= 2 and gmm_params is not None:
                ests.append(cenie)
            per_est = jnp.stack(ests, axis=0)                                    # (N, M)
            mixed, w, bids = auction_mix_scores(
                per_est, float(config.get("AUCTION_LAMBDA", 1.0)))               # (M,)
            print('auction weights', w, 'bids', bids, 'stage', _auc_stage, 'N', len(ests))
            score = mixed
        else:
            score = learnability
        top_1000 = jnp.argsort(score)[-config["NUM_TO_SAVE"]:]
        print('top 1000', top_1000)

        top_1000_instances = jax.tree_map(lambda x: x.at[top_1000].get(), flat_env_instances)
        print('top 1000 instances', top_1000_instances)
        # 第 3 个返回值 p_pool=海选 pool 全体真实 p（与 generator 路径 gen_p_* 同口径，仪表盘用）。
        return score.at[top_1000].get(), top_1000_instances, p_pool


    def get_probe_signals(rng, network_params):
        """V1 正交性探针 rollout（jit 内，只产 host 侧原料；GMM/相关在 jit 外 train_probe 算）。

        独立于 get_learnability_set（零回归）：同样的 reset→rollout，但额外 emit
          ① per-level success_rate / num_episodes（难度+learnability 信号）；
          ② dones + GAE advantages（PVL 信号，照 train_step 的 _calculate_gae 同法补算）；
          ③ 每步 GRU hidden + per_level_index（CENIE 信号，host 侧拟 GMM 用）。
        返回设备数组 dict（调用方 device_get 后喂 train_probe.log_orthogonality_step）。
        """
        BATCH_ACTORS = config["BATCH_SIZE"] * env.num_agents

        def _batch_step(unused, rng):
            def _env_step(runner_state, unused):
                env_state, start_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, BATCH_ACTORS)
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                new_hstate, pi, value, _ = network.apply(network_params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["BATCH_SIZE"], env.num_agents)
                env_act = {k: v.squeeze() for k, v in env_act.items()}
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["BATCH_SIZE"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, 0))(rng_step, env_state, env_act, start_state)
                if env.do_sep_reward:
                    reward = listify_reward(reward, do_batchify=True)
                else:
                    reward = batchify(reward, env.agents, BATCH_ACTORS).squeeze()
                done_batch = batchify(done, env.agents, BATCH_ACTORS).squeeze()
                train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents), last_done,
                    action.squeeze(), value.squeeze(), reward, log_prob.squeeze(),
                    obs_batch, train_mask, info,
                )
                # hstate 是 emit 给 CENIE 的 GRU hidden（(BATCH_ACTORS, HIDDEN_SIZE)）。
                # 取 last_obs 对应的 hstate（即进入这步前的 hidden），与 value 对齐。
                runner_state = (env_state, start_state, obsv, done_batch, new_hstate, rng)
                return runner_state, (transition, hstate)

            # 同 _calc_outcomes_by_agent（复算 success_rate / num_episodes，per-actor）。
            @partial(jax.vmap, in_axes=(None, 1, 1, 1))
            @partial(jax.jit, static_argnums=(0,))
            def _calc_outcomes_by_agent(max_steps, dones, returns, info):
                idxs = jnp.arange(max_steps)
                @partial(jax.vmap, in_axes=(0, 0))
                def __ep_outcomes(start_idx, end_idx):
                    mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                    r = jnp.sum(returns * mask)
                    success = jnp.sum(info["GoalR"] * mask)
                    l = end_idx - start_idx
                    return r, success, l
                done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
                mask_done = jnp.where(done_idxs == max_steps, 0, 1)
                ep_return, success, length = __ep_outcomes(
                    jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)
                return {"num_episodes": mask_done.sum(),
                        "success_rate": success.mean(where=mask_done)}

            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["BATCH_SIZE"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            init_hstate = ScannedRNN.initialize_carry(BATCH_ACTORS, t_config["HIDDEN_SIZE"])
            runner_state = (env_state, env_state, obsv,
                            jnp.zeros((BATCH_ACTORS), dtype=bool), init_hstate, rng)
            runner_state, (traj_batch, hstates) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_STEPS"])
            # hstates: (ROLLOUT_STEPS, BATCH_ACTORS, HIDDEN_SIZE)
            # ⚠ 内存：全量 hidden = NB×ROLLOUT_STEPS×BATCH_ACTORS×H ≈ 5×1000×1000×512×4B
            #   ≈ 10GB（GPU scan 累积爆显存 + device_get 爆 24G host RAM → OOM）。CENIE 的
            #   GMM 只需少量代表性 hidden 估密度（官方 cenie_subsample=4096），探针测正交性
            #   更无需逐步全量。故**在 jit 内沿时间维均匀抽 PROBE_HIDDEN_STEPS 步**，把 hidden
            #   降到 5×K×1000×512（K=16 → ~160MB），既不爆显存也不爆 host。
            K = int(config.get("PROBE_HIDDEN_STEPS", 16))
            t_idx = jnp.linspace(0, config["ROLLOUT_STEPS"] - 1, K).astype(jnp.int32)
            hstates = hstates[t_idx]                          # (K, BATCH_ACTORS, HIDDEN_SIZE)

            # —— PVL：照 train_step 的 _calculate_gae 同法补算 advantages（last_val 用末步 value）——
            last_val = traj_batch.value[-1]                                  # (BATCH_ACTORS,)
            def _get_adv(gae_next, transition):
                gae, next_value = gae_next
                done, value, reward = transition.global_done, transition.value, transition.reward
                delta = reward + t_config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + t_config["GAMMA"] * t_config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae
            _, advantages = jax.lax.scan(
                _get_adv, (jnp.zeros_like(last_val), last_val), traj_batch,
                reverse=True, unroll=16)                                     # (ROLLOUT_STEPS, BATCH_ACTORS)

            info_by_actor = jax.tree_map(
                lambda x: x.swapaxes(2, 1).reshape((-1, BATCH_ACTORS)), traj_batch.info)
            o = _calc_outcomes_by_agent(config["ROLLOUT_STEPS"], traj_batch.done,
                                        traj_batch.reward, info_by_actor)
            # per-env（对 num_agents 维聚合；num_agents=1 时即原值）。
            succ_env = o["success_rate"].reshape((env.num_agents, config["BATCH_SIZE"])).mean(axis=0)
            nep_env = o["num_episodes"].reshape((env.num_agents, config["BATCH_SIZE"])).sum(axis=0)
            return None, {
                "success_rate": succ_env,                     # (BATCH_SIZE,)
                "num_episodes": nep_env,                       # (BATCH_SIZE,)
                "dones": traj_batch.done,                      # (ROLLOUT_STEPS, BATCH_ACTORS)
                "advantages": advantages,                      # (ROLLOUT_STEPS, BATCH_ACTORS)
                "hidden": hstates,                             # (ROLLOUT_STEPS, BATCH_ACTORS, H)
            }

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, batched = jax.lax.scan(_batch_step, None, rngs, config["NUM_BATCHES"])
        # 跨 NUM_BATCHES 拼成全量 per-level（与 get_learnability_set 的 flatten 同序）。
        # success_rate/num_episodes: (NUM_BATCHES, BATCH_SIZE) → (M,)
        success_rate = batched["success_rate"].reshape(-1)                  # (M,)
        num_episodes = batched["num_episodes"].reshape(-1)                  # (M,)
        M = success_rate.shape[0]
        # dones/advantages: (NUM_BATCHES, ROLLOUT_STEPS, BATCH_ACTORS) → (ROLLOUT_STEPS, M_actors)
        # PVL 按 (T, M) 喂；这里 M_actors = NUM_BATCHES*BATCH_ACTORS，num_agents=1 时 = M。
        def _stack_TM(x):  # (NB, T, A) → (T, NB*A)
            return x.transpose(1, 0, 2).reshape(x.shape[1], -1)
        dones_TM = _stack_TM(batched["dones"])                             # (T, M_actors)
        adv_TM = _stack_TM(batched["advantages"])                          # (T, M_actors)
        # hidden: (NB, K, A, H) → (NB*K*A, H)；per_level_index = 每行的 env 列（0..M-1）。
        # K = PROBE_HIDDEN_STEPS（已在 jit 内沿时间维抽样，非全 ROLLOUT_STEPS）。
        nb, K, A = batched["hidden"].shape[:3]
        H = batched["hidden"].shape[-1]
        hid = batched["hidden"].reshape(-1, H)                             # (NB*K*A, H)
        # reshape 后行序 = (nb, k, a)。env 列号 = nb*BATCH_SIZE + (a % BATCH_SIZE)
        # （num_agents=1 时 A=BATCH_SIZE，a%BATCH_SIZE=a）。用 broadcasting 构造，避免 tile 易错。
        nb_g = jnp.arange(nb)[:, None, None]                               # (nb,1,1)
        a_g = jnp.arange(A)[None, None, :]                                 # (1,1,A)
        pli_grid = (nb_g * config["BATCH_SIZE"] + (a_g % config["BATCH_SIZE"]))
        pli_grid = jnp.broadcast_to(pli_grid, (nb, K, A))                  # (nb,K,A)
        per_level_index = pli_grid.reshape(-1).astype(jnp.int32)          # (NB*K*A,)
        return {
            "success_rate": success_rate, "num_episodes": num_episodes,
            "dones": dones_TM, "advantages": adv_TM,
            "hidden_feats": hid, "per_level_index": per_level_index, "n_levels": M,
        }


    # TRAIN LOOP
    def train_step(runner_state_instances, unused):
        # COLLECT TRAJECTORIES
        runner_state, instances = runner_state_instances
        num_env_instances = instances.agent_pos.shape[0]

        def _env_step(runner_state, unused):
            train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            hstate, pi, value, dormancy = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            env_act = unbatchify(
                action, env.agents, t_config["NUM_ENVS"], env.num_agents
            )
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, t_config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, 0)
            )(rng_step, env_state, env_act, start_state)
            if env.do_sep_reward:
                reward = listify_reward(reward, do_batchify=True)
            else:
                reward = batchify(reward, env.agents, t_config["NUM_ACTORS"]).squeeze()
            done_batch = batchify(done, env.agents, t_config["NUM_ACTORS"]).squeeze()
            train_mask = info["terminated"].swapaxes(0, 1).reshape(-1)
            # train_mask = batchify(info["terminated"], env.agents, t_config["NUM_ACTORS"]).squeeze()
            transition = Transition(
                jnp.tile(done["__all__"], env.num_agents),
                last_done,
                action.squeeze(),
                value.squeeze(),
                reward,
                log_prob.squeeze(),
                obs_batch,
                train_mask,
                info,
            )
            runner_state = (train_state, env_state, start_state, obsv, done_batch, hstate, update_steps, rng)
            return runner_state, (transition, dormancy)

        initial_hstate = runner_state[-3]
        runner_state, traj_batch_dormancy = jax.lax.scan(
            _env_step, runner_state, None, t_config["NUM_STEPS"]
        )
        traj_batch, dormancy = traj_batch_dormancy
        dormancy = jax.tree_map(lambda x: x.mean(), dormancy)
        
        @partial(jax.vmap, in_axes=(1, 1))
        def _calc_ep_return_by_agent(dones, returns):
            idxs = jnp.arange(t_config["NUM_STEPS"])
            
            @partial(jax.vmap, in_axes=(None, 0, 0))
            def __ep_returns(rews, start_idx, end_idx): 
                mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != t_config["NUM_STEPS"])
                r = jnp.sum(rews * mask, axis=0)
                l = end_idx - start_idx
                return r, l
            
            done_idxs = jnp.argwhere(dones, size=t_config["NUM_STEPS"]//4, fill_value=t_config["NUM_STEPS"]).squeeze()
            mask_done = jnp.where(done_idxs == t_config["NUM_STEPS"], 0, 1)
            r, l = __ep_returns(returns, jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)                
            return {"episodic_return_per_agent": r.mean(where=mask_done), "episodic_length_per_agent": l.mean(where=mask_done)}
        
        if env.do_sep_reward:
            reward_by_env = traj_batch.reward.sum(axis=-1)
        else:
            reward_by_env = traj_batch.reward
        episodic_return_length = _calc_ep_return_by_agent(traj_batch.done, reward_by_env)
        episodic_return_length = jax.tree_map(lambda x: x.mean(), episodic_return_length)
        # CALCULATE ADVANTAGE
        train_state, env_state, start_state, last_obs, last_done, hstate, update_steps, rng = runner_state
        last_obs_batch = batchify(last_obs, env.agents, t_config["NUM_ACTORS"])
        ac_in = (
            last_obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze()
        print('last_val shape', last_val.shape)
        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition: Transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.global_done, 
                    transition.value,
                    transition.reward,
                )
                delta = reward + t_config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + t_config["GAMMA"] * t_config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn_masked(params, init_hstate, traj_batch, gae, targets):
                                            
                    # RERUN NETWORK
                    _, pi, value, _ = network.apply(
                        params,
                        init_hstate.transpose(),
                        (traj_batch.obs, traj_batch.done),
                    )
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-t_config["CLIP_EPS"], t_config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(
                        value_losses, value_losses_clipped
                    )
                    if env.do_sep_reward:
                        value_loss_sparse = value_loss[..., REWARD_COMPONENT_SPARSE].mean(where=(1 - traj_batch.mask))
                        value_loss_dense  = value_loss[..., REWARD_COMPONENT_DENSE].mean(where=(1 - traj_batch.mask))
                        
                        critic_loss = t_config["VF_COEF"] * (value_loss_sparse + value_loss_dense)
                    else:
                        critic_loss = t_config["VF_COEF"] * value_loss.mean(where=(1 - traj_batch.mask))
                    
                    # CALCULATE ACTOR LOSS
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    if env.do_sep_reward:
                        gae = gae.sum(axis=-1)
                    gae = (gae - gae.mean(where=(1-traj_batch.mask))) / (gae.std(where=(1-traj_batch.mask)) + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - t_config["CLIP_EPS"],
                            1.0 + t_config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean(where=(1 - traj_batch.mask))
                    entropy = pi.entropy().mean(where=(1 - traj_batch.mask))
                    
                    # debug
                    approx_kl = jax.lax.stop_gradient(
                        ((ratio - 1) - logratio).mean()
                    )
                    clipfrac = jax.lax.stop_gradient(
                        (jnp.abs(ratio - 1) > t_config["CLIP_EPS"]).mean()
                    )

                    total_loss = (
                        loss_actor
                        + critic_loss
                        - t_config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clipfrac)

                grad_fn = jax.value_and_grad(_loss_fn_masked, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, _rng = jax.random.split(rng)

            init_hstate = jnp.reshape(
                init_hstate, (t_config["HIDDEN_SIZE"], t_config["NUM_ACTORS"])
            )
            batch = (
                init_hstate,
                traj_batch,
                advantages.squeeze(),
                targets.squeeze(),
            )
            permutation = jax.random.permutation(_rng, t_config["NUM_ACTORS"])

            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )

            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], t_config["NUM_MINIBATCHES"], -1]
                        + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            # total_loss = jax.tree_map(lambda x: x.mean(), total_loss)
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        # init_hstate = initial_hstate[None, :].squeeze().transpose()
        init_hstate = jax.tree_map(lambda x: x[None, :].squeeze().transpose(), initial_hstate)
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, t_config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        metric = traj_batch.info
        metric = jax.tree_map(
            lambda x: x.sum(axis=-1).reshape(
                (t_config["NUM_STEPS"], t_config["NUM_ENVS"])  # , env.num_agents
            ),
            traj_batch.info,
        )
        rng = update_state[-1]

        def callback(metric):
            wandb.log(
                {
                    "train-term": metric["terminations"],
                    #"reward": metric["returned_episode_returns"],
                    
                    # "eval-collision": metric["test-metrics"]["collision-by-env"].mean(),
                    # "eval-timeout": metric["test-metrics"]["timeout-by-env"].mean(),
                    "env_step": metric["update_steps"]
                        * t_config["NUM_ENVS"]
                        * t_config["NUM_STEPS"],
                    "dormancy/": metric["dormancy"],
                    # [卡死修复] env-metrics 已移除(每update的Dijkstra致generator地图卡死,纯日志无方法影响)。
                    # "mean_ued_score": metric["mean_ued_score"],
                    **metric["episodic_return_length"],
                    **metric["loss_info"],
                    "mean_lambda_val": metric["mean_lambda_val"],
                }
            )

        dormancy_log = {
            "actor": dormancy.actor,
            "embedding": dormancy.embedding,
            "hidden": dormancy.hidden,
            "rnnout": dormancy.rnnout,
            "critic": dormancy.critic,
        }
        ratio0 = jnp.around(loss_info[1][3].at[0,0].get().mean(), decimals=6)
        loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
        metric["loss_info"] = {
            "total_loss": loss_info[0],
            "value_loss": loss_info[1][0],
            "actor_loss": loss_info[1][1],
            "entropy": loss_info[1][2],
            "ratio": loss_info[1][3],
            "ratio_0": ratio0,
            "approx_kl": loss_info[1][4],
            "clipfrac": loss_info[1][5],
            "mask_percentage": jnp.mean(traj_batch.mask),
        }
        metric["episodic_return_length"] = episodic_return_length
        metric["update_steps"] = update_steps
        metric["terminations"] = {k: traj_batch.info[k] for k in ["NumC", "GoalR", "AgentC", "MapC", "TimeO"]}
        metric["terminations"] = jax.tree_map(lambda x: x.sum(), metric["terminations"])
        metric["dormancy"] = dormancy_log
        # [卡死修复·零方法影响] 移除 env-metrics。原 `jax.vmap(env.get_env_metrics)(start_state)`
        #   每 update 对 256 env 各跑一次 dikstra_path(grid_map.py:411 while_loop 迭代到开集为空、
        #   无提前终止、步数数据依赖地图)。generator 的 PCGRL 迷宫地图让 Dijkstra 迭代爆炸,经
        #   ordered io_callback 串到每 update 同步点 → student scan 卡死(8h 黑洞真因,见
        #   STAGE4_提速方案.md §0)。env-metrics 仅 wandb 训练曲线日志,不参与课程/student PPO/
        #   SOTA 评估,且 get_env_metrics 注释明示"only valid for grid map type"(jaxnav 是连续避障)。
        #   删除即根除卡死,零数值/方法影响。剥离对照实测:student-scan RUN 70.6s 跑通 vs 带它卡死 15min+。
        metric["mean_lambda_val"] = env_state.rew_lambda.mean()
        jax.experimental.io_callback(callback, None, metric)

        # SAMPLE NEW ENVS
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, t_config["NUM_ENVS_TO_GENERATE"])
        obsv_gen, env_state_gen = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        
        rng, _rng = jax.random.split(rng)
        sampled_env_instances_idxs = jax.random.randint(_rng, (t_config["NUM_ENVS_FROM_SAMPLED"],), 0, num_env_instances)
        sampled_env_instances = jax.tree_map(lambda x: x.at[sampled_env_instances_idxs].get(), instances)
        obsv_sampled, env_state_sampled = jax.vmap(env.set_env_instance, in_axes=(0,))(sampled_env_instances)
        
        obsv = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), obsv_gen, obsv_sampled)
        env_state = jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0), env_state_gen, env_state_sampled)
        
        start_state = env_state
        hstate = ScannedRNN.initialize_carry(t_config["NUM_ACTORS"], t_config["HIDDEN_SIZE"])
        
        update_steps = update_steps + 1
        runner_state = (train_state, env_state, start_state, obsv, jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool), hstate, update_steps, rng)
        return (runner_state, instances), metric
    
    def log_buffer(learnability, states, epoch):
        num_samples = states.pos.shape[0]
        rows = 2 
        fig, axes = plt.subplots(rows, int(num_samples/rows), figsize=(20, 10))
        axes=axes.flatten()
        for i, ax in enumerate(axes):
            # ax.imshow(train_state.plr_buffer.get_sample(i))
            score = learnability[i]            
            state = jax.tree_map(lambda x: x[i], states)
                        
            env.init_render(ax, state, lidar=False, ticks_off=True)
            ax.set_title(f'learnability: {score:.3f}')
            ax.set_aspect('equal', 'box')
                        
        plt.tight_layout()
        fig.canvas.draw()
        im = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba()) 
        wandb.log({"maps": wandb.Image(im)}, step=epoch)
    
    # [拆 jit 提速·纯工程零数值影响] 把 generator 段抽成独立 jit 编译单元。
    #   原因：原 train_and_eval_step 把 N=3 generator(各 363步scan+PPO,Python for 展开 3 份)
    #   + 50步 student scan + auction + CENIE 全内联进一个巨型 jit，XLA 编译复杂度超线性爆炸
    #   (实测单 seed 卡编译 >8h 编不完;而单独 jit 的 gen_train_iter 仅 45s)。拆成独立 jit 后
    #   generator 段单独编译、student 段单独编译,图规模回到可控。
    #   纯工程:generator 先训→student 后训的顺序与数据流完全不变,数值不变,不碰课程/方法。
    @jax.jit
    def gen_phase_jit(student_params, gen_state, learnability_rng, gmm_params, curriculum_thr):
        """generator 注入段(独立编译)。返回 (scores, instances, new_gen_state, gen_metrics)。

        curriculum_thr: 当前难度上限标量(host 侧按 eval_step 算,每 epoch 变,故穿 jit 参数;
                        arm=none 时传 0.0 占位,get_generator_set 内 arm=none 不读它)。
        """
        return get_generator_set(
            learnability_rng, student_params, gen_state, gen_optimizer,
            gen_estimator_ids, gen_env, gen_network, env, network,
            t_config["HIDDEN_SIZE"], config["ROLLOUT_STEPS"],
            num_levels_per_gen=gen_num_levels, num_to_save=gen_num_to_save,
            place_env=gen_place_env, gmm_params=gmm_params,
            auction_lambda=gen_auction_lambda,
            signal_mode=_auction_signal_mode, auction_weight_factors=_auction_weight_factors,
            alp_quota_coef=float(config.get("ALP_QUOTA_COEF", 0.0)),
            gate_weight=float(config.get("GATE_WEIGHT", 0.0)),
            gate_w_hard=float(config.get("GATE_W_HARD", 2.0)),
            gate_w_easy=float(config.get("GATE_W_EASY", 1.0)),
            curriculum_arm=_curriculum_arm, curriculum_thr=curriculum_thr,
            curriculum_beta=_curriculum_beta,
            gen_outer_steps=gen_outer_steps, ppo_epochs=gen_ppo_epochs,
            gen_rollout_chunk=gen_rollout_chunk, pool_per_gen=gen_pool_per_gen,
            gamma=t_config["GAMMA"], gae_lambda=t_config["GAE_LAMBDA"])

    @jax.jit
    def learnability_phase_jit(student_params, learnability_rng, gmm_params):
        """随机海选段(GENERATOR_INJECTION=false,独立编译)。返回 (scores, instances, p_pool)。"""
        return get_learnability_set(learnability_rng, student_params, gmm_params)

    @partial(jax.jit, static_argnums=())
    def train_and_eval_step(runner_state, eval_rng, instances, learnabilty_scores):
        # [拆 jit] generator/海选段已在外部独立 jit 算好,这里只接 instances/scores 作参数。
        #   本 jit 现在只含: student 50步 scan + 评估,图规模大幅缩小、编译快。
        _, eval_singleton_rng, eval_sampled_rng = jax.random.split(eval_rng, 3)
        runner_state_instances = (runner_state, instances)
        runner_state_instances, metrics = jax.lax.scan(train_step, runner_state_instances, None, t_config["EVAL_FREQ"])
        # EVAL

        test_metrics = {
            "learnability_set_scores": learnabilty_scores,
            "learnability_set_mean_score": learnabilty_scores.mean(),
            # STAGE4 §1.3 第二指标：方差复活 + 分布。learnability_set_scores 已是全量打分数组，
            # 离线可直接算 var/std/p_std；这里额外落盘 jit 内方差，省得事后重算。
            "learnability_set_var": learnabilty_scores.var(),
        }
        # [拆 jit] gen_metrics 的 wandb 键处理移到 eval loop(host 层)，这里只算核心 test_metrics。
        test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, runner_state[0].params)
        test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, runner_state[0].params)

        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]

        top_instances = jax.tree_map(lambda x: x.at[-20:].get(), instances)
        _, top_states = jax.vmap(env.set_env_instance)(top_instances)

        return runner_state, (learnabilty_scores.at[-20:].get(), top_states), test_metrics

    rng, _rng = jax.random.split(rng)
    runner_state = (
        train_state,
        env_state,
        start_state,
        obsv,
        jnp.zeros((t_config["NUM_ACTORS"]), dtype=bool),
        init_hstate,
        0,
        _rng,
    )
    # max(...,1)：小规模（NUM_UPDATES//EVAL_FREQ < NUM_CHECKPOINTS）下原式会得 0，
    # 致下方 eval_step % checkpoint_steps 除零崩。钳到 ≥1 让小规模冒烟/CPU 短跑健壮。
    checkpoint_steps = max(1, t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"] // t_config["NUM_CHECKPOINTS"])
    print('eval freq', t_config["EVAL_FREQ"])
    # V1 探针：相关轨迹 + checkpoint 都落到 run 专属目录。
    probe_enabled = bool(config.get("PROBE_ORTHOGONALITY", True))
    probe_out_dir = os.path.join(config["SAVE_PATH"] or "checkpoints", run.name, "probe") if config["SAVE_PATH"] else None

    # ── CENIE 第 3 维：GMM 参数显式穿过 jit 边界（不进 flax TrainState）。──
    # AUCTION_USE_CENIE 开 → 冷启 init_gmm_params(valid=False，前几 epoch CENIE 返回 0)；
    # 每 eval epoch 用探针收集的 hidden host 重拟（fit_visitation_gmm），写回供下一 epoch。
    # 关 → gmm_params=None，auction 走 N=2(difficulty+PVL)。
    auction_use_cenie = bool(config.get("AUCTION_USE_CENIE", False))
    cenie_max_k = int(config.get("CENIE_MAX_COMPONENTS", 15))
    cenie_min_k = int(config.get("CENIE_MIN_COMPONENTS", 6))
    cenie_subsample = int(config.get("CENIE_SUBSAMPLE", 4096))
    cenie_reg_covar = float(config.get("CENIE_REG_COVAR", 1e-2))
    gmm_params = None
    if auction_use_cenie:
        gmm_params = init_gmm_params(cenie_max_k, t_config["HIDDEN_SIZE"])       # valid=False 冷启

    # ── 方案 B generator 注入（GENERATOR_INJECTION，默认 false 零回归）──
    # 开 → 建 N 个 PCGRL generator（独立 params/opt_state，宿主 Python 持有，与
    # gmm_params 同模式穿过 train_and_eval_step），每 outer round 阶段 G 用冻结的
    # student 训 generator 再造 level 注入；关 → gen_state=None，走原随机海选。
    generator_injection = bool(config.get("GENERATOR_INJECTION", False))
    gen_state = None
    gen_optimizer = None
    gen_env = gen_network = None
    gen_place_env = None      # 放起终点用 valid_path_check=True env（§R5，根除不可解图）
    # generator 数 N 与各自 estimator 绑定（冒烟 N=1 全 difficulty；N=3 各绑
    # difficulty/PVL/CENIE 是 part2 待做）。estimator_ids 长度即 N。提到块外定义，
    # 保证 train_and_eval_step 闭包在两分支下都能解析此名（关时是空 list，不被引用）。
    gen_estimator_ids = list(config.get("GEN_ESTIMATOR_IDS", [GEN_ESTIMATOR_DIFFICULTY])) \
        if generator_injection else []
    gen_num_levels = int(config.get("GEN_NUM_LEVELS_PER_GEN", 32))
    # auction 漏斗输出宽度（注入 buffer 的候选课程数）：默认沿用全局 NUM_TO_SAVE；
    # 须 < pool(=N×gen_num_levels)，让漏斗按 score 排序时把 incomplete(-inf) level 挤掉。
    gen_num_to_save = int(config.get("GEN_NUM_TO_SAVE", config["NUM_TO_SAVE"]))
    gen_outer_steps = int(config.get("GEN_OUTER_STEPS", 1))
    gen_ppo_epochs = int(config.get("GEN_PPO_EPOCHS", 4))
    # [大基数 OOM 修复 2026-06-26] 注入候选池与 PPO 训练 level 数解耦 + 分批造关/测信号：
    #   GEN_POOL_PER_GEN：每 gen **注入池**关数（造关+测信号，无 PPO backprop，可大）。
    #     <=0(默认) → 退回 gen_num_levels（pool=PPO 数，零回归，旧 sbatch 不变）。
    #   GEN_ROLLOUT_CHUNK：分批每批关数（仿 baseline NUM_BATCHES，峰值显存=单 chunk）。
    #     <=0(默认) → 不分批走一次性路径（零回归）。大基数时设 250-500（pool/chunk=批数，别太多致编译慢）。
    #   ⚠ gen_num_levels(PPO) 在 pool 大时仍应保持小（如 64），否则 gen_train_iter PPO backprop OOM。
    gen_pool_per_gen = int(config.get("GEN_POOL_PER_GEN", 0))
    gen_rollout_chunk = int(config.get("GEN_ROLLOUT_CHUNK", 0))
    # auction 出价漏斗温度（idea 核心）：N estimator 对整 pool 出价混合决定注入哪些课程。
    # None/"none" → fallback 每 gen 自评（消融）；"inf" → single-winner；有限值 → fractional。
    _gal = config.get("GEN_AUCTION_LAMBDA", 1.0)
    if _gal is None or (isinstance(_gal, str) and _gal.lower() == "none"):
        gen_auction_lambda = None
    elif isinstance(_gal, str) and _gal.lower() == "inf":
        gen_auction_lambda = float("inf")
    else:
        gen_auction_lambda = float(_gal)
    # ── [护栏 §injp] GEN_LEARNGATE CLI 覆盖：可学性门总开关，用于 on/off 对照实验 ──
    #   GEN_LEARNGATE 是 pcgrl_generator 模块常量，terminal_reward_euc 在 jit trace 时按 Python
    #   静态分支读取。运行前 setattr 覆盖 → trace 读到新值 → 生效。CLI: GEN_LEARNGATE=false 关门
    #   （=纯 euc，护栏前对照基线）；GEN_LEARNGATE_SIGMA=<float> 调门宽。默认沿用模块常量(True/0.25)。
    if generator_injection:
        import pcgrl_generator as _pg
        if "GEN_LEARNGATE" in config:
            _lg = config["GEN_LEARNGATE"]
            _lg = (_lg.lower() == "true") if isinstance(_lg, str) else bool(_lg)
            _pg.GEN_LEARNGATE = _lg
        if "GEN_LEARNGATE_SIGMA" in config:
            _pg.GEN_LEARNGATE_SIGMA = float(config["GEN_LEARNGATE_SIGMA"])
        print(f"[generator] GEN_LEARNGATE={_pg.GEN_LEARNGATE} sigma={_pg.GEN_LEARNGATE_SIGMA}")
    # ── [结构课程 §结构课程方案定稿 2026-06-25] 事前引导 + 固定时间表 ──
    #   CURRICULUM_ARM: none(关闭=旧行为) / geo(测地绝对长度,主轴) / fill(墙占比,反例对照)。
    #   塑形项 -β·relu(struct-thr(t)) 加在 generator terminal reward 上,把生成分布推向当前难度档。
    #   thr(t) host 侧按 eval_step 算(分三档),作标量穿 gen_phase_jit(与 gmm_params 同模式)。
    #   arm/beta 是静态(行为分支),thr 每 epoch 变故穿参。none 时 thr 不参与、塑形跳过。
    _curriculum_arm = str(config.get("CURRICULUM_ARM", "none")).lower()
    _curriculum_beta = float(config.get("CURRICULUM_BETA", 1.0))
    # 三档阈值默认 = 测地标定(5/10/22格); fill arm 须在 config 改成 fill 量纲(如 0.30/0.45/0.60)。
    _curr_thr_easy = float(config.get("CURRICULUM_THR_EASY", 5.0))
    _curr_thr_mid = float(config.get("CURRICULUM_THR_MID", 10.0))
    _curr_thr_hard = float(config.get("CURRICULUM_THR_HARD", 22.0))
    # ── [自适应课程推进 2026-06-26] 由 student 能力(注入关高 p 占比)驱动升档，替换固定时间表 ──
    #   CURRICULUM_ADAPTIVE=true → 状态机推进(_curr_stage 跨 eval 维护，由 frac(p>0.8) 触发升档)；
    #   false(默认) → 旧固定时间表(按 eval_step 进度三等分)，保证 arm=none/旧 arm 零回归。
    #   CURRICULUM_P_THRESHOLD τ: 每 eval 若 frac(p>0.8)>τ 则连续计数+1，连续 2 次超 τ 升一档(单调不退)。
    _curriculum_adaptive = config.get("CURRICULUM_ADAPTIVE", False)
    _curriculum_adaptive = (_curriculum_adaptive.lower() == "true") \
        if isinstance(_curriculum_adaptive, str) else bool(_curriculum_adaptive)
    _curriculum_p_threshold = float(config.get("CURRICULUM_P_THRESHOLD", 0.6))
    _curr_thr_by_stage = [_curr_thr_easy, _curr_thr_mid, _curr_thr_hard]
    # ── [双向滑落退档 2026-06-26] 允许退档但有地板，防 student 在中档轻松时瞬时升 hard 卡死 ──
    #   CURRICULUM_ALLOW_DEMOTE=true → 在升档基础上加退档：连续 2 次 frac(p<0.2)>τ 且未触地板则降一档。
    #   地板 floor = max(_max_stage_reached-1, 0)：升过 hard(max=2)则永不退回 easy(地板=mid=1)。
    #   false(默认) → 单调升档(只升不退，对照组)，保证已实现的单调 arm 零回归。
    #   升/退两个 streak 独立计数（升看 p>0.8，退看 p<0.2），共用同一 τ。
    _curriculum_allow_demote = config.get("CURRICULUM_ALLOW_DEMOTE", False)
    _curriculum_allow_demote = (_curriculum_allow_demote.lower() == "true") \
        if isinstance(_curriculum_allow_demote, str) else bool(_curriculum_allow_demote)
    if generator_injection and _curriculum_arm != "none":
        print(f"[curriculum] arm={_curriculum_arm} beta={_curriculum_beta} "
              f"thr={_curr_thr_easy}/{_curr_thr_mid}/{_curr_thr_hard} (easy/mid/hard) "
              f"adaptive={_curriculum_adaptive} tau={_curriculum_p_threshold} "
              f"allow_demote={_curriculum_allow_demote}", flush=True)
    # ── 方向1/3 (STAGE4_phase2): auction 信号模式 + difficulty 固定权重 ──
    # AUCTION_SIGNAL_MODE: 'anchored'(旧,[difficulty,anchored-PVL,cenie]) / 'euc'([euc,difficulty,cenie])
    _auction_signal_mode = str(config.get("AUCTION_SIGNAL_MODE", "anchored"))
    # DIFFICULTY_WEIGHT_FACTOR: >1 时放大 difficulty 在 auction 的竞价权重(方向1),
    #   乘在 auction 自动分配的 difficulty 维权重上再归一化——保留 euc/cenie 相对关系,
    #   只增强 difficulty(不像绝对覆盖那样拍平另两维)。=1 即不放大(auction 自动)。
    #   维度顺序: signal_mode='euc' → [euc, difficulty, cenie](difficulty 是 index 1);
    #            'anchored' → [difficulty, anchored, cenie](difficulty 是 index 0)。
    _dwf = float(config.get("DIFFICULTY_WEIGHT_FACTOR", 1.0))
    # ALP_WEIGHT_FACTOR: >1 抬 euc_alp 模式的 ALP 维(index 1)权重（乘 auto w 后归一，作用在 w 层
    #   不被 z-score 抹平）。f=2.3 让 ALP 追平 euc/cenie(三方均势)，2.5≈0.35(略超)。专给 ALP，
    #   与 DIFFICULTY_WEIGHT_FACTOR 互斥（euc_alp 模式 difficulty 已被 ALP 顿替，不用后者）。
    _awf = float(config.get("ALP_WEIGHT_FACTOR", 1.0))
    import jax.numpy as _jnp_fw
    if _auction_signal_mode == "euc_alp" and _awf != 1.0:
        # [euc, ALP, cenie]：抬 ALP(index 1)。
        _auction_weight_factors = _jnp_fw.array([1.0, _awf, 1.0], dtype=_jnp_fw.float32)
    elif _dwf != 1.0:
        if _auction_signal_mode == "euc":
            _auction_weight_factors = _jnp_fw.array([1.0, _dwf, 1.0], dtype=_jnp_fw.float32)
        else:
            _auction_weight_factors = _jnp_fw.array([_dwf, 1.0, 1.0], dtype=_jnp_fw.float32)
    else:
        _auction_weight_factors = None
    if generator_injection:
        gen_map_size = int(config.get("GEN_MAP_SIZE", env._map_obj.width))
        gen_max_board_scans = float(config.get("GEN_MAX_BOARD_SCANS", 3.0))
        gen_env, _gen_env_params = make_pcgrl_env(
            map_size=gen_map_size, max_board_scans=gen_max_board_scans)
        gen_network, _gen_meta = make_generator_network(gen_env)
        rng, _gen_rng = jax.random.split(rng)
        gen_state, gen_optimizer = make_generator_state(
            _gen_rng, gen_env, gen_network, gen_estimator_ids,
            gen_lr=float(config.get("GEN_LR", 2.5e-4)),
            max_grad_norm=float(config.get("GEN_MAX_GRAD_NORM", 0.5)))
        # 放起终点用的 jaxnav env：与训练 env 同参数，但 valid_path_check=True（§R5：
        # goal 只从 start 连通块采，根除不可解图——difficulty 信号对 p=0 给最难端高分，
        # 否则 generator 会 reward-hack 造不可解关卡。研究确认 JAX flood-fill 连通检验廉价）。
        # rollout 仍在原 env（valid_path_check=False，与训练分布一致）。
        _ep = OmegaConf.to_container(config["env"]["env_params"]) \
            if not isinstance(config["env"]["env_params"], dict) else dict(config["env"]["env_params"])
        _ep = {**_ep, "map_params": {**dict(_ep["map_params"]), "valid_path_check": True}}
        gen_place_env = JaxNav(num_agents=config["env"]["num_agents"], **_ep)
        # CENIE generator 依赖 gmm_params（每 eval epoch host 重拟）。若 estimator_ids 含
        # cenie 但 gmm_params 仍 None（AUCTION_USE_CENIE 关），补建冷启 GMM（valid=False，
        # 前几 epoch CENIE 返 0），并强制开启每 epoch GMM 重拟（否则 CENIE generator 永远拿 0）。
        if "cenie" in gen_estimator_ids and gmm_params is None:
            gmm_params = init_gmm_params(cenie_max_k, t_config["HIDDEN_SIZE"])
            auction_use_cenie = True   # 触发下方 eval-loop 每 epoch fit_visitation_gmm 重拟
            print("[generator] estimator_ids 含 cenie 但 AUCTION_USE_CENIE 原为关："
                  "已自动冷启 GMM + 开启每 epoch 重拟（否则 CENIE generator 恒 0）", flush=True)
        print(f"[generator] GENERATOR_INJECTION on: N={len(gen_estimator_ids)} "
              f"estimators={gen_estimator_ids} map_size={gen_map_size} "
              f"levels/gen(PPO)={gen_num_levels} "
              f"pool/gen={gen_pool_per_gen if gen_pool_per_gen > 0 else gen_num_levels}(eff) "
              f"rollout_chunk={gen_rollout_chunk} outer_steps={gen_outer_steps} "
              f"place_env.valid_path_check={getattr(gen_place_env._map_obj, 'valid_path_check', '?')}",
              flush=True)

    _total_eval_steps = int(t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])
    # ── [自适应课程状态机 2026-06-26] host 层 Python 变量跨 eval 维护，不穿 jit ──
    #   _curr_stage ∈ {0:easy,1:mid,2:hard} 初始 0；_high_p_streak = frac(p>0.8)>τ 的连续计数。
    #   每 eval 后更新；连续 2 次超 τ 升一档(单调不退，hard 封顶)；nan/无 p 统计时不升(streak 不+)。
    _curr_stage = 0
    _high_p_streak = 0
    # [双向滑落退档] _max_stage_reached = 曾达到的最高档（地板=max-1）；_low_p_streak = frac(p<0.2)>τ 连续计数。
    _max_stage_reached = 0
    _low_p_streak = 0
    for eval_step in range(_total_eval_steps):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        # [拆 jit] RNG 一致性：原 train_and_eval_step 内 split(eval_rng,3) 的第 1 份是
        #   learnability_rng。这里 host 层 split 出同一份喂 generator 段；train_and_eval_step
        #   仍收同一个 eval_rng,内部 split 第 2/3 份(singleton/sampled)与原版完全一致。
        learnability_rng, _, _ = jax.random.split(eval_rng, 3)
        # [结构课程] host 侧按训练进度算当前难度上限 thr(t)，作标量穿 gen_phase_jit。
        #   progress = eval_step/(总 eval 步-1) ∈[0,1]；分三档(easy/mid/hard)。arm=none 时传 0.0
        #   占位(get_generator_set 内 arm=none 不读 thr，塑形跳过)。改 thr 不触发重编译(标量参数)。
        if _curriculum_arm != "none":
            if _curriculum_adaptive:
                # 自适应：thr 由状态机档位 _curr_stage 决定（升档逻辑在本 eval 末尾按 frac(p>0.8) 更新）。
                _curr_thr = _curr_thr_by_stage[_curr_stage]
            else:
                # 固定时间表（旧行为，零回归）：按训练进度三等分。
                _progress = eval_step / max(_total_eval_steps - 1, 1)
                if _progress < (1.0 / 3.0):
                    _curr_thr = _curr_thr_easy
                elif _progress < (2.0 / 3.0):
                    _curr_thr = _curr_thr_mid
                else:
                    _curr_thr = _curr_thr_hard
        else:
            _curr_thr = 0.0
        # 阶段 G：generator 注入段 / 随机海选段（独立 jit 编译单元）。
        if generator_injection:
            learnabilty_scores, instances, gen_state, gen_metrics = gen_phase_jit(
                runner_state[0].params, gen_state, learnability_rng, gmm_params,
                jnp.float32(_curr_thr))
        else:
            learnabilty_scores, instances, _p_pool = learnability_phase_jit(
                runner_state[0].params, learnability_rng, gmm_params)
            # [护栏仪表盘 §injp] baseline 海选 pool 也算 gen_p_*（与 generator 路径同口径），
            #   答"baseline 海选关真实 p 分阶段如何"。复用下方 gen_metrics 白名单转发到 wandb。
            _pf = jnp.where(jnp.isfinite(_p_pool), _p_pool, jnp.nan)
            gen_metrics = {
                "gen_p_mean": jnp.nanmean(_pf),
                "gen_p_median": jnp.nanmedian(_pf),
                "gen_p_learnable_frac": jnp.mean((_pf >= 0.2) & (_pf <= 0.8)),
                "gen_p_extreme_frac": jnp.mean((_pf < 0.05) | (_pf > 0.95)),
                # [自适应课程] 高 p 占比 = p>0.8 占比（incomplete 排除）。baseline 路径也算（同口径），
                #   保证 adaptive 状态机在 generator/baseline 两路径取键一致。
                "gen_p_high_frac": (jnp.where(jnp.isfinite(_p_pool),
                                              (_p_pool > 0.8).astype(jnp.float32), 0.0).sum()
                                    / jnp.maximum(jnp.isfinite(_p_pool).sum(), 1)),
                # [双向滑落退档] 低 p 占比 = p<0.2 占比（incomplete 排除，同口径）。baseline 路径也算。
                "gen_p_low_frac": (jnp.where(jnp.isfinite(_p_pool),
                                             (_p_pool < 0.2).astype(jnp.float32), 0.0).sum()
                                   / jnp.maximum(jnp.isfinite(_p_pool).sum(), 1)),
                "gen_p_perlevel": _p_pool,
            }
        # 阶段 S+EVAL：student 50步 scan + 评估（独立 jit；instances/scores 作参数传入）。
        runner_state, instances_top, metrics = train_and_eval_step(
            runner_state, eval_rng, instances, learnabilty_scores)
        # 诊断: 区分编译 vs 运行时。第 0 个 epoch 含编译，第 1 个起纯运行（已编译）。
        jax.block_until_ready(runner_state)
        curr_time = time.time()
        if eval_step <= 1:
            print(f"[DIAG] eval_step={eval_step} gen+train+eval 耗时 {curr_time-start_time:.1f}s "
                  f"(stage={config.get('AUCTION_STAGE','-')}, "
                  f"auction={config.get('AUCTION_SCORING')}, cenie={auction_use_cenie}) "
                  f"{'[含编译]' if eval_step==0 else '[纯运行]'}", flush=True)
        # [拆 jit] gen_metrics wandb 键 host 层处理（原在 jit 内）。
        if gen_metrics is not None:
            # baseline 海选路径只填 gen_p_*（无 gen_mean_score 等 generator 专属键）→ 用 .get 防 KeyError。
            for _gk in ("gen_mean_score", "gen_injected", "gen_n_incomplete"):
                if _gk in gen_metrics:
                    metrics[_gk] = gen_metrics[_gk]
            # ── [自适应课程状态机更新 2026-06-26] 由本 eval 注入关 p 分布驱动升/退档 ──
            #   仅 adaptive 模式且课程开启时生效。冷启/全 incomplete → frac 缺失或 nan → 两 streak 都不动。
            #   升档：连续 2 次 frac(p>0.8)>τ 升一档(hard=2 封顶)，升档后 high_streak 归 0、更新 max_stage。
            #   退档(仅 ALLOW_DEMOTE=true)：连续 2 次 frac(p<0.2)>τ 且未触地板则降一档，退后 low_streak 归 0。
            #     地板 floor=max(_max_stage_reached-1,0)：升过 hard 则永不退回 easy(地板=mid)。
            #   升/退 streak 独立计数；ALLOW_DEMOTE=false 时退档逻辑整段不执行(单调对照组零回归)。
            if _curriculum_arm != "none" and _curriculum_adaptive:
                _hpf = gen_metrics.get("gen_p_high_frac", None)
                _hpf = float(_hpf) if _hpf is not None else float("nan")
                _lpf = gen_metrics.get("gen_p_low_frac", None)
                _lpf = float(_lpf) if _lpf is not None else float("nan")
                metrics["curriculum_high_p_frac"] = _hpf
                metrics["curriculum_low_p_frac"] = _lpf
                # —— 升档 —— (nan!=nan → nan 时不计)
                if _hpf == _hpf and _hpf > _curriculum_p_threshold:
                    _high_p_streak += 1
                else:
                    _high_p_streak = 0
                if _high_p_streak >= 2 and _curr_stage < (len(_curr_thr_by_stage) - 1):
                    _curr_stage += 1
                    _high_p_streak = 0
                    _max_stage_reached = max(_max_stage_reached, _curr_stage)
                    print(f"[curriculum] eval_step={eval_step} 升档 → stage={_curr_stage} "
                          f"thr={_curr_thr_by_stage[_curr_stage]} (frac(p>0.8)={_hpf:.3f}>"
                          f"{_curriculum_p_threshold})", flush=True)
                # —— 退档（双向滑落，有地板）——
                if _curriculum_allow_demote:
                    if _lpf == _lpf and _lpf > _curriculum_p_threshold:
                        _low_p_streak += 1
                    else:
                        _low_p_streak = 0
                    _floor = max(_max_stage_reached - 1, 0)
                    if _low_p_streak >= 2 and _curr_stage > _floor:
                        _curr_stage -= 1
                        _low_p_streak = 0
                        print(f"[curriculum] eval_step={eval_step} 退档 → stage={_curr_stage} "
                              f"thr={_curr_thr_by_stage[_curr_stage]} (frac(p<0.2)={_lpf:.3f}>"
                              f"{_curriculum_p_threshold}, floor={_floor})", flush=True)
                metrics["curriculum_stage"] = _curr_stage
                metrics["curriculum_max_stage"] = _max_stage_reached
            # [机制判据 STAGE4_phase2 §方向3] 注入关 fill/euc 分布 → wandb。
            # [护栏仪表盘 §injp 2026-06-25] gen_p_* = 注入关 per-level student p 分布（可学区/两极占比）；
            #   验护栏是否把 p 从 injp 的"100%两极"拉回可学区。gen_p_perlevel 是数组→存直方图。
            for _mk in ("gen_inj_fill_median", "gen_inj_fill_mean", "gen_inj_fill_hi_frac",
                        "gen_inj_euc_median", "gen_inj_euc_mean",
                        # [结构课程] 注入关测地长度(课程主轴注入侧验证) + 当前难度档 + 超档比例。
                        "gen_inj_geo_median", "gen_inj_geo_mean",
                        "curriculum_threshold", "gen_inj_frac_over_threshold",
                        "gen_struct_value_mean", "gen_frac_over_threshold", "gen_frac_unreachable",
                        "gen_p_mean", "gen_p_median", "gen_p_learnable_frac", "gen_p_extreme_frac",
                        "gen_p_high_frac", "gen_p_low_frac"):
                if _mk in gen_metrics:
                    metrics[_mk] = gen_metrics[_mk]
            if "gen_p_perlevel" in gen_metrics:
                import numpy as _np
                _pl = _np.asarray(gen_metrics["gen_p_perlevel"])
                _pl = _pl[_np.isfinite(_pl)]
                if _pl.size > 0:
                    metrics["gen_p_hist"] = wandb.Histogram(_pl)
            # [结构课程] 注入关测地长度直方图（看课程是否把结构分布推向 Nav2/3 走廊拓扑 16~22）。
            if "gen_inj_geo_perlevel" in gen_metrics:
                import numpy as _np
                _gl = _np.asarray(gen_metrics["gen_inj_geo_perlevel"])
                _gl = _gl[_np.isfinite(_gl)]
                if _gl.size > 0:
                    metrics["gen_inj_geo_hist"] = wandb.Histogram(_gl)
            if "auction_weights" in gen_metrics:
                # auction 维名按 signal_mode 取（≠ gen_estimator_ids，后者是 generator reward 信号）：
                #   euc_alp → [euc, alp, cenie]；euc → [euc, difficulty, cenie]；
                #   anchored/pvl → [difficulty, <pvl/anchored>, cenie]。否则退回 gen_estimator_ids。
                if _auction_signal_mode == "euc_alp":
                    _auc_names = ["euc", "alp", "cenie"]
                elif _auction_signal_mode == "euc":
                    _auc_names = ["euc", "difficulty", "cenie"]
                elif _auction_signal_mode == "pvl":
                    _auc_names = ["difficulty", "pvl", "cenie"]
                elif _auction_signal_mode == "anchored":
                    _auc_names = ["difficulty", "anchored", "cenie"]
                elif _auction_signal_mode == "anchored_cenie":
                    _auc_names = ["anchored", "cenie"]      # [结构课程 Run B] 去 difficulty 的 2 维
                elif _auction_signal_mode == "anchored_cenie_gate":
                    _auc_names = ["anchored", "cenie"]      # [单边可学性 gate] auction 仍这 2 维（gate 走 host 层）
                else:
                    _auc_names = list(gen_estimator_ids)
                for _i in range(len(_auc_names)):
                    metrics[f"auction_weight_{_auc_names[_i]}"] = gen_metrics["auction_weights"][_i]
                    metrics[f"auction_bid_{_auc_names[_i]}"] = gen_metrics["auction_bids"][_i]
        log_buffer(*instances_top, metrics["update_count"])
        # -- [diag DIAG_DUMP_MAZE] dump injected-level math repr; storage-only, no mechanism change.
        #   windows by real update_count (multiples of EVAL_FREQ=50, last=2250): early/mid/late x5 frames.
        try:
            _uc = int(metrics["update_count"])
            _windows = {50, 100, 150, 200, 250,
                        1100, 1150, 1200, 1250, 1300,
                        2050, 2100, 2150, 2200, 2250}
            if _uc in _windows:
                import numpy as _np, os as _os
                _lrn, _st = instances_top
                _dd = _os.path.join("_maze_dump", str(config.get("GROUP_NAME", "run")),
                                   "seed%s" % config.get("SEED", 0))
                _os.makedirs(_dd, exist_ok=True)
                _np.savez(
                    _os.path.join(_dd, "inj_%05d.npz" % _uc),
                    learnability=_np.asarray(_lrn),
                    map_data=_np.asarray(_st.map_data),
                    agent_pos=_np.asarray(_st.pos),
                    goal_pos=_np.asarray(_st.goal),
                    theta=_np.asarray(_st.theta),
                    update_count=_uc,
                )
                print("[DIAG_DUMP_MAZE] saved inj_%05d.npz (n=%d) -> %s"
                      % (_uc, _np.asarray(_st.map_data).shape[0], _dd))
        except Exception as _e:
            print("[DIAG_DUMP_MAZE] dump skipped: %s" % _e)
        metrics['time_delta'] = curr_time - start_time
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]) / metrics['time_delta']
        wandb.log(metrics, step=metrics["update_count"])

        # ── V1 正交性探针 + CENIE GMM 重拟（jit 外，host 侧，复用同一份 hidden）──
        # 探针测四信号相关 + CENIE 用探针收集的 hidden 重拟 GMM（写回 gmm_params 供下一 epoch）。
        probe_data = None
        # probe_data 收集条件解耦 SAVE_PATH（STAGE4 阶段 1 暴露：注入档未设 SAVE_PATH 致 probe_out_dir
        # =None → 整条 probe 不跑、拿不到 probe/p_std 等饱和诊断）。现只要 probe_enabled 或 cenie 需要
        # hidden 就收集；落 wandb 指标无条件，落 trace 文件才看 probe_out_dir。
        if probe_enabled or auction_use_cenie:
            try:
                rng, probe_rng = jax.random.split(rng)
                probe_dev = get_probe_signals(probe_rng, runner_state[0].params)
                probe_data = jax.device_get(probe_dev)  # 设备数组 → host numpy
                probe_data["n_levels"] = int(probe_data["n_levels"])
            except Exception as e:
                print(f"[probe/cenie][step {int(metrics['update_count'])}] hidden 收集异常：{e}")

        # probe/p_std 等 wandb 曲线指标无条件落（注入档/海选档都拿得到，用于判饱和）；
        # out_dir 传 probe_out_dir（None 时 log_orthogonality_step 只算指标、不写 trace 文件）。
        if probe_enabled and probe_data is not None:
            try:
                probe_res = log_orthogonality_step(
                    probe_data, int(metrics["update_count"]), probe_out_dir, threshold=0.5)
                wandb.log(probe_res["wandb_metrics"], step=int(metrics["update_count"]))
            except Exception as e:
                print(f"[probe][step {int(metrics['update_count'])}] 跳过（探针异常）：{e}")

        # CENIE GMM 重拟（每 eval epoch）：用探针 hidden（已 host 侧 subsample 过）拟 GMM 写回。
        if auction_use_cenie and probe_data is not None:
            try:
                feats = np.asarray(probe_data["hidden_feats"])                   # (Nh, H)
                if feats.shape[0] > cenie_subsample:
                    idx = np.linspace(0, feats.shape[0] - 1, cenie_subsample).astype(int)
                    feats = feats[idx]
                gmm_params = fit_visitation_gmm(
                    feats, max_components=cenie_max_k, min_components=cenie_min_k,
                    reg_covar=cenie_reg_covar)
                print(f"[cenie][step {int(metrics['update_count'])}] GMM 重拟 OK (valid={bool(gmm_params.valid)})")
            except Exception as e:
                print(f"[cenie][step {int(metrics['update_count'])}] GMM 重拟异常，保留上轮：{e}")

        # ── checkpoint：带 step 后缀存多份（不覆盖），便于事后对比训练阶段 ──
        if (eval_step % checkpoint_steps == 0) & (eval_step > 0):
            if config["SAVE_PATH"] is not None:
                params = runner_state[0].params

                save_dir = os.path.join(config["SAVE_PATH"], run.name)
                os.makedirs(save_dir, exist_ok=True)
                # 末期覆盖式（向后兼容现有 eval/部署脚本）
                save_params(params, f'{save_dir}/model.safetensors')
                # 带 step 后缀的多 checkpoint（新增，供探针/事后分析挑非饱和阶段）
                step_tag = int(metrics["update_count"])
                save_params(params, f'{save_dir}/model_step{step_tag}.safetensors')
                print(f'Parameters of saved in {save_dir}/model.safetensors (+ model_step{step_tag}.safetensors)')

                # upload this to wandb as an artifact
                artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
                artifact.add_file(f'{save_dir}/model.safetensors')
                artifact.save()

    if config["SAVE_PATH"] is not None:
        params = runner_state[0].params

        save_dir = os.path.join(config["SAVE_PATH"], run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/model.safetensors')
        print(f'Parameters of saved in {save_dir}/model.safetensors')
        
        # upload this to wandb as an artifact
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        artifact.add_file(f'{save_dir}/model.safetensors')
        artifact.save()

    # ── V1 探针总裁决：训练结束读相关轨迹，给非饱和 epoch 上的多数 verdict ──
    if probe_enabled and probe_out_dir is not None:
        try:
            summary = summarize_trace(probe_out_dir)
            print("=" * 60)
            print(f"[probe] 正交性总裁决（基于非饱和 epoch）: {summary.get('verdict')}")
            print(f"[probe]   {summary}")
            print("=" * 60)
            with open(os.path.join(probe_out_dir, "verdict_summary.txt"), "w", encoding="utf-8") as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")
        except Exception as e:
            print(f"[probe] 总裁决跳过（异常）：{e}")


if __name__ == "__main__":
    main()
