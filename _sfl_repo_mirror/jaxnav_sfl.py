"""
Run SFL on JaxNav, both single and multi-agent variations.
"""

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
            return None, (learnability_by_env, env_instances,
                          difficulty_by_env, pvl_by_env, cenie_by_env)

        rngs = jax.random.split(rng, config["NUM_BATCHES"])
        _, (learnability, env_instances, difficulty, pvl, cenie) = jax.lax.scan(
            _batch_step, None, rngs, config["NUM_BATCHES"])

        flat_env_instances = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), env_instances)
        learnability = learnability.flatten()
        difficulty = difficulty.flatten()                                       # (M,)
        pvl = pvl.flatten()                                                      # (M,)
        cenie = cenie.flatten()                                                  # (M,)

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
        return score.at[top_1000].get(), top_1000_instances


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
                    "env-metrics/": metric["env-metrics"],
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
        metric["env-metrics"] = jax.tree_map(lambda x: x.mean(), jax.vmap(env.get_env_metrics)(start_state))
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
    
    @partial(jax.jit, static_argnums=())
    def train_and_eval_step(runner_state, eval_rng, gmm_params=None, gen_state=None):
        # gmm_params/gen_state 显式穿过 jit 边界（pytree 参数，与 student params 同理）。
        #   gmm_params: CENIE 的 GMM（None=CENIE 关）。
        #   gen_state : N 个 generator 的 params/opt_state（None=随机海选）。env/network/
        #               optimizer 是静态对象，走闭包捕获（gen_env/gen_network/gen_optimizer），
        #               不作参数；generator_injection 是 Python 静态 bool，False 分支不 trace。
        learnability_rng, eval_singleton_rng, eval_sampled_rng = jax.random.split(eval_rng, 3)
        # TRAIN —— 阶段 G：generator 注入 or 随机海选，产同型 (scores, instances)。
        new_gen_state = gen_state
        if generator_injection:
            # 交替训练阶段 G：冻 student（runner_state[0].params）→ 更新 generator →
            # 产 top-K instances。NUM_TO_SAVE 对齐 buffer 规模；ROLLOUT_STEPS = student
            # 测 level 难度的 rollout 步数（与 get_learnability_set 海选时一致）。
            learnabilty_scores, instances, new_gen_state, gen_metrics = get_generator_set(
                learnability_rng, runner_state[0].params, gen_state, gen_optimizer,
                gen_estimator_ids, gen_env, gen_network, env, network,
                t_config["HIDDEN_SIZE"], config["ROLLOUT_STEPS"],
                num_levels_per_gen=gen_num_levels, num_to_save=gen_num_to_save,
                place_env=gen_place_env, gmm_params=gmm_params,
                auction_lambda=gen_auction_lambda,
                gen_outer_steps=gen_outer_steps, ppo_epochs=gen_ppo_epochs,
                gamma=t_config["GAMMA"], gae_lambda=t_config["GAE_LAMBDA"])
        else:
            learnabilty_scores, instances = get_learnability_set(
                learnability_rng, runner_state[0].params, gmm_params)
            gen_metrics = None
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
        # STAGE4 §1.3：auction_weights/bids 轨迹（哪个 teacher 何时出价高、λ 怎么调尖锐度）。
        # gen_metrics 是 jit 内 traced 数组，并进 test_metrics 一并穿出 jit 边界 → wandb.log。
        # 基线（GENERATOR_INJECTION=false）gen_metrics=None，不落这些键，零回归。
        if gen_metrics is not None:
            test_metrics["gen_mean_score"] = gen_metrics["gen_mean_score"]
            test_metrics["gen_injected"] = gen_metrics["gen_injected"]
            test_metrics["gen_n_incomplete"] = gen_metrics["gen_n_incomplete"]
            if "auction_weights" in gen_metrics:
                # (N,) 数组拆成标量键，wandb 才能各画一条曲线（N=3：difficulty/pvl/cenie）。
                for _i in range(len(gen_estimator_ids)):
                    test_metrics[f"auction_weight_{gen_estimator_ids[_i]}"] = gen_metrics["auction_weights"][_i]
                    test_metrics[f"auction_bid_{gen_estimator_ids[_i]}"] = gen_metrics["auction_bids"][_i]
        test_metrics["singleton-test-metrics"] = eval_singleton_runner.run(eval_singleton_rng, runner_state[0].params)
        test_metrics["sampled-test-metrics"] = eval_sampled_runner.run(eval_sampled_rng, runner_state[0].params)

        runner_state, _ = runner_state_instances
        test_metrics["update_count"] = runner_state[-2]

        top_instances = jax.tree_map(lambda x: x.at[-20:].get(), instances)
        _, top_states = jax.vmap(env.set_env_instance)(top_instances)

        return runner_state, (learnabilty_scores.at[-20:].get(), top_states), test_metrics, new_gen_state

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
    # auction 出价漏斗温度（idea 核心）：N estimator 对整 pool 出价混合决定注入哪些课程。
    # None/"none" → fallback 每 gen 自评（消融）；"inf" → single-winner；有限值 → fractional。
    _gal = config.get("GEN_AUCTION_LAMBDA", 1.0)
    if _gal is None or (isinstance(_gal, str) and _gal.lower() == "none"):
        gen_auction_lambda = None
    elif isinstance(_gal, str) and _gal.lower() == "inf":
        gen_auction_lambda = float("inf")
    else:
        gen_auction_lambda = float(_gal)
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
              f"levels/gen={gen_num_levels} outer_steps={gen_outer_steps} "
              f"place_env.valid_path_check={getattr(gen_place_env._map_obj, 'valid_path_check', '?')}",
              flush=True)

    for eval_step in range(int(t_config["NUM_UPDATES"] // t_config["EVAL_FREQ"])):
        start_time = time.time()
        rng, eval_rng = jax.random.split(rng)
        # gen_state 宿主侧穿入/穿出（与 gmm_params 同模式）：阶段 G 更新后写回，供下轮。
        runner_state, instances, metrics, gen_state = train_and_eval_step(
            runner_state, eval_rng, gmm_params, gen_state)
        # 诊断: 区分编译 vs 运行时。第 0 个 epoch 含编译，第 1 个起纯运行（已编译）。
        jax.block_until_ready(runner_state)
        curr_time = time.time()
        if eval_step <= 1:
            print(f"[DIAG] eval_step={eval_step} train_and_eval_step 耗时 {curr_time-start_time:.1f}s "
                  f"(stage={config.get('AUCTION_STAGE','-')}, "
                  f"auction={config.get('AUCTION_SCORING')}, cenie={auction_use_cenie}) "
                  f"{'[含编译]' if eval_step==0 else '[纯运行]'}", flush=True)
        log_buffer(*instances, metrics["update_count"])
        metrics['time_delta'] = curr_time - start_time
        metrics["steps_per_section"] = (t_config["EVAL_FREQ"] * t_config["NUM_STEPS"] * t_config["NUM_ENVS"]) / metrics['time_delta']
        wandb.log(metrics, step=metrics["update_count"])

        # ── V1 正交性探针 + CENIE GMM 重拟（jit 外，host 侧，复用同一份 hidden）──
        # 探针测四信号相关 + CENIE 用探针收集的 hidden 重拟 GMM（写回 gmm_params 供下一 epoch）。
        probe_data = None
        if (probe_enabled and probe_out_dir is not None) or auction_use_cenie:
            try:
                rng, probe_rng = jax.random.split(rng)
                probe_dev = get_probe_signals(probe_rng, runner_state[0].params)
                probe_data = jax.device_get(probe_dev)  # 设备数组 → host numpy
                probe_data["n_levels"] = int(probe_data["n_levels"])
            except Exception as e:
                print(f"[probe/cenie][step {int(metrics['update_count'])}] hidden 收集异常：{e}")

        if probe_enabled and probe_out_dir is not None and probe_data is not None:
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
