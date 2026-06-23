"""PCGRL generator 注入层 —— 第 1 块：map 桥接 + 起终点（复用 jaxnav 采样器）+ 输出合法 EnvInstance。

设计见 mechanism_UED/_sfl_repo_mirror/GENERATOR_DESIGN.md（§2 map 桥接、§3 reward、§5 交替训练）。

本文件分块落地（§8），当前只含**与 PPO 无关、最易错的几何桥接**，可独立单测：
  1. pcgrl_map_to_jaxnav(env_map): PCGRL {BORDER=0,EMPTY=1,WALL=2} → jaxnav map_data {0=自由,1=墙}。
  2. place_start_goal_on_map(map_obj, jaxnav_map, rng): 复用 jaxnav grid_map 的 inflate+连通区采样在给定 map
     上放合法可达的 start/goal（不自己手写碰撞/可达，调 jaxnav 已校验逻辑）。
  3. map_to_env_instance(...): 组装成 jaxnav EnvInstance（单 level），可直接喂 env.set_env_instance 跑 student。

PPO / generator policy / estimator terminal reward / N generator 在跑通本块后的下一版加。

载体 = SFL repo sampling-for-learnability（sfl env, 真 jaxnav）。本机无 jaxnav，只能 Oscar 单测。
"""
import jax
import jax.numpy as jnp
from typing import NamedTuple
import optax


# PCGRL binary tile 编码（来自 pcgrl-jax envs/probs/binary.py BinaryTiles）。
PCGRL_BORDER = 0
PCGRL_EMPTY = 1
PCGRL_WALL = 2


def pcgrl_map_to_jaxnav(env_map):
    """PCGRL env_map {BORDER=0,EMPTY=1,WALL=2} → jaxnav map_data {0=自由, 1=墙}。

    BORDER 与 WALL 都映射成 1（jaxnav 里边界即墙）；EMPTY→0。纯逐元素，jit 安全。

    Args:
      env_map: (H, W) int，PCGRL tile 值。
    Returns:
      map_data: (H, W) int32，jaxnav 占用栅格。
    """
    return jnp.where(env_map == PCGRL_EMPTY, 0, 1).astype(jnp.int32)


def place_start_goal_on_map(map_obj, jaxnav_map, rng):
    """在给定 jaxnav_map（占用栅格）上放合法 start/goal，复刻 jaxnav Grid-Rand-Poly 的采样逻辑。

    对齐 grid_map.GridMapCircleAgents.grid_sample_test_case 的「放起终点」半段（去掉 sample_map，
    map 用 generator 的）：在 **inside_grid（去边界内部 (H-2)×(W-2)）** 上，非占用格随机选 start，
    连通/自由格里选 goal，坐标转换 = (内部坐标 + 1.5)。num_agents=1，故 scan 退化为一次。

    可达性：与 jaxnav 同款受 valid_path_check 控制。配置 valid_path_check=False 时连通区=全体自由内部格
    （不做真连通检查，与 jaxnav 原生行为一致）；True 时用 component_mask_with_pos 真连通。

    Args:
      map_obj: jaxnav env._map_obj（GridMapPolygonAgents）。提供 width/height/valid_path_check。
      jaxnav_map: (H, W) int 占用栅格（generator 造，0=自由 1=墙）。
      rng: PRNGKey。
    Returns:
      start_xy, goal_xy: (2,) float [x, y]（含边界坐标系，格中心）。
      valid: bool 标量。
    """
    import jaxmarl.environments.jaxnav.jaxnav_graph_utils as _graph_utils

    iwidth = map_obj.width - 2
    map_data = jaxnav_map
    inside_grid = map_data[1:-1, 1:-1]               # (H-2, W-2) 内部，1=占用
    valid_path_check = bool(getattr(map_obj, "valid_path_check", False))

    # [卡死修复·A 止血] 加迭代上界，根除无界 while_loop 死循环。原 _cond 只判 `while not
    #   valid`，无上界：当 generator 造出病态全墙图（连通区只剩孤立单格，valid_path_check=True
    #   时 component_mask 算出无合法 goal）→ valid 永远 False → device 上无限重采、GPU 满频自旋、
    #   日志 mtime 死（与 GPU=0% 的 XLA 死锁画像不同）。正常图 1-2 次必中，上界不触发、行为零变化；
    #   病态图耗尽 MAX_PLACE_TRIES 后带 valid=False 退出，由下游（B 注入端 -inf / C 奖励端罚分）处理。
    MAX_PLACE_TRIES = 64

    def _cond(val):
        _valid, _it = val[0], val[1]
        return jnp.bitwise_not(_valid) & (_it < MAX_PLACE_TRIES)

    def _body(val):
        _valid, _it, rng, _s, _g = val
        rng, k1, k2 = jax.random.split(rng, 3)
        flat_occ = inside_grid.flatten()
        # start：内部非占用格里随机选。
        start_idx = jax.random.choice(k1, flat_occ.shape[0], (1,), p=(1 - flat_occ).astype(jnp.float32))[0]
        start = jnp.array([start_idx % iwidth, start_idx // iwidth])  # [x, y]（内部坐标）
        actual_idx = (start + 1).astype(jnp.int32)                    # 含边界坐标
        if valid_path_check:
            connected = _graph_utils.component_mask_with_pos(map_data, actual_idx)[1:-1, 1:-1]
        else:
            connected = 1 - inside_grid                               # 全体自由内部格
        masked_start = connected.at[start[1], start[0]].set(0)        # 排除 start 自身
        goal_poss = masked_start.flatten()
        valid = jnp.any(goal_poss > 0)
        goal_idx = jax.random.choice(k2, flat_occ.shape[0], (1,),
                                     p=goal_poss.astype(jnp.float32))[0]
        goal = jnp.array([goal_idx % iwidth, goal_idx // iwidth])
        return (valid, _it + 1, rng, start.astype(jnp.float32), goal.astype(jnp.float32))

    init = (False, jnp.int32(0), rng, jnp.zeros((2,)), jnp.zeros((2,)))
    valid, _, _, start, goal = jax.lax.while_loop(_cond, _body, init)
    # 坐标转换：内部坐标 +1.5（+1 回含边界，+0.5 取格中心），与 grid_sample_test_case 的 pos+1.5 一致。
    return start + 1.5, goal + 1.5, valid


def map_to_env_instance(env, env_map, rng, rew_lambda=None):
    """PCGRL env_map → 合法 jaxnav EnvInstance（单 level, num_agents=1）。

    Args:
      env: jaxnav JaxNav 实例（提供 _map_obj、num_agents、sample_lambda）。
      env_map: (H, W) PCGRL tile 值。
      rng: PRNGKey。
      rew_lambda: 可选，None 则用 env.sample_lambda 采样。
    Returns:
      EnvInstance（agent_pos/agent_theta/goal_pos/map_data/rew_lambda），valid 标量。
    """
    from jaxmarl.environments.jaxnav.jaxnav_env import EnvInstance

    rng, r_sg, r_th, r_lam = jax.random.split(rng, 4)
    jaxnav_map = pcgrl_map_to_jaxnav(env_map)
    start_xy, goal_xy, valid = place_start_goal_on_map(env._map_obj, jaxnav_map, r_sg)

    theta = (jnp.pi / 2) * jax.random.choice(r_th, jnp.arange(4))    # 4 朝向之一，与 barn_test_case 同
    n = env.num_agents                                              # =1
    agent_pos = start_xy[None, :]                                   # (1, 2)
    agent_theta = theta[None]                                       # (1,)
    goal_pos = goal_xy[None, :]                                     # (1, 2)

    if rew_lambda is None:
        rew_lambda = env.sample_lambda(r_lam)
    rew_lambda = jnp.asarray(rew_lambda).reshape(())

    inst = EnvInstance(
        agent_pos=agent_pos,
        agent_theta=agent_theta,
        goal_pos=goal_pos,
        map_data=jaxnav_map,
        rew_lambda=rew_lambda,
    )
    return inst, valid


# ============================================================================
# 第 2 块：generator policy（复用 pcgrl-jax ConvForward CNN）+ single rollout
# ----------------------------------------------------------------------------
# 设计确认（全部 Oscar sfl env 真跑验证，见对话 2026-06-21）：
#   env   : PCGRLEnv(map_shape=(11,11)) → max_steps=363, act_shape=(1,1),
#           action_dim=2(Discrete: EMPTY/WALL), rf_shape=(31,31)
#   reset : gen_dummy_queued_state(env) → reset_env(rng,params,qs)
#           → obs.map_obs=(31,31,4) float32, obs.flat_obs=(1,) float32
#   policy: ConvForward(action_dim=2, act_shape=(1,1), arf=vrf=31 看全图,
#           hidden_dims=(64,64)) 输入 (map_obs, flat_obs) 出 (logits(B,2), value(B,))
#   action 桥接（根因，非 bug）: env 期望单 env action 形状 (n_agents, *act_shape)
#           =(1,1,1)。policy logits reshape (B,n_agents,*act_shape,action_dim)
#           =(B,1,1,1,2) → Categorical.sample → (B,1,1,1) → per-env (1,1,1) 喂
#           step_env。喂标量/(1,)/(1,1) 都会被各层 [...,0]/[0] 剥成标量索引致
#           builds[scalar] rank0 撞 dynamic_update_slice（原版同理，故原版从不
#           直喂标量，而是经 ActorCriticPCGRL reshape 出 (1,1,1) 维 action）。
#
# 本块只验证「随机初始化 policy 能跑满 363 步逐 tile rollout → 出 1 张成品 map
# → 过第 1 块 map_to_env_instance → 喂 student 不崩」。PPO/estimator reward/
# N generator 是第 3 块。policy 参数随机初始化、固定不训。
# ============================================================================
import dataclasses


def make_pcgrl_env(map_size=11, max_board_scans=3.0):
    """造 binary-narrow PCGRL env（generator 的关卡设计 MDP）。

    Returns:
      env: PCGRLEnv 实例。
      env_params: 对应 PCGRLEnvParams（map_shape=(map_size,map_size)）。
    """
    from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams
    env_params = dataclasses.replace(
        PCGRLEnvParams(), map_shape=(map_size, map_size), max_board_scans=max_board_scans)
    env = PCGRLEnv(env_params)
    return env, env_params


def make_generator_network(env, hidden_dims=(64, 64), activation="relu"):
    """实例化 generator policy = pcgrl-jax ConvForward（CNN actor-critic）。

    arf_size=vrf_size=rf 全尺寸（看整张 obs 视野，11×11 图无需局部裁剪）。
    action_dim=可编辑 tile 数（binary=2: EMPTY/WALL）。act_shape=env.rep.act_shape=(1,1)。

    Returns:
      network: ConvForward（未初始化，调用方 network.init 拿 params）。
      meta: dict（action_dim/act_shape/rf 备查）。
    """
    from models_pcgrl import ConvForward
    rep = env.rep
    action_dim = int(rep.action_space.n) // (rep.act_shape[0] * rep.act_shape[1])
    rf = int(rep.rf_shape[0]) if hasattr(rep.rf_shape, "__len__") else int(rep.rf_shape)
    network = ConvForward(
        action_dim=action_dim,
        act_shape=tuple(int(s) for s in rep.act_shape),
        arf_size=rf, vrf_size=rf,
        hidden_dims=hidden_dims, activation=activation,
    )
    meta = {"action_dim": action_dim, "act_shape": tuple(int(s) for s in rep.act_shape), "rf": rf}
    return network, meta


def init_generator_params(network, env, rng):
    """随机初始化 generator policy 参数（第 2 块固定不训，第 3 块接 PPO 更新）。"""
    from envs.pcgrl_env import gen_dummy_queued_state
    rng, r_reset, r_init = jax.random.split(rng, 3)
    qs = gen_dummy_queued_state(env)
    obs, _state = env.reset_env(r_reset, _env_params_of(env), qs)
    # ConvForward 吃 batch 维：map_obs (B,31,31,4), flat_obs (B,1)。
    map_b = obs.map_obs[None, ...]
    flat_b = obs.flat_obs[None, ...]
    params = network.init(r_init, map_b, flat_b)
    return params


def _env_params_of(env):
    """从 env 取回它构造时的 PCGRLEnvParams（reset_env 需要）。"""
    from envs.pcgrl_env import PCGRLEnvParams
    # env 自身不存 params，但 map_shape/max_board_scans 可重建（与构造一致）。
    ms = tuple(int(s) for s in env.map_shape)
    mbs = float(env.rep.max_board_scans)
    return dataclasses.replace(PCGRLEnvParams(), map_shape=ms, max_board_scans=mbs)


def logits_to_action(logits, rng, n_agents, act_shape, action_dim):
    """policy logits (B, action_dim*prod(act_shape)) → 采样 action (B, n_agents, *act_shape)。

    复刻 ActorCriticPCGRL: reshape 出 (n_agents, *act_shape) 维度，使喂进 step_env 后
    经 [...,None]/[0]/[...,0] 剥维到 (act_shape) 时 builds[action] 出 (1,1) patch。
    """
    import distrax
    B = logits.shape[0]
    logits = logits.reshape((B, n_agents, *act_shape, action_dim))
    pi = distrax.Categorical(logits=logits)
    action = pi.sample(seed=rng)            # (B, n_agents, *act_shape)
    return action


def gen_rollout_single(env, network, params, rng):
    """随机初始化 policy 跑满 max_steps 逐 tile 编辑 → 出 1 张成品 env_map。

    单 env（无 batch）。每步：policy(obs)→logits→采样 action(1,1,1)→env.step_env→新 obs。
    用 jax.lax.scan 跑 env.rep.max_steps 步。

    Returns:
      final_env_map: (H, W) int32，成品 PCGRL map（{EMPTY=1, WALL=2}）。
    """
    from envs.pcgrl_env import gen_dummy_queued_state
    rep = env.rep
    n_agents = 1
    act_shape = tuple(int(s) for s in rep.act_shape)
    action_dim = int(rep.action_space.n) // (act_shape[0] * act_shape[1])
    env_params = _env_params_of(env)
    max_steps = int(rep.max_steps)

    rng, r_reset = jax.random.split(rng)
    qs = gen_dummy_queued_state(env)
    obs0, state0 = env.reset_env(r_reset, env_params, qs)

    def _step(carry, _):
        rng, obs, state = carry
        rng, r_act, r_step = jax.random.split(rng, 3)
        logits, _val = network.apply(params, obs.map_obs[None, ...], obs.flat_obs[None, ...])
        action = logits_to_action(logits, r_act, n_agents, act_shape, action_dim)  # (1,1,1,1)
        action = action[0]                       # 去 batch → (n_agents,*act_shape)=(1,1,1)
        obs2, state2, _r, _d, _i = env.step_env(r_step, state, action, env_params)
        return (rng, obs2, state2), None

    (rng, _obs, state_final), _ = jax.lax.scan(_step, (rng, obs0, state0), None, length=max_steps)
    return state_final.env_map


def gen_rollout_batch(env, network, params, rng, n_levels):
    """vmap gen_rollout_single → 一次造 n_levels 张成品 env_map。

    queued_state 是常量（全 False/zeros），N 张共用，故只对 rng 轴 vmap。
    每个 sub-rng 不同 → reset 初始 map + rollout 采样都不同 → N 张多样。

    Args:
      env, network, params: 同 gen_rollout_single。
      rng: PRNGKey。
      n_levels: int，造几张（静态）。
    Returns:
      env_maps: (n_levels, H, W) int32。
    """
    rngs = jax.random.split(rng, n_levels)
    return jax.vmap(lambda r: gen_rollout_single(env, network, params, r))(rngs)


def gen_batch_to_env_instances(env_maps, jaxnav_env, rng):
    """batch 成品 env_map → batch jaxnav EnvInstance（每张过第 1 块 map_to_env_instance）。

    Args:
      env_maps: (n_levels, H, W) PCGRL tile 值（gen_rollout_batch 产出）。
      jaxnav_env: jaxnav JaxNav 实例（注意：≠ PCGRL env；提供 _map_obj/set_env_instance）。
      rng: PRNGKey（每张用独立 sub-rng 放起终点）。
    Returns:
      insts: EnvInstance（各字段前置 batch 维 n_levels）。
      valids: (n_levels,) bool。
    """
    n = env_maps.shape[0]
    rngs = jax.random.split(rng, n)
    return jax.vmap(lambda m, r: map_to_env_instance(jaxnav_env, m, r))(env_maps, rngs)


# ============================================================================
# 第 3 块（part 1）：terminal reward —— 造 map → 喂 student rollout → estimator 分
# ----------------------------------------------------------------------------
# 设计见 GENERATOR_DESIGN.md §3/§4：generator 的 reward = 对它造的 level 跑 student
# rollout 测出 success_rate，再过有方向的 estimator（difficulty ，§5.0 主力）
# 得到一个 **terminal 标量** —— 不是 PCGRL 几何 reward，而是 estimator 信号驱动。
#
# 本 part 只做「rollout → success_rate → difficulty 分」（estimator stage 0），
# 即第 3 块最易错的「跑 student 拿分」骨架，可独立单测。PVL/CENIE（需 GAE/hidden）
# 与 PPO 更新循环是 part 2/3。
#
# 全部接口 Oscar sfl env 真验证（2026-06-21）：
#   student env  : jaxnav JaxNav(num_agents=1)，obs=dict{agent_0}，action 连续 (2,)
#   student net  : ActorCriticRNN(action_dim, config=t_config)；
#                  t_config 需 HIDDEN_SIZE/FC_DIM_SIZE/USE_LAYER_NORM/LOG_DORMANCY
#   apply        : network.apply(params, hstate, ac_in) → (hstate, pi, value, _)
#                  ac_in = (obs_batch[None,:], done[None,:])
#   set_env_inst : env.set_env_instance(inst) → (obs, state)（把指定 level 装进去）
#   success_rate : 复用 SFL _calc_outcomes_by_agent 的 episode 切分（done_idxs+mask），
#                  非粗算 n_goal/n_done，保证与 get_learnability_set 语义一致。
# ============================================================================
from functools import partial as _partial


def _sfl_batchify(x, agents, num_actors):
    """复刻 jaxnav_sfl.batchify（dict obs → (num_actors, -1)）。"""
    return jnp.stack([x[a] for a in agents]).reshape((num_actors, -1))


def _sfl_unbatchify(x, agents, num_envs, num_actors):
    """复刻 jaxnav_sfl.unbatchify（action → dict per agent）。"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agents)}


def _calc_success_by_level(rollout_steps, dones, goalr_actor, num_agents, num_levels):
    """复刻 SFL _calc_outcomes_by_agent 的 success_rate / num_episodes（per level）。

    用 episode 边界切分（done_idxs，size=10），mask 出每个完成 episode 的 GoalR 均值。
    Args:
      rollout_steps: int（静态）。
      dones: (T, BATCH_ACTORS) bool，每步是否 episode 结束。
      goalr_actor: (T, BATCH_ACTORS) float，每步 GoalR（到达目标=1）。
      num_agents, num_levels: int。
    Returns:
      success_rate: (num_levels,) p∈[0,1]（agent 维取均值）。
      num_episodes: (num_levels,) 完成 episode 数（agent 维求和）。
    """
    @_partial(jax.vmap, in_axes=(1, 1))      # over BATCH_ACTORS 轴
    def _per_actor(dones_a, goalr_a):
        idxs = jnp.arange(rollout_steps)
        done_idxs = jnp.argwhere(dones_a, size=10, fill_value=rollout_steps).squeeze()
        mask_done = jnp.where(done_idxs == rollout_steps, 0, 1)
        starts = jnp.concatenate([jnp.array([-1]), done_idxs[:-1]])

        @_partial(jax.vmap, in_axes=(0, 0))
        def _ep(start_idx, end_idx):
            mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != rollout_steps)
            return jnp.sum(goalr_a * mask)
        succ = _ep(starts, done_idxs)
        return succ.mean(where=mask_done), mask_done.sum()

    succ_actor, nep_actor = _per_actor(dones, goalr_actor)   # (BATCH_ACTORS,)
    succ = succ_actor.reshape((num_agents, num_levels)).mean(axis=0)
    nep = nep_actor.reshape((num_agents, num_levels)).sum(axis=0)
    return succ, nep


def student_rollout_on_levels(env, network, network_params, env_instances,
                              rng, hidden_size, rollout_steps, num_levels,
                              emit_hidden=True):
    """对一批指定 level（generator 造的 EnvInstance）跑 student rollout → 完整轨迹。

    起点 = env.set_env_instance（非 env.reset），其余复刻 get_learnability_set 内循环。
    一次 rollout emit **全部信号**（done/goalr/value/reward/hidden），供三种 terminal
    reward（difficulty/PVL/CENIE）各取所需（统一 rollout，不重跑）。

    Args:
      env: jaxnav JaxNav（student env，num_agents=1）。
      network: ActorCriticRNN（student policy，GRU）。
      network_params: student 参数。
      env_instances: EnvInstance（batch 维 num_levels；generator 产出经第1/2块桥接）。
      rng: PRNGKey。
      hidden_size, rollout_steps, num_levels: int。
      emit_hidden: 是否收集 GRU hidden（CENIE 需要；difficulty/PVL 不需可省显存）。
    Returns:
      traj: dict {
        "dones":   (T, BATCH_ACTORS) bool，每步是否 episode 结束,
        "goalr":   (T, BATCH_ACTORS) float，每步 GoalR（到达目标=1）,
        "values":  (T, BATCH_ACTORS) float，进入该步前的 value（与 reward 对齐，GAE 用）,
        "rewards": (T, BATCH_ACTORS) float，该步 env reward（PVL 的 GAE 用）,
        "hidden":  (T, BATCH_ACTORS, H) 进入该步前的 GRU hidden（CENIE 用；emit_hidden=False 时占位标量）,
      }
    """
    from sfl.train.common.network import ScannedRNN
    BATCH_ACTORS = num_levels * env.num_agents
    # 把 batch EnvInstance 装进 env → 初始 obs/state（vmap over level 轴）。
    obsv, state = jax.vmap(env.set_env_instance, in_axes=(0,))(env_instances)

    def _step(carry, _):
        state, obs, done, hstate, rng = carry
        rng, _r = jax.random.split(rng)
        ob = _sfl_batchify(obs, env.agents, BATCH_ACTORS)
        # hstate_in = 进入该步前的 hidden（emit 它作 CENIE 特征，与 value/action 对齐，
        # 复刻 get_learnability_set 第 249 行「emit 进入该步前 hidden」）。
        hstate_in = hstate
        hstate, pi, value, _ = network.apply(network_params, hstate, (ob[None, :], done[None, :]))
        action = pi.sample(seed=_r)
        env_act = _sfl_unbatchify(action, env.agents, num_levels, env.num_agents)
        # reshape((num_levels,-1)) 而非 squeeze()：squeeze 在 num_levels=1 时会把 (1,2)
        # 误塌成 (2,) 丢掉 level 轴致 vmap 轴错（num_levels=1 退化 bug）；reshape 对任意 NL 健壮。
        env_act = {k: v.reshape((num_levels, -1)) for k, v in env_act.items()}
        rng, _r = jax.random.split(rng)
        rs = jax.random.split(_r, num_levels)
        obsv, state, reward, dn, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(rs, state, env_act, state)
        done_b = _sfl_batchify(dn, env.agents, BATCH_ACTORS).reshape((BATCH_ACTORS,))
        goalr_b = info["GoalR"].swapaxes(0, 1).reshape(-1)    # (BATCH_ACTORS,) per step
        # reward：env.step 出的 per-agent reward → (BATCH_ACTORS,)。share_only_sparse/do_sep_reward
        # 下 reward 可能多分量，这里取 batchify 后的标量（与 get_learnability_set 的 traj.reward 一致）。
        reward_b = _sfl_batchify(reward, env.agents, BATCH_ACTORS).reshape((BATCH_ACTORS,))
        value_b = value.reshape((BATCH_ACTORS,))              # network 出 (1, BATCH_ACTORS) → 压平
        # hidden：hstate_in 是 (BATCH_ACTORS, H)。emit_hidden=False 时收标量占位省显存。
        hid_emit = hstate_in if emit_hidden else jnp.zeros((), dtype=jnp.float32)
        return (state, obsv, done_b, hstate, rng), (done_b, goalr_b, value_b, reward_b, hid_emit)

    hstate0 = ScannedRNN.initialize_carry(BATCH_ACTORS, hidden_size)
    init = (state, obsv, jnp.zeros(BATCH_ACTORS, dtype=bool), hstate0, rng)
    _, (dones, goalr, values, rewards, hidden) = jax.lax.scan(_step, init, None, rollout_steps)
    return {"dones": dones, "goalr": goalr, "values": values,
            "rewards": rewards, "hidden": hidden}


def _success_from_traj(traj, rollout_steps, num_agents, num_levels):
    """从 student rollout 轨迹 dict 算 per-level success_rate / num_episodes。"""
    return _calc_success_by_level(
        rollout_steps, traj["dones"], traj["goalr"], num_agents, num_levels)


def _gae_from_traj(traj, gamma, gae_lambda):
    """从轨迹 dict 算 per-step advantages（reverse-scan GAE，复刻 get_learnability_set §324-333）。

    traj 各字段 (T, BATCH_ACTORS)。last_val=最后一步 value（bootstrap）。
    Returns: advantages (T, BATCH_ACTORS)。
    """
    dones, values, rewards = traj["dones"], traj["values"], traj["rewards"]
    last_val = values[-1]

    def _body(carry, x):
        gae, next_value = carry
        done, value, reward = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    _, adv = jax.lax.scan(
        _body, (jnp.zeros_like(last_val), last_val),
        (dones, values, rewards), reverse=True, unroll=16)
    return adv


def terminal_reward_difficulty(env, network, network_params, env_instances,
                               rng, hidden_size, rollout_steps, num_levels):
    """generator terminal reward —— difficulty-match -(p-0.5)^2（estimator 主力，§5.0）。

    串：student_rollout_on_levels → success_rate → estimators.difficulty_match。
    incomplete level（num_episodes=0）→ -inf（对齐 estimator 契约）。
    Returns:
      reward: (num_levels,) per-level terminal 标量；info: dict 备查。
    """
    from estimators import difficulty_match
    traj = student_rollout_on_levels(
        env, network, network_params, env_instances, rng, hidden_size, rollout_steps,
        num_levels, emit_hidden=False)
    succ, nep = _success_from_traj(traj, rollout_steps, env.num_agents, num_levels)
    reward = difficulty_match(succ, num_episodes=nep)
    return reward, {"success_rate": succ, "num_episodes": nep}


def terminal_reward_pvl(env, network, network_params, env_instances,
                        rng, hidden_size, rollout_steps, num_levels,
                        gamma=0.99, gae_lambda=0.95):
    """generator terminal reward —— PVL（positive value loss，regret 系，§4 generator-C）。

    串：rollout → GAE（reverse-scan）→ jaxued.positive_value_loss(dones, advantages)。
    复刻 get_learnability_set 的 PVL 路径（§324-335）：per-actor PVL → reshape 到 per-level。
    incomplete level（episode_count=0）→ positive_value_loss 内部置 -inf。
    Returns:
      reward: (num_levels,) ；info: dict 备查。
    """
    from sfl.util.jaxued.jaxued_utils import positive_value_loss
    traj = student_rollout_on_levels(
        env, network, network_params, env_instances, rng, hidden_size, rollout_steps,
        num_levels, emit_hidden=False)
    adv = _gae_from_traj(traj, gamma, gae_lambda)                 # (T, BATCH_ACTORS)
    # positive_value_loss(dones, advantages) → per-actor (BATCH_ACTORS,)；reshape per-level 取均值
    # （与 get_learnability_set 第 334-335 行一致：reshape((num_agents, num_levels)).mean(axis=0)）。
    pvl_actor = positive_value_loss(traj["dones"], adv)           # (BATCH_ACTORS,)
    pvl = pvl_actor.reshape((env.num_agents, num_levels)).mean(axis=0)   # (num_levels,)
    # incomplete actor 是 -inf，reshape.mean 会传染 -inf 到该 level（与原版同：incomplete→-inf）。
    return pvl, {"pvl_actor": pvl_actor}


def terminal_reward_cenie(env, network, network_params, env_instances,
                          rng, hidden_size, rollout_steps, num_levels,
                          gmm_params, probe_hidden_steps=16):
    """generator terminal reward —— CENIE（GMM 反密度，exploration 主力，§4 generator-B）。

    串：rollout(emit hidden) → 时间维 subsample(防 OOM，K=probe_hidden_steps) →
    cenie_neg_logp(gmm, hidden) → per-level mean。复刻 get_learnability_set §339-348。
    gmm_params=None 或 valid=False（冷启）时 cenie_neg_logp 返回 0（CENIE 暂不参与）。
    incomplete level（无完整 episode）→ -inf（对齐其它 estimator 契约）。
    Returns:
      reward: (num_levels,) ；info: dict 备查。
    """
    from cenie_density import cenie_neg_logp
    traj = student_rollout_on_levels(
        env, network, network_params, env_instances, rng, hidden_size, rollout_steps,
        num_levels, emit_hidden=True)
    hidden = traj["hidden"]                                       # (T, BATCH_ACTORS, H)
    BATCH_ACTORS = num_levels * env.num_agents
    H = hidden.shape[-1]
    # 时间维 subsample K 帧（jit 内降 OOM，见 memory probe-hidden-oom-subsample）。
    Kc = int(probe_hidden_steps)
    t_idx = jnp.linspace(0, rollout_steps - 1, Kc).astype(jnp.int32)
    hs_sub = hidden[t_idx]                                        # (Kc, BATCH_ACTORS, H)
    nlp = cenie_neg_logp(gmm_params, hs_sub.reshape(-1, H))       # (Kc*BATCH_ACTORS,)
    nlp = nlp.reshape(Kc, BATCH_ACTORS).mean(axis=0)             # (BATCH_ACTORS,) 时间均值
    cenie = nlp.reshape((env.num_agents, num_levels)).mean(axis=0)   # (num_levels,)
    # incomplete level（无完整 episode）→ -inf。用 success 的 num_episodes 判定 complete。
    _succ, nep = _success_from_traj(traj, rollout_steps, env.num_agents, num_levels)
    cenie = jnp.where(nep > 0, cenie, -jnp.inf)
    return cenie, {"cenie_raw": cenie, "num_episodes": nep}


def all_signals_on_levels(env, network, network_params, env_instances, rng,
                          hidden_size, rollout_steps, num_levels, gmm_params=None,
                          gamma=0.99, gae_lambda=0.95, probe_hidden_steps=16):
    """对一批 level **一次 student rollout** → difficulty/PVL/CENIE 三信号分（组 (3, M)）。

    供 auction 出价漏斗用：N 个 estimator 对**全部候选 level**都打分（同一轨迹各算一列），
    才能组 (N, M) 矩阵竞价。与 get_learnability_set 同法（一次 rollout 出三信号），
    避免三个 terminal_reward 各跑一次 rollout（省 3× 算力）。

    顺序固定 [difficulty, PVL, CENIE]，与 auction estimator 维度对齐。CENIE 需 gmm_params
    （None/valid=False 时该列恒 0=不参与竞价，对齐 get_learnability_set 冷启行为）。
    incomplete level：difficulty/CENIE → -inf，PVL → -inf（positive_value_loss 内部）。

    Returns:
      per_est: (3, num_levels) —— 行 = [difficulty, pvl, cenie] 的 per-level 分。
      info: dict（success_rate/num_episodes 备查）。
    """
    from estimators import difficulty_match
    from sfl.util.jaxued.jaxued_utils import positive_value_loss
    from cenie_density import cenie_neg_logp

    traj = student_rollout_on_levels(
        env, network, network_params, env_instances, rng, hidden_size, rollout_steps,
        num_levels, emit_hidden=True)   # emit_hidden=True：CENIE 要 hidden
    succ, nep = _success_from_traj(traj, rollout_steps, env.num_agents, num_levels)

    # difficulty：-(p-0.5)²，incomplete→-inf。
    difficulty = difficulty_match(succ, num_episodes=nep)            # (num_levels,)

    # PVL：GAE → positive_value_loss → per-level。
    adv = _gae_from_traj(traj, gamma, gae_lambda)
    pvl_actor = positive_value_loss(traj["dones"], adv)
    pvl = pvl_actor.reshape((env.num_agents, num_levels)).mean(axis=0)

    # CENIE：hidden subsample → cenie_neg_logp → per-level mean，incomplete→-inf。
    hidden = traj["hidden"]                                          # (T, BATCH_ACTORS, H)
    BATCH_ACTORS = num_levels * env.num_agents
    H = hidden.shape[-1]
    Kc = int(probe_hidden_steps)
    t_idx = jnp.linspace(0, rollout_steps - 1, Kc).astype(jnp.int32)
    hs_sub = hidden[t_idx]
    nlp = cenie_neg_logp(gmm_params, hs_sub.reshape(-1, H)) if gmm_params is not None \
        else jnp.zeros((Kc * BATCH_ACTORS,))
    nlp = nlp.reshape(Kc, BATCH_ACTORS).mean(axis=0)
    cenie = nlp.reshape((env.num_agents, num_levels)).mean(axis=0)
    cenie = jnp.where(nep > 0, cenie, -jnp.inf)

    per_est = jnp.stack([difficulty, pvl, cenie], axis=0)           # (3, num_levels)
    return per_est, {"success_rate": succ, "num_episodes": nep}


# ============================================================================
# 第 3 块（part 3a）：generator PPO —— 轨迹收集 + GAE（纯 terminal reward）
# ----------------------------------------------------------------------------
# 设计（用户拍板 2026-06-21）：纯 terminal reward —— generator 跑满 363 步逐 tile
# 造完整张 map，只在最后一步给 difficulty 分（estimator 对成品 map 打分），前 362
# 步 reward=0。GAE reverse-scan 把终端信号反传到各步。先单 generator 闭环跑通
# 「rollout 收轨迹 → GAE → PPO loss → 更新」，验证 generator 真能被信号训动
# （方案 B 第一核心 claim）。N=3 / PVL / CENIE 留后续。
#
# 关键：rollout 与 PPO loss 必须共用同一个 distrax 分布构造（make_gen_pi），否则
# log_prob 维度（n_agents,*act_shape）对不齐 —— PPO 最常见的坑。
# ============================================================================


class GenTransition(NamedTuple):
    """generator PPO 轨迹的单步记录（per env step）。"""
    map_obs: jnp.ndarray      # (31,31,4)
    flat_obs: jnp.ndarray     # (1,)
    action: jnp.ndarray       # (n_agents,*act_shape)
    log_prob: jnp.ndarray     # 标量（对 action 各维 log_prob 求和）
    value: jnp.ndarray        # 标量
    reward: jnp.ndarray       # 标量（仅末步 = difficulty 分，其余 0）
    done: jnp.ndarray         # 标量 bool（仅末步 True）


def make_gen_pi(logits, n_agents, act_shape, action_dim):
    """logits (B, action_dim*prod(act_shape)) → distrax.Categorical（rollout/loss 共用）。

    reshape 出 (B, n_agents, *act_shape, action_dim)，与 logits_to_action 一致。
    log_prob/entropy 对 (n_agents,*act_shape) 各维独立 → 用时 sum 到标量。
    """
    import distrax
    B = logits.shape[0]
    logits = logits.reshape((B, n_agents, *act_shape, action_dim))
    return distrax.Categorical(logits=logits)


def gen_rollout_traj(env, network, params, rng):
    """单 generator 跑满 max_steps，收集完整 PPO 轨迹 + 成品 map。

    Returns:
      traj: GenTransition（前置 T=max_steps 维；每字段 leading dim = 步数）。
      final_env_map: (H,W) 成品 map。
      last_val: 标量，末状态 value（GAE bootstrap 用；terminal 故实际乘 0）。
    """
    from envs.pcgrl_env import gen_dummy_queued_state
    rep = env.rep
    n_agents = 1
    act_shape = tuple(int(s) for s in rep.act_shape)
    action_dim = int(rep.action_space.n) // (act_shape[0] * act_shape[1])
    env_params = _env_params_of(env)
    max_steps = int(rep.max_steps)

    rng, r_reset = jax.random.split(rng)
    qs = gen_dummy_queued_state(env)
    obs0, state0 = env.reset_env(r_reset, env_params, qs)

    def _step(carry, step_i):
        rng, obs, state = carry
        rng, r_act, r_step = jax.random.split(rng, 3)
        logits, val = network.apply(params, obs.map_obs[None, ...], obs.flat_obs[None, ...])
        pi = make_gen_pi(logits, n_agents, act_shape, action_dim)
        action = pi.sample(seed=r_act)                 # (1, n_agents, *act_shape)
        log_prob = pi.log_prob(action).sum()           # 对各维求和 → 标量
        action_env = action[0]                         # 去 batch → (n_agents,*act_shape)
        obs2, state2, _r, _d, _i = env.step_env(r_step, state, action_env, env_params)
        # 纯 terminal reward：rollout 内不知 difficulty（要喂 student），故先记 0；
        # 末步 reward 由调用方在拿到 terminal_reward 后回填（见 assign_terminal_reward）。
        is_last = step_i == (max_steps - 1)
        tr = GenTransition(
            map_obs=obs.map_obs, flat_obs=obs.flat_obs,
            action=action[0], log_prob=log_prob, value=val[0],
            reward=jnp.float32(0.0), done=is_last,
        )
        return (rng, obs2, state2), tr

    (rng, _obs, state_final), traj = jax.lax.scan(
        _step, (rng, obs0, state0), jnp.arange(max_steps))
    # 末状态 value（terminal，GAE 里乘 (1-done) 归零，仅占位）。
    last_val = jnp.float32(0.0)
    return traj, state_final.env_map, last_val


def assign_terminal_reward(traj, terminal_reward):
    """把整 episode 的 terminal difficulty 分写到轨迹最后一步的 reward（纯 terminal）。

    Args:
      traj: GenTransition（reward 当前全 0）。
      terminal_reward: 标量，= terminal_reward_difficulty 出的该 map 的 difficulty 分。
    Returns:
      traj: reward 末步 = terminal_reward，其余 0。
    """
    T = traj.reward.shape[0]
    new_reward = traj.reward.at[T - 1].set(terminal_reward)
    return traj._replace(reward=new_reward)


def compute_gae(traj, last_val, gamma=0.99, gae_lambda=0.95):
    """对 generator 轨迹算 GAE advantage + returns（标准 reverse-scan）。

    纯 terminal reward：只有末步 reward≠0，GAE 把它按 gamma*lambda 反传到各步。
    Args:
      traj: GenTransition（reward 已 assign_terminal_reward）。
      last_val: 标量，bootstrap value。
    Returns:
      advantages: (T,) ；returns: (T,) = advantages + values。
    """
    def _body(carry, x):
        gae, next_val = carry
        reward, value, done = x
        delta = reward + gamma * next_val * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae

    rev = (traj.reward[::-1], traj.value[::-1], traj.done[::-1])
    _, adv_rev = jax.lax.scan(_body, (jnp.float32(0.0), last_val), rev)
    advantages = adv_rev[::-1]
    returns = advantages + traj.value
    return advantages, returns


# ============================================================================
# 第 3 块（part 3b）：generator PPO loss + update —— 闭环最后一块
# ----------------------------------------------------------------------------
# loss：对轨迹各步 re-apply ConvForward 拿新 logits/value → make_gen_pi 算新
# log_prob（与 rollout 共用分布构造，维度对齐）→ PPO clip ratio + value MSE +
# entropy bonus。第一版 full-batch 多 epoch（轨迹仅 363 步，无需 minibatch 切分）。
# update：optax adam。gen_train_iter = 完整一次迭代（rollout→terminal reward→
# GAE→多 epoch update），返回新 params + 指标。验证 loss 下降 / 参数真动。
# ============================================================================


def gen_ppo_loss(params, network, traj, advantages, returns,
                 n_agents, act_shape, action_dim,
                 clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, norm_adv=True):
    """generator PPO loss（actor clip + critic MSE + entropy）—— 单 level 版（保留作单测/参考）。

    对轨迹 T 步 vmap re-apply network（每步独立 map_obs/flat_obs）。
    Returns:
      total_loss, (actor_loss, value_loss, entropy)。
    """
    def _fwd(map_o, flat_o):
        logits, val = network.apply(params, map_o[None, ...], flat_o[None, ...])
        return logits[0], val[0]
    logits, values = jax.vmap(_fwd)(traj.map_obs, traj.flat_obs)   # (T, ...), (T,)
    # logits: (T, action_dim*prod(act_shape))；make_gen_pi 内部 reshape 出分布维度。
    pi = make_gen_pi(logits, n_agents, act_shape, action_dim)
    # log_prob/entropy 在 (n_agents,*act_shape) 各维 → sum 到 per-step 标量（与 rollout 一致）。
    new_log_prob = pi.log_prob(traj.action).reshape((traj.action.shape[0], -1)).sum(axis=-1)
    entropy = pi.entropy().reshape((traj.action.shape[0], -1)).sum(axis=-1).mean()

    adv = advantages
    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ratio = jnp.exp(new_log_prob - traj.log_prob)
    a1 = ratio * adv
    a2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    actor_loss = -jnp.minimum(a1, a2).mean()

    value_loss = 0.5 * ((values - returns) ** 2).mean()

    total = actor_loss + vf_coef * value_loss - ent_coef * entropy
    return total, (actor_loss, value_loss, entropy)


def gen_ppo_loss_batched(params, network, trajs, advantages, returns,
                         n_agents, act_shape, action_dim,
                         clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, norm_adv=True):
    """[A2 提速·数值等价] 整批 L levels 的 generator PPO loss，消除 nested vmap。

    与「vmap(gen_ppo_loss) over L 个 level 再 .mean()」**逐元素数值等价**，但把 conv 前向
    从 vmap(L)∘vmap(T)=batch L*T 的怪形状(64*363=23232,cuDNN 无快算法→慢 autotune)摊平成
    单 batch (L*T,)，让 cuDNN 选到高效算法。等价性保证：

      - conv 逐样本独立 → flatten (L,T)->(L*T) 计算再 reshape 回 (L,T) 数值不变；
      - norm_adv 仍 **per-level**（沿 T 轴 mean/std，keepdims，不跨 level 全局化）；
      - actor/value/entropy 先在每 level 内 .mean()(沿 T) → 再跨 level .mean()，
        等于原「每 level gen_ppo_loss().mean() 再外层 ls.mean()」。

    输入形状：trajs.map_obs (L, T, *obs)，advantages/returns (L, T)，trajs.action (L, T, *act)。
    Returns: total_loss 标量, (actor_loss, value_loss, entropy) 标量三元组。
    """
    L = trajs.map_obs.shape[0]
    T = trajs.map_obs.shape[1]

    # --- conv 前向：flatten (L,T,...) -> (L*T,...) 单 batch 过网络（make_gen_pi 要单 batch 维）---
    flat_map = trajs.map_obs.reshape((L * T,) + trajs.map_obs.shape[2:])
    flat_flat = trajs.flat_obs.reshape((L * T,) + trajs.flat_obs.shape[2:])
    logits_f, values_f = network.apply(params, flat_map, flat_flat)   # (L*T, A), (L*T,)

    # --- 分布 / log_prob / entropy：全程在 (L*T,) 单 batch 维算，再 reshape 回 (L,T) ---
    pi = make_gen_pi(logits_f, n_agents, act_shape, action_dim)       # B=L*T
    act_f = trajs.action.reshape((L * T,) + trajs.action.shape[2:])   # (L*T, *act)
    new_log_prob = pi.log_prob(act_f).reshape((L * T, -1)).sum(axis=-1).reshape((L, T))
    entropy_lt = pi.entropy().reshape((L * T, -1)).sum(axis=-1).reshape((L, T))
    entropy = entropy_lt.mean(axis=1).mean()            # per-level mean(T) → mean(L)
    values = values_f.reshape((L, T))

    # --- advantage 标准化：per-level（沿 T 轴），keepdims 广播回 (L,T) ---
    adv = advantages
    if norm_adv:
        m = adv.mean(axis=1, keepdims=True)
        s = adv.std(axis=1, keepdims=True)
        adv = (adv - m) / (s + 1e-8)

    ratio = jnp.exp(new_log_prob - trajs.log_prob)      # (L, T)
    a1 = ratio * adv
    a2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    actor_loss = (-jnp.minimum(a1, a2)).mean(axis=1).mean()   # per-level mean(T) → mean(L)

    value_loss = (0.5 * ((values - returns) ** 2)).mean(axis=1).mean()

    total = actor_loss + vf_coef * value_loss - ent_coef * entropy
    return total, (actor_loss, value_loss, entropy)


# 各 generator 绑定的 estimator id（reward 信号源）。N=3 各绑一个（§4 防 mode collapse）。
# 定义在 compute_terminal_reward/gen_train_iter 之前：gen_train_iter 用其作**默认参数值**，
# 而默认参数在函数定义时求值（非调用时），故常量须先于函数定义存在（否则 NameError）。
GEN_ESTIMATOR_DIFFICULTY = "difficulty"   # -(p-0.5)²，造"刚好能学"（part1）
GEN_ESTIMATOR_PVL = "pvl"                 # positive value loss，regret 系（part2）
GEN_ESTIMATOR_CENIE = "cenie"             # GMM 反密度，exploration 主力（part2，需 gmm_params）
# [卡死修复·C] invalid 图（放不下合法起终点）的固定强负 terminal reward。difficulty=-(p-0.5)²∈
#   [-0.25,0]，故 -1.0 显著低于任何 valid 图 → 形成清晰"少造 invalid"梯度，量级又不至于炸 PPO。
GEN_INVALID_PENALTY = -1.0


def compute_terminal_reward(estimator_id, student_env, student_network, student_params,
                            insts, rng, hidden_size, student_rollout_steps, num_levels,
                            gmm_params=None, gamma=0.99, gae_lambda=0.95):
    """按 estimator_id 分派到三种 terminal reward（difficulty/pvl/cenie）。

    estimator_id 是静态字符串（jit 下走静态分支，非 traced）。CENIE 需 gmm_params。
    Returns: reward (num_levels,)，info dict。
    """
    if estimator_id == GEN_ESTIMATOR_DIFFICULTY:
        return terminal_reward_difficulty(
            student_env, student_network, student_params, insts, rng,
            hidden_size, student_rollout_steps, num_levels)
    elif estimator_id == GEN_ESTIMATOR_PVL:
        return terminal_reward_pvl(
            student_env, student_network, student_params, insts, rng,
            hidden_size, student_rollout_steps, num_levels,
            gamma=gamma, gae_lambda=gae_lambda)
    elif estimator_id == GEN_ESTIMATOR_CENIE:
        return terminal_reward_cenie(
            student_env, student_network, student_params, insts, rng,
            hidden_size, student_rollout_steps, num_levels, gmm_params=gmm_params)
    else:
        raise ValueError(f"未知 estimator_id={estimator_id}（应为 "
                         f"{GEN_ESTIMATOR_DIFFICULTY}/{GEN_ESTIMATOR_PVL}/{GEN_ESTIMATOR_CENIE}）")


def gen_train_iter(env, network, params, opt_state, optimizer, rng,
                   student_env, student_network, student_params,
                   hidden_size, student_rollout_steps, num_levels,
                   estimator_id=GEN_ESTIMATOR_DIFFICULTY, gmm_params=None,
                   place_env=None,
                   ppo_epochs=4, gamma=0.99, gae_lambda=0.95,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """generator 一次完整 PPO 迭代（单 generator，terminal reward 由 estimator_id 决定）。

    rollout（造 map + 收轨迹）→ 喂 student 算 terminal reward（difficulty/PVL/CENIE）→
    回填 → GAE → ppo_epochs 次 full-batch 更新。num_levels 张轨迹各自训。

    Args:
      estimator_id: 该 generator 的信号源（GEN_ESTIMATOR_*，静态）。CENIE 需 gmm_params。
      gmm_params: CENIE 用的已拟合 GMM（None/valid=False 时 CENIE 返 0）。
      place_env: 放起终点用 jaxnav env（应 valid_path_check=True，§R5）；None 退回 student_env。
                 rollout 仍在 student_env 上（与训练同分布）。
    Returns:
      params, opt_state, metrics(dict: total/actor/value/entropy/mean_reward)。
    """
    if place_env is None:
        place_env = student_env
    rep = env.rep
    n_agents = 1
    act_shape = tuple(int(s) for s in rep.act_shape)
    action_dim = int(rep.action_space.n) // (act_shape[0] * act_shape[1])

    # 1) num_levels 条 generator 轨迹（vmap rollout）。
    rng, r_roll = jax.random.split(rng)
    roll_rngs = jax.random.split(r_roll, num_levels)
    trajs, env_maps, last_vals = jax.vmap(
        lambda r: gen_rollout_traj(env, network, params, r))(roll_rngs)
    # trajs: GenTransition，各字段前置 (num_levels, T, ...)。

    # 2) batch map → EnvInstance（起终点用 place_env=valid_path_check=True 根除不可解）
    #    → student（在 student_env 上 rollout）→ terminal reward（按 estimator_id 分派）。
    rng, r_inst, r_stud = jax.random.split(rng, 3)
    insts, valids = gen_batch_to_env_instances(env_maps, place_env, r_inst)
    terminal_rew, _info = compute_terminal_reward(
        estimator_id, student_env, student_network, student_params, insts, r_stud,
        hidden_size, student_rollout_steps, num_levels,
        gmm_params=gmm_params, gamma=gamma, gae_lambda=gae_lambda)   # (num_levels,)
    # incomplete level（-inf）→ 置 0（不参与梯度，避免 NaN）。
    terminal_rew = jnp.where(jnp.isfinite(terminal_rew), terminal_rew, 0.0)
    # [卡死修复·C 奖励端罚分] invalid 图（place_start_goal 耗尽 MAX_PLACE_TRIES，放不下合法
    #   起终点）→ 固定强负 reward，让 generator PPO 学到"造 invalid=亏"，从源头抑制 reward-hack。
    #   不做则 difficulty 信号会把 success=0 的不可解图当"最难=最优"正向激励，invalid 产出率随训练
    #   单调升高（阶段二尤甚）。GEN_INVALID_PENALTY 量级与正常 difficulty 分相当（默认 -1.0）。
    terminal_rew = jnp.where(valids, terminal_rew, GEN_INVALID_PENALTY)

    # 3) 回填 terminal + GAE（per level，vmap）。
    def _backfill_gae(traj, trew, lv):
        traj = assign_terminal_reward(traj, trew)
        adv, ret = compute_gae(traj, lv, gamma, gae_lambda)
        return traj, adv, ret
    trajs, advs, rets = jax.vmap(_backfill_gae)(trajs, terminal_rew, last_vals)

    # 4) PPO 更新（ppo_epochs 次；loss 对 num_levels 条轨迹平均）。
    #    [A2 提速] 用 gen_ppo_loss_batched 替代 vmap(_per_level over L)——消除 nested vmap，
    #    conv 前向摊平成单 batch 让 cuDNN 选到快算法。与原 vmap+mean 逐元素数值等价。
    def _epoch(carry, _):
        params, opt_state = carry
        def _loss_fn(p):
            return gen_ppo_loss_batched(p, network, trajs, advs, rets,
                                        n_agents, act_shape, action_dim,
                                        clip_eps, vf_coef, ent_coef)
        (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (loss, aux)

    (params, opt_state), (losses, auxes) = jax.lax.scan(
        _epoch, (params, opt_state), None, ppo_epochs)
    actor_l, value_l, ent = auxes
    metrics = {
        "total_loss": losses,                # (ppo_epochs,)
        "actor_loss": actor_l,
        "value_loss": value_l,
        "entropy": ent,
        "mean_terminal_reward": terminal_rew.mean(),
    }
    return params, opt_state, metrics


# ============================================================================
# 第 4 块：get_generator_set —— 交替训练阶段 G + 产 instances（接 jaxnav_sfl）
# ----------------------------------------------------------------------------
# 设计见 GENERATOR_DESIGN.md §0/§5/§6：替换 SFL get_learnability_set 的随机海选。
# 交替训练 outer round 的「阶段 G（冻 student，更新 generator）」=
#   for g in 1..N: gen_train_iter(student 冻结) 更新 generator-g
#   → 用更新后的 generator 造 n_inject 张 level → 拼成 instances 注入固定 buffer
#   → 返回 (scores, instances, new_gen_state)；student 训练（阶段 S）由 SFL 原
#     train_step scan 不动消费 instances。
#
# generator 状态（params/opt_state）不进 jit 的 runner_state，而作宿主 Python
# 变量穿过 train_and_eval_step（与 gmm_params 同模式，见 jaxnav_sfl.py §924）。
#
# 冒烟版 N=1 + difficulty（GENERATOR_DESIGN §8.3）；N=3 各绑 difficulty/PVL/CENIE
# 是 §8.4，扩 N 只需 make_generator_state 多建几个 + estimator 分派（接口已按 N 设）。
# ============================================================================
import functools


def make_generator_state(rng, gen_env, gen_network, estimator_ids, gen_lr=2.5e-4,
                         max_grad_norm=0.5):
    """初始化 N 个 generator 的 (params, opt_state, optimizer, estimator_id)。

    各 generator 独立 params/optimizer（防 mode collapse，§4）；共用同一 network 结构。
    optimizer 与 SFL student 同款：clip_by_global_norm + adam。

    Args:
      rng: PRNGKey。
      gen_env: make_pcgrl_env 出的 PCGRL env（提供 reset/obs 形状给 init）。
      gen_network: make_generator_network 出的 ConvForward（未初始化）。
      estimator_ids: list[str]，长度 = N，各 generator 的 reward 信号（GEN_ESTIMATOR_*）。
      gen_lr, max_grad_norm: PPO 超参。
    Returns:
      gen_state: dict {
        "params": list[N] 各 generator params,
        "opt_state": list[N] 各 opt_state,
      }
        —— **纯数组 pytree**，可安全穿过 train_and_eval_step 的 jit 边界（不含 str）。
        estimator_ids 是静态信息，由外层闭包持有、作 get_generator_set 的参数传入，
        **不放进 gen_state**（jit 输出 pytree 不能含 Python str 叶子，会报非法 JAX 类型）。
      optimizer: optax GradientTransformation（N 个 generator 共用同一 tx 定义，
                 各自 opt_state 独立）。
    """
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(gen_lr, eps=1e-5),
    )
    params_list, opt_list = [], []
    rngs = jax.random.split(rng, len(estimator_ids))
    for r in rngs:
        p = init_generator_params(gen_network, gen_env, r)
        params_list.append(p)
        opt_list.append(optimizer.init(p))
    gen_state = {
        "params": params_list,
        "opt_state": opt_list,
    }
    return gen_state, optimizer


def get_generator_set(rng, student_params, gen_state, optimizer, estimator_ids,
                      gen_env, gen_network, student_env, student_network,
                      hidden_size, student_rollout_steps,
                      num_levels_per_gen, num_to_save, place_env=None,
                      gmm_params=None, auction_lambda=None,
                      gen_outer_steps=1, ppo_epochs=4,
                      gamma=0.99, gae_lambda=0.95,
                      clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """交替训练阶段 G：更新 N 个 generator（冻 student）→ 产 top-K instances 注入 buffer。

    替换 SFL get_learnability_set 的随机海选，输出**同型** (scores, instances)
    供 train_and_eval_step 下游 train_step scan 原样消费。

    流程（每个 generator g）：
      gen_outer_steps 次 gen_train_iter（冻 student_params，更新 gen-g）
      → 用更新后 gen-g params 造 num_levels_per_gen 张 level（rollout_batch）
      → 桥接成 EnvInstance + 算各 level 的 difficulty 分（terminal reward）当 score
    N 个 generator 的 level 拼起来（N×num_levels_per_gen 张）→ 按 score top-K 选 num_to_save。

    冒烟版只用 difficulty 信号（estimator_ids 全 "difficulty"）；N>1 也跑得通
    （各 generator 独立 params 自然分化），PVL/CENIE 分派是 part2 待做。

    Args:
      rng: PRNGKey。
      student_params: 冻结的 student 参数（= runner_state[0].params）。
      gen_state: make_generator_state 出的 dict（params/opt_state，纯数组 pytree）。
      optimizer: make_generator_state 出的 optimizer。
      estimator_ids: list[str]，各 generator 的信号源（静态，外层闭包持有）。
      gen_env, gen_network: PCGRL env + generator network。
      student_env, student_network: jaxnav env + student ActorCriticRNN。
      hidden_size, student_rollout_steps: student rollout 配置。
      num_levels_per_gen: 每个 generator 造几张 level。
      num_to_save: top-K 注入数（= SFL NUM_TO_SAVE，对齐 buffer 规模）。
      place_env: 放起终点的 jaxnav env（应 valid_path_check=True 根除不可解图，§R5）；
                 None 退回 student_env（False，生产慎用——difficulty 信号会激励 reward-hack
                 造不可解关卡）。rollout 始终在 student_env（与训练同分布）。
      auction_lambda: auction 出价漏斗温度（idea 核心，§6）。非 None → 对整 pool 跑
                 all_signals_on_levels 得 (3,M) → auction_mix_scores（z-score+bid+softmax）
                 → 混合 score 决定"现在最该学谁的课程"。λ=inf single-winner、有限大 uniform。
                 None → fallback 每 gen 用自己信号评自己 level（part2 行为，消融用）。
      gen_outer_steps: 每 outer round 每个 generator 跑几次 PPO 迭代（冒烟=1）。
      其余: PPO 超参。
    Returns:
      scores: (num_to_save,) top-K level 的 difficulty 分。
      instances: EnvInstance（batch 维 num_to_save），喂 set_env_instance 即可。
      new_gen_state: 更新后的 gen_state（params/opt_state 已 step，纯数组 pytree）。
      metrics: dict（per-generator PPO 指标 + 注入统计）。注意：调用方若在 jit 内
               且不返回 metrics，则其中的 Python int（gen_injected）不进 jit 输出。
    """
    if place_env is None:
        place_env = student_env
    params_list = list(gen_state["params"])
    opt_list = list(gen_state["opt_state"])
    N = len(estimator_ids)

    all_insts = []           # 各 generator 造的 EnvInstance（待 concat）
    all_scores = []          # 各 level 的 difficulty 分（待 concat）
    all_valids = []          # 各 level 是否放下了合法起终点（待 concat，B 注入端过滤用）
    per_gen_metrics = []

    for g in range(N):
        # ── 阶段 G：更新 generator-g（冻 student），terminal reward = 该 gen 的信号 ──
        est_g = estimator_ids[g]      # 静态字符串（difficulty/pvl/cenie）
        p, os_ = params_list[g], opt_list[g]
        last_metrics = None
        for _it in range(gen_outer_steps):
            rng, r_iter = jax.random.split(rng)
            p, os_, last_metrics = gen_train_iter(
                gen_env, gen_network, p, os_, optimizer, r_iter,
                student_env, student_network, student_params,
                hidden_size, student_rollout_steps, num_levels_per_gen,
                estimator_id=est_g, gmm_params=gmm_params,
                place_env=place_env,
                ppo_epochs=ppo_epochs, gamma=gamma, gae_lambda=gae_lambda,
                clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef)
        params_list[g], opt_list[g] = p, os_
        per_gen_metrics.append(last_metrics)

        # ── 用更新后的 generator-g 造注入 level（buffer 用 step-G 之后的产物）──
        # 起终点用 place_env（valid_path_check=True 根除不可解图）；rollout 仍 student_env。
        rng, r_roll, r_inst, r_score = jax.random.split(rng, 4)
        env_maps = gen_rollout_batch(gen_env, gen_network, p, r_roll, num_levels_per_gen)
        insts, valids_g = gen_batch_to_env_instances(env_maps, place_env, r_inst)
        all_insts.append(insts)
        all_valids.append(valids_g)   # [B] 接住 valid，下游把 invalid 图 score→-inf 挤出 buffer
        # fallback（auction_lambda=None）：每 gen 用自己信号评自己 level（part2 行为，消融用）。
        if auction_lambda is None:
            scores_g, _info = compute_terminal_reward(
                est_g, student_env, student_network, student_params, insts, r_score,
                hidden_size, student_rollout_steps, num_levels_per_gen,
                gmm_params=gmm_params, gamma=gamma, gae_lambda=gae_lambda)
            all_scores.append(scores_g)

    # ── 拼 N×num_levels_per_gen 张 pool ──
    cat_insts = jax.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_insts) \
        if N > 1 else all_insts[0]
    M = N * num_levels_per_gen

    # ── 评分漏斗：auction 多 teacher 出价混合（idea 核心）or fallback 单信号（消融）──
    if auction_lambda is not None:
        # auction：N 个 estimator 对**整个 pool** 都打分（一次 rollout 出三信号，§6）→
        # (3, M) → z-score 抹平量级 → bid → softmax(bid/λ) → 混合 score。决定"现在最该学谁的课程"。
        from auction_bid import mix_scores as auction_mix_scores
        rng, r_auc = jax.random.split(rng)
        per_est, _ainfo = all_signals_on_levels(
            student_env, student_network, student_params, cat_insts, r_auc,
            hidden_size, student_rollout_steps, M, gmm_params=gmm_params,
            gamma=gamma, gae_lambda=gae_lambda)                       # (3, M)
        cat_scores, auc_w, auc_bids = auction_mix_scores(per_est, float(auction_lambda))  # (M,)
        auction_info = {"weights": auc_w, "bids": auc_bids, "per_est": per_est}
    else:
        # fallback（auction_lambda=None，各 gen 自评不跨 gen auction）：各 gen 用的是**自己**
        # 信号的原始量级（difficulty 可达数百、pvl/cenie ~O(1)），直接 concat 会让量纲悬殊的 gen
        # 在 top-K 排序里恒赢/恒输，且使 learnability_set_var 落在原始量纲（阶段 1 实测 none 档
        # var~1090、scores~154–274），与 auction 档（已 z-score 到 ~O(1)）口径不可比、消融无法横比。
        # 修复（STAGE4 阶段 1 暴露）：各 gen 的 scores_g 先各自做 per-gen z-score（与 auction 路径
        # 的 standardize_per_estimator 同口径），再 concat。这样 none 与 auction 档同量纲、可横比，
        # 且 top-K 不再被原始量级悬殊主导。incomplete/invalid 仍走下方 -inf 通道（z-score 对
        # 非 finite 位原样保留）。
        from auction_bid import standardize_per_estimator
        per_gen_scores = jnp.stack(all_scores, axis=0)                # (N, num_levels_per_gen)
        per_gen_z = standardize_per_estimator(per_gen_scores)         # (N, ...) 各行 z-score
        cat_scores = per_gen_z.reshape(-1)                            # (M,) z-score 后拼接
        auction_info = None

    # ── [B 注入端过滤] invalid 图（放不下合法起终点）score→-inf，复用下方 incomplete 同款
    #    "-inf 被 argsort 排末尾→top-K 挤掉"通道，不污染 student 课程。物理删除会破坏 jit 静态
    #    形状（buffer 固定 num_to_save 槽位），故判分而非删除——与 incomplete 处理一致。
    #    ⚠ 同 incomplete：须保证有效图数(valid 且 finite)≥ num_to_save，否则 -inf 图被迫选入。
    #    阶段二 invalid 产出率升高会吃掉这个余量，届时需调大 GEN_NUM_LEVELS_PER_GEN（见文档）。
    cat_valids = jnp.concatenate(all_valids, axis=0)                  # (M,) bool
    cat_scores = jnp.where(cat_valids, cat_scores, -jnp.inf)

    # ── top-K 选 num_to_save ──
    # top-K 用**原始 cat_scores（含 -inf）排序**：incomplete level(-inf)被排到末尾，
    # pool(M)≥num_to_save 时被自然挤掉（对齐原版 get_learnability_set 靠大 pool 过滤）。
    # ⚠ 须保证 config GEN_NUM_LEVELS_PER_GEN×N ≥ NUM_TO_SAVE，否则 incomplete 会被选入。
    k = min(num_to_save, M)
    top_idx = jnp.argsort(cat_scores)[-k:]
    top_insts = jax.tree_map(lambda x: x.at[top_idx].get(), cat_insts)
    top_scores_raw = cat_scores.at[top_idx].get()
    # 返回的 scores 清洗 -inf（incomplete）→ 置为该批最小有限分，防 .mean()/wandb.log 被
    # -inf 污染。buffer 只用 instances 不用 score，故清洗不影响注入；仅净化统计/日志数值。
    finite = jnp.isfinite(top_scores_raw)
    min_finite = jnp.where(jnp.any(finite),
                           jnp.min(jnp.where(finite, top_scores_raw, jnp.inf)),
                           0.0)
    top_scores = jnp.where(finite, top_scores_raw, min_finite)

    new_gen_state = {
        "params": params_list,
        "opt_state": opt_list,
    }
    # gen_mean_score 也清洗 -inf（同理防污染）。
    cat_finite = jnp.isfinite(cat_scores)
    gen_mean = jnp.where(jnp.any(cat_finite),
                         jnp.mean(jnp.where(cat_finite, cat_scores, 0.0),
                                  where=cat_finite),
                         0.0)
    metrics = {
        "gen_per_gen": per_gen_metrics,
        "gen_pool_size": M,
        "gen_injected": int(k),
        # n_incomplete 现含 invalid（都被设 -inf）；单列 n_invalid 监控病态率（C 应使其随训练下降）。
        "gen_n_incomplete": (~cat_finite).sum(),     # traced 标量（jit 内不可 int()）
        "gen_n_invalid": (~cat_valids).sum(),        # [B/C 监控] 放不下合法起终点的图数
        "gen_mean_score": gen_mean,
        "gen_top_mean_score": top_scores.mean(),
    }
    if auction_info is not None:
        # auction 权重/bid（诊断：看三 teacher 谁出价高=谁的课程当前被选；λ 控 single-winner↔uniform）。
        metrics["auction_weights"] = auction_info["weights"]   # (N,)
        metrics["auction_bids"] = auction_info["bids"]         # (N,)
    return top_scores, top_insts, new_gen_state, metrics
