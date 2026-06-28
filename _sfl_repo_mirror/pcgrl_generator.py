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


# ============================================================================
# 正确的 4-邻接 BFS 距离场（根治 jaxmarl component_mask_with_pos 的碎图失效 bug）
# ----------------------------------------------------------------------------
# 背景（用户 2026-06-26 抓出，Oscar 实测坐实）：jaxmarl jaxnav_graph_utils.component_mask_with_pos
#   的魔改 DFS（步数上界 2·h·w + n_visited≥3 脏栈回溯启发式）在中高墙密度图（fill≥0.4）上**系统性
#   算错连通块**（误判率 54–74%），把墙另一侧的格误标进起点连通块 → place 端把 goal 放到真实不可
#   达区、valid 仍判 True → 不连通关被放行注入训练（dump inj_00050 #0 即铁证）。
# 修复：用**固定迭代 H·W 次的并行 BFS 松弛**（数学保证收敛：最长最短路 ≤ 自由格数 ≤ H·W），
#   100% 正确、jit+vmap 安全、计算量与坏 DFS 同量级（每迭代 4 次移位+min，11×11 廉价）。
# 一函数根治三处：连通性（place 放 goal + reachable_mask）与测地长度（geodesic）本质同源——
#   皆取自此距离场。坏函数在本文件调用链中**直接移除**，无重复计算。
_BFS_UNREACH = jnp.int32(1_000_000)   # 不可达距离哨兵（远大于任何 11×11 真实最短路）


def bfs_dist_field(wall, src_xy):
    """从 src 出发的 4-邻接 BFS 最短距离场（格步数）。jit+vmap 安全，固定 H·W 次迭代。

    Args:
      wall: (H, W) bool/int，True/1 = 墙（不可通行）。
      src_xy: (2,) int [x, y]，源格（含边界坐标系，与 map_data 同）。
    Returns:
      dist: (H, W) int32，每格到 src 的最短步数；墙格与不可达格 = _BFS_UNREACH。
    """
    wall_b = jnp.asarray(wall).astype(bool)
    H, W = wall_b.shape
    sx, sy = src_xy[0].astype(jnp.int32), src_xy[1].astype(jnp.int32)
    # 初始：源格 0，其余 UNREACH；墙格强制 UNREACH（永不被更新）。
    dist = jnp.full((H, W), _BFS_UNREACH, dtype=jnp.int32)
    src_is_free = ~wall_b[sy, sx]
    dist = dist.at[sy, sx].set(jnp.where(src_is_free, jnp.int32(0), _BFS_UNREACH))

    def _relax(d, _):
        # 4 邻居距离 +1，与自身取 min；墙格钳回 UNREACH。一次迭代把 BFS 波前推进一格。
        # ⚠ jnp.roll 是**环绕**的：必须屏蔽绕到对边的假邻居（generator map 边界**不保证全墙**，
        #   实测 dump #0 边界含自由格 → 不屏蔽会让上下/左右边界假连通、误判 goal 可达，这正是
        #   2026-06-26 接入时单测抓出的 bug）。屏蔽法：把环绕进来的那一行/列置 UNREACH。
        up    = jnp.roll(d, -1, axis=0).at[-1, :].set(_BFS_UNREACH)   # 来自下方；最后一行的邻居绕到第0行→屏蔽
        down  = jnp.roll(d,  1, axis=0).at[0, :].set(_BFS_UNREACH)    # 来自上方；第0行的邻居绕到最后一行→屏蔽
        left  = jnp.roll(d, -1, axis=1).at[:, -1].set(_BFS_UNREACH)   # 来自右方
        right = jnp.roll(d,  1, axis=1).at[:, 0].set(_BFS_UNREACH)    # 来自左方
        neigh = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
        cand = jnp.minimum(d, neigh + 1)
        cand = jnp.where(wall_b, _BFS_UNREACH, cand)
        return cand, None

    dist, _ = jax.lax.scan(_relax, dist, None, length=H * W)
    return dist


def bfs_component_mask(wall, src_xy):
    """src 的连通块 mask（替换坏的 component_mask_with_pos）：dist < UNREACH 的自由格。

    Returns: (H, W) bool，True = 与 src 4-邻接连通的自由格（含 src 自身）。
    """
    dist = bfs_dist_field(wall, src_xy)
    return dist < _BFS_UNREACH


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
            # [根治 2026-06-26] 改用正确的 BFS 连通块（替换碎图失效的 component_mask_with_pos，
            #   见 bfs_component_mask docstring）。返回 (H,W) bool，切内部 (H-2,W-2)，转 int 兼容下游。
            connected = bfs_component_mask(map_data, actual_idx)[1:-1, 1:-1].astype(jnp.int32)
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

    # 初始朝向：连续全方位 uniform(-π,π)，与 baseline DR(jaxmarl grid_map.py:140) 完全对齐。
    # （曾误写成离散 (π/2)·choice(arange(4)) 只 4 个朝向，致 student 学不到连续转向、
    #  测试关 Nav2(θ=180°背朝目标) 全 0；2026-06-25 修复，见 memory generator-theta-discrete-bug。）
    theta = jax.random.uniform(r_th, (), minval=-jnp.pi, maxval=jnp.pi)
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



def terminal_reward_euc(env, network, network_params, env_instances,
                        rng, hidden_size, rollout_steps, num_levels):
    """generator terminal reward —— 起终点欧氏距离 euc(关卡内禀, **不依赖 student**, 不可被堆墙操纵)。

    euc = ||agent_pos - goal_pos||_2。起终点由 place_start_goal_on_map 随机放、非 generator 输出,
    故 generator 摆墙改不了 euc(已核实, STAGE4_phase2 §方向3)。补"长程导航"维度——所有 student-centric
    信号(difficulty/PVL/CENIE)盲视该维度, 而 test-set 最难关恰是起终点远的长程关。
    与 geodesic_from_instances 同法取坐标。

    [护栏·选项一 可学性门，2026-06-25] 纯 euc 不看 student p → generator 堆墙拉远起终点不管
    student 死活 → curriculum-too-hard（injp 实测注入关 p 两极塌缩、可学区 0%）。修复：跑一次
    student rollout 取 per-level p，euc 乘 gate(p)=exp(-((p-0.5)/sigma)²)：p≈0.5 时 gate≈1、
    两极 gate→0，把 reward 双边对准可学区。GEN_LEARNGATE=False → 退化纯 euc（旧行为，不跑 rollout）。
    incomplete level（nep=0）：gate=0（无完整 episode 不知 p，按"不可学"压到 0）。
    Returns: reward (num_levels,) = euc*gate; info dict（含 success_rate 供 per-level p log）。
    """
    pa = env_instances.agent_pos.reshape((num_levels, -1))[:, :2]     # (M,2) [x,y]
    pb = env_instances.goal_pos.reshape((num_levels, -1))[:, :2]
    euc = jnp.sqrt(((pa - pb) ** 2).sum(axis=1) + 1e-8)              # (M,)
    if not GEN_LEARNGATE:
        # 消融/兼容：退回旧纯 euc，不跑 student rollout（最廉价 estimator）。
        return euc, {"euc": euc, "success_rate": jnp.full((num_levels,), jnp.nan),
                     "num_episodes": jnp.zeros((num_levels,), dtype=jnp.int32)}
    # 可学性门：跑一次 student rollout 取 per-level p（口径同 terminal_reward_difficulty）。
    traj = student_rollout_on_levels(
        env, network, network_params, env_instances, rng, hidden_size, rollout_steps,
        num_levels, emit_hidden=False)
    succ, nep = _success_from_traj(traj, rollout_steps, env.num_agents, num_levels)
    sigma = float(GEN_LEARNGATE_SIGMA)
    gate = jnp.exp(-(((succ - 0.5) / sigma) ** 2))                   # (M,) ∈(0,1]，p≈0.5→1
    gate = jnp.where(nep > 0, gate, 0.0)                             # incomplete→0（不可学）
    reward = euc * gate                                              # (M,) 双边对准可学区
    return reward, {"euc": euc, "gate": gate, "success_rate": succ, "num_episodes": nep}


def geodesic_from_instances(env_instances, num_levels):
    """从 EnvInstance batch 算每关的起→终测地最短路长（格步数，jit+vmap 安全）。

    关卡内禀几何量（只读 map_data + agent_pos + goal_pos），**不依赖 student**，故免疫
    "generator 堆墙→student 退化→信号偏堆墙"那个闭环（见 STAGE4 §6.3）。复用 jaxnav 现成
    sfl.util.graph.shortest_path_len（APSP，11×11 廉价）。连续坐标 floor 到格点；map_data
    1=墙→grid true=墙（shortest_path_len 约定 false=free）。不可达返回 1（其内部约定）。

    Returns: (num_levels,) int 测地步数。
    """
    def _one(map_wall, pa_xy, pb_xy):
        # [根治 2026-06-26] 改用正确 BFS 距离场（替换 sfl.util.graph.shortest_path_len，后者内部第一步
        #   即 ~component_mask_with_pos，同受碎图失效 bug 污染）。BFS 不可达返 _BFS_UNREACH 大值
        #   （非残值）——顺带修掉旧"不可达返残值骗过课程/reachable 反推漏判"陷阱（见 geodesic_reachable_mask）。
        pa = jnp.floor(pa_xy).astype(jnp.int32)            # 连续坐标→格点 (x,y)
        pb = jnp.floor(pb_xy).astype(jnp.int32)
        dist = bfs_dist_field(map_wall, pa)                # (H,W) int32；墙/不可达=_BFS_UNREACH
        return dist[pb[1], pb[0]]

    # EnvInstance: agent_pos (M,1,2) / goal_pos (M,1,2) / map_data (M,H,W)。取 agent0。
    pa = env_instances.agent_pos.reshape((num_levels, -1))[:, :2]     # (M,2) [x,y]
    pb = env_instances.goal_pos.reshape((num_levels, -1))[:, :2]
    wall = (env_instances.map_data == 1)                             # (M,H,W) true=wall
    return jax.vmap(_one, in_axes=(0, 0, 0))(wall, pa, pb)           # (M,)


def geodesic_reachable_mask(env_instances, num_levels):
    """每关起终点是否**真可达**（bool）。配 geodesic_from_instances 用于"不可达重罚"。

    ⚠ 陷阱（用户 2026-06-25 抓出，Oscar 实测纠正）：原以为 shortest_path_len 不可达返回 1（小值），
      **实测不可达返回的是残值（如 5），不是 1**——靠 geo 返回值反推可达性不可靠（会漏判）。
      故改用 **component_mask_with_pos 直接判连通**：终点格是否落在起点的连通块里。这是 place_start_goal
      放 goal 时用的同一套真连通逻辑（valid_path_check=True），权威可靠。

    不可达关测地课程会误当"超简单关"放行（让 generator 造割裂图漏网）→ 此 mask 标出不可达关供塑形端重罚。
    Returns: (num_levels,) bool，True=可达。
    """
    def _one(map_data, pa_xy, pb_xy):
        pa = jnp.floor(pa_xy).astype(jnp.int32)              # [x,y] 含边界格点
        pb = jnp.floor(pb_xy).astype(jnp.int32)
        # [根治 2026-06-26] 改用正确 BFS 连通块（替换碎图失效的 component_mask_with_pos）。
        comp = bfs_component_mask(map_data, pa)              # (H,W) bool
        return comp[pb[1], pb[0]]                            # 终点是否在起点连通块

    pa = env_instances.agent_pos.reshape((num_levels, -1))[:, :2]    # (M,2) [x,y]
    pb = env_instances.goal_pos.reshape((num_levels, -1))[:, :2]
    md = env_instances.map_data.astype(jnp.int32)                   # (M,H,W) 0=自由1=墙
    return jax.vmap(_one, in_axes=(0, 0, 0))(md, pa, pb)            # (M,) bool


def _geodesic_anchor_factor(env_instances, num_levels):
    """测地几何锚定因子 = 测地最短路 / 该 batch 最大测地（归一化到 ~[0,1]，防乘法爆量纲）。

    用作 PVL 的乘性锚（方案丙）：堆墙但起终点近的关测地短→因子小→把 PVL 在那压低，
    堵死"堆墙刷瞬时 value-loss"捷径。geo 是关卡内禀几何（不读 student），免疫闭环。
    Returns: (num_levels,) float ∈ ~[0,1]。
    """
    geo = geodesic_from_instances(env_instances, num_levels).astype(jnp.float32)  # (M,)
    # [根治 2026-06-26] BFS 不可达返 _BFS_UNREACH 大值。须排除：否则 geo_max 爆成 1e6 让所有可达关
    #   锚因子≈0。不可达关：①不计入 geo_max；②锚因子强制 0（不可达关 anchored 分=0，不被选中——
    #   与 geodesic_reachable_mask + UNREACHABLE_PENALTY 一致，不可达永远不该当"长程好关"）。
    reachable = geo < (_BFS_UNREACH.astype(jnp.float32))             # (M,) bool
    geo_valid = jnp.where(reachable, geo, 0.0)                       # 不可达置 0 再求 max
    geo_max = jnp.maximum(geo_valid.max(), 1.0)                      # 防除零；只含可达关
    factor = jnp.where(reachable, geo / geo_max, 0.0)               # 不可达锚=0
    return factor, geo_valid


def terminal_reward_anchored(env, network, network_params, env_instances,
                             rng, hidden_size, rollout_steps, num_levels,
                             gamma=0.99, gae_lambda=0.95):
    """generator terminal reward —— 几何锚定 PVL（anchored PVL，方案丙，STAGE4 §6.4，新主力）。

    anchored_pvl = PVL × (geodesic_shortest_path / max_geodesic)
      = "student 价值损失(regret 代理)" × "这关的几何最优代价(归一)"。
      **保留 PVL 的价值误差角度**（含 PVL 量），又乘几何锚把堆墙捷径压下去。

    机制（堵死堆墙捷径，恢复 PAIRED 缺失的可达性参照点）：
      • 堆墙但起终点近的关 → 测地短 → 因子小 → 即使 PVL 高也被压低 → generator **无法靠堆墙刷分**。
      • 中等墙密度但起终点远、student 绕不出来 → 测地大 + PVL 高 → anchored 大 → **自然被选中**。
    离线实测（base/l1 ckpt 各 1000 关）：vs 裸 PVL，ρ(fill) 从 +0.35 降到 +0.004（堆墙偏好消除），
    ρ(euc) 从 +0.24 升到 +0.70（长程信号反增），长程/堆墙近 比值 1.3→4.9，最难关分位 81%→86%。
    优于方案甲 (1-p)×测地（其 ρ(euc) 仅 0.09，强 student 上 (1-p)→0 信息塌缩）。

    incomplete level（无完整 episode）→ -inf（PVL 内部，对齐其它 estimator 契约）。
    Returns: reward (num_levels,)；info dict。
    """
    from sfl.util.jaxued.jaxued_utils import positive_value_loss
    traj = student_rollout_on_levels(
        env, network, network_params, env_instances, rng, hidden_size, rollout_steps,
        num_levels, emit_hidden=False)
    adv = _gae_from_traj(traj, gamma, gae_lambda)                   # (T, BATCH_ACTORS)
    pvl_actor = positive_value_loss(traj["dones"], adv)            # (BATCH_ACTORS,)；incomplete→-inf
    pvl = pvl_actor.reshape((env.num_agents, num_levels)).mean(axis=0)   # (M,)
    anchor, geo = _geodesic_anchor_factor(env_instances, num_levels)    # (M,) ∈~[0,1]
    anchored = pvl * anchor                                        # (M,)；incomplete 的 -inf 透传
    return anchored, {"pvl": pvl, "geodesic": geo, "anchor_factor": anchor}


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
                          gamma=0.99, gae_lambda=0.95, probe_hidden_steps=16,
                          signal_mode="anchored", prev_bucket_p=None,
                          alp_num_buckets=6, alp_euc_lo=0.0, alp_euc_hi=16.0,
                          gate_w_hard=2.0, gate_w_easy=1.0):
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

    # 第二维信号按 signal_mode 选 (STAGE4_phase2 §方向1/3):
    #   'pvl'      → 裸 PVL (=第一版注入配置 s4p2-lambda-1_0, 难关CVaR 0.655, 纯方向1扫描基底);
    #   'anchored' → anchored-PVL = PVL×测地锚 (已证更差0.630且被"堆墙+绕路"组合捷径操纵, 已弃);
    #   'euc'      → 不在此算 (euc 是几何量, 不依赖 rollout, 见下方 per_est 组装)。
    # [诊断仪表盘 2026-06-27] per-level 裸 PVL / mean|adv| 全过程记录（与 gen_p_perlevel 同口径）。
    #   裸 PVL 在 line 778 会被 anchored 覆盖，故此处先存 pvl_raw_perlevel；absadv 顺路从同一 traj 算，
    #   零额外 rollout。仅 adv 分支(anchored/pvl/...)有定义；euc 等无 rollout-adv 路径置 None。
    _pvl_raw_perlevel = None
    _absadv_perlevel = None
    if signal_mode in ("anchored", "pvl", "anchored_cenie", "anchored_cenie_gate",
                       "absadv_anchored_cenie"):
        adv = _gae_from_traj(traj, gamma, gae_lambda)
        pvl_actor = positive_value_loss(traj["dones"], adv)
        pvl = pvl_actor.reshape((env.num_agents, num_levels)).mean(axis=0)   # 裸 PVL
        _pvl_raw_perlevel = pvl                                          # 存裸 PVL（覆盖前），incomplete→-inf
        # per-level mean|adv|：(T, BATCH_ACTORS)→(T, num_agents, num_levels) 在 T 和 agent 维取均值。
        _absadv_perlevel = jnp.abs(adv).reshape(
            (rollout_steps, env.num_agents, num_levels)).mean(axis=(0, 1))   # (num_levels,)
        # [absadv_anchored_cenie 2026-06-27] |adv| 攻 p≈0 区 + anchored 攻 p≈1 区两侧夹击(§5.7)。
        #   |adv| 是梯度幅度，在 p≈0 桶分辨率最高(std 最大)，覆盖 p(1-p)+PVL 双盲的 p≈0 区。
        #   incomplete level(nep==0)的 |adv| 仍是有限值(rollout 跑了但没 episode)，须显式置 -inf 对齐契约。
        _absadv_bid = jnp.where(nep > 0, _absadv_perlevel, -jnp.inf)    # (M,) bid 用，incomplete→-inf
        if signal_mode in ("anchored", "anchored_cenie", "anchored_cenie_gate",
                          "absadv_anchored_cenie"):
            _anchor, _geo = _geodesic_anchor_factor(env_instances, num_levels)   # (M,) ∈~[0,1]
            pvl = pvl * _anchor                                          # ← anchored PVL (乘测地锚)

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

    _cur_bucket_p = None   # ALP 路径才算；非 ALP 路径返回 None（host 层据此决定是否更新状态）
    _alp_quota_perlevel = None   # 配额模式才算 per-level ALP bonus；其余路径 None
    _gate_perlevel = None        # 单边可学性 gate 修正项（host 层加性叠加）；其余路径 None
    if signal_mode in ("euc", "euc_alp", "euc_cenie_alpquota"):
        # 方向3/ALP: euc=起终点欧氏距离(内禀几何,不依赖student,不可被堆墙操纵)。
        #   起终点由 place_start_goal_on_map 随机放、非generator输出 → generator摆墙改不了euc。
        pa = env_instances.agent_pos.reshape((num_levels, -1))[:, :2]   # (M,2)
        pb = env_instances.goal_pos.reshape((num_levels, -1))[:, :2]
        euc = jnp.sqrt(((pa - pb) ** 2).sum(axis=1) + 1e-8)            # (M,)
    if signal_mode == "euc_alp":
        # [euc, ALP, cenie]：ALP 顿替 difficulty（ρ(diff,ALP)=0.91 同信号，见 ALP_design_phase2.md）。
        # ALP=per-level euc 桶配对进步率：|cur_bucket_p[bucket] - prev_bucket_p[bucket]|。
        # 冷启（prev_bucket_p=None）→ 退化为全哨兵 → ALP 全 0（首 epoch 不竞价，对齐 cenie 冷启）。
        from estimators import alp_by_euc_bucket, ALP_BUCKET_SENTINEL
        _prev = prev_bucket_p
        if _prev is None:
            _prev = jnp.full((alp_num_buckets,), ALP_BUCKET_SENTINEL, dtype=jnp.float32)
        alp, _cur_bucket_p = alp_by_euc_bucket(
            succ, nep, euc, _prev, alp_num_buckets,
            euc_lo=alp_euc_lo, euc_hi=alp_euc_hi)                       # (M,), (B,)
        per_est = jnp.stack([euc, alp, cenie], axis=0)                 # (3, M): [euc, ALP, cenie]
    elif signal_mode == "euc_cenie_alpquota":
        # 配额调节器（modulation 层）：auction 只 [euc, cenie] 选关（2 维），ALP 作 per-level
        #   加性 bonus（host 层叠加，见 get_generator_set），按 euc 桶调配额，不进 bid 竞争。
        #   ALP 仍用 alp_by_euc_bucket 算（per-level，桶 broadcast），从 _ainfo["alp_perlevel"] 传出。
        from estimators import alp_by_euc_bucket, ALP_BUCKET_SENTINEL
        _prev = prev_bucket_p
        if _prev is None:
            _prev = jnp.full((alp_num_buckets,), ALP_BUCKET_SENTINEL, dtype=jnp.float32)
        _alp_pl, _cur_bucket_p = alp_by_euc_bucket(
            succ, nep, euc, _prev, alp_num_buckets,
            euc_lo=alp_euc_lo, euc_hi=alp_euc_hi)                       # (M,), (B,)
        per_est = jnp.stack([euc, cenie], axis=0)                      # (2, M): [euc, cenie]（不含 ALP）
        _alp_quota_perlevel = _alp_pl                                  # host 层叠加 bonus
    elif signal_mode == "euc":
        # 方向3: [euc, difficulty, cenie]。
        # euc 是 finite 的(几何恒有定义); incomplete level 仍由 difficulty/cenie 的 -inf 标记。
        per_est = jnp.stack([euc, difficulty, cenie], axis=0)          # (3, M): [euc, difficulty, cenie]
    elif signal_mode == "anchored_cenie":
        # [结构课程并行 Run B, 2026-06-25] 去 difficulty 的 2 维 auction：[anchored-PVL, cenie]。
        #   加课程后 difficulty 与课程难度轴冗余/打架(都用 student p/难度轴)→删它,只留 anchored(造什么样
        #   的难,与课程测地协同) + cenie(探索,与难度轴正交)。pvl 此处已乘测地锚(上方 if 分支)。
        per_est = jnp.stack([pvl, cenie], axis=0)                      # (2, M): [anchored, cenie]
    elif signal_mode == "absadv_anchored_cenie":
        # [两侧夹击 2026-06-27, §5.7] 3 维 auction：[|adv|, anchored-PVL, cenie]。
        #   动机(§5.6 信号区分度实验): p(1-p) 双零盲区(p≈0/p≈1 都给0)。anchored-PVL 偏向 p≈1 侧
        #   "快会"关(Spearman(PVL,1-p)=-0.68); |adv| 在 p≈0 桶分辨率最高(std 1.516>PVL 0.845),
        #   补 p≈0 侧高梯度真难关 → 两侧夹击覆盖盲区。cenie 给结构多样性(正交)。
        #   z-score 抹平三者量级悬殊(|adv|~O(1) vs cenie~O(100))，话语权由 bid top-k 尖度+λ 定，
        #   非绝对量级(auction_bid.standardize_per_estimator)。pvl 已乘测地锚(上方 if 分支)。
        per_est = jnp.stack([_absadv_bid, pvl, cenie], axis=0)         # (3, M): [|adv|, anchored, cenie]
    elif signal_mode == "anchored_cenie_gate":
        # [双向可学性 gate 修正, 2026-06-26 改] auction 仍 [anchored-PVL, cenie] 2 维选关（gate 不进
        #   竞价），gate 作 per-level 加性修正项走 host 层（仿 euc_cenie_alpquota 的 alp_quota 路径）。
        #   双向 gate（两个独立单边项，避开 -(p-0.5)² 对称无梯度坑）：
        #     gate = -W_HARD·relu(0.5-p) - W_EASY·relu(p-0.5)
        #   W_HARD 压太难(p<0.5)，W_EASY 压太简单(p>0.5)。p=0→-0.5·W_HARD，p=1→-0.5·W_EASY，
        #   p=0.5→0。W_HARD≠W_EASY 时两极拿不同值故 z(gate) 不塌成常数、保留梯度。
        #   incomplete（nep==0）置 0 不参与修正（避免污染）。host 层 final=z(bid)+GATE_WEIGHT·z(gate)：
        #   W_HARD/W_EASY 定两侧相对形状（z 抹绝对量级，只剩两侧比值），GATE_WEIGHT 定总幅度。
        #   pvl 已乘测地锚（上方 if 分支）。
        gate = (-gate_w_hard * jnp.maximum(0.0, 0.5 - succ)
                - gate_w_easy * jnp.maximum(0.0, succ - 0.5))          # (M,) p<0.5罚太难, p>0.5罚太简单
        gate = jnp.where(nep > 0, gate, 0.0)                           # incomplete 不参与 gate
        per_est = jnp.stack([pvl, cenie], axis=0)                      # (2, M): [anchored, cenie]（不含 gate）
        _gate_perlevel = gate                                         # host 层叠加修正项
    else:
        # 'anchored' (旧主线): [difficulty, anchored-PVL, cenie];
        # 'pvl' (纯方向1): [difficulty, 裸PVL, cenie] (=第一版注入, difficulty 是 index 0)。
        per_est = jnp.stack([difficulty, pvl, cenie], axis=0)          # (3, M)
    return per_est, {"success_rate": succ, "num_episodes": nep,
                     "cur_bucket_p": _cur_bucket_p,
                     "alp_quota_perlevel": _alp_quota_perlevel,
                     "gate_perlevel": _gate_perlevel,
                     "pvl_raw_perlevel": _pvl_raw_perlevel,
                     "absadv_perlevel": _absadv_perlevel}


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
GEN_ESTIMATOR_ANCHORED = "anchored"       # (1-p)×测地最短路，参照锚定 regret（STAGE4 §6.4，堵堆墙捷径）
GEN_ESTIMATOR_EUC = "euc"                 # 起终点欧氏距离, 内禀几何不依赖student(STAGE4_phase2 §方向3)
# [卡死修复·C] invalid 图（放不下合法起终点）的固定强负 terminal reward（**只作用于 A 训练端的
#   generator PPO 梯度**；注入端另有 -inf 硬挡，见 get_generator_set）。原 -1.0：difficulty=-(p-0.5)²∈
#   [-0.25,0] 时够低；但 auction 路径下 terminal 已 z-score 到 ~O(1)，-1.0 会落进分布中段、压不出
#   "少造 invalid"的清晰梯度。2026-06-26 加强到 -5.0（与 CURRICULUM_UNREACHABLE_PENALTY 同量级），
#   更强地惩罚 invalid 产出、减轻注入端"合法关不够被迫凑 -inf 图"的压力；-5 仍不至于炸 PPO。
GEN_INVALID_PENALTY = -5.0

# [护栏·选项一 可学性门调制，STAGE4_phase2 §injp 决定性，2026-06-25]
#   injp 实测：generator 注入关 p 两极塌缩、可学区 0%（前期 73% p≈0 必死）。真因 = terminal_reward_euc
#   纯几何不看 student p，generator 堆墙拉远起终点不管 student 死活 → curriculum-too-hard。
#   修复：euc 乘可学性门 gate(p)=exp(-((p-0.5)/sigma)²)，p≈0.5 时 gate≈1、两极 gate→0，
#   把 reward 双边对准可学区（既压"太难"也压"太简单"），推 generator 造 student 学得动的关。
#   GEN_LEARNGATE=False → gate≡1，退化回纯 euc（消融/兼容旧行为）。sigma 越小门越窄越锁 p=0.5。
GEN_LEARNGATE = True       # 可学性门总开关；False 时 gate≡1（=旧纯 euc，消融对照）
GEN_LEARNGATE_SIGMA = 0.25 # 门宽：p 偏离 0.5 达 sigma 时 gate≈0.37；0.25 → 可学区 ~[0.25,0.75]


# ============================================================================
# 结构课程（structure curriculum，STAGE4_phase2 §结构课程方案定稿，2026-06-25）
# ----------------------------------------------------------------------------
# 用户洞察：generator 输 baseline 的真因 = 课程难度与 student 能力不匹配（generator 塌到
# 单一难度档=全悬崖）。题眼 = 让难度始终贴 student 能力前沿（先易后难匹配），非 p、非堆墙。
#
# 主轴 = 测地绝对长度（geodesic_from_instances，已有；start→goal 最短路格数，关卡内禀几何、
#   **不依赖 student 瞬时态**，故不会像 p/PVL 在 student 学会后归零失向）。用户看 baseline
#   step100 图抓出 B 类关（高墙但起终点近=简单）→ 证伪 fill（高 fill≠难），故 fill 仅作反例对照。
#
# 事前引导（非事后过滤）：在 generator 的 terminal reward 上加塑形项 -β·relu(struct-thr(t))，
#   罚的是驱动生成策略本身的 reward（非 buffer 注入端删关）。超档关 reward 被压低 → generator
#   PPO 梯度推离超档 → 收敛后几乎不再生成超档关（最接近"事前硬引导"的可实现形式）。
#
# 固定时间表：thr(t) 随训练进度分段抬升（host 侧按 eval_step 算好标量穿 jit，与 gmm_params 同模式）。
#   标定档（测地格数，看 mazes/ 真实图，2026-06-25）：thr_easy=5 / thr_mid=10 / thr_hard=22。
# ============================================================================
CURRICULUM_ARM_NONE = "none"   # 关闭塑形（= 旧行为，对照 arm）
CURRICULUM_ARM_GEO = "geo"     # 主轴：测地绝对长度（geodesic_from_instances）
CURRICULUM_ARM_FILL = "fill"   # 反例对照：墙占比（预期因误杀 B 类高墙近距关而表现差）
# 默认时间表三档阈值（测地格数；fill arm 时调用方须改传 fill 量纲阈值，见 jaxnav_sfl config）。
CURRICULUM_THR_EASY = 5.0
CURRICULUM_THR_MID = 10.0
CURRICULUM_THR_HARD = 22.0
CURRICULUM_BETA = 1.0          # 塑形强度；β 越大越接近硬引导（扫 {1,3,10} 定多硬才够推动生成分布）。
# [不可达重罚，用户 2026-06-25] 测地课程压制"造割裂图"的生成倾向：起终点真不可达(geodesic_reachable_mask
#   =False)的关，除走 place_start_goal 的事后剔除外，再在塑形端给固定强负罚，让 generator PPO 从生成
#   策略层学到"造割裂图=重亏"。量级须显著负于任何 valid 关的 base+塑形，又不炸 PPO。
CURRICULUM_UNREACHABLE_PENALTY = -5.0


def fill_ratio_of_map(jaxnav_map):
    """墙占比 = 墙格数 / 总格数（含边界）。纯逐元素，jit 安全。

    Args:
      jaxnav_map: (H, W) int 占用栅格（0=自由 1=墙；map_to_env_instance 出的 map_data 口径）。
    Returns:
      标量 float ∈[0,1]。
    反例对照轴（arm-fill）：用户 B 类关证伪它当难度轴（高墙近距关 fill 高却简单）。
    """
    return (jaxnav_map == 1).astype(jnp.float32).mean()


def fill_ratio_from_instances(env_instances, num_levels):
    """batch EnvInstance → 每关 fill ratio（vmap fill_ratio_of_map）。

    Returns: (num_levels,) float ∈[0,1]。
    """
    walls = (env_instances.map_data == 1).astype(jnp.float32)        # (M,H,W)
    return walls.reshape((num_levels, -1)).mean(axis=1)              # (M,)


def curriculum_threshold(progress, thr_easy=CURRICULUM_THR_EASY,
                         thr_mid=CURRICULUM_THR_MID, thr_hard=CURRICULUM_THR_HARD):
    """固定时间表：训练进度 progress∈[0,1] → 当前难度上限 thr(t)（分段常量）。

    分三段（对齐 STAGE4_phase2 标定）：
      progress ∈ [0,   1/3):  thr_easy   早期只许简单结构（短测地），打牢基础
      progress ∈ [1/3, 2/3):  thr_mid    中期放开到对角线量级
      progress ∈ [2/3, 1.0]:  thr_hard   后期解锁到目标关（Nav2/3）难度
    progress 由 host 侧 eval_step/(总 eval 步) 算出（标量），穿 jit 边界。

    Args:
      progress: 标量 ∈[0,1]（训练进度）。可为 Python float 或 traced 标量。
    Returns:
      thr: 标量 float（当前难度上限）。
    """
    p = jnp.asarray(progress, dtype=jnp.float32)
    return jnp.where(p < (1.0 / 3.0), thr_easy,
                     jnp.where(p < (2.0 / 3.0), thr_mid, thr_hard))


def curriculum_shaping_penalty(struct_value, threshold, beta=CURRICULUM_BETA,
                               thr_lo=0.0, beta_lo=0.0):
    """事前引导塑形项（双边）= -β·relu(struct-thr_hi) - β_lo·relu(thr_lo-struct)。

    [§5.9 双边窗口 2026-06-28] 原单边只罚"太难"(上界)，geo<thr_hi 一律零罚 → generator 可躲
      简单区躺平(seed0 病根)。加下界项罚"太简单" → 把生成分布逼进窗口 [thr_lo, thr_hi]。
      thr_lo=0/beta_lo=0 时退化为原单边(零回归)。easy 档 thr_lo=0 不设下界，保"先易后难"。

    叠加到 terminal reward 上：超档/不及档关被压低 → generator PPO 学到造关须落窗口内。
    geo 是关卡内禀几何(不读 student)，下界罚不可被 student 行为操纵，reward-hack 风险低。

    Args:
      struct_value: (M,) 各关的结构难度（测地长度 or fill，按 arm）。
      threshold: 标量，当前难度上限 thr_hi(t)。
      beta: 上界塑形强度。
      thr_lo: 标量，当前难度下界 thr_lo(t)（0=不设下界）。
      beta_lo: 下界塑形强度（0=关闭下界，退化为单边）。
    Returns:
      penalty: (M,) ≤0 的塑形项（落窗口内=0）。
    """
    too_hard = -beta * jnp.maximum(struct_value - threshold, 0.0)
    too_easy = -beta_lo * jnp.maximum(thr_lo - struct_value, 0.0)
    return too_hard + too_easy


def struct_value_from_instances(env_instances, num_levels, arm):
    """按 curriculum arm 算每关的结构难度（测地长度 or fill ratio）。

    Args:
      arm: CURRICULUM_ARM_GEO（测地绝对长度，主轴）/ CURRICULUM_ARM_FILL（墙占比，反例对照）。
           静态字符串（jit 下走静态分支）。
    Returns:
      (num_levels,) float。CURRICULUM_ARM_NONE 不应进此函数（调用方应跳过塑形）。
    """
    if arm == CURRICULUM_ARM_GEO:
        return geodesic_from_instances(env_instances, num_levels).astype(jnp.float32)
    elif arm == CURRICULUM_ARM_FILL:
        return fill_ratio_from_instances(env_instances, num_levels)
    else:
        raise ValueError(f"未知 curriculum arm={arm}（应为 "
                         f"{CURRICULUM_ARM_GEO}/{CURRICULUM_ARM_FILL}）")


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
    elif estimator_id == GEN_ESTIMATOR_ANCHORED:
        return terminal_reward_anchored(
            student_env, student_network, student_params, insts, rng,
            hidden_size, student_rollout_steps, num_levels)
    elif estimator_id == GEN_ESTIMATOR_EUC:
        return terminal_reward_euc(
            student_env, student_network, student_params, insts, rng,
            hidden_size, student_rollout_steps, num_levels)
    else:
        raise ValueError(f"未知 estimator_id={estimator_id}（应为 "
                         f"{GEN_ESTIMATOR_DIFFICULTY}/{GEN_ESTIMATOR_PVL}/"
                         f"{GEN_ESTIMATOR_CENIE}/{GEN_ESTIMATOR_ANCHORED}/{GEN_ESTIMATOR_EUC}）")


def gen_train_iter(env, network, params, opt_state, optimizer, rng,
                   student_env, student_network, student_params,
                   hidden_size, student_rollout_steps, num_levels,
                   estimator_id=GEN_ESTIMATOR_DIFFICULTY, gmm_params=None,
                   place_env=None,
                   curriculum_arm=CURRICULUM_ARM_NONE, curriculum_thr=None,
                   curriculum_beta=CURRICULUM_BETA,
                   curriculum_thr_lo=0.0, curriculum_beta_lo=0.0,
                   ppo_epochs=4, gamma=0.99, gae_lambda=0.95,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """generator 一次完整 PPO 迭代（单 generator，terminal reward 由 estimator_id 决定）。

    rollout（造 map + 收轨迹）→ 喂 student 算 terminal reward（difficulty/PVL/CENIE）→
    [结构课程] 加塑形项 -β·relu(struct-thr) → 回填 → GAE → ppo_epochs 次更新。

    Args:
      estimator_id: 该 generator 的信号源（GEN_ESTIMATOR_*，静态）。CENIE 需 gmm_params。
      gmm_params: CENIE 用的已拟合 GMM（None/valid=False 时 CENIE 返 0）。
      place_env: 放起终点用 jaxnav env（应 valid_path_check=True，§R5）；None 退回 student_env。
                 rollout 仍在 student_env 上（与训练同分布）。
      curriculum_arm: 结构课程难度轴（CURRICULUM_ARM_NONE/GEO/FILL，静态字符串）。
                 NONE=关闭塑形（旧行为）；GEO=测地长度主轴；FILL=墙占比反例对照。
      curriculum_thr: 当前难度上限 thr(t)（host 侧按训练进度算好的标量；arm≠NONE 时必传）。
      curriculum_beta: 塑形强度 β（越大越接近硬引导）。
    Returns:
      params, opt_state, metrics(dict: total/actor/value/entropy/mean_reward + struct 统计)。
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

    # [结构课程·事前引导] 塑形项 -β·relu(struct-thr(t))：超档关 reward 被压低 → generator PPO
    #   学到"造超档=亏" → 生成分布被推向当前难度档（先易后难匹配）。只罚 valid 关（invalid 已
    #   是固定 -1.0，不再叠塑形，否则双重惩罚污染梯度）。arm=NONE 时跳过（=旧行为，对照）。
    # [不可达重罚，用户 2026-06-25] geo arm 下不可达关测地返回 1（小值）会被误当达标关放行；
    #   geodesic_reachable_mask 显式标出 → 给 CURRICULUM_UNREACHABLE_PENALTY 重罚，压制造割裂图。
    struct_mean = jnp.float32(0.0)
    frac_over = jnp.float32(0.0)
    frac_under = jnp.float32(0.0)
    frac_unreach = jnp.float32(0.0)
    if curriculum_arm != CURRICULUM_ARM_NONE:
        struct_val = struct_value_from_instances(insts, num_levels, curriculum_arm)  # (M,)
        penalty = curriculum_shaping_penalty(struct_val, curriculum_thr, curriculum_beta,
                                             thr_lo=curriculum_thr_lo, beta_lo=curriculum_beta_lo)  # (M,)≤0
        terminal_rew = jnp.where(valids, terminal_rew + penalty, terminal_rew)
        # 不可达重罚（geo arm 主治割裂图；fill arm 也复用此安全网，不可达永远该罚）。
        reachable = geodesic_reachable_mask(insts, num_levels)        # (M,) bool
        unreach_valid = valids & (~reachable)        # valid(放下了起终点)但起终点不可达=割裂图漏网
        terminal_rew = jnp.where(unreach_valid, CURRICULUM_UNREACHABLE_PENALTY, terminal_rew)
        # struct 统计排除不可达关（其 geo=1 是假值，留下会把均值拉低、误导仪表盘）。
        _reach_f = reachable.astype(jnp.float32)
        _cnt = jnp.maximum(_reach_f.sum(), 1.0)
        struct_mean = (struct_val * _reach_f).sum() / _cnt
        frac_unreach = (valids & (~reachable)).astype(jnp.float32).mean()
        frac_over = (struct_val > curriculum_thr).astype(jnp.float32).mean()
        # [§5.9 下界仪表盘] 不及下界(太简单)关比例。下界生效则应随训练→0(generator 学会不造太简单)。
        frac_under = ((struct_val < curriculum_thr_lo) & reachable).astype(jnp.float32).mean()

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
        # [结构课程仪表盘] 训练侧 struct 统计（arm=NONE 时恒 0）。注入侧的分布指标在 get_generator_set。
        "gen_struct_value_mean": struct_mean,    # 该 gen 造关测地/fill 均值（已排除不可达关）
        "gen_frac_over_threshold": frac_over,    # 超档关比例（应随训练→0 证明事前引导生效）
        "gen_frac_under_lo": frac_under,         # [§5.9] 不及下界(太简单)关比例（下界生效应→0）
        "gen_frac_unreachable": frac_unreach,    # 不可达关(割裂图)比例（重罚应使其随训练→0）
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
                         max_grad_norm=0.5, alp_num_buckets=6):
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
    # ALP（euc 桶配对）跨 epoch 状态：每桶上一 epoch 平均 p（哨兵 -1=无历史）。纯数组，穿 jit。
    from estimators import ALP_BUCKET_SENTINEL
    prev_bucket_p = jnp.full((alp_num_buckets,), ALP_BUCKET_SENTINEL, dtype=jnp.float32)
    gen_state = {
        "params": params_list,
        "opt_state": opt_list,
        "prev_bucket_p": prev_bucket_p,
    }
    return gen_state, optimizer


# ============================================================================
# [大基数分批 / OOM 修复 2026-06-26] chunked 造关 + 分批测信号
# ----------------------------------------------------------------------------
# 背景：generator 一次性 vmap 造 num_levels_per_gen 张 level（gen_rollout_batch），
#   PCGRL 造关 CNN 在 pool=1000/2000 时 RESOURCE_EXHAUSTED（f32[N×181,64,16,16]，
#   23.8GB）。baseline 海选 5000 关不 OOM 是因它分 5 批（NUM_BATCHES×BATCH_SIZE）。
#   本块仿 baseline 把"造关 + 测信号"按 GEN_ROLLOUT_CHUNK 分批，峰值显存 = 单 chunk 而非全 pool。
#
# 设计原则（零回归）：
#   - chunk<=0 或 chunk>=n_levels → 调用方走原一次性路径，逐字等价（不进本块）。
#   - chunk>0 且 <n_levels → Python for 展开 ceil(n/chunk) 批（chunk 数是编译期常量，
#     在 gen_phase_jit trace 期静态展开），每批独立 sub-rng（从入参 rng 确定性 split），
#     concat 回完整形状。RNG 从 r_roll/r_score split → same config+seed 可复现。
#   - 拼回的 insts/valids/per_est/info 形状与一次性路径完全一致，下游 auction/top-K 不动。
#
# ⚠ ALP 限制：alp_by_euc_bucket 的 cur_bucket_p 是**全 pool 聚合**（按桶平均 p），
#   分批会改变桶统计语义。故 ALP signal_mode（euc_alp / euc_cenie_alpquota）下分批不安全，
#   chunk_all_signals 会 raise（调用方应让这些模式走一次性路径）。目标 bigpool 跑 anchored
#   模式（[difficulty,anchored,cenie] 全 per-level 独立），分批数值精确等价。
# ============================================================================
def _n_chunks(n_levels, chunk):
    """ceil(n_levels/chunk)，编译期常量（n_levels、chunk 均为 Python int）。"""
    return (n_levels + chunk - 1) // chunk


def _chunk_bounds(n_levels, chunk):
    """生成 [(lo, hi), ...] 切片边界（Python list，trace 期静态）。末批可能不足 chunk。"""
    return [(i * chunk, min((i + 1) * chunk, n_levels))
            for i in range(_n_chunks(n_levels, chunk))]


def chunked_gen_rollout_and_bridge(gen_env, gen_network, params, jaxnav_env,
                                   r_roll, r_inst, n_levels, chunk):
    """分批造关 + 桥接成 EnvInstance，concat 回 (n_levels,...)。峰值显存 = 单 chunk。

    每批用从 r_roll/r_inst 确定性 split 的独立 sub-rng（批数=ceil(n/chunk) 静态）。
    与一次性 gen_rollout_batch + gen_batch_to_env_instances 形状/语义等价（仅分批跑 CNN）。

    Returns:
      insts:  EnvInstance（batch 维 n_levels）。
      valids: (n_levels,) bool。
    """
    bounds = _chunk_bounds(n_levels, chunk)
    nc = len(bounds)
    roll_rngs = jax.random.split(r_roll, nc)
    inst_rngs = jax.random.split(r_inst, nc)
    insts_parts, valids_parts = [], []
    for ci, (lo, hi) in enumerate(bounds):
        sz = hi - lo
        env_maps_c = gen_rollout_batch(gen_env, gen_network, params, roll_rngs[ci], sz)
        insts_c, valids_c = gen_batch_to_env_instances(env_maps_c, jaxnav_env, inst_rngs[ci])
        insts_parts.append(insts_c)
        valids_parts.append(valids_c)
    if nc == 1:
        return insts_parts[0], valids_parts[0]
    insts = jax.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *insts_parts)
    valids = jnp.concatenate(valids_parts, axis=0)
    return insts, valids


def chunked_compute_terminal_reward(estimator_id, student_env, student_network,
                                    student_params, insts, r_score, hidden_size,
                                    student_rollout_steps, n_levels, chunk,
                                    gmm_params=None, gamma=0.99, gae_lambda=0.95):
    """分批 student rollout 测 fallback 信号（compute_terminal_reward），concat 回 (n_levels,).

    每批独立 sub-rng（从 r_score split）。incomplete level 的 -inf 标记原样保留。
    与一次性 compute_terminal_reward 形状/语义等价。
    Returns: scores (n_levels,)，info dict（success_rate (n_levels,) concat 回来）。
    """
    bounds = _chunk_bounds(n_levels, chunk)
    nc = len(bounds)
    score_rngs = jax.random.split(r_score, nc)
    scores_parts, succ_parts = [], []
    for ci, (lo, hi) in enumerate(bounds):
        sz = hi - lo
        insts_c = jax.tree_map(lambda x, lo=lo, hi=hi: x[lo:hi], insts)
        sc_c, info_c = compute_terminal_reward(
            estimator_id, student_env, student_network, student_params, insts_c,
            score_rngs[ci], hidden_size, student_rollout_steps, sz,
            gmm_params=gmm_params, gamma=gamma, gae_lambda=gae_lambda)
        scores_parts.append(sc_c)
        succ_parts.append(info_c.get("success_rate", jnp.full((sz,), jnp.nan)))
    if nc == 1:
        return scores_parts[0], {"success_rate": succ_parts[0]}
    scores = jnp.concatenate(scores_parts, axis=0)
    succ = jnp.concatenate(succ_parts, axis=0)
    return scores, {"success_rate": succ}


def chunked_all_signals_on_levels(student_env, student_network, student_params,
                                  cat_insts, r_auc, hidden_size, student_rollout_steps,
                                  M, chunk, gmm_params=None, gamma=0.99, gae_lambda=0.95,
                                  signal_mode="anchored", prev_bucket_p=None,
                                  gate_w_hard=2.0, gate_w_easy=1.0):
    """分批 student rollout 测 auction 三信号（all_signals_on_levels），concat 回 (K, M).

    每批独立 sub-rng（从 r_auc split）。per_est 沿 axis=1 concat；info 的 per-level 数组
    （success_rate/gate_perlevel/alp_quota_perlevel）concat 回 (M,)。

    ⚠ ALP 模式（cur_bucket_p 是全 pool 聚合）分批语义不等价 → 这些模式禁止分批（raise）。
       目标 bigpool 用 anchored（per-level 独立），分批精确等价。
    """
    if signal_mode in ("euc_alp", "euc_cenie_alpquota"):
        raise ValueError(
            f"chunked signal path 不支持 ALP signal_mode={signal_mode}"
            "（cur_bucket_p 全 pool 聚合，分批改语义）；这些模式请设 GEN_ROLLOUT_CHUNK=0 走一次性路径。")
    bounds = _chunk_bounds(M, chunk)
    nc = len(bounds)
    auc_rngs = jax.random.split(r_auc, nc)
    per_est_parts, succ_parts, gate_parts = [], [], []
    # [诊断仪表盘 2026-06-27 修] chunked 路径也必须 concat absadv/pvl per-level，否则 _ainfo 缺这两 key
    #   → get_generator_set 取不到 → gen_absadv_mean 不落 → |adv| 课程判据读 NaN 死锁档0（实测 bug）。
    absadv_parts, pvl_parts = [], []
    have_absadv = False; have_pvl = False
    have_gate = False
    for ci, (lo, hi) in enumerate(bounds):
        sz = hi - lo
        insts_c = jax.tree_map(lambda x, lo=lo, hi=hi: x[lo:hi], cat_insts)
        per_est_c, info_c = all_signals_on_levels(
            student_env, student_network, student_params, insts_c, auc_rngs[ci],
            hidden_size, student_rollout_steps, sz, gmm_params=gmm_params,
            gamma=gamma, gae_lambda=gae_lambda, signal_mode=signal_mode,
            prev_bucket_p=prev_bucket_p, gate_w_hard=gate_w_hard, gate_w_easy=gate_w_easy)
        per_est_parts.append(per_est_c)                  # (K, sz)
        succ_parts.append(info_c.get("success_rate", jnp.full((sz,), jnp.nan)))
        _g = info_c.get("gate_perlevel", None)
        if _g is not None:
            have_gate = True
            gate_parts.append(_g)
        else:
            gate_parts.append(jnp.zeros((sz,)))
        _av = info_c.get("absadv_perlevel", None)
        if _av is not None:
            have_absadv = True; absadv_parts.append(_av)
        else:
            absadv_parts.append(jnp.full((sz,), jnp.nan))
        _pv = info_c.get("pvl_raw_perlevel", None)
        if _pv is not None:
            have_pvl = True; pvl_parts.append(_pv)
        else:
            pvl_parts.append(jnp.full((sz,), jnp.nan))
    if nc == 1:
        return per_est_parts[0], {
            "success_rate": succ_parts[0],
            "cur_bucket_p": None,
            "alp_quota_perlevel": None,
            "gate_perlevel": (gate_parts[0] if have_gate else None),
            "absadv_perlevel": (absadv_parts[0] if have_absadv else None),
            "pvl_raw_perlevel": (pvl_parts[0] if have_pvl else None),
        }
    per_est = jnp.concatenate(per_est_parts, axis=1)     # (K, M)
    succ = jnp.concatenate(succ_parts, axis=0)           # (M,)
    info = {
        "success_rate": succ,
        "cur_bucket_p": None,            # ALP 模式已被 raise 拦截，非 ALP 路径恒 None
        "alp_quota_perlevel": None,
        "gate_perlevel": (jnp.concatenate(gate_parts, axis=0) if have_gate else None),
        "absadv_perlevel": (jnp.concatenate(absadv_parts, axis=0) if have_absadv else None),
        "pvl_raw_perlevel": (jnp.concatenate(pvl_parts, axis=0) if have_pvl else None),
    }
    return per_est, info


def get_generator_set(rng, student_params, gen_state, optimizer, estimator_ids,
                      gen_env, gen_network, student_env, student_network,
                      hidden_size, student_rollout_steps,
                      num_levels_per_gen, num_to_save, place_env=None,
                      gmm_params=None, auction_lambda=None,
                      signal_mode="anchored", auction_weight_factors=None,
                      alp_quota_coef=0.0, gate_weight=0.0,
                      gate_w_hard=2.0, gate_w_easy=1.0,
                      curriculum_arm=CURRICULUM_ARM_NONE, curriculum_thr=None,
                      curriculum_beta=CURRICULUM_BETA,
                      curriculum_thr_lo=0.0, curriculum_beta_lo=0.0,
                      gen_outer_steps=1, ppo_epochs=4,
                      gen_rollout_chunk=0, pool_per_gen=0,
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
      num_levels_per_gen: 每个 generator **PPO 训练**用 level 数（gen_train_iter，PPO backprop
                 持全轨迹 → 显存上限低，保持小如 64）。
      pool_per_gen: 每个 generator **注入候选池** level 数（造关+测信号，无 PPO backprop，可大如
                 1000，靠 gen_rollout_chunk 分批控显存）。<=0 → 退回 num_levels_per_gen（零回归）。
      gen_rollout_chunk: 分批造关/测信号的每批 level 数（仿 baseline NUM_BATCHES）。<=0 或 >= 池规模
                 → 不分批走一次性路径（零回归）。>0 且 < 池规模 → ceil(池/chunk) 批静态展开。
                 ⚠ ALP signal_mode（euc_alp/euc_cenie_alpquota）禁止分批（cur_bucket_p 全 pool 聚合）。
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

    # [大基数解耦 + 分批] 区分两个数量级（关键修复，2026-06-26）：
    #   - num_levels_per_gen：generator **PPO 训练**用的 level 数（gen_train_iter 内 rollout +
    #     PPO backprop 持全轨迹 + flatten L*T 过 conv→OOM 上限低，应保持小，如 64）。
    #   - pool_per_gen：**注入候选池**每 gen 造关数（只做 inference 造关 + student 测信号，无
    #     PPO backprop，可大，如 1000；靠 GEN_ROLLOUT_CHUNK 分批控峰值显存）。
    #   pool_per_gen<=0 → 退回 num_levels_per_gen（零回归：pool=PPO 数，逐字等价旧行为）。
    #   ⚠ 旧行为下二者相等，故旧 sbatch 不动数值不变。大基数 sbatch 应设 per_gen 小 + pool 大。
    gen_rollout_chunk = int(gen_rollout_chunk)
    pool_per_gen = int(pool_per_gen)
    _pool_pg = pool_per_gen if pool_per_gen > 0 else num_levels_per_gen
    # 造关/测信号按 _pool_pg 跑；chunk>0 且 < _pool_pg → 分批（峰值显存=单 chunk，避 OOM）。
    _pg_chunked = (gen_rollout_chunk > 0) and (gen_rollout_chunk < _pool_pg)

    all_insts = []           # 各 generator 造的 EnvInstance（待 concat）
    all_scores = []          # 各 level 的 difficulty 分（待 concat）
    all_valids = []          # 各 level 是否放下了合法起终点（待 concat，B 注入端过滤用）
    all_succ = []            # [护栏仪表盘] fallback 路径各 gen 的 per-level p（待 concat）
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
                curriculum_arm=curriculum_arm, curriculum_thr=curriculum_thr,
                curriculum_beta=curriculum_beta,
                curriculum_thr_lo=curriculum_thr_lo, curriculum_beta_lo=curriculum_beta_lo,
                ppo_epochs=ppo_epochs, gamma=gamma, gae_lambda=gae_lambda,
                clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef)
        params_list[g], opt_list[g] = p, os_
        per_gen_metrics.append(last_metrics)

        # ── 用更新后的 generator-g 造注入 level（buffer 用 step-G 之后的产物）──
        # 起终点用 place_env（valid_path_check=True 根除不可解图）；rollout 仍 student_env。
        rng, r_roll, r_inst, r_score = jax.random.split(rng, 4)
        if _pg_chunked:
            # 分批造关 + 桥接（峰值显存=单 chunk，避 PCGRL CNN OOM）。造 _pool_pg 张（注入池规模）。
            insts, valids_g = chunked_gen_rollout_and_bridge(
                gen_env, gen_network, p, place_env, r_roll, r_inst,
                _pool_pg, gen_rollout_chunk)
        else:
            env_maps = gen_rollout_batch(gen_env, gen_network, p, r_roll, _pool_pg)
            insts, valids_g = gen_batch_to_env_instances(env_maps, place_env, r_inst)
        all_insts.append(insts)
        all_valids.append(valids_g)   # [B] 接住 valid，下游把 invalid 图 score→-inf 挤出 buffer
        # fallback（auction_lambda=None）：每 gen 用自己信号评自己 level（part2 行为，消融用）。
        if auction_lambda is None:
            if _pg_chunked:
                # 分批 student rollout 测 fallback 信号（pool 也大，student rollout 同需分批）。
                scores_g, _info = chunked_compute_terminal_reward(
                    est_g, student_env, student_network, student_params, insts, r_score,
                    hidden_size, student_rollout_steps, _pool_pg,
                    gen_rollout_chunk, gmm_params=gmm_params, gamma=gamma, gae_lambda=gae_lambda)
            else:
                scores_g, _info = compute_terminal_reward(
                    est_g, student_env, student_network, student_params, insts, r_score,
                    hidden_size, student_rollout_steps, _pool_pg,
                    gmm_params=gmm_params, gamma=gamma, gae_lambda=gae_lambda)
            all_scores.append(scores_g)
            # [护栏仪表盘 §injp] fallback 路径也收集 per-level p（difficulty/euc reward 的 info 带
            #   success_rate）→ 下方 gen_p_* log。auction 路径从 _ainfo 取，这里从各 gen 自评 info 取。
            all_succ.append(_info.get("success_rate", jnp.full((_pool_pg,), jnp.nan)))

    # ── 拼 N×_pool_pg 张 pool（_pool_pg=注入池规模，默认=num_levels_per_gen 零回归）──
    cat_insts = jax.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_insts) \
        if N > 1 else all_insts[0]
    M = N * _pool_pg

    # ── 评分漏斗：auction 多 teacher 出价混合（idea 核心）or fallback 单信号（消融）──
    if auction_lambda is not None:
        # auction：N 个 estimator 对**整个 pool** 都打分（一次 rollout 出三信号，§6）→
        # (3, M) → z-score 抹平量级 → bid → softmax(bid/λ) → 混合 score。决定"现在最该学谁的课程"。
        from auction_bid import mix_scores as auction_mix_scores
        rng, r_auc = jax.random.split(rng)
        # pool 级测信号：M=N×per_gen 关一次 student rollout 也会 OOM（大 pool）→ 同需分批。
        #   chunk>0 且 < M 时分批（ALP 模式禁止分批，见 chunked_all_signals_on_levels）。
        _pool_chunked = (gen_rollout_chunk > 0) and (gen_rollout_chunk < M)
        if _pool_chunked:
            per_est, _ainfo = chunked_all_signals_on_levels(
                student_env, student_network, student_params, cat_insts, r_auc,
                hidden_size, student_rollout_steps, M, gen_rollout_chunk,
                gmm_params=gmm_params, gamma=gamma, gae_lambda=gae_lambda,
                signal_mode=signal_mode,
                prev_bucket_p=gen_state.get("prev_bucket_p", None),
                gate_w_hard=gate_w_hard, gate_w_easy=gate_w_easy)  # (K, M)
        else:
            per_est, _ainfo = all_signals_on_levels(
                student_env, student_network, student_params, cat_insts, r_auc,
                hidden_size, student_rollout_steps, M, gmm_params=gmm_params,
                gamma=gamma, gae_lambda=gae_lambda, signal_mode=signal_mode,
                prev_bucket_p=gen_state.get("prev_bucket_p", None),
                gate_w_hard=gate_w_hard, gate_w_easy=gate_w_easy)  # (3, M)
        cat_scores, auc_w, auc_bids = auction_mix_scores(
            per_est, float(auction_lambda), weight_factors=auction_weight_factors)  # (M,)
        # ── ALP 配额调节器（modulation 层，signal_mode='euc_cenie_alpquota'）──
        #   auction 只 [euc,cenie] 选关；ALP 作 per-level 加性 bonus 叠在分数上，按 euc 桶调配额：
        #   进步档(ALP高)的 level 分数被抬→更多进 top-K；饱和档不抬。冷启 ALP=0→bonus=0→退化纯 auction。
        #   bonus = COEF × z(alp_perlevel)（z 抹量级，与 cat_scores ~O(1) 可比）。加性、独立、不与他维耦合
        #   ——区别于 bid 层（euc_alp）的乘性权重竞争。
        _alp_pl = _ainfo.get("alp_quota_perlevel", None)
        if alp_quota_coef != 0.0 and _alp_pl is not None:
            _apl = jnp.where(jnp.isfinite(_alp_pl), _alp_pl, 0.0)        # incomplete→0 不污染 z
            _fin = jnp.isfinite(_alp_pl)
            _cnt = jnp.maximum(_fin.sum(), 1)
            _mean = jnp.where(_fin, _apl, 0.0).sum() / _cnt
            _std = jnp.sqrt(jnp.where(_fin, (_apl - _mean) ** 2, 0.0).sum() / _cnt) + 1e-8
            _z_alp = jnp.where(_fin, (_apl - _mean) / _std, 0.0)        # (M,) z-score，incomplete=0
            cat_scores = cat_scores + float(alp_quota_coef) * _z_alp    # 加性 bonus
        # ── 单边可学性 gate 修正项（host 层，signal_mode='anchored_cenie_gate'）──
        #   auction 只 [anchored-PVL,cenie] 选关；gate=-relu(0.5-p) 作 per-level 加性修正，
        #   final = auction_bid + GATE_WEIGHT·z(gate)。z 抹量级（incomplete 已在信号函数置 0，
        #   z 时再屏蔽非 finite 位）。p<0.5 的太难关被压低分→更难进 top-K（p 越低压越狠）。
        #   仿 ALPquota 加性叠加，独立不与他维耦合（区别 bid 层乘性竞争）。
        _gate_pl = _ainfo.get("gate_perlevel", None)
        if gate_weight != 0.0 and _gate_pl is not None:
            _gfin = jnp.isfinite(_gate_pl)
            _gpl = jnp.where(_gfin, _gate_pl, 0.0)                       # incomplete/非finite→0 不污染 z
            _gcnt = jnp.maximum(_gfin.sum(), 1)
            _gmean = jnp.where(_gfin, _gpl, 0.0).sum() / _gcnt
            _gstd = jnp.sqrt(jnp.where(_gfin, (_gpl - _gmean) ** 2, 0.0).sum() / _gcnt) + 1e-8
            _z_gate = jnp.where(_gfin, (_gpl - _gmean) / _gstd, 0.0)     # (M,) z-score，incomplete=0
            cat_scores = cat_scores + float(gate_weight) * _z_gate       # 加性修正项
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

    # ── [§5.9 筛选层 geo 下界] 候选池里 geo<thr_lo 的"太简单"关 score→-inf，被 top-K 挤掉。
    #    与 reward 层下界罚配合：reward 层让 generator 学会多造难关(把池子做厚,治本,慢)，筛选层
    #    保证**这一代注入的 60 关当下就**不含太简单的(立即止血+安全网,即使 generator 偶尔躺平)。
    #    ⚠ 与 gate 失败的本质区别：筛 geo(不塌缩,1~24 连续展开,中难关池里有足量)而非 p(两极塌缩,
    #    中间难度候选≈0=筛空集)。geo 是内禀几何不读 student,免疫 reward-hack。
    #    thr_lo=0(easy 档/未配置) → 条件恒真不过滤(零回归)。靠 reward 层把池子做厚保多样性余量。
    #    [复用] _pool_struct 在此算一次,下方反推下界 log 直接复用(避免对 3000 池重复 BFS)。
    _pool_struct = None
    if curriculum_arm != CURRICULUM_ARM_NONE:
        _pool_struct = struct_value_from_instances(cat_insts, M, curriculum_arm)  # (M,) 全候选池
        if curriculum_thr_lo is not None:
            _too_easy = _pool_struct < curriculum_thr_lo                 # (M,) bool
            cat_scores = jnp.where(_too_easy, -jnp.inf, cat_scores)

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

    # ALP 状态写回：euc_alp 路径下 _ainfo["cur_bucket_p"] 非 None → 更新；否则保留旧 prev（非 ALP 模式）。
    _new_bucket_p = gen_state.get("prev_bucket_p", None)
    if auction_lambda is not None and _ainfo.get("cur_bucket_p", None) is not None:
        _new_bucket_p = _ainfo["cur_bucket_p"]
    new_gen_state = {
        "params": params_list,
        "opt_state": opt_list,
    }
    if _new_bucket_p is not None:
        new_gen_state["prev_bucket_p"] = _new_bucket_p
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
    # ── [机制判据 log] 注入关(top_insts, 真进 buffer 的)的 fill/euc 分布 (STAGE4_phase2 §方向3) ──
    #   s4p2A 没存注入关只能 PNG 反推(memory 诊断教训); 这里直接算注入关墙密度+起终点欧氏距离,
    #   判 buffer 主体是否健康长程(euc大,fill适中) vs 少数堆墙(fill>0.5)。纯几何, 不依赖 student。
    _inj_md = top_insts.map_data                                  # (k, H, W)
    _inj_fill = (_inj_md == 1).reshape(_inj_md.shape[0], -1).mean(axis=1)   # (k,)
    _inj_pa = top_insts.agent_pos.reshape((top_insts.map_data.shape[0], -1))[:, :2]
    _inj_pb = top_insts.goal_pos.reshape((top_insts.map_data.shape[0], -1))[:, :2]
    _inj_euc = jnp.sqrt(((_inj_pa - _inj_pb) ** 2).sum(axis=1) + 1e-8)      # (k,)
    metrics["gen_inj_fill_median"] = jnp.median(_inj_fill)
    metrics["gen_inj_fill_mean"] = _inj_fill.mean()
    metrics["gen_inj_fill_hi_frac"] = (_inj_fill > 0.5).mean()    # 堆墙关占注入比例
    metrics["gen_inj_euc_median"] = jnp.median(_inj_euc)
    metrics["gen_inj_euc_mean"] = _inj_euc.mean()
    # ── [结构课程 log] 注入关测地长度分布（课程主轴的注入侧验证，STAGE4_phase2 §结构课程，2026-06-25）──
    #   euc 是直线距离(可被堆墙绕过)，测地是真实最短路格数=课程难度轴。看课程是否把注入关测地从
    #   塌缩谱推向 Nav2/3 走廊拓扑(测地 16~22)。geodesic_from_instances 复用 jaxnav APSP，不依赖 student。
    _inj_geo = geodesic_from_instances(top_insts, top_insts.map_data.shape[0]).astype(jnp.float32)  # (k,)
    metrics["gen_inj_geo_median"] = jnp.median(_inj_geo)
    metrics["gen_inj_geo_mean"] = _inj_geo.mean()
    metrics["gen_inj_geo_perlevel"] = _inj_geo            # (k,) wandb 直方图用（看课程推进结构分布）
    if curriculum_arm != CURRICULUM_ARM_NONE and curriculum_thr is not None:
        # 当前难度档 + 注入关超档比例（课程生效则注入关测地应贴 thr 之下，超档比例随训练→0）。
        metrics["curriculum_threshold"] = jnp.asarray(curriculum_thr, dtype=jnp.float32)
        metrics["curriculum_threshold_lo"] = jnp.asarray(curriculum_thr_lo, dtype=jnp.float32)
        _inj_struct = _inj_geo if curriculum_arm == CURRICULUM_ARM_GEO else _inj_fill
        metrics["gen_inj_frac_over_threshold"] = (_inj_struct > curriculum_thr).astype(jnp.float32).mean()
        # ── [§5.9 反推精准下界用] 候选池(全 3000 关)的 geo 完整分布 + 分位数 ───────────────
        #   下界 [0,3,8,12] 是猜的。要反推精准值须看**候选池**(注入前)各档 generator 实际造关的
        #   geo 分布——注入后 60 关是 top-K by score(被选择偏置污染)，候选池才是 generator 真实
        #   产出。记录全分布直方图 + 关键分位(p10/p25/p50/p75) → 事后按"想要的注入难度档"反查
        #   合理 thr_lo(如想让中难档注入关 geo≥某值,看候选池该分位够不够供给)。
        # _pool_struct 已在筛选层算过(复用,不重复 BFS)；arm≠none 时必非 None。
        _pool_reach = geodesic_reachable_mask(cat_insts, M)          # 排不可达(geo=大值)假值
        _pf = _pool_reach.astype(jnp.float32); _pn = jnp.maximum(_pf.sum(), 1.0)
        _pg = jnp.where(_pool_reach, _pool_struct, jnp.nan)
        metrics["gen_pool_geo_mean"] = (jnp.where(_pool_reach, _pool_struct, 0.0).sum() / _pn)
        metrics["gen_pool_geo_p10"] = jnp.nanquantile(_pg, 0.10)
        metrics["gen_pool_geo_p25"] = jnp.nanquantile(_pg, 0.25)
        metrics["gen_pool_geo_p50"] = jnp.nanquantile(_pg, 0.50)
        metrics["gen_pool_geo_p75"] = jnp.nanquantile(_pg, 0.75)
        metrics["gen_pool_geo_perlevel"] = _pool_struct              # (M,) 全分布直方图(含不可达大值,看时滤)
        # 候选池里 geo≥各候选下界的供给量(反推:某下界够不够 60 关注入)。
        for _cand_lo in (3.0, 5.0, 8.0, 10.0, 12.0, 16.0):
            metrics[f"gen_pool_frac_geo_ge_{int(_cand_lo)}"] = (
                (_pool_struct >= _cand_lo) & _pool_reach).astype(jnp.float32).mean()
    # ── [护栏仪表盘 log] pool 全体 level 的 per-level student p 分布 (STAGE4_phase2 §injp, 2026-06-25) ──
    #   injp 离线测出注入关 p 两极塌缩(可学区0%)；这里把训练中**实时** per-level p 存进 wandb，
    #   作仪表盘：看 p 分布是否在可学区(0.2≤p≤0.8) vs 两极。口径对齐 injp。
    #   auction 路径从 _ainfo 取 pool p；fallback(各gen自评)从 all_succ concat 取。
    _p_pool = None
    if auction_lambda is not None and _ainfo.get("success_rate", None) is not None:
        _p_pool = _ainfo["success_rate"]                            # (M,) auction 路径
    elif auction_lambda is None and len(all_succ) > 0:
        _p_pool = jnp.concatenate(all_succ, axis=0)                 # (M,) fallback 路径
    if _p_pool is not None:
        _pf = jnp.where(jnp.isfinite(_p_pool), _p_pool, jnp.nan)
        metrics["gen_p_mean"] = jnp.nanmean(_pf)
        metrics["gen_p_median"] = jnp.nanmedian(_pf)
        metrics["gen_p_learnable_frac"] = jnp.mean((_pf >= 0.2) & (_pf <= 0.8))   # 可学区占比
        metrics["gen_p_extreme_frac"] = jnp.mean((_pf < 0.05) | (_pf > 0.95))     # 两极占比
        # [自适应课程 2026-06-26] 高 p 占比 = 注入关里 p>0.8 的比例（incomplete 排除：仅 finite 位
        #   计入分母）。host 层状态机据此判 student 是否学会当前难度档 → 决定升档（升档准则）。
        _p_finite = jnp.isfinite(_p_pool)
        _p_hi = jnp.where(_p_finite, (_p_pool > 0.8).astype(jnp.float32), 0.0)
        metrics["gen_p_high_frac"] = _p_hi.sum() / jnp.maximum(_p_finite.sum(), 1)
        # [双向滑落退档 2026-06-26] 低 p 占比 = 注入关里 p<0.2 的比例（incomplete 排除，同 high_frac
        #   口径）。host 层退档状态机据此判 student 是否被当前难度档压垮 → 触发退档（有地板）。
        _p_lo = jnp.where(_p_finite, (_p_pool < 0.2).astype(jnp.float32), 0.0)
        metrics["gen_p_low_frac"] = _p_lo.sum() / jnp.maximum(_p_finite.sum(), 1)
        metrics["gen_p_perlevel"] = _p_pool                        # (M,) 全量，wandb 直方图用
    # [诊断仪表盘 2026-06-27] per-level 裸 PVL / mean|adv| 全过程记录（与 gen_p_perlevel 同口径）。
    #   仅 auction 路径(_ainfo 带出)且 adv 分支(anchored/pvl/...)有定义；其余路径键缺失→跳过（零回归）。
    #   动机(§5.6 信号区分度实验)：p(1-p) 塌缩时 PVL/|adv| 仍有区分度，需看它们全过程如何变化、
    #   如何对照课程升降档。mean/median + perlevel 数组(host 层存直方图)。incomplete(-inf)用 finite 过滤。
    _pvl_pl = _ainfo.get("pvl_raw_perlevel", None) if auction_lambda is not None else None
    _adv_pl = _ainfo.get("absadv_perlevel", None) if auction_lambda is not None else None
    if _pvl_pl is not None:
        _pvf = jnp.where(jnp.isfinite(_pvl_pl), _pvl_pl, jnp.nan)
        metrics["gen_pvl_mean"] = jnp.nanmean(_pvf)
        metrics["gen_pvl_median"] = jnp.nanmedian(_pvf)
        metrics["gen_pvl_perlevel"] = _pvl_pl                       # (M,) 全量
    if _adv_pl is not None:
        _avf = jnp.where(jnp.isfinite(_adv_pl), _adv_pl, jnp.nan)
        metrics["gen_absadv_mean"] = jnp.nanmean(_avf)
        metrics["gen_absadv_median"] = jnp.nanmedian(_avf)
        metrics["gen_absadv_perlevel"] = _adv_pl                    # (M,) 全量
    if auction_info is not None:
        # auction 权重/bid（诊断：看三 teacher 谁出价高=谁的课程当前被选；λ 控 single-winner↔uniform）。
        metrics["auction_weights"] = auction_info["weights"]   # (N,)
        metrics["auction_bids"] = auction_info["bids"]         # (N,)
    return top_scores, top_insts, new_gen_state, metrics
