"""集成冒烟：get_generator_set 一个 outer round 端到端跑通（交替训练阶段 G）。

仅 Oscar sfl env（真 jaxnav + pcgrl vendor）。**必须 srun 跑**（含 PPO 梯度/student rollout，
非 login 安全）：见 memory oscar-login-node-no-compute。

验证（小参数，快）：
  G1 make_generator_state 建 N 个 generator（params/opt_state 非空，纯数组 pytree）。
  G2 get_generator_set 一个 outer round 不崩，返回 (scores, instances, new_gen_state, metrics)。
  G3 instances 形状 = (num_to_save, ...)，map_data∈{0,1}，能喂 student_env.set_env_instance。
  G4 scores 有限（无 NaN/inf；incomplete level 的 -inf 已在 get_generator_set 内 top-K 排末）。
  G5 generator params 真动（|Δθ|>0，对比 outer round 前后）。
  G6 new_gen_state 结构 = 传入 gen_state（纯数组 pytree，无 str；jit 输出兼容）。

跑：conda activate sfl; cd ~/sampling-for-learnability;
    PYTHONPATH=sfl/train python sfl/train/_test_generator_integration.py
"""
import sys
sys.path.insert(0, "sfl/train")
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from sfl.train.common.network import ActorCriticRNN, ScannedRNN
from pcgrl_generator import (
    make_pcgrl_env, make_generator_network, make_generator_state, get_generator_set,
    GEN_ESTIMATOR_DIFFICULTY, GEN_ESTIMATOR_PVL, GEN_ESTIMATOR_CENIE,
)

PASS = 0; FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS {name}")
    else: FAIL += 1; print(f"  FAIL {name}")

# ── 小参数（冒烟求快；真实训练值在 config）──
MAP_SIZE = 11          # 对齐 jaxnav-sfl.yaml map_params.map_size [11,11]
# part2: N=3 异质 teacher generators，各绑 difficulty/PVL/CENIE。
ESTIMATOR_IDS = [GEN_ESTIMATOR_DIFFICULTY, GEN_ESTIMATOR_PVL, GEN_ESTIMATOR_CENIE]
N_GEN = len(ESTIMATOR_IDS)   # 3
# N=3 在 CPU 上跑三份独立 generator rollout+PPO+编译，规模砍到最小（只验管道通+三 params 各动）。
NUM_LEVELS_PER_GEN = 3   # pool = N×此值 = 9
NUM_TO_SAVE = 6          # K=6 < pool=9：验证漏斗排序挤掉 incomplete(-inf)
STUDENT_ROLLOUT = 30   # 远小于训练 ROLLOUT_STEPS=1000，只为管道跑通
GEN_BOARD_SCANS = 0.5  # generator episode 步数 = 11×11×此值 ≈ 60（vs 默认 3.0=363），CPU 提速
# HIDDEN_SIZE 必须 = FC_DIM_SIZE：ScannedRNN 的 GRU features 取自 embedding 末维(=FC_DIM_SIZE)，
# initialize_carry(B, HIDDEN_SIZE) 须与之一致，否则 jnp.where 广播失败。真实训练两者都 512。
HIDDEN_SIZE = 512
FC_DIM_SIZE = 512
PPO_EPOCHS = 2
OUTER_STEPS = 1

# t_config：student ActorCriticRNN 需要的最小字段（见 pcgrl_generator part1 接口注释）。
t_config = {"HIDDEN_SIZE": HIDDEN_SIZE, "FC_DIM_SIZE": FC_DIM_SIZE,
            "USE_LAYER_NORM": False, "LOG_DORMANCY": True}

# ── student env（valid_path_check=False，对齐训练）+ student network ──
student_env = JaxNav(num_agents=1, map_id="Grid-Rand-Poly",
                     map_params={"map_size": [MAP_SIZE, MAP_SIZE], "fill": 0.6,
                                 "start_pad": 1.5, "valid_path_check": False,
                                 "sample_test_case_type": "grid"})
student_network = ActorCriticRNN(student_env.agent_action_space().shape[0], config=t_config)

# ── place_env（valid_path_check=True）：generator 注入放起终点用，根除不可解图（§R5）。
#    rollout 仍在 student_env（False，与训练分布一致）。
place_env = JaxNav(num_agents=1, map_id="Grid-Rand-Poly",
                   map_params={"map_size": [MAP_SIZE, MAP_SIZE], "fill": 0.6,
                               "start_pad": 1.5, "valid_path_check": True,
                               "sample_test_case_type": "grid"})

rng = jax.random.PRNGKey(0)
rng, r_init = jax.random.split(rng)
# student params init（与 jaxnav_sfl 一致：init_hstate + (obs,done) dummy）。
init_hstate = ScannedRNN.initialize_carry(1, HIDDEN_SIZE)
init_x = (jnp.zeros((1, 1, student_env.lidar_num_beams + 5)), jnp.zeros((1, 1)))
student_params = student_network.init(r_init, init_hstate, init_x)

# ── generator env + network + state ──
gen_env, _gp = make_pcgrl_env(map_size=MAP_SIZE, max_board_scans=GEN_BOARD_SCANS)
gen_network, gen_meta = make_generator_network(gen_env)
print("gen_meta", gen_meta)

estimator_ids = ESTIMATOR_IDS
rng, r_gs = jax.random.split(rng)
gen_state, gen_optimizer = make_generator_state(r_gs, gen_env, gen_network, estimator_ids)

# ── CENIE 需 gmm_params：先冷启（valid=False，cenie 返 0），再用随机 hidden fit 一个
#    valid=True 的 GMM，验证 CENIE generator 拿到真实非零 reward（而非恒 0 冷启）。
from cenie_density import init_gmm_params, fit_visitation_gmm
rng, r_feat = jax.random.split(rng)
_feats = jax.random.normal(r_feat, (128, HIDDEN_SIZE))   # 假 visitation hidden（128 样本，CPU 提速）
gmm_params = fit_visitation_gmm(np.asarray(_feats), max_components=4, min_components=2,
                                reg_covar=1e-2)
print("gmm valid=", bool(gmm_params.valid))

print("=== G1 make_generator_state ===")
check("params list 长度 = N", len(gen_state["params"]) == N_GEN)
check("opt_state list 长度 = N", len(gen_state["opt_state"]) == N_GEN)
check("gen_state 无 estimator_ids 键（纯数组 pytree）", "estimator_ids" not in gen_state)
leaves = jax.tree_util.tree_leaves(gen_state)
check("gen_state 全数组叶子（无 str）", all(hasattr(x, "dtype") for x in leaves))

# 记录 outer round 前的 params（验 G5 每个 generator 都真动）。
params_before_all = [jax.tree_map(lambda x: jnp.array(x), p) for p in gen_state["params"]]

print("=== G2 get_generator_set 一个 outer round (N=3 + auction λ=1.0) ===")
rng, r_gen = jax.random.split(rng)
try:
    scores, instances, new_gen_state, metrics = get_generator_set(
        r_gen, student_params, gen_state, gen_optimizer, estimator_ids,
        gen_env, gen_network, student_env, student_network,
        HIDDEN_SIZE, STUDENT_ROLLOUT,
        num_levels_per_gen=NUM_LEVELS_PER_GEN, num_to_save=NUM_TO_SAVE,
        place_env=place_env, gmm_params=gmm_params, auction_lambda=1.0,
        gen_outer_steps=OUTER_STEPS, ppo_epochs=PPO_EPOCHS)
    check("get_generator_set(auction) 不崩", True)
except Exception as e:
    check("get_generator_set 不崩", False)
    import traceback; traceback.print_exc()
    print(f"\n==== {PASS} PASS / {FAIL} FAIL ====")
    sys.exit(1)

print("  metrics:", {k: (v if not hasattr(v, "shape") else f"arr{v.shape}")
                      for k, v in metrics.items() if k != "gen_per_gen"})
print("  scores:", np.asarray(scores))

print("=== G3 instances 形状 + 可注入 ===")
check("scores 形状 (num_to_save,)", scores.shape == (NUM_TO_SAVE,))
check("instances.map_data 形状 (K,H,W)",
      instances.map_data.shape == (NUM_TO_SAVE, MAP_SIZE, MAP_SIZE))
check("instances.agent_pos 形状 (K,1,2)", instances.agent_pos.shape == (NUM_TO_SAVE, 1, 2))
check("map_data ∈{0,1}", bool(jnp.all((instances.map_data == 0) | (instances.map_data == 1))))
try:
    obsv, state = jax.vmap(student_env.set_env_instance, in_axes=(0,))(instances)
    check("vmap set_env_instance 不崩", True)
    check("state.map_data 对上", bool(jnp.array_equal(state.map_data, instances.map_data)))
except Exception as e:
    check("vmap set_env_instance 不崩", False)
    print("   异常:", type(e).__name__, str(e)[:200])

print("=== G4 scores 有限 ===")
check("scores 全有限（无 NaN/inf）", bool(jnp.all(jnp.isfinite(scores))))

print("=== G5 每个 generator params 真动（N=3 三信号各自被训动）===")
for g, est in enumerate(estimator_ids):
    pa = new_gen_state["params"][g]
    pb = params_before_all[g]
    delta = jax.tree_util.tree_reduce(
        lambda acc, ab: acc + jnp.sum(jnp.abs(ab[0] - ab[1])),
        jax.tree_map(lambda a, b: (a, b), pa, pb), 0.0)
    print(f"  generator-{g}({est}) |Δθ| = {float(delta):.4f}")
    check(f"generator-{g}({est}) params 真动（|Δθ|>0）", float(delta) > 0)

print("=== G6 new_gen_state 结构兼容 jit（纯数组 pytree）===")
check("new_gen_state 键 = {params, opt_state}",
      set(new_gen_state.keys()) == {"params", "opt_state"})
new_leaves = jax.tree_util.tree_leaves(new_gen_state)
check("new_gen_state 全数组叶子", all(hasattr(x, "dtype") for x in new_leaves))
# 结构必须与传入 gen_state 完全一致（jit 多轮传回不重编译/不报错）。
in_def = jax.tree_util.tree_structure(gen_state)
out_def = jax.tree_util.tree_structure(new_gen_state)
check("new_gen_state pytree 结构 == 传入 gen_state", in_def == out_def)

print("=== G7 注入 level 起终点连通（valid_path_check=True 生效，§R5）===")
check("place_env.valid_path_check=True",
      bool(getattr(place_env._map_obj, "valid_path_check", False)))
# 每张注入 level：start 含边界格 = floor(agent_pos)，goal = floor(goal_pos)；用 place_env
# 的真连通分量验 goal 落在 start 的连通块（component_mask_with_pos）。不连通=不可解图。
import jaxmarl.environments.jaxnav.jaxnav_graph_utils as _gu
def _connected(map_data, apos, gpos):
    s = jnp.floor(apos).astype(jnp.int32)              # [x,y] 含边界格
    g = jnp.floor(gpos).astype(jnp.int32)
    comp = _gu.component_mask_with_pos(map_data, s)     # start 连通块 mask (H,W)
    return bool(comp[g[1], g[0]] > 0)                   # goal 格在该连通块内
conn = [_connected(instances.map_data[i], instances.agent_pos[i, 0], instances.goal_pos[i, 0])
        for i in range(NUM_TO_SAVE)]
frac_conn = float(np.mean(conn))
print(f"  注入 level 起终点连通比例 = {frac_conn:.2f} ({NUM_TO_SAVE} levels)")
check("注入 level 起终点全连通（无不可解图）", frac_conn >= 0.99)

print("=== G8 三种 terminal_reward 函数直测（形状/有限性）===")
from pcgrl_generator import (terminal_reward_difficulty, terminal_reward_pvl,
                             terminal_reward_cenie, gen_rollout_batch,
                             gen_batch_to_env_instances)
rng, r_b, r_i = jax.random.split(rng, 3)
_maps = gen_rollout_batch(gen_env, gen_network, gen_state["params"][0], r_b, NUM_LEVELS_PER_GEN)
_insts, _ = gen_batch_to_env_instances(_maps, place_env, r_i)
for name, fn, kw in [
    ("difficulty", terminal_reward_difficulty, {}),
    ("pvl", terminal_reward_pvl, {}),
    ("cenie", terminal_reward_cenie, {"gmm_params": gmm_params}),
]:
    rng, r_t = jax.random.split(rng)
    rew, info = fn(student_env, student_network, student_params, _insts, r_t,
                   HIDDEN_SIZE, STUDENT_ROLLOUT, NUM_LEVELS_PER_GEN, **kw)
    finite_frac = float(jnp.mean(jnp.isfinite(rew)))
    print(f"  {name}: shape={rew.shape} finite_frac={finite_frac:.2f} "
          f"sample={np.asarray(rew)[:4]}")
    check(f"{name} reward 形状 (num_levels,)", rew.shape == (NUM_LEVELS_PER_GEN,))
    # 非 incomplete 的应有限；至少不能全 NaN（incomplete 的 -inf 允许）。
    check(f"{name} reward 无 NaN", bool(jnp.all(~jnp.isnan(rew))))
# CENIE 用 valid GMM 应产非零（区别于冷启恒 0）：至少有一个 finite 且非 0。
rng, r_c = jax.random.split(rng)
cenie_rew, _ = terminal_reward_cenie(student_env, student_network, student_params, _insts,
                                     r_c, HIDDEN_SIZE, STUDENT_ROLLOUT, NUM_LEVELS_PER_GEN,
                                     gmm_params=gmm_params)
fin = jnp.isfinite(cenie_rew)
nonzero = bool(jnp.any(fin & (jnp.abs(cenie_rew) > 1e-6)))
print(f"  CENIE(valid gmm) 非零? {nonzero}  values={np.asarray(cenie_rew)[:4]}")
check("CENIE 用 valid GMM 产非零 reward（非冷启恒 0）", nonzero)

print("=== G9 auction 出价漏斗（all_signals_on_levels → auction_mix_scores）===")
from pcgrl_generator import all_signals_on_levels
from auction_bid import mix_scores as auction_mix_scores
# G2 已确认 auction 模式端到端不崩；这里直测 (N,M) 矩阵 + 混合 + 量级抹平。
rng, r_as = jax.random.split(rng)
per_est, as_info = all_signals_on_levels(
    student_env, student_network, student_params, _insts, r_as,
    HIDDEN_SIZE, STUDENT_ROLLOUT, NUM_LEVELS_PER_GEN, gmm_params=gmm_params)
print(f"  per_est shape={per_est.shape}  "
      f"difficulty~{np.asarray(per_est[0])}  pvl~{np.asarray(per_est[1])}  cenie~{np.asarray(per_est[2])}")
check("per_est 形状 (3, M)", per_est.shape == (3, NUM_LEVELS_PER_GEN))
mixed, w, bids = auction_mix_scores(per_est, 1.0)
print(f"  auction weights={np.asarray(w)}  bids={np.asarray(bids)}  mixed={np.asarray(mixed)}")
check("auction weights 形状 (3,)", w.shape == (3,))
check("auction weights 和≈1（单纯形）", bool(jnp.abs(w.sum() - 1.0) < 1e-4))
check("auction weights 非负", bool(jnp.all(w >= -1e-6)))
# 量级抹平验证：difficulty(≈-0.25)/PVL(≈0)/CENIE(≈数百) 量级悬殊，但 z-score 后 bid 应同量级
# （不会 CENIE 的大数值碾压另两个）。bid 是 z 后 top-k 均值，三者绝对值应在同一数量级（<10×差）。
fin_bids = bids[jnp.isfinite(bids)]
if len(fin_bids) >= 2:
    ratio = float(jnp.max(jnp.abs(fin_bids)) / (jnp.min(jnp.abs(fin_bids)) + 1e-6))
    print(f"  bid 量级比（max/min |bid|）= {ratio:.1f}（z-score 抹平后应 O(1~10)，非原始 ~2000×）")
    check("三信号量级被 z-score 抹平（bid 比 < 100）", ratio < 100)
# 混合 score：complete level 应有限（incomplete 仍 -inf）。
mfin = jnp.isfinite(mixed)
check("混合 score 至少部分有限", bool(jnp.any(mfin)))
check("混合 score 无 NaN", bool(jnp.all(~jnp.isnan(mixed))))

# G2 的 auction metrics（端到端跑出的权重）。
if "auction_weights" in metrics:
    print(f"  [G2 end-to-end] auction_weights={np.asarray(metrics['auction_weights'])} "
          f"bids={np.asarray(metrics['auction_bids'])}")
    check("G2 metrics 含 auction_weights/bids", True)

print(f"\n==== {PASS} PASS / {FAIL} FAIL ====")
sys.exit(0 if FAIL == 0 else 1)
