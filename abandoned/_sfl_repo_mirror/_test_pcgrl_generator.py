"""单测：pcgrl_generator 第 1 块（map 桥接 + 起终点 + EnvInstance）。仅 Oscar sfl env（真 jaxnav）。

验证：
  T1 pcgrl_map_to_jaxnav: {0,1,2}→{0,1} 正确（EMPTY→0，BORDER/WALL→1）。
  T2 map_to_env_instance: 产出 EnvInstance 各字段 shape/dtype 正确，map_data∈{0,1}。
  T3 起终点合法：start/goal 落在自由格、互不重合、在连通区内（valid=True 比例够高）。
  T4 闭环：产出的 EnvInstance 能喂进 env.set_env_instance 不崩，得到 obs。

跑：conda activate sfl; cd ~/sampling-for-learnability; PYTHONPATH=sfl/train python sfl/train/_test_pcgrl_generator.py
"""
import sys
sys.path.insert(0, "sfl/train")
import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from pcgrl_generator import (pcgrl_map_to_jaxnav, map_to_env_instance,
                             PCGRL_BORDER, PCGRL_EMPTY, PCGRL_WALL)

PASS = 0; FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS {name}")
    else: FAIL += 1; print(f"  FAIL {name}")

# jaxnav 与 jaxnav-sfl.yaml 对齐：Grid-Rand-Poly, map_size [11,11], num_agents 1。
env = JaxNav(num_agents=1, map_id="Grid-Rand-Poly",
             map_params={"map_size": [11, 11], "fill": 0.6, "start_pad": 1.5,
                         "valid_path_check": False, "sample_test_case_type": "grid"})
H = W = 11
print("=== T1 map 编码桥接 ===")
m = jnp.array([[PCGRL_BORDER, PCGRL_EMPTY, PCGRL_WALL],
               [PCGRL_EMPTY, PCGRL_EMPTY, PCGRL_BORDER]])
jm = pcgrl_map_to_jaxnav(m)
check("EMPTY→0", bool(jm[0, 1] == 0 and jm[1, 0] == 0 and jm[1, 1] == 0))
check("WALL→1", bool(jm[0, 2] == 1))
check("BORDER→1", bool(jm[0, 0] == 1 and jm[1, 2] == 1))
check("只含{0,1}", bool(jnp.all((jm == 0) | (jm == 1))))

print("=== T2/T3 map_to_env_instance 合法性 ===")
# 造一个稀疏障碍的 PCGRL map（多数 EMPTY，边界 WALL），保证有连通自由空间。
base = jnp.full((H, W), PCGRL_EMPTY, dtype=jnp.int32)
base = base.at[0, :].set(PCGRL_BORDER).at[-1, :].set(PCGRL_BORDER)
base = base.at[:, 0].set(PCGRL_BORDER).at[:, -1].set(PCGRL_BORDER)
base = base.at[5, 2:8].set(PCGRL_WALL)   # 一道墙

f = jax.jit(lambda rng: map_to_env_instance(env, base, rng))
valids = []
insts = []
for s in range(20):
    inst, valid = f(jax.random.PRNGKey(s))
    valids.append(bool(valid)); insts.append(inst)
inst0 = insts[0]
check("map_data shape (H,W)", inst0.map_data.shape == (H, W))
check("map_data ∈{0,1}", bool(jnp.all((inst0.map_data == 0) | (inst0.map_data == 1))))
check("agent_pos shape (1,2)", inst0.agent_pos.shape == (1, 2))
check("goal_pos shape (1,2)", inst0.goal_pos.shape == (1, 2))
check("agent_theta shape (1,)", inst0.agent_theta.shape == (1,))
frac_valid = np.mean(valids)
print(f"  valid 比例 = {frac_valid:.2f} (20 seeds)")
check("valid 比例 ≥0.8", frac_valid >= 0.8)
# 起终点不重合
sep = [bool(jnp.any(i.agent_pos != i.goal_pos)) for i in insts]
check("起终点不重合", all(sep))

print("=== T4 闭环：喂 set_env_instance ===")
try:
    obs, state = env.set_env_instance(inst0)
    check("set_env_instance 不崩", True)
    check("obs 非空", obs is not None and len(obs) == env.num_agents)
    check("state.map_data 对上", bool(jnp.array_equal(state.map_data, inst0.map_data)))
except Exception as e:
    check("set_env_instance 不崩", False)
    print("   异常:", type(e).__name__, str(e)[:200])

print("=== T5 valid_path_check=True 根除不可解图 ===")
# 用 valid_path_check=True 的 env，造一张「完整横墙隔断」的图：上半区 vs 下半区互不连通。
# 验证 start/goal 必在同一连通块（即不会一个上半、一个下半）。
env_vpc = JaxNav(num_agents=1, map_id="Grid-Rand-Poly",
                 map_params={"map_size": [11, 11], "fill": 0.6, "start_pad": 1.5,
                             "valid_path_check": True, "sample_test_case_type": "grid"})
check("valid_path_check 属性=True", bool(getattr(env_vpc._map_obj, "valid_path_check", False)))
# 隔断图：第 5 行（含边界坐标）整行 WALL，把 11x11 分成上下两块。
split = jnp.full((H, W), PCGRL_EMPTY, dtype=jnp.int32)
split = split.at[0, :].set(PCGRL_BORDER).at[-1, :].set(PCGRL_BORDER)
split = split.at[:, 0].set(PCGRL_BORDER).at[:, -1].set(PCGRL_BORDER)
split = split.at[5, :].set(PCGRL_WALL)        # 整行墙，彻底隔断上下
jm_split = pcgrl_map_to_jaxnav(split)
fv = jax.jit(lambda rng: map_to_env_instance(env_vpc, split, rng))
same_region = []
for s in range(30):
    inst, valid = fv(jax.random.PRNGKey(100 + s))
    # 含边界坐标系下，agent_pos/goal_pos 的 y 坐标（[x,y] 第二维）应在墙同侧（都 <5 或都 >5）。
    ay = float(inst.agent_pos[0, 1]); gy = float(inst.goal_pos[0, 1])
    same = (ay < 5 and gy < 5) or (ay > 5 and gy > 5)
    same_region.append(same)
frac_same = np.mean(same_region)
print(f"  start/goal 同侧比例 = {frac_same:.2f} (30 seeds, 隔断图)")
check("valid_path_check=True 下起终点必同连通块(同侧≥0.99)", frac_same >= 0.99)

print(f"\n==== {PASS} PASS / {FAIL} FAIL ====")
sys.exit(0 if FAIL == 0 else 1)
