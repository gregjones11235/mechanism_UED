# PCGRL Generator 注入层 —— 实现设计（写代码前的 review 锚点）

对应方案B §2 注入层 + §3「PCGRL tile-MDP + PPO（已拍板）」。本文档把"动手前必须定死的接线事实"固化，
代码以此为准。所有事实来自真实代码（pcgrl-jax @ Oscar ~/pcgrl-jax；SFL @ ~/sampling-for-learnability），非文档转抄。

---

## 0. 一句话目标

把 SFL `get_learnability_set` 的**随机海选**（`env.reset` 造 5000 random level）换成
**N 个 PCGRL generator**：每个 generator 是一个 narrow tile-MDP 的 PPO agent，被一个**异质 estimator
信号**（difficulty / PVL / CENIE）当 reward 驱动，造出 level → 转成 jaxnav `EnvInstance` → 注入固定共享
buffer。打分层（auction + top-K）不变，收敛定理仍罩打分层。

---

## 1. 复用的现成零件（零或少改）

| 零件 | 来源 | 用法 |
|---|---|---|
| narrow tile-MDP | pcgrl-jax `envs/reps/narrow.py` `NarrowRepresentation` | 逐 tile 编辑；action=Discrete(n_editable_tiles)；obs=egocentric patch(rf_shape=31×31) |
| binary problem | pcgrl-jax `envs/probs/binary.py` `BinaryProblem` | tile_enum={BORDER=0,EMPTY=1,WALL=2}；**原生 reward(diameter/n_regions)丢弃** |
| PCGRL env | pcgrl-jax `envs/pcgrl_env.py` `PCGRLEnv` | reset_env→N步 step_env→成品 env_state.env_map |
| PPO update | SFL `jaxnav_sfl.py` `update_actor_critic_rnn` / loss | generator 独立 TrainState 复用同一 PPO 逻辑 |
| estimator 信号 | SFL `jaxnav_sfl.py:355` scan 已产 (difficulty,pvl,cenie) per-level | generator reward 直接接此，不重写信号 |
| buffer 注入点 | SFL `jaxnav_sfl.py:886` `get_learnability_set` 产 `instances` | 新增 `get_generator_set` 产同型 `instances` |

---

## 2. map 表示桥接（PCGRL ↔ jaxnav）

- PCGRL `env_map`: (H,W) int，值 ∈ {BORDER=0, EMPTY=1, WALL=2}。
- jaxnav `map_data`: (H,W) int，值 ∈ {0=自由, 1=墙}。
- **映射（一行）**：`jaxnav_map = jnp.where(pcgrl_map == BinaryTiles.WALL, 1, 0)`（BORDER 也映射成 1=墙，因 jaxnav 边界即墙）。
- **起点/终点**：PCGRL binary 不产 agent/goal（tile_enum 只有 {BORDER,EMPTY,WALL}，无 agent/goal tile）。
  generator 只造**障碍布局**；agent_pos/goal_pos/theta 在造完后补。
  → **V1 复用 jaxnav 自己的起终点采样器**（比手写低方差补更优）：jaxnav `grid_map.py` 的 `barn_test_case`
    逻辑 = sample_map → smooth → **inflate（按 agent 半径膨胀障碍）→ 非膨胀格随机选 start → flood-fill 连通区里
    选 goal → 校验可达（while_loop 重采直到 valid）**。这后半段（inflate+连通区采样）**与 map 解耦**，接受任意
    map_data。故：generator 造 map_data → 喂进 barn_test_case「放起终点」那半段 → 拿到合法可达的 start/goal。
    **不自己手写连续坐标/碰撞校验，完全复用 jaxnav 已校验的逻辑**，R5 几乎免费解决。
  → ⚠ **代价（须标注）**：一张障碍图的难度同时取决于起终点位置。若起终点随机/方差大，generator 收到的
    estimator reward（尤其 difficulty/learnability 线）会混入「起终点随机性」这部分它控制不了的噪声，削弱
    「障碍布局 → reward」的因果清晰度、加重 R1 的稀疏信用分配。低方差规则补正是为压住这个噪声。
  → 让 generator 自己造起终点（加 AGENT/GOAL tile + 改动作空间 + 唯一性约束）列 future work。

---

## 3. reward 时序：terminal（episode 末）而非 PCGRL 原生 dense

**核心抉择**：PCGRL 原生每步从 `prob.step` 算几何 reward（diameter/n_regions）。该 reward 衡量的是
**地图静态几何美观**（通路够长、是否单连通），**与 student 学到哪了无关** —— 它在回答另一个问题（怎么造
好看的迷宫），不是方案B 要的「对 student 的学习价值」。**故几何 reward 不作主信号**（用它作主 = 放弃方案B
全部卖点：异质 estimator 驱动 / learnability 复活 / 收敛叙事）。改用 estimator 信号当 **terminal reward**：

```
generator episode（max_board_scans×H×W 步逐 tile 编辑）
  → 成品 env_map
  → 转 jaxnav map_data + 随机起终点 → EnvInstance
  → 冻结的 student 在该 level 上 rollout（复用 get_learnability_set 的 rollout+信号计算路径）
  → estimator-j 分数 = 该 generator 的 terminal reward（中间步 reward=0）
  → PPO 用此 terminal reward 反传更新 generator-j
```

符合方案B §3「生成完一个 level → 跑 student rollout 测 → reward」。中间步稀疏（reward=0）可接受：
narrow episode 不长（jaxnav 用小图），且 terminal reward 是 level 级信号，本就该 episode 末给。

> ⚠ 风险点 R1：稀疏 terminal reward + 长 episode → 信用分配难、REINFORCE 方差大。缓解：(a) 小图先验证；
> (b) 若方差大，把 PCGRL 原生几何 reward 作**可选 dense shaping 辅助项**加回：
>    `reward = estimator信号(主, terminal, 定方向) + λ·几何reward(辅, dense, 帮收敛)`。
>    几何项与最终学习价值正相关但不等价，只当配角帮 generator 在稀疏主信号下别乱走；主方向仍由 estimator 定。
>    **先用纯 estimator reward 跑**，实测方差大/学不动再加（避免过早引入 λ 超参 + 几何项带偏主信号的新问题）。

---

## 4. N 个 generator 各被异质 estimator 驱动（防 mode collapse）

- generator-A reward = difficulty-match `-(p-0.5)²`（造"刚好能学"）
- generator-B reward = CENIE 反密度（造没去过的，exploration 主力）
- generator-C reward = PVL（regret 系，造高 value-loss）
- N 个各自独立 TrainState/params/optimizer，**reward 用不同 estimator → 自然分到 buffer 不同区域**（§2 §37）。
- 各 generator 造的 level **拼起来**进 buffer（N×batch → 替代原 5000 随机海选规模的一部分或全部）。

---

## 5. 交替训练循环（已拍板，与收敛定理"固定 buffer"兼容）

嵌进 SFL `train_and_eval_step`（line 882）：

```
每个 outer round:
  阶段G（冻 student，更新 generator）:
    for g in 1..N:
      generator-g rollout 造 batch level → student rollout 算 estimator-g reward → PPO 更新 generator-g
    汇总 N×batch level → instances（替代 get_learnability_set 的随机海选输出）
  阶段S（冻 generator，更新 student）:
    instances 注入 buffer → lax.scan(train_step, EVAL_FREQ)  ← 原样不动
```

- 阶段G 内 student 冻结 → 对 generator 而言 student 固定；阶段S 内 buffer 固定 → 对 student 而言 buffer 外部固定。
  两段都满足各自的"对方固定"，与 proof_skeleton §8.4「buffer 刷新当外部」一致。
- **不做 min-max 对抗**：generator reward = estimator 信号（非 regret 对抗），不与 student 算零和。
  借 PCGRL/tile-MDP 的**关卡构造机制**，不借 PAIRED 的**对抗目标**（见 memory [[generator-teacher-breaks-convergence]]）。

---

## 6. 开关与回退

- `GENERATOR_INJECTION`（默认 false）：false→走原随机海选（零回归）；true→走 get_generator_set。
- N、各 generator 的 estimator 绑定、generator PPO 超参 走 config。
- 回退：关 GENERATOR_INJECTION 即回到当前已验证的 auction-on-random-pool 主线。

---

## 7. 工程风险清单（诚实标注）

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 稀疏 terminal reward 方差大 | 小图先验；必要时加 PCGRL 几何 reward 作 dense shaping |
| R2 | pcgrl-jax 依赖（gymnax 版本等）与 sfl env 冲突 | 只搬 narrow/binary/env 几个文件进 sfl，不整包安装；逐个 import 自检 |
| R3 | generator episode = max_board_scans×H×W 步，长 rollout 显存 | egocentric patch(31×31) 已降单步显存；小图(如 13×13)起步；max_board_scans 调小 |
| R4 | map_shape 必须 generator 与 jaxnav 一致 | 读 jaxnav-sfl.yaml 的地图尺寸，generator map_shape 对齐 |
| R5 | 起终点不连通 → 不可解地图（generator 会主动 reward-hack 放大） | **开 `valid_path_check=True`**：goal 只从 start 的连通块采（component_mask_with_pos 真连通），根除不可解。已在 `place_start_goal_on_map` 实现并单测验证（T5：隔断图上起终点 100% 同连通块）。⚠ generator 注入必须用 valid_path_check=True 的 env（False 是 jaxnav 原生默认，会产不可解图）。 |

---

## 8. 落地顺序（小步快跑，每步可单测）

1. `pcgrl_generator.py`：搬 narrow+binary+env，封装 `make_generator(map_shape, estimator_id)` →
   `gen_rollout(rng, gen_params, student_params, gmm) -> (levels_EnvInstance, gen_grad/loss)`。先单 generator。
2. 单元：本机/Oscar 跑 `gen_rollout` 出一批合法 jaxnav EnvInstance（map_data∈{0,1}、起终点合法可达）。
3. 接 `get_generator_set` 进 jaxnav_sfl.py，GENERATOR_INJECTION 开关；冒烟 1 generator 跑通一个 outer round。
4. 扩到 N=3，各绑 difficulty/CENIE/PVL；看 buffer 里 level 是否分化（estimator 分布差异）。
5. 对照实验：generator 注入 vs 随机海选，看 learnability estimator 方差是否复活（§10.2 地基2）。
