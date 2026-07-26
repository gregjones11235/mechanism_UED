# P7-EGOMAP-WAVE1 设计文档（GPU1）

**标签**: P7-EGOMAP-WAVE1 / 局内显式空间记忆 / 自我中心 / 无特权
**总监**: A（仅 GPU0+GPU1；GPU0 POSTHOC 已交付；P1-NavAux 归队友、禁触）
**目标**: 在健康 GTrXL-PPO（ckpt17500）上增加由 Student **自身观测+动作**构建的局内显式空间记忆（EgoMap），帮助突破 ENTER_SEWERS / DEFEAT_KOBOLD 零率与长程探索。

---

## 1. 接地结论（已读源码核验）

### 1.1 Obs 格式（8335 维平铺向量；`craftax/craftax/renderer.py::render_craftax_symbolic` + 任务嵌入拼接）

三段布局（已通过算术核验 8217+42+76=8335）：

| 段 | 偏移 | 维 | 内容 |
|---|---|---|---|
| 空间局部视图 | `[0:8217]` | 8217 | reshape `(9,11,83)`，OBS_DIM=(9,11)，以玩家为中心、**世界轴对齐** |
| renderer 标量 | `[8217:8259]` | 42 | inventory(16)+potions(1)+intrinsics(9)+**direction one-hot(4)**+armour(1)+armour_ench(3)+special(8) |
| 任务嵌入 | `[8259:8335]` | 76 | `ConcatEmbeddingWrapper` 拼接的固定任务向量（**非空间，不作地图来源**） |

**空间块 83 通道（`[0:8217]` reshape (9,11,83)）**：
- 通道 `[0:37]` = **BlockType one-hot（地形）**。37 类。可通行（passable）：GRASS(2)/PATH(7)/SAND(13)/PLANT(15)/RIPE_PLANT(16)/FIRE_GRASS(25)/ICE_GRASS(26)/GRAVEL(27)。固体障碍（solid）：WATER(3)/STONE(4)/TREE(5)/WOOD(6)/WALL(17)/WALL_MOSS(19)/STALAGMITE(20)/LAVA(14)/CHEST(23)/FOUNTAIN(24)/各类 ENCHANTMENT_TABLE/GRAVE/NECROMANCER 等。`OUT_OF_BOUNDS(1)`=地图边界。`INVALID(0)`/`DARKNESS(18)` 视情况。
- 通道 `[37:42]` = **ItemType one-hot（物品/楼梯）**。`NONE(0)/TORCH(1)=38/LADDER_DOWN(2)=39/LADDER_UP(3)=40/LADDER_DOWN_BLOCKED(4)=41`。**楼梯是 ITEM，不是 Block！** 火炬=通道38。
- 通道 `[42:82]` = mob map（5 类×8 型=40）。
- 通道 `[82]` = **light/可见性掩码**（1=被照亮/观察到）。

**关键：所有空间通道都乘以 `light_map_view`（L125 `all_map = all_map * light_map_view`）→ 只有被观察（照亮）的格子非零，暗格清零。** 这天然满足「地图只记录 Student 实际获得的信息」。

### 1.2 玩家状态可从 obs 提取（无特权）
- **朝向**：`direction one-hot` 在 `[8243:8247]`（= `one_hot(player_direction-1, 4)`）。
- **楼层**：`special_values[5] = player_level/10.0` 在 `[8256]`（special 段起于 8251，索引5 → 8251+5=8256）。`player_level = round(obs[8256]*10)`。
- **obs 不含绝对 player_position** → 必须用**动作里程计**累积位置。

### 1.3 动作语义（`game_logic.py::move_player` L2021；43 动作）
```
proposed = pos + DIRECTIONS[action]        # DIRECTIONS[1]=(0,-1) [2]=(0,1) [3]=(-1,0) [4]=(1,0)；其余动作=(0,0)
valid    = in_bounds & not_in_mob & not_colliding  (| god_mode)
pos      = pos + valid * DIRECTIONS[action]        # 阻挡 → 实际不移动
facing   = action  (当 action∈{1,2,3,4})           # 移动动作设置朝向=动作值
```
- 移动动作 **LEFT=1/RIGHT=2/UP=3/DOWN=4** 是**绝对世界方向**（注意 Action 名与 DIRECTIONS 索引语义不直接同名，按上表为准：action1=(0,-1)向上、action2=(0,1)向下、action3=(-1,0)向左、action4=(1,0)向右）。
- 其余动作（NOOP=0/DO=5/craft/PLACE/DESCEND=18/ASCEND=19…）不移动、不改朝向。
- **楼梯换层**（L2431-2471）：站在 LADDER 上执行 DESCEND(18)/ASCEND(19) → `player_level ± 1`，位置跳到对应梯子。换层是楼层语义边界。

### 1.4 基座代码（嫁接目标，只读参考、不改）
- 网络：`dicode_v7fix58_armB/src/dicode/network.py::ActorCriticTransformer`（setup 建 `transformer`=GTrXL trunk + actor 头 actor_ln1/ln2/out + critic 头 critic_ln1/ln2/out；trunk 输出 `x` 在 model_forward_* 内部，返回 (pi, value, memory_out)）。obs 编码在 `transformer/` 内（posthoc 计 encoder=2134016 / trunk=2497536 / actor_head=142635 / value_head=131841）。
- PPO 主循环：`ppo_tr.py`（make_train→train→_update_step 外层 PPO scan→_env_step 内层 rollout scan；L353-356 done 重置 memories_mask；L375 model_forward_eval；L390 env.step 后 env_state 可得；_calculate_gae 含 value-target clip）。
- env wrapper：`wrappers_cl.py`（AutoReplayWrapper/CL wrapper，task_embeddings 经 training.py:134 `normalized_table` 传入）。
- **基座 checkpoint**：`/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500`。
- craftax 包：`/home/oseasy/miniconda3/envs/dicode310/lib/python3.10/site-packages/craftax/craftax/`。

---

## 2. EgoMap 架构

### 2.1 总体结构（满足「当前观测 CNN + 原 GTrXL + EgoMap 编码器 → 原 Actor/Value」）
```
obs[8335] ─┬─> 原 obs 编码 + GTrXL trunk ──────────────> x_trunk  ─┐
           │                                                       ├─> fuse ─> 原 actor 头 -> pi
           └─> [空间块解析 (9,11,83)] ─> EgoMap 更新(状态) ─> map  ─┤        原 critic 头 -> value
                                                                   └─> EgoMap 编码器 ─> x_ego ─┘
```
- **不改动原 Actor/Value 头的结构**；EgoMap 编码器输出 `x_ego` 与原 trunk 输出 `x_trunk` **融合**（concat 后过一个新增融合 Dense，或相加——默认 concat+Dense 到 hidden_layers），再喂给**原** actor/critic 头。
- **feature-off 开关**：`egomap_enabled: bool`。关闭时 `x_ego` 恒为 0 向量且融合层退化为恒等（融合权重初始化为使 off 时逐位等于原模型——见 Gate1）。

### 2.2 EgoMap 状态（per-vector-env，世界锚定累积）
- **每楼层一张地图**：`map_bank[num_floors=9, H, W, C]`（默认 H=W=32；C=9 通道，见下）。episode 起点位置锚定在地图中心 (H//2, W//2)；用动作里程计维护 `ego_offset`（当前估计世界位移，整数）。写入/读取均以 `ego_offset` 为中心。越界用 clamp（记录 `overflow` 计数用于诊断，不泄漏）。
- **9 个通道**（全部仅来自观测/动作）：
  1. `ever_observed`：该格曾被观察（light 通道曾=1）→ OR 累积。
  2. `visit_count`：玩家曾位于该格的次数（归一化，如 `1 - exp(-n/τ)`）。
  3. `last_visit_recency`：距最近一次访问的步数（归一化/封顶）。
  4. `passable`：观察到的可通行地形（BlockType∈passable 集）。
  5. `obstacle`：观察到的固体障碍/边界（BlockType∈solid 集 或 OUT_OF_BOUNDS）。
  6. `torch`：观察到火炬（Item 通道38）。
  7. `stair_down`：观察到下行楼梯（Item 通道39 或 41=DOWN_BLOCKED）。
  8. `stair_up`：观察到上行楼梯（Item 通道40）。
  9. `frontier`（计算量）：已观察-可通行 且 邻接未观察格 → 未探索边界。
- **楼层上下文**：读取当前 `player_level` 对应那张地图（`map_bank[player_level]`）；楼层变化（由 obs[8256] 推得）即切换到对应楼层地图——**不清空**任何楼层（每楼层独立记忆）。

### 2.3 EgoMap 更新规则（每 env step，纯 JAX、确定性）
每步输入：当前 obs 空间块 `(9,11,83)`、上一步动作 `a`、当前 `ego_offset`、`map_bank`。
1. **里程计（动作 + 观测阻挡修正，无特权）**：
   - 若 `a∈{1,2,3,4}`：`intended = DIRECTIONS[a]`；取当前观察 patch 中「移动方向相邻格」（中心 + intended）的地形/mob：若该格 ∈ solid 集 或 OUT_OF_BOUNDS 或 有 mob（mob 通道非零）→ 判定阻挡 `Δ=(0,0)`；否则 `Δ=intended`。
   - 否则 `Δ=(0,0)`。
   - `ego_offset ← ego_offset + Δ`。
   - （阻挡修正只用当前观测中可见的相邻格——相邻格必在被照亮的玩家邻域内，故无特权信息。这是 SLAM-lite，纯由观测+动作构建。）
2. **写入**：把当前观察 patch（`ever_observed`=light通道；passable/obstacle/torch/stair 由 one-hot 通道解码）以 `ego_offset` 为中心 stamp 进 `map_bank[player_level]`。`ever_observed` 用 OR 累积；passable/obstacle/torch/stair 用「观察到则置位」覆盖。`visit_count`/`last_visit_recency` 在玩家当前格更新。
3. **frontier 重算**：对当前楼层地图，`frontier = passable_observed AND (任一4邻格未 ever_observed)`。
4. **读出**：以 `ego_offset` 为中心裁出 `(H,W,C)`（默认全图），作为 EgoMap 编码器输入。

### 2.4 地图状态生命周期（严格满足规格）
- **per-vector-env 独立**：map_bank/ego_offset 是 carry 状态，按 env 维 vmap，互不干扰。
- **rollout 边界不重置**：rollout 切分（128 步窗）处地图状态作为 carry 跨窗传递，**不清空**。
- **true done 才清空**：仅当 `info` 标记 true done（非 timeout 截断）时清零该 env 的 map_bank+ego_offset。需区分 true done vs truncation（沿用基座 done/timeout 语义；Gate3 验证）。
- **checkpoint 完整保存/恢复**：map_bank/ego_offset 纳入 train state pytree，随 checkpoint 一并 save/restore（Gate5 验证逐位 exact resume）。

### 2.5 EgoMap 编码器
- 小 CNN：`(H,W,9)` → 2-3 层 Conv（小核、少量通道）→ flatten → Dense → `x_ego`（维 = hidden_layers，与 trunk 输出维对齐）。
- **初始化使 feature-on 起步不破坏基座**：融合层中 `x_ego` 一侧权重初始化为 0（zero-init gate），保证训练初始 `x_ego` 贡献=0 → Gate6 迁移门（64-world SR 与 Baseline 差≤5pp）。

---

## 3. 六门（开跑前必须全过；任一失败停止）

| 门 | 定义 | 通过判据 |
|---|---|---|
| **G1 feature-off 逐位一致** | `egomap_enabled=False` 时前向 | Actor logits 与 Value 与原 ckpt17500 模型**逐位 bit-exact 相等**（同 obs/memory/mask，max abs diff = 0.0） |
| **G2 无地图信息泄漏** | (a) 地图通道只含曾被观察（light 掩蔽）的信息；(b) 把 obs 空间块全部置零（或仅保留非空间段）跑一段，EgoMap 不得「凭空」出现 passable/stair/torch；(c) 断言 EgoMap 编码器输入中**不含** obs 未提供的字段（无 sim 全图、无未观察楼梯坐标、无未来帧、无 env_state.player_position/map） | 黑暗/遮挡格 `ever_observed=0` 且 passable/obstacle/stair/torch 全 0；置零空间块后地图无新增结构；代码审计无特权读取 |
| **G3 map 更新/done-reset/隔离测试** | 单元测试：(a) 给定确定性假 obs+动作序列，地图累积符合 §2.3；(b) true done → 该 env 地图清零；timeout/truncation → 不清零；(c) 多 vector-env 并行，env i 的地图不受 env j 影响 | 三类断言全过 |
| **G4 4096 smoke 无数值异常** | feature-on 跑 4096 env steps | 无 NaN/Inf（loss/value/log_prob/map 值全 finite） |
| **G5 checkpoint exact resume** | 存 ckpt → 恢复 → 续跑 | 恢复后 map_bank/ego_offset/params/opt_state 逐位一致；续跑首步输出与不中断运行逐位一致 |
| **G6 迁移门（64-world）** | 新模块初始化后（未训练，仅 zero-init gate），64-world 评测 SR | 与 Baseline（原 ckpt17500）SR 差 **≤ 5pp**；否则**停止长训**（说明初始化破坏了已学行为） |

---

## 4. 训练与评测协议

### 4.1 共同配置
- 起点 ckpt17500；deterministic ops；seed=42；LR=2e-5；Adam eps=1e-5；γ=0.999；GAE λ=0.8；rollout=128；Stage4-native；total_steps=98304。
- **双跑**：Control（原 GTrXL-PPO，无 EgoMap）+ EgoMap，同 seed/同配置/同步数，公平对比。

### 4.2 Checkpoint 节点
- 保存 `0 / 4096 / 24576 / 49152 / 73728 / 98304`（Control 与 EgoMap 各 6 个）。

### 4.3 评测（冻结 256-world evaluator，vs 同 step Control）
- 指标：DK SR / floor3 到达 / conditional kill / death-timeout / episode length / unique cells / revisit ratio / coverage / paired CI / McNemar。

### 4.4 98304 正向门
- `SR ≥ Control + 8pp` **且** `floor3 ≥ Control` **且**（`coverage` 或 `revisit ratio` 至少一项改善）**且** 无数值/entropy 坍塌。
- 满足 → 标 `P7_EXPLORATORY_POSITIVE_SIGNAL`；否则 `NO_POSITIVE_SIGNAL`。
- **不得**自动跑第二 seed / 512 / Official FULL。

---

## 5. 边界与 provenance

- **GPU**：仅 GPU1（UUID `GPU-3c7a2864-755b-7045-b293-6f80e748283f`，index 1，启动前经 UUID 复核，当前空闲）。禁触 GPU0/2/3。
- **禁触**：D052、P0/P01、P1-NavAux（归队友）的代码/目录/产物；禁子 Agent/Task/background agent；只用普通 Python/bash/tmux/GPU 进程。
- **只读**：ckpt17500 与基座 `dicode_v7fix58_armB` 源码只读参考，不改原文件；EgoMap 代码全部新写在 `gpu1_p7_egomap/src/`。
- **不泄漏 secrets**：env 文件只读、绝不打印。
- **产物目录**：服务器 `/home/oseasy/experiments/student_upgrade_wave1_4gpu/gpu1_p7_egomap/`（src/checkpoints/outputs/reports/logs/tests 已建）。
- **基座 ckpt**：`…/Henry_work/base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500`。

## 6. 待实现工件（`gpu1_p7_egomap/src/`）
1. `egomap.py`：EgoMap 状态 pytree + `egomap_init/update/encode` + obs 解析（按 §1.1 偏移）+ 里程计（§2.3）+ feature-off 开关。
2. `network_egomap.py`：在 `ActorCriticTransformer` 外挂 EgoMap 编码器 + 融合层（zero-init gate），保留原 actor/critic 头。
3. `ppo_tr_egomap.py`：在 `ppo_tr.py` rollout/update 中携带 EgoMap carry（per-env、跨 rollout、true-done 清零、checkpoint 纳入）。
4. `launcher_p7.py`：Stage4-native 启动器（GPU1 UUID 绑定、seed42、双跑 Control/EgoMap、6 节点保存）。
5. `tests/test_egomap_gates.py`：G1-G6 的 CPU 测试。
6. `eval_p7.py`：冻结 256-world evaluator + 指标 + paired CI/McNemar。
