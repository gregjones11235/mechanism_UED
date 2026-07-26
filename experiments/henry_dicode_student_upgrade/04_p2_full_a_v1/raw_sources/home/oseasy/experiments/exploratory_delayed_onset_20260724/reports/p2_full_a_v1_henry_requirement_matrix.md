# P2-Full-A-v1 × Henry 原始 P2 需求矩阵（只读差距审计）

- 阶段：`POSTHOC_REPLAY_ATTRIBUTION_AND_HENRY_GAP_AUDIT`（§四 交付物）
- 性质：**只读**。不训练、不改算法/超参/阈值、不改网络结构、不 commit optimizer、不第二 seed。
- 结论冻结标志：`P2_FULL_A_V1_ENGINEERING_PASS=true`、`P2_FULL_A_V1_PERFORMANCE_PASS=false`、
  `NO_DELAYED_ONSET_WITHIN_98304=true`、`EXPLORATORY_POSITIVE_SIGNAL=false`、`FORMAL_HENRY_P2_TESTED=false`。
- 本矩阵只回答一件事：**Henry 原计划 P2 的每一条要求，当前 P2-Full-A-v1 到底实现了没有、缺在哪、
  以及对"负结果能解释到什么范围"施加什么限制。**

---

## 0. 冻结的终极目标（来自 Henry_work/尝试指导方案.md §0/§1.1/§4，文件使用说明.md §0）

> 做出一个**强于健康 PPO-GTrXL（ckpt17500）**的 Student，在 **Official FULL、floor0 自然出生**口径下，
> **可复现地**同时突破 `ENTER_SEWERS` 与 `DEFEAT_KOBOLD` 的零率。

- `S4_dark`（spawn_floor=2，floor2→3 暗搜索滤镜）**只是瓶颈定位探针**，不是终极目标本身；它隔离的是
  「8怪走廊」第二道门——清8怪解锁后的**暗搜索找梯**。
- 瓶颈链（Henry 实测定案）：floor0→1(~89% 已解) → floor1→2(20–28% 跨方法平台) →
  **floor2→3(organic 1–3%，主攻环=8怪走廊)** → floor3 kobold 战(~33%，R0 脚手架证过 87.5%)。
- `P(floor2→3) = P(清8怪解锁) × P(解锁后暗搜索找到梯)`。第一道门（战斗）可学；**第二道门（暗搜索）
  被四次独立实验证明现架构下练不动**——这正是 Henry 提出 P0/P1/P2 的原因。
- 暗搜索墙双因子（§1.5）：**表示能力**（128 窗外历史物理丢失，无法维持航位推算）× **学习信号**
  （找到梯前奖励为零 + 50% 运气赢局虹吸梯度，系统性搜索从未被奖励）。
- 健康基座 = `base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500`（pre-P0′ 事故、held-out mines 23.9%），
  代码基座 = `dicode_v7fix58_armB`（fix5.5→5.8 网络结构一行未动，新码加载老权重无损）。
- **γ/λ 严禁改动（0.999/0.8）**；评测只认 fresh-world 零样本（512×4096）；S4_dark 判决线 ≥ +8pp vs 对照。

---

## 1. Henry 原始 P2 的明确要求（§4 P2 逐字 + §1.3/§1.5 支撑）

Henry §4 P2 原文：

> 「**整轨迹 replay + 长上下文 transformer + hindsight 重标注**（失败轨迹按实际达成的子目标重标注为
> 成功样本），**off-policy TD 在全轨迹上下文上学**。参考 AMAGO (ICLR 2024)……同时换掉记忆（整局上下文）
> 与信用分配（跨 1500 步的投资链不再被 128 步截断切碎），还把稀有幸运赢局重复利用几十次。」

拆解为可核验需求项：

| 编号 | Henry 需求 | 出处 |
|---|---|---|
| **R1** | 整轨迹（whole-episode）replay：以完整 episode 为单位存储与重放 | §4 P2、§1.3 |
| **R2** | **长上下文 transformer 策略结构**：Actor 在**整局/全轨迹上下文**上输出动作；记忆贯穿整局，**无 128 步窗口悬崖**（AMAGO 式全轨迹自注意力 / 贯穿整局的循环态） | §4 P2、§1.3 核心矛盾、§1.5 表示因子、§1.4 AMAGO |
| **R3** | hindsight 重标注：失败轨迹按**实际达成的子目标**重标注为成功样本 | §4 P2 |
| **R4** | off-policy 的 Actor/Value 学习，且作用在**长轨迹（全轨迹上下文）**上 | §4 P2、§1.5 |
| **R5** | 解决「跨 128 步截断的信用分配」：输家 episode 中位 **643 步 >> 128 记忆窗**，跨段搜索历史信用不可学 → P2 必须让 >128 步的长程依赖进入策略表示 | §1.3 核心矛盾、§1.5、§1.2 r16 取证（失败签名=无航位推算/搜索历史整合） |
| **R6** | 稀有幸运赢局重复利用几十次（replay 复用） | §4 P2 |
| **R7** | 归因纪律：P2 换掉的自变量太多，其读数**不参与 P0/P1 单因子归因**，定位为独立「上限侦察臂」；判决协议与其他臂共用（S4_dark ≥ +8pp + 行为取证：背向漂移消失） | §4 P2 归因纪律段 |

---

## 2. 当前 P2-Full-A-v1 的实现（源码逐条证据，只读）

实现位于 `/home/oseasy/experiments/p2_full_20260723/src/`（模块未改动，仅作证据引用）：
`replay_buffer.py / hindsight.py / awr.py / vtrace.py / memory_anchor.py / full_p2_core.py /
full_p2_learner.py / run_p2_full_levelB.py`。网络基座 = `dicode_v7fix58_armB` 的原生
`ActorCriticTransformer`（GTrXL，`compat_init.NET_DIMS`：window_mem=128、num_layers=2、embed/qkv=256、
num_heads=8、gating=True、gating_bias=2.0、action_dim=43、obs_dim=8335）。

训练协议（`run_p2_full_levelB.py` 冻结常量）：`NUM_ENVS=16`、`ROLLOUT_STEPS=128`、`K_BATCH=4`、
`L_SEQ=129`（损失窗长）、`lr=2e-5`、`adam_eps=1e-5`、`γ=0.999`、`seed=42`、`ReplayBuffer(capacity=64)`。

| 编号 | 当前实现 | 实现证据（源码/常量） | 完全满足? | 未满足部分 |
|---|---|---|---|---|
| **R1** 整轨迹 replay | **已实现**。`collect_rollout` 用 `PendingEpisodeBuffers` 跨 rollout 持久化每条 episode，done 时产出含稀疏锚点的完整 `Trajectory`；`ReplayBuffer(capacity=64)` 以整条 episode 存储；`Trajectory.dones[-1]` 必须为 True（拒收截断片段，Gate 4） | `full_p2_core.collect_rollout` L110-133；`replay_buffer.Trajectory/insert` L171-189；`MIN_SEQUENCE_LENGTH=129` | **是（存储/重放层面）** | 存储的是整条 episode，但**喂给网络的只是 129 步损失窗**（见 R2/R5），整条轨迹并未作为上下文进入前向 |
| **R2** 长上下文 transformer | **未实现**。Actor 仍是**原生 GTrXL，window_mem=128**。损失区用 `jax.lax.scan` 前向 129 步，但每步可访问的记忆**结构性地≤128**（GTrXL 窗口 + 从最近锚点≤128 步 burn-in 重建）。没有 AMAGO 式全轨迹自注意力，也没有贯穿整局的循环态（S5/RNN） | `compat_init.NET_DIMS window_mem=128`；`full_p2_learner._scan_lax`/`reconstruct_batch`（burn-in≤128，`stop_gradient`）；`memory_anchor.reconstruct_state`（gap≤128）；`ANCHOR_INTERVAL=128` | **否** | **Actor 的显式长上下文结构完全未替换**；策略在任意时刻的条件上下文上限 = 128 步记忆窗，与 Henry「整局上下文 / 无窗口悬崖」要求不符 |
| **R3** hindsight 重标注 | **已实现**。`relabel_sample/relabel_trajectory` 将 obs 尾 67 维替换为**实际达成**子目标的 embedding，并按该目标重算 per-step reward；AWR 路径用 relabeled obs/reward。Gate 5（只 relabel 已达成目标）/ Gate 6（拒绝伪造目标，抛 ValueError）保持未弱化 | `hindsight.relabel_sample` L101-132、`_select_goal_index` L50-66、`apply_goal_conditioning`、`recompute_reward_for_goal`；`run_p2_full_levelB` L304-308（`relabel_sample(s)`，goal_index=None→最小达成目标） | **是** | relabel 目标是**窗内最小达成子目标**（常为 wood/stone 等早期成就），并非必然 DEFEAT_KOBOLD；这是「按实际达成子目标重标注」的合法实现，但对终极任务的定向性有限（详见 §七 分析） |
| **R4** off-policy Actor/Value（长轨迹上） | **部分实现**。off-policy 机制完备：**原目标路径** V-trace Actor+Value（`rho=pi/mu` 重要性修正，`vtrace.py`）；**relabeled 路径** hindsight AWR Actor（加权 BC+KL，无跨目标 IS 比，`awr.py`）+ relabeled-return Value。组合损失 `E=0.5·(A+0.5B)+0.5·(C+0.5D)`，事务性单步 KL 门（≤0.05）+ critic-only 独立 commit | `full_p2_learner.compute_loss` L152-206、`full_p2_update` L346-453；`vtrace.py`/`awr.py` 全文 | **机制是；"长轨迹上下文"否** | off-policy Actor/Value **确实存在且正确**，但其作用域是 **129 步损失窗 + ≤128 步记忆**，**不是 Henry 要求的"全轨迹上下文"**。即"off-policy"成立，"在长轨迹上下文上学"不成立 |
| **R5** 跨 >128 步信用分配 | **未实现**。128 窗悬崖原样保留；replay 序列长 129 仅比窗口大 1。实测 buffer 内 episode 最长 **1435/2048/2771/4096** 步（@24576/49152/73728/98304，manifest），远超 128，但这些长程依赖**从未作为上下文进入网络** | manifest `replay_longest_trajectory`；`L_SEQ=129`、`window_mem=128`；`memory_anchor`（窗口滚动覆盖 >128 历史） | **否** | 这正是 Henry「643 步输家 >> 128 窗」核心矛盾，**当前实现完全没有触及**；§八 长上下文审计直接检验此点 |
| **R6** 稀有赢局重复利用 | **部分实现**。整条 episode 入 buffer 后可被多次重采样（稀有 DK 赢局理论上可复用）；但 **FIFO capacity=64** 会驱逐旧轨迹，稀有高质量轨迹可能被永久驱逐（不可恢复，仅余计数器） | `ReplayBuffer.insert` L182-184（`pop(0)` FIFO 驱逐）；manifest `replay_buffer_size` 73728/98304 已=64（满）；`counters.trajectories_inserted` 记录累计插入 | **部分** | 复用发生在 **129 步窗**粒度而非整局上下文；FIFO 满后稀有赢局可能被驱逐（§七 量化 n_evicted 与赢局稀缺度） |
| **R7** 归因纪律 | **已遵守**。本审计与上游延迟涌现判定均把 P2-Full-A-v1 作为**独立上限侦察**处理，未将其负结果归因为 P0/P1 单因子无效；判决用 fresh-world 256 世界配对 + S4 类读数 + 行为面 | 上游 `eval_paired_256.py`（256×4096 fresh，配对 McNemar）；本审计 §九 只产出一个候选 | 是 | —— |

---

## 3. 必须明确的五条裁定（directive §四 强制项）

1. **replay / hindsight / off-policy Actor 链条确实已实现。**
   R1（整轨迹 replay）+ R3（hindsight 重标注）完整落地；R4 的 **off-policy Actor/Value 机制**
   （V-trace 原目标 + AWR relabeled，事务性 KL 门）正确实现。`P2_FULL_A_V1_ENGINEERING_PASS=true`
   指的是这条链在工程上跑通、无 NaN、守恒、KL 门生效、可精确续训——**这是真的**。

2. **Actor 的显式长上下文结构仍未替换。**
   R2 未实现：Actor 仍是原生 GTrXL，**没有任何 AMAGO 式全轨迹自注意力或贯穿整局的循环态**。
   这是 Henry P2 与当前实现之间**最本质的结构性缺口**。

3. **GTrXL 仍 window_mem=128。**
   `compat_init.NET_DIMS["window_mem"]=128` 与训练/评估一致；本审计**不修改**网络结构。
   策略在任意时刻的条件记忆上限 = 128 步。

4. **更长的序列 replay ≠ Actor 拥有整局显式上下文。**
   把整条 episode（最长 4096 步）存进 replay、并从中切 129 步窗做 off-policy 更新，**并不等于**
   Actor 在前向时"看见"了整局。网络每次前向只以 ≤128 步记忆窗为条件；窗外历史被 GTrXL 滚动覆盖、
   物理丢失。**"序列变长的 replay" 与 "Actor 的整局上下文" 是两件不同的事**——当前只做到了前者。

5. **当前负结果只针对 P2-Full-A-v1。**
   `P2_FULL_A_V1_PERFORMANCE_PASS=false` / `NO_DELAYED_ONSET_WITHIN_98304=true` /
   `EXPLORATORY_POSITIVE_SIGNAL=false` 描述的是**这个特定实现**（整轨迹 replay + V-trace + hindsight AWR
   + 稀疏锚点重建 + **原生 GTrXL window_mem=128**）在 98,304 步内不优于 Control。
   **`FORMAL_HENRY_P2_TESTED=false`**：由于 R2/R5（长上下文策略结构）根本未实现，
   **本结果不能解释为"Henry 原始 P2 / AMAGO 式整轨迹长上下文路线失败"**。它只否证了
   "在保留 128 窗 GTrXL 的前提下，仅补 replay+hindsight+off-policy 即可在 98,304 步内带来收益"这一
   **弱化版本**。

---

## 4. 对"负结果解释范围"的限制（不可外推清单）

- ❌ 不得把当前负结果外推为「AMAGO / Henry P2 路线无效」——**长上下文结构这一最大自变量根本没动**。
- ❌ 不得外推为「hindsight / off-policy replay 无效」——它们在 128 窗约束下运行，**约束本身未解除**。
- ❌ 不得外推为「稀有赢局复用无效」——复用粒度是 129 窗、且 FIFO 可能已驱逐稀有赢局（待 §七 量化）。
- ✅ 可以否定：「保持 window_mem=128 的 GTrXL，仅靠整轨迹 replay + V-trace + hindsight AWR，
  能在 98,304 步内对健康 Control 产生可复现收益。」
- ✅ 可以判定：若 §五–§八 未发现单一 replay 梯度异常，则负结果**更一致于** Henry 早已指出的
  **表示瓶颈（128 窗外历史丢失）**仍未被解决——指向真正的长上下文**结构**候选（§九 候选 D），
  而非 replay 优化细节。

---

## 5. 证据路径与 SHA256（只读绑定）

| step | 路径 | params_sha256 | update_count | replay_size | longest_traj |
|---|---|---|---|---|---|
| 24576 | `p2_full_20260723/checkpoints/p2_full_levelB_24576_20260724/24576` | `bd08422042788f6322b76730…` `…25a10d28` | 11 | 24 | 1435 |
| 49152 | `exploratory_delayed_onset_20260724/p2_resume_RUN1/ckpt/49152` | `6b2a4fc5035a1c86…461b232c` | 23 | 45 | 2048 |
| 73728 | `…/p2_resume_RUN1/ckpt/73728` | `2d93352f238ec447…0cfe9d84` | 35 | 64(FIFO满) | 2771 |
| 98304 | `…/p2_resume_RUN1/ckpt/98304` | `67689592cd10f6c9…02080a67` | 47 | 64(FIFO满) | 4096 |

- 源码证据（未改动，只读引用）：`p2_full_20260723/src/{replay_buffer,hindsight,awr,vtrace,memory_anchor,full_p2_core,full_p2_learner,run_p2_full_levelB,compat_init}.py`
- 网络基座：`/home/oseasy/incoming/henry_work_20260721T105300/extracted/Henry_work/code/dicode_v7fix58_armB/src/dicode/network.py`
- Henry 需求出处：`Henry_work/尝试指导方案.md`（§0/§1.1/§1.3/§1.5/§4 P2/§5）、`Henry_work/文件使用说明.md`（§0/§2/§3）
