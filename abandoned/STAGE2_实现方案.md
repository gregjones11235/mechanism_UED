> # ⛔ 本方案已作废为主线（2026-06-19，⚑V5）——降级为 PAIRED fallback 探索记录
> #
> # **作废原因**：本方案走 PAIRED 范式（teacher = 真正的关卡生成器网络）。但 main.tex §3.1 + 收敛性分析
> # （[proof_skeleton §8](proof_skeleton_entropy_regularized_heterogeneous_UED.md)）确立：**teacher ≡ estimator（打分函数），
> # 不是会训练的生成器 agent**。把 teacher 实现成可训练生成器网络会**破坏收敛定理**（对手决策变量从单纯形 `y`
> # 变成非凸网络参数 `φ`，熵正则罩不住）——见 [UED实验步骤指导.md §0.4 / §4.3.2 ⚑V5](UED实验步骤指导.md)。
> #
> # **且"同质多 teacher Stage 2"本身已取消**：teacher≡estimator ⇒ "多 teacher 却仍同质"是矛盾的伪状态，
> # 不存在可隔离"多 teacher 本身有没有用"的中间阶段。Stage 1（单 teacher 复现）之后**直接进 Stage 3**。
> #
> # **当前主线实体** = `multi_estimator_plr` runner（**PLR 血统、`n_students=1`、无 teacher 生成器网络**，
> # 3 个异质 estimator PVL/SFL/CENIE 各出 bid → non-IC auction 聚合）。本方案的 PAIRED `multi_teacher`
> # runner **不再用于主线**。
> #
> # **本文档保留作什么**：① PAIRED 式"会训练的多 teacher 生成器网络"作为 **fallback 探索方向**的实现参考
> # （若 estimator-only 主方案性能不足）；② 但加它**必须走分层路线**（生成器只刷 buffer、auction 仍在固定
> # buffer 上跑课程）才能保住定理，**不可**直接 PAIRED 式对抗（见 §4.3.2 ⚑V5 决策建议）；③ AgentPop / vmap N teacher
> # 的工程调研仍有参考价值。下文内容**按 PAIRED-fallback 语境阅读，不是当前主线规划**。

---

# ~~Stage 2 实现方案：N 个真 Teacher 种群 + 三种聚合~~（PAIRED fallback 记录）

> ~~目标（来自 `UED实验步骤指导.md` §4.2）：隔离"多 teacher 本身有没有用"这一个变量。~~
> ~~N∈{1,2,4,8} 扫描，**所有 teacher 同质（全用 PVL 评分）**，三种聚合方式对照：~~
> ~~round-robin / bandit(EXP3 或 UCB) / argmax。预期是"持平"零结果，为 Stage 3 异质性铺路。~~
>
> **（PAIRED fallback 方向）**：走 PAIRED 范式——teacher 是真正的关卡生成器网络，N 个组成种群（复用 AgentPop）。
> 前置依赖：PAIRED 单 teacher 先跑通（脚本已修 `teacher_recurrent_hidden_dim=256`）。
> ⚠ 仅在 estimator-only 主方案失效、且走分层路线（agent 刷 buffer）时才启用——直接对抗会破收敛定理。

---

## 1. 现状：minimax 单 teacher 的数据流（已调研确认）

PAIRED runner 一个 update step 的顺序（`paired_runner.py:454-596`）：

```
1. teacher rollout：teacher 网络生成关卡          (L473-505)
   - reset_teacher → teacher 在 ued_env 里走 N 步"放墙"动作 → ued_state（即关卡）
   - 用 self.teacher_pop（当前 AgentPop(teacher, n_agents=1)）
2. student rollout：学生在 teacher 生成的关卡上解迷宫  (L507-536)
   - reset_student(ued_state) → 把关卡装进学生环境 → 学生 rollout
3. student PPO 更新                                (L538-549)
4. teacher 奖励 + PPO 更新                          (L551-569)
   - ued_score = compute_ued_scores(RELATIVE_REGRET, student_batch)  ← regret
   - set_final_reward(ued_rollout, ued_score) → teacher 把 regret 当奖励
   - teacher_pop.update()
```

关键抽象：
- **AgentPop**（`util/rl/agent_pop.py`）：vmap 管理 n_agents 个 agent 的 params/act/update，**已支持任意 N**，teacher 只是被硬编码成 `n_agents=1`（`paired_runner.py:113`）。
- **PVL 评分**（`util/rl/ued_scores.py:189-193`）：`mean(relu(advantage))`，同质、只依赖学生 advantage，与 teacher 身份无关 → N 个同质 teacher 天然共用同一评分函数。

---

## 2. 核心设计问题：N 个 teacher 怎么协作？

单 teacher 时"teacher 生成关卡 → 学生在上面训"是 1:1。N 个 teacher 时必须定义清楚**三件事**，这是整个方案的地基：

### 2.1 N 个 teacher 各自生成 N 批关卡（这步天然并行）
把 `teacher_pop` 的 `n_agents` 从 1 改成 N，`reset_teacher` + `_rollout` 自动 vmap 出 **N 套 ued_state**（N 批关卡）。AgentPop 已支持，改动小。

### 2.2 聚合：从 N 批关卡里选哪批喂给学生？（三种方式的本质）
这是 Stage 2 的核心变量。学生只有一个，每个 update step 只能在"某一批"关卡上训练。三种聚合 = 三种"选哪个 teacher 的关卡"策略：

| 聚合方式 | 选择规则 | 实现要点 |
|---|---|---|
| **round-robin**（控制组） | 第 t 步选 teacher `t % N`，轮流 | 确定性，无需打分即可选 |
| **argmax**（first-price） | 选 PVL 评分最高的那个 teacher 的关卡 | 需先对 N 批关卡各跑一次学生评估拿 PVL → 选最高 |
| **bandit**（EXP3/UCB） | 按历史 regret 维护每个 teacher 的权重，采样选 | 需跨 step 维护 bandit 状态（权重/计数） |

> ⚠ **关键实现难点**：argmax 和 bandit 都需要"先知道每个 teacher 的关卡有多好"才能选。
> 但"有多好"(PVL/regret) 要让学生在关卡上 rollout 才算得出。两种解法：
> - **(a) 先评估再训练**：N 批关卡各让学生 rollout 一次（只算 PVL、不更新），选中后再正式训练。代价：每步 N 次学生 rollout，N=8 时约 8 倍开销。
> - **(b) 用上一步的评分**：bandit 天然是"用历史反馈选臂"，不需要当步先评估全部 → 开销小。argmax 若也用"上一轮该 teacher 的 regret"近似，则同样省开销但有滞后。
>
> **建议**：round-robin 和 bandit 走 (b) 低开销路线；argmax 必须走 (a)（否则失去"挑当步最优"的语义）。这意味着 argmax 配置会显著更慢，扫 N 时要把这点算进墙钟。

### 2.3 N 个 teacher 各自怎么更新？
每个 teacher 只在"它的关卡被选中、且学生在上面训过"时才有真实 regret 信号。两种处理：
- **只更新被选中的 teacher**：语义最干净（选中才有学生反馈），但被选中的 teacher 才学习，其余 step 空转。
- **所有 teacher 都更新**：需给未选中 teacher 一个 regret 估计（如用它自己关卡的"预估难度"或 0）。偏离标准 PAIRED。

> **建议**：Stage 2 先用"只更新被选中 teacher"，对应 bandit/argmax 的标准语义（被选中→拿到 reward→更新）。round-robin 下每个 teacher 轮流被选中，长期都更新到。

---

## 3. 落地路线：新建 `MultiTeacherRunner`（不改 PAIRED）

调研结论一致推荐**新建 runner**，不改造 PAIRED（避免破坏已有单 teacher 逻辑，PAIRED 还要当 Stage 1 基线）。

### 3.1 改动清单（4 处）

| # | 文件 | 改动 | 量 |
|---|---|---|---|
| 1 | `runners/multi_teacher_runner.py`（**新建**） | 继承 PAIREDRunner，重写 `run()` 的 teacher 段：N teacher 生成 + 聚合选择 + 选中者更新 | 大头 |
| 2 | `runners/xp_runner.py:39-50` | `RUNNER_INFO` 加 `'multi_teacher': RunnerInfo(runner_cls=MultiTeacherRunner, is_ued=True)` | 1 行 |
| 3 | `arguments.py` `--train_runner` choices | 加 `'multi_teacher'`；新增 `--n_teachers`、`--teacher_aggregation` 参数 | ~15 行 |
| 4 | `config/configs/maze/` 新建 `multi_teacher.json` | 照 paired.json，加 `n_teachers:[1,2,4,8]`、`teacher_aggregation:[...]` 扫参 | 配置 |

### 3.2 MultiTeacherRunner.run() 伪代码（核心）

```python
def run(self, rng, ...):
    # 1. N 个 teacher 各生成一批关卡（teacher_pop.n_agents=N，自动 vmap）
    ued_rollout, ued_states_N, ... = self._rollout(self.teacher_pop, ...)
    #    ued_states_N: (N, n_parallel, ...) ← N 批关卡

    # 2. 聚合：选 teacher 索引 sel_idx ∈ [0,N)
    if agg == 'round_robin':
        sel_idx = self.n_updates % N
    elif agg == 'bandit':
        sel_idx = sample_from_bandit(self.bandit_state)      # 用历史 regret 权重
    elif agg == 'argmax':
        pvl_per_teacher = eval_each_teacher(ued_states_N)    # (a) 每批先评估一次
        sel_idx = argmax(pvl_per_teacher)

    # 3. 取选中 teacher 的关卡，学生在其上训练（复用 PAIRED 的 student 段）
    chosen_levels = tree_map(lambda x: x[sel_idx], ued_states_N)
    student_batch = rollout_and_update_student(chosen_levels)

    # 4. 算 regret，只更新被选中的 teacher
    regret = compute_ued_scores(PVL, student_batch)
    teacher_state = update_one_teacher(ued_train_state, sel_idx, regret)
    #    并更新 bandit 状态（若 bandit）：bandit_state.update(sel_idx, regret)
```

### 3.3 状态扩展
- `teacher_pop = AgentPop(teacher_agent, n_agents=N)` —— N 套 teacher 参数（vmap 维 = N）。
- **bandit 状态**（仅 bandit 用）：每个 teacher 一个权重/计数数组，需进 train_state 以便 checkpoint 续跑。
- checkpoint：`get_checkpoint_state` 要把 N teacher params + bandit 状态都存上（否则续跑丢 teacher 学习进度）。

---

## 4. 实现顺序（建议分阶段，每步可验证）

1. **前置**：确认 PAIRED 单 teacher 跑通（重训 + eval），作为 N=1 的退化基线对照。【待办：PAIRED 脚本已修 teacher_hidden_dim，待重训】
2. ✅ **骨架 + round-robin**（2026-06-14 完成，本地 CPU sanity 通过）：建 `runners/multi_teacher_runner.py`（继承 PAIRED）。N=2 验证 `selected_teacher=[0,1,0,1,0]` 完美交替。
3. ✅ **argmax**（同日完成）：vmap 并行评估 N 批关卡（非 Python 循环，避免 N 倍墙钟），选 PVL 最高。验证 `score_spread>0` 时挑高分 teacher、打平时选索引 0。
4. ✅ **EXP3 bandit**（同日完成）：log 权重 + γ 探索 + 重要性加权更新；bandit_state 随 runner_state 流转并进 checkpoint（reset/run/get_checkpoint_state 已重写）。验证概率随 regret 反馈从 0.5 分化、保留探索。
5. **下一步：集群端验证**——把改动同步回 GPU 集群，短训确认①三种聚合都能跑出 checkpoint；②**N=1 等价性**（MultiTeacherRunner N=1 ≈ PAIRED，验证新 runner 无偏差，见 §6 决策4）。
6. **扫描**：N∈{1,2,4,8} × 三种聚合，MiniGrid + BipedalWalker，出 scaling curve。
7. **分支判断**（§4.2.4）：MiniGrid 上 argmax 看不出 N>1 增益 → 按计划 pivot Craftax，不在 MiniGrid 反复调。

> 已改动文件（本地 minimax，待同步回 GPU 集群）：
> `runners/multi_teacher_runner.py`（新建）、`runners/__init__.py`、`runners/xp_runner.py`（注册）、
> `arguments.py`（train_runner 加 multi_teacher + 13 处 teacher 依赖扩展 + n_teachers/teacher_aggregation/exp3 参数）。
> 新增 CLI 参数：`--n_teachers`、`--teacher_aggregation={round_robin,argmax,exp3}`、`--exp3_gamma`、`--exp3_eta`。

---

## 5. 风险与开销提醒

- **argmax 的 N 倍学生 rollout 开销**：扫 N=8 时这一配置最慢，墙钟按 §sbatch 规约给 48h。
- **"持平"是预期结果**（§4.2.2）：N>1 ≈ ACCEL 不是 bug，是 Stage 2 的诊断价值，别因没涨点反复调参。
- **同质性必须严格**：N 个 teacher 必须同架构、同 PVL 评分，唯一变量是 N 和聚合方式，否则混入 Stage 3 的异质性变量。
- **本地只做 sanity**，正式扫描回 GPU 集群。改完同步 `src/` 回集群。

---

## 6. 设计决策（已拍板，2026-06-14）

1. **argmax 选择语义 = (a) 每步先评估 N 批关卡**：N 批关卡各让学生 rollout 一次（只算 PVL、不更新参数），选 PVL 最高的 teacher 关卡再正式训练。语义准确（挑当步真正最难的关卡），代价是 N=8 时约 8 倍学生 rollout 开销 → 该配置墙钟按 48h 给足。
2. **teacher 更新 = 只更新被选中的 teacher**：被选中→学生在其关卡训练→拿到真实 regret→更新；其余 teacher 该步不更新。贴合 bandit/argmax 标准语义。round-robin 下每个 teacher 轮流被选，长期都更新到。
3. **bandit 算法 = EXP3**：对抗性 bandit，不假设回报分布稳定，天然适配 UED 非平稳性（teacher 的最优性随学生变强而漂移）。对应文档点名的 prediction-with-expert-advice 路线（Hrithik 建议）。
   - 实现要点：每个 teacher 维护权重 `w_i`，按 `p_i ∝ exp(η·累积regret估计_i)` + 均匀探索 γ 采样选 teacher；用重要性加权 `regret/p_sel` 修正"只观察被选臂"的偏差。bandit 状态（权重数组）需进 train_state 以便 checkpoint 续跑。
   - UCB 不选：其"固定回报分布"假设被 UED 非平稳性破坏，对"teacher 已过时"反应迟钝。
4. **N=1 基线 = MultiTeacherRunner(N=1) 与 PAIRED 都跑**：两者结果应一致，用于验证新 runner 没引入偏差（退化到单 teacher 时必须等价于 PAIRED）。

> 三种聚合最终定为：**round-robin（控制组）/ EXP3（bandit）/ argmax（first-price）**。
