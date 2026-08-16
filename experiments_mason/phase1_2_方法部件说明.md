# Phase 1 & 2：方法核心部件（Skill Graph Scheduler + Preflight Gate）

> 作者：Mason　｜　日期：2026-07-07　｜　分支：`skill-preflight-ued_Mason`
> 概述：方案"Skill-Guided Generation with Learnability Preflight"的两个核心部件已实现并单元测试通过。
> 代码位置：`dicode_src/src/dicode/skill_preflight/`。两者均为薄封装，复用仓库现有机制，逻辑层已离线测通。

---

## 方法回顾

在小模型 code-level UED 中，生成既慢又贵，且并非每个生成的关卡都值得训练。本方案在 DiCode 基础上加两步：

1. **Skill Graph Scheduler**：用学生的 per-achievement 成功率定位"最浅的未掌握 tier"（学习前沿），
   引导生成聚焦到该 tier —— 让有限的生成用在刀刃上。
2. **Preflight Gate**：候选关卡进训练前，用当前 policy 做一次 cold rollout，
   基于学生的真实"部分进度信号"（而非二值成功率、也非 LLM 自评）判断该关卡"当前是否可学习"，
   过滤掉太易/太难的，避免浪费。

Phase 0 已验证地基假设（本地 14B 首次编译通过率 100%）。Phase 1/2 实现上述两个部件。

---

## Phase 1：Skill Graph Scheduler

文件：`skill_scheduler.py`（~93 行）　｜　测试：`tests/test_skill_scheduler.py`（8 个测试，全通过）

### 做什么
输入学生评估指标（`process_evaluation_metrics` 输出的 `skill_<name>` 成功率，0–100），
定位学习前沿 tier，输出该 tier 内未掌握、按最难在前排序的目标成就。

### 接口
- `pick_target(evaluation_metrics, *, threshold=0.60, max_target_achievements=6) -> SchedulerTarget`
  - 复用 `auction.craftax_achievements.profile_to_target_gap` 把 SR 转成 gap；
  - 复用 `tier_mastery` 算每层掌握度、`reachable_ceiling` 定位最浅未掌握 tier；
  - 返回 `SchedulerTarget(tier, target_achievements, tier_mastery, frontier_mastery, gap_type)`。
- `format_target_for_prompt(target) -> str`：把目标渲染成注入生成 prompt 的约束文字。

### 复用（未重新实现）
`auction.craftax_achievements` 的 `reachable_ceiling` / `tier_mastery` / `DEPTH_TIERS` /
`MASTERY_THRESHOLD_DEFAULT`，以及 `dicode.dreaming.auction_integration.profile_to_target_gap`。

### 测试覆盖
- 前沿 tier 判定：无掌握→tier1、掌握 tier1→tier2、tier1-2→tier3、tier1-3→tier4、全掌握→tier4；
- `gap_type`：前沿未掌握=advance，全掌握=consolidate；
- 目标成就落在正确 tier、最难在前、空输入默认 tier1、上限截断、prompt 格式化。

---

## Phase 2：Preflight Gate

文件：`preflight.py`（~269 行）　｜　测试：`tests/test_preflight.py`（19 个测试，全通过）

### 做什么
候选关卡进训练前判断"当前是否可学习"。分两层：**纯判据逻辑**（已测通）与
**接 rollout 的执行部分**（集成时接入，见下）。

### 接口
**纯逻辑（已单元测试）：**
- `partial_progress_signals(state) -> dict`：从单局最终 EnvState 抽物理子信号
  —— 到达楼层、是否造出基础工具、是否采到矿、死因、存活步数、解锁成就数，汇总为 `made_progress`。
  用真实 minicraftax EnvState/Inventory 字段（`player_level` / `player_health` / `inventory.pickaxe` 等）。
- `infer_death_reason(health, food, drink, energy) -> str`：饿死/渴死/累死/被打死/存活。
- `route(sr, any_partial_progress, *, learnable_low=0.05, too_easy=0.85) -> Decision`：核心判据
  - SR≥0.85 → reject（太易）；0.05≤SR<0.85 → accept（可学习区）；
  - SR≈0 但有部分进度 → accept（前沿：现在失败但够得着）；SR≈0 且无进度 → reject（太难）。
- `staged_preflight(candidates, static_check, run_short, run_full, ...)`：L1 静态→L2 短 rollout→
  L3 完整 rollout 的漏斗（越贵评越少），回调注入以便测试与解耦。

**接 rollout（集成时接入，非单测覆盖）：**
- `cold_preflight(env, env_params, train_state, rng, config, target_achievements, ...) -> PreflightResult`
  复用 `dicode.craftax_evaluation.make_evaluate` 在候选关卡上跑当前 policy，取 SR/return 后 `route`。

### 设计要点（重要架构结论）
DiCode 的生成+编译在后台线程（`executor.submit`）跑，而 cold_preflight 需要读 policy（`rl_train_state`，
主循环中不断更新）。**在后台线程读边训边变的 policy 会有并发问题**，因此 preflight 必须放主循环
（policy 手边、单线程），不能放 validation。这决定了 `cold_preflight` 与纯函数分开、集成落点在主循环
（详见 `phase3_hooks.md`）。

### 测试覆盖
- `infer_death_reason` 各分支；`partial_progress_signals` 各子信号（工具/楼层/矿/成就/库存提取）；
- `route` 四类分支（太易/可学习/前沿/太难）+ 边界；`staged_preflight` 漏斗跳级（L1 静态失败/去重、
  L2 reject 跳过 L3、通过到 L3）。

---

## 运行测试

```bash
cd /workspace/mechanism_UED/dicode_src
uv run pytest src/dicode/skill_preflight/tests/ -v
```
纯 CPU，不依赖 GPU / 模型 / 训练。skill_scheduler 8 项 + preflight 19 项，全部通过。

## 状态与下一步
- Phase 1/2 部件逻辑层完成并测通；`cold_preflight` 的 rollout 接线在 Phase 3 集成时完成并验证。
- 下一步：按 `phase3_hooks.md` 把两个部件 hook 进主循环（★A skill graph 注入生成、
  ★B preflight 过滤新任务），再跑三组消融（纯 baseline / +skill graph / +preflight）对比。
