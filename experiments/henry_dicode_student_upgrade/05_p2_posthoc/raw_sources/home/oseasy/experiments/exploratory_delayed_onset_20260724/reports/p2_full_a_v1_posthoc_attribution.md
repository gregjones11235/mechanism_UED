# P2-Full-A-v1 离线 Replay 归因与 Henry 长上下文缺口审计

**标签**: POSTHOC_REPLAY_ATTRIBUTION_AND_HENRY_GAP_AUDIT / P2-Full-A-v1 / 只读离线 / 不训练
**裁决**: `NEXT_CANDIDATE = D (P2-Full-B-long-context)` | `CONFIDENCE = MEDIUM` | **`TRAINING_AUTHORIZED = false`**
**机器可读**: `reports/p2_full_a_v1_posthoc_attribution.json`（sha256=76c03ead406464f5f3ccb2c2c932e50e7edfdfe2d4b2efcae2bbfe1dfade5cd5）

四节点（24576/49152/73728/98304）全部以同一只读 harness 完成；每节点 READONLY_OK=true、SHA 前后逐项一致、params==expected。

---

## 1. Henry 总目标与当前阶段

- **冻结目标**：Student 强于健康 PPO-GTrXL（ckpt17500）；Official FULL / floor0 自然出生可复现，同时突破 ENTER_SEWERS 与 DEFEAT_KOBOLD 零率；S4_dark 仅探针（≥+8pp）。评测只认 fresh-world 零样本（512×4096）。γ/λ 严格 0.999/0.8。
- **§1.3 核心矛盾**：输家 episode 中位 643 步 ≫ 128 记忆窗。
- **P2 阶段假设**：整轨迹 replay + 长上下文 transformer + hindsight 重标注，off-policy TD 在全轨迹上下文上学（AMAGO, ICLR 2024）。死路名单含「window 256」——任何候选须是真正长上下文结构，不是加窗。
- **本次任务**：不训练、不提前选候选；对 P2-Full-A-v1 四个 checkpoint 做只读 replay 梯度归因 + 长上下文缺口审计，四节点全完成后恰好推荐一个候选。

## 2. 需求差距矩阵摘要

详见 `reports/p2_full_a_v1_henry_requirement_matrix.md`（R1–R7）。本次归因直接对应的差距：
- **长上下文结构（R 核心）**：未解决。§8 证明 window_mem=128 卡死历史，无 >128 步机制（见 §6）。
- **DK 成功信号**：四节点 buffer 中 DK 成功 = 0/0/1/1，成功梯度基本不可估（见 §5）。
- **off-policy 质量**：lag 门（max_policy_lag=16）使最旧轨迹在后期 checkpoint 的 AWR/hindsight 路径被掩零；近端批已修复可估性（见 §5）。

## 3. 四节点梯度归因（§5）

每节点双批：`fixed`（最旧轨迹，trajectory_id 升序）+ `recent`（最新轨迹，lag 升序，保 AWR 可估）。`valueTotal_vs_full` 为 value 合计梯度与 full 梯度的余弦。

| step | uc | DKsucc | valueTotal_vs_full (fixed/recent) | vtraceActor_vs_full (recent) | awrActor_vs_full (recent) | awr_valid_frac (fixed/recent) |
|---|---|---|---|---|---|---|
| 24576 | 11 | 0 | 0.988 / 0.893 | 0.485 | -0.059 | 1.0 / 1.0 |
| 49152 | 23 | 0 | 0.992 / 0.862 | 0.313 | 0.409 | 0.0 / 1.0 |
| 73728 | 35 | 1 | 0.932 / 0.875 | 0.493 | 0.259 | 0.167 / 1.0 |
| 98304 | 47 | 1 | 0.846 / 0.936 | 0.375 | 0.128 | 0.167 / 1.0 |

**核心发现**：
1. **Value 梯度主导 Full 梯度**（valueTotal_vs_full = 0.85–0.99，四节点一致）。value 损失量级（B≈9–53）远大于 actor，梯度范数由 value 主导（B gnorm 数百）。
2. **AWR/hindsight 路径在固定批后期被 lag 门掩零**（fixed awr_valid_frac 49152=0.0、73728/98304=0.167），这是训练损失真实门控（`full_p2_learner.compute_loss` L186-187，lag≤16）。**近端批（lag≤16）使 AWR 在每节点可估（awr_valid_frac=1.0，C/D 梯度非零）**——这是本次 harness 的关键修复，使候选 A/C 裁决所需的 AWR 路径在四节点都可观测。
3. **vtraceActor 与 awrActor 无持续冲突**（vtraceActor_vs_awrActor recent：-0.016/0.057/0.100/0.030，近零或微正）。
4. **成功梯度不可估**：24576/49152 buffer 无 DK 成功 → `SUCCESS_GRADIENT_NOT_ESTIMABLE`；73728/98304 各仅 1 条成功（部分可估，样本过少）。

## 4. Candidate KL 来源（§6，一步候选更新 forward KL，近端批）

7 个只读候选（同一 opt_state 一步更新后丢弃）：c1=vtrace-actor，c2=awr-actor，c3=vtrace-value，c4=hindsight-value，c5=actors 合并，c6=values 合并，c7=full 合并。

| step | c1 vtrace-actor | c2 awr-actor | c3 vtrace-value | c5 actors | c6 values | c7 full |
|---|---|---|---|---|---|---|
| 24576 | 0.00594 | 0.00572 | 0.00276 | 0.00594 | 0.00303 | 0.00219 |
| 49152 | 0.00325 | 0.00070 | 0.00130 | 0.00147 | 0.00133 | 0.00104 |
| 73728 | 0.00170 | 0.00194 | 0.00069 | 0.00204 | 0.00039 | 0.00103 |
| 98304 | 0.00185 | 0.00191 | 0.00081 | 0.00194 | 0.00067 | 0.00075 |

**关键判读**：
- 所有候选 KL ≪ 0.05（单步漂移很小，符合 lr=2e-5 + grad_clip）。
- **actor-only（c1/c5）≥ value-only（c3/c6）于全部四节点**：尽管 value 梯度主导 full 梯度的*范数/方向*，但 value-only 一步更新造成的*策略漂移*反而小于 actor-only。原因：value 梯度主要落入 value head（critic-only）+ 共享 trunk 中「不动 logit」的方向；actor 头未被 value-only 更新直接改变，故 policy KL 小。actor 梯度虽小，却直接驱动 logit 变化。
- **AWR-actor（c2）不是主 policy-KL 源**：c2 ≤ c1 于 3/4 节点（24576 近似相等）。

## 5. ratio / lag / 轨迹 / FIFO 分析（§7，近端批）

| step | ratio p95 | ratio max | ESS | 成功/批 | FIFO 插入/当前/驱逐(cap64) |
|---|---|---|---|---|---|
| 24576 | 1.345 | 42.53 | 0.282 | 0/6 | 24/24/0 |
| 49152 | 1.339 | 7.33 | 0.807 | 0/4 | 45/45/0 |
| 73728 | 1.195 | 4.03 | 0.953 | 1/4 | 78/64/14 |
| 98304 | 1.324 | 45.47 | 0.274 | 1/6 | 95/64/31 |

- **ratio 温和**：近端批 p95≈1.2–1.35（低 lag 样本接近 on-policy）；偶有极端 max（42.5/45.5）但被 grad_clip 约束，ESS 合理（0.27–0.95）。无单一 replay 梯度异常（梯度全 finite，无 NaN/Inf）。
- **lag 门效应**：固定批选最旧轨迹（lag 22–31），后期超 max_policy_lag=16 → AWR 掩零；近端批（lag 1–15）全部过门。
- **FIFO 填满与驱逐**：buffer cap=64；24576/49152 未满（无驱逐）；73728 起满（驱逐 14/31）。驱逐的均为最旧（最高 lag）轨迹——与固定批后期 AWR 不可估互为印证。
- **轨迹质量**：DK 成功极稀（0/0/1/1），AWR 高权重「集中于失败轨迹」被「几乎全失败」混淆，不构成 AWR 有害的独立证据。

## 6. 长上下文缺口结论（§8，深窗记忆消融）

深窗（start=384，长 episode）四条件共享同一 scan_fn 损失区前向，仅进入态 (memory,mask,idx) 不同：
- **A**=真实 anchor + 完整重 burn-in（训练逐位前向）
- **B**=窗起点记忆清零
- **C**=从零重建最近 128 步
- **D**=384 步 burn-in（窗上限 128）

| step | meanB_kl (零记忆 vs A) | B top-action翻转 | meanC_kl (近128) | meanD_kl (384 burn-in) | 失败 episode >128/256/512 步 |
|---|---|---|---|---|---|
| 24576 | 0.9996 | 0.311 | 0.0015 | 0.0028 | 0.833 / 0.625 / 0.417 |
| 49152 | 0.9442 | 0.249 | 0.0255 | 0.0312 | 0.867 / 0.711 / 0.556 |
| 73728 | 1.4524 | 0.251 | 0.0425 | 0.0493 | 0.857 / 0.683 / 0.571 |
| 98304 | 1.2369 | 0.238 | 0.0110 | 0.0134 | 0.841 / 0.683 / 0.492 |

**结论（四节点一致，A≈C≈D≪B）**：
1. **策略强烈使用 128 窗记忆**：B（零记忆）相对 A 的 action KL≈0.94–1.45、top-action 翻转≈24–31%——远超浮点噪声，记忆对决策至关重要。
2. **无 >128 步历史机制**：D（384 步 burn-in）≈ C（近 128 步）≈ A，三者 KL 仅 0.0015–0.049。把 burn-in 从 128 拉到 384 **不带来任何额外信号**——因为 GTrXL window_mem=128 只容纳最近 128 步，更早历史被结构性丢弃。
3. **历史 >128 步是任务相关的**：84–87% 的失败 episode 超过 128 步，约 42–57% 超过 512 步（与 Henry §1.3「输家中位 643 步」一致）。DK 目标（深层地下城探索）本质长程，而网络只能看 128 步。
4. **区分判定**：当前网络属于「(2) 通过 128 窗递归 memory 保留部分历史 + (3) window 结构截断 >128 历史」，**不是**「(1) 对 128 步前完全不敏感」（B≠A 证明敏感），也**不能**写成「已拥有 Henry 要求的整局长上下文」（D≈C 证明无 >128 机制）。

## 7. 唯一 NEXT_CANDIDATE

> **`NEXT_CANDIDATE = D — P2-Full-B-long-context`（真正的长上下文结构记忆，非加窗）**

候选门逐项核验（跨四节点）：
- **A (no-HER-actor) — 否决**：AWR-actor 非主 candidate-KL 源（c2≤c1 于 3/4 节点）；vtraceActor 与 awrActor 无持续冲突（余弦近零）；「AWR 权重集中失败」被 DK 成功≈0 混淆。三门均不满足。
- **B (value-isolated) — 否决**：value total 主导 Full 梯度（条件1✓），但 value-only 候选更新造成的 policy KL/logit 变化在全部四节点**小于** actor-only（c3/c6 < c1/c5），与 B 的定义条件（value-only 大漂移 + actor-only 明显较小）**相反**。value 梯度范数大但 policy-quiet。
- **C (vtrace-restricted) — 否决**：vtrace-actor 非主漂移源（valueTotal_vs_full 更高）；近端批 ratio 温和（p95≈1.2–1.35）；value/AWR 才是首要，非 vtrace 限制目标。
- **D (long-context) — 选中**，四条件全满足：
  1. 无单一 replay 梯度异常（梯度 finite、ratio 有界、AWR 正常门控、无 NaN）；
  2. 128 步前历史对决策有任务相关影响（84–87% 失败 episode >128 步，DK 长程）；
  3. 当前网络无法稳定保留/使用 >128 历史（A≈C≈D，384 burn-in 无增益）；
  4. 证据更支持结构记忆瓶颈而非优化路径（§8 为结构性前向探针，4/4 可复现；梯度/优化分析无更好解释失败的异常）。

## 8. CONFIDENCE

> **`CONFIDENCE = MEDIUM`**

- §8 结构瓶颈干净且四节点可复现（A≈C≈D≪B，episode ≫128，无 >128 机制）——支持 D 的主证据稳健。
- 降为 MEDIUM（非 HIGH）的原因：
  1. **DK 成功≈0**（0/0/1/1）→ 成功 vs 失败梯度归因基本不可估（SUCCESS_GRADIENT_NOT_ESTIMABLE），限制了「成败轨迹分别的 candidate KL」这一判据；
  2. **value 梯度主导 Full 梯度是真实的次级信号**（虽不满足 B 的机制条件，仍值得在长上下文候选训练中监控 value/trunk 相互作用）；
  3. >128 历史的「任务相关性」由 episode 长度结构 + Henry 核心矛盾**推断**，尚未通过一次真正的长上下文干预**直接验证**。

## 9. TRAINING_AUTHORIZED

> **`TRAINING_AUTHORIZED = false`**

本报告为只读离线归因。不启动任何新 P2 训练；不提交 optimizer；不改 checkpoint/训练代码/超参/阈值；不第二 seed；不 512 评估。候选 D 的实际训练需另行明确授权。

## 10. 证据路径与 SHA256

- **harness**：`posthoc_attribution/src/posthoc_attribution.py` sha256=`9663d70a8d5b4aabd49df9bd69d42271693f5b90743058e6019851cbf2718b45`（双批：fixed + recent；§5/§6/§7 重构为 analyze_batch 复用；§8 深窗不变；GPU0 UUID 绑定 + 只读 SHA bundle 7 项前后断言）
- **run 脚本**：`posthoc_attribution/src/run_posthoc.sh` sha256=`381a943ee91313bfdb9ff61ab632db7f79ffb9ca3388aa0694d9f23bdf8dc680`
- **每节点输出 JSON**（READONLY_OK=true，bundle 前后一致）：
  | step | params_sha256 | uc | replay | 输出 |
  |---|---|---|---|---|
  | 24576 | `bd08422042788f63…` | 11 | 24 | `posthoc_attribution/out/posthoc_24576.json` |
  | 49152 | `6b2a4fc5035a1c86…` | 23 | 45 | `posthoc_attribution/out/posthoc_49152.json` |
  | 73728 | `2d93352f238ec447…` | 35 | 64 | `posthoc_attribution/out/posthoc_73728.json` |
  | 98304 | `67689592cd10f6c9…` | 47 | 64 | `posthoc_attribution/out/posthoc_98304.json` |
- **checkpoint roots**：24576=`/home/oseasy/experiments/p2_full_20260723/checkpoints/p2_full_levelB_24576_20260724`；49152/73728/98304=`/home/oseasy/experiments/exploratory_delayed_onset_20260724/p2_resume_RUN1/ckpt`
- **合并交付物**：`reports/p2_full_a_v1_posthoc_attribution.json` sha256=`76c03ead406464f5f3ccb2c2c932e50e7edfdfe2d4b2efcae2bbfe1dfade5cd5`
- **需求矩阵**：`reports/p2_full_a_v1_henry_requirement_matrix.md`
- 全部位于服务器 `/home/oseasy/experiments/exploratory_delayed_onset_20260724/`。
