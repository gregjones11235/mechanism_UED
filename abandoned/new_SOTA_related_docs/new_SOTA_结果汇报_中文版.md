# 新 SOTA 结果汇报：定向生成关卡在 jaxnav 上的 worst-case CVaR 超越 SFL

> 创建 2026-06-26。口径：100-map 测试集 + returns CVaR10（最难 10% 关）。currA = 我们的方法（learned PCGRL generator + geodesic-anchored 信号 + learnability 课程，注入固定 PLR buffer）。

---

## 一句话结论

**在 jaxnav 连续导航主场，用比 SFL 更小的候选池（3000 < 5000），我们的方法在最难 10% 关的 worst-case CVaR 上稳定超过近期 SOTA SFL（CVaR10 均值 1.75 vs 1.27，+38%），且三个 seed 一致、几乎不崩盘。** 均值持平、差距全在尾部——这正是 UED 方法该有的卖点：定向供给难关 → student 在最难关上更鲁棒。

---

## 1. 核心结果表（多 seed，统一口径 = returns CVaR10）

> baseline = [SFL](https://arxiv.org/pdf/2408.15099)（Rutherford, NeurIPS 2024）。**选 SFL 作为对标对象**：经 deep research 文献核查，SFL 是当前 UED / 课程学习里最强的方法（在 jaxnav 上以 learnability curation 击败 DR / PLR / ACCEL / PAIRED 一众 baseline），所以超过它即等于刷新该设定下的 SOTA。

| 方法 | 候选池 | overall win | overall return | **CVaR10 return（最难10%）** | worst-bin return |
|---|---|---|---|---|---|
| **SFL baseline**（learnability curation） | 5000 | 0.979 | 3.911 | 1.269 | −4.94 |
| **currA seed0** | 3000 | 0.982 | 3.933 | **1.653** | −2.11 |
| **currA seed1** | 3000 | 0.983 | 3.954 | **1.730** | −4.09 |
| **currA seed2** | 3000 | 0.986 | 3.947 | **1.855** | −0.87 |
| **currA 均值 ± 范围** | 3000 | 0.984 | 3.945 | **1.746（1.65~1.86）** | — |

**关键读数：**

- **CVaR10 全面超越 SFL**：三个 seed 全部 ≥ 1.65，均值 **1.746 vs SFL 1.269（+0.48，+38%）**。三 seed 一致（范围仅 1.65~1.86），不是单 seed 运气。
- **overall return / win 持平**：均值 3.945 vs 3.911、0.984 vs 0.979。差距**不在均值，全在尾部**——100-map 大部分关不难，谁都能解；分水岭在最难那批关。
- **几乎不崩盘**：SFL 在最惨那关 return = **−4.94**（撞墙 / 超时深度崩盘）；currA 最好的 seed 只 −0.87（接近解开），即使最差 seed 也只 −4.09，整体比 SFL 更不容易在难关上彻底失败。

---

## 2. 为什么能赢 SFL（机制）

SFL 用同样的 learnability 信号，但**没有 generator，只能被动从 `env.reset` 的无偏先验里挑关**——挑不出池里本就没有的特定结构难关（如 Nav2/Nav3 那类背朝目标 + 贴墙起点 + 长程测地的走廊拓扑）。

我们有 **learned PCGRL generator**，能**主动定向生成** SFL 永远海选不到的难关，再注入固定 PLR buffer。所以这是：

> **"少而精的定向生成" > "多而杂的随机海选"。** 同样喂 PLR，我们用更少、更稀缺的原料喂出了更鲁棒的 student。

---

## 3. 卖点的正确表述：赢在质量 / 训练效率，不是基数

**同口径数字直接否定"靠候选池基数赢"这个外推：**

- **基数最大的是 SFL（5000），它恰恰是 CVaR 输家。** 我们用 3000（更小的池子）反而赢。若靠基数，5000 该赢。
- **可学关（p≈0.5）我们还少一个数量级**：SFL 池里可学区占 1~21%，我们只 0.07~1.4%。我们在**可学原料更稀缺**的劣势设定下赢了。

**所以真正的卖点 = 关卡质量 / 训练效率，不是数量：**

> 用更小的候选池、少一个数量级的可学关，**定向生成的关卡单位价值远高于随机海选**，在最难 10% 关的 worst-case CVaR 上超过近期 SOTA SFL（+38%）。

---

## 4. 方法构成（currA 当前最佳架构）

| 组件 | 作用 |
|---|---|
| **learned PCGRL generator** | 主动生成关卡（非被动海选），注入固定 PLR buffer |
| **geodesic-anchored 信号** | 用测地距离场几何锚定 value-loss / regret，防 generator 堆墙刷分 |
| **CENIE 信号** | 覆盖度 / 新颖度信号 |
| **learnability 课程** | 引导生成分布对准可学区（p≈0.5） |

> 注：difficulty 信号已证明在大基数下多余（全程权重仅 8%，且 difficulty ≡ learnability − 0.25 在两极池失区分度），可砍。当前最佳 = **anchored + cenie + geo 课程**的简化架构。

---

## 5. 诚实账（这份成果目前的边界）

1. **"战胜 SFL" 的当前严格含义 = 内部对照式战胜**：SFL baseline 是我们自己 repo 里 `GENERATOR_INJECTION=false` 跑的，同环境 / 同评测 / 同代码，唯一变量是"有没有 generator"。这是最干净的消融对照，已成立。
2. **尚未做"论文同表战胜"**：和 SFL / PLR / ACCEL / PAIRED **论文报告的数字**同表对标（需 10 seed + 外部 baseline）还没做。投稿时两者都可说，但不能把前者说成后者。
3. **CVaR 是尾部统计**——目前 3 seed 已给出一致性证据，但投稿级强度需要推到 10 seeds。

---

## 6. 数据来源

- **wandb 项目**：`gregjones11235-brown-university/multi_robot_ued`
