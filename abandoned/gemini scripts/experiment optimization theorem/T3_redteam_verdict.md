# T3（multi-winner ⊇ single-winner 单调）红队终判存档

> **角色分工**：Gemini 3.1 Pro 证明（`gemini_answer3.md`）→ Claude Opus 4.8 红队 + SymPy/数值机械核验。
> **裁决日期**：2026-06-19。
> **对应任务书**：`gemini_prompt_T3_multiwinner_monotone.md`（L0/L1/L2 分层 + D1–D4 先证伪）。
> **命题出处**：[mathematical_tasks.md](../../mathematical_tasks.md) T3；机制定义 [main.tex](../../overleaf/main.tex) §177-189。

---

## ⭐ 一句话结论

**T3 通过（分层），且 Gemini 纠正了出题方 prompt 里的一个错误方差直觉（SymPy 确认）。** 核心结论：one-hot 顶点（single-winner）非局部最优、multi-winner 有实质增益的**确切条件是 `ρ_jk < √(Σ_kk/Σ_jj)`**（teacher j、k 的 score 相关系数低于该阈值）。**w 轴不只是 ×3→×2，而是"先测一次 score 协方差 Σ → 算 ρ 阈值 → 满足才测 fractional（×2），不满足则 argmax 即够（×1）"。** 与 T4 共用同一个 Σ 矩阵、互证。

> ⚠⚠ **最关键的防误读（见下方"两轴区分"）**：T3 说的"argmax 即够"**只针对 `w` 的分配丰度轴（给几个赢家），绝不意味着 non-IC 机制轴（`λ`，按什么规则选赢家）退化**。这两条是正交的轴，T3 只碰前者。**审稿人会攻"冗余情形机制退化成 baseline 吗"，正确回答：分配退化 ≠ 机制退化。**

---

## 分层判定

| 层 | 命题 | 红队判定 | 强度 |
|---|---|---|---|
| **L0** | `sup_{w} G(w) ≥ G(e_k)` | ✅ **被显式拒绝**（平凡恒真、无信息，不构成省实验依据） | 不算数 |
| **L1** | `G(w)` 在 Δ_N 连续 | ✅ 成立，挂在 `H_cont`（收敛点对 y* 连续依赖、无灾难性遗忘断崖） | 🟢/🟡 |
| **L2** | one-hot 顶点非局部最优 | ✅ **代理层证出条件 `ρ_jk < √(Σ_kk/Σ_jj)`** | 🟢 代理层 |
| **桥 H_T3** | 代理多样性 → 真实泛化 | ⚠ 诚实标为工作假设，易证伪（混入无效噪声关卡会害泛化） | 🟡 |

---

## SymPy/数值机械核验（全 PASS，含一处出题方直觉被推翻）

| CHECK | 断言 | 结果 |
|---|---|---|
| 1 | 顶点方向导数 `dV/dt\|₀ = 2(Σ_jk − Σ_kk)`（V=wᵀΣw，v=e_j−e_k） | ✅ PASS（Gemini 的 D=−V 方向导数 `2(Σ_kk−Σ_jk)` 符号一致） |
| 2 | **混入低相关 teacher 降低方差**（非出题方 prompt 直觉的"增大"） | ✅ PASS：`ρ_jk < √(Σ_kk/Σ_jj)` ⇒ `dV/dt<0` ⇒ 方差减小 |
| 3 | logit 方差小 ⇒ softmax 熵大（更 uniform、多样性高） | ✅ PASS（数值：Var 6.0→H 0.21；Var 0.17→H 1.02） |

**出题方直觉被推翻**：prompt 原写"ρ 小⇒增大方差⇒多样性高"。Gemini D4 反转为"ρ 小⇒**降低**方差⇒softmax 熵增⇒多样性高"，SymPy + 数值确认 **Gemini 对、prompt 错**。链条正确版：**低相关混入 → logit 方差降 → 课程分布更均匀 → 有效支撑（多样性）增**。

---

## L2 核心：阈值的精确形式与非对称性

```
顶点 e_k 非局部最优（multi-winner 有增益） ⟺ Σ_jk < Σ_kk ⟺ ρ_jk < √(Σ_kk / Σ_jj)
```
- **阈值不是 ρ<1，而是 ρ < √(Σ_kk/Σ_jj)**，取决于两 teacher 的方差比：
  - 要混入的 j 方差更大（`Σ_jj > Σ_kk`）⇒ 阈值 <1 ⇒ 更难触发增益；
  - j 方差更小（`Σ_jj < Σ_kk`）⇒ 阈值 >1 ⇒ 几乎总有增益。
- `ρ_jk → 1` 且 `Σ_jj ≥ Σ_kk`（j 被 k 包含）⇒ 方向导数 ≤0 ⇒ **顶点局部最优、混入无效**。

---

## ⚠⚠ 两轴区分（防止把 T3 误读成"non-IC 机制该退化"）

机制（[main.tex](../../overleaf/main.tex) §177-189）有**两条正交的轴**，T3 只碰第一条：

| 轴 | 是什么 | 由谁定 | T3 管不管 |
|---|---|---|---|
| **轴1：分配丰度** | single-winner ↔ top-k ↔ fractional（**给几个 teacher 中标**） | `w∈Δ_N` 的分布形状 | ✅ **T3 管这条**：`ρ_jk` 高 ⇒ `w=e_{j*}`（单赢家）即够 |
| **轴2：机制偏离（创新点）** | VCG-truthful ↔ non-IC（**按什么 payment rule 选赢家**） | `λ`：`w=argmax_w{Σw_j b_j − λ·ICV}` | ❌ **T3 完全不碰**，`ρ_jk` 与 λ 无关 |

**正确推论**：
- `ρ_jk` 高 ⇒ 轴1 退化成 single-winner（argmax）——**省掉 fractional 那个实验点**。
- **但"输出哪一个赢家"仍由 non-IC payment rule（λ）决定**——VCG 选一个 teacher，non-IC 可能选另一个。**argmax 退化丝毫不削弱 non-IC 创新点。**

**错误推论（必须避免）**：「ρ 高 ⇒ argmax ⇒ 不需要机制」。错在把 **allocation（给几个赢家）混同 payment（按什么准则选）**。argmax 是分配规则、non-IC 活在选择规则。

**反审稿话术**：审稿人问"冗余情形下你的机制不就退化成 baseline 了吗？"——答：**分配轴退化 ≠ 机制轴退化**。single-winner + VCG ≈ 已有 baseline（main.tex L179 自承）；但 single-winner + **non-IC** 仍是新机制（选的赢家不同）。`ρ` 只决定开不开轴1（multi-winner），轴2（non-IC）永远是因变量、不退化。

---

## 与 T4 的互证（结构红利）

T3 的 `Σ`（teacher 间 score 协方差，N×N）= T4（异质增益 ∝ score 相关）要用的**同一对象**：
- T3：`ρ_jk → 1` ⇒ 方向导数 ≤0 ⇒ multi-winner 无增益。
- T4：高相关 ⇒ 加第三类 teacher 不增课程方差 ⇒ 冗余、可砍。
- **⇒ 在已有 Stage 2 checkpoint 上测一次 Σ，同时回答 T3（要不要 multi-winner）和 T4（要不要砍 CENIE）。** 一次廉价测量、两个决策。

---

## 落到 Stage 3 的实操结论

1. **先测一次 score 协方差 `Σ`**（已有 Stage 2 checkpoint 上廉价打分，不重训）。
2. **对每对要考虑混合的 teacher (j,k) 算 `ρ_jk` 与阈值 `√(Σ_kk/Σ_jj)`**：
   - 满足 `ρ_jk < √(Σ_kk/Σ_jj)` ⇒ 测 argmax（Stage 2 复用）+ 1 fractional 内部点（w 轴 ×2），预期 fractional 赢；
   - 不满足（高相关）⇒ **argmax 即最优，连 fractional 都省（w 轴 ×1）**，并据此判 multi-winner 在 maze 上冗余。
3. **泛化层是工作假设** ⇒ "满足阈值"时那个 fractional 点仍需**实测验证**（不能纯理论宣布它赢）。
4. **non-IC（λ 轴）不受 T3 影响**——无论轴1 是否退化，λ 仍是要扫的机制因变量（地板：λ 的 3 锚点 {∞,中,0}）。

---

## 收口决定

T3 **就此收口**（L1 连续/L2 阈值代理层已证 + SymPy 确认 + 出题方直觉已纠正 + 两轴区分已固化）。下一步：**T4（异质增益 ∝ score 相关，与 T3 共用 Σ）**。
