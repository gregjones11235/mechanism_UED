# ★C 提案：级联修复（Repair Cascade）—— 用"验证+修复"分工救回小模型的幻觉废件

> 起草 2026-07-08。性质 = **第二轮实验候选之四**（与 seeds / 延长对照 / 阈值敏感性并列，优先级请 PI 拍板）。
> 依据 = baseline v2 与三臂 run 的第一手失败模式数据 + 官方口径 eval v1。

---

## 1. 动机：失败模式数据指向一个"便宜的洞"

三个 run 一致的观测：14B 生成任务的失败**几乎全是语义幻觉，零语法错误**——
- `AttributeError` 为主（15 次 @baseline）：编造不存在的枚举成员（`Achievement.DESCEND`、`BlockType.BAT`）；
- `TypeError` 次之（10 次）：参数用错；
- `SyntaxError` = 0（fence-strip 修复后）。

当前管线对失败候选的处理 = **整件丢弃 + 从头重新生成**。这浪费在两处：
1. 一个候选 ~几百到几千 token 的生成成本，因为**一行**编造的枚举名全部作废；
2. 每设计 session 12 个候选只有 ~5-9 个入库（含重试），课程吞吐被失败率压着。

**关键的不对称性：验证/修复比生成便宜得多。** 判断 `Achievement.DESCEND` 存不存在，只需要对照一张真实 API 清单——这远比"设计一个好任务"简单。这正是级联/分工能赚钱的地方：**不需要第二个模型会设计任务，只需要它会查表和改一行代码。**

## 2. 机制设计：三级级联（诚实地从最便宜的开始）

```
14B 生成候选代码
   ↓
[C-0] 静态 API lint（确定性，零 LLM 成本）
      AST 抽取 Achievement.X / BlockType.Y / 函数签名引用 → 对照真实枚举白名单
      → 命中幻觉：不丢弃，转 C-1；干净：直通编译校验
   ↓
[C-1] LLM 修复（级联的核心）
      prompt = 原代码 + 报错/lint结果 + 合法成员白名单 → 小模型输出修正版
      → 重新编译校验；上限 K=1-2 次修复尝试，仍失败才丢弃
   ↓
[C-2]（可选，后置）语义验证器：独立小模型审"任务是否忠于 docstring/target"
      —— 这是"协作"的完整形态，但先不做（见 §5 风险 3）
```

**诚实的设计要点**：主导失败模式（幻觉枚举）**大部分可被 C-0 静态检查捕获**，根本不需要第二个 LLM——所以消融必须把 C-0 和 C-1 分开（`lint-only` vs `lint+repair`），否则说不清增益来自"分工协作"还是来自"一个正则表达式"。这个诚实拆分本身就是实验设计的卖点。

**两个科学问题（本提案真正要回答的）**：
- **Q1（级联 vs 重试）**：等 LLM 调用预算下，"修复失败件"是否优于"丢弃后从头重生成"？（度量：每入库任务的 token 成本、每 session 有效新任务数）
- **Q2（异构 vs 自修）**：用**不同的**小模型当修复器（如 deepseek-coder 系）是否优于 qwen 自己修自己？——能力分工的异构性问题，与 Alec 线的 persona 竞争在概念上呼应但机制上正交（他是提案竞争选优，我是能力分工修错）。

## 3. 最小实验（两阶段，第一阶段不占训练 GPU）

### Stage 1：离线修复率测试（~1 天，只用 Ollama，不训练）
从日志/重放收集 30-50 个真实失败候选（幻觉/TypeError 样本管线里现成）→ 分三组处理：
| 组 | 处理 | 读数 |
|---|---|---|
| retry（现状对照） | 丢弃+重生成 1 次 | 通过率、token 成本 |
| lint-only (C-0) | 静态检查+机械替换最近邻合法成员 | 同上 |
| lint+repair (C-1) | lint 定位 + 14B 修复(K=2) | 同上 + 修复后语义抽查 |

**通过判据**：`lint+repair` 的"每入库任务 token 成本"显著低于 retry，且修复件编译通过率 >70%。**不达标就止损**，不进 Stage 2——这是本提案对算力最负责任的地方。

### Stage 2：接入管线跑 +A+B+C 臂（Stage 1 达标才做）
- hook 位置：`check_compilation` 失败分支（flag-gated：`+skill_preflight.use_repair=true`，默认关）；
- 跑到 session 10（157M 对齐）→ 官方协议 eval `[300,600,900,1200]`，与现有三臂同表；
- **预期读数**：每 session 有效新任务 ~7 → ~10-11；幻觉致弃率 → 近零；若课程吞吐确是瓶颈，mean_return 曲线整体左移/上移。

## 4. 在联合论文里的位置

补全 §5 的机制三部曲：**supply steering (★A) / admission gating (★B) / repair salvage (★C)** —— "小模型 regime 里，机制在三个侧面替代规模"。与 Alec 线的关系一句话讲清：auction = persona 竞争选优（proposal competition），级联 = 能力分工修错（capability division）——同属"引导生成"论点，正交实现，且 ★C 直接长在本线独有的失败模式量化数据上。
相关工作注意：LLM self-repair/Self-Refine 一族有先例 `[待核实]`；本提案的收窄主张 = **小模型 code-level UED 中，修复级联作为"规模替代物"的成本效益刻画**，非修复概念首创。

## 5. 诚实风险

1. **修复件可能"能编译但变味"**（改掉幻觉枚举后任务语义退化/变平庸）——Stage 1 加人工抽查 10 件；Stage 2 靠 ★B preflight 兜底（语义废件会被 learnability 闸拦，两道闸正好互补：C 拦代码层、B 拦可学性层）。
2. **C-0 可能吃掉大部分增益**，让 C-1 的"协作"增量很小——那结论就是"一个静态检查器抵得上半个验证模型"，同样是有信息量的诚实发现，写进论文不丢人。
3. C-2（语义验证器）先不做：在 Q1/Q2 有数据前引入第三个角色会让归因变浑。
4. 本提案与 seeds/延长对照竞争算力——**Stage 1 不占训练 GPU，可与任何选项并行**；Stage 2 才需要排队。

## 6. 给 PI 的一句话（英文）

> "Our failure-mode data shows the 14B's dominant error is semantic hallucination — inventing API members — which is far cheaper to *verify and repair* than to regenerate. I propose a repair cascade (static API lint → small-model repair, flag-gated as ★C): Stage 1 is an offline repair-rate test costing no training GPU; Stage 2 plugs into the pipeline as a fourth arm under the same official eval. It completes the mechanism story — steering (A), gating (B), salvaging (C) — and is orthogonal to the auction line's persona competition. Where should it sit relative to seeds and longer baselines?"
