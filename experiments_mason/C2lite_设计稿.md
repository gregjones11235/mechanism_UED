# C-2-lite 设计稿：三层修复冲 43（调度 / 生成 / 验证）

> 2026-07-11 起草。目标 = 修复版 +A+B 上 2e9,把 held-out mean_return 从 ~30 推向 43+(DiCode 45 的 3% 内)。
> 依据 = 五臂消融 + 302 任务审计 + prompt 溯源 + 三 regime 裸复验(全部在 `probe_结果与瓶颈定位.md` v4)。
> 状态 = **设计稿,待实现**;实现顺序与验证判据见 §5。

---

## 0. 墙的物理学(为什么是这三层)

长链任务学不会的根本原因是**终端奖励到达概率 = 中间步成功率的连乘**。以钻石剑为例,代入 probe 末点实测裸 SR:

```
P(铁镐)≈2.7% × P(挖钻|铁镐)≈7-10% × P(合成|钻石) → 联合 ≈ 0.2-0.3%
```

与裸复验实测 orig SR(task_19/20 = 0.6-0.9%)同量级,数据自洽。奖励采样不到 → 零梯度 → **不是学得慢,是学习信号不存在**;链每长一步难度指数增长 → 断裂带 = 依赖链长度。

当前管线对此有三处系统性失明,构成三层修复(每层独立可测、互相兜底):

| 层 | 缺陷(实证) | 修复 |
|---|---|---|
| **调度侧** | tier **均值**判掌握:早熟成员(collect_iron 64%)灌高均值,短板(iron_pickaxe 2.7%)躺地上,前置未齐即发钻石任务 | pick_target 改**前置齐备**判据 |
| **生成侧** | 脚手架压缩**已掌握前缀**(木器族预标),完整豁口(铁→钻 5 步)原封不动——梯子搭在不缺梯子的地方 | **mastery 驱动的脚手架放置:裸露步数=1** |
| **验证侧** | preflight 只测脚手架版可学性,对截断/错位全部失明 → 系统无法自愈 | C-0 静态检测 + C-2 裸复验抽查 |

## 1. 调度侧:前置齐备判据(替换 tier 均值)

**规则**:技能 X 可进 target 列表 ⟺ X 的**直接前置**全部单独过线(各自裸 SR ≥ `prereq_threshold`,建议 0.3)且 X 自身未过(< `mastery_threshold`)。

- 依赖边:skill-graph 已有(auction.craftax_achievements 的层级/依赖数据);
- 效果(按末点数据推演):iron_pickaxe 前置齐(collect_iron 64%/furnace 95%)→ **该练**;diamond_sword 前置不齐(iron_pickaxe 2.7%)→ **不发**——火力自动集中在真豁口;
- 实现:改 `skill_scheduler.pick_target` 的候选生成逻辑(tier 均值 → 逐节点+依赖检查),flag `+skill_preflight.frontier_mode=prereq`(默认 tier,旧行为不变);
- 附带修复 0.2 阈值的"早熟均值"假解锁问题(probe session 3 过早推 tier3 的直接原因)。

## 2. 生成侧:one-step frontier 脚手架(prompt 层,零代码闸)

**原则**:每个任务只允许**一个未掌握步骤裸露**;未掌握的更早前置可以脚手架,**已掌握的前缀不用脚手架也不许预标**(预标已掌握项是纯噪声,还污染审计)。

- 实现 = 改 persona/evolve prompt(替换现行 "SCAFFOLD them...list as Completed" 段):
  1. 注入**当前 mastery 快照**(skill: SR 列表,调度 hook 已有该数据,格式化进 prompt);
  2. 规则文本:"Scaffold ONLY prerequisites the agent has NOT mastered (SR<30%) that are NOT the training focus; the focus skill and its immediate prerequisite must be performed bare. Do NOT pre-mark or provision skills the agent already masters (SR>70%)。"
- 与 v6fix7 的关系:他们"保护焦点直接链"是静态规则;本设计加了 **mastery 数据驱动**的维度(压缩什么取决于 student 会什么)——两线互补,论文可合写;
- 风险:14B 对复杂 prompt 规则的服从率未知 → 靠第 3 层兜底,且短跑验证首要读数就是它。

## 3. 验证侧:C-0 静态闸 + C-2 抽查(两档,按成本选)

- **C-0 静态闸(必上,零 LLM 成本)**:`scaffold_audit.audit_code` 已有——preflight 前对候选跑四签名检测,违规判据 = "预标含已掌握技能" 或 "焦点/直接前置被脚手架" → 带罪证重 prompt 重生成(≤2 次,仿 v6fix7 的 validator 流程);
- **C-2 动态抽查(可选,控制成本)**:每设计 session 随机抽 2 个通过件跑裸复验(`bare_reverify` 机器现成),`bare/orig SR 比值` 落 wandb 作监控指标——**不做逐件闸**(逐件会使 preflight 开销翻倍,2e9 多 ~12h);若抽查显示裸依赖度恶化,再升级为闸;
- 保留 preflight(B)原样:剥脚手架后任务更难,恰恰更需要 B 判可学边界(三闸分工:C-0 保真码+保忠实,B 保可学,C-2 抽查保监控)。

## 4. 配置汇总(修复版 = 现 +A+B 之上叠加)

```
+skill_preflight.use_scheduler=true +skill_preflight.use_preflight=true
+skill_preflight.mastery_threshold=0.2          # 已有 flag
+skill_preflight.frontier_mode=prereq            # 新,§1
+skill_preflight.prereq_threshold=0.3            # 新,§1
+skill_preflight.use_scaffold_gate=true          # 新,§3 C-0
(prompt 改动随代码走,one-step 规则文本,§2)
```
全部 flag 默认关 → 旧 run 永久可复现。

## 5. 实现与验证计划

1. **实现顺序**(半天~1天):§1 判据(~40行) → §2 prompt(文本) → §3 C-0 hook(复用 audit,~60行) → 单测(判据推演用末点 JSON 回放;C-0 用 task_19 复刻件);
2. **短跑验证**(session 10,~7h 挂夜),三判据:
   - 泄漏语义改善:预标已掌握技能的比例 ↓(audit 复扫);
   - **tier-3 裸指标动**:iron_pickaxe/diamond 族 held-out SR 相对 probe 同期 ↑(核心判据);
   - preflight 拒绝率不失控(<30%;若 one-step 任务太难被 B 大量拒,prereq_threshold 上调);
3. 判据过 → 终配置定稿 → **2e9 主跑**(`total_timesteps=2000000000` 从头,~55-65h,resume 保险已验证);判据不过 → 按失败模式回退(prompt 服从率低→加重 C-0;任务太难→调阈值)。
4. eval:官方协议 seed 0,steps=[300..15300 量级](2e9 的 keep_period 留存),gap 报告 vs 44.58/48.33。

## 6. 诚实预期

- 三层修复直接攻击的是"课程供给质量",**不保证** 43(第二层未知数:即便课程完美,14B 策略容量/2e9 预算是否够长链,只有跑了才知道);
- fallback:即使停在 35-40,三层修复 + 断裂带机制分析本身构成 §5 的完整故事;
- 消融纪律:修复三层打包为"最终方法"上 2e9(论文标准做法);单层归因如需要,3e8 短跑补(prereq-only / prompt-only 各一臂,候选队列排后)。

## 7. ★ 总工作队列与文档关系(新会话从这里开始)

**执行顺序(严格按此,勿乱)**:

| # | 事项 | 状态 | 说明 |
|---|---|---|---|
| 1 | **本稿三层实现**(§5 顺序:prereq 判据→one-step prompt→C-0 闸)+ 单测 | ⬜ 待做 | 关键路径第一步 |
| 2 | **短跑验证**(session 10,~7h,三判据见 §5.2) | ⬜ | 全周最重要的闸门,不过不上 2e9 |
| 3 | **修复版 +A+B @2e9**(total_timesteps=2e9 从头,~55-65h,resume 已验证) | ⬜ | 主数字 |
| 4 | 官方 eval(seed 0,steps=[300..15300 量级])→ **gap 报告 vs 44.58/48.33** | ⬜ | 目标 43+ |
| 5 | 条件触发:★C-1 修复级联接入 | 🅿 板凳待命 | **仅当**短跑显示 one-step 约束使生成失败/重造率飙升(吞吐成瓶颈)时上;离线验证已完(30/30),接入=一个 hook |
| 6 | 押后队列:baseline@2e9(可请队友)/ seeds / PLR接learnability / 跨线审计(找Alec要fix7前后graphml) | 🅿 | 出主数字后再分派 |

**文档关系与命名消歧**:
- `C2lite_设计稿.md`(本稿)= **冲 43 的三层修复方案**(调度/生成/验证),主线作战图;
- `proposal_C_repair_cascade.md` = **幻觉修复级联(C-0 API lint + C-1 修复)**,治吞吐量(幻觉→丢件),离线验证已完、**未接入**,对应上表第 5 项——治的不是 30 分的墙(墙是质量问题非吞吐问题),故不在关键路径;
- 两处 "C-2" 是同一概念(忠实性/裸复验)的不同落地形态:proposal_C 的 C-2 = 完整语义验证器构想,本稿 §3 的 C-2 = 抽查监控版(成本考量);工具同为 `bare_reverify.py`;
- C-0 有两个含义:proposal_C 的 C-0 = API 幻觉 lint(`stage1_repair_bench.lint`),本稿 §3 的 C-0 = 脚手架静态闸(`scaffold_audit.audit_code`)——**两者挂同一 hook 位置,实现时可合并为一次 AST 遍历**(顺路零成本);
- 证据底座:`probe_结果与瓶颈定位.md` v4(五臂表/审计/溯源/三 regime)、`official_eval_v2.md`(主结果)、`frontier_lock_分析与修复.md`(阈值 flag 由来)。
