# v6fix7 整体修复设计方案（隐患 #1-#8 一体化）

> 起草 2026-07-06。输入 = 同目录《idea代码实现风险审计结果.md》的 8 项隐患 + 《idea设计审核与竞品分析结果.md》的可借鉴机制。
> 设计原则（用户拍板）：**工程量不是代价，唯一目标 = held-out SOTA 性能最大化**。因此默认代码硬闸 > prompt 软约束。
> 两条不放松的真约束（非工程成本，是实验有效性）：① baseline 对标不变性 —— 全部改动 siege-gated，siege off 时全路径 byte 级不变；② 知识泄露边界 —— 不引入 (a) 课程链先验，代码闸用的链数据只来自 notebook 里 LLM 自推的 prereq_tree。

---

## 0. 方案总骨架：一个基础设施吃掉五个隐患

审计的核心发现是：**所有 prompt 层的约束都在"LLM 心情"这条单点故障上**（superset 冲突、送成品、TYPE 无校验、gate 静默搬运）。与其逐条打 prompt 补丁，不如上一层统一的代码基础设施，让每条规则都有代码兜底。这就是为什么 #1 的选项 C 单独做没必要 —— 它被 P0 吸收并超越。

```
P0  结构化关卡元数据 + 生成环校验器 (SiegeLevelValidator)     ← 吸收 #1C
 ├─ #1 superset vs drill      → prompt 豁免 + validator 规则 1
 ├─ #2 真采真熔真造           → prompt 优先级声明 + validator 规则 2/3
 └─ #5 gate 搬运不可解        → 静默搬运改为"带罪证回炉重造"
P1a #3+#8 攻坚生命周期重设计（escalation ladder + conquest 修正）
P1b #4  siege 关的生成/训练生态位豁免（配额 + priority floor + gap 毕业）
P1c #6  style_note 生命周期（AutoManual-lite）
P1d #7  毛刺清理（wandb 双写、恒真条件）
P2  时序链挖掘 + 失败局采集（把 (c) 从"相关"升级为"有向+断链定位"）
```

---

## P0 基础设施：结构化元数据 + 生成环校验器

### P0.1 结构化关卡元数据

**问题**：proposer 选的 TYPE 只存在于 `<reasoning>` 自由文本里，代码读不到 → 一切按 TYPE 分派的规则都无法代码化；archive 里也无法识别哪些关是 drill/siege 关 → #4 的豁免无从谈起。

**改法**：
- proposer 输出模板新增机器可读块（与 `<docstring>` 平级）：
  `<level_meta>{"type": "DEPTH|BREADTH|CONSOLIDATE", "drill_target": "<skill或null>", "siege_wall": "<skill或null>"}</level_meta>`
- gen_manager 解析：失败 → 重试一次（重prompt 该 proposer，附解析错误）→ 仍失败 fallback `{"type":"DEPTH"}` + WARN（保守：DEPTH 不触发任何豁免）。
- 元数据写入 task_graph 节点属性（与 session_created/performance_history 同级）→ 训练采样、复盘脚本、监控都能读。
- **文件**：persona_ambitious_coop.py（模板）、gen_manager.py（解析+入图）、新增 auction/level_meta.py（schema+校验，纯 python 可离线测）。

### P0.2 SiegeLevelValidator（生成环校验器，docstring 阶段、编译前）

对每个新造关按元数据跑规则；**违规不静默修正，而是带具体罪证重prompt 该 proposer 重造**（至多 2 次），仍违规才 fallback 到"代码自动修正 + ERROR 日志"（保底不断训练）。重造成本 = 一次 LLM 调用，按用户原则可忽略。

| 规则 | 内容 | 数据源 | 治哪个隐患 |
|---|---|---|---|
| R1 形态 | type=CONSOLIDATE：Relevant ⊆ {drill_target} ∪ 其执行链环 ∪ directive 强制 unmastered 环，且不得含无关战斗/生存成就；type≠CONSOLIDATE：superset(父关) 检查 | level_meta + notebook prereq_tree | #1 |
| R2 禁送成品 | **任何 TYPE**：active focus 技能 & drill_target 禁入 Completed | notebook.foci | #2 |
| R3 保执行链 | type=CONSOLIDATE：焦点 prereq_tree 的**直接链环**禁入 Completed（必须留 Relevant，真采真熔真造）；更深的祖先环若 CONSOLIDATED 仍可压缩 | notebook prereq_tree（LLM 自推，无新先验） | #2 |
| R4 世界自证 | 被 R1-R3 或 completed_gate 拉回 Relevant 的每个成就，docstring World 段必须提供对应资源/怪物/工具（重prompt 时把"缺什么"列出来） | docstring 文本（LLM 自查+声明） | #5 |

**注意 R1 的正确语义**（讨论 #1 时已厘清）：drill 的 Relevant **不是**纯单技能 —— 目标自己的真实执行链（如铁镐的 collect_iron/coal/place_furnace/table）本来就该在 Relevant 里（这就是"真采真熔真造"），剥掉的是**无关**目标（defeat_zombie/eat_cow/wake_up 这类父关继承物）和环境干扰。

### P0.3 prompt 层配套（消矛盾 + 定优先级）

代码闸兜底不等于 prompt 可以继续自相矛盾（矛盾会拉高重造率、浪费轮次）。三处改动：
1. `persona:166` superset 规则加 CONSOLIDATE 例外（= 原 #1 方案 A）；`persona:147` point 0 呼应。
2. GUIDING PRINCIPLE（`persona:100`）与 mandate（`persona:58`）各加一句适用范围："压缩原则**不适用于** active siege focus 自身、drill 目标的直接执行链、以及任何 SIEGE_DIRECTIVE 列为 unmastered 的环。"；SIEGE_DIRECTIVE 的 "mastered — may be compressed" 标注（gen_manager.py:1463）对焦点直接链环改为 "kept in Relevant for drills"。
3. 新增一段**全局优先级声明**（放 system prompt 顶部规则区）："指令冲突时优先级：SIEGE_DIRECTIVE/drill 规则 > TYPE 定义 > 通用 scaffold 原则。"

---

## P1a 攻坚生命周期重设计（#3 退休死循环 + #8 conquest 过早）

**现状病灶**：棘轮 stall（+3pp 新高即清零）→ 震荡可无限推迟退休；退休无冷却无记忆 → 同 session 原地重开（fix4 实锤）；conquest 疑似 delta 触发 → 44% 就进 verified_chains/protected_set。

**改法 —— 把"退休/重开"二元开关换成升级阶梯（escalation ladder）**，性能逻辑：墙是真墙就该继续攻，但**必须换打法**，而不是同一套打法无限循环或干脆放弃。

**★核心修订（用户 2026-07-06 质询后定稿）：耐心自适应 —— stall 只在"整棵攻坚树全冻结"时计数。**
tier4 战斗墙的焦点 SR 可能合理地在 0% 趴 20+ session（student 在链条下游打地基），只盯焦点 SR 会把健康的地基期误判为僵局、杀死本会成功的长攻。stall 计数器改为：**三路进度信号任一命中即清零，全部冻结才 +1**：
- ① 焦点 SR 实质动静（新高 ≥3pp 或 W=6 窗口回归斜率 >1pp/session；窗口斜率同时防"棘轮震荡推迟退休"的旧病）；
- ② 任何未掌握链环 SR 实质动静（地基在涨 = 攻坚在起效）；
- ③ **断链环前移**（P2 提供，tier4 关键）：失败局"最远打到哪一环"的分布在往深处移 —— SR 为 0 但死得越来越近也是进度。
→ 耐心无上限：树里任何地方有进展，阶梯永不推进。触发线的语义变为"连续 N session 全树零进展"（参照：session≈100 PPO 更新≈15-30 分钟墙钟；fix4 实测打法对时 10 session 可从 1%→44%，故 12 session 全冻结 = 充分止损证据）。

**阶梯（强度随冻结时长升级：先问责、后强制）**：
| 级别 | 触发(全树冻结 session) | 动作 | 强度 |
|---|---|---|---|
| L1 | ≥3 | 换形态，**或**写出留守的具体新理由+新计划（理由非空且与上次不同，代码查重） | 建议+问责 |
| L2 | ≥6 | 强制换形态（DEPTH↔CONSOLIDATE；modeler 不换则代码翻转 recommended_type 并下轮告知） | 强制 |
| L3 | ≥9 | 强制换战术（新 style_note 与旧规范化比对须实质不同，一次重写机会） | 强制 |
| L4 | ≥12 | 退休 + 冷却 6 session（_reconcile_foci 与 expand gate 拒绝重开）+ 失败战术归档 | 强制 |

- **重开检查**：冷却期满重开同一技能，modeler 必须声明"这次有何不同"，且新战术 ≠ 归档失败战术（代码比对）；累计退休 ≥2 次 → 黑名单，直到其某个链环 SR 出现新证据（或断链环前移）才解锁。
- **"强制 vs 建议"的设计依据**：L1 保留 LLM 的辩护权（墙硬需要时间是合法判断，但必须说出新理由，不得沉默续杯）；到 L2 已辩护过一轮且再冻结 3 session，强制的证据充分 —— 本库血泪史（prompt 说了不算 + fix4 modeler 30 session 行动惯性）排除纯建议，tier4 长攻风险排除纯强制。
3. **退休史可读化**：`render_for_prompt` 渲染最近退休事件 + 失败战术摘要给 modeler（现在 history 只写不读）。
4. **conquest 修正（#8）**：verified_chains / protected_set 准入 = **连续 2 个快照 held-out SR ≥ mastered(70)**；废除 delta/record 触发进 verified 的路径（record 只进"进展日志"）。防止未攻破技能污染 tier4 复用地基。
5. **焦点位战略约束（部分回应"H1 上不了场"风险）**：max_foci=3 现在实际只开 1 个。改为：当排名第一的墙是 CRAFT/enabler 类（如铁镐）时，modeler 被鼓励（prompt）同时开一个 COMBAT 类第二焦点 —— enabler 走 drill 快攻，战斗墙走 DEPTH 长攻，H1 提早上场。（软约束即可，代码只保证不超 3。）

**文件**：siege_notebook.py（双轨 stall、阶梯状态机、冷却/黑名单、conquest 准入、渲染退休史）、modeler.py（阶梯提示注入 + 强制换挡/换战术的 schema 字段）、gen_manager.py（注入点）。

---

## P1b siege 关的生成/训练生态位豁免（#4）

**现状病灶**：生成期 drill 按父系连坐被 pure-lrn cull；训练期 drill 高 SR 后 p(1-p)→0 被 CAS/PLR 挤出 —— "drill 学会了就被扔"vs"held-out 还没传导上去"结构性冲突。

**改法**（三层，全部 siege-gated）：
1. **生成期分区保送**：`_coop_select` 改为分区 —— 带 siege 标记（level_meta.siege_wall 非空或 type=CONSOLIDATE 且 drill_target ∈ active foci）的候选**全保**，top-k 只在其余候选里选，总保留数不变（siege 关占掉的名额从非 siege 池里扣）。
2. **训练期焦点配额 + priority floor**：
   - 新增 `reserve_siege_quota`（与 append_rehearsal_tasks 同构，接在 sample_tasks_for_training 之后）：焦点 active 期间，保证训练 batch 内 ≥ Q=4 槽给"教当前焦点"的关（按 lineage 最新优先）；不足配额时从 archive 强制补入。
   - 教 active focus 的关，`priority_score` 加 floor：`max(p(1-p), siege_floor=0.15)`，**墙未破期间**不随 p 饱和归零；墙 conquest 后 floor 撤销，自然衰减。
   - CAS 激活：siege 关跳过与 worst-active 的比分，强制进 active set。
3. **drill 毕业机制（train-eval gap，借 SCALAR p18 Fig.9）**：跟踪 `gap = drill 关 trained SR − 焦点 held-out SR`。drill trained SR > 90% 而 held-out 焦点 SR 停滞 → 给 modeler 注入明确信号："drill 已过拟合安全场景，按 CONSOLIDATE 定义加回干扰、向完整游戏收敛"（这正是 §3.10 修复里"reduce 非清零、SR 升后加回"的量化触发器）；gap 收敛（< 10pp）→ 该 drill 退出配额与 floor，让位新关。

**文件**：gen_manager.py（_coop_select 分区）、selection.py（reserve_siege_quota + floor）、evolution_efficient.py（CAS 豁免）、training.py 或 gen_manager（gap 统计，喂 modeler prompt）。

---

## P1c style_note 生命周期（#6，AutoManual-lite）

- notebook 每条 style_note 附 `last_supported_session` + `status ∈ {active, contradicted, stale}`。
- modeler siege schema 加一个必填小字段 `evidence_check`：本 session 的 behav fingerprint/SR 证据**支持/矛盾/无证据**当前战术（一次调用顺带完成，不加预算）。
- 连续 M=4 session 无支持 → status=stale，prompt 强制"重审此战术而非继续 refine"；contradicted → 强制重写（新战术需与旧的实质不同，复用 P1a-L2 的比对代码）。
- 已在 verified_chains 里的 note 不回改（那是攻破时的历史记录）。

**文件**：siege_notebook.py、modeler.py（schema+prompt）。

## P1d 毛刺清理（#7）

- **wandb 双写**：Step 4b 的 `run_session_evaluation` 落盘 cooc/behav 后**不再重复 log `evaluation/*`**（或改前缀 `evaluation_heldout/*`）—— 曲线口径唯一化，这直接关系到你后面判胜负的数据可信度。
- `log.support(name) >= 0` 恒真 → 改为 `>= 1`（意图恢复，MIN_SR 仍是主护栏）。
- （顺带）siege 阈值语义注释订正：写明 SR 序列来源是训练 rollout。

---

## P2 时序链挖掘 + 失败局采集（性能向增强，建议随 fix7 一起上）

**动机**：调研确认共现矩阵丢弃顺序是最大 naive 点；"断链在哪一环"是攻坚焦点选择的最强证据，目前完全缺失。因为要改 eval jit，需要重启才能生效 —— 而 fix7 反正要重启，**建议一起上**。

1. **采集**（craftax_evaluation.py，与 cooc 同一 jit）：per-env 记录 `first_step[67]`（成就首达成步，`jnp.where(newly & (first<0), t, first)`，jit 安全）+ episode 终止步 + 是否 finished。
2. **落盘**（新 auction/chain_order_log.py，与 cooc log 同构持久化）：
   - 成功局：按 first_step 排序的成就序列 → 累积 2-gram/3-gram 有向链计数；
   - 失败局（针对当前焦点：达成了部分前置但没达成焦点的局）：记录"最后达成的环 + 到死/超时为止缺的环" → **断链环分布**。
3. **渲染**（gen_manager → modeler prompt）：焦点的 "CHAIN EVIDENCE" 块 ≤4 行："成功局主导路径 A→B→C→焦点（xx%）；失败局最常断在 B→C（xx%）"。MIN_SR 护栏同 cooc。
4. **用途**：① modeler 的攻坚回溯从"共现相关"升级为"有向断链定位"；② P1a 重开检查的"新证据"判据；③ 论文里是独立可消融组件，也是与 SkillGraph/SCALAR-PTA 的差异化实现。

（可选 P3，暂不做：CODE-SHARP 式 opportunistic replay —— eval 偶然攻破 deep 成就的 episode 指纹加权喂 modeler。等 P2 数据证明有需求再说。）

---

## 实施顺序与验收

1. P0（元数据+validator+prompt 配套）→ 2. P1a → 3. P1b → 4. P1c/P1d → 5. P2 → 全量单测（现 204 个全绿为底线，每个新模块配离线单测）→ sanity 脚本（siege off byte 级不变校验）→ rsync 上机 → **杀 3682819，重启为 fix7（新 wandb run + 新 output dir，从零）**。
2. 监控沿用审计清单，新增：validator 违规/重造率（`[siege][validator]`）、gap 曲线、阶梯级别变迁（`[siege][ladder]`）、断链环分布。
3. 判据不变：make_iron_pickaxe 能否脱离 29% 天花板只是烟雾测试；真判据 = tier3 战斗类 held-out 裸 SR 突破 ~10% + H1 的"后攻的墙爬得更快"。

## 风险与对策

- **validator 误杀**（prereq_tree 是 LLM 推的，可能错）：validator 只 enforce notebook 已有内容，链错误由 P1c 证据机制+P2 断链数据自然修正；fallback 永不断训练。
- **重prompt 循环成本**：至多 2 次/关，API 成本可忽略（用户原则）；违规率本身是监控指标，若居高不下说明 prompt 还有矛盾没清。
- **配额+floor 过强导致课程僵化**：所有豁免都绑定"墙未破"状态，conquest 即撤销；rehearsal_total_cap 式的绝对上限沿用（batch 总量不失控）。
- **归因**：全部改动 siege-gated；P2 数据采集 siege off 时不运行。对 baseline 的对比公平性无损。
