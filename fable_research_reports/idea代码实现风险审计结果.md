# v6fix6 "铁镐死循环"修复 — 代码实现风险审计结果

> 审计日期 2026-07-06。审计对象 = 本地 `dicode_v6/`（含未提交改动，即 v6fix6）。
> 已用 rsync checksum 干跑确认与 Oscar `~/dicode_v6` **逐字节一致** —— 审计的就是 job 3682819
> （wandb `DiCode-v6siege/dicode-v6fix6-s0`，2026-07-06 15:19 EDT 从零起跑）正在跑的代码。
> 审计时 run 处于 session 1，siege 未激活（成熟度闸 ~s11-15），一切结论来自静态代码 + v6fix4 旧 run 数据。

---

## 一、修复链验证结果：三件套全部真实闭环 ✅

（这个代码库有"写好没接线"前科：cooc 采集、rehearsal holder、style_note 本身都犯过。这次逐跳追了调用链，**没有断点**。）

### 1. style_note → proposer prompt：链路闭合

1. **modeler 产出**：siege 系统 prompt 明确要求每个 focus 带 `style_note`（`auction/modeler.py:620-636`）；`_validate_siege` 保留该字段且无长度截断（`modeler.py:261-266, 279-291`）。
2. **入笔记本**：`evolve_mastered_coop` 调 `diagnose_siege` 后把 `siege_update` 喂给 `apply_llm_update`（`src/dicode/dreaming/gen_manager.py:1147-1172`）；`_merge_style_notes` 在 step 1b 和 step 6 两次合并（`auction/siege_notebook.py:628, 661, 489-505`），随后 `_save()` 落盘，resume 原样恢复。
3. **入 SIEGE_DIRECTIVE**：`_render_siege_directive` 读 `foc["style_note"]` 拼出 `ATTACK TACTIC for <skill>: ...` 行（`gen_manager.py:1473-1478`）—— 即 diff 里新增的 hunk。
4. **入最终 LLM 调用**：directive → `extra[pid]["SIEGE_DIRECTIVE"]`（`gen_manager.py:1242`）→ `_build_mastered_prompts` fields（`1628-1632`）→ `_safe_format` 填入 persona 模板 `{SIEGE_DIRECTIVE}` 占位符（`persona_ambitious_coop.py:248`）→ `self.llm.query(...)`（`gen_manager.py:1258`）。

细节：tactic 只对 **ACTIVE** foci 渲染；焦点退休后其 tactic 从 directive 消失（仅存 verified_chains 供 modeler 读，`siege_notebook.py:804-807`）。

### 2. DEPTH-vs-CONSOLIDATE 决策：字段闭环存在，但对 proposer 是纯软约束

- 决定 TYPE 的字段：`guidance_per_parent[pid].recommended_type` ∈ `("DEPTH","BREADTH","CONSOLIDATE")`（`modeler.py:39, 307-309`）。
- 传递：`render_guidance_for_parent` → `"Modeler-recommended level TYPE: X"`（`modeler.py:329-330`）→ `extra[pid]["MODELER_GUIDANCE"]`（`gen_manager.py:1235`）→ persona `{MODELER_GUIDANCE}`（`persona:224-226`）。
- 决策规则文本确认在 modeler system prompt："DEPTH vs CONSOLIDATE — HOW TO CHOOSE"（前置未起=DEPTH / 前置全绿但卡=CONSOLIDATE）位于 `modeler.py:422-429`，siege prompt 经字符串拼接继承（`modeler.py:540`）。
- ⚠️ **代码从不解析/校验 proposer 实际选的 TYPE**（TYPE 只出现在 `<reasoning>` 自由文本里）—— proposer 可自由偏离，无 code backstop。

### 3. CONSOLIDATE=隔离演练 6 处 prompt 改动：全部到位，旧文本清零

LLM 可见 6 处（diff 逐一核对）：persona THREE LEVEL TYPES（`persona:50`）、reasoning point 0（`persona:147`）、user MODELER_GUIDANCE 导语（`persona:218-221`）、user SIEGE DIRECTIVE 块（`persona:228-246`）、modeler CONSOLIDATE 定义（`modeler.py:410-420`）、DEPTH-vs-CONSOLIDATE 规则（`modeler.py:422-429`）。全库 grep `NON-siege|NOT for the hard wall` 零残留。
但有 3 处**未改的模板残留**与新定义打架（见隐患 #1/#2/#3）。

### 4. rehearsal 与 cooc/behav 落盘：均已接通

- rehearsal：`run_dicode.py:179` 调 `append_rehearsal_tasks`；holder 解析 `getattr(gen_manager,"task_generator",None) or gen_manager`（`src/dicode/selection.py:125`），`_siege_notebook`/`_profile_log` 确实挂在 TaskGenerator 上（`gen_manager.py:657, 1075, 2409-2416`）。✓
- cooc/behav：`cooc_names_static` 移到 `make_evaluate` 外层并随元组返回（`src/dicode/craftax_evaluation.py:25-36, 331`，有回归测试 `test_make_evaluate_returns_cooc_names.py`）；`run_session_evaluation` pop 后按同一 holder accumulate（`online_evaluation.py:82-131`）；`run_dicode.py:238-250` Step 4b 以 holder 上的 `_cooc_log` 为闸每 session 调一次。✓
- 3682819 启动日志：SiegeNotebook / CooccurrenceLog / BehaviorFingerprintLog 三件套全部正常初始化。✓

---

## 二、确认的残留隐患（按严重度排序）

### ★★★ #1 superset 规则残留 —— 第 7 处漏改，与隔离演练正面互斥
`persona_ambitious_coop.py:166`："New Task Relevant Achievements: [your list — **must be a valid superset of the trained task's**]"。隔离演练本义 = 把 Relevant 剥到单技能；这条 DiCode 原生模板规则强制子关 Relevant ⊇ 父关 —— proposer 要么违反模板（不稳定），要么把父关战斗/生存目标全带回 drill（Relevant 是终止/成功条件，带上就必须做 → 隔离失效）。

### ★★★ #2 "真采真熔真造"零代码保障，且 prompt 四处自相矛盾
- `unmastered_links` 只遍历各焦点的 `prereq_tree`（`siege_notebook.py:376-393`），而 `_attach_prereq_trees` 把**焦点自身从链里剔除**（`siege_notebook.py:743-744`）→ proposer 把 `MAKE_IRON_PICKAXE` 本身塞进 Completed（初始库存直接给成品铁镐），`enforce_completed_gate` 完全放行；唯一防线是 prompt 一句 "never gift the finished item"（`persona:50`）。上一轮病灶正是这类压缩。
- 闸只搬 unmastered/RISING 链（`completed_gate.py:79-114`；判定 `state != CONSOLIDATED` 即 SR<70）。铁镐前置全绿场景下 collect_iron/coal/furnace 全 ≥70 → 全部合法进 Completed → 初始库存给铁+煤+熔炉，drill 退化为按一次合成键。
- prompt 内部打架：CONSOLIDATE 定义要求 "whole real sequence (mine→smelt→craft)"（`persona:50`）+ reasoning 要求声明 "not gifted via starting inventory"（`persona:147`）；但 GUIDING PRINCIPLE 命令 "Compress away already-mastered prerequisites via the initial state"（`persona:100`）、mandate 命令 "SCAFFOLD them...list them as Completed"（`persona:58`），SIEGE_DIRECTIVE 又逐链标注 "mastered — may be scaffolded/compressed"（`gen_manager.py:1463-1464`）。四段指令对同一情形相反指示，最终行为取决于 LLM。

### ★★★ #3 焦点退休→重开：无冷却、无黑名单，退休可被震荡无限推迟
- 退休慢：`_update_focus_stall` 的 best_sr 是棘轮，`sr >= best+3pp` 即清零 stall（`siege_notebook.py:440-454`）；29-44% 震荡中每创新高 stall 归零，10 连停滞才退休（`456-470`）可被反复推迟 —— v6-OLD 锁死 30+ session 与此机制完全一致，**fix6 未改动此逻辑**。
- 重开无阻拦：退休后 `foci` 空 → expand gate "first wall is always allowed"（`472-487`）；`_reconcile_foci` 只查 scope（SR<80，`412-424, 686-720`）；history 的 `focus_retired_stalled` 事件**只写不读**（`466-469`），`render_for_prompt` 不向 modeler 展示退休史（`758-822`）。
- **fix4 实锤**：siege_notebook.json history：s45 `focus_retired_stalled` → **同 session** `focus_opened` 同一个 make_iron_pickaxe。若 fix 后 SR 仍不升，retire→reopen 循环必然复现。

### ★★ #4 learnability 生态位对 drill 无豁免
- 生成期：`_coop_select` 无 siege/CONSOLIDATE 豁免（`gen_manager.py:1293-1352`）；v6siege 纯 Learnability（`conf/gen_manager/auction_c_v6siege.yaml:55-58`）。新关**没有自己的 p** —— lrn 用父关 archive 的 p(1-p)（`auction/learnability.py:33-57`；`gen_manager.py:1657-1685`），杀伤模式 = **按父系连坐**（砍 lrn 最低父系的整对候选，含从未训过的父=0）；平手时 GreedyTopK 严格 `>` 偏向池序靠前（`auction/selectors.py:151-159`）。
- 训练期（真正咬人处）：drill 训到高 SR 后 `priority_score = p(1-p)` 归零（`src/dicode/training.py:386-389`）；CAS 激活以首训 p(1-p) 与 worst active 比分，饱和快的 drill 可能进不了 active set（`src/dicode/evolution_efficient.py:49-67`）；PLR rank 采样系统性降权，仅 staleness_coeff=0.3 兜底（`selection.py:63-88`）。**"drill 学会了就被扔" vs "held-out 传导需要继续干净重复" 结构性冲突，无任何 siege 豁免。**

### ★★ #5 闸搬运链目标但不改 World → 可能制造不可解关
`enforce_completed_gate` 只重写 Relevant/Completed 两行（`completed_gate.py:112-114`），不碰 World/初始库存描述。proposer 按"该链已压缩"设计的世界（如没放可达铁矿）被强制加上 Relevant `COLLECT_IRON` 后可能无解（Relevant 即终止条件）。只有 prompt 侧软约束（`persona:129-130`）。

### ★ #6 style_note 无质量闸、无回滚，且有自我强化回路
非空即覆盖、无验证（`siege_notebook.py:489-505`）；错误战术随 `_record_experience` 进 verified_chains（`551-583`），`render_for_prompt` 以 "style-so-far" 喂回 modeler（`777-778`），prompt 要求 "refine, don't restate"（`modeler.py:633-636`）—— 错误认知可跨 session 固化并经 ATTACK TACTIC 同时毒化 proposer。唯一纠偏 = behav_hint 真实行为证据（`modeler.py:625-629`），但目标 SR<MIN_SR=3% 时为空 —— **最卡的墙恰恰没有行为证据可纠偏**。note 无长度上限。

### ★ #7 其他毛刺
- `log.support(name) >= 0` 恒真（`gen_manager.py:1384`）—— 过滤形同虚设，靠 `render_prereq_hint` 的 MIN_SR 兜底。
- **wandb 双写**：每 session `evaluation/*` 写两次 —— 训练口径（`run_dicode.py:215-222`）+ Step 4b held-out 口径（`online_evaluation.py:133-137`），曲线分析会被污染；且 siege 路径每 session 多付一次 1024env×8192step held-out rollout。
- 阈值带冲突：前置落在 20-70（RISING）时 directive/gate 强制进 Relevant → "隔离"演练被迫携带整条链目标，drill 与 DEPTH 形态趋同。
- 数据口径：siege 的 SR 序列来自训练循环 original_craftax 的**训练 rollout**（`training.py:159-161`），非 held-out；阈值语义按 held-out 措辞（沿 v5 口径，非新 bug）。
- 首 session 时序：notebook/_cooc_log 后台 worker 惰性创建，第一 session 主线程读到 None 静默跳过（良性）。

### ★ #8 conquest 疑似过早（fix4 数据观察，未读码确认）
fix4 notebook：make_iron_pickaxe 在 s25、SR 44.3% 时已进 verified_chains + protected_set（mastered 阈值 70；疑 record_delta_pp=10 跳变触发记录）→ **"已验证链条库"可被未攻破技能污染**，tier4 复用地基时会踩。

---

## 三、不确定 / 需要 run 数据才能判断的点

1. pure-lrn cull 的实际连坐对象：看 `[coop][select]` 被剔候选的 parent 分布（#4 前半）。
2. drill 生命周期竞赛：drill 训练 p 饱和速度 vs held-out SR 传导速度 —— 看 drill 关 `priority_score` 轨迹与 `is_active` 翻转（#4 后半）。
3. LLM 实际服从哪条矛盾指令：抽查生成 docstring —— CONSOLIDATE 关 Relevant 是否真剥离（vs superset 规则）、mastered 前置是否被 gift（#2/#1 实际发生率）。
4. retire→reopen 是否再发生：`[siege][focus-decision]` 序列 + notebook history 同名事件间隔（#3）。
5. style_note 是否向 behav 证据收敛：焦点 SR<3% 期间 note 是否停留在"想象战术"（#6）。
6. wandb 面板双写污染核对（#7）。

---

## 四、总评与行动建议

三条声称的修复**代码层全部属实且链路闭合**，cooc/behav/rehearsal 三条历史断线确已接通。但"真采真熔真造"的隔离演练目前**只有 prompt 保障、没有代码保障**，且被 superset 规则、scaffold 总纲、"mastered 可压缩"标注三处旧文本正面对冲；焦点生命周期（棘轮退休+无冷却重开）与 learnability 生态位（drill 无豁免）是下一轮最可能复现"铁镐式僵局"的两个结构点。

**建议**：趁 siege 未激活（~s11-15）杀掉重启，先打三个低成本补丁 —— ① superset 规则给 CONSOLIDATE 豁免；② completed_gate 加"焦点自身禁入 Completed"硬闸；③ 退休冷却 + 退休史渲染进 modeler prompt。#4（PLR 豁免）动训练采样层、有偏离对标风险，先监控确认再动。

**监控清单**（siege 激活后）：①抽查 CONSOLIDATE 关 docstring（Relevant 是否单技能 / Completed 是否 gift 成品或全链材料）；②`[siege][focus-decision]` + history 看 retire→reopen；③drill 关 priority_score/is_active；④`[coop][select]` 被剔父系分布；⑤style_note 演变。
