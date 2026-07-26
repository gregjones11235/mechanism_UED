# v6 SIEGE 阈值合理化方案（2026-07-05 讨论产物）

> 背景：4.1 超参表指出 siege_notebook 的 9 个阈值原是"按 80% mastery 设计"，但学生在
> tier3 真实天花板只有 ~12%，导致 siege 可能空转（focus 永不 conquered / protected set 永远空 /
> rehearsal 永不触发）。本文是把阈值**重新对齐 v6 因果结构**的方案。

## 0. 推导起点（不用旧数据锚定）

**关键纠正（用户 2026-07-05）**：不能用 baseline/v1 在 tier3/4 的历史 SR 去锚定阈值——那些数据是
"方法失效"状态下产生的，用它锚定 = 把失败固化进阈值。正确起点是**推导 v6 生效后的预期轨迹**：

1. **阶段1**：少量 tier3 **装备**成就（enabler：diamond gear、iron sword…）被达成 + 巩固。
   装备类不受对抗天花板压制，巩固后 SR **能到较高**（50-70%+）。
2. **阶段2**：某**一个** tier3 **战斗**成就被**第一次打成**（focus 攻坚的历史性成果）。
   第一次打成时 SR 可能只有 **10-25%**（偶尔赢）。
3. **阶段3**：**同一个**战斗成就被**越攻越熟**（15%→40%→更高），并向其它战斗成就迁移
   （自我风格 H1，一通百通）。

**关键纠正（用户 2026-07-05，第二轮）**：**不存在"战斗类因对抗天花板只能到 ~40%"这回事**。
人类高手打困难 boss 也能接近 100% 胜率——低 SR 是"当前方法还没教会"，不是"能力上限"。
所以：**战斗类的目标 SR 不设低天花板**，saturated 等"已经很强"的判据战斗/装备**同样设高**（见 §2③）。
战斗类唯一特殊的地方是**起步阶段** SR 低（第一次打成才 10-25%）——这只影响"识别第一次攻破"的
**起点判据**，不影响"最终能到多高"。

**核心洞察**：阈值不该问"学生能不能到 80%"，该问"这条轨迹的每个转折点，用什么信号能被**正确识别**"；
且要区分**"识别第一次攻破"（低起点信号）**与**"判定已经很强/该退休"（高水平信号）**两类判据——
前者用低地台起点，后者战斗/装备一律用高值。

## 1. 两个已定型的设计决策（用户拍板）

- **决策 A — 形状判据，不用绝对低阈值**：conquered / protected / rehearsal 的判定，看
  "**从 ~0 稳定爬到一个非零地台并保持 K 个 snapshot**"的**形状**，而不是单点 SR 过某个绝对线。
  理由：eval num_envs=1024，单次 SR 有噪声；"偶尔蒙对一次" ≠ "真正学会了"。形状判据抗噪且更贴合
  "真正学会"的语义。
- **决策 B — 分档轴是"起步识别 vs 高水平判定"，不是"战斗 vs 装备天花板"**（用户 2026-07-05 第二轮修正）：
  - **战斗类唯一特殊处 = 起点低**：第一次攻破只有 10-25%，所以**"识别第一次攻破"的起点地台**用低值
    （~15%）。这**不是**说战斗类到 15% 就到顶。
  - **"已经很强/该退休/是地基"的判定，战斗与装备一律用高值**（saturated、mastered 巩固都朝 80% 靠）——
    因为战斗类同样能越攻越强、没有 40% 天花板。
  - `skill_family()` 仍用来分类**成功经验条目**（combat_milestone vs enabler，见 §2.5），而**不是**用来给
    战斗类整体降天花板。
  - 代码前提已具备：`craftax_achievements.skill_family()` 已把技能分成 COMBAT/GATHER/CRAFT/EXPLORE。

## 2. 逐阈值方案（值 + 推导依据）

### ① CONQUERED（重构：不再是"到阈值退休"，改成"增量触发经验记录 + 不强制退休"）
- **病 1（旧）**：原 conquered_sr =MASTERED_SR=80。第一次 tier3 战斗攻破时 SR 仅 10-25%，
  永远判不出 → focus 卡死在第一个 wall。方法自己否定自己的成果。
- **病 2（用户 2026-07-05 第二轮指出的更深问题）**：把 conquered 当成一个**一次性阈值门**
  （过了就退休、进 protected、不再当 focus）本身就错。**conquered ≠ saturated**：
  第一次打到 15%、第二次打到 40%，**第二次的进步同样值得记录**——这正是 H1"越攻越熟"。
  一到阈值就冻结，会把一个还能继续变强的战斗技能过早退休。
- **方案（增量触发 + 去重 + 不强制退休）**：
  1. **不再有"conquered 阈值门→强制退休"**。focus 是否换，只由 §2⑤ 的 stall 判据决定
     （连续 N session 不涨才换）。一个墙可以一直攻到 stall 为止，越攻越熟。
  2. **经验记录由"增量"触发**：当 focus 的 SR 相比**该 target 上次记录时**上涨
     **≥ `record_delta_pp`（默认 10pp）**，就写/更新一条成功经验（见 §2.5）。
     - 15%→40% 触发（+25pp，真实进步，记）；
     - 15%→14%→16% 抖动不触发（达不到 +10pp，噪声，不记）——**天然去冗余**。
  3. **保护/rehearsal** 仍要覆盖已记录的战斗里程碑（见 §2⑥ 联动）。
- **落地**：删掉 `_focus_conquered`/`_retire_conquered_focus` 的"到阈值即退休"语义，
  改为在 apply 时比较"该 target SR vs 上次记录 SR"，达增量则调用 §2.5 的经验记录逻辑；
  focus 退休交回 stall 路径。

### ② MASTERED_SR / UNMASTERED_SR（链接 CONSOLIDATED / UNMASTERED 判定）
- **MASTERED_SR（巩固）**：分档。
  - 装备/enabler 链接：`mastered_craft = 55%`（enabler 该真学会，标准可以高）
  - 战斗类链接：`mastered_combat = 20%`（战斗类"巩固"= 站稳低地台，同 conquered 逻辑）
- **UNMASTERED_SR**：**基本不改**，方向本就对（防"偶尔蒙对一次就当掌握、跳过前置"）。
  可略放宽 `20% → 25%`，在低天花板场景多防一点假掌握。这个不是杠杆点。

### ③ SATURATED_SR（scope：太易的技能不能当 focus）
- 解绑，独立设。作用是把"已是地基"的技能排除出 focus 候选。
- **战斗与装备一律设高（用户 2026-07-05 第二轮拍板）**：`saturated = 80%`，两类同值。
  理由：战斗类没有 40% 天花板，到 40% 远不算"饱和到不必再攻"——高手能接近 100%。只有真正
  站到 80% 高位，才算"这个技能已是地基、不该再占攻坚预算"。
- （注意：saturated 是"排除出 focus 候选"，与 §2① 的"经验记录/退休"是两回事——focus 攻到很高
  但没到 80% 时，仍可继续攻、继续记增量经验，只是不会因 saturated 被踢出候选。）

### ④ MATURITY_*（决定 siege 何时开——防"永不开"或"早开误判"）
- **病**：需 12 个技能 SR≥50 才"成熟"。若学生长期只有 ~10-15 个技能能到 50%，siege 开启过晚甚至永不。
- **方案**：
  - `maturity_skill_sr: 50 → 35`（放宽"算一个成熟技能"的线，让基础盘计数更快满）
  - `maturity_min_mastered: 12 → 10`（略降，配合上一条）
  - `maturity_min_snapshots: 4`（**不改**，4 个快照才敢判 wall 是合理的抗噪下限）
- 目标：siege 在**阶段1刚起步、阶段2还没到**时就能开——因为 siege 本身就是促成"第一次攻破"的手段。

### ⑤ FOCUS_MIN_STALL_SESSIONS / FOCUS_IMPROVE_PP（防漂移 / stall 计数）
- **FOCUS_IMPROVE_PP**：原 3.0pp。tier3 在 12% 噪声带振荡时，±3pp 可能全是噪声 → stall 计数不可靠。
  - 方案：**改用相对判据 + 抗噪窗口**：把"进步"定义为"最近 K 个 snapshot 的 SR **中位数**比
    focus 开始时**高 ≥ 2pp**"，而非单点 diff ≥ 3pp。降低噪声误触发 stall/reset。
  - 若嫌改动大，退而求其次：`focus_improve_pp: 3.0 → 2.0` + 用平滑后的 SR。
- **FOCUS_MIN_STALL_SESSIONS**：`4` **不改**（连续 4 session 不涨才允许换 focus，防漂移，合理）。

### ⑥ forgetting min_peak / drop_pp（rehearsal 触发——与"已记录经验"联动）
- **病**：`min_peak=40`。阶段2/3 的 tier3 战斗成就峰值可能就 15-25%，一旦滑落 rehearsal **永不救**——
  而它们恰是最该保护、最易在 focus 转移后遗忘的。
- **铁律**：**凡被写进成功经验（§2.5）的技能，都必须能触发对它的 rehearsal**。所以 `min_peak` 必须
  ≤ 触发经验记录的起点（战斗类 ~15%）。
  - 分档：战斗类 `forgetting_min_peak_combat = 15%`；装备类 `forgetting_min_peak_craft = 40%`。
  - `drop_pp`：战斗类 `10pp`（低地台上掉 10pp 已是明显滑落）；装备类保持 `20pp`。

## 2.5 成功经验笔记：分类 + 增量触发 + 去重（用户 2026-07-05 第二轮提出）

**动机**：现状 `_retire_conquered_focus` 每次过阈值就往 `verified_chains` **无条件 append 一条**
`{target, links, conquered_session, evidence}`，注入 modeler prompt。三个冗余病：
1. 同一 target 因 SR 噪声反复过阈值 → 多条近乎重复记录；
2. 攻 A / 攻 B 的前置链高度重叠（都下矿、造铁镐）→ links 大面积重复；
3. 全平铺、无"重点"维度——**战斗里程碑（H1 风格来源）和装备 enabler（脚手架）混在一起**，
   LLM 找不到重点。

**方案（三合一，全部作用在 `verified_chains` 的写入路径）**：

- **(a) 分类** `category`：写经验时用现成 `skill_family()` 判 target——
  - COMBAT → `combat_milestone`（**H1 自我风格来源，注入时是"重点"分区**）
  - CRAFT/GATHER/EXPLORE → `enabler`（**脚手架/地基，注入时是"背景"分区**）
- **(b) 增量触发**（替代"到固定阈值"）：只有当该 target SR 相比**上次记录时**上涨
  **≥ `record_delta_pp`（默认 10pp）** 才写。噪声抖动 < 10pp 不写 → 天然去冗余；
  真实进步（15%→40%）写，且体现"越攻越熟"。
- **(c) 按 target 去重**：同 target 已有条目则**更新那一条**（覆盖 evidence、
  刷新 `last_recorded_sr` / session、可保留 `first_recorded_sr` 看成长），**不 append 第二条**。
  → 笔记里每个 target 永远至多一条，`defeat_zombie` 的 evidence 从"SR 15%"升级到"SR 40%，越攻越熟"。

**新条目结构**（扩 `verified_chains` 元素）：
```
{
  target, links,
  category: "combat_milestone" | "enabler",
  first_recorded_sr, last_recorded_sr, last_recorded_session,
  evidence,                # 人读串，随增量更新
}
```

**注入 modeler prompt 时分两区**：
- 「**已攻破的战斗里程碑**（重点·自我风格来源）」← category==combat_milestone
- 「**已具备的 enabler 地基**（背景·勿当风格）」← category==enabler

（可选进阶，非本轮必做）links 共享 enabler 全局去重：跨 chain 重复的下矿/造铁镐只在笔记留一份，
chain 只引用——留到 §2.5 落地后视 prompt 长度再决定。

## 2.6 多 focus：并行攻坚 + 成就间互助（用户 2026-07-05 第三轮提出）

**动机**：单 focus 一次只攻一个墙。但当一个墙已经打得不错（SR 起来了），学生有余力**分心**去带一个
新的成就；且成就间可互助——**升级装备利好后续所有攻关**，同时推进多个战斗成就也能共享打法迁移，
提高整体攻克效率。所以从"单 focus"放开到**同时最多 3 个 focus**。

**规则（用户拍板）**：
- **上限**：`max_focus = 3`，同时至多 3 个 active focus。
- **开新 focus 的解锁闸**：**任一**现有 focus 的 SR **≥ `focus_expand_sr`（默认 50%）** 时，允许再开一个
  新 focus（直到 3 个）。"有一个打得不错了 → 才有余力分心带新的"。
  - 用户明确：50% 这条闸看**任一**现有 focus 达标即可（不是要求全部达标）。
- **新 focus 类型**：**任何 tier3+ 成就都可以**（含装备类）——用户理由"升级装备有利于攻关"。
  （注意：这是"能不能当 focus"的规则；§2.5 成功经验分类仍按 skill_family 把 combat 与 enabler
  分区，两者不冲突——focus 类型放开 ≠ 经验记录不分类。）
- **退休**：每个 focus 独立按 §2⑤ stall 判据退休；退休腾出的槽可被新 focus 补上。

**数据结构改造**（把单标量 focus 升成 focus 列表）：
- `focus: str|None` → `foci: list[dict]`，每个元素：
  ```
  { "skill", "started_session", "best_sr", "stall_sessions" }
  ```
  （原来散在顶层的 focus_started_session/focus_best_sr/focus_stall_sessions 收进每个 focus 元素，
  变成 **per-focus** 计数。）
- `prereq_tree`（单棵）→ **每个 focus 自带一棵 `prereq_tree`**（放进 foci 元素）。
  **落地简化（2026-07-05 实现时定）**：不单独建"共享链池 + refs"——因为唯一需要跨 focus 合并的消费方
  是 Completed 闸（`unmastered_links`），它本来就用 **set 并集**，逐 focus 遍历再并集即可，共享池的
  唯一收益（enabler 去重）在消费端一个 `set()` 就解决了，不值得为它改 schema。装备 enabler 若同时
  是多个 focus 的前置，会在各自 tree 里各存一份 link（几个字节），但注入/闸都按并集去重，无副作用。

**prereq 链的消费方（决定为何用共享池）**：
1. **§3.4 Completed 准入闸**（`unmastered_links`）——问"当前**所有** active focus 的前置里，哪些还
   没掌握"，要的是**合并集合**，不关心前置归属哪个 focus。→ 改成遍历所有 focus 的 prereq_refs 并集。
   `completed_gate.py` 本身**不用改**（它只吃一个 unmastered set）。
2. **conquest/经验沉淀**（§2.5）——需要知道"某 target 的链是哪些 links"，从该 focus 的 prereq_refs 取。
3. **modeler prompt 注入**——按 focus 分别渲染各自的链。

**受影响的方法（单 focus → 多 focus 全部要改）**：
- `_empty_notebook`（schema）、`_coerce`（旧单-focus 笔记的向后兼容迁移：把旧 focus 包成单元素 foci）、
- `focus` property → `foci` / 增 `active_foci()`、
- `unmastered_links`（并集）、`_refresh_link_flags`（刷共享池）、
- `_is_valid_focus`（不变，逐个候选判）、`_update_focus_stall`（改成遍历 foci 各更各的）、
- `_may_switch_focus` → per-focus stall 判 + `_may_open_new_focus`（查 max_focus 上限 + focus_expand_sr 闸）、
- `_focus_conquered`/经验记录（§2.5，per-focus 增量）、
- apply 第4步 focus 变更：接受 LLM 的 foci 增删提议，逐个过 maturity+scope+（新增）expand 闸+上限。

**向后兼容**：`_coerce` 检测到旧 schema（有顶层 `focus`）时迁移成 `foci=[{...}]`，保证旧笔记/旧单测
不炸。siege off 路径完全不碰这些，保持 byte 级不变。

### ⑦ modeler_recent_k（时间序列窗口，已可调）
- `6` 暂不改。它影响 forgetting/stall 判定的灵敏度——等形状判据落地后再回看是否要加长窗口配合。

## 3. 落地改动清单（供后续实现）

1. `SiegeThresholds`：扩键——
   - 新增 `record_delta_pp`（默认 10，§2.5 增量触发）、`saturated_sr` 统一 80（§2③）、
     forgetting min_peak 拆 combat/craft、mastered 拆 combat/craft（§2②⑥）、
     maturity 三键放宽（§2④）。
   - **删** `conquered_sr` 的"到阈值退休"语义（§2①）。
2. **conquered 重构**（§2①）：删 `_focus_conquered`/`_retire_conquered_focus` 的阈值退休逻辑；
   focus 退休只走 stall 路径（§2⑤）。
3. **成功经验写入路径重写**（§2.5，回答"冗余/找不到重点"的核心）：
   - 增量触发：apply 时比较 target SR vs `last_recorded_sr`，≥ `record_delta_pp` 才写；
   - 按 target 去重：同 target 更新那一条，不 append；
   - 分类：写时 `skill_family()` 判 `category`（combat_milestone / enabler）。
4. **modeler prompt 注入**：`verified_chains` 按 category 分两区渲染（重点战斗里程碑 / 背景 enabler）。
5. `saturated` scope 判据：战斗/装备统一 80%（§2③）。
6. **多 focus 重构**（§2.6）：schema 升 foci 列表 + 共享链池；`_coerce` 向后兼容迁移；
   stall/refresh/unmastered_links/apply 全部 per-focus 化；新增 `focus_expand_sr`(50)、`max_focus`(3)
   两个阈值与 `_may_open_new_focus` 闸。
7. config yaml：把新键全部暴露（延续 2026-07-05 "全可调"原则）。
8. 单测（判别性回归，旧逻辑必挂、新逻辑必过）：
   - 低起点战斗技能（峰值 15%）**能被写进经验**且能触发 rehearsal；
   - 同 target 从 15%→40% **只留一条**、evidence 升级、category==combat_milestone；
   - 15%→14%→16% 抖动**不产生**新记录（增量去冗余）；
   - enabler 与 combat 条目在注入时分属不同区；
   - **多 focus**：任一 focus 到 50% 才解锁开第 2 个；至多 3 个；旧单-focus 笔记 `_coerce` 迁移不炸；
     Completed 闸取所有 focus 前置并集；siege off 路径 byte 级不变。

## 4. 一句话总结

旧阈值把"学会"等同于"到 80% 就退休"，在战斗类**起步只有 10-25%** 的现实里 siege 卡在第一个墙、
成功笔记还堆满重复流水账。新方案分三刀：
1. **识别 vs 判定分离**——"识别第一次攻破"用低起点（~15%），"已很强/是地基"战斗装备一律用高值（80%），
   **不给战斗类设 40% 天花板**（高手能接近 100%）。
2. **conquered 从"阈值门退休"改成"增量触发经验记录"**——SR 每涨 ≥10pp 记一次"越攻越熟"，
   focus 攻到 stall 才换，不过早冻结还能变强的技能。
3. **成功经验分类 + 去重**——按 `skill_family()` 分 combat_milestone（H1 重点）/ enabler（背景），
   同 target 只留一条并随增量升级 evidence，让 LLM 一眼看清主次、不被冗余淹没。
