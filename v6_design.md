# v6 设计文档 — 起点:v5 的核心负结果(scaffold 切断 tier3 长链→不可迁移)

> 起草 2026-07-04。v6 的设计动机 = 诊断出 v5(modeler + 合作补位 + Amb/Lrn 筛选)为何仍未突破
> tier3 天花板。本文档先固化 v5 的实测负结果与失败模式,再给改进方向。数据来自 v5yA(24→18)
> 的 wandb 曲线 + task_graph.graphml(session 62-73,108 关)亲验。

---

## 1. 现象:三方(v5yA / v5yB / baseline)全部卡在共同天花板

公平同 step 对齐(裸 tier SR,非增量),step 6319-7033:

| tier | baseline | v5yA(24→18) | v5yB(24→12) |
|---|---|---|---|
| tier1 | ~89% | ~87% | ~85% |
| tier2 | ~66-71% | ~68% | ~64% |
| **tier3** | **~10%** | **~10%** | **~9%** |
| **tier4** | **0%** | **0%** | **0%** |

- **三方在 tier2/tier3 咬死**,mean_return(37-42)剧烈振荡且互相穿插,**排名无意义**(纯噪声 + 浅成就饱和抖动)。
- 共同天花板 = **tier3 ~10% + tier4 = 0**。谁都没捅破。v1 参照:最终也卡 tier3 ~12%/tier4 0(§10.7)。
- mean_return 振荡的根因(推测已证实,见 §2):课程全砸在攻不破的 tier3 墙上,student 反复尝试深层、
  成功率低且不稳 → 分数抖动。

> ⚠️ mean_return 已在此阶段**饱和失效**:它是"加权解锁成就数 + 少量血量项"(craftax_evaluation.py:
> mean_performance = mean_return/226*100),被大量已饱和的 tier1/tier2 浅成就稀释,看不出深层突破。
> 判胜负只能看 **tier3/tier4 的裸 per-skill SR**。

---

## 2. ★★★ 核心负结果:v5yA 83% 关卡在攻 tier3,但 scaffold 方式破坏了 tier3 的完整长链条

### 2.1 v5yA 现在几乎全在攻 tier3(graphml 亲验)

最近 12 session(62-73)造的 108 关,目标 tier(Relevant Achievements 里最深的)分布:

| 目标 tier | 关卡数 | 占比 |
|---|---|---|
| tier3 | **90** | **83%** |
| tier2 | 17 | 16% |
| tier4 | 1 | ~1% |
| tier1 | 0 | 0 |

TYPE 标记:DEPTH 16 / CONSOLIDATE 4 / BREADTH 2。**课程绝大多数在 DEPTH 攻 tier3**,而 tier3 SR 卡 ~10% 久攻不破 → mean_return 振荡。**modeler 诊断没错(该攻 tier3),问题在 proposer 的 scaffold 方式。**

### 2.2 失败模式:scaffold 把 tier3 长依赖链切成"孤立的最后一跳"

graphml 逐关证据(session 73):

- **task_666**(目标 defeat_orc_mage/solider,纯 t3 打精英怪):scaffold(Completed)只给 tier1 一堆
  + enter_dungeon,**没给**打精英怪所需的铁装/钻石装、到达 orc 层级的完整路径、生存物资 → student 被
  "空投"到 orc 面前只练"打这一招"。
- **task_670**(目标 enter_gnomish_mines + collect_diamond):scaffold 给了铁镐/铁,把 collect_diamond
  拆成"给你铁镐你只管挖钻石" → student 学"有铁镐时挖钻石",但到不了钻石层/铁镐半路丢/死在半路。

**机制**:scaffold 为了让**单关可解**,把 tier3 的长依赖链(到达层级 → 装备 → 维持生存 → 执行招式)
**压缩成孤立的最后一跳**。student 在每个孤立跳上练得还行(所以 tier3 SR 有 ~10% 而非 0),但
**held-out 真实 Craftax 没有 scaffold** —— 它必须自己串起整条链,而中间环节(到达/装备/生存)student
**从没在完整情境里练过** → 串不起来 → tier3 不可迁移。

**一句话:scaffold 治好了"可解性",却制造了"不可迁移性"。** 学会接精英怪一招,但忘了/还不会:到达它
所在层级、造钻石装、维持生存 —— 用户 2026-07-04 的假设,graphml 逐条印证。

### 2.3 与 v1 §10.7 的关系(升级版失败模式)

- v1(§10.7):过早推 tier3 → tier2 铁器链因灾难性遗忘倒退(make_iron_pickaxe 斜率 −0.68)。
- v5:modeler **防住了** v1 的遗忘(v5yA tier2 铁器链在升:make_iron_sword +25、pickaxe +18,不倒退),
  且过早推的问题也缓解了。**但暴露出更深一层的问题**:即便正确地推 tier3、且 scaffold 保证可解,
  **scaffold 的"切成最后一跳"方式本身破坏了 tier3 迁移所需的完整链条**。
- 结论:病根从"课程锚点太激进"(v1)转移到"**scaffold 粒度太激进——压掉了 tier3 迁移必需的中间链条**"(v5)。

---

## 3. v6 设计:聚焦攻坚课程 (Focused Milestone-Chain Curriculum)

> 本节为 2026-07-04 与用户 brainstorm 收敛的正式设计(纯设计,未写代码)。核心 = 把 v5 的
> "modeler 诊断状态" 升级为 "modeler 主动规划一条攻坚路径并守护它",像真人玩家卡关时那样
> **聚焦攻透一个难关→沉淀可迁移风格→固化不遗忘→在稳固地基上挑战更深**。

### 3.0 先厘清 baseline 为什么会切断长链(读 DiCode.pdf 原文,2026-07-04)

不是 bug,是**针对短链的正确设计被无意识套用到长链**:

- **机制**:DiCode 把成就分两类(原文 p23):`Relevant Achievements`(本关必须主动达成=终止/成功
  条件) vs `Completed Achievements`(初始状态隐式满足、agent 不该再做的前置)。"压缩长链" =
  把前置塞进初始库存、标为 Completed。原文明确指令(p21):"Initial state is a tool to **compress
  away already-mastered prerequisites**... initialize inventory consistent with 'an agent that
  reached here competently', so training **focuses on the NEW dependency**"。
- **三条设计理由(都合理)**:① 信用分配/探索——长链稀疏奖励,RL 从零走不完,压掉前置留"最后一跳"
  才学得动;② thin-slice + one-main-change(p22/p24)——每次只改一处、避免难度尖峰、保证关卡
  存活(太难的被 learnability 淘汰);③ ZPD——靠此把 avg SR 稳在 ~0.5。
- **关键**:这套在**短链**(iron armour 45%、进 floor2 30%)有效——压掉的前置 student 真实游戏里
  本来也会(tier1/2 已 90%+),能自己补→迁移。但**长链**(gnome warrior、diamond sword)失效——
  中间环节(到达深层/装备/生存)student 真实游戏里**也不会**,压掉后 held-out 没人给它→补不上→
  不迁移。**DiCode 原文自己 tier3/4 也只 ~10%**(gnome warrior 11%、diamond sword 6%,p6),它把
  0%→10% 当成功就停了。**v5 完美复现了 DiCode 的天花板,因为继承了同一 scaffold 判断。捅破这个
  DiCode 自己都没捅破的天花板 = v6 的 novelty 与对标价值。**

### 3.1 核心假设 H1:战斗类攻坚沉淀可迁移"自我风格"

- **自我风格 = student 在攻克战斗类 tier3 过程中习得的一套动态策略**(走位、gear-up 时机、拉怪、
  逃生、深层导航+资源管理的组合技)。**只有战斗类产生风格**——打赢强精英怪需要动态决策;装备类
  (钻石剑/甲)是确定性 crafting,是让战斗**可尝试**的钥匙(enabler),本身不产生风格但不可或缺。
- **H1**:一次攻透一个战斗目标(含其完整前置链)→ 沉淀可迁移的风格内核 → 迁移到下一个战斗目标使
  其加速。平摊 23 个 tier3(v5 的病)→ 每个风格都残缺 → 都卡 ~10%。
- **可验证**:攻克顺序靠后的战斗类 tier3,其 held-out SR 爬升应更快(风格迁移的信号)。这是 v6
  论文的核心论点。
- **★风格的物理载体 = 笔记的 `style_note` 字段(§3.5)**:这套"动态策略"本质是文字性经验,必须有字段承接
  才能跨 session/跨目标迁移。见 §3.5 的 style_note 实现——没有它,H1 的"沉淀风格"在数据结构层面是空的。

### 3.2 攻坚单位与排序:战斗目标 + 自适应回溯的攻坚树

- **攻坚单位 = 战斗类难关目标**(一次只攻一个;且 H1:攻越多风格越成熟、后续越易)。装备类**不是独立
  攻坚目标**,是某个战斗目标前置链里的 enabler 环节,跟着它服务的战斗目标一起被纳入关卡。
- **战斗/装备二分直接告知 LLM**(合法:是成就的类别标签 COMBAT/CRAFT,非课程链先验;Craftax 成就
  本就分类,见 craftax_achievements.FAMILIES)。

- **★难关如何判定 = modeler 从 SR 自己"感到卡关",不查 tier 表(用户 2026-07-04 拍板,关键)**:
  就像玩家自己感觉到"卡关了"才开始攻坚。modeler **不吃**成就→tier 映射(那是我们引入的弱先验,
  见 §3.3),而是从 **student 的 held-out achievement SR + (b)机制** 自己推深浅、自己判"哪个是当前
  卡住的难关"。判定信号 = 某战斗成就 **SR flat-near-zero 或长期低位不升**(=真难关),其**机制前置
  已大体具备但该成就仍攻不破**(=卡在这一关,不是还没到)。
  - **★作用域硬约束:攻坚机制只服务"让 baseline/v5 性能振荡的难关",绝不对 tier1/2 简单关做攻坚。**
    判据(不靠 tier 表,靠 SR):student 已**高 SR/饱和**的成就(collect_wood 等 90%+)→ modeler **不得**
    把它列为攻坚目标或围绕它造攻坚关(它不是难关,是已掌握的地基)。攻坚只针对 **flat-near-zero 且
    student 反复尝试不成的深层战斗成就**——正是 §1 里三方振荡的那面墙。简单关继续走 baseline 的
    正常 evolve(EXPAND/VARY),不进攻坚流程。→ 防止把火力浪费在已饱和成就上(那既无收益又扰动)。
  - **★★早期误判防护(用户 2026-07-04,关键补充):"SR 低"有两种成因,只有一种是难关。**
    早期 tier1/2 阶段 student 整体很弱,**简单成就(make_wood_pickaxe 等)SR 也 flat-near-zero**——
    若只看"SR 低不升"就判卡关,会把简单早期成就误当难关攻坚(违反上条作用域,还扰动早期正常爬坡)。
    必须区分(= modeler prompt 里 NORMAL_EARLY vs STALLED 判据,现已明确接进 focus 选择):
    - **(A) 真难关(该攻)**:student 已**过早期**(很多成就已中高 SR)、该成就**机制前置大体具备**,
      却仍 flat-near-zero → 卡在真墙(§1 那面 tier3 墙)。
    - **(B) 早期未成熟(绝不攻,走 baseline)**:student 整体还弱(达标成就少)、该低分成就只是**还没轮到**
      → NORMAL_EARLY,耐心等正常课程。
    - **A+B 双保险落地**:①**A层 prompt** 教 LLM 用上述 litmus(student 是否已整体胜任+是否前置已具备)
      判 focus,拿不准/看着早就**不设 focus**;②**B层代码硬闸**(SiegeNotebook `_student_is_mature`,
      不可违反,纯 SR 统计不查 tier 表):student 未跨过**成熟度阈值**(快照数 ≥ MATURITY_MIN_SNAPSHOTS
      且达中高 SR 的成就数 ≥ MATURITY_MIN_MASTERED)前,**拒绝任何 focus**(focus 保持 None,全程走
      baseline evolve)。→ 即便 LLM 误判要攻简单关,代码兜底挡住。与既有 SATURATED_SR 作用域闸同源。

- **攻坚不是线性队列,是"攻坚依赖树"**:tier3 内部是**空间依赖交织的 DAG**(证据见 §3.3),装备后置
  于战斗普遍存在。modeler 选定一个战斗难关 → 沿 (b)floor/空间依赖回溯 → 找"当前前置都已固化的
  最深可攻点"先攻。粒度 = **student 实际卡点,非 tier 标签**(神似 DiCode frontier,但更细)。
- **★自适应回溯(展开规则,用户 2026-07-04 拍板)**:回溯到每一环时看 student 该环的 held-out SR:
  - **rising / mastered 的环** → 不单独攻,**可合法压进 Completed**(真掌握,压掉不伤迁移)。
  - **flat-near-zero 的环** → 展开成子攻坚目标,**必须留在关卡里单独训**(未掌握,压掉必复发 v5 病)。
  - 好处:已掌握的环自动截断(不无限回溯),未掌握的环一定单独训(不复发切链病)。
  - 注意:这里的"深浅"也是 modeler 从 SR + (b)机制自推,不查 tier 表(与难关判定同源)。

### 3.3 攻坚链条的推断:(b)机制依赖图 + (c)student 共现,禁 (a) 课程先验

**知识泄露边界(用户 2026-07-04 厘清,关键)**:

| 形式 | 内容 | 泄露? |
|---|---|---|
| (a) 完整成就 tech-tree / 课程链先验 | "gnome_warrior 的最优攻克链 = [...]" | ❌ 重度泄露,baseline 无,禁用 |
| (b) crafting/世界机制依赖图 | "make_diamond_sword 需 collect_diamond;钻石在 floor2/7" | ✅ **可用**——baseline 已吃 |
| (c) 纯 student 行为共现 | "tier3 成功的 episode 里同时达成了啥" | ✅ 可用,纯经验 |

- **(b) 不算新增泄露**:DiCode proposer 的 system prompt 本就注入完整 `<game_rules>`
  ({CONSTANTS}/{MOBS}/{GAME_MECHANICS}/{WORLD_GEN},见 world_gen_nl.py)。我们只是把散在文本里的
  机制**结构化**成图便于推理,未引入 baseline 没有的信息。
- **(a) 禁用**:"该走哪条链、链的顺序"必须 LLM **自己推理**,不查表。这是 novelty 所在。
- **modeler 用 (b) 当"物理上可能的依赖骨架" + (c) 当"这个 student 实际走通/走断了哪段"**,两者
  结合推断当前攻坚目标的**完整前置链**(混合序列,非纯装备清单)。

**★★★ DEPTH_TIERS 是我们引入的弱先验,baseline 没有 —— v6 不再依赖它排序(用户 2026-07-04)**:
- 事实核查(2026-07-04):**"tier" 既不在 student 观测里,也不在 baseline DiCode 的 teacher prompt 里。**
  student conditioning = 关卡成就名列表的 embedding(envs/craftax.py `self.label`),无 tier/难度字段;
  baseline teacher 只拿到 **游戏机制 `<game_rules>` + 成就 Category(CRAFTING/COMBAT,非 tier) + student
  achievement SR**,靠 `learnability=p(1-p)` 和 SR profile **自己摸索深浅**,不查 tier 表。DiCode 原文
  连 "tier" 一词都没有(讲 "hierarchical depth"/instrumental/late-game,是论文描述性排序,未结构化喂入)。
- **`DEPTH_TIERS`(craftax_achievements.py:113)是我们 v5 手工划的分层**(注释已注明 "hand-curated,
  NOT an authoritative craftax constant"),v5 的 AmbitionGain depth 轴 / ability-gate 都用了它 ——
  **这是 v5 相对 DiCode 多吃的一个弱先验**,以前没当回事。
- **v6 决定:攻坚机制的"难关判定 + 深浅排序"改为 modeler 从 (b)机制 + student SR 自推(§3.2),
  不再喂 DEPTH_TIERS 做攻坚排序** —— 更干净、更贴 baseline("像玩家自己感到卡关"),也回避了这个
  泄露点。DEPTH_TIERS 若仍在 v5 遗留的 auction bid 里出现,v6 应逐步剥离或在论文里 disclose 为
  "a coarse hand-curated depth prior we introduce";攻坚新机制**不新增**对它的依赖。

**★真实 Craftax 结构证实"装备后置于战斗"系统性存在(权威,来自 world_gen_nl.py 注入知识)**:
9 层垂直:0 Overworld / 1 Dungeon / **2 Gnomish Mines(全黑,gnome warrior+archer,钻石/宝石)** /
3 Sewers(冰附魔台) / 4 Vaults(火附魔台) / 5 Troll Mines / 6 Fire / 7 Ice(钻石) / 8 Boss(necromancer)。
- **钻石装链交织**:make_diamond_sword/armour 需 collect_diamond;钻石在 Overworld 仅 0.1%(拿不到)
  → 实际去 **floor2(全黑+gnome 战斗)** 或 floor7 挖 → **造钻石装(装备 t3)后置于"在 floor2 应对
  gnome(战斗 t3)"**。装备依赖战斗,同 tier3。
- **附魔链交织**:enchant_sword/armour 需 floor3/4 的附魔台 → 要穿过 floor2 → 装备后置于深层战斗/到达。
- **全黑层隐藏门槛**:floor2/5/7/8 pitch black → 隐含依赖 tier2 火把(make/place_torch)。这种"因黑
  需火把"的依赖 **crafting 配方表查不到**,只有 (b)世界结构+(c)student 表现能推 → 再证必须 LLM 推断。

### 3.4 干预点:收紧 proposer 的 Completed/Relevant 准入(不推翻 scaffold)

**改哪里**:DiCode 的 scaffold 核心是 proposer 决定哪些成就进 `Completed`。现指令"compress away
**already-mastered** prerequisites",v5 病 = proposer 把中间链环也当 mastered 压掉了。v6 给
`Completed` 准入**加一道闸**:

- 一个前置**只有 student held-out SR 真的高(真 mastered)** 才能进 Completed 压掉。
- 攻坚链条里 student **还没稳定掌握的中间环节**(§3.2 的 flat-near-zero 环:在黑暗层生存、到达
  floor2 等)→ **必须留在 Relevant(关卡里实际训练),禁止压进 Completed**。

**不改**:Completed/Relevant 两分机制本身、初始状态工具、compilation 流程、learnability 淘汰。
**改**:Completed 准入判断从"proposer 拍脑袋"→"modeler 基于 held-out 真实掌握度 + 攻坚链条裁定";
强制未掌握中间环节留 Relevant。→ 改动小、不破坏对标公平(仍同一套 scaffold 框架,只是判断更准)。

### 3.5 modeler 的持久攻坚笔记 (A+B 混合,核心数据结构,用户 2026-07-04 拍板)

**"攻坚依赖树"不是每 session 现推的临时结构,而是 modeler 长期维护、跨 session 存盘增量更新的
持久"作战笔记"** —— 像真人玩家卡关时脑内那张随经验生长的笔记。这是 v6 的核心数据结构,也是
§3.5(固化)的上游:已验证链条库/protected set 是这本笔记的组成部分。

**为何必须持久(非每 session 重推)**:① (c)共现证据稀疏(tier3 SR~10%,单 session 成功 episode 太少),
须跨 session 累积才稳;② 攻透一个难关(含回溯子目标)跨很多 session,须记住"正在攻哪个/到链条哪一环/
哪些环已固化",否则东一榔头西一棒退回 v5 平摊病;③ H1 的"自我风格"本就依赖记忆生长。

**笔记五块内容 + 随 session 怎么变**:
| 笔记块 | 内容 | 更新规则 |
|---|---|---|
| **当前攻坚焦点** | 正在攻哪个战斗难关 | 攻破/放弃才换,否则保持——**保证聚焦不漂移** |
| **焦点的前置依赖树** | 回溯出的链条 + 每环 student 掌握状态 | 每 session 用新 SR + (c)共现**增量更新每环"已固化/未掌握"标记** |
| **★自我风格心得(style_note)** | 每个焦点的**攻坚 know-how 文字**("怎么打赢的/难在哪/什么策略管用"——走位、gear-up 时机、拉怪、逃生) | 每 session **无门槛**更新(focus active + LLM 写了新非空 note 即覆盖旧的;空则保留旧的→心得跨 session 累积生长) |
| **已验证链条库** | 攻破并固化的链条(技能组成 + 验证证据 + **该链的 style_note**) | 只增/dedup-by-target(攻破一个加/更新一条),供复用为 tier4 地基(§3.6) |
| **protected set** | 强制 rehearsal 的技能(§3.6 固化) | 攻破验证后加入 |

**★★★ style_note 是 H1"自我风格"的物理载体(用户 2026-07-05,补上核心缺口)**:此前笔记只存骨架
(skill/links/category/SR),H1 说的"可迁移风格"(走位/gear-up/拉怪/逃生这类**动态策略经验**)**没有任何字段
承接**——modeler 每 session 脑内想的攻坚经验用完即弃。这是"核心卖点无数据载体"的设计-实现落差。**修复=给
notebook 加自由文字 `style_note` 字段**,让 modeler 每 session 写下攻坚心得。落地五处:①modeler siege
prompt schema 加 style_note + 要求"精简致密、每字有用、不漏重点、随理解精炼、留空则丢失 know-how";
②`_validate_siege`/`_normalise_proposal` 保留该字段(含 legacy 形式);③`_merge_style_notes` 每 session 把新非空
note 贴到 active focus(无门槛,只覆盖非空→累积);④`_upsert_experience` 写进 verified_chains(dedup-by-target,
非空才覆盖);⑤`render_for_prompt` 三处渲染(active focus 的 style-so-far / milestone 的 style: / enabler 的
note:)喂回下一轮。**对装备类同样生效**(category 只是标签不阻断)。不截断(信任 prompt 要求的精简)。181 单测绿。

**★心得 vs rehearsal 两套独立机制(易混淆,用户 2026-07-05 澄清)**:
- **style_note 心得**:**每 siege session 无条件**注入 modeler prompt(gen_manager 唯一条件=`siege_active`,无遗忘门槛)→ **指导 modeler(teacher)的每一次攻坚决策**。这是全程持续生效的。
- **rehearsal(§3.6)**:**仅当检测到 FORGETTING** 才触发 → 给 **student(RL网络)**追加复习关防遗忘。
- 两者不同触发/不同对象/完全独立:心得是"每次都指导 teacher",rehearsal 是"遗忘时才救 student"。

**★style_note 是"铁镐困境"的修复(用户 2026-07-05 厘清)**:v6 首跑(job 3653226)观察到 siege 30 session
卡在 make_iron_pickaxe(SR 32-58% 震荡、卡 47%、focus 反复退役重开、从没推进到 tier3 战斗、H1 上不了场)。
病根**不是**独立的"退役死循环 bug",而是**攻坚经验用完即弃→student 学一点忘一点→SR 永远稳不下来**。→
style_note 让 modeler 带着积累心得连续指导攻坚 → SR 应单调爬升而非震荡 → 铁镐真正攻透→退役→focus 推进到
战斗类(H1 上场)→ 死循环自愈。**修好病根=症状自愈,不需额外的退役冷却代码**。★方法论教训:不能用"无心得
时的历史震荡轨迹"去推测"有心得后的行为"(那是循环论证);笔记有效则那条轨迹本就不会发生。验证 run =
job 3658849(DiCode-v6siege-style,新代码带 style_note,旧 3653226 已杀防重启污染)。

**★A+B 混合的维护方式(用户拍板)** —— 代码保证硬约束 + 骨架,LLM 负责判断:
- **代码保证(B,防漂移兜底)**:笔记的 schema/存盘(跨 session resume,像 StudentProfileLog 存 JSON)、
  §3.2 的**作用域硬约束**(焦点不得是简单关/已饱和成就)、每环"已固化/未掌握"标记由 **SR 阈值自动算**、
  焦点切换的最低条件(如连续 K session 不升才允许换)。这些**不能让 LLM 违反**。
- **LLM 负责(A,玩家直觉)**:读上一版笔记 + 本 session 新证据 → 推断/更新前置链条(用 (b)机制+(c)共现)、
  判断焦点该不该换、决定下一个攻谁、语义层面判"这一环是不是真前置"。**LLM 输出更新后的笔记(受 schema
  约束),不是每次从零重写。**
- **好处**:既有 LLM 的"玩家脑内笔记"生长感(novelty),又有代码兜底防 LLM 长期状态漂移/自相矛盾
  (LLM agent memory 的老大难)。

### 3.6 固化:两层分工(防遗忘,让风格累积)

战斗目标攻破 + 链条经**多次成功验证**成立后,固化以防灾难性遗忘(v1 病根之一):

- **student 侧 — rehearsal(安全底线)**:链条攻破 → 焦点+其链环入 **protected set** → 在后续训练
  batch **额外追加**复习关(教 protected 技能的 active 关)。**★实现决策(用户 2026-07-04,偏离原文)**:
  - **触发改为"只在检测到 FORGETTING 时才复习"**,非原文的"主动锁定/每 session 都复习"。理由=早期
    链少时每 session 强塞复习浪费攻坚火力;FORGETTING 信号(StudentProfileLog.forgetting_candidates,
    SR 峰后跌)足够灵敏。作用对象是已固化 protected 链(非任意技能)。protected 技能没在掉 → quota=0
    → 全火力留攻坚焦点。
  - **★术语分家(用户 2026-07-05,重要)**:救遗忘的职责**完全归 rehearsal(系统自动)**,不再借道
    proposer 的 `CONSOLIDATE` 级别类型。**FORGETTING → 系统 rehearsal**(本节),**CONSOLIDATE → 重
    定义为"精益求精"**:proposer/modeler 用它把一个**已学会但不可靠的非-siege 技能**(中庸 SR 卡在
    未 solid)往上推向掌握——既不是救遗忘(系统管),也不是攻 siege 硬墙(那是 DEPTH)。三处 prompt
    (persona_ambitious_coop / modeler system prompt / 诊断注入)已按此改齐;modeler 诊断出的
    FORGETTING 不再注入给 proposer(只 STALLED 注入),避免 proposer 去抢 rehearsal 的活。
  - **复习是"额外追加"非"batch 内替换"**(用户强调):攻坚满额(training_sample_size_n,如16)**一关不删**,
    复习关追加在后。→ 攻坚永远满额,复习是加法不是从攻坚里抠。
  - **一条链可用多个复习槽**:quota = rehearsal_per_forgetting_skill × (FORGETTING 的 protected 技能数)。
    链的多个环节同时掉 → 按比例多救(应对 tier4 同时维护多链/多环节)。
  - **两道上限防墙钟失控**:① quota ≤ 攻坚满额 × rehearsal_max_frac(默认0.5);② **攻坚+复习总数 ≤
    rehearsal_total_cap(暂定24关)** 绝对天花板。代码=selection.append_rehearsal_tasks,接 run_dicode.py
    在攻坚 batch 组好后追加;siege off/protected 空/无 FORGETTING 均严格 no-op(baseline 零改)。9 单测绿。
- **modeler 侧 — 已验证链条库(优雅,agent 记忆)**:modeler 维护"已验证链条库"(哪些链条已攻破、
  由哪些技能组成、验证证据)。用途:① 造 tier4 关时**合法复用**这些链条作地基前置(合法进 Completed,
  因真掌握且在固化中);② 排序下一个攻坚目标。这是 LLM 侧的结构化记忆,天然适配。
- **分工**:记忆库负责"知道该固化什么、下一步攻什么";rehearsal 负责"让 student 权重真的记住"。
- (存疑,超 v6 范围,不碰)让 student RL 网络自带可持久读写的外部记忆 = 研究级难题,会动到"student
  锁 RL 不动"的地基,不做。

### 3.7 归属:先升级 modeler,重了再拆

这套(攻坚树 + 战斗/装备分类 + 链条推断 + 已验证链条库 + 固化调度)全依赖 modeler 已有的
StudentProfileLog / guidance_per_parent / ModelerArchiveView → **升级 modeler(纯叠加 state),
不新起 agent**。若 modeler 职责过重致 prompt 塞不下或诊断与规划混淆降质 → 再拆成"状态诊断"+
"攻坚规划"两 agent。**先升级,重了再拆。**

### 3.8 数据缺口与待办(实现前必须解决)

- **★per-episode 成就共现数据(c) —— 已实现(2026-07-05)**:原缺口=craftax_evaluation.py 在 `.sum()`
  聚合时扔掉 per-env 成就 multi-hot。**已实现**:eval 在 jit 内额外堆 per-env multihot(finished&达成
  →0/1)→算 `count[67]`+`cooc[67,67]`(固定 shape,jit 安全,`multihotᵀ·multihot`)→用 `_cooc_*` key 带
  出;`make_evaluate` 额外返回静态 `cooc_names`(jit 外,对齐列名);`online_evaluation` 在 wandb log **前**
  pop `_cooc_*` 落盘(不污染 wandb),喂 `auction/cooccurrence_log.py` CooccurrenceLog(照 student_profile_log
  持久化,**跨 session 累加计数**+**累加总 finished 局数(SR 分母)**,按名重排到 craftax 顺序,resume 幂等)。
  modeler 侧 `render_prereq_hint` 给 siege prompt 注入"student 攻破深层成就时实际共现了啥(附经验 SR)"→让链条
  来自真实轨迹非想象。gen_manager `_cooc_log` 懒建(仅 siege on)。16 单测。
  仍**分阶段**:先用聚合 SR 跑通 §3.4(验证"强制完整链"),(c) 作增强叠加(SR 够高才生效,无害回退)。
- **★共现可信护栏 = 相对 SR 门槛 MIN_SR=3%(user 2026-07-05,订正)**:原设计用绝对计数 `MIN_SUPPORT=5`,
  基于"tier3 成功稀疏"的**错误前提**。实测 held-out eval `num_envs=1024`/session → tier3(SR~12%)单 session
  就有 ~120 次成功局,**绝对计数从不稀疏**;`MIN_SUPPORT=5` 几乎不起作用。真正该防的是"deep 成就极少被攻破
  时,条件共现频率来自单样本噪声"→护栏应**相对**:`count[deep]/累加总 finished 局数 ≥ MIN_SR(3%)` 才信其
  共现,否则空(回退 (b)机制)。3% 对齐 v5 实测 tier3 天花板~12%(tier3 稳过,真正近乎不可能的成就 SR<3% 被滤)。
  `MIN_SUPPORT` 保留为无用别名。
- **实现野心分阶段**:①prompt 要求完整链 + ②校验兜底(Completed 准入闸)先做,快验证 H1;有效但
  不够再上 ③递进多关(= 强化 DiCode 已有的 112→287→532 lineage 演化,每关多留一中间环,非发明新结构)。

**待验证(v6 成败判据)**:tier3 held-out 裸 per-skill SR 能否突破 ~10% 天花板(当前三方+DiCode
原文共同的墙);且攻克顺序靠后的战斗类 tier3 是否爬升更快(H1 的直接证据)。

### 3.9 coop 质量筛选:v6 采纯-Learnability top-k(24→18,用户 2026-07-05)

**决策**:v6siege 开启 coop 层质量筛选,`coop_select_k=18`(2 proposer × 12 = 24 候选,留 18,弃约
25% 最不可学的),**但权重改为纯 Learnability**:`coop_w_amb=0 / coop_w_lrn=1 / coop_w_cov=0 /
coop_w_end=0`。config 一行覆盖,`_coop_select`/`GreedyTopKSelector` 代码零改(w_amb=0 时 ambition 项
干净清零,target_gap 为 None 也已兜底)。

**为何筛**:v5 证据(v5yA,24→18,`coop_w_amb=1/coop_w_lrn=1`)显示适度 coop cull 有**微弱正效果**(早
期 step1117=19.7 首次探出 baseline +1.2)。注意这是**弱信号**,且 v5 整体仍是负结果(全卡 tier3 天花板,
见 [[v5-negative-result-scaffold-breaks-tier3-chain]])——所以 v6 开筛是**低风险尝试**,不是"v5 证明有效
故照搬"。

**为何去掉 AmbitionGain(v6 特有的关键改动)**:`AmbitionGain = Σ gap × DEPTH_WEIGHT × reach`,depth_weight
使它**系统性偏好深关**。而 v6 攻坚课程的整个立命之本是**保住浅层但必需的前置链**(§3.4 准入闸强制未掌握
中间环留 Relevant、别 scaffold 成"孤立最后一跳")。若 topk 仍带 AmbitionGain,就会把浅前置链当低分**剔掉
**——与 §3.4 直接对冲。故 v6 让**筛选只判"现在学得动吗"(Learnability=p(1−p)),把"往深处推"的职责整个交
给 siege 焦点 + completed_gate**。职责分离后两者不再打架。

**★管线阶段隔离 → rehearsal 关天然免疫此筛选(用户关切,已代码核实)**:
- **生成阶段**(gen_manager):2 proposer 造 24 关 → `_coop_select` topk 留 18 → 存 archive。筛选对象
  **只有当轮新造的 24 关**。
- **训练采样阶段**(selection.py,独立后续):`sample_tasks_for_training` 从 archive 全部 active 关 PLR
  采样 → siege_batch → `append_rehearsal_tasks` **额外追加**复习关。复习关是从 **archive 里的旧关**捞的
  (教某正在 FORGETTING 的 protected 技能),**不在**当轮新造 24 关池里。
- 结论:`_coop_select` 根本看不到 rehearsal 关(不同池、不同阶段),且 rehearsal 是 `siege_batch +
  rehearsal_extra` 纯追加、siege_batch 原样不缩。**开不开筛、用什么权重,都碰不到复习关**,无需额外护栏。
- 另注:底层 `topk_k`(selection.py 的 PLR replay 采样)是 DiCode/baseline 共用的原生机制,与本节 coop
  `coop_select_k` 是**两回事**,不动(动它会偏离对标基线)。

---

## 4. 数据与复现
- v5yA graphml:`/oscar/scratch/jzhu223/dicode_outputs/v5yA_s0_r2/task_graph.graphml`(670 节点,
  session_created / description / performance_history 字段);解析脚本 scratchpad/_parse_v5yA_levels.py。
- 裸 tier SR 对比脚本:scratchpad/_tier_absolute.py;mean_return 横比:_xcompare_v1shift.py。
- run:v5yA=DiCode-v5debate/dicode-v5yA-s0-r2,v5yB=dicode-v5yB-s0,base=DiCode-repro/dicode-repro-s0-v1。
- 相关:[[v5-runs-and-selection-variants-2026-07-03]]、[[v1-step-1900-pollution-early-lead-real]]、
  experiment_design.md §10.7(v1 tier漂移+遗忘)、v5_design.md §8(筛选策略)。
