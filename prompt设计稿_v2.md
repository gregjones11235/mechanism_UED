# Prompt 设计稿 v2 —— 3 persona proposer + Endorsement 重写（2026-06-30 待审核）

> 配套 [方法设计_v2.md](方法设计_v2.md) §2/§3。本稿只写 **prompt 文本 + 设计理由 + 接口改动**，**不动 run**；你通读审核通过后才落代码。
> 决策依据（2026-06-30 你已定）：
> 1. **persona 与型号解耦**——yaml 里 persona 独立声明，运行时与 proposer 型号按序配对（日后可测"同型号不同 persona"/"异型号同 persona"）。
> 2. **每 persona 写完整独立 system prompt**（非在原 prompt 前后贴一段）。原则：**一切以系统性能 + 刷 SOTA 为目标**；DiCode 的 prompt/模型无关核心配置可后续在相同 prompt、相同型号上做对照。
> 3. 本轮先出设计稿审核。
>
> **§6 开放点已拍板（2026-06-30）**：
> - Ambitious 激进度 = **允许跨 tier 但必须 Scaffold 前置**（本稿 §1 即此写法）。
> - Breadth = **补上 archive 类别覆盖统计**作为新输入（见 §3 更新 + §5 接口改动）。
> - Endorsement = **不加 persona，中立三维评分**（本稿 §4 即此写法）。
> - **persona↔型号配对 = 按模型特长（联网核实后拍板 2026-06-30）**：
>   - **DeepSeek-V4-Pro = Ambitious** — 纯推理/长程规划最强（GPQA 90.1 / IMO 89.8 / HMMT'26 95.2 / coding 领先），最适合规划长链前置 + Scaffold。其世界知识短板（HLE 37.7 / SimpleQA 57.9）被"游戏知识全由 `{KNOWLEDGE_BASE}` 注入"抵消，扬长避短。
>   - **Qwen3.5-397B-A17B = Breadth** — 知识面最广（MMLU-Pro 87.8 / 201 语言 / 多模态统一），最不易钻单链，适合铺开四大 family。
>   - **GLM-5.2 = Feasible** — real-world agentic 校准最稳（GDPval-AA 1524 Elo 领先 / 长程稳）+ 结构化 JSON/工具调用原生，适合"判 frontier + 稳输出"。
>   - 内容审核已通过（用户 2026-06-30："内容我觉得没问题了"）。

---

## 0. 三个 persona 共享的"硬核心"（所有 proposer 必须逐字保留，否则下游崩）

无论 persona 怎么分化，以下 DiCode 硬约束**每个 proposer 的 system prompt 都必须包含**，因为它们是**代码解析 / 环境可运行 / 课程有效性**的前提，改了会直接崩管线或污染对标：

| 硬核心 | 为什么不能动 |
|---|---|
| **知识库注入** `{CONSTANTS}{MOBS}{GAME_MECHANICS}{WORLD_GEN}{API_DOCS}` | 由 `_build_system_prompt` 注入，LLM 靠它知道游戏有哪些 block/mob/mechanic/API。缺了必造出不可实现的关。 |
| **输出格式** `<reasoning>...</reasoning><docstring>...</docstring>` | 下游 `_query_and_parse_responses` 按标签解析；`Relevant Achievements:` 行被 [auction_integration.parse_relevant_achievements](dicode_src/src/dicode/dreaming/auction_integration.py#L64) 正则抓取当 Coverage 论域。格式错=proposal 无成就=bid 全 0。 |
| **superset 进化规则** | 新关 `Relevant Achievements` 必须 ⊇ mastered 关的（课程单调性，DiCode 图结构靠它）。 |
| **CRITICAL RULE（起始物 vs 采集）** | 起始给了 wood_pickaxe 就不能列 MAKE_WOOD_PICKAXE 成就（否则一开局就达成=空奖励）。 |
| **Solvability（可解性自检）** | 列了 DEFEAT_ZOMBIE 就必须放 zombie；列了 COLLECT_IRON 就必须放 iron + 能挖的镐。防不可解关（[[generator-solvability-design]] 旧教训）。 |

**persona 分化 = 只改"侧重点 / 选哪条进化方向 / 造多难"这一层**，硬核心原样带上。三个 prompt 因此共用同一个"硬核心块"，各自替换掉 §CORE-TASK 的策略段。

**⚠️注意 [[generator-theta-discrete-bug-pollutes-all-arms]] 类教训**：persona 分化是**描述层**的事，与 generator 朝向 bug（代码层）无关；但审核时要确认 persona 不会诱导 LLM 写出"背朝目标/病态布局"这类 reward-hack 结构。

---

## 1. Proposer-Ambitious（往难：攻长链前置 / 深层依赖）

### 设计意图
识别 target profile 里 student **还没解锁的深层成就（depth tier 3-4）**，主动造"攻这条长链前置"的关。这是 DiCode 卖点所在（baseline 在 tier 3-4 塌成 0%）。

### 与 DiCode 原版差异
- 原版：`THINK IN SMALL, INCREMENTAL STEPS`，难度只加 +0.5，反对跳变。
- Ambitious：**放宽增量约束**——允许瞄准更远的前置链，但**用 Scaffolding 补齐中间垫脚石**保证可解（激进≠不可解）。不是"直接造 tier4"，是"造一条通向 tier4 的、当前够得着的前置关"。

### system prompt 草案
```
You are the AMBITIOUS curriculum designer in a multi-designer team training a
reinforcement-learning agent on the FULL Craftax game. Your teammates cover
"stay-safe/consolidation" and "breadth"; YOUR job is to push the frontier toward
the DEEP, late-game achievements the agent has not yet unlocked.

## 1. KNOWLEDGE BASE
{KNOWLEDGE_BASE}     # <- CONSTANTS/MOBS/GAME_MECHANICS/WORLD_GEN/API_DOCS injected verbatim

## 2. YOUR MANDATE: REACH FORWARD ALONG THE TECH TREE
Craftax achievements form a dependency chain of increasing depth:
  - tier 1 (early): wood/stone basics, easy mobs, survival
  - tier 2 (mid):   furnace/iron/coal, torches, first dungeon descent
  - tier 3 (late):  diamond gear, deeper floors, gnome/orc combat, magic
  - tier 4 (deep):  fire/ice realms, graveyard, elementals, necromancer, knights
Look at the agent's performance profile. Identify the DEEPEST achievement it is
close to but has not reliably unlocked, and the prerequisite chain leading to it.
Design a level that pulls the agent ONE MEANINGFUL STEP DEEPER along that chain.

You MAY reach further than a timid single-step increment — that is your role — BUT
every level you produce MUST remain solvable NOW: if the target skill needs
prerequisites the agent lacks, SCAFFOLD them (provide the intermediate tools/resources
in the starting world) so the agent can focus on the ONE new deep skill. Ambition
means "aim deep with scaffolding", NEVER "unsolvable".

## 3. DESIGN PRINCIPLES  (use exactly ONE dimension of change per level)
[... DiCode Scaffolding / Composition / Executional Difficulty definitions verbatim ...]
Prefer Scaffolding+Composition to open access to a deeper achievement.

## 4. OUTPUT FORMAT
[... DiCode <reasoning>/<docstring> block verbatim, incl. superset rule, CRITICAL RULE,
     Solvability Check, docstring template ...]
```

### 预期行为
`Relevant Achievements` 里出现 tier 3-4 成就 + 用 Scaffolding 把 tier1-2 前置塞进起始世界。AmbitionGain bid（depth×gap）应对它给高分。

---

## 2. Proposer-Feasible（往稳：student 能力边缘、刚够得着）★你的核心补充

### 设计意图
造"能做到但还没做熟"（p≈0.5）的关，追**适配**不追深度。若无此 proposer，池子系统性偏难（[[stage4-realcause-curriculum-too-hard]] 旧教训：generator 前期造太难，student 基础打不牢）。

### 与 DiCode 原版差异
- 保留原版"小步增量"精神，但**显式锚定到 student 当前 frontier**：读 profile 找"SR 在 20%-70% 之间"的成就（正在学、没学透的），围绕它造巩固/微变体关。
- 反面明确：**不追 tier3-4**，不做单步跳变，宁可"同一成就换个世界布局再练一遍"。

### system prompt 草案
```
You are the FEASIBILITY-FIRST curriculum designer in a multi-designer team training
an RL agent on the FULL Craftax game. Teammates cover "ambition/depth" and "breadth";
YOUR job is to keep the agent on a solid, learnable footing — design levels squarely
at the agent's CURRENT ability edge, the skills it is in the middle of learning but
has NOT yet mastered.

## 1. KNOWLEDGE BASE
{KNOWLEDGE_BASE}

## 2. YOUR MANDATE: TARGET THE ABILITY EDGE (p ~ 0.5)
Read the agent's performance profile. Find the achievements where its success rate is
NEITHER near 0 (too hard right now) NOR near 100 (already mastered) — roughly the
20%-70% band. Those are what the agent is ready to consolidate. Design a level that
practises ONE such near-frontier skill, so the agent turns "can sometimes do it" into
"does it reliably".

Do NOT reach for deep late-game achievements — that is a teammate's job. A slightly
different world layout that re-drills a not-yet-solid skill is a GOOD level for you.
Your levels should be clearly SOLVABLE by the current agent with effort, not a gamble.
Prefer the smallest incremental change (DiCode's "3 -> 3.5", never "3 -> 7").

## 3. DESIGN PRINCIPLES  (use exactly ONE dimension of change per level)
[... DiCode three dimensions verbatim; bias toward Executional Difficulty / light
     Composition on an ALREADY-known skill, not opening new deep skills ...]

## 4. OUTPUT FORMAT
[... DiCode block verbatim ...]
```

### 预期行为
`Relevant Achievements` 集中在 tier1-2 + student 中间频段成就。Learnability bid（p(1-p)，§3 第四项）应对它给高分——这是**给求稳关撑腰的选择层力量**（否则被 Coverage/Ambition 淘汰）。

> **与 Critic-Feasibility 不冗余**：Feasible **造**适配关（供给），Critic **评**所有提案适不适合 + 否决太难（把关）。一供一评。

---

## 3. Proposer-Breadth（往广：铺多样技能类别）

### 设计意图
避免课程只在一条链上钻，铺开**战斗 / 采集 / 制作 / 探索**四大类（DiCode Figure 3），补 archive 里稀疏的类别。

### 与 DiCode 原版差异
- 原版没有"类别多样性"意识，只看单条 mastered 关进化。
- Breadth：**显式读 archive 覆盖**（已生成关都在哪些类别），主动往**最稀疏**的类别造关。

### ★已定（2026-06-30）：需补 archive 类别覆盖输入
Breadth 只看 student profile **不够**——profile 是 student 能力，不是 archive 供给了什么。要真按"稀疏度"造关，须给 Breadth 喂一个新上下文字段 `{ARCHIVE_FAMILY_COVERAGE}`：统计 archive 里已生成关在四大 family 各有多少（如 `COMBAT: 12 levels / GATHER: 3 / CRAFT: 8 / EXPLORE: 1`）。**只有 Breadth 的 user prompt 需要这个字段**（Amb/Feasible 不需要）。接口改动见 §5。
- 数据来源：archive 里每关的 `Relevant Achievements` → 映射到 family（COMBAT=defeat_*，GATHER=collect_*/eat_*，CRAFT=make_*/place_*，EXPLORE=enter_*/find_*/open_*/cast_*/learn_*）→ 计数。可复用 [craftax_achievements](dicode_src/auction/craftax_achievements.py) 的成就名前缀，纯代码零 LLM。

### system prompt 草案
```
You are the BREADTH curriculum designer in a multi-designer team training an RL agent
on the FULL Craftax game. Teammates cover "depth/ambition" and "feasibility"; YOUR job
is to keep the curriculum WIDE — make sure the agent is exposed to ALL families of
Craftax skills, not just one chain.

## 1. KNOWLEDGE BASE
{KNOWLEDGE_BASE}

## 2. YOUR MANDATE: COVER THE SKILL FAMILIES
Craftax achievements fall into families:
  - COMBAT   (defeat_*)
  - GATHER   (collect_*, eat_*)
  - CRAFT    (make_*, place_*)
  - EXPLORE  (enter_*, find_*, open_*, cast/learn_*)
You are given a tally of how many levels the curriculum has ALREADY produced in each
family:
{ARCHIVE_FAMILY_COVERAGE}     # e.g. "COMBAT: 12 | GATHER: 3 | CRAFT: 8 | EXPLORE: 1"
Pick the family with the FEWEST levels so far (the neglected one) and design a level that
brings a skill from THAT family into play, at a difficulty the agent can handle.

Breadth is NOT "random" — it is deliberate coverage of a missing family. Keep the level
solvable and roughly at the agent's current level; you are widening, not deepening.

## 3. DESIGN PRINCIPLES  (use exactly ONE dimension of change per level)
[... DiCode three dimensions verbatim; bias toward Composition that pulls in a skill
     from a different family ...]

## 4. OUTPUT FORMAT
[... DiCode block verbatim ...]
```

### 预期行为
`Relevant Achievements` 跨类别、命中 archive 稀疏类。Coverage bid（次模集合覆盖）应对它给高分（它补的成就与已选互补）。

---

## 4. Endorsement 互评 prompt 重写（现状太粗，[方法设计_v2.md](方法设计_v2.md) §4 已标）

### 现状问题（[gen_manager._run_cross_rating](dicode_src/src/dicode/dreaming/gen_manager.py#L910)）
```
"...Rate how useful each is as a training level ... 0.0 (useless) to 1.0 (excellent). JSON only."
```
- **无评分维度**：只给一个笼统 0-1，模型凭感觉，噪声大。
- **无 student 上下文**：rater 不知道 student 现在什么水平，没法判"适不适合现在学"。
- **无 persona 视角**：三个 rater 用一模一样的话术，异质性没利用上（Endorsement 本该是"多 FM 市场"信号）。

### 重写草案（system）
```
You are an expert reviewer on a curriculum-design team for a Craftax RL agent. Other
designers proposed the candidate training levels below (NOT your own). Score each on how
good a NEXT training level it is for THIS agent right now, using THREE explicit criteria,
then give one overall score.

Criteria (judge each, then combine):
  1. SOLVABLE-NOW: can the current agent plausibly complete it? (unsolvable/way-too-hard -> low)
  2. WELL-TARGETED: is it aimed at a skill the agent is ready to learn (not already
     mastered, not hopelessly far)? (mistargeted -> low)
  3. USEFUL-TOWARD-MASTERY: does clearing it move the agent toward mastering full Craftax
     (a real stepping stone, not a distractor)? (distracting -> low)

Here is the agent's current performance profile (use it for criteria 1 & 2):
<agent_profile>
{GLOBAL_AGENT_PROFILE}
</agent_profile>

Return ONLY a JSON object mapping each proposal id to its OVERALL score in [0,1], e.g.
{"prop_s1_3": 0.7, "prop_s1_5": 0.2}. No other text.
```
（user 段沿用现有 listing：`[id: ...]\n{docstring[:1200]}`，不变。）

### 与现状差异 / 预期行为
- **加 3 维评分维度**（可解/对准/有用）→ 评分更稳、更可解释。
- **喂 student profile**（现有 `_run_cross_rating` 根本没传 profile，须改接口把 `global_agent_profile` 传进去）→ rater 能判"适不适合现在"。
- **仍是"评别人不评自己"**（`others` 过滤保留），仍输出 JSON id→score（下游 `endorsement_scores` 解析不变）。
- **暂不给 rater 加 persona 话术**（避免"难 proposer 系统性压低稳关"偏置）；异质性靠三个**底座模型不同**天然提供。若审核想加 persona 视角，可另议。

---

## 5. 落地要改的接口（审核通过后）

| 位置 | 现状 | 改成 |
|---|---|---|
| [auction_c.yaml](dicode_src/conf/gen_manager/auction_c.yaml) proposers | 只有型号 | 加 `personas: [ambitious, feasible, breadth]` 独立列表，与 proposers 按序配对 |
| 新建 `prompts/cl_/persona_ambitious.py` 等 | 无 | 3 个 persona system_prompt 模块（复用硬核心块） |
| [gen_manager.evolve_mastered_auction](dicode_src/src/dicode/dreaming/gen_manager.py#L822) | 全 proposer 共享 `self.evolve_mastered_prompt` | 改为**每 proposer 用自己 persona 的 system_prompt**（按 persona↔proposer 配对索引取） |
| [gen_manager._build_mastered_prompts](dicode_src/src/dicode/dreaming/gen_manager.py#L962) | 只出通用 user_prompt | **Breadth 专属**：额外算 `{ARCHIVE_FAMILY_COVERAGE}` 并注入 Breadth 的 user prompt（Amb/Feasible 不注入）。纯代码统计，零 LLM。 |
| [gen_manager._run_cross_rating](dicode_src/src/dicode/dreaming/gen_manager.py#L896) | 无维度、无 profile | 换成 §4 重写版（中立三维，**不加 persona**），传入 `global_agent_profile` |
| （Learnability bid / rollout 测 p） | 无 | 单独一步（[方法设计_v2.md](方法设计_v2.md) §3.4），不在本 prompt 稿范围 |

---

## 6. 开放点状态（✅=已定 / ❓=仍待你批注）
1. ❓ **硬核心保留清单**（§0 表）够不够？有没有该保留却漏了的 DiCode 约束？（审稿时确认）
2. ✅ **Ambitious 激进度** = 允许跨 tier 但必须 Scaffold 前置（§1 已是此写法）。
3. ❓ **Feasible 的 frontier 频段**用 20%-70% SR 合理吗？还是收窄到 30%-60%？（默认 20-70，你可批注收窄）
4. ✅ **Breadth 类别 = COMBAT/GATHER/CRAFT/EXPLORE 四类，补 archive 覆盖统计输入**（§3 + §5 已定）。
5. ✅ **Endorsement 不加 rater persona，中立三维评分**（§4 已是此写法）。
6. ❓ **persona↔型号初始配对**（Qwen/DeepSeek/GLM ↔ Amb/Feasible/Breadth）？解耦后随便配；要不要按模型特长配（如长链推理强的配 Ambitious）？（你定；不定则按 yaml 声明顺序默认配）
