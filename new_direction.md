# new_direction —— DiCode 拆解 + 我方向的落子判断（2026-06-30）

> 来源：与 Claude 的多轮讨论，全部基于 DiCode 论文全文（`new_direction/DiCode.pdf`，90 页含附录）逐条核对原文。
> 论文：*Dreaming in Code for Curriculum Learning in Open-Ended Worlds*（DiCode），arXiv 2602.08194。
> 配套记忆：`new-direction-dicode-multi-llm-auction`、`ued-benchmark-saturation-map-2026`。
> 可信度：除标注外，关键数字/引用均来自论文正文或附录原文。

---

## 0. 名词：FM 是什么

**FM = Foundation Model（基础模型）。** 在 DiCode 里具体就是 **Qwen3-235B**（开源，论文 Table 5：`Qwen3-235B-A22B-Thinking-2507-FP8`，HuggingFace API 调用）。

- 论文用 "FM" 而非 "LLM" 是 open-endedness 圈（Hughes/Clune 派）的约定叫法，强调"海量人类数据预训练、当通用工具用"。
- 在 DiCode 里 FM 扮演**环境架构师（environment architect）**：不优化 agent、不直接玩游戏，而是**生成训练环境的代码**。§5 原话区分：别人用 FM 做 reward 设计/直接决策/技能分解，DiCode 用 FM "dynamically shaping the level distribution itself"。
- 对我方向：我说的"多 LLM teacher"，在 DiCode 术语里 = "多 FM generator"。我要把它的**单 FM 换成 N 个 FM + auction**。

---

## 1. Craftax 到底是什么（澄清一个根本误会）

**Craftax 不是一张写死的固定关卡，而是"固定规则下随机生成关卡的引擎"**（Minecraft/Roguelike 风格）。

- 每局用随机种子程序化生成新世界（地图/矿藏/怪物/楼层都不同），但**生成分布固定** = "原始 Craftax 分布" `Θ_target`。
- 特点：开局空手站地表第 0 层；要钻石装备/打深层 Gnome 得自己走通**十几步依赖链**（砍树→木镐→挖石→石镐→挖铁→熔炉→铁镐→下楼→…）；**奖励极稀疏**（只有解锁新成就才给一点）。
- **最终考试就在 `Θ_target` 上**（1024 关 held-out 测试集，mean return 口径）。

> "Craftax 限定好了"对一半：**规则和生成分布限定好了，但具体关卡每局随机。**

### 为什么直接在 `Θ_target` 上训练会卡死 → 这才是要"改关卡"的根本原因

baseline `PPO-GTrXL` 直接在原始分布上跑，**深层成就永远卡 0%**（Defeat Gnome Warrior/Archer 都是 0%）。根因 = **信用分配悬崖（credit assignment cliff）**：

> 要打到 Gnome Warrior，得在一局里碰巧连续做对十几步。随机探索下这几乎永不发生一次 → 这条路径永远没有奖励信号 → 梯度永远 0 → 永远学不会。
> 死锁：**需要先成功一次才能学会，但需要先学会才能成功一次。**

### "改关卡"改的不是考试，是练习题

- **考试 `Θ_target` 永不变**（最终评测就在它上面）。被"修改/生成"的是**训练用的中间关卡**，它们在原始分布里不存在。
- DiCode 的 Level 112：直接给 3 铁 3 煤 + 现成工作台熔炉，目标"造铁甲然后下楼"——把"造铁甲"单独拎出来、前置全包办，让 agent 第一次拿到非 0 梯度。
- 之后 Level 143 撤脚手架逼自采集 → 287 加怪压 → 532 起点放第 1 层只给一点资源逼取舍。

| 直觉误会 | 实际 |
|---|---|
| Craftax 是一个固定关卡 | 固定规则下**随机生成关卡的引擎**；考试 = 原始生成分布 |
| 改关卡 = 作弊改考卷 | 改的是**训练用练习题**，考试分布永不变 |
| 既然限定好了为什么还要改 | 考试太难、信号太稀，直接刷刷不出深层成就（baseline 卡 0%） |
| 改关卡有什么用 | 把长链拆成短链当垫脚石搭梯子，让稀疏奖励变密、信用分配可解 |

---

## 2. DiCode 为什么能当"纪录保持者"——五大机制（逐条对原文）

**① 它本质是 UED 方法，沿用全套成熟积木（所以能与 SFL/PLR/DR 同台且赢）**
§5 自列血统：**PLR 优先回放 + ACCEL 变异演化 + SFL(Rutherford 2024) learnability curation**。没发明新选关准则——learnability 选 parent（式7 `f(λ)` = SFL 的 p(1−p)）、PLR 公式（式2）采样 buffer。**唯一换掉的零件 = 把"随机/变异生成器"换成"FM 写代码生成器"。** 故能用同一 PPO-GTrXL + 1024 held-out 把 SFL/PLR/DR 全比一遍（Fig 2）。

**② 闭环反馈是真正的胜负手（ablation 实锤）**
**DiCode-OL（开环，去反馈）= 40.91 ≈ baseline PPO-GTrXL 41.54**（Appendix D.2 / Fig 7）。即**裸 FM 生成环境 = 没用，和不做课程一样**。增益全部来自闭环——FM 每轮收两份成绩单：
- `perf_p`：在 parent 关上 agent 学到了什么（§B.1.3）
- `perf_target`：agent 在**真 Craftax** 上各成就成功率
这就是 LLM 语义优势的落点——它读懂"铁甲只有 14%、是后续生存前置瓶颈"，定向造"先练铁甲"的关。p(1−p) 标量做不到这种**跨成就因果推理**。

**③ FM 当老师涌现出教科书式脚手架行为（Fig 4 + §C.1，Level 112→143→287→532）**
- 112：发铁+煤+工作台，只练"造铁甲+下楼"（压掉前置）
- 143：撤工作台和资源，逼自采集（同目标去脚手架）
- 287：加怪物压力，逼战斗中造甲
- 532：起点直接放 Floor 1，只给 1 铁 1 煤（不够同时造剑和镐）逼取舍，去打地精
FM 自动撤脚手架维持成功率 ~0.5（最近发展区 ZPD，Fig 4 底部曲线全程稳定 0.5）。**"理解科技树依赖、按依赖链设计渐进课程"——LLM 优势被实证。**

**④ 固定引擎执行 FM 代码 = 保证物理有效（避开世界模型幻觉）**
§3：FM 只写代码，由**固定 Craftax 引擎**执行 → 不可能幻觉出会飞的僵尸。只能在 `TaskParams`（怪物血量/spawn/需求衰减）+ `WorldBuilder`（放方块/给装备/设起点楼层）这个**受限但富表达力的 API**（§A.1/§B.1.2）里造关。**编译检查（§3.2）+ 固定引擎双保险**，编译失败直接丢弃，不自我修正。

**⑤ 突破点是结构性的（不是均值微涨）**
Fig 3 + §4.2：赢不是均匀提升，而是**攻破 baseline 卡 0% 的后期成就**。

---

## 3. DiCode 在 complete Craftax 上**没饱和**——数字证据

**"饱和"在不同 benchmark 含义相反**：jaxnav 是顶被摸到（win→1.0 无区分度）；Craftax 是底刚够到、顶还很远。**DiCode 是当前纪录，但绝对分数离天花板差得远。**

| 成就 | DiCode | 最强 baseline | 满分 |
|---|---|---|---|
| Make Iron Armour | **45%** | 14% | 100% |
| 进入 Gnomish Mines (Floor 2) | **30%** | 9% | 100% |
| Defeat Gnome Warrior | **11%** | 0% | 100% |
| Defeat Gnome Archer | **9%** | 0% | 100% |
| Make Diamond Sword | **6%** | 3% | 100% |
| mean return | **48.33** | 41.54 | Craftax 满成就上限远高于此 |

**关键判读**：相对 baseline 是 16% 提升、是把 0% 拉起来的"结构性突破"，但 DiCode 自己在后期成就**只有 6%–11%**。这根本不是饱和——是刚把门撞开一条缝。**它能当擂台恰恰因为没饱和**：天花板高、方法间区分空间巨大，谁能把那些 6%/9%/11% 往上推谁就刷新记录。与 jaxnav（顶被摸到、没空间）正好镜像。

---

## 4. DiCode 撞到的四大瓶颈（论文自承 + 数字暴露）

**瓶颈一：信用分配悬崖仍是主导，FM 只架了第一座桥。**
增益全来自闭环（去掉=回落 baseline），但最深成就仍只 6%–11%。FM 能架"先铁甲再下楼"的两三步依赖链，更深更长程的完整链路（钻石全套→深层 Gnome 全谱）还供给不稳——**架桥能力本身有上限**。

**瓶颈二（§6 明写）：固定引擎=物理有效，但限制"发明"。**
> 原话："it cannot yet invent entirely new physical laws or mechanics from scratch."
FM 只能在 `TaskParams` + `WorldBuilder` 里**配置**世界，不能造新机制。表达力被引擎旋钮数量锁死。详见 §5（配置 vs 发明）。

**瓶颈三（§6 明写）：FM 推理延迟/成本。**
> 原话："the inference cost of large FMs introduces latency that simpler UED methods avoid."
每轮调 Qwen3-235B 生成描述+代码，比 PLR/SFL 纯标量打分慢一个数量级，限制课程迭代次数和 archive 规模。

**瓶颈四（§6 点名 open challenge，最深一条）：无法可靠区分"有用垫脚石" vs "无用/干扰关"。**
> 原话："A key open challenge remains: how to reliably distinguish useful stepping-stone environments from uninformative or distracting ones."
DiCode 把判断关卡价值的信号交给**单 FM 语义直觉 + learnability(p(1−p)) + 谱系多样性约束**，论文承认这套**还不可靠**——FM 常生成无信息量/带偏的关。

### → 第四条瓶颈正是我方法的靶心
DiCode 用**单个** FM 判断"下一关教什么" → 单点偏差、会生成干扰关、覆盖不到的科技树分支是永久盲区。**多 FM auction 给的正是单 FM 结构上拿不到的信号**：N 个 FM 各提议下一关，用"互相分歧 / 互补覆盖度 / 被其他 FM 认同度"当 bid，直接攻这条 open challenge。

---

## 5. "配置 vs 发明"——FM 造关卡的自由度边界（瓶颈二展开）

DiCode 给 FM 的硬约束（§B，prompt 原文 1920–1924 行）：
> **"The Logic Flow is Fixed: You cannot change *how* the game processes actions**（e.g., you cannot rewrite the code for `update_mobs` to **make zombies fly**）."
> **"The Parameters are Mutable**: ... You CAN change those values via the API to control difficulty."

**FM 能做 = 配置（拧引擎已有旋钮）：** 调怪物血量/攻击/spawn/伤害倍率；摆已有方块（树/石/铁矿/工作台/熔炉/楼梯）；给初始装备资源；设起点楼层、改下楼击杀数；改成就目标组合。**全是引擎写死的物体和规则，只填数值和摆放。**

**FM 不能做 = 发明（造引擎里不存在的东西）：** 造新物体（如"魔法传送门"方块）；造新机制（僵尸会飞、铁直接合成钻石、水里能呼吸）；改物理规则本身（`update_mobs`/`do_crafting`/`change_floor` 的逻辑代码碰不了，只能改它们读取的参数）。

> 一句话：**FM 是关卡设计师，不是游戏设计师。** 能在 Craftax 积木里搭无穷关卡，但搭不出新积木块。

**双刃剑**：固定引擎保证物理有效（根除世界模型幻觉），代价是课程表达力被引擎旋钮锁死——等 agent 学完所有现有机制变体，FM 就没更难/更新的东西可教，无法像人类设计师那样"加一层新玩法"延续开放性。

---

## 6. 「假如允许造积木，造什么理想」+ 落子判断（关键决策）

### ⚠️ 强建议：第一篇（对标 DiCode）**不要开"造积木"口子**
立论根基 = DiCode 已把 SFL/PLR/DR/ACCEL 在固定引擎 + 同一 `Θ_target` 上比过 = 现成擂台。一旦允许 FM 造新积木：
1. **改变游戏本身** → agent 见过引擎里没有的东西，没法在原始 1024 held-out 上公平评测。
2. **失去对标对象** → 没人在"可发明机制"设定下比过 → 回到 jaxnav"自造擂台自己赢"的空中楼阁（正是要逃离的坑）。
3. **撞回世界模型老问题** → "造出来是否合法/可玩/可解"重新出现（DiCode 用固定引擎正为根除此）。

### 假如真要造积木，理想的新积木须同时满足三约束
1. **可执行、可验证**：以可编译、可由确定性引擎执行的代码形式存在（OMNI-EPIC / Genie 线），不是神经网络梦出的像素 → 能自动查"能跑/能解"（接 `generator-solvability-design` 的所有坑）。**理想积木 = 通过编译检查 + 可解性检查（flood-fill）的环境代码。**
2. **是"新依赖链的源头"，不是"换皮旧机制"**：最没价值 = 铁矿改名钛矿+数值微调。**最理想 = 引入原引擎不存在的新前置依赖**（如"必须先造火把照明否则深层不可见"→ 逼出采煤→造火把→管理光照→深挖的新子技能链）。**判据：能否在科技树长出一条新的非平凡分支。**
3. **难度可调、落在 ZPD 可达范围**：不能直接变不可解。**自带难度旋钮**，能从"脚手架拉满"平滑过渡到"完整版"，可编进课程维持 ~0.5 成功率。**判据：能否参数化成难度连续谱，而非一个开关。**

> 合起来：**理想新积木 = 一段可验证可执行的代码，引入一条全新的、可参数化难度的依赖链，扩展科技树深度/广度，而非给现有机制换皮。**

### 更深的陷阱 → 又绕回 auction
固定引擎安全是因为所有积木被人类验证过"是有意义的一部分"。FM 自由发明会造大量"花哨但无用甚至带偏"的机制（= 瓶颈四的 open challenge：区分有用垫脚石 vs 干扰物）。**评判一块自造积木好不好本身就是未解难题。**
→ **这恰恰让多 FM auction 在"可发明设定"下价值比固定引擎更大**：单 FM 自由发明无外部校验极易跑偏；N 个 FM 互相竞价/证伪/投票正是筛掉无用积木的天然机制。

### 落子建议
- **第一篇（对标 DiCode）**：守住固定引擎，只在"配置/排序练习题"层面用多 FM auction 赢 48.33。干净、有擂台、可发表。
- **第二篇 / Discussion 的 future work**：提"多 FM auction 天然适合校验自造积木"，把"造积木"作为愿景写进去，**不实做**。

---

## 7. 看全文后给方法架构的两个新插入点

**① auction 必须 bid 与 learnability 正交的东西。** DiCode 的 parent 选择已用 learnability（式7），archive 选择（式6）还加"促进谱系多样性"约束（只选子代全失败的 A/B 档关当 parent）。**learnability + 基础多样性两个位置 DiCode 都占了。** 我的 bid 信号得是单 FM 拿不到的——最自然 = **多 FM 之间的分歧/互补性**（谁覆盖了别人没覆盖的科技树盲区/谁的提议最被认同）。

**② 两阶段生成给 auction 天然插入点。** §3.2：FM 先生成自然语言**描述**（式8），再由第二个 FM 生成**代码**（式9）。auction 插在**描述层**——N 个 FM 各出一个 level 描述，auction 选 1~k 个进入代码生成。竞价的是"教学计划"（语义，FM 擅长），不是底层代码，干净且发挥 LLM 长处。

---

## 8. Craftax 上做 SOTA 的学术正当性（防"游戏场景无价值"质疑）

**结论：在 UED/open-ended RL 子领域完全站得住，只要 claim 钉在通用能力而非"会玩游戏"。**
- Craftax = **ICML 2024**（arXiv 2402.16801，Oxford FLAIR，jaxnav/SFL 同系）。它是**学术界专门设计的开放式 RL 标准 benchmark**（Crafter 放大加深 + JAX 加速），不是游戏移植。
- 它测的不是"会不会打游戏"，而是一组公认难的通用 RL 能力：**稀疏奖励探索、长程信用分配、层级技能、零样本泛化**。把 0%→11% = 解决了信用分配悬崖，这个 claim 不绑定"游戏"。
- DiCode 来自 Imperial College，整个 UED 圈（Oxford/Imperial）都用 Craftax/Kinetix/jaxnav 这类 JAX benchmark = 子领域通用货币。审稿人本就接受 benchmark 环境。

**挡"应用价值有限"的 framing**：不辩称 Craftax 像真实世界，而是定位贡献为"引擎无关的方法机制"。DiCode §1/§3 自己说引擎"可以是 game engine 也可以是 robotics 的 MuJoCo physics engine"——**同一套框架换引擎就能上机器人**。论文里写一句"多 FM auction 与引擎无关，Craftax 是 controlled testbed，可迁移 MuJoCo/Isaac"即可框住质疑。
**真正风险不是"游戏被鄙视"，而是"增益是否只是调参/换大模型"** → 由 auction on/off ablation 挡（同 DiCode 用 DiCode-OL 挡）。

---

## 9. 多 LLM debate 的角色架构（关键设计决策）

**原设想 coder + ambitionist + practitionist 的问题**：把"野心 vs 谨慎"做成两个固定人格拔河 = 本质是 DiCode 闭环已解决的"调难度维持 0.5 成功率"。审稿人会问"这比 DiCode 单 FM 闭环多了什么？" 而且两人格吵架只产生一个折中方案，**丢失了"多独立视角分歧/互补"这个核心信号**——而那正是我 vs DiCode 的命门（§4）。

**重新定位的架构 = 多 Proposer + 多维 Critic + Coder：**

| 角色 | 数量 | 职责 | 为什么 |
|---|---|---|---|
| **Proposer** | **N 个（≥3），异质底座** | 各自独立提一个 level 描述 | **auction 的弹药**——N 个独立提案的分歧/互补覆盖是单 FM 拿不到的核心信号。**最该"多 agent"的就是这里** |
| **Critic / 评审** | 多维度 | 沿不同维度打分：①可学性（太超前？=原 practitionist）②进步价值（够野心、攻盲区？=原 ambitionist）③覆盖新颖性（补别人没覆盖的科技树分支）④可解性/可编译 | 多维评审 = auction 的 bid 信号来源 |
| **Coder** | 1（描述层选 k 个后可并行） | 描述→可执行代码 + 编译/可解性检查 | DiCode 已验证必需 |

**核心判断**：①最该多 agent 的是 **Proposer**（必须异质，否则提案趋同、auction 退化成单 FM、卖点没了）；②ambitionist/practitionist **不该各是一个对立 agent，应降为评审维度**（两把尺子量每个 proposal，而非两 agent 对骂出折中）。
→ 原三角色**不够**：缺"多个独立 Proposer"这个最关键的多 agent 维度。

---

## 10. 模型选型（HuggingFace API，2026-06 核实）

**约束变更**：放弃本地 Ollama（235B 跑不动），改 **HuggingFace Inference Providers**（聚合 Novita/Together/Fireworks/DeepInfra/SambaNova/Groq，统一 OpenAI 兼容接口，一个 HF token）。
✅ 这自动对齐 DiCode "open-weights、可复现、不依赖闭源 API"约束（§1）——**不要用 GPT/Claude 闭源当 Proposer**，否则丢复现性 + 丢与 DiCode 可比性。

| 角色 | 推荐模型 | HF provider | 理由 |
|---|---|---|---|
| **Proposer #1** | **Qwen3-235B-A22B-Instruct-2507** | Novita/Together/DeepInfra | **DiCode 同款**（Table 5），复现锚点；至少一个 Proposer 用它 → "同等 FM 下靠 auction 赢"claim 才干净 |
| **Proposer #2** | DeepSeek-V3.2 / R1 | Novita/DeepInfra | 异质底座；V3.2 便宜（~$0.27/$0.41）适合开发期 |
| **Proposer #3** | GLM-5.x（Zhipu） | 多 provider | 第三个异质底座 |
| **Coder** | Qwen3-Coder 或 Kimi-K2-Code | 多 provider | 受限 API 代码，不需顶配 |

**核心判断**：三个 Proposer **必须不同底座**（Qwen/DeepSeek/GLM），不能三个都 Qwen——否则是"伪多样性"，退化成 self-consistency 采样，丢方法 motivation。开发期用便宜模型跑通 pipeline，正式实验再开齐三大底座。

---

## 11. 对标策略 + 那 8 篇 prior art 定位

**唯一要在主结果表上击败的 = DiCode（48.33）**。它是唯一在 complete Craftax 上把 SFL/PLR/DR/ACCEL 全比过的 = 现成擂台。其余全进 Related Work，无需重跑实验。

| 论文 | arXiv | 定位 | 处理 |
|---|---|---|---|
| PAIRED | 2012.02096 | UED 起点，DiCode 的先验 | 引为背景 |
| PLR | 2010.03934 | DiCode 复用其回放公式 | DiCode 的零件 |
| ACCEL | 2203.01302 | DiCode 复用其变异演化 | DiCode 的零件 |
| LLM-parkour 课程 | 2411.01775 | 同思路前驱，非 Craftax/无闭环 grounding | 引为相关 |
| **Eureka** | 2310.12931 | "FM 优化 agent（reward 设计）"正交赛道 | 引为对比 + **可偷机制见下** |
| **Text2Reward** | 2309.11489 | reward 设计类，正交 | 引为对比 |
| **Auto MC-Reward** | 2312.09238 | reward 设计类，正交 | 引为对比 + **可偷机制见下** |
| **EnvGen** | 2403.12014 | **最近祖先**，Crafter+JSON，被 DiCode 超越（表达力） | 详见 `EnvGen_拆解.md` |
| **GenEnv** | 2512.19682 | **同主题近邻**（标题"Co-Evolution LLM Agents & Env Simulators"表面像我），低撞车 | 详见 §15，必引区分 |

### reward 三篇：DiCode 没用，但有机制能用于我的 Proposer
- **DiCode 一篇都没"使用"它们**，只在 §5 归为"FM 优化 agent"正交赛道、与"FM 当环境架构师"划界。
- **Eureka（2310.12931）**：①一次生成多 reward 候选 + 用 RL 结果当 fitness 演化 ②reward reflection 把训练统计喂回 LLM 自改。→ **①是我 multi-Proposer + 下游信号择优的先例支撑；②是 Critic→Proposer 反馈范式**。
- **Auto MC-Reward（2312.09238）**：**多角色 LLM**（Reward Designer + Critic + Trajectory Analyzer）分析失败轨迹改 reward。→ **几乎是我 debate 架构在 reward 域的雏形**，引为"多角色 LLM 协作"的 prior art，我的区别 = 搬到 environment/课程域 + auction 竞价。
- **Text2Reward（2309.11489）**：LLM 生成 reward 代码 + 执行反馈 refine → 借鉴到 Coder 的"代码生成+校验+refine"环节。
- **新贡献定位**：把 Eureka 的"多候选+下游择优" + AutoMC 的"多角色协作"从 **reward 域搬到 environment/课程域**，并用 **auction**（异质 FM 市场 + 竞价）替代简单 argmax —— DiCode（单 FM）和这三篇（reward 域）都没有的组合。

⚠️ **待核实**：~~AutoMC 是否真三角色~~ 已读原文确认（见 §12）；Eureka 进化搜索 / Text2Reward 细节仍基于标题+§5 定位判断，**未读原文**，写 Related Work 前各读一遍。

---

## 12. Auto MC-Reward 拆解 + 与我方法的七维区分度（已读全文，2312.09238）

**论文**：Auto MC-Reward，CUHK/Shanghai AI Lab/SJTU/Tsinghua/SenseTime，CVPR 2024，PDF 在 new_direction 文件夹。

**它的精确机制（三角色串行流水线 + 迭代）：**
- **Reward Designer**：读 [任务描述+游戏信息+reward 要求]，写**可执行 Python reward 函数**（带 CoT 注释、Multi-Step Memory 字典、dense/sparse 两子函数，LLM 只定正负号、数值预设）。
- **Reward Critic**：**纯代码静态审查**（self-consistent / 语法语义/格式）——注意：**只是 sanity check 代码能不能跑，不评估 reward 好不好**。
- **Trajectory Analyzer**：训练后读 agent 轨迹，**归纳失败原因**（如"被岩浆烧死"）→ 给 Designer 改进建议（加岩浆惩罚 + pitch 约束）。
- 循环：Designer 出码 → Critic 过审 → 训 RL → Analyzer 诊断 → Designer 改。
- 测试床 = **Minecraft（MineDojo）单任务**（挖钻石 36.5%、找树、找动物）；产出 = **reward 函数**；student = RL。

**★七维区分度（我的方法绝不是 AutoMC 换领域）：**

| 维度 | Auto MC-Reward | 我的方法 | 真区分？ |
|---|---|---|---|
| LLM 产出物 | **reward 函数**（改奖励信号） | **environment/level**（改训练分布 Θ） | ✅ reward shaping vs environment design 是两个研究问题 |
| 目标 agent 能力 | **单一固定任务**（把挖钻石学好） | **通用 agent**（1024 held-out 整体变强，UED） | ✅ 单任务 vs 课程通用 |
| **多角色性质** | **流水线分工**（Designer/Critic/Analyzer 串行**不同工种**，无人竞争同一件事） | **竞争性同工种**（N 个 Proposer 并行**同岗位竞标**，互相竞争/分歧） | ✅✅ **最关键**：AutoMC 是装配线，**无 auction/竞争结构**；我是市场竞标 |
| 多样性/分歧信号 | **无**（三角色目标一致协同调一个 reward） | **核心机制**（bid=多 FM 分歧/互补覆盖） | ✅ AutoMC 不存在"多 LLM 互不同意" |
| Critic 作用 | 代码 sanity check（能不能跑） | **价值评审**（可学性/进步/覆盖多维打分喂 auction） | ✅ 查语法 vs 评教学价值 |
| 反馈信号 | 失败轨迹归因（why died） | agent 在 target 各成就成功率（learnability frontier） | ✅ 单任务诊断 vs 课程调度 |
| 底层范式 | reward 域，非 UED | **UED**（PLR+learnability+课程，DiCode 骨架） | ✅ 不同子领域 |

**一句话定位**：AutoMC = "三个不同工种的 LLM **串行协作**把**一个任务的 reward** 调好"；我 = "**N 个同工种 LLM 并行竞标**，用 auction 选**通用课程的下一关**"。**最硬区分 = AutoMC 里根本没有竞争/auction，它的"多 LLM"是装配线不是市场。**

**⚠️ 必防攻击**：审稿人可能说"你的 Proposer+Critic+反馈迭代 ≈ AutoMC 的 Designer/Critic/Analyzer 三角色"。**防御 = 强调竞争性**：AutoMC 是 **1 个 Designer 无竞争**，我是 **N 个 Proposer 竞标（auction 是核心）**。只要 N≥3 异质 Proposer + 竞价立住，攻击不成立——竞争性多 agent 市场是 AutoMC 结构上没有的。
**致敬+区别写法**：AutoMC 首次展示"多 LLM 角色协作设计训练信号"（致敬），但①reward 域非 env 域②协作流水线非竞争市场③无 UED 课程——我把"协作流水线"升级成"竞争市场"+搬到 environment design+UED。

---

## 13. reward vs environment 边界（澄清"我是否必须设计 reward 函数"）

**结论：不必，也不应该。在 Craftax/UED 设定里 reward 是 benchmark 给定的常量，我只设计 environment。**

- **Craftax reward 是引擎自带、写死的**：解锁新成就 +1（XP），死亡/饥饿给负（见 DiCode 附录 `change_floor`/starving rate）。**DiCode/SFL/PLR/我全用同一个原生 reward**。碰它就不能在同一 Θ_target 公平对标（mean return 48.33 就是用这个 reward 算的）。
- **病同（稀疏奖励够不到后期成就），药正交**：

| 路线 | 做法 | 谁 |
|---|---|---|
| 改 reward（shaping） | reward 不变=稀疏 → 额外加 dense 辅助信号把它变密 | AutoMC/Eureka/Text2Reward |
| **改 environment（我+DiCode）** | reward 不变=稀疏 → 改训练关让稀疏 +1 **更容易被拿到**（发铁+工作台让"造铁甲+1"第一次能拿到） | **我/DiCode** |

- **reward 在我方法里的角色**：训练时 agent 在我生成的每个练习关上拿的就是 **Craftax 原生 reward**（关变简单→原生稀疏 reward 第一次够得到→梯度非0→学会）；评测时在 Θ_target 上用原生 reward 算 return。**改 env 不需要改 reward。**
- **干净边界**：**AutoMC 改 reward 不改世界，我改世界不改 reward。我从不写新 reward 函数；AutoMC 的整个产出物就是 reward 函数。**
- ⚠️ 边界提醒：DiCode `TaskParams` 改"需求衰减/怪物伤害倍率"轻微影响训练关 reward 动态，但那是"配置关卡难度"非"设计 reward 函数"，且评测关 Θ_target 永远原生 reward——不模糊我与 AutoMC 的区别。

---

## 14. ★核心实验骨架：三档对照 B < A < C（模型无关性证明）

**策略**：先不锁模型刷 SOTA，再用相同机制做模型无关对照。升级成三档递进对照，是整篇论文的骨架结果图。

| 档 | 配置 | 证明什么 |
|---|---|---|
| **B = DiCode 复现** | 1 × Qwen3-235B（单 FM） | baseline 擂台（=48.33） |
| **A = 同模型 auction** | **N × 同一个 Qwen3-235B** + auction | **增益来自 auction 机制本身，与模型强弱无关**（控制变量=模型固定成 DiCode 同款） |
| **C = 异质 auction** | Qwen3-235B + DeepSeek-V3.2 + GLM-5.x + auction | **主结果/刷 SOTA + 异质底座额外加成** |

- 三者**同 student（PPO-GTrXL）、同预算、同 1024 held-out、同 seed 数**。
- **A>B** → 机制有用（不是换强模型赢的）；**C>A** → 异质性进一步放大增益。
- **B<A<C 这条递进曲线 = 论文核心图**，一次讲清"auction 有用 + 异质加成"两件事，且彻底挡死"你只是用了更强模型"的攻击（A 用 DiCode 同款模型也赢）。

**⚠️ A 档的微妙点**：N 个**同一个** Qwen3-235B 实例，多样性只能靠 **prompt persona 差异 / 温度采样**（比 C 的异质底座弱）。所以 A 证明"auction 结构本身有用（即便仅采样多样性）"，C 证明"异质底座额外放大"。两阶段定位清晰：
- **阶段一（刷 SOTA）**：直接上 C（最强异质组合），先把 >48.33 主结果坐实。
- **阶段二（模型无关对照）**：补 B 和 A，证明赢在机制不在模型。

---

## 15. GenEnv 撞车重核（已读全文 2512.19682，低风险需引用区分）

**GenEnv**：Princeton/Columbia/Michigan/Chicago，2025-12，*Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators*。
**背景**：旧记忆 [[novelty-check-currA-gate-prior-art]] 把它列为"头号撞车"，因其 **α-Curriculum Reward `R_env(p̂)=exp(-β(p̂-α)²), α=0.5`**（§2.2.2 式3）与旧方向的 gate 公式结构同构。**那是旧方向（jaxnav+gate）的事，gate 已弃。**

**对新方向重判 = 低撞车，但必引区分。三个根本不同：**

| 维度 | GenEnv | 我的新方向 |
|---|---|---|
| **student** | **LLM agent**（πagent，GRPO 微调 7B LLM） | **RL agent（PPO-GTrXL，非 LLM）** |
| **environment** | LLM 生成的**文本 task/工具调用**，**无固定引擎** | **可执行 Craftax 代码 + 固定引擎执行** |
| 测试床 | API-Bank/ALFWorld/BFCL/Bamboogle/TravelPlanner（LLM agent bench） | **complete Craftax**（RL bench），零重叠 |
| 生成器训练 | 单个 πenv 被 **RWR 微调**（权重∝exp(λR_env)）= 被训练的 LLM | Proposer **不微调**（FM 冻结，靠 prompt+auction 选择） |
| 多生成器/auction | **单个** πenv 自我进化 | **N 个异质 FM + auction 竞标** |

**关键洞察**：GenEnv 的 α=0.5 钟形门 ≡ ZPD 维持 ≡ **DiCode 已有的"维持 0.5 成功率"** = 我 baseline 的一部分，**不是我的创新点，无需在此公式争 novelty**（我靠的是多 FM 分歧/互补 bid，不是 p 钟形门）。

**行动**：GenEnv 标题表面像我（"Co-Evolution LLM Agents & Env Simulators"），审稿人必问 → **必引，用上表三根本区分划清（student=RL非LLM / 固定引擎代码非文本task / 多FM auction非单环境LLM微调）**。撞车点（α门）已随旧方向被抛弃，不构成 novelty 威胁。
⚠️ 子领域快移（DEGen Jan2026 等），camera-ready 前再扫 2026 预印本，novelty 陈述带时间戳。
