# v6 idea 设计审核与竞品分析结果

> 调研日期 2026-07-06。方法：5 路并行调研（LLM 课程流派 / 经典 RL / UED 新作与 DiCode 竞品 / Craftax SOTA / 轨迹技能链提取），约 40 次搜索 + 19 次 arXiv 原页核实。
> 标注约定：[已核实]=打开过 arXiv abstract 页确认标题+编号；[搜索结果]=来自搜索摘要未开原文；[记忆，未核实]=编号可能有误，引用前需再核对。
> ⚠️ SCALAR / CODE-SHARP 两篇 PDF 精读补充见文末附录（后补）。

**DiCode 本体确认** [已核实]：*Dreaming in Code for Curriculum Learning in Open-Ended Worlds*（arXiv 2602.08194，2026-02，Mitsides / Faldor / Cully，Imperial College，OMNI-EPIC 直系）。Semantic Scholar 被引数 0，**尚无正式后续引用，我们在窗口期内**。

---

## 一、long-horizon sparse reward 技能链的流派全景

### 1a. LLM-guided curriculum / skill library 路线

| 论文 | arXiv | 一句话核心机制 | 相关度 |
|---|---|---|---|
| Voyager [已核实] | 2305.16291 | 自动课程+可执行代码技能库+迭代 prompting，GPT-4 当 policy 玩 Minecraft | 技能库=防遗忘鼻祖，但 LLM-as-policy，无 RL student |
| DEPS [已核实] | 2302.01560 | 描述-解释失败-再规划-选择，failure-driven 修计划 | "解释失败→修课程"循环的原型 |
| JARVIS-1 [已核实] | 2311.05997 | 多模态记忆增强 LLM planner + goal-conditioned controller | 记忆机制可借鉴，不出课程 |
| Plan4MC [已核实] | 2303.16563 | RL 学基础技能，LLM 先验建技能依赖图做规划 | 链条知识在规划器、不在 student policy —— 与 v6 的关键对照 |
| OMNI [已核实] | 2306.01711 | FM 当"有趣性模型"挑任务训 RL（Crafter 域） | FM 选任务喂 RL student 的直系祖先 |
| OMNI-EPIC [已核实] | 2405.15568 | FM 生成任务+环境+奖励代码，档案库+有趣性过滤 | DiCode 直接前身 |
| CurricuLLM [已核实] | 2409.18382 | LLM 三步造机器人课程（ICRA 2025） | LLM teacher→RL student，链短 |
| EnvGen [已核实] | 2403.12014 | LLM 按 agent 弱项反馈自适应生成训练环境（Crafter/Heist） | "teacher 看弱点出题"闭环 2024 锚点；**无持久 learner 档案** |
| Eurekaverse [已核实] | 2411.01775 | LLM 写环境代码做渐进 parkour 课程（CoRL 2024） | LLM-UED 代表作，无依赖链概念 |
| **SCALAR** [已核实] | **2603.09036** | LLM 提技能规范(precondition/effect)，RL 逐技能 grounding，轨迹反馈(Pivotal Trajectory Analysis)反向纠 LLM 前置；Craftax diamond 88.2%（1.9×最佳基线）、Gnomish Mines 9.1% | ⚠️**最高优先级竞品**，详见附录精读 |
| **CODE-SHARP** [已核实] | **2602.10085** | FM 演化层级奖励程序档案（技能=带前置链的 Python 奖励程序），Craftax-Classic 中位数 6×，Extended 训 90+ 技能 | ⚠️**Cully 组兄弟作、与 DiCode 同月发布**，详见附录精读 |
| SkillGraph [已核实] | 2605.12039 | 技能=图节点，**带类型边编码 prerequisite/co-occurrence**，图随轨迹演化，依赖序检索+渐进解锁成自动课程 | 与 v6 机制图+共现最同构的 2026 工作，值得引 |
| From Trainee to Trainer [已核实] | 2606.17682 | RL checkpoint 自己(当 LLM)读失败轨迹→提环境配置修改；"训练中 checkpoint 更会诊断自身弱点" | 角度新但环境小、能力模型每阶段重建、无持久性 |
| DataEnvGym [搜索结果] | 2410.06215 | teacher 生成训练数据框成序贯决策，student 每轮回传 per-skill 错误（ICLR 2025 Spotlight） | teacher 维护 learner 进度最正式的工作，但 student 非 RL policy |
| HPRL-UED [已核实] | 2602.09813 | teacher 从 student 策略嵌入表征生成匹配能力的环境（非 LLM） | student 能力建模的非语言对照组 |
| Generative World Models of Tasks [已核实] | 2509.04731 | LLM 动态生成 HTN 指导多智能体 RL | 可引 |
| SGRL [已核实] | 2509.22008 | LLM 生成优先级目标函数+动作掩蔽引导 RL 探索（Crafter/Craftax-Classic） | 探索先验路线，仅 Classic |

另有一簇 2025-2026 "curriculum RL for LLM reasoning"（E2H 2506.06632、VCRL 2509.19803、Goldilocks RL 2602.14868 等）—— student 是 LLM 本身，related work 一句话划界。**"GenEnv"/"GenPool"为记忆虚构名，勿引用。**

### 1b. 经典 RL 路线

| 论文 | 编号 | 一句话核心机制 | 对 v6 的启示 |
|---|---|---|---|
| Reverse Curriculum Generation (Florensa 2017) [已核实] | 1707.05300 | 从目标态反向扩散生成起始态课程，逐步退回真实初始分布 | **概念上最重要**：反向课程练"以目标结尾的越来越长的后缀"，且 scaffold 最终完全撤除回真实初始态 —— DiCode 失败正是 scaffold 从不撤除。v6 的"回溯+已掌握才压缩"=依赖图上的 reverse curriculum，应显式引用 |
| RFCL [已核实] | 2405.03379 | 反向课程(demo 态重置)+前向课程结合 | 用了 demo 我们不能；"反向后必须接前向(真实初始态)"结构可借 |
| Go-Explore [记忆，未核实编号] | 1901.10995 / Nature 2021 | 档案化已发现状态，回到最有希望的状态再探索 | 依赖 state reset；造关可近似"回到状态" |
| Intelligent Go-Explore [已核实] | 2405.15143 | 用 FM 的"有趣性直觉"替换 Go-Explore 手工启发式 | "FM 判断哪个中间态值得攻坚"与 v6 焦点选择同构 |
| HER [记忆，未核实编号] | 1707.01495 | 失败轨迹重标记为达成了的目标 | student 非 goal-conditioned，适用性低 |
| Deep Skill Chaining [记忆，未核实] | ICLR 2020 | 从目标反向逐个学 option | back-chaining 的 option 化；"后缀完整"原则 |
| Robust Subtask Learning [搜索结果] | 2302.02984 | **证明子任务分开训→组合成完整链失败**，提出对抗性子任务加权 | **v6 隔离演练关的风险警告来源** |
| Subgoal Discovery via Free Energy [搜索结果] | 2412.16687 | 自由能+状态聚合发现瓶颈子目标 | 非 LLM 瓶颈发现参考 |

**灾难遗忘×课程**：replay/rehearsal 是持续学习正统（van de Ven 综述 2403.05175 [搜索结果]），但**没有找到把 protected rehearsal set 显式接进"自动课程生成器"的 RL 工作** —— v6 的 rehearsal 槽位在此交叉点无直接先例。

### 1c. UED 主线 2025-2026 新工作

| 论文 | arXiv | 一句话核心机制 |
|---|---|---|
| An Optimisation Framework for UED（含 Gen-SFL）[已核实] | 2505.20659 | 可证明收敛框架；**Gen-SFL 在 Craftax 有更优成绩**（RLC 2025）。⚠️记忆缩写"NCC"未在页面出现，引用用全名 |
| DEGen [已核实] | 2601.14957 | 动态环境生成 + Maximised Negative Advantage regret 近似 |
| CENIE [已核实] | 2502.05726 | state-action 覆盖+GMM 量化环境新颖性 |
| TRACED [已核实] | 2506.19997 | 转移预测误差入 regret + **Co-learnability**（训 A 关对 B 关的溢出）指标（ICLR 2026）—— 与技能链互动微弱交叠，值得引 |
| ATLAS [已核实] | 2511.12706 | UED 扩展到任务-关卡对联合课程，reward machine 形式化（AAAI 2026） |
| DRED [已核实标题] | 2402.03479 | 数据正则化环境设计；论证 UED 无监督生成造成分布漂移→zero-shot 差 |

### 1d. Craftax / Crafter 专项 SOTA

- **Craftax full（我们的战场）**：官方口径 PPO-GTrXL ≈ 18.3% of max（max=226），"remains unsolved"。打 full 版长链的只有 DiCode / SCALAR / CODE-SHARP 三家。**深层成就（4-9 层）held-out 完整口径下仍无人真正攻破 —— tier4=0 的天花板全领域都还立着。**
- **Craftax-Classic**：DeepMind TWM [已核实] 2502.01591 —— Dyna+NN tokenizer+block teacher forcing，1M 步 67.42%，首超人类 65.0%（MBRL 路线，Classic only）。
- **Crafter**：Achievement Distillation [搜索结果] 2307.03486（NeurIPS 2023）、Curious Replay 19.4%、DreamerV3 14.5%。历史线参考。
- **无监督层次技能发现** [已核实] 2601.23156（ICML 2026）：语法归纳从无标注轨迹分割层次技能结构，**就在 Craftax + Minecraft 上评测** —— 与轨迹提取需求直接相关。
- Multi-Agent Craftax [搜索结果] 2511.04904：多智能体版，非竞品。

---

## 二、从行为轨迹提取"关键技能链/时序模式"

按"输入=成就时间戳序列+胜负标签，预算=本地轻量计算+少量 LLM 调用"约束评估：

| 方法族 | 代表 | 强于共现矩阵之处 | 成本 | 判定 |
|---|---|---|---|---|
| **时序化成就序列（首达成时间排序+n-gram 链统计）** | PrefixSpan 系 [记忆]；SkillGraph 2605.12039 | 共现矩阵**丢弃顺序**，而链的本质就是顺序；2/3-gram+胜负对照几乎零成本把"共现"升级成"有向链" | 纯本地零 API | ★首选升级 |
| **判别性模式对比（成功 vs 失败）** | 判别性序列挖掘 [记忆] | 失败 episode 里"链断在哪一环"是攻坚焦点的最强证据（当前只用获胜局） | 纯本地 | ★首选升级 |
| 语法归纳层次分割 | 2601.23156 [已核实] | 自动归纳技能组成关系，就在 Craftax 验证过 | 中 | 中期可选 |
| LLM 轨迹总结经验 | Reflexion [记忆] 2303.11366；ExpeL [已核实] 2308.10144；**AutoManual [已核实] 2405.16247**；AutoRefine [搜索结果] 2601.22758 | style_note 的正统血统。AutoManual 教训直接适用：**规则要在线验证、合并、淘汰**，不能只追加 | 每 session 1-2 次调用 | 已在做，按 AutoManual 补生命周期 |
| 因果发现/前置推断 | AGWM [搜索结果] 2605.06841 | **"从 gameplay 推 tech tree"的专门工作不存在** —— 这件事本身有 novelty 空间 | 完整因果发现贵且脆 | 用时序统计近似，别上重型因果 |
| Successor features | 1905.05731 [搜索结果]；2412.16687 | 能找瓶颈态但需额外训练、输出非 LLM 可读 | 高(GPU) | 不推荐 |

**结论**：比"共现矩阵+动作直方图"更强且几乎免费的升级只有一个方向 —— **把时间序放回去**：首达成时间排序 → n-gram/前缀链统计 → 成功/失败对照 → 输出"有向链+断链环"给 modeler。同时与 SkillGraph 差异化：我们的依赖边从 student 自己的行为统计长出来，不违反"禁 tech-tree 先验"约束。

---

## 三、对 v6 设计的批判性评估

### 撞车区（必须引用并区分）

1. **"LLM 维护技能前置结构+RL grounding"** ↔ SCALAR、Plan4MC、CODE-SHARP。区分点：它们把链条知识放在规划器/奖励程序/per-skill policy 里，执行时拼链；我们要求**单一 PPO student 在权重里内化完整链**并在 held-out 零辅助复现 —— 更硬也更干净的主张。
2. **"teacher 看 student 弱点出题"** ↔ EnvGen、DataEnvGym、From Trainee to Trainer。区分点=**持久性**：它们每轮重建诊断，SiegeNotebook 是跨 session 持久作战档案+代码驱动 mastery/遗忘判定。
3. **共现/依赖边** ↔ SkillGraph、TRACED Co-learnability。
4. **rehearsal 防遗忘** ↔ 持续学习 replay 正统 —— 非 novelty，标准手段的正确应用，引文即可。
5. **反向回溯课程** ↔ Florensa 1707.05300：v6"沿机制图回溯、已掌握才压缩"=依赖图上的 reverse curriculum —— 30 年血统的理论支撑，主动引。

### Naive 区（前沿有更好做法）

1. **共现矩阵丢顺序**（见第二节）—— 最大低垂果实。
2. **隔离演练关有理论风险**：2302.02984 证明"子任务隔离训→组合失败"正是我们诊断 DiCode 的病。剥离干扰的单技能重复训练 = 可能变成我们自己引入的另一种 chain-severing。**演练关 mastery 判定必须以完整链整合关的 SR 为准，隔离关 SR 只当中间信号**；"逐步加回干扰"要绑 SR 闸而非固定步数。
3. **style_note 自由文本无验证/淘汰**：AutoManual 核心教训 —— 离线追加的 insight 会漂移，规则需在线验证可靠性、合并冗余、淘汰失效。style_note 应带"最近 N session 是否仍被证据支持"的生命周期。

### 真 Novelty 区（可主张，两路独立搜索一致确认空位）

1. **显式命名并实证"scaffold 切断长链→held-out 迁移失败"失败模式** + 保链修复。空位干净（最近邻 DRED 2402.03479 分布漂移视角、2302.02984 组合视角）。建议命名如 *chain-severing scaffolds*。
2. **LLM 持久 student 能力档案（SiegeNotebook）接 UED 课程生成** —— 无先例。
3. **多 LLM auction 市场机制选课程** —— 零竞品（HARBOR 2502.12149 仍是"LLM 当买家"行为模拟）。
4. **从 student 自身行为统计推断依赖图（不喂先验）** —— 连"从 gameplay 推 tech tree"专门工作都没有。

### 5 条落地 design suggestion（满足全部约束：student 锁 RL / 无人类演示 / 无 tech-tree 先验 / API 预算）

1. **共现→时序链挖掘（本地零 API，一天实现量）**：per-episode 按成就首达成时间排序，2/3-gram 链频率统计，成功/失败判别对比，"断链最频繁的一环"直接写进 modeler prompt。
2. **completed gate 形式化为 "suffix-preserving compression"**：压缩只能从链头进行（已掌握前缀→初始库存），且每个焦点课程必须始终保留 ≥1 个**从真实初始态出发的完整链"整合考试关"**，conquest 判定只认它的 SR —— 把 Florensa"撤 scaffold"原则写成代码闸，也是可消融对照。
3. **隔离演练关加"再整合闸"**：隔离关 SR 达标 ≠ mastery；K session 内须在整合关复现该环节，否则自动回退未掌握 —— 回应 2302.02984。
4. **style_note 加 AutoManual 式生命周期**：每条心得带 evidence counter（最近被成功 episode 支持次数），连续 M session 无支持则 modeler 复审淘汰/降级 —— 一次 LLM 调用顺带完成。
5. **IGE 思想做焦点选择第二信号**：卡关候选并列时，从失败 episode 断链统计里选"离已掌握前沿最近的一环"而非最深的一环 —— 防焦点漂移到 tier4 空转，与早期误判双闸互补。

### 竞品行动项

- **必须精读**：CODE-SHARP 2602.10085（Cully 组兄弟作、同月发布）与 SCALAR 2603.09036（核对 88.2% 评测口径）。→ 见附录。
- Gen-SFL (2505.20659) 在 Craftax 有报告成绩，考虑纳入 baseline 讨论。

---

## 附录：SCALAR / CODE-SHARP PDF 精读补充

（待补：本地 PDF 精读结论 —— 评测口径、与 v6 的逐点对比、论文里如何回应。）
