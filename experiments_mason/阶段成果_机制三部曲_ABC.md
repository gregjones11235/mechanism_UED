# 阶段成果:机制三部曲 ★A / ★B / ★C(supply steering / admission gating / curriculum fidelity)

> 2026-07-11。定位 = 本线(Skill-Preflight UED,14B code-level)截至本日的机制级成果汇总,
> 供组会/PI 汇报与联合论文素材调用。评测口径全文统一:官方协议 `eval_checkpoints.py`,
> 1024 个 seed=0 固定 held-out Craftax 世界,mean episode return,与 Alec 线(235B)同批世界。
> 证据底座:`official_eval_v2.md`(主结果)/ `probe_结果与瓶颈定位.md` v4(泄漏审计+裸复验)/
> `C2lite_实现记录.md` + `prereq_autoextract_勘察.md`(★C 实现与防御)/ `stage1_C/`(修复级联)。
>
> **状态口径(重要):★A、★B = 已验证 + 已跑(有官方 eval 数字);★C = 已实现 + 单测/离线
> 验证完成,训练读数待短跑(session 10)——本文档不为 ★C 主张任何训练数字。**

---

## 0. 一览表

| 机制 | 一句话 | 状态 | 关键证据 |
|---|---|---|---|
| **★A** skill-graph scheduler(supply steering) | 从 held-out 逐技能 SR 定位学习前沿,定向生成 | ✅ 已验证已跑(`85qid2ev` / ext `z8jygtyw`) | 裸 14B 平台 ~18 → **31.07**(+13 配对差);深层技能从死零冒头 |
| **★B** preflight gate(admission gating) | 新任务入库前用当前 policy 滚一轮,拒不可学件 | ✅ 已验证已跑(`u1qjqror`→`u1gjqror`) | 终点 **29.81**(与 +A 噪声带内不可分);机制审计已完成诚实降级(见 §2) |
| **★C** curriculum fidelity(三层修复,C-2-lite) | 治 100% 脚手架泄漏:prereq 前沿 + one-step prompt + C-0 静态闸 | 🔶 **已实现未跑**:51/51 单测过,离线证据齐,短跑(队列第 2 项)待开 | 302 任务审计定位病因;task_19 复刻件过闸实测;前置图 78% 边可从环境源码编译 |

三者分工一句话:**A 决定练什么,B 决定收什么,C 保证任务真的是它声称的东西**——三条正交质量轴
(供给方向 / 可学性 / 忠实性),第三条是本线独有发现(现有全部闸门零覆盖,泄漏由此穿过)。

## 1. ★A:frontier 调度 = 天花板抬升的主要来源(已验证)

**机制**:每 session 用 held-out 逐成就 SR 计算 tier 掌握度 → 定位最浅未掌握层 → 未掌握成就
清单注入生成 prompt([Curriculum focus])。flag:`use_scheduler=true`。

**证据(315M 全程配对,official_eval_v2)**:
- 裸 14B 平台正式实证:baseline_ext 1200 后四点钉死 17.7-18.8,另含 @900 的 -6.4 中段回撤;
- +A_ext 突破至 **31.07**(+13.2 配对差),路径剧烈震荡(-7.1 @1500)后冲顶;
- 一代 157M 数据(ablation_v1)给出分层细节:深/中层技能(open_chest/enter_dungeon/find_bow/
  fire_bow/iron 族/diamond 资源)机制臂大面积从零冒头,baseline 全程死零;tier0-1 三臂重合饱和
  ——深层收益不牺牲基础;
- 因果链在日志可见:scheduler 先切 targets → 生成随之定向 → 技能随后冒头(先定向、后生成、再学会)。

## 2. ★B:preflight 闸 = 可学性轴唯一闸门,画像已诚实校准(已验证)

**机制**:evolution 产出的新任务在入库前,复用生产路径(load_tasks_from_env_codes →
evaluate_new_tasks → calculate_scores_from_snapshot)做冻结 policy 短滚,`route()` 按 SR
接受/拒绝。flag:`use_preflight=true`。成本 ~10min/设计 session。

**主数字**:+A+B 终点 29.81,与 +A 31.07 在 ±1.5 噪声带内不可分 → **B 不抬天花板**。

**代码级机制审计后的诚实校准(probe 文档 §4,汇报时请用此口径)**:
- B-1 写入的 `learnability_score` 全管线**零读取方**(采样与激活均读 priority_score)→
  本管线中 +A+B ≈ +A + ~10min + rng 偏移;
- "path quality(-0.6 vs -7.1 回撤)"降级为 **observed but unattributed**(n=1;probe 臂零
  干预却最平滑,进一步削弱因果归因);
- 保留的主张:**拒必有理**(2/2 真 too_easy)、**廉价**、**保险角色**(为激进 frontier 推进
  封顶下行——probe 即受保运行,零赔付 ≠ 零价值);
- 白送升级项(押后队列):PLR 采样接入 learnability 字段,"只写不读"→"闸+信号源",config 级改动。
- **C 上线后 B 反而更关键**:剥脚手架的裸任务更难,正需 B 判可学边界。C-2 抽查直接复用 B 的
  rollout 机器。

## 3. ★C:curriculum fidelity 三层修复(已实现,未跑 —— 本节为补充部分)

### 3.1 为什么需要 C:发现链(全部第一手)

1. **五臂消融**定位:三种机制配置(+A / +A+B t=0.6 / probe t=0.2)收敛 29-31 平台,阈值
   probe 排除"解锁太晚"假说(frontier 22/25 session 稳定 tier 3,tier-3 技能仍全零);
2. **302 任务全量 AST 审计**(scaffold_audit,四签名):泄漏率 **100%,含无定向 baseline**
   ——脚手架是范式固有生成形态,非机制诱发;溯源双重结案:种子任务 few-shot 4/4 示范 +
   persona prompt 明文规定("SCAFFOLD them...list as Completed");
3. **裸复验三 regime**(bare_reverify,冻结 checkpoint 配对 rollout)与迁移模式逐点对齐:
   已内化(0.76→0.70)/ 脚手架承重(**0.65→0.01**)/ 给脚手架也不会(钻石族 ~0→~0);
4. 断裂带物理学:终端奖励到达概率 = 中间步成功率连乘(钻石剑联合 ≈0.2-0.3%,与裸复验实测
   同量级)→ 长链**不是学得慢,是学习信号不存在**;
5. 概念产出:**可学性 vs 忠实性两条正交质量轴**——编译/lint/preflight 全部只覆盖前者,
   后者零覆盖 → 泄漏由此穿过;所有五臂成绩都是"脚手架课程的迁移残余"。

### 3.2 三层设计与实现(2026-07-11 完成,commit f6bfeff + d1e773d)

| 层 | 治什么 | 实现 | flag |
|---|---|---|---|
| 调度侧(§1) | tier 均值被早熟成员灌高,前置未齐即发深层任务 | `pick_target(frontier_mode="prereq")`:候选 ⟺ 自身未过 且**每个直接前置**单独 ≥0.3;底层新建 67 节点直接前置图(逐条对照 craftax 1.4.5 源码) | `frontier_mode=prereq`,`prereq_threshold=0.3` |
| 生成侧(§2) | prompt 明文 + few-shot 教出的"压缩已掌握前缀" | one-step 契约文本(恰好一步裸露;只许脚手架未掌握非焦点前置;已掌握绝不预标)+ mastery 三档快照,双注入(设计段 [Curriculum focus] / 编码段规则块**明文压过示例**) | `scaffold_prompt=auto`(随 prereq 自动开) |
| 验证侧(§3) | 忠实性轴零闸门,系统无法自愈 | C-0 静态闸:R1 预标已掌握 / R2 预标 relevant / R3 焦点直接前置被脚手架(预标∪库存授予∪起始层);违规罪证进现有 reflection 模板重生成 ≤2 次 | `use_scaffold_gate=true` |

全部 flag 默认关,默认路径与旧 run 逐位一致(回归测试钉死)。刻意不禁近距刷怪(S3)——
终端技能重复练习正是机制臂 +12 分的来源。

### 3.3 离线证据(实现质量,非训练效果)

- **单测 51/51**(24 新 + 13 旧回归 + preflight 旧套),含设计稿指定的两种验证:probe 末点
  真实 eval JSON 回放判据(不变量:发出的每个目标零断前置、钻石族全拦)+ task_19 复刻件过闸
  (真末点快照下精准打出 R1+R3,罪证逐行点名可直接喂修复 prompt);
- **前置图先验防御(勘察报告)**:抽取原型实测 **78% 的边(72/92)今日即可从 craftax 源码
  全自动编译**,半天工程可至 ~90%;真正的人类输入收敛为一张 ~10 行的 AND 投影表(析取源
  选正则分支)。且 spike 反向抓出手工图 3 条漏边 + 1 条不精确边(已修)——**环境规格级先验
  用编译获得比人手更可靠**,UED 纯度质疑的正面回应;
- 顺带确认语义地基:COLLECT_*/FIND_BOW/MAKE_* 为库存状态成就(inventory.x>0 即触发)→
  闸的库存授予检查为字面游戏真值。

### 3.4 待验证(短跑三判据,队列第 2 项,~7h)

1. 泄漏语义:audit 复扫,"预标已掌握技能"比例 ↓;
2. **tier-3 裸指标动**(核心判据):iron/diamond 族 held-out SR 相对 probe 同期 ↑;
3. 拒绝率不失控:preflight <30%;ScaffoldGate repaired/dropped 比 = 14B 对 one-step prompt
   服从率的首个实测(dropped 飙高 → 触发 ★C-1 修复级联接入条件)。

已知观察点(回放发现):末点处合格池 > cap 6,断裂带锚点 make_iron_pickaxe 排第 7——判据
不变量未破,但 cap/排序对"火力集中"有影响,短跑看调度日志再定(cap 提 8-10 为一行改动)。

## 4. 命名消歧(汇报时防混淆)

- **本文 ★C = curriculum fidelity(C-2-lite 三层)**,治 30 分的墙(质量问题),关键路径;
- **proposal_C = 幻觉修复级联(C-0 API lint + C-1 修复)**,治吞吐(幻觉→丢件),
  Stage 1 离线已过判据(30/30 检出、修复成功率 1.0、~2.1k token/次,附带发现:幻觉 =
  跨游戏常识污染而非随机噪声,最典型是 14B 把源码 typo `SOLIDER` 纠正回 `SOLDIER` 反而报错;
  prompt 清单干净 → 幻觉为模型内生),**未接入**,板凳待命(仅当短跑显示 one-step 约束使
  重造率飙升时上,接入 = 一个 hook);
- 两处 "C-0" 挂同一 hook 位,实现时可合并为一次 AST 遍历;两处 "C-2" 同为裸复验概念的
  不同落地形态。

## 5. 主数字表(五臂,官方口径,mean_return)

| update (steps) | baseline_ext | +A_ext | +A+B (t=0.6) | probe (t=0.2) |
|---|---|---|---|---|
| 1200 (157M) | 18.27 | 20.00 | 20.25 | **23.19** |
| 1500 (197M) | 18.76 | 12.94 ▼ | 29.16 | 26.38 |
| 2100 (275M) | 17.66 | 28.93 | 32.46 | 29.61 |
| **2400 (315M)** | **17.88** | **31.07** | **29.81** | 29.06 |

尺度轴:14B + 机制在 315M(≈235B 线 2e9 预算的 16%)达 29.8-31.1,对照 235B 的 44.58(中段)
/ 48.33(终值)→ 约 16% 算力达 ~2/3 成绩(regime 对照,非受控比较)。同 seed LLM 采样方差
≈ ±1 分(跨 run 复现测得)。**目标:★C 修复版 +A+B @2e9 冲 43+(45 的 3% 内)。**

## 6. 汇报口径(英文,一段话)

> "Our mechanism story has three legs. Steering (A, verified & run): frontier scheduling
> breaks the bare-14B plateau from ~18 to ~31 — a +13 paired gap under the official protocol,
> with deep skills rising from flat zero. Gating (B, verified & run): the preflight gate
> matches A's endpoint at ~10min/session; a code-level audit honestly downgrades its
> path-quality claim to observed-but-unattributed, while its rejections remain justified and
> its insurance role stands — and it becomes MORE important once fidelity repair lands.
> Fidelity (C, implemented, short-run pending): a full audit of 302 generated tasks found
> 100% scaffolding leakage across all arms including baseline — traced to the paradigm's own
> prompts and few-shot seeds — with bare re-verification showing per-task collapse up to
> 0.65→0.01. C is a three-layer fix (prerequisite-frontier scheduling on a dependency graph
> 78% mechanically compiled from the environment source, a one-bare-step scaffolding
> contract in both prompts, and a static fidelity gate with evidence-carrying regeneration),
> fully flag-gated, 51/51 tests passing, replay-validated on real endpoint evals. Training
> readouts await a session-10 short run; the 2e9 target is 43+ against the 235B line's
> 44.58/48.33."

## 7. 下一步与诚实清单

**顺序(队列)**:短跑三判据(第 2 项)→ 过则 2e9 主跑(第 3 项,~55-65h)→ 官方 eval +
gap 报告(第 4 项);★C-1 条件触发;baseline@2e9 / seeds / PLR 接 learnability / 跨线审计押后。

**诚实清单**:全部 n=1;★C 零训练数字,三层直接攻击课程供给质量但**不保证** 43(第二层
未知数:课程完美后 14B 容量/2e9 预算是否够长链);fallback 即使停在 35-40,三层修复 +
断裂带机制分析构成完整故事;+A vs +A+B 终点不可分;四签名清单可能不完备;"终端技能定向
解释 +12 分"为 post-hoc(可做技能级归因验证)。
