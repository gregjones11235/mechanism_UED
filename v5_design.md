# v5-debate 设计文档

> 起草 2026-07-03。在 v4(auction, 无gate无endorsement, ambitious读SR序列+thinking) 基础上,
> 新增 **modeler agent + proposer↔modeler 沟通**,治 tier2→tier3 断档。

---

## 0. 动机(为什么要 modeler)

根因诊断(2026-07-03 亲验 v3c graphml):tier2→tier3 断档**不是**式6 parent 闸门失效,
而是式6依赖的 **A/B/C/D status 标签会"过期冻结"**——`update_node_status` 只更新本session训过的关,
一个关评上A后不再重训则永久钉A。v3c 中 51%(48/94) 的A/B parent 是≥5 session前评的旧快照
(最老 task_67 是27 session前评的A,SR96%)。式6忠实地选A/B parent,但那个A/B是历史快照,
**系统性高估 student 当前掌握度** → ambitious 从"名义已掌握实则遗忘"的tier-2 parent推tier-3 → 断档。

**modeler 的不可替代价值 = 全局 + 当前(held-out,非过期status) + 双向纠偏。**
- 用当前 held-out SR 实时判断 student 真实状态(绕开status过期)
- 全局视角(跨archive跨tier),看到单proposer看不到的"哪个前置在遗忘、哪个方向盲目堆砌"
- 双向纠偏:遗忘→补训、堆难→修正

modeler 不替代式6,而是给式6的过期status打补丁。

---

## 1. 对照设置(验证"竞争是否必要")

两个对照,**都配 modeler=GLM-5.2**(受控变量):

| 对照 | proposer | modeler | 差异 |
|---|---|---|---|
| **X** | 1× DeepSeek-V4(ambitious, thinking on) | GLM-5.2 | 无竞争(单proposer,auction退化为直接采纳) |
| **Y** | 2× DeepSeek-V4 + Qwen3.5-397B(都ambitious) | GLM-5.2 | 有竞争(2 proposer竞标top-k) |

唯一差异 = 第二个 proposer 带来的竞争。干净隔离"竞争"变量。
两个GPU同时跑。

## 2. 模型分配(v1三模型全用上)
- proposer: DeepSeek-V4(推理最强, ambitious, thinking on) [+ Qwen3.5-397B(异质竞争者, 仅Y组)]
- modeler: **GLM-5.2**(native structured output + calibration,最适合诊断student状态)

v1原三模型: Qwen3.5-397B(breadth) / DeepSeek-V4(ambitious) / GLM-5.2(feasible)。
v5重新分工: DeepSeek+Qwen当ambitious proposer, GLM当modeler。

---

> **注意**: 本文档描述的是 **方案A(合作补位式)**,只是 v5 的一种尝试。
> 后续会补充其它方案的 seed(如竞争式等),此文档不是最终定稿。

## 3. 方案A(合作补位式)的设计决策

**核心性格: 合作补位,无竞争。** 两 proposer 不抢地盘,后手补前手空位,产出互补都保留。

- soft: modeler给建议,proposer保持ambitious自主(不倾向质疑modeler)
- A/B/C/D式6分级 = **和baseline完全一致**(strict frontier选parent + 编译检查)
- **取消 auction 的"竞争性"淘汰**(无中标率、无 endorsement 互评),但**保留一道非竞争的质量筛选**:
  每轮 24 候选选 12 注入(对齐 baseline 产量),标准 = AmbitionGain+Learnability(方案A)。**详见 §8**。
  (注:早期设计曾定"两关全保留不筛",2026-07-03 修正为筛到对齐 baseline——见 §8.1 公平性论证)
- **顺序补位造关(先后手每轮轮换)**:
  - LLM_first: 看modeler诊断 → 造关(标注类型: DEPTH/BREADTH/CONSOLIDATE)
  - LLM_second: 看modeler诊断 + LLM_first已造类型 → 补未覆盖的**有价值**类型
- **无二次校准**(砍掉原"1轮双向"第5步): modeler只在造关前诊断一次,proposer据此造关,直接编译注入
- modeler = **GLM-5.2**
- 新增 **StudentProfileLog**: 每session落盘held-out profile快照→时序,供随时评估+modeler建模
- modeler 可**调取任意历史关卡详情**(复用/微调推给proposer作参考)
- **proposer不再追"中标率"**(无竞争), 改追"补modeler诊断出的、前手未覆盖的有价值空位"

### 三种关卡类型(proposer标注,modeler据student状态建议)
1. **DEPTH 深度前进**: 打通更深的关(tier前进)
2. **BREADTH 广度探索**: 探索更多没碰过的技能点(新成就家族)
3. **CONSOLIDATE 巩固补缺**: 补modeler发现的能力遗忘缺口 (**仅modeler能发现**,需对比profile时序)

## 4. modeler 的核心能力要求
必须能区分三种现象(prompt需专门设计):
1. **NORMAL_EARLY**: 早期student正常起步弱(能力会涨,别慌)
2. **STALLED**: 中期(tier3出现后)走错方向学不会(真断档,要纠)
3. **NOISY**: 随机采样噪声(单点低≠真失败,别被骗)
额外: **FORGETTING**: 曾掌握现遗忘(补训信号→CONSOLIDATE)
判据数据: SR时间序列(判rising/flat) + 训练阶段(判早/中期) + 父子关K=10序列(判抖动)

---

## 5. 方案A 每 session 流程(合作补位式)

```
1. [baseline一致] 式6选frontier parents (A/B/C/D strict + 编译检查口径不变)
2. [新] StudentProfileLog: 落盘当前held-out profile → 时序
3. [新] modeler(GLM)诊断: 每个能力域状态(NORMAL/STALLED/NOISY/FORGETTING)
        + 每个parent的建议类型 + 可复用的历史参考关
4. [新] 顺序造关(先后手本轮轮换):
   - LLM_first : modeler诊断 → 造关(标注DEPTH/BREADTH/CONSOLIDATE)
   - LLM_second: modeler诊断 + LLM_first已造类型 → 补未覆盖的有价值类型
5. [新] 筛选: 24 候选 → 选 12(方案A: AmbitionGain+Learnability, 对齐 baseline 产量, 见 §8)
6. [baseline一致] 编译检查 → 注入训练(无 auction 竞争淘汰, 但有 §8 质量筛选)
```

对照(方案A内):
- **对照 X = 1 proposer(DeepSeek) + modeler**: 无补位(单proposer,无先后手)
- **对照 Y = 2 proposer(DeepSeek+Qwen) + modeler**: 有顺序补位分化
- X vs Y 验证: **加一个"补位proposer"是否提升覆盖/性能**(验证"协作补位是否有效")
- **先实现对照Y**(对照X暂放)

---

## 6. 新增组件清单(代码)
- A. `auction/student_profile_log.py`: 每session存profile→时序查询
- B. `auction/modeler.py`: GLM调用, structured output schema, 可查archive任意关卡
- C. gen_manager 加 modeler+顺序补位分支(config开关控制, 纯叠加不改v4路径)
- D. 新建 persona 副本 `persona_ambitious_coop.py`(不覆盖原persona_ambitious.py):
     加 {MODELER_GUIDANCE}/{PEER_ALREADY_MADE}/{REFERENCE_LEVEL}/{MY_TURN_ORDER} 占位符
     去掉"追中标率", 改"补前手未覆盖的有价值类型"
- E. 先后手轮换逻辑
- F. config: auction_c_v5y.yaml(2 proposer + modeler)  [对照X的v5x暂放]

## 7. 系统完成后: 冒烟测试(用户要求)
smoke test 验证 modeler调用/structured输出/顺序补位流程/profile落盘/config可跑。

---

## 8. 关卡筛选策略(每轮生成 vs 保留)

**背景(2026-07-03 亲验真实日志)**: 三种 arm 每轮的生成/保留数根本不同,直接影响对比公平性——
若 v5y 每轮往 archive 灌的关比 baseline 多,优势可能来自"多生成"而非"生成得更好"。故 v5y
**引入非竞争质量筛选**,砍掉最差候选。当前方案A 定 **保留 18**(24→18,砍 ~25% 最差),
介于不筛(24)与 baseline(12)之间 —— 产量为 baseline 的 1.5×(分析时须注意此产量差,
不是严格等产量对比;若要严格对齐 baseline 可把 coop_select_k 设 12)。

### 8.1 三种 arm 的真实生成/保留数

| Arm | proposer 数 | 每轮**生成** | 筛选机制 | 每轮**保留**(注入archive) | 保留率 |
|---|---|---|---|---|---|
| **baseline** (DiCode 原版) | 1 | 12 | 无淘汰 | **12** | 100% |
| **v1–v4 (auction)** | 3 (ambitious/breadth/feasible) | 36 = 3×12 | top-k `auction_k=10`(Coverage+AmbitionGain+Learnability bid) | **10** | ~28% |
| **v5y (coop)** | 2 (都 ambitious) | 24 = 2×12 | **见 8.2(三变种)** | **18**(方案A选定后) | 75% |

- baseline 生成数 12 = `num_generations` 相关(每轮对采样出的 frontier 各演化;40 轮零方差)。
- baseline **不淘汰关卡**(日志 `removed from archive`=0);A/B/C/D status 只作**采样优先级**输入,
  决定下一轮优先训谁,不决定留不留。archive 只增不减,`Active tasks` 稳定 100 是采样窗口非存量。
- auction 是"多生成、狠筛选、留精华"(留 10 < baseline 12);coop(方案A)= "多生成、轻筛选"(留 18,砍 ~25%)。

### 8.2 v5y 的筛选变种(三个,方案A 为当前默认)

v5y 每轮 2 proposer 各对 12 个 frontier parent 造关 = **24 候选**。选择标准有三个变种,**变种0和A都要跑**
(对照,看筛选是否提升性能):

| 变种 | 保留数 | bid 组成 | 语义 | 状态 |
|---|---|---|---|---|
| **(0) 不筛** | 24 全留 | — | 24 全保留(每轮 2×baseline) | ✅ **在跑**(job 3639732,`auction_c_v5y.yaml`,`coop_select_k=null`);作为"筛选是否有用"的对照,不杀 |
| **(A) Ambition + Learnability** | **18**(24→18) | `w_amb=1, w_lrn=1, w_cov=0, w_end=0` | AmbitionGain 提供**深度压力**(gap×depth,选针对弱点+深的);Learnability(p(1−p))保**可学性**(筛掉学不会/太简单的,留学习区) | ✅ **待提交**(`auction_c_v5yA.yaml`,`coop_select_k=18`) |
| **(B) 纯 Learnability** | 可配 | `w_lrn=1, 其余=0` | 只保可学性,方向完全交给 modeler+proposer分工;选择层不管深度 | 📌 **备手**(记录留档,暂不跑) |

**开关实现**:`coop_select_k`(config)= null → 不筛(变种0);= K → 从 24 选 top-K(方案A K=18)。
纯叠加,变种0 行为不受新代码影响。变种间用**独立目录 + 独立 run id + 独立输出目录**隔离(§8.3)。

**为什么去掉 Coverage 和 Endorsement**:
- **Coverage 去掉**:广度已由**两 proposer 的分工提示**在生成层保证(后手补前手未覆盖的 TYPE),
  选择层再压 Coverage 会与分工重复、甚至互相打架。
- **Endorsement 去掉**:cross-rating 是 auction 的竞争性互评,和 v5 合作精神冲突;v4 已实测它是
  "trust-everyone"信号、与质量负相关([[auction-voice-logging-added]] 系列),弃用。

**为什么方案A 保留 AmbitionGain(不是冗余)**:
核对 modeler 提示词确认——**没有任何组件"保证"课程往深走**。modeler 的 DEPTH 只是三选一里的中立选项
(和 BREADTH/CONSOLIDATE 平权),且 DEPTH 有"前置须 MASTERED/RISING"门槛,学生弱时反而不推 DEPTH;
HARD CONSTRAINT 还禁止 modeler 用 tier 数字推理。故**唯一系统性的深度驱动力只有 proposer 的 ambitious
persona**。纯 Learnability 有"畏难"倾向(偏好 p≈0.5 的浅关),若与"modeler 不推 DEPTH"叠加,课程可能
卡在舒适区(DiCode 原始难题,[[difficulty-bid-drowned-equals-learnability]])。**AmbitionGain 在选择层
主动抬高深关的分,补上这个深度压力**——它与 Learnability 正交(Ambition 管"值不值得往深推",
Learnability 管"现在能不能学会")。

**方案B(纯 Learnability)作为备手的条件**:若相信 modeler 生成侧的方向诊断足够强,可切纯 Learnability
让职责链更干净(modeler 定方向 / proposer 分工给广度 / Learnability 保可学);风险是深度只剩 proposer
persona 单点支撑,须盯 tier 推进速度,若卡舒适区则回去加强 modeler/proposer 的深度倾向,而非在选择层堆 bid。

### 8.3 实现要点
- 复用 `auction/selectors.py` 的 `GreedyTopKSelector` + `SelectionContext`,只把权重设为
  `w_cov=0, w_end=0, w_amb=1, w_lrn=1`(方案A);`target_gap` 来自 global_agent_profile,
  `parent_learnability` = 父关官方 p(1−p)(训练副产品,[[dicode-learnability-p-is-training-byproduct]])。
- `reachable_ceiling=None`(v4 已关 ability gate,不复活)。
- 在 `evolve_mastered_coop` 收齐 24 候选后、注入前插入 select(k=12);对照X(1 proposer,12候选)
  k=12 即全保留,不受影响。
