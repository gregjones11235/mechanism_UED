# baseline 方法原理分析 & non-baseline（auction/coop）对比

> 记录 2026-07-03。本文档固化"DiCode baseline 到底靠哪些机制超旧 UED""A/B/C/D 四档的真实作用"
> "12 关的来源""我们的 auction/coop idea 覆盖/绕过了 baseline 哪些机制"这几个问题的**代码级**结论。
> 所有结论均带 PDF / dicode_src 代码行号锚点，可复核。区分**论文文本口径**与**实际代码口径**（两者不完全一致，见 §1）。
>
> 相关记忆/文档：[[v1-step-1900-pollution-early-lead-real]]、`最早的non-baseline run v1.md`、
> `v5_design.md`、[[dicode-repro-injection-point]]、[[dicode-step-semantics-and-official-eval]]。

---

## 0. 一句话总纲

- **baseline（= DiCode 主方法）的增益来自 8 套叠加机制**（§2），核心是"有向图 archive + learnability 选父关 + 血缘多样性约束 + 20% target 锚定"。
- **A/B/C/D 四档 ≠ 淘汰父关**；它是**从 archive buffer 池筛"哪些关够格当这一轮父关"的准入标签**（§3，即论文公式(6)的代码落地 `_select_tasks_frontier`）。
- **"12" = `num_generation_tasks(10) + additional_num_parents(2)`**，是"每回合选几个父关去演化"的名额，**恒为 12**，但**去重后真实父关数常 < 12**（不足时 `_replicate_to_fill` 复制填满，§1）。
- **我们的 auction/coop idea 对 baseline 8 机制是"纯叠加"**：①–⑧ 全部复用，只在"给定 12 父关 → 生成新子关"这一步内部插入"多 proposer + bid 筛选"。**唯一被显式弃用的是 v4 的 ability-gate（非 baseline 机制）**（§4）。
- **v1 灾难性遗忘的精确根因**：baseline 机制全部健在，但我们在"生成候选 → 保留"之间插的 **AmbitionGain 加权筛选**，把机制②③辛苦守住的 **tier-2 供给在最后一步筛掉了**（§5）。

---

## 1. "12" 到底指什么？—— 两套口径必须分清

### ⚠️ 论文文本口径 ≠ 实际代码口径

| 出处 | "关卡数"指什么 | 单位 |
|---|---|---|
| **论文 Table 4**（`DiCode.pdf`，pdftotext line 1209-1223） | RL 训练时 env **worker 的比例分配**：0.20 target / replay / new；unique replayed=15/10，unique new=0/5 | env worker 比例 |
| **实际代码** | 每个 evolution session **选多少个"父关"去演化生成新关** = `k` | 父关名额 |

**"每回合喂给 student 的关卡数"这个问题，指向的是代码口径的 `k`，不是论文 Table 4 的 worker 比例。**

### 12 的代码来源链（三处，已锁定）

1. **配置**：`dicode_src/conf/dicode_manager/default.yaml:53-54`
   ```yaml
   num_generation_tasks: 10
   additional_num_parents: 2
   ```
2. **求和**：`dicode_src/src/dicode/evolution_efficient.py:125-130`
   ```python
   num_to_evolve = (
       config.dicode_manager.num_generation_tasks + config.dicode_manager.additional_num_parents
   )  # = 10 + 2 = 12
   tasks_to_evolve = select_tasks_for_evolution(config, archive, session_idx-1, num_to_evolve)
   ```
3. **auction 配置明写对标**：`dicode_src/conf/gen_manager/auction_c_v5yB.yaml:3-19`
   > "per-round retained count = baseline's 12 (strict equal-yield comparison)"
   > `coop_select_k: 12  # = baseline yield`

### "是不是总是 12 关？"—— 确定答案

**父关名额恒为 12，但去重后的真实父关数经常 < 12。**

- `_select_tasks_frontier`（`selection.py:183-247`）先按公式(6)筛出 frontier tasks（早期/稀疏时可能只有 3、5 个）。
- 若不足 12，`_replicate_to_fill`（`selection.py:277-298`）**复制并洗牌填满到 12**。
- ⟹ **交给 evolve 的列表长度恒 = 12**；**不重复真实父关数 = min(len(frontier), 12)**；不足时 12 个名额里有重复 → 12 个新关中有多个源自同一父关。

### auction/coop 变种下每回合保留关数（可变）

12 父关 × N proposer → 候选池，再由 `coop_select_k` 筛回：

| 变种 | proposer 数 | 候选池 | coop_select_k | 每回合保留关数 |
|---|---|---|---|---|
| baseline | 1 | 12 | — | **12** |
| v5y（不筛，已 scancel） | 2 | 24 | null（全留） | 24 |
| v5yA（方案A 24→18） | 2 | 24 | 18 | 18 |
| v5yB（方案A 24→12，strict 对标） | 2 | 24 | 12 | 12 |

（v5yA/v5yB job 映射见 [[v5-runs-and-selection-variants-2026-07-03]]。）

---

## 2. baseline（DiCode 主方法）的 8 套增益机制

除"单 FM 235B（Qwen3-235B-A22B-Thinking，`DiCode.pdf` line 1230）+ code 生成"外，相对 PLR/SFL/DR 的增益来自以下叠加机制：

| # | 机制 | 论文锚点（DiCode.pdf line） | 代码锚点 |
|---|---|---|---|
| ① | **有向图 Archive + 血缘（parent→offspring edges）**，节点存 code+metadata+SR | 187-190 | `TaskArchive`（gen_manager.py 内） |
| ② | **learnability 父关选择 + 血缘多样性约束（公式6）**：`A_cand={ℓ∈A \| S(ℓ)∈{A,B} ∧ ∃c∈C(ℓ),S(c)=D}` | 200-217 | `_select_tasks_frontier`（selection.py:183） |
| ③ | **learnability 直接作分**（弃 PVL/MaxMC），`f(ℓ)=p(1−p)` | 149-159 | `score_function:"learnability"`（default.yaml:20） |
| ④ | **分层 batch（三源混合）+ 固定 20% target 锚定**（防分布漂移） | 260-267 | `original_task_proportion:0.2`（default.yaml:51） |
| ⑤ | **自适应 bonus scaling** `Bt=max(d,2·R_{t-1})` + `I_init` mask（已掌握成就不给奖） | 219-229 | `completion_bonus_*` / `dynamic_bonus_k`（default.yaml:40-47） |
| ⑥ | **每 v=2 迭代注入新关 + 异步生成**（保 policy 稳定 & GPU 利用率） | 269-281 | `evolution_interval:2`（default.yaml:6） |
| ⑦ | **冗余生成 + compilation check（不做 self-correction）** | 249-256 | `generate_code_only` / `evolve_and_validate_tasks`（evolution_efficient.py:149） |
| ⑧ | **4 个 seed tasks 冷启动**（survival/combat/crafting/gathering） | 804-806 | `A.2 Seed Tasks` |

**消融证据**（`DiCode.pdf` line 6310-6316，Table 7）：把闭环反馈拿掉（DiCode-OL，生成不看 agent 当前能力 + 父关）→ **48.33 → 40.91**。闭环 = "条件在 parent + agent perf 上生成"，是命门。

---

## 3. A/B/C/D 四档的真实作用（= 从 buffer 池筛父关）

### 定义（论文口径，`DiCode.pdf` line 196-198）
status mapping `S(ℓ)`，按 agent 最近成功率 SR 离散：
- **A**: SR ≥ 0.75（已掌握）
- **B**: SR ∈ [0.50, 0.75)
- **C**: SR ∈ [0.25, 0.50)
- **D**: SR < 0.25（还打不动）

### 代码落地（`selection.py:160-178`）—— 这就是"从 buffer 池筛父关"
`parent_selection: strict`（default.yaml:57）时：
```python
_select_tasks_frontier(
    parent_statuses={"A", "B"},        # ← 只有 A/B 档才够格当父关
    success_statuses={"A", "B", "C"},  # ← 子关到 A/B/C 就算"毕业"
    max_children=5,                    # ← 血缘多样性：单父关最多 5 子
    sort_by_score=True,
)
```

三重过滤（`_select_tasks_frontier`，selection.py:204-247）：
1. **状态门**：`status ∈ {A,B}` 才够格当父关（agent SR≥0.5）。C/D 不当父关。
2. **分支上限**：已有 ≥5 子关的父关跳过（= 论文的血缘多样性，防一条线无限深挖）。
3. **毕业门**：某父关**已有** A/B/C 档子关 → 说明这条线推进过 → 不再选它（`has_viable_child` → skip）。

### 结论
- **A/B/C/D ≠ 淘汰/删除父关**；它是**准入标签**：决定"archive 里哪些关这一轮够格进 12 父关名额"。
- 用户判断"从 buffer 池中的筛选机制"是**准确**的。
- **坑**：`lenient` 模式（selection.py:170-178）把 `parent_statuses` 放宽到 `{A,B,C,D}` 全放行，那时 A/B/C/D 退化为只影响排序不影响准入。**当前 baseline 用 `strict`**，A/B 门真生效。
- 另注：`_get_valid_parent_statuses`（gen_manager.py:1455）返回**所有**状态，但那是用于选 **few-shot example**，**不是**父关准入——别混淆这两个函数。

---

## 4. 我们的 auction/coop idea 覆盖/弃用了 baseline 哪些机制？

**逐条核对 `evolve_mastered_auction`（gen_manager.py:829）/ `evolve_mastered_coop`（gen_manager.py:1054）对 8 机制的触碰情况。**

| baseline 机制 | 在 idea 里 | 证据 |
|---|---|---|
| ① 有向图 Archive + 血缘 | ✅ **完全复用**（共用 `TaskArchive`，auction 只产 `_organize_data` 结构后走同一注入） | gen_manager.py:1176 返回同结构 |
| ② frontier 父关筛选（公式6） | ✅ **完全复用**（`select_tasks_for_evolution` 在 auction **之前**，与 baseline 同函数选出 12 父关） | evolution_efficient.py:128 |
| ③ learnability 选父关 + score_function | ✅ **复用**（archive priority_score/learnability 不变） | default.yaml:20 |
| ④ 20% target + 三源 replay 混合 | ✅ **完全复用**（训练侧不碰） | `original_task_proportion:0.2` 未被 auction 引用 |
| ⑤ 自适应 bonus scaling / I_init | ✅ **完全复用**（grep 确认 auction/coop 无一处触碰 bonus/reward） | grep `completion_bonus\|dynamic_bonus` in auction 路径 = 空 |
| ⑥ 每 v=2 注入 + 异步 | ✅ **复用** | `evolution_interval:2` 不变 |
| ⑦ 冗余生成 + compile check | ✅ **复用**（同一 `generate_code_only`/`evolve_and_validate`） | evolution_efficient.py:172-184 |
| ⑧ 4 seed 冷启动 | ✅ **复用** | seed 不变 |

**只有两处被改/被绕过，都在"给定 12 父关 → 生成新子关"这一步内部：**

**改动 A（核心叠加点）：单 FM → 多 proposer 合作补位 + 筛选**
- baseline：每父关调 1 次 LLM 生成 1 关。
- coop（gen_manager.py:1109-1156）：每父关 N proposer **轮流**生成，第二个能看到第一个造了啥（`PEER_ALREADY_MADE`）→ 合作补位；再由 `_coop_select`（gen_manager.py:1180）用 `w_amb·AmbitionGain + w_lrn·Learnability` 从候选池筛回 k 个。
- **不覆盖任何 baseline 机制**，是插在"父关已选定"↔"子关送编译"之间的一层。

**改动 B（唯一真正被绕过的机制 = ability-gate，且它非 baseline 机制）：**
- gen_manager.py:1212：`reachable_ceiling=None  # v4 already retired the ability gate; do not revive it`
- 这是**我们 v4 自己加的**"可达性天花板"门，v5 已弃用。**不是 baseline 机制**，不影响 baseline。

---

## 5. v1 灾难性遗忘的精确根因（收口，证据完整）

**关键**：我们的 idea **没有覆盖/削弱 baseline 机制②③④⑤，它们全在**。那 v1 为何仍遗忘？

代码给出精确答案 —— 遗忘**不是**因为 auction 破坏机制②③，而是因为 **bid 排序架在机制②③选出的 12 父关之上，改变了"从这 12 父关生成的候选里保留哪些"的分布**：

1. 机制②③照常选出 12 个合法 frontier 父关（含 tier-2）✅
2. 但候选经 `w_amb=1.0` 的 **AmbitionGain（gap×depth 深度轴）**加权筛选后，**tier-3 候选被排到前面，tier-2 铁器链子关被筛掉**（gen_manager.py:1208-1211）。
3. baseline 没有这层筛选，12 父关各留 1 关，**tier-2 供给自然守住**。

**精确机制链**：
> baseline 机制全部健在 → 但我们在"生成候选 → 保留"之间插入的 **AmbitionGain 加权筛选**，把机制②③辛苦守住的 **tier-2 供给在最后一步筛掉了** → tier-2 铁器链关 replay 变稀 → student 深关大量失败进不到"打铁"那步 → 该技能正梯度消失 → RL 权重被深关梯度覆盖 → **make_iron 斜率转负（灾难性遗忘）** → 被 baseline 反超。

**对照 v1 现象记录**（`最早的non-baseline run v1.md` §4）：
- graphml：C 档造关目标 tier 中期漂移（s1-10 纯 t2 72% → s26-45 t3=72%）✅ 吻合
- make_iron_pickaxe 斜率 −0.68、collect_iron −0.12（倒退）✅ 吻合

**`coop_w_lrn`（Learnability 权重）就是设计来对冲这个的**：v5yB 把 `coop_w_amb=1.0`/`coop_w_lrn=1.0` 都设 1.0，正是在调这个平衡（让 Learnability 把过深/过易的候选拉回可学 band）。v1（实为 v4 auction）当时 AmbitionGain 压过 Learnability，是漂移主因。

---

## 6. 待验证 / 未决

- **v5 早期偏弱风险**：v5 用 2×同质 `ambitious_coop` 替换 v1 的 breadth/ambitious/feasible 三元异质，同时丢了 breadth 广度(28%)+feasible 稳度(20%)，早期"广铺快刷简单成就"供给变薄（`最早的non-baseline run v1.md` §3b）。**待观察**两 r2 run 到真实步 1000-2000 是否复现 v1 早期领先。
- **tension 未解**：早期领先（ambitious 放开冲）vs 中期不遗忘（压回 learnable band）可能不能同时靠"压回 band"拿到。modeler 若只诊断不强制供给 tier-2，冲突点仍在。
- **可能的冗余**：Learnability 在公式(7)选父关用了一次，`_coop_select` 的 bid 里又用一次（`w_lrn`）——是否双重计数需审计权重语义。

---

## 附：本次分析读过的关键文件
- `DiCode.pdf`（pdftotext -layout，见 scratchpad/dicode.txt）：method §3.2-3.3、Table 4、Table 7、A.2
- `dicode_src/conf/dicode_manager/default.yaml`：baseline buffer/generation 管理
- `dicode_src/conf/gen_manager/auction_c_v5yB.yaml` 等变种
- `dicode_src/src/dicode/selection.py`：`select_tasks_for_evolution` / `_select_tasks_frontier` / `_replicate_to_fill`
- `dicode_src/src/dicode/evolution_efficient.py`：12 的求和 + 派发
- `dicode_src/src/dicode/dreaming/gen_manager.py`：`evolve_mastered` / `_auction` / `_coop` / `_coop_select`
