# Threshold Probe 结果与瓶颈定位：脚手架泄漏（Scaffolding Leakage）

> 2026-07-09。probe 臂 = +A+B + `mastery_threshold=0.2`（wandb `oun2yfm6`，改名 `AB_threshold02_probe`）。
> 对照 = u1gjqror（同配置，threshold 0.60），同 seed=1 同环境单 flag 差。
> 数据：`experiments_mason/eval/eval_PROBET02_seed0.json`、`run_probe_t02.log`、probe task_graph.graphml。
> **本文档结论：机制臂 ~30 平台的成因最终定位为"生成任务的脚手架泄漏"——有 14B 自写代码注释级的实锤。**

---

## 1. 五臂终表（官方口径，mean_return，seed 0 同批 1024 世界）

| update (steps) | baseline_ext | +A_ext | +A+B (t=0.6) | **probe (t=0.2)** |
|---|---|---|---|---|
| 300 (39M) | 12.75 | 12.20 | 13.46 | 13.19 |
| 600 (79M) | 17.24 | 18.32 | 17.81 | 16.70 |
| 900 (118M) | 10.84 ▼ | 18.15 | 18.54 | 18.75 |
| 1200 (157M) | 18.27 | 20.00 | 20.25 | **23.19** |
| 1500 (197M) | 18.76 | 12.94 ▼ | 29.16 | 26.38 |
| 1800 (236M) | 18.74 | 21.34 | 28.59 | 28.71 |
| 2100 (275M) | 17.66 | 28.93 | 32.46 | 29.61 |
| 2400 (315M) | 17.88 | 31.07 | 29.81 | **29.06** |

（一代 157M 三臂另存 v1；五臂身份：mc75k0nx / z8jygtyw / u1gjqror / oun2yfm6 + 32v02vi9/85qid2ev。）

## 2. Probe 过程指标（预注册判据逐条对账）

| 判据 | 结果 |
|---|---|
| frontier 解锁 tier 3？ | ✅ session 3 首现；session 1-4 短暂 2↔3 振荡后 **22/25 session 稳定 tier 3** |
| preflight 拒绝率变化？ | **0/76 全收**（0.6 臂为 2/78）——见 §4，全收本身成为证据 |
| 平台抬升？ | ❌ 终点 29.06，与 0.6 臂 29.81 / +A 31.07 同噪声带——**三种机制配置收敛 29-31** |
| tier-3 技能非零？ | ❌ 调度 targets（defeat_lizard / learn_fireball / enter_vault / gnome_warrior）**20 个定向 session 后仍全 0**；仅 make_diamond_sword 0.6 |
| 副作用（tier-2 回撤）？ | 无实证：probe iron 2.7/2.6 vs 正当对照 ARMAB 4.6/2.8（同带）；tier-2.5 照常巩固至 61-66；钻石资源非零（7.4/5.9/4.7）。+A_ext 的 iron 28.9/39.2 为无闸臂独立高点（n=1 方差带）——机会成本仅 hint |
| 中段效应 | ✅ 1200 处 23.19 vs 两个 0.6 臂 20.0/20.3——早解锁买到中段加速 |

其他：零故障；轨迹为五臂最平滑之一（300→2100 单调，末点 -0.5）；振荡期为事实上的 tier-2/3 混合课程（方案二的免费近似）。

## 3. ★ 瓶颈定位：脚手架泄漏（实锤）

**证据链**（每环第一手）：
1. 调度器忠实指向 tier 3（22/25 session，targets 为真身枚举名，含 `defeat_orc_solider` 源码拼法）；
2. 14B 照 target 生成 → **生成代码内建脚手架**。probe task_graph 抽查实录（14B 自写注释原文）：

```python
# task_19 "MAKE_DIAMOND_SWORD" — generate_world():
# --- ADDED SCAFFOLDING ---
# 1. Give a stone pickaxe and 50 coal for crafting
builder.set_player_inventory({"pickaxe": 1, "coal": 50})
# 2. Place lizards as combat practice
builder.add_mobs_randomly_near(..., min_dist=4, max_dist=8, ...)
# --- END SCAFFOLDING ---
```
   并且 `completed_achievements` 预标记前置链（task_20 一次预标 **14 个**前置成就；task_17 预标 MAKE_WOOD_SWORD）——前置依赖被代码级跳过；
3. 任务因此可解 → **preflight 全收**（sr≥0.05 恰恰由脚手架制造）；
4. policy 学会"持现成资源做最后一步"→ 裸 held-out 需从空手走完整链 → **技能不迁移** → 平台。

**两条正交质量轴**（本轮最重要的概念产出）：
- **可解性（learnability）**：当前 policy 能否推进——编译校验、C-0 lint、preflight 全部只覆盖此轴；
- **忠实性（fidelity）**：任务是否真的要求它声称的技能——**现有全部闸门零覆盖**，泄漏即由此穿过。

**跨尺度呼应**：v6 线（235B）文档记录的同一风险（"LLM 可能直接把成品铁镐塞进初始库存"，v6fix7 以代码硬闸防御）——本线在 14B 上以五臂消融独立复现并定位 → **脚手架泄漏是 code-level UED 的尺度无关缺陷**（联合论文 discussion 素材）。

**公允注记**：LLM 的行为符合人类教学直觉（递工具练最后一步）；缺陷在于无任何机制传达"评测是裸的/脚手架须递减"。修复方向因此不是禁止脚手架，而是约束或验证它（§5）。

## 4. Preflight（★B）终版定位

机制审计结论（代码级）：B-1 写入的 `learnability_score` **全管线零读取方**（采样与激活均读 `priority_score`；training.py 的 learnability 分支与 task_utils 的复活逻辑均从 `latest_sr` 现算，与 B 无关且默认休眠）。故本管线中 **+A+B ≈ +A + ~10min/设计session + rng 偏移**。

据此校准：
- "path quality（-0.6 vs -7.1 回撤）" 降级为 **observed but unattributed**（n=1；同 seed 方差 ±1 分；probe 零干预却最平滑，进一步削弱因果归因）；
- 保留的主张：**拒必有理**（2/2 真 too_easy）、**廉价**（~10min）、**保险角色**（为激进 frontier 推进封顶下行——probe 即受保运行，零赔付≠零价值）；
- **白送的改进项**：将 PLR 采样接入 B 实测的 learnability 字段（"只写不读"→"闸+信号源"），config 级改动，Stage-2 相邻工程。

> **注意：以上校准均不改变 B 的机制地位** —— B 仍是可学性轴的唯一闸门，且 §5 的 C-2 动态检测直接复用 B 的 rollout 机器（"裸初始条件下重跑 preflight"）。三闸终局分工：**C-0 保代码真、B 保可学、C-2 保忠实**——各管一条正交质量轴，互补而非替代；C-2 上线后 B 反而更关键（剥除脚手架后的裸任务更难，正需 B 判定可学边界）。

## 5. 移交：★C-2 / 方案二的精确规格

平台成因既定为忠实性缺陷，下一机制的规格随之明确（候选，待排序）：

- **C-2 忠实性闸（fidelity gate）**：候选任务在**裸初始条件**下复验——静态层：AST 检查 `set_player_inventory` / `completed_achievements` 预标 / 近距刷怪等脚手架模式（脚手架有清晰代码签名,14B 还自写注释,检测成本低）；动态层：将任务的 init 替换为裸初始态后重跑 preflight rollout，SR 崩塌幅度即"脚手架依赖度"量化指标；
- **脚手架递减课程（方案二升级版）**：不禁止脚手架，prompt/代码层要求随 session 递减（对齐 v6 线"隔离演练→代码保障"思路，两线可合写）；
- 判据：fidelity gate 上线后，tier-3 定向任务的"裸复验 SR"与 held-out tier-3 技能增长应恢复相关。

## 6. 局限

1. 全部 n=1；probe vs 0.6 臂差异多在 ±1.5 噪声带内。
2. 脚手架抽查 n=3（task_17/19/20，全部命中）——泄漏率的系统统计未做（可对 graphml 全量跑静态检测，~1 小时脚本活）。
3. 振荡期（session 1-4）的混合课程效应与 0.2 阈值本身的效应未分离。
4. "预算不足"假设未完全排除（tier-3 链条长），但脚手架实锤使其从主嫌疑降为次要因素。
