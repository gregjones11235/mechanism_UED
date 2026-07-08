# 三组消融 v1 结果记录（baseline / +A / +A+B，seed=1）

> 起草 2026-07-08。数据 = wandb project `mechanism_UED/Skill_Preflight_UED`。
> 性质 = 第一轮（单 seed）结论，措辞已按"方向可主张、幅度需谨慎"校准。

---

## 1. 三臂身份表（引用/评测/归档一律按 run id）

| arm | wandb run id | run name | 步数 | 环境 |
|---|---|---|---|---|
| baseline（纯 DiCode） | **`32v02vi9`** | singleLLM_baseline | 157,286,400（session 10） | 同一套：A100 / jax 0.6.2 / qwen2.5-coder:14b 49-49全GPU / seed 1 / max_tokens 8192 / MEM_FRACTION 0.75 |
| +A（scheduler） | **`85qid2ev`** | skillgraph_only_A | ~157M（session 10 完整，跑到 11 开头停） | 同上 + `use_scheduler=true`。⚠️ preflight 因旧 B-1 bug 静默失效（日志 7 处 `[Preflight] skipped` 为证），**机制上即 +A 臂**；rng 相对纯净 +A 有轻微偏移 |
| +A+B（完整方法） | **`u1gjqror`** | skillgraph_preflight_AB | ~3e8（session 23，超额部分为额外数据） | 同上 + 两 flag + 修复后 preflight（fix_preflight_hook.py） |

**对比纪律：三臂比较一律在 x=global_env_steps ≤ 157,286,400 区间内做**；+A+B 157M 之后的数据只描述、不与另两臂对比。

## 2. 主结果（≤157M，单 seed）

### 2.1 机制 vs 无机制：分层清晰（本轮最强结论）
深层/中层技能上，**两个机制臂大面积"从零冒头"，baseline 全程死零**：
- +A 末段（%）：open_chest ~22、enter_dungeon ~27、find_bow ~20、fire_bow ~17、eat_snail ~17、drink_potion ~9、defeat_orc_solider ~8、collect_diamond ~3、ruby/sapphire ~2-2.5、make_iron_sword/pickaxe ~0.7-0.85；
- +A+B 同族技能同样冒头（iron 族更早，见 2.3）；
- baseline 上述全部 ≈ 0（个别瞬时凸起后回落）。
- 低层技能（tier0-1）三臂重合饱和 —— 深层收益不以牺牲基础为代价。

**机制因果链**（两臂日志一致）：scheduler 先把 targets 从 tier 1 切到 tier 2（铁器族）→ 生成随之定向 → 深层技能随后冒头。先定向、后生成、再学会。

### 2.2 总量指标：未分层
`evaluation/mean_return` @157M：baseline ~19 / +A 末点 ~22 / +A+B ~18.8。三线大部分区间纠缠（17-19），+A 的 +3 为**末段单点**（其在 100-120M 区间曾落后），单 seed 下属噪声带内，**不主张总量优势**。

### 2.3 +A+B vs +A：效率信号（earlier onset），非幅度信号
- iron 族 onset：+A+B **~110-120M** vs +A **~130M**（提早 10-20M 步）；
- `make_iron_armour`：**仅 +A+B 非零**（~0.12%），+A 与 baseline 均为 0；
- 与 preflight 的干预量自洽：整个 run 拒绝率仅 **2/78 ≈ 3%**（两次均 `too_easy`：sr=0.96、0.86），闸干预小 → 增量体现在"预算花在前沿"的效率上，而非天花板。

### 2.4 preflight 机制验证（★B 首次真实运行，全绿）
11 个设计 session 全部执行、**0 次 ERROR/静默降级**；kept 记录：9/10、7/8、3/3、10/10、5/5、7/7、3/3、9/9、6/6、7/7、10/10。单次开销 486-718s（~10 min/设计session）。零 `unlearnable` 拒绝——14B+scheduler 生成的任务均在可学区（或编译校验已筛掉最差的），本身是个观察。

### 2.5 长预算观察（+A+B 独有，~3e8）
- task graph 4→**136** 节点持续增长；
- **frontier 到最后仍停在 tier 2（铁器族）**：按 mastery 0.6 阈值，铁器 tier 在 3e8/14B 预算下未被掌握，tier2→3 跃迁未发生。tier-3/4（钻石装、附魔、深层怪）三臂全程为 0。
- → 诚实定位：机制能让深层技能**从 0 冒头**，但此预算下**不足以攻克 tier 跃迁**。

## 3. 汇报口径（英文，直接可用）

> "At 157M steps (single seed): both mechanism arms produce broad first-nonzero movement on deep-tier skills where the same-model baseline stays at zero — the mechanism-vs-none separation is clear, with onset following the scheduler's shift to tier-2 targets. Between the arms, the preflight gate adds an *efficiency* effect: deep-tier onset arrives 10-20M steps earlier and iron-armour shows the only nonzero reading, consistent with its low (~3%) but well-targeted rejection rate (both rejections were too-easy tasks, sr 0.86-0.96). Aggregate mean return does not yet separate the three arms at this budget, and tier-3/4 remains at zero for all arms."

## 4. 局限（诚实清单）

1. **单 seed（=1）**，末点排名不稳定；结论均为"方向"级，幅度待 seed 2,3。
2. +A 臂含旧 preflight bug 的 except 痕迹（rng 轻微偏移）；对机制结论无影响，已注记。
3. preflight 拒绝率 3% —— 当前阈值（learnable_low 0.05 / too_easy 0.85）下闸干预弱；"B 增量温和"可能是阈值问题而非机制问题（见 §5.2）。
4. 深层技能读数为低百分比（0.1-27%），稀有事件统计噪声大。
5. 三臂对比口径为本管线的 session 评估；**与 Alec 线同表须先跑共享协议**（见 §5.1）。

## 5. 下一步

1. **共享评测协议**：`eval_checkpoints.py`（1024 held-out 世界 / mean episode return / one-hot 67）评三臂 checkpoint → 与 Alec 线（235B：baseline 44.6 / auction 40.7 中段）进同一张表。先与 Alec 对齐 held-out 世界 seed。
2. **阈值敏感性（第二轮候选）**：too_easy 0.85→0.7 重跑一臂，验证"提高干预率是否放大 B 的效率增量"。
3. **seed 2,3**（算力允许时）：优先补 baseline 与 +A+B。
4. 归档：run_AB.log（85qid2ev）、run_AB2.log（u1gjqror）进 `experiments_mason/logs/`；preflight 修复 commit 进 git（fix_preflight_hook.py，验证记录=u1gjqror 11 sessions / kept 76/78 / 0 errors）。
5. 失败模式清单补充：`BlockType.BAT` 幻觉样本（幻觉不限于 Achievement 枚举）。
