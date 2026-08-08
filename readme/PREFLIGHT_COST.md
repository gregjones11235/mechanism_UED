# Preflight 的时间成本:基线、判据与已知陷阱

> 面向 Henry 关于"preflight 能否更便宜"的工作。总体环境/启动/评测见
> [REPO_GUIDE.md](REPO_GUIDE.md);本文只讲 preflight 这一件事。
> 所有数字来自 2026 夏季实验季的实测(数据总账 08-07 收官版为唯一底本)。

---

## 1. preflight 做什么

每个新生成的关卡在进入训练批之前,用**当前策略**跑一批 rollout 验证,拒掉"太容易"和"无进展"的关卡。

- 代码:`src/dicode/skill_preflight/preflight.py`;调用点在 `experiments/training/run_dicode.py`(见 §4 的日志锚点)。
- 开关:`+skill_preflight.use_preflight=true`。
- 它与调度器是串联关系:调度器**选**目标,preflight **筛**生成结果。两者只开一个,课程形态就变了。

---

## 2. 成本基线(已实测,不必重测)

| 量 | 全栈臂(带 preflight) | 上游原版臂(无 preflight) |
|---|---|---|
| 周期速率 | ~1.2 周期/h | ~3.1 周期/h |
| 2e9 墙钟 | 60.3–66.0 h(三个 seed) | 52.1 h |
| 每轮任务数 | 1899–1900 | 同量级 |
| 每轮 chat 调用 | 4515–4810(≈2.4 次/任务) | ≈2.0 次/任务 |

两条对定位很重要:

1. **成本花在生成侧墙钟,不在 GPU**。preflight 期间 GPU 基本在等,所以优化方向是"少跑/快跑 rollout 验证",不是省显存。
2. **preflight 不是纯开销**。5e8 的组件阶梯里,可裁决的增益几乎全部来自调度器,而调度器选出的目标要经 preflight 过滤才进批;**直接关掉 = 换成另一套课程**,不是"同样效果更便宜"。真问题因此是:**能否用更便宜的代理替代 rollout 验证** —— 例如缓存已验证关卡、缩短验证 horizon、只验证新生成而非重取的关卡、用 embedding 近邻预判等。

---

## 3. 改动怎么判定有效(沿用本季协议,请勿自建口径)

- **对照**:同 seed、单变量、短预算(5e8 级)先跑对照臂;`from scratch`,不要从 ckpt 续跑(见 REPO_GUIDE 的 fork/resume 学习率法条)。
- **成本三件套一起报**:周期速率 / 墙钟 / chat 调用数。只报墙钟会被生成侧抖动骗。
- **效果只认离线评测**:last-10 检查点、1024 held-out、`use_wandb=false`;训练期曲线不作裁决(本季被它误导四次,其中一次是结构性泄漏)。
- **实测噪声地板**(判据请在跑之前写死):

| 量 | 值 |
|---|---|
| 种子差(晚期匹配位) | 1.13 |
| last-10 估计量 | 0.68 |
| 单臂 ckpt 抖动 | 0.37–0.54 |
| 同配置重跑,早窗离线单点 | ±2.31 |
| **带宽随阶段变化** | 技能起飞窗内可达 **~7**;s15 之后收敛到 **0.3–1.1** |

最后一行是关键:**早窗的差值几乎都不可裁决**。要在 5e8 尺度比较,请取窗口末段的多点均值,并把差值放在对应阶段的带宽上读。

---

## 4. 日志锚点(量成本与查失效都靠它们)

```bash
grep -a "\[Preflight\] kept"   /root/run_<arm>.log | tail -5    # kept N/M new tasks,每周期的过滤率
grep -a "\[Preflight\] reject" /root/run_<arm>.log | tail -20   # 逐条拒绝理由
grep -ac "chat/completions \"HTTP/1.1 200" /root/run_<arm>.log  # 教师 chat 调用总数
grep -ac "embeddings \"HTTP/1.1 200"       /root/run_<arm>.log  # 任务数(≈ 每任务一次)
grep -a "SkillGraph" /root/run_<arm>.log | tail -5              # 当期 frontier 组成
```

---

## 5. 陷阱

1. **`+validation=default` 永远不要传。** `validation` 已在基础配置里;缺了它 preflight 会抛异常,而异常被吞成一行:
 ```
 [Preflight] ERROR (kept all, gate inactive!): <e>
 ```
 (出处:`experiments/training/run_dicode.py:317`)门会**静默失效**、全部关卡照单全收,实验白跑。每次开跑后先 grep 这行确认没有。
2. **`[Preflight] WARNING: rollouts returned no scoring data`**(`run_dicode.py:288`)同样意味着这一周期的门没生效,量成本时要把这些周期剔除。
3. **槽位上限 6**(`skill_scheduler.py` 的 `max_target_achievements`):频繁是它而不是阈值在决定谁被教。改 preflight 前先看 frontier 列表,别把槽位效应算到 preflight 头上。
4. **`ep_len` 单独不作证据**:同配置两臂可差 55–82% 而 return 只差 0.3–1.1。
5. 环境相关的坑(ollama 端口/模型目录、`jax[cuda12]`、`craftax==1.4.5`、MooseFS fsync)见 REPO_GUIDE §1 与 §5。

---

## 6. 需要更多背景时

- 五机制记分牌、噪声地板全表、成本章:数据总账(08-07 收官版,Mason 处)。
- 先验图/打乱图消融:REPO_GUIDE §"Prerequisite graph and the shuffle ablation"——与本工作不重叠,但同在 `skill_preflight/` 目录下,改动请勿波及。
