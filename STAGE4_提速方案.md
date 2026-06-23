# Stage 4 generator-on 提速方案（纯工程优先，不碰方法/不影响 SOTA）

> ## ⛔ 2026-06-22 晚 终极诊断更正：真因不是编译，是 io_callback × Dijkstra
>
> **本文档下方第 0 节"编译爆炸"诊断结论已被证伪作废**（保留作诊断弯路记录）。真因经 srun 单变量对照实测坐实：
>
> **根因链**：student `train_step`(jaxnav_sfl.py:858) 每个 update 调 **ordered** `io_callback(callback)` → callback log 的 `env-metrics` = `jax.vmap(env.get_env_metrics)(start_state)` → `get_env_metrics`(jaxnav_env.py:658) 对每个环境跑 `dikstra_path`(grid_map.py:411 `jax.lax.while_loop` 迭代到开集为空，**步数数据依赖地图结构、无提前终止**) → **generator 的 PCGRL 迷宫地图让 Dijkstra while_loop 迭代爆炸**(海选随机地图则快) → ordered io_callback 把它串到每 update 同步点 → 卡死。
>
> **决定性对照**(SFL_DIAG_NO_IOCB)：剥离 io_callback → student-scan RUN **70.6s 跑通**；带 io_callback 同配置 → **卡死 15min+**。generator-OFF(海选地图)110s 正常(同跑 io_callback 但地图不让 Dijkstra 爆炸)。
>
> **已逐一实测排除**：①instances 致编译爆炸(train_and_eval 编译仅 41s，gen/海选 jaxpr eqn 数 20177≈20185) ②A0 缓存(开 64s vs 关 57s 无差异) ③wandb online(disabled 也卡)。
>
> **为何历史误判成"编译"**：所有"诊断跑通"的实验要么只 `.compile()` 不 run、要么 run 不含 io_callback 的 gen_phase——**没有一个真正 RUN 过含 io_callback 的 student scan**。判死活铁律：主线程 R/CPU100% + GPU mem-util 0% + 满频 + 无慢算子告警 = host ordered callback 自旋等 device 的 compute-bound while_loop，**不是编译**(编译时 GPU≈0)。
>
> 详见 [[stage4-stall-root-cause-dijkstra-iocallback]]。**修法见下方第 0.5 节**(约束:不影响刷 SOTA)。

---

## 0.6 Oscar 资源追问应答（2026-06-22 收到 CCV 邮件，备查）

> CCV Systems 于 2026-06-22 发邮件给 PI(nopi/ccv-nopi 组)，称用户 jzhu223 过去一周「allocated 15 GPU-hours but not utilized」，点名 jobid **`3363545_6` 和 `3363547_0`**（注明"may not be exhaustive"）。这是发给 PI 的全组资源效率提醒，非针对个人警告。以下是 `jobstats` + 日志 + 代码三方核实的事实，若被追问可直接引用。

**这两个 job 当时在干什么**：SFL auction ablation 的 array job（job 名 `sfljnAbl`），各自卡死在 **Dijkstra × io_callback 黑洞**（即本文档 §0.5 真因），空占 GPU 显存数小时无推进，最终被取消。

| 事实 | `3363545_6`（内部 3365430） | `3363547_0`（内部 3370954） |
|---|---|---|
| 配置 | seed=6, `auction_lambda=inf`, group=auction-abl-linf | seed=0, `auction_lambda=3.0`, group=auction-abl-l3_0 |
| 同 array 兄弟 task 正常时长 | ~50min | ~50min |
| 实际 | 卡 **11:35:40** 后 CANCELLED | 卡 **03:40:26** 后 CANCELLED |
| GPU 利用率 | **0%** | **0%** |
| GPU 显存 | **34.4GB / 45GB（76%）一直占着** | 同左 |
| CPU 时间（整段 run） | 仅 **31 秒**（efficiency 0%） | 仅 **14 秒** |
| 日志停在 | `JAX devices: [CudaDevice(id=0)]` → `auction+cenie import OK` **之后再无输出** | 同左 |

**为何 GPU 0% 利用却占着 34GB 显存**：JAX 已初始化并把模型/buffer 搬上卡（→ 显存占用），但主循环第一步即撞进 ordered `io_callback`，把 generator PCGRL 迷宫地图上的 Dijkstra `while_loop`（数据依赖、无提前终止、迭代爆炸）串到每个 update 同步点 → 进程 hang，GPU kernel 不再推进 → 利用率恒 0%。**不是代码退回 CPU 跑**（那样 CPU 利用率会很高，实测 CPU 整段只动十几秒），是死锁式卡死。

**重要佐证（说明非系统性浪费）**：同两个 array（3363545/3363546/3363547）中十余个兄弟 task 全部 `COMPLETED` 正常跑完（各 ~50min–1h22min），仅这两个个例触发卡死路径。

**已修复**：致卡根因（每-update 的 env-metrics/Dijkstra）已于 jaxnav_sfl.py:823/856 移除并注释存档，env-metrics 仅为 wandb 训练曲线日志、不参与课程/训练/SOTA 评估，移除 0 数值影响。后续重跑不再触发。另已规划看门狗（§4 第 5 条）使"慢/卡但未死"也能告警，避免再次长时间空占。

> 与"墙钟（walltime）刻意留长"无关：邮件查的是 GPU utilization（运行期实际使用率），walltime 只是申请的运行窗口，两者独立。

---

## 0.5 修法（真因对症，约束=不影响 SOTA）

> 待定，2026-06-22 晚讨论中。候选方向(均不碰 generator 课程/方法)：
> - io_callback 降频(每 N update 才 log，而非每 update)
> - env-metrics(Dijkstra)移出每-update 路径(只在 eval_step 边界算一次，或彻底不在训练循环 log)
> - dikstra_path 加提前终止(goal 到达即停，源码 TODO 已标)/封顶迭代步数
> - io_callback 改 unordered
>
> **关键约束**：env-metrics 只是 wandb 训练曲线日志，**不参与 generator 课程/student 训练/SOTA 评估** → 改它不动方法。需逐条确认哪个改动 0 数值影响。

---

## 0. 诊断结论（确证，不是猜）

| 段(n=64) | 首次含编译 | 纯运行复跑 | 结论 |
|---|---|---|---|
| gen_rollout_batch (363步scan) | 4.08s | **0.073s** | 运行极快 |
| compute_terminal_reward (student评估64lv×1000步) | 3.05s | **0.176s** | 运行极快 |
| gen_train_iter 整体(单generator) | **45.75s** | **0.733s** | **编译=运行的62×** |

- 整训练 40 个 eval_freq × N=3 generator 的**纯运行总开销 ≈ 1.5 分钟**。
- 真实 8h run 卡在 `[generator] GENERATOR_INJECTION on` 后 8h 无输出、`.err` 无重编译痕迹 → **卡在单次巨型 jit 首次编译没编完**。
- **元凶**：`get_generator_set` 用 Python `for g in range(N=3)` 展开 3 个异质 generator(estimator 各异，无法 scan 折叠)，每个完整 gen_train_iter(363步scan+student评估+PPO4epoch)，再叠 `lax.scan(train_step, EVAL_FREQ=50)` student PPO + auction + CENIE，全塞进一个 `train_and_eval_step` jit → 编译图爆炸到小时级。GPU 60-70% = 编译期 XLA autotune 搜索(batch=23232 conv 那个 8.6s 慢算子 × 海量算子)。

**rollout 的 batch 缩放**(n64→256→1024 复跑 0.073→0.198→0.662s，亚线性 2.48×/9×) 说明小 batch 也有空转、拉 batch 有空间，但**不是 8h 黑洞主因**——主因是编译。

---

## 1. A 类：纯工程，零数值影响 / 仅浮点重排（不碰方法，可放心用）⭐ 优先

| # | 手段 | 治什么 | 预期 | 改哪 | 验证 |
|---|---|---|---|---|---|
| **A0** | **持久化编译+autotune 缓存** `jax.config.update("jax_compilation_cache_dir", ...)` + `jax_persistent_cache_min_compile_time_secs=0` | 10-seed sweep 第 2 个 seed 起跳过重编译；autotune 搜索(那个 8.6s×N)只付一次 | 第 2+ seed 省掉小时级编译 | jaxnav_sfl.py 启动处加几行 | 看第 2 个 seed 启动是否秒级进训练 |
| **A1** | **N=3 generator 共享编译**：把 `for g in range(N)` 的 3 个异质 generator 折叠——estimator 分支用 `jax.lax.switch`/数据驱动而非 Python 分支，使 3 个 generator 编译同一份子图(只是 estimator_id 作运行时参数) | 编译图从"3 份独立完整 generator 训练子图"缩成"1 份 ×3 调用" | 编译时间 ~3× 缩减(可能是从编不完到能编完的关键) | `pcgrl_generator.py get_generator_set` | 编译时间对比；输出数值不变 |
| **A2** | **flatten nested vmap**：`gen_ppo_loss` 把 `(64 levels × 363步)` 摊平成单 batch 过 conv(注意 norm_adv/mean 仍按 level 分组) | 消掉 batch=23232 怪形状，cuDNN 不再为它跑慢 autotune | 2-5×(编译+运行) | `pcgrl_generator.py gen_ppo_loss` | 单测断言输出与原版数值一致(rtol 1e-5) |
| **A3** | **jax.checkpoint 套 363步 scan body** | 省显存，解锁更大 batch | 间接 | rollout/loss 的 scan | 梯度数值等价(FP重排级) |
| **A4** | **donate_argnums** 原地复用 PPO buffer/optimizer | 省显存 | 省显存 | jit 调用处 | 功能等价 |
| **A5** | **stdout flush + 编译进度打印** | "在编译"不被误判为卡死 | 0提速,去焦虑 | generator 注入前后加 flush 打印 | — |

**A0+A1 是核心**：黑洞是编译，这俩直接砍编译。A1(N=3 共享编译)很可能是"从 8h 编不完 → 几分钟编完"的决定性一招。

---

## 2. ⚠️ B 类：碰训练动态（要当方法改动，需消融验证，不可借"提速"名义偷改）

| 手段 | 改了什么 | 何时考虑 |
|---|---|---|
| bf16 混合精度 | 改数值、影响 PPO 稳定性("always bf16"被 deep-research 0-3 否决) | A 类不够时，且需 fp32 对照验证 |
| 降低 generator 重训频率(不每50update) | 改课程耦合/staleness | A 类不够时，作方法实验 |
| 换 NCA/wide 一次性生成替代 363步自回归 | 改生成分布("wide消串行"被0-3否决,只NCA可信) | 仅当 profile 证明序列scan是主因——**但已证不是(编译才是)**，故不需要 |
| generator/student 异步(Cleanba式) | 改博弈动态 | 不建议 |

---

## 3. ❌ 不可复用的幻觉
- 15×/120×(PCGRL-jax/minimax)那些大数字来自 **CPU→GPU 迁移**——jaxnav 早在 GPU 上，**拿不到**。
- 拉大并行 env(SFL 25000 env)主要利好 student 主循环吞吐，对 generator 编译瓶颈帮助有限。

---

## 4. 执行顺序建议
1. **先 A0**(改动最小、风险零)：加编译缓存，单看能否让重复 run 免编译。
2. **再 A1**(核心)：N=3 generator 共享编译，砍掉编译图爆炸。单测验数值不变。
3. **A2**(flatten vmap)：进一步砍编译+运行，消 cuDNN 怪形状。单测验数值不变。
4. 跑一个单 seed 短验证：确认编译时间从小时级降到分钟级、纯运行对齐 SFL ~1h 量级。
5. 补**看门狗**(STAGE4 §4 一直没实现的)：generator 注入阶段加超时打印 + 旁路监控 .out mtime，让"慢/卡"也告警(当前 mail-type=END,FAIL 不报"慢但没死")。
6. 全绿后再重提 Stage4 阶段1 的 35 run。

> 所有 A 类改动**不改变 generator 生成什么 level、不改 auction 混合、不改课程分布** → 不影响 idea 方法的 SOTA 能力。改完必须单测断言输出数值与原版一致。
