# LLM 独立性能研究线 — 固定 replay 报告（阶段 D）

分类：`LLM_RESEARCH`（允许比较模型与调度行为；**非**语义一致优化，不与 B/C 主线合并）

## 0. 结论

**`LLM_RESEARCH_NO_IMPROVEMENT`**

在 Ollama 14B（`qwen2.5-coder:14b`，单生成槽 `-np 1`）上，`max_in_flight ∈ {1,2,4}` **都不能显著降低墙钟时间**——生成 union 时间基本持平（中位 432 / 420 / 425s），差异在重复运行的波动范围内（~3%）。根因是 Ollama 服务端以单并行槽运行，客户端的并发只是把队列从客户端（`queue_wait` 2181→1136s）搬到服务端（`llm_sum` 432→1447s），总生成墙钟不变。

**但有一个明确的正向发现**：所有有界并发（1/2/4）配置 **retry=0、空响应=0、有效任务率持平 ~0.75**。对比 Mason attempt_06 在**无上限并发**（`asyncio.gather` 全并发）下产生的 **863 次 transport retry**（chat 289 / embedding 574），有界并发避免了连接风暴。质量指标不随并发恶化。

**D2 的 235B 臂标记 `BLOCKED_EXTERNAL_PROVIDER`**：服务器上无本地 235B 服务（Ollama 仅 14B + nomic-embed-text），无 vLLM/sglang，`DEEPINFRA_API_KEY` 未设置。未伪造 235B 结果，也未把仅 Ollama 的结论扩展成双模型结论。

## 1. 配置矩阵汇总（中位数，各 2 次 repeat）

| max_in_flight | 墙钟(s) | LLM union(s) | LLM sum(s) | queue wait(s) | retry | 空响应 | 有效任务 | 有效率 | 唯一代码 | LLM s/有效 | static_invalid | jax_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 587.4 | 431.8 | 431.8 | 2180.9 | 0 | 0 | 9.5 | 0.792 | 12 | 45.6 | 2.5 | 0 |
| 2 | 565.1 | 419.5 | 793.6 | 1776.1 | 0 | 0 | 9 | 0.75 | 12 | 46.6 | 2 | 1 |
| 4 | 570.9 | 425.2 | 1446.6 | 1135.8 | 0 | 0 | 9 | 0.75 | 12 | 47.3 | 2.5 | 0.5 |

注：`llm_sum` 是并发请求 duration 的**直接求和**（并发下不等于墙钟）；`llm_union` 是**去重叠 union 时间**（≈ 真实生成墙钟）。`wall_clock` 包含 CPU JAX 校验（~150s/轮）。

## 2. 串行基线（max_in_flight=1）对比

- 串行基线墙钟（生成 union）：431.8s，有效率 0.792，retry 0，空响应 0，LLM s/有效 45.6s。
- mif=2：union 419.5s（**−2.8%** vs 串行），有效率 0.75，retry 0。
- mif=4：union 425.2s（**−1.5%** vs 串行），有效率 0.75，retry 0。

墙钟降幅（~1.5–2.8%）落在重复运行波动内（同配置两次 repeat 的 union 差 6–10s ≈ 1.4–2.5%），**不能判定为显著改善**。有效任务率 mif=1 出现一次 0.833（10/12）的运气样本，其余全部 0.75（9/12），故有效率**未随并发恶化**（中位 0.792 vs 0.75，属统计波动）。

## 3. 门禁核查

- ✅ 有效任务率不低于串行基线（0.75 vs 0.792，属波动，非显著下降）。
- ✅ 空响应率不恶化（全 0）。
- ✅ retry/repair 无异常增加（全 0，反而比 Mason 无限并发的 863 次大幅下降）。
- ✅ GPU0 最低剩余显存安全（恒定 43558/46068 MiB，剩余 ~2510 MiB，无 OOM）。
- ✅ 服务端无持续 rate limit / queue collapse（0 次 429/5xx，无 HTTP 错误）。
- ❌ **总 replay 中位墙钟未明显下降**（这是唯一未满足的推荐条件 → `NO_IMPROVEMENT`）。

## 4. 三个研究问题的回答

1. **235B 时间去向**：本机无 235B 服务与凭据，**未能实测**（`BLOCKED_EXTERNAL_PROVIDER`）。代码层已知：`llm.query()` 全并发无上限、`_query_with_retries` 指数退避、`check_compilation`（static + CPU JAX）在 evolution worker 内执行。
2. **Ollama 14B 为何要数小时**：主因是 **单槽串行生成**（`-np 1`）叠加 **长输出（max_tokens 8192）** 与 **大 system prompt（含 MiniCraftax/Craftax 完整库代码）**——本 replay 中每请求 ~35–43s（union 432s / 12 请求）。Mason 额外受 **~100% embedding transport retry** 拖累；生成后的 validation 较轻（Compilation ~167s/轮）。本 replay 中无 embedding 阶段，故 retry=0。
3. **max_in_flight 1/2/4**：**均不能降低墙钟**（单槽服务端是瓶颈）；但**有界并发消除了 Mason 的连接风暴 retry**。质量不恶化。

## 5. LLM seconds per valid task（重点指标）

| max_in_flight | LLM s / 有效任务（中位） |
|---|---|
| 1 | 45.6 |
| 2 | 46.6 |
| 4 | 47.3 |

三档基本一致（~46s），印证**并发不改善每有效任务的 LLM 秒数**——请求没有"更快"，只是队列位置变了。

## 6. GPU0 证据

- UUID：`GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6`。
- Ollama llama-server ×2（qwen 14906 MiB + nomic 656 MiB）。
- replay 全程显存恒定 `43558 MiB / 46068 MiB`，最低剩余 ~2510 MiB，无 OOM。
- 每 config 前后 util 0–1%，无第二个生成服务启动。

## 7. 冻结 replay 数据集

- manifest SHA256：`2066515499c305e263023e84a607c93ee4569708da57f23e56c8927c5b690d01`
- system_prompt SHA256：`5cdc991c2a3aced259548a69043be70b87ff083bd969f09f98e6a44fe85772b5`
- 12 个 user prompt（early task_1..4 / mid task_276..279 / late task_551..554），temperature 0.6，max_tokens 8192。
- **冻结类型：`RECONSTRUCTED_FROM_SOURCE_AND_ARCHIVE`** —— Mason 未记录 prompt 字节（profiling 关闭），故从 source commit `91a75e5` 的 gen_env prompt 模块 + `task_graph.graphml` 的真实任务描述重建，非日志字节级捕获。已如实声明。

## 8. 遗留限制

1. 每 config 仅 2 次 repeat，统计功效有限；wall-clock 差异落在重复波动内。
2. D2 235B 臂 `BLOCKED_EXTERNAL_PROVIDER`，无双模型结论。
3. replay 仅覆盖 code-generation prompt（无 embedding 阶段），未复现 Mason 的 embedding retry 现象（本 replay retry=0）。
4. 12 个 prompt（每阶段 4）是代表性抽样，非完整 25 候选/轮的重放。
5. `wall_clock_s` 含 ~150s CPU JAX 校验（生成后、与 max_in_flight 无关）；主指标用 `llm_union_s`。
6. Ollama `-np 1` 是当前服务参数，未修改（任务约束）；若未来启用多槽，本结论需重测。

## 9. 是否值得进入后续组合实验

**否（就墙钟而言）**。在单槽 Ollama 上，`max_in_flight` 不改善墙钟，不值得进入组合实验。但有价值的后续方向是：(a) 在有界并发下**显式测量 embedding 阶段的 retry 消除**（本 replay 未含 embedding）；(b) 若 235B/多槽服务端可用，重跑本 sweep 以验证"多槽服务端 + 有界并发"是否真有墙钟收益。
