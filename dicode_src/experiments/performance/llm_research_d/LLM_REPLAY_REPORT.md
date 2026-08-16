# LLM 独立性能研究线 — 固定 replay 报告（阶段 D，修订版）

分类：`LLM_RESEARCH`（允许比较模型与调度行为；**非**语义一致优化，不与 B/C 主线合并）

> 本文件为 D1 结论修订版，收窄到证据实际支持的范围。原结论与修订差异见 `D1_AUDIT_ADDENDUM.md`。

## 0. 结论

**D1 正式结论（收窄后）：`D1_CHAT_ONLY_SINGLE_SLOT_NO_SIGNIFICANT_SPEEDUP`**

**阶段 D 总状态：`LLM_RESEARCH_PASS_WITH_CONCERNS`**

在当前单槽 Ollama 14B（`qwen2.5-coder:14b`，服务端 `-np 1`）、**12 个合成重建的 chat/code-generation prompts**、每档 2 次 repeat 的条件下，`max_in_flight ∈ {1,2,4}` **没有显示稳定显著的生成 union 或总 replay 墙钟收益**（生成 union 中位 432 / 420 / 425s，差异落在重复运行波动内 ~3%）。

本次 D1 **只覆盖 chat/code-generation**，**未覆盖 embedding**。各配置观察到的 `retry=0` 只表示"这 12 个 chat 请求在该次运行中没有重试"，**不能据此解释或解决 Mason 的 863 次 retry**（其中 574 次来自 embedding，D1 未含 embedding）。

**D2 235B 臂：`BLOCKED_EXTERNAL_PROVIDER`**（无本地 235B 服务、无 `DEEPINFRA_API_KEY`），未伪造双模型结论。

## 1. 配置矩阵汇总（中位数，各 2 次 repeat）

| max_in_flight | 墙钟(s) | 生成 union(s) | 生成 sum(s) | client semaphore wait sum(s) | retry | 空响应 | 有效任务 | 有效率 | 唯一代码 | LLM s/有效 | static_invalid | jax_failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 587.4 | 431.8 | 431.8 | 2180.9 | 0 | 0 | 9.5 | 0.792 | 12 | 45.6 | 2.5 | 0 |
| 2 | 565.1 | 419.5 | 793.6 | 1776.1 | 0 | 0 | 9 | 0.75 | 12 | 46.6 | 2 | 1 |
| 4 | 570.9 | 425.2 | 1446.6 | 1135.8 | 0 | 0 | 9 | 0.75 | 12 | 47.3 | 2.5 | 0.5 |

注：
- `生成 sum` 是并发请求 duration 的**直接求和**（并发下不等于墙钟）；`生成 union` 是**去重叠 union 时间**（≈ 真实生成墙钟）。
- `client semaphore wait sum(s)` 是客户端 coroutine 在 asyncio semaphore 上**逐请求等待的累加值**（历史字段名 `queue_wait_sum_s` 的语义澄清，见 §4）。**它不是**关键路径等待、服务端排队时间，也不能直接加入墙钟。
- `wall_clock` 包含 CPU JAX 校验（~150s/轮），与 max_in_flight 无关；主指标用 `生成 union`。

## 2. 允许保留的结论

在当前单槽 Ollama 14B、12 个合成重建 code-generation prompts、每档 2 repeat 的条件下，`max_in_flight=2/4` 相对 `1` 没有显示稳定显著的生成 union 或总 replay 墙钟收益。因此**暂不将客户端 chat 并发作为 48 小时主线加速项**。

## 3. 串行基线（max_in_flight=1）对比

- 串行基线生成 union：431.8s，有效率 0.792，retry 0，空响应 0，LLM s/有效 45.6s。
- mif=2：union 419.5s（−2.8% vs 串行），有效率 0.75，retry 0。
- mif=4：union 425.2s（−1.5% vs 串行），有效率 0.75，retry 0。

墙钟降幅（~1.5–2.8%）落在重复运行波动内（同配置两次 repeat 的 union 差 6–10s ≈ 1.4–2.5%），**不能判定为显著改善**。有效任务率 mif=1 出现一次 0.833（10/12）的样本，其余全部 0.75（9/12）；中位 0.792 vs 0.75 属统计波动，**不能断言并发提升或降低有效率**。

## 4. 计量语义澄清（对应审计第 6 条）

- `queue_wait_sum_s` 是客户端 semaphore wait 的**逐请求累加**，不是服务端 queue wait。
- 本次报告不将客户端 semaphore wait 解释为服务端排队。
- 未来运行将改用 `client_semaphore_wait_sum_s` / `client_semaphore_wait_union_s` / `client_semaphore_wait_critical_s`，并保留 `queue_wait_sum_s` 为 `legacy_alias=true`。

## 5. 门禁核查（修订后）

- ✅ 空响应率不恶化（全 0）。
- ✅ 本 12 请求 chat workload 未观察到 retry（retry=0；但**不涉及** Mason 的 embedding retry）。
- ✅ 服务端无持续 rate limit / queue collapse（0 次 429/5xx）。
- ⚠️ GPU0 显存只有每组运行前后快照（见 §7），**不宣称连续采样或精确峰值**。
- ⚠️ 有效任务率与串行基线同属统计波动，不构成"不低于基线"的强证据。
- ❌ **总 replay 中位生成墙钟未明显下降**（唯一未满足的推荐条件）。

## 6. 三个研究问题的回答（修订后）

1. **235B 时间去向**：本机无 235B 服务与凭据，**未能实测**（`BLOCKED_EXTERNAL_PROVIDER`）。代码层已知：`llm.query()` 全并发无上限、`_query_with_retries` 指数退避、`check_compilation` 在 evolution worker 内执行。
2. **Ollama 14B 为何要数小时**：主因是 **单槽串行生成**（`-np 1`）叠加长输出与大 system prompt。Mason 额外受 embedding retry 拖累，但 **D1 未含 embedding，不能在本报告中建立该归因**。
3. **max_in_flight 1/2/4**：在 chat/code-generation workload 上**未显示显著墙钟收益**。`retry=0` 仅反映本次 chat workload，**不构成"有界并发消除 Mason retry"的结论**。

## 7. GPU0 证据（仅快照）

- UUID：`GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6`；Ollama llama-server ×2（qwen 14906 MiB + nomic 656 MiB）。
- 每组运行前后快照显存为 43558 MiB（总 46068 MiB）；**快照之间未连续采样**，不宣称"全程恒定"，不给出精确峰值/最低剩余。
- 每 config 前后 util 0–1%；未启动第二个生成服务。

## 8. 冻结 replay 数据集（真实 workload 标签）

- **工作负载标签：`SYNTHETIC_RECONSTRUCTED_CODEGEN_WORKLOAD`**。
- manifest SHA256：`2066515499c305e263023e84a607c93ee4569708da57f23e56c8927c5b690d01`
- system_prompt SHA256：`5cdc991c2a3aced259548a69043be70b87ff083bd969f09f98e6a44fe85772b5`
- 12 个 user prompt（early task_1..4 / mid task_276..279 / late task_551..554），temperature 0.6，max_tokens 8192。
- 真实性说明：system prompt 来自固定 source commit `91a75e5`；task description 来自真实 `task_graph.graphml`；**CODE_EXAMPLES 为统一 seed code（collecting.py），未复刻生产 selector 的候选特定 examples**。因此该 workload **只用于受控调度比较，不用于候选质量的生产等价判断**。

## 9. 遗留限制

1. 每 config 仅 2 次 repeat，统计功效有限。
2. D2 235B 臂 `BLOCKED_EXTERNAL_PROVIDER`。
3. D1 未含 embedding 阶段，未复现 Mason 的 embedding retry。
4. 12 prompt 是抽样，非完整 25 候选/轮。
5. 原始 `events.jsonl / events.csv / critical_path.json / RESULT.json` 已随 `/tmp` 沙箱删除，**无法完成请求级独立复核**（见 addendum）。
6. GPU0 只有快照，无连续采样。

## 10. 后续

- 补齐 embedding 专项（D1b）与 chat 无界对照，判断 embedding retry 是否与客户端并发相关。
- 若 235B/多槽服务端可用，重跑以验证多槽下是否有墙钟收益。
