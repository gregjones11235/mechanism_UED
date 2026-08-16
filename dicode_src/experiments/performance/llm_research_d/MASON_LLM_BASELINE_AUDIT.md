# Mason attempt_06 — LLM 基线只读审计报告

分类：`LLM_RESEARCH_BASELINE_AUDIT`（独立研究线，非语义一致优化，不与 B/C 主线合并）

生成时间：2026-08-13

## 0. 结论摘要

Mason attempt_06 是一个**被有意停止**的 full-budget 试运行（`STOPPED_FOR_ENGINEERING_OPTIMIZATION`，SIGTERM/-15，非训练失败），约完成计划的 30%。它跑在 **Ollama 14B（qwen2.5-coder:14b）+ nomic-embed-text**（不是 235B），全部请求指向 `127.0.0.1:11434/v1`。

核心事实（直接日志证据）：

1. **Ollama 服务以 `-np 1`（单并行槽）运行**。生产 `llm.query()` 用 `asyncio.gather` 把所有 prompt 并发发出，但在服务端被强制串行。每轮 evolution 的 25 个描述 + 25 个代码 chat 请求在单槽上排队。
2. **大量传输层 retry**：863 次 retry（chat 289 / embed 574），全部请求最终 `200 OK`（无 4xx/5xx）。**embedding 请求几乎 100% 被 retry**（574/575）。
3. **零语义失败**：0 parse 失败、0 空响应、0 编译失败、0 重排队、0 reflection。所有候选一次编译通过。
4. **Evolution (LLM) 每轮 ~1488s，且随轮次增长 ~30%**（1195s→1747s），主要由 embedding 批次从 ~5 涨到 ~15-16 tasks、以及单槽串行生成长响应（max_tokens 8192）驱动。

## 1. 运行与配置（直接证据）

- 起点：`2026-08-10T12:13:41Z`，停止：`2026-08-11T09:26:32Z`，墙钟 ~21.2h。
- source_commit：`91a75e5`，source_tree_sha：`78c9b0e`。
- 生成：`qwen2.5-coder:14b`，temp 0.6，top_p 0.95，max_tokens **8192**。
- embedding：`nomic-embed-text`，embedding_size 1024。
- Ollama 服务（process_manifest argv）：
  - qwen llama-server：`-c 32768 -np 1 -b 1024 -ub 1024 --chat-template chatml`
  - nomic llama-server：`-c 2048 -np 1 --embedding -b 2048 -ub 2048`

> `-np 1` = 单生成槽。这是解释 "Ollama 14B 为什么需要数小时" 的关键：客户端的并发不会带来吞吐，反而在单槽上排队。

## 2. 请求量与错误分类（直接证据）

| 指标 | 值 |
|---|---|
| chat completions | 1626 |
| embeddings | 575 |
| HTTP 200 | 2201 |
| HTTP 4xx/5xx | 0 |
| retry 事件 | 863（chat 289 / embed 574） |
| parse 失败 / 空响应 / 编译失败 / 重排队 | 0 / 0 / 0 / 0 |

retry 特征：全部是 `openai._base_client` 的 **transport 层 retry**（backoff ~0.40–0.49s），每个请求最终都拿到 `200 OK`。**没有** rate-limit、server-error、timeout 的 HTTP 状态证据。

**embedding ~100% retry 率**是最突出的异常：日志呈现稳定的 `POST /embeddings "200 OK"` 后紧跟 `Retrying request to /embeddings` 的交替模式。即每个 embedding 请求都被重发一次。粗略估算仅 backoff 睡眠 + 重发就使 embedding 开销接近翻倍。

## 3. 阶段耗时（直接证据，来自 timings.csv）

- Evolution (LLM)：22 轮，累计 32729.41s，均值 1488s，范围 1181→1747s。
- Compilation (LLM)：22 轮，累计 3669.25s，均值 166.8s。
- Training：44 轮，累计 59767.62s，均值 1358.4s。

> Evolution 与 Compilation 在后台线程跑，与 GPU3 上的 Training 重叠，因此组件秒数之和 > 墙钟。

每轮 Evolution 组成（trainer.stdout）：`25 个描述 chat` + `25 个代码 chat` + `1 次批量 embedding（N tasks）` + `25 个 static/CPU-JAX 校验`。embedding 批次 N 从早期 ~5 涨到稳态 ~15–16，archive 停止时 554 节点。

## 4. GPU 证据（直接证据）

- GPU0（Ollama）：`GPU-e8c08612...`，观察时 util 0%，qwen 14906 MiB + nomic 656 MiB。
- GPU3（训练）：gpu_memory_5s.csv 采样 util 0%，内存 1→850 MiB 热身，长时间低利用（与 40-update 小 rollout 相对 LLM 墙钟占比较小一致）。

## 5. 直接证据 vs 推断（明确区分）

**直接日志证据**：provider/model/base_url/temp/max_tokens、请求数/retry 数/HTTP 状态、零 parse/compile/empty、逐轮 Evolution/Compilation/Training 时长、`-np 1` 单槽、embedding 批次增长、554 节点。

**推断（非日志直接证明）**：
- retry 的精确触发机制（连接 reset vs 连接池争用 vs 服务端关闭 keep-alive）未被日志捕获；只能确证"全部最终 200，且 transport 层重发"。
- "Evolution 变慢 30% 由 embedding 批次增长 + archive 上下文增长驱动"是相关推断，非严格因果归因。
- **返回文本长度、prompt 文本、prompt SHA256、message 哈希均未记录**（profiling 关闭，无 manifest），无法直接冻结，只能在重放时用"相同代码路径重建 prompt"来冻结。
- 首 token 延迟（TTFT）无法从这些日志观测。

## 6. 关键局限

1. `runtime_profiling.enabled = false`：无 events.jsonl / critical_path.json，只有 timings.csv。
2. 无 prompt/response 文本与哈希，无法直接冻结原始字节；冻结需依赖"用 source commit + 相同输入重建"的只读证据。
3. 无凭据泄露风险（local provider 用 `api_key='token-'`）。

## 7. 对三个研究问题的初步定位

- **Q1（235B 时间去哪）**：本 Mason 数据是 14B，不直接回答 235B；235B 需在 D2 中用当前服务实测。代码层已知：`llm.query()` 全并发无上限、`_query_with_retries` 指数退避、`check_compilation`（static lint + CPU JAX）串行在 evolution worker 内。
- **Q2（14B 为何要数小时）**：主因是 **单槽串行生成**（`-np 1`）叠加 **长输出（max_tokens 8192）** 与 **~100% embedding retry**，再加 embedding 批次随 archive 增长。生成后的 validation/preflight 反而轻（Compilation 仅 ~167s/轮）。
- **Q3（max_in_flight 1/2/4）**：由于 Ollama 单槽，客户端并发上限对 Ollama 墙钟影响有限（服务端本就串行）；但 235B/多槽服务端可能不同。D1 需实测验证。
