# D1c 生产形状 embedding retry 诊断 — 最终报告

分类：`LLM_RESEARCH`（独立研究线，非语义一致优化）

## 0. 结论

**D1c 结论：`D1C_RETRY_NOT_REPRODUCED`**

**D2 结论：`D2_BLOCKED_EXTERNAL_PROVIDER`**

**阶段 D 总状态：`LLM_RESEARCH_PASS_WITH_CONCERNS`**

本轮优先目标（回答"Mason 的 batched embedding 为什么几乎每次都触发 SDK transport retry"）**未能复现**：所有受控生产形状 arm（persistent contiguous / fresh client / 30s idle / 120s idle，各 2 repeat）SDK transport retry 全 0、错误全 0、空结果全 0。stale keep-alive 假设在 ≤120s 空闲尺度下未被证实。

## 1. 冻结 manifest

- SHA256：`a8ae10167a44825b21487ad9d4ea77110fe8b429dab434ed43673e70d2757649`
- 16 个真实任务描述（`FROZEN_PRODUCTION_EMBEDDING_TEXTS_FROM_MASON_ARCHIVE`）
- 生产 batch-size 序列：`[5, 10, 12, 13, 16, 10, 15, 16, 9, 16, 16, 10]`（12 批，每批 = pool[0:size]）
- embedding model：nomic-embed-text，size 768，base_url `http://127.0.0.1:11434/v1`

## 2. 实验矩阵与结果（每臂 12 批 × 2 repeat）

| arm | lifecycle | idle gap | r1 SDK retry | r2 SDK retry | error | empty |
|---|---|---|---|---|---|---|
| A | persistent contiguous | 0 | 0 | 0 | 0 | 0 |
| B | fresh client/request | 0 | 0 | 0 | 0 | 0 |
| C30 | persistent idle | 30s | 0 | 0 | 0 | 0 |
| C120 | persistent idle | 120s | 0 | 0 | 0 | 0 |

所有 96 个 batched 请求（8 arm × 12 批）SDK retry=0、HTTP 状态正常、item count / shape / 有限性校验通过。

## 3. 根因假设逐项结论

| 假设 | 判定 |
|---|---|
| H1 stale keep-alive（≤120s 空闲触发 retry） | **NOT_OBSERVED_AT_IDLE_LE_120S**（仅测到 120s 空闲，未排除更长空闲触发） |
| H2 asyncio.run 跨 event loop 复用失败 | **NOT_TESTED**（每个 arm/repeat 通过新 asyncio.run 建 loop，client 仅在单 loop 内存活，未测同一 client 跨 loop 复用） |
| H3 并发连接池争用 | **NOT_OBSERVED_UNDER_TESTED_WORKLOAD**（受控实验全 0 retry，但不构成对所有生产条件的反证） |
| H4 特定时间窗口服务端状态 | **NOT_OBSERVED**（当前环境无法复现/证实） |
| H5 remote protocol error（200 后断流） | **NOT_OBSERVED**（无 retry 可诊断） |

**未宣称根因已确定**。Mason 574/575 retry 的触发条件仍未定位；可能依赖分钟级以上的 chat-generation 空闲（本任务明确禁止为模拟历史状态等待数小时），或特定时间窗口的服务状态。

### 代码修复说明（供未来运行使用，历史结果未改动）

- `run_d1c.py`：`ollama_pid_after` 原在 `await embed()` 前即求值（并非真正请求后 PID）。已改为向 `embed()` 传入 getter，在 await 返回/异常后才采样。
- `d1c_harness.validate_embedding`：现已强制 `len(embedding) == configured embedding_size`，维度不匹配 fail-closed。
- 历史 `D1C_ALL_RESULTS.json` **未修改**；其 request-level `ollama_pid_after` 不作为真正请求后证据；run-level baseline/final PID 与连续 GPU sampler 证据仍保留。

## 4. GPU0 连续采样证据

- 采样文件：`llm_research_d1c/gpu0_memory_2s.csv`，**1601 行样本**（2s 间隔，覆盖全部 8 臂墙钟区间）。
- `uuid_consistent=true`，`peak_memory_used_mib=15573`，`min_memory_free_mib=30495`，`max_temperature=45`。
- compute PIDs 全程 `[3154045, 3156249]`，前后一致，无 OOM/Xid，无第二个生成服务。

## 5. D2：235B provider 检查

- `DEEPINFRA_API_KEY`：NOT SET。
- 无本地 235B 服务（:5000 无响应），Ollama 仅 `qwen2.5-coder:14b` + `nomic-embed-text`。
- 结论：**`D2_BLOCKED_EXTERNAL_PROVIDER`**（只读探测一次，不重复探测）。

## 6. 生产修改建议（门禁未满足）

- retry 未复现 → **不修改生产调度**；建议增强 request-level transport telemetry，等真实 full-budget 再捕获。
- synthetic concurrency 有收益但生产是单 batched request → **不建议 max_in_flight**；archive embedding 的增量计算/缓存应另立任务。
- 不提出任何生产代码修改。

## 7. 是否纳入研究线组合

- **不建议纳入组合**：chat 无大幅稳定收益（已知）；embedding retry 未复现、合成并发收益不映射生产单批路径。

## 8. 遗留限制

1. 空闲尺度仅测到 120s，未测分钟级（任务禁止为模拟历史状态等待数小时）。
2. Mason 574/575 retry 根因仍未定位，保留 `D1B_EMBEDDING_RETRY_CAUSE_NOT_CONFIRMED` 结论。
3. 235B 臂 `BLOCKED_EXTERNAL_PROVIDER`。
4. 未修改生产代码、未触碰 GPU1/2/3、未阻塞 B/C。
