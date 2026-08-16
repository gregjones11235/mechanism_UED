# D1/D1b 最终审计报告（结论修正版）

分类：`LLM_RESEARCH`（独立研究线，非语义一致优化，不与 B/C 合并）
审计 HEAD：`c69b177966481d5017c2e139da6697641f0ac4a4`

## 0. 总状态

**`LLM_RESEARCH_PASS_WITH_CONCERNS`**（不得升级为完全 PASS）

工程含义：Chat 并发没有显示大幅稳定收益，暂不进入组合；embedding 并发在合成压力 replay 中确实提速，但没有复现 Mason retry，也尚未证明生产 Mason 路径能获得同等收益。

## 1. Chat 并发

结论：**`D1_CHAT_CONCURRENCY_NO_LARGE_STABLE_GAIN`**

- max_in_flight=1 → 12：wall 中位 589.4s → 569.8s（约 **+3.33% 改善**），LLM union 中位 431.1s → 417.1s（约 **+3.25% 改善**）。
- 有效率逐 repeat 配对一致（c1 均 0.833，c2 均 0.75），retry 均为 0。
- 每档仅 2 个 repeat，实验前未定义置信区间/噪声阈值，**不能证明该改善稳定**。
- 当前证据只排除了"大幅稳定收益"；这既不代表零收益，也不代表无收益已被证实。
- 暂不把 chat 并发纳入 48 小时组合主线。

## 2. 非批处理 embedding

结论：**`EMBEDDING_CONCURRENCY_SPEEDUP_OBSERVED_ON_SYNTHETIC_WORKLOAD`**

- max_in_flight=1 → 25：中位墙钟 1.172s → 0.277s（约 **+76% 改善**）。
- 这是合成的 25 个独立单文本请求 workload，**不能直接外推 Mason 生产 session**。
- 所有档 retry=0、SDK retry=0、empty=0、error_counts={}。

## 3. 批处理 embedding 压力测试

结论：**`BATCH_EMBEDDING_CONCURRENCY_SPEEDUP_OBSERVED_IN_STRESS_REPLAY`**

- 25 个并发 16-text batch：mif=1 3.948s → mif=4 2.757s / mif=25 2.870s（约 **+27%～30% 改善**）。
- workload 是 25×16-text 合成压力 replay；Mason 生产通常是一轮一个 batched 请求。
- 结果只证明压力 replay 中存在并发吞吐收益，**不能直接外推为 Mason session 会获得 27%～30% 提速**。

## 4. Mason retry 根因

结论：**`D1B_EMBEDDING_RETRY_CAUSE_NOT_CONFIRMED`**

- 所有受控并发档（chat 1/2/4/12，embedding 1/2/4/25）retry 全 0。
- 未复现 Mason 的 574/575 embedding transport retry。
- **不能确认 Mason retry 来自客户端并发**，也**不能确认有界并发已解决 retry**。
- 潜在原因仍包括：连接池、keep-alive、服务部署状态、transport reset、特定时间窗口。

## 5. 235B

结论：**`BLOCKED_EXTERNAL_PROVIDER`**

无本地 235B 服务、无 `DEEPINFRA_API_KEY`。不伪造、不推断 235B 与 14B 的速度/质量对比。

## 6. 哈希契约（修复后）

`run_replay()` 返回契约已修复，哈希作用域显式化：

- `result_sha256`：继续验证 raw RESULT 语义内容，算法 `legacy_default_json_sha256`，作用域 `RESULT_FIELDS_EXCLUDING_RESULT_SHA256_AND_ARTIFACT_INVENTORY`。
- `artifact_inventory_sha256`：规范化算法 `canonical_json_sha256`（json.dumps + sort_keys + separators=(",",":") + ensure_ascii=False + UTF-8 + SHA256）。
- `enriched_summary_sha256`：所有 enriched 字段追加后计算，排除自身，算法 `canonical_json_sha256`。
- `legacy_result_hash_algorithm` / `canonical_summary_hash_algorithm`：分别记录两种算法。
- 旧 enriched summary 标记 `LEGACY_ENRICHED_HASH_SCOPE_AMBIGUOUS`，不伪称旧条目整体哈希通过。

## 7. GPU0 采样证据（限定采样窗口）

- `D1B_GPU0_CONTINUOUS.csv` 仅 1 行 header + 5 行数据，采样**不覆盖**约 570~600 秒的 chat 对照窗口（主要覆盖很短的 embedding 实验）。
- 仅允许声明：已采样窗口内 UUID 一致、PID 一致、峰值显存 15573 MiB、最低 free 30495 MiB、未观察 OOM/Xid。
- 禁止声明：chat 对照全程 GPU0/PID/显存稳定、5 个样本足以代表整个 D1b 与 chat 无界对照、chat 无界对照有完整连续 GPU 监督。

## 8. 遗留限制

1. chat 每档仅 2 repeat，无预设置信区间。
2. Mason retry 未复现，根因未定位。
3. embedding 为合成 workload，非生产等价。
4. GPU 采样仅 5 样本，不覆盖 chat 对照窗口。
5. 235B 臂 `BLOCKED_EXTERNAL_PROVIDER`。
6. 未修改生产调度、未进入 full-budget、未触碰 B/C 与 GPU1/2/3。
