# D1b embedding 专项 + chat 无界对照 — 最终交付报告

分类：`LLM_RESEARCH`（独立研究线，非语义一致优化，不与 B/C 合并）

## 0. Git 与交付

| 项 | 值 |
|---|---|
| branch | `perf/llm-replay-research-d` |
| base | `453dc356d29dce783dfb7c6e915f5195dc272fe1` |
| HEAD | `6d0bbd4`（本报告后追加最终提交） |
| 新增 commit | `b5bf29e` fix(d): narrow D1 conclusions and preserve replay evidence；`6d0bbd4` feat(d): add controlled embedding concurrency replay (D1b) |
| git status | 干净（本报告写入后重新提交） |

修改文件（本研究线，未碰生产 `llm.py`/`gen_manager.py`/`run_dicode.py`/`preflight_replay.py`）：
- `experiments/performance/llm_replay_{manifest,harness,benchmark,gpu}.py`
- `experiments/performance/llm_research_d/`（报告、addendum、冻结/运行脚本、结果 JSON）
- `src/dicode/skill_preflight/tests/test_llm_replay_d.py`

## 1. D1 修订前后结论

- 原：`LLM_RESEARCH_NO_IMPROVEMENT` → 修订：`D1_CHAT_ONLY_SINGLE_SLOT_NO_SIGNIFICANT_SPEEDUP`
- 阶段 D 总状态：`LLM_RESEARCH_PASS_WITH_CONCERNS`
- 详见 `D1_AUDIT_ADDENDUM.md`。

## 2. D1 保留数据清单

六组汇总数字（`result_sha256` 均验证通过）；Mason 日志统计（HTTP200=2201/chat=1626/embedding=575/retry=863）；GPU0 UUID+PID；本地 33 项测试；manifest SHA `2066515499…`。

## 3. D1 缺失且无法恢复的证据

六次运行原始 `events.jsonl/events.csv/critical_path.json/RESULT.json`（随 `/tmp` 沙箱删除）；GPU0 连续采样（从未采集）。**不补造。**

## 4. D1b manifest SHA

`533f724bcf33d004d4812bf1beec9b65829c91b1dc36521c4f712b7adbacdd32`（25 个真实任务描述文本，`FROZEN_EMBEDDING_TEXTS_FROM_MASON_ARCHIVE`）

## 5. D1b embedding 结果（每档每次 repeat）

### 5.1 无批处理（25 独立 1-text 请求）

| max_in_flight | repeat | wall(s) | union(s) | sum(s) | sem_wait_sum(s) | retry | SDK_retry | empty | error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | r1 | 1.254 | 1.244 | 1.244 | 14.46 | 0 | 0 | 0 | {} |
| 1 | r2 | 1.090 | 1.080 | 1.080 | 13.01 | 0 | 0 | 0 | {} |
| 2 | r1 | 0.656 | 0.655 | 1.259 | 7.13 | 0 | 0 | 0 | {} |
| 2 | r2 | 0.637 | 0.636 | 1.251 | 7.17 | 0 | 0 | 0 | {} |
| 4 | r1 | 0.350 | 0.349 | 1.331 | 3.62 | 0 | 0 | 0 | {} |
| 4 | r2 | 0.356 | 0.355 | 1.328 | 3.88 | 0 | 0 | 0 | {} |
| 25 | r1 | 0.264 | 0.264 | 4.210 | 0.00 | 0 | 0 | 0 | {} |
| 25 | r2 | 0.289 | 0.289 | 4.391 | 0.00 | 0 | 0 | 0 | {} |

### 5.2 批处理（25 个 16-text 请求）

| max_in_flight | wall(s) | SDK_retry | error |
|---|---|---|---|
| 1 | 3.948 | 0 | {} |
| 4 | 2.757 | 0 | {} |
| 25 | 2.870 | 0 | {} |

## 6. chat 无界对照结果（12 合成 prompt，`PRODUCTION_STYLE_CLIENT_UNBOUNDED_FOR_12_REQUESTS`）

| max_in_flight | repeat | wall(s) | union(s) | retry | SDK_retry | valid_rate |
|---|---|---|---|---|---|---|
| 1 | c1 | 600.77 | 428.92 | 0 | 0 | 0.833 |
| 12 | c1 | 572.24 | 409.91 | 0 | 0 | 0.833 |
| 12 | c2 | 567.36 | 424.32 | 0 | 0 | 0.75 |
| 1 | c2 | 578.02 | 433.34 | 0 | 0 | 0.75 |

chat 中位 union：mif=1 ≈ 431.1s，mif=12 ≈ 417.1s（−3.3%，在重复波动内）。retry 全 0。

## 7. GPU0 连续采样证据

- CSV：`llm_research_d1b/gpu0_continuous.csv`（2s 间隔，5 样本——embedding 实验极快，样本数少）。
- `uuid_consistent=true`，`peak_memory_used_mib=15573`，`min_memory_free_mib=30495`，`max_temperature=42`，`compute_pids_observed=[3154045, 3156249]`，前后 PID 一致。
- 无 OOM/Xid，无第二个生成服务。

## 8. 原始 artifact 持久目录

`/home/oseasy/e2_data_disk2/skill_preflight_runs/llm_research_d1b/`（含 `out/`（9 个 embedding RESULT）、`out_batch/`、`out_chat_unbounded/`、`frozen_embedding_manifest.json`、`gpu0_continuous.csv`）。每次 `run_replay`/`run_embedding_config` 运行目录内写有 `SHA256SUMS` + `ARTIFACT_INVENTORY.json`（批处理 run 仅 RESULT+events.csv，见限制）。

## 9. 测试结果

`test_llm_replay_d.py`：**33 passed**；py_compile PASS；git diff --check clean。（相关 skill_preflight 测试需 craftax/jax，本地不可跑，未在本地执行。）

## 10. 命题判定

| 命题 | 判定 | 依据 |
|---|---|---|
| chat 并发没有墙钟收益 | **CONFIRMED** | D1 union 432/420/425s + chat 无界 union 431/417s，均无显著差异（`-np 1` 单槽瓶颈） |
| embedding retry 由并发导致 | **NOT_CONFIRMED** | D1b 无批+批处理，1/2/4/25 并发下 SDK retry 全 0，未复现 Mason 的 ~100% embedding retry |
| 有界并发减少 embedding retry | **NOT_CONFIRMED** | 各并发档 retry 均为 0，"减少"无从谈起；Mason 的 retry 来源未由并发解释 |
| 235B 优于或慢于 14B | **BLOCKED_EXTERNAL_PROVIDER** | 无本地 235B、无 `DEEPINFRA_API_KEY`，未伪造双模型结论 |

## 11. 最终状态

**`LLM_RESEARCH_PASS_WITH_CONCERNS`**

附加子结论：
- D1 修订：`D1_CHAT_ONLY_SINGLE_SLOT_NO_SIGNIFICANT_SPEEDUP`。
- D1b：`D1B_EMBEDDING_NO_IMPROVEMENT`（就"并发导致 retry"这一命题而言无改善，且未复现 retry）。

## 12. 遗留限制

1. D1b 仅 2 repeat；embedding 实验极快导致 GPU 连续采样仅 5 样本。
2. Mason 的 embedding 是**批处理**请求（16-text/请求），D1b 批处理 arm 未写完整 artifact 清单（仅 RESULT+events.csv）。
3. Mason 的 863 retry 未能复现，可能源于：特定时间段服务器状态、连接池配置、或生产 `asyncio.gather` 无界并发的更大队列——本实验 25 并发/12 并发均未触发。
4. 235B 臂 `BLOCKED_EXTERNAL_PROVIDER`。
5. 未修改生产调度、未进入 full-budget、未触碰 B/C 与 GPU1/2/3。

本任务结束，不自动修改生产调度、不自动进入 full-budget。
