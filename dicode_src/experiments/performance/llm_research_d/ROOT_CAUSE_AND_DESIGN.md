# 阶段 D — LLM 独立性能研究线：root-cause 与实验设计

分类：`LLM_RESEARCH`（独立研究线，允许比较模型与调度行为，**非**语义一致优化，不与 B/C 主线合并）。

## 1. Root cause（基于只读审计 + 代码走读）

### 1.1 生产 LLM 调用链（代码证据）

```
run_dicode.py (main thread)
  └─ dispatch_evolution_worker (ThreadPoolExecutor max_workers=1, background)
       └─ evolve_and_validate_tasks
            ├─ gen_manager.evolve_tasks  → "Evolution (LLM)" 计时
            │    ├─ evolve_mastered: 构建 system_prompt + 25 个 user_prompts
            │    │    └─ _query_and_parse_responses → llm.query()  ← 全并发 asyncio.gather
            │    ├─ selector.select_similar_desc_tasks → embedding_model.get_embedding()
            │    └─ env_generator.generate → llm.query() (代码生成，25 prompts)
            └─ check_compilation × 25  → "Compilation (LLM)" 计时
                 ├─ _static_lint (AST 校验 Craftax 枚举/kwargs)
                 └─ _check_compilation_uncached (CPU JAX compile+run)
  └─ (main thread) preflight: evaluate_new_tasks (40-update rollout) → route
```

关键点：
- `llm.query()` 用 `asyncio.gather` 把**所有** prompt 并发发出，**无 `max_in_flight` 上限**。
- `_query_with_retries` 在空响应/异常时做指数退避（2/4/8s）。
- `check_compilation`（static lint + CPU JAX）在 evolution worker 内**串行**（`evolve_and_validate_tasks` 用 ThreadPoolExecutor 并行校验，见 evolution_efficient.py:182）。

### 1.2 Mason attempt_06（14B）时间去向（直接日志证据 + 推断）

- Ollama 服务 `-np 1` 单槽 → 客户端并发被服务端串行化。
- 22 轮 Evolution 均值 1488s，末段比初段慢 ~30%，与 embedding 批次 5→16 增长相关。
- **embedding ~100% transport retry**（574/575），chat ~18%（289/1626），全部最终 200。
- 零 parse/compile/empty 失败 → 时间不在 repair/validation 失败重试上。

### 1.3 三个研究问题的当前答案假设（待 D1/D2 实测验证）

1. **235B 时间去向**：需 D2 实测；代码层已知 chat 全并发无上限、retry 指数退避、validation 串行。
2. **14B 为何数小时**：单槽串行生成 + 长输出（max_tokens 8192）+ ~100% embedding retry + 批次增长。
3. **max_in_flight**：Ollama 单槽下客户端并发收益有限；235B/多槽服务端收益待测。

## 2. 隔离与工具边界

- 独立 worktree：`skill_preflight_ued_llm_research_worktree`，分支 `perf/llm-replay-research-d`，base `453dc356d29dce783dfb7c6e915f5195dc272fe1`。
- 新工具文件（**不修改生产** `llm.py`/`gen_manager.py`/`run_dicode.py`/`preflight_replay.py`）：
  - `llm_replay_manifest.py` — 冻结 manifest 的构建/原子写/重载/篡改校验。
  - `llm_replay_harness.py` — 独立 OpenAI-compatible 客户端 + max_in_flight 信号量 + 单调时钟事件 + 错误分类 + validation（static/CPU-JAX 只读调用，副本隔离目录）。
  - `llm_replay_benchmark.py` — D1/D2 编排、顺序交替、repeat、去重叠 union/关键路径计算。
  - `llm_replay_report.py` — RESULT.json / LLM_REPLAY_REPORT.md / events.csv / critical_path.json。
- 生产接口通过 `AsyncOpenAI` 直连 `base_url` 调用（与 `llm.py._create_client` 同协议），**不 import 生产 `LLM`**；provider 不可用时 fail-closed。

## 3. 冻结 replay 数据集

Mason attempt_06 无 prompt/response 日志（profiling 关闭），无法直接冻结原始字节。冻结策略（如实声明为"重建式冻结"）：

- **system prompt**：从 source commit 的 prompt 模块（`dicode.dreaming.prompts.*`）读取并 SHA256。
- **user prompts**：从 `task_graph.graphml` 读取代表性 mastered 任务的 description，按 `evolve_mastered_prompt.user_prompt` 模板重建；early/mid/late 每阶段 ≥2 个 cycle。
- **manifest 字段**：system_prompt、user_prompts（有序）、prompt_sha256、请求顺序、请求数、token 上限、temperature、candidate_slots、repair 上限、validation 配置、response_sha256（若有）、provider/model、source_commit、tool SHA。
- 原子写 + 自哈希；加载时重新校验，篡改即拒。
- 不写入任何 API key / token / Authorization。

## 4. 实验矩阵

### D1：同模型并发调度
- 同一冻结 prompt replay，对每个可用模型 × `max_in_flight ∈ {1,2,4}`。
- 请求数/顺序/槽位/temperature/token 上限/timeout-retry 策略全同；不削减候选预算。
- 每配置 ≥2 次 repeat，顺序交替（避免首轮偏差）。
- GPU0 始终单队列；`max_in_flight` 是队列内并发；不并发启动多个 evolution cycle。

### D2：235B vs Ollama 14B
- 同一 replay 协议；明确标记 `RESEARCH_NON_SEMANTIC_MODEL_COMPARISON`。
- 比较：总生成墙钟、chat 墙钟、queue wait、TTFT（接口可观测时）、输出 token 数、retry/backoff、空响应、repair、static 非法率、CPU-JAX 通过率、有效任务率、每有效任务 LLM 秒、GPU0 峰值/最低剩余显存。
- 235B 或凭据不可用 → 停该臂，标记 `BLOCKED_EXTERNAL_PROVIDER`，不伪造、不把单模型结论扩成双模型结论。

## 5. 事件与计时框架

- 单调时钟 `time.monotonic_ns`，追加式 JSONL，字段固定（见 harness 常量）。
- phase：`replay_wall / queue_wait / chat_request / embedding_request / retry_backoff / response_parse / static_validation / repair_request / cpu_jax_validation / candidate_finalize / result_write`。
- error_class：`timeout / connection_error / server_error / rate_limited / empty_response / invalid_json / static_invalid / jax_validation_failed / cancelled / unknown_error`。
- 输出：`events.jsonl`、`events.csv`、`critical_path.json`、`RESULT.json`、`LLM_REPLAY_REPORT.md`。
- 并发 duration 不直接相加；报告计算：duration 总和、去重叠 union、关键路径、queue wait、主线程同步等待、实际重叠收益。

## 6. 候选验证边界

- 不优化 preflight，不跑 40-update PPO preflight。
- 只读调用当前 static lint 与 CPU JAX validation（`gen_manager.EnvGenerator._static_lint` / `_check_compilation_uncached` 的等价逻辑或直接 import，副本隔离 tmp 目录）。
- 相同返回代码按哈希识别重复项，但**不删槽位、不改顺序、不削减预算**；报告同时给出原始槽位数与唯一代码数。
- GPU2 仅最小功能 smoke；正式 LLM 墙钟计时不与 GPU2 JAX 混跑。

## 7. 研究门禁（通过标准）

调度方案只有同时满足以下才可推荐：
- 总 replay 中位墙钟明显下降；
- 有效任务率不低于串行基线（超过允许统计波动）；
- 空响应率不恶化；retry/repair 无异常增加；
- GPU0 最低剩余显存安全；服务端无持续 rate limit / queue collapse。

必须重点报告 `LLM seconds per valid task`。

## 8. 最终结论（五选一）

`LLM_RESEARCH_REPLAY_PASS` / `LLM_RESEARCH_PASS_WITH_CONCERNS` / `LLM_RESEARCH_NO_IMPROVEMENT` / `LLM_RESEARCH_BLOCKED_EXTERNAL` / `LLM_RESEARCH_REJECTED`
