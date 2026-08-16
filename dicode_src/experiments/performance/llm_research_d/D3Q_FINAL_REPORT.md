# D3Q 最终报告 — 小模型 vs 大模型 端到端 Preflight 对比

classification: D3Q_FINAL_REPORT
schema_version: 1
recorded_utc: 2026-08-16T08:50:00Z
branch: perf/llm-small-large-quality-cost-d3
base: 91a75e5（未 push / 未 merge）

## 最终结论

**D3_SMALL_MODEL_FASTER_END_TO_END**

本地 Ollama `qwen2.5-coder:14b` 在“生成 + repair + CPU-JAX 验证 + Preflight”全链路中，
平均每个被 Preflight 接受的任务耗时 **310.7 秒**，DeepSeek 官方 `deepseek-v4-flash` 为
**730.8 秒**（小模型快约 2.35 倍），三次重复实验方向全部一致（278.1 / 335.9 / 318.2 vs
1202.0 / 473.9 / 516.7 秒）。小模型虽最终有效任务率更低（83–86% vs 100%）且存在 5 个
被丢失槽位（事故 01），但 Preflight 接受率显著更高（65.4% vs 30.6%），每个接受任务的
总耗时仍全面占优。**“小模型因无效任务反而更慢”不成立**（五条件中的第 3 条不满足）。

## 主指标（秒 / Preflight 接受任务）

| arm | 候选 | 接受 | pipeline_wall_s | preflight_wall_s | 秒/接受 |
|---|---:|---:|---:|---:|---:|
| large_r1 | 12 | 2 | 1525.1 | 878.8 | 1202.0 |
| large_r2 | 12 | 5 | 1489.7 | 879.6 | 473.9 |
| large_r3 | 12 | 4 | 1226.6 | 840.1 | 516.7 |
| small_r1 | 6 | 5 | 718.2 | 672.1 | 278.1 |
| small_r2 | 10 | 6 | 1078.8 | 936.4 | 335.9 |
| small_r3 | 10 | 6 | 1107.8 | 801.4 | 318.2 |

均值：small **310.7**（17 个接受任务）vs large **730.8**（11 个接受任务）。

## 质量与成本

- 最终有效任务率：large 3×100%；small r1 6/7=85.7%、r2/r3 10/12=83.3%。
- Preflight 接受率：large 11/36=30.6%；small 17/26=65.4%（不含 5 个丢失槽位）。
- repair 请求：large 9/7/5，small 4/5/6；空响应 / 超时 / 连接错误均为 0。
- 错误分类：ollama `api_enum_error=17`、`cpu_jax_error=3`、`duplicate_code=3`；
  deepseek `extract_error=16`、`syntax_error=1`、`cpu_jax_error=4`。
- 成本：DeepSeek 官方 API 共 57 次 POST，3,676,850 输入 / 313,057 输出 token
  （缓存命中 3,630,976），按 2026-08-16T16:00Z 前官方平峰价快照
  （$0.14/M 未命中、$0.0028/M 命中、$0.28/M 输出）计算 **$0.104**；Ollama 为本地 GPU0
  服务，无 API 费用。

## 事件披露

- **事故 01**（r1-small 丢槽）：launcher 超量记账 + 批量收集丢证据；已修复并
  `D3Q_BUDGET_RECONCILIATION.json` 对账；`slot_r1_small_p01–p05` 永久标记
  attrition_no_rerun，r1-small 为 7/12 prompts，相关速率均按可得槽位计算。
- **事故 02**（r3 后置门禁）：外部 perf48 进程临时占 GPU2，经
  `recover-completed-chunk` 恢复（162 文件 SHA256 重验）。
- **事故 03**（根分区满）：服务器 `/dev/sda2` 100% 满阻断部署；`pip cache purge`
  释放 3.7G 后恢复。
- **事故 04**（SSH 断开）：orchestrator 单长连接被服务端断开，远程 driver 成孤儿
  且 finally 误删 exec root；已改为 setsid 分离驱动 + 60s 轮询 + 存活进程保护的清理。
- **事故 05**（后置门禁外部进程）：外部 `pytest tests/e3_litesim`（PID 3753771）
  在 small_r3 窗口占 GPU2 262 MiB；量化干扰分析显示无劣化（small_r3 execute 202.6s
  < small_r2 232.9s，ratio 1.04），PID 退出后经 `recover-completed-run` 恢复为 PASS。

## 限制（诚实声明）

- 所有 Preflight 在 `--xla_gpu_deterministic_ops=true` 下运行（训练约慢 31%），绝对秒数
  偏高；两臂同条件，相对结论不受影响。
- r1-small 丢失槽位使 small 臂仅 26 个候选进入 Preflight；速率按可得槽位计算。
- DeepSeek 定价按 2026-08-16T16:00Z 前的平峰窗口快照；16:00Z 起官方切换峰谷计价。
- 生成阶段每档仅一次运行（3 次重复 × 2 模型），无置信区间；主指标方向 3/3 一致。
- 本地测试 `test_d3_cpu.py::test_manifest_and_contract_are_frozen` 在默认 cwd 下因
  `llm_replay_manifest` 不在 `sys.path` 而失败（既有路径问题，`PYTHONPATH=..` 后通过），
  与本次改动无关。

## 交付物

权威输出：`d3q_artifacts/d3q_final_20260816T083944Z/`
（D3Q_FINAL_RESULT.json、D3Q_SLOT_RESULTS.csv、D3Q_PREFLIGHT_RESULTS.json、
D3Q_ERROR_TAXONOMY.json、D3Q_COST_AND_TOKENS.json、D3Q_ARTIFACT_INVENTORY.json、
SHA256SUMS）；`d3q_artifacts/D3Q_REQUEST_LEDGER.jsonl` 随证据提交入库。
被取代输出 `d3q_final_20260816T083433Z` / `d3q_aggregate_20260816T064133Z`
已放置 `SUPERSEDED.md` 标记并保留作审计证据。

## Git 与测试

- HEAD：本报告与全部证据所在的交付提交（branch `perf/llm-small-large-quality-cost-d3`，未 push / 未 merge），
  工作区除有意保留未跟踪的失败/被取代产物外干净。
- 本地 pytest：116 passed + 1 pre-existing path failure（见限制）；preflight/finalize
  专项 17 passed（8 finalize + 9 preflight）。
- 独立审核结论：**PASS_WITH_CONCERNS** —— 实验与门禁完整通过，残余关注点全部
  在上方“限制”中如实披露。
