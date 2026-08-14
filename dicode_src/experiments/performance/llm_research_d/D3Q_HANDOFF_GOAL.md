# D3Q 总目标（交接）

## 一句话目标

在服务器上公平对比 **本地 Ollama `qwen2.5-coder:14b`** 与 **官方 API `deepseek-v4-flash`**，回答：小模型虽然单次生成可能更快，是否因为生成更多无效任务、需要更多 repair、真实 Preflight 接受率更低，而导致"每个最终有效任务的总耗时"反而更慢。

## 主要指标

```
(生成 wall + repair wall + CPU-JAX validation wall + Preflight wall)
────────────────────────────────────────────────────────────────
Preflight accepted task 数
```

accepted=0 时结果必须为 infinity/null 并判失败，不得零除或隐藏。

## 两条模型臂

- Arm A（小）：Ollama OpenAI-compatible，`qwen2.5-coder:14b`，GPU0，非 thinking。
- Arm B（大）：DeepSeek 官方 API，`deepseek-v4-flash`，base `https://api.deepseek.com`，thinking `{"type":"enabled"}`。
- 禁止 `deepseek-v4-pro`、`deepseek-chat`、`deepseek-reasoner`、Qwen API、DeepInfra、任何模型名 fallback。
- 参数规模：DeepSeek 官方将 V4 Flash 描述为 284B total / 13B active；本地 Qwen 约 14.8B 量化部署。这是"服务/模型整体工程效果"对比，不是纯参数规模消融。

## 必须交付

`D3Q_FINAL_RESULT.json`、`D3Q_FINAL_REPORT.md`、`D3Q_REQUEST_LEDGER.jsonl`、`D3Q_SLOT_RESULTS.csv`、`D3Q_PREFLIGHT_RESULTS.json`、`D3Q_ERROR_TAXONOMY.json`、`D3Q_COST_AND_TOKENS.json`、`D3Q_ARTIFACT_INVENTORY.json`、`SHA256SUMS`，加上 Git 状态、测试结果、独立审核结论、一段中文结论。

## 最终结论（七选一）

`D3_SMALL_MODEL_FASTER_END_TO_END` / `D3_DEEPSEEK_FLASH_FASTER_END_TO_END` / `D3_SMALL_MODEL_INVALIDITY_ERASES_LOCAL_SPEED_GAIN` / `D3_NO_CLEAR_WINNER` / `D3_BLOCKED_PROVIDER` / `D3_BLOCKED_RUNTIME` / `D3_REJECTED_EVIDENCE_INTEGRITY`

## 能答"小模型因无效任务反而更慢"的条件（缺一不可）

1. Ollama 14B 生成阶段更快；2. 其有效率更低或 repair 更多；3. 其每 accepted task 总秒数更高；4. 至少 2/3 repeats 同方向；5. 结果不能被 API outage、GPU 干扰或协议差异解释。否则只能写"证据不足 / 无明确赢家"。

详见 `D3Q_HANDOFF_PROMPT.md` 的完整协议、约束、当前状态与压缩计划。
