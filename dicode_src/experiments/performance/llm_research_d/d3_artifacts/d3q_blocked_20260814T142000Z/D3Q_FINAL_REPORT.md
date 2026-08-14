# D3Q 模型质量—速度—成本对比 — 最终报告（BLOCKED）

## 结论

**`D3_BLOCKED_PROVIDER`**

D3Q 实验在 **DeepSeek metadata gate** 阶段被阻断，无法进入生成矩阵与 GPU2 Preflight。

## 根因链（逐步定位，全部只读/授权后操作）

1. **原始 request_count=0 阻断**：launcher 曾把合法的 `request_count=0`（PRE_REQUEST_BLOCKED）误写成 `artifact_request_count_invalid`，丢失了精确 failure_class。**已修复**（commit `1993495` + `9e872f2`）。

2. **launcher 环境适配**：OpenSSH 10.x 的 post-quantum WARNING 写进 stderr，使 launcher 的严格 stderr 校验误判。**已修复**（commit `a756e7f`）。

3. **模型配置冲突**（第一层真实阻断）：远端 env 的模型是 `EXP_GENERATOR_MODEL_ID=deepseek-v4-pro`（任务禁止），而 D3Q 要求 `deepseek-v4-flash`。经用户授权，新增 `EXP_DEEPSEEK_MODEL=deepseek-v4-flash` 与 `EXP_DEEPSEEK_PROVIDER=deepseek`（未改动既有 `EXP_GENERATOR_*`）。

4. **网络不可达**（最终阻断）：修复模型后，gate 进入 `request_count=1` 并发出 `GET https://api.deepseek.com/models`，但得到 `transport_error`（`http_status=null`）。只读诊断确认：
   - DNS 能解析 `api.deepseek.com`（→ `43.141.130.88` / `43.141.68.41`）。
   - **TCP 443 连接失败**（防火墙阻断外网 HTTPS 到该域）。
   - 同一服务器可连 `github.com:443`（说明是选择性白名单，DeepSeek 不在其中）。
   - 无 http/https 代理、无本地代理端口。

## 为什么无法继续

- D3Q 的 Arm B（DeepSeek Flash）凭据只能从服务器 env 读取，API 调用必须发生在服务器；
- 但服务器到不了 `api.deepseek.com`；
- 没有可用的代理，也不能在本地读取凭据（任务禁止凭据离开服务器 env）。

因此两臂公平对比（Ollama 14B vs DeepSeek Flash）无法成立，`seconds_per_preflight_accepted_task` 无法计算。

## 证据与交付状态

- metadata gate evidence：`d3_artifacts/deepseek_flash_metadata_gate_rerun_20260814T142004Z/`（`reason=request_attempted_blocked`, `gate_reason=transport_error`, `request_count=1`）。
- `D3Q_FINAL_RESULT.json`：本报告同目录。
- 未产生：`D3Q_REQUEST_LEDGER.jsonl`、`D3Q_SLOT_RESULTS.csv`、`D3Q_PREFLIGHT_RESULTS.json`、`D3Q_ERROR_TAXONOMY.json`、`D3Q_COST_AND_TOKENS.json`（实验未运行，无请求/槽位/preflight 数据）。
- 实验矩阵、GPU2 Preflight、token/成本统计：`not_reached`。

## Git

- branch：`perf/llm-small-large-quality-cost-d3`
- base：`62b7d115b6de6506cb955733beaf1f5b8e79d521`（+ 本 session 3 个修复 commit：`1993495`、`9e872f2`、`a756e7f`）
- 未 push/merge。

## 遗留限制

1. 服务器防火墙阻断 `api.deepseek.com:443`，这是本实验无法从实验环境触达 DeepSeek API 的直接原因。
2. 若要继续 D3Q，需运维放行 `api.deepseek.com:443`（或提供 HTTP 代理），并把凭据仍留在服务器 env。
3. Ollama 14B 臂未启动（无配对对象），GPU0 未受影响、GPU1/GPU3 未触碰。

## 中文结论

尚无充分证据回答"小模型是否因无效任务反而整体更慢"——因为外部模型 `deepseek-v4-flash` 在当前实验服务器上**不可达**（网络阻断），无法构成公平对比。这不是模型质量问题，而是 provider 可达性阻断。
