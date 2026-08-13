# D2 审计修复说明

分类：`D2_AUDIT_REPAIR`。本次为证据修复，不是新的性能实验，不运行 235B 推理，不调用 chat/embedding API，不触碰 GPU。

## 修复内容

1. **D2 独立结果文件补齐**：新增 `D2_PROVIDER_PROBE.json` 与 `D2_RESULT.json`，使"235B provider 不可用"成为可独立复算的证据（此前仅存在于对话日志）。

2. **provider 探测工具**：新增 `d2_provider_probe.py`，只读读取 provider 配置、检查 `DEEPINFRA_API_KEY` 是否导出（仅布尔值）、只做一次 localhost:5000 metadata health 与 Ollama models 元数据检查，不发送任何模型推理、不访问 DeepInfra 外部 endpoint。

3. **凭据措辞收窄**：把"无凭据"精确为"指定非交互 shell 中未发现已导出的 `DEEPINFRA_API_KEY`"，不扩大为服务器所有位置均无凭据。

4. **D2 结论措辞收窄**：明确写 `D2 provider availability gate blocked`（阻塞于可用性门禁），不写"D2 实验完成"或"D2 benchmark 已运行"。

5. **D1c 证据边界修复**：
   - `run_d1c.py`：`ollama_pid_after` 原在 await 前求值，改为 getter 在 await 返回/异常后采样。
   - `d1c_harness.validate_embedding`：强制 `len(embedding) == configured embedding_size`，维度不匹配 fail-closed。
   - `D1C_FINAL_REPORT.md`：H1 收窄为 `NOT_OBSERVED_AT_IDLE_LE_120S`，H2 改为 `NOT_TESTED`，H3 改为 `NOT_OBSERVED_UNDER_TESTED_WORKLOAD`。

## 最终结论

`D2_BLOCKED_EXTERNAL_PROVIDER_EVIDENCE_COMPLETE`

含义：**阻塞证据完整**。不代表 D2 benchmark 已运行，不代表 235B 与 14B 存在任何速度或质量结论，不进入组合优化。

## 关键 SHA

- `D2_PROVIDER_PROBE.json` artifact_sha256：`58bdb6e90ae5cd05c46d0562744278943b1e5afb76292cae8df5a3d5e7bcbfb6`
- `D2_RESULT.json` artifact_sha256：`634b2150b3d496796ed94101f69e2e2b22038e38904dc6439ae3cebaa0b2fc17`
