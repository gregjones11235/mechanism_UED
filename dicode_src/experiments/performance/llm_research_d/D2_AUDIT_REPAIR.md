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

6. **首次 repair probe 的退出状态披露**：首次且唯一一次真实 repair probe 已写出 `D2_PROVIDER_PROBE.json`，但在 post-write CLI 摘要阶段访问不存在的顶层 `probe['local_model_available']`，发生 `KeyError` 并以非零状态退出。因此不能把首次 probe 描述为成功完成；已写出的 artifact 可独立复算且保持 byte-identical。

7. **离线工具修复**：CLI 摘要改从 `decision_inputs.local_model_available` 读取；DeepInfra target 使用 `deepinfra.yaml`，localhost target 独立使用 `local_gen.yaml` 的 FP8 model ID；连接拒绝、超时、HTTP 错误和 invalid JSON 均显式分类。修复只使用 fake/offline 测试，**未执行第二次真实 provider probe**，没有网络、API、LLM 或 GPU 调用。

8. **持久路径与 provenance**：`D2_RESULT.json` 的 `provider_probe_path` 改为仓库内可读取的持久相对路径；原始 `/tmp/llm_repair_probe_VFAed5/out/D2_PROVIDER_PROBE.json` 保留在 provenance 中，并标明沙箱已清理、原路径已不存在。

9. **终态证据绑定**：新增 `D2_EVIDENCE_FINAL.json`，绑定原 probe 的 probe UTC、internal canonical SHA、file SHA，旧/新工具 file SHA，首次 post-write failure，以及旧/新 `D2_RESULT.json` 的 internal/file SHA。

## 最终结论

`D2_BLOCKED_EXTERNAL_PROVIDER_EVIDENCE_COMPLETE`

含义：**阻塞证据完整**。不代表 D2 benchmark 已运行，不代表 235B 与 14B 存在任何速度或质量结论，不进入组合优化。

## 关键 SHA

- `D2_PROVIDER_PROBE.json` internal canonical `artifact_sha256`：`58bdb6e90ae5cd05c46d0562744278943b1e5afb76292cae8df5a3d5e7bcbfb6`
- `D2_PROVIDER_PROBE.json` raw file SHA256：`e3746fb814f2ac64fa96744b54902d620e6175dac41ec776d27d225351b268fc`
- 原 `D2_RESULT.json` internal canonical `artifact_sha256`：`634b2150b3d496796ed94101f69e2e2b22038e38904dc6439ae3cebaa0b2fc17`
- 原 `D2_RESULT.json` raw file SHA256：`f872e6086120d65ed50c82a8ccb974cb0f5b6b24c101529efbcd561200e6e795`
- 修复后 `D2_RESULT.json` internal canonical `artifact_sha256`：`f0e9b453e64582e583019b0259d0e6a7a8bb4708649ca35bda979a22a5cc9b1e`
- 修复后 `D2_RESULT.json` raw file SHA256：`ef1054fd596d1ff7afc93c13b48f9067785532897a274462e7a50104f56fa362`
- `d2_provider_probe.py` 修复前/后 raw file SHA256：`7e155a1d4cc1c876a2985b82b5a4e86c633f42770499b895f959790dbf63c400` / `0f67a7aea730c0b0e4a660bb729be67afe259fbe85d8c0b3e5e59ed2c02d5c09`
