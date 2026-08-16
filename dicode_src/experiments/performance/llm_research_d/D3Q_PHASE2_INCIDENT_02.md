# D3Q Phase-2 Incident 02 — chunk3 r3 完成后 GPU2 外部进程门禁拦截（环境干扰，证据完整）

classification: D3Q_PHASE2_INCIDENT
incident_id: D3Q_PHASE2_INCIDENT_02
recorded_utc: 2026-08-16T06:22:00Z
recorded_by: codex-primary（用户 goal 授权直接执行）
severity: 环境干扰（实验计算未受影响；证据完整）

## 1. 事件经过

- 05:41:59Z 启动 chunk3：`run-chunk --run-id d3q_p2_chunk3_r3_20260816T054159Z --repeat r3`（24 slots，small 先行）。
- 24 个 slot 全部正常完成（12 small + 12 large），增量证据收集全部落盘（slots/ 24 目录，slot_results 24 项）。
- post-run GPU 门禁发现 GPU2 出现外部 compute PID 3269422（34590 MiB，util 93%），launcher 判 `gpu2_external_app`，整 run BLOCKED（rc=3）。
- 关键点：slot 生成阶段不使用 GPU（LLM 生成 + CPU JAX 验证），GPU2 外部进程未参与任何本 chunk 的计算。

## 2. 外部进程身份（只读识别）

- PID 3269422（GPU2）：`/home/oseasy/venvs/skill_preflight_e0e1/bin/python ... perf48_async_pipeline_harness.py --component A`（perf48 异步流水线并行组件，属服务器另一工作流）。门禁时刻后已退出；06:20Z 复核 GPU2 = 1 MiB / 0%（空闲）。
- PID 3268245（GPU3）：同一 harness 的 component B（--required-gpu-uuid GPU-f56a59b4...），仅在 GPU3 运行，未触碰 GPU2。GPU1/GPU0 未受影响。
- 未对任何外部进程执行干预。

## 3. 证据

- artifact：`d3q_artifacts/d3q_p2_chunk3_r3_20260816T054159Z/`（BLOCKED 发布，含完整 slots/ 24 目录与 SHA256SUMS）
- D3Q_SLOT_LAUNCHER_RESULT.json sha256=0f3d00a0f53d75d496d5a81ea538f0e68378c641bd386e13d5ca22674f923e5f
- status=BLOCKED、reason=gpu2_external_app、slot_results=24、cleanup_verified=true、gpu_pre 干净（GPU2 1 MiB）。

## 4. 处置决定：completed_chunk_recovery（完整证据回收）

- 该 chunk 的全部 POST 真实发生且逐 POST 元数据完整（classification D3Q_REQUEST_METADATA + ledger_event + result 链），若不回收将导致全局 ledger 与实际预算永久不符，且 r3 repeat 不可重建（预算不可重放）。
- 回收仅允许于：launcher 状态 BLOCKED 且 reason 属于 post-run 环境门禁集合（gpu2_external_app / ollama_pid_changed / ollama_digest_changed），且 chunk SHA256SUMS 全量复算一致、所有 slot 元数据链通过 merge 校验。budget_exceeded / artifact_tamper / no_secret_scan_failed 等一律禁止回收。
- 回收通过 driver 的 `recover-completed-chunk` 命令执行，产出 D3Q_CHUNK_RECOVERY.json 与本 incident 绑定；ledger 正常合并（非对账占位）。
- GPU2 的外部进程干扰不影响本 chunk 数值（无 GPU 计算参与）；对后续 preflight 阶段的影响：preflight 每 arm 前置 GPU 门禁，若再现外部进程则停该 arm 并择时重跑该 arm（replay 在确定性 XLA 下可幂等重放，不消耗 LLM 预算）。

## 5. 边界确认

- 未触碰 GPU0/1/3；未干预外部进程；无密钥读取/输出；远端 exec root 已由 launcher 清理并验证（cleanup_verified=true）。
