# D3Q Phase-2 Incident 01 — chunk1 r1 budget_exceeded 过度拦截与证据损失

classification: D3Q_PHASE2_INCIDENT
incident_id: D3Q_PHASE2_INCIDENT_01
recorded_utc: 2026-08-16T04:32:00Z
recorded_by: codex-primary（用户 goal 授权直接执行）
severity: 工具故障（未发生真实预算超限；未触碰 GPU0/1/3；无密钥暴露）

## 1. 事件时间线

- 04:15:40Z 启动 chunk1：`python d3q_phase2_driver.py run-chunk --run-id d3q_p2_chunk1_r1_20260816T041540Z --repeat r1`（22 slots）。
- 远端 `/tmp/d3q_exec_20260816T041540Z` 部署通过（deployed_hashes_verified=true），GPU2 UUID 门禁通过，ollama digest 门禁通过。
- 顺序完成 slot_r1_small_p01..p04（各 1 POST），p05 使用 3 POST（预算上限内合法用尽）。
- p05 完成后 launcher 判定 `budget_exceeded`，整 run BLOCKED（rc=3），随后清理远端 exec root（cleanup_verified=true）。
- driver 按 fail-closed 语义拒绝合并：全局 ledger 未变化（仍为 seed 2 行），chunk_not_pass。

## 2. 证据（byte-identical 保留，已提交 git）

- artifact 目录：`d3q_artifacts/d3q_p2_chunk1_r1_20260816T041540Z/`
  - D3Q_SLOT_LAUNCHER_RESULT.json sha256=bf65fca0291735a47193318f9353467459b6dc5e228078cfc3175115ac3b2d2b
  - D3Q_RUN_MANIFEST.json sha256=a7d4d43f28b0b7d7e59d9688bf431b5542b5ea35dee0dddf7e7d9cff732436f5
  - SHA256SUMS sha256=281856a2337ae532e7b421fdace08b979b61c0b3988448d573ac34615a8a0968
- 关键字段：status=BLOCKED、reason=budget_exceeded、ledger_post={ollama:7, slots p01..p04:1, p05:3}、slot_results={}、gpu_pre 正常（GPU2 空闲、UUID 匹配）、cleanup_verified=true。

## 3. 根因（两个叠加的工具缺陷，均在 launcher，不涉及冻结的 runner/budget 模块）

1. **过度拦截**：dispatch loop 在每个 slot 完成后，用同一 slot_id 再次调用 `_enforce_budget_after` 之外还调用了 `_enforce_budget_before_slot(after, spec)`，其条件为 `slot_counts[slot] >= MAX_POSTS_PER_SLOT`。任何合法用尽 3 POST 的 slot 都会因此 BLOCK 整个 run。冻结预算语义是"每 slot ≤3 POST"，用满 3 POST 是合法实验结果（无效性/repair 正是被测对象），不是违规。
2. **证据集中收集**：per-slot 产物在"全部 slot 跑完后"才统一收集到本地。中途 BLOCK 时远端 cleanup 删除 exec root，已完成 slot 的 request metadata、raw response、候选代码全部丢失，只剩 launcher result 中的 ledger_post 计数。

## 4. 真实预算消耗（不可撤销，必须入账）

chunk1 实际消耗 ollama provider 7 POST（真实请求已到达 Ollama）：

| slot | 消耗 POST | 备注 |
|---|---|---|
| slot_r1_small_p01 | 1 | 产物丢失 |
| slot_r1_small_p02 | 1 | 产物丢失 |
| slot_r1_small_p03 | 1 | 产物丢失 |
| slot_r1_small_p04 | 1 | 产物丢失 |
| slot_r1_small_p05 | 3 | 产物丢失，预算用尽 |

## 5. 处置决定：attrition_no_rerun（损耗、不重放）

- **不重放 p01-p04**：冻结 runner（hash 绑定，不可改）在远端使用全新 per-run ledger，无法把它限制在"剩余 2 POST"；重放存在突破全局 slot 预算的真实风险，违反冻结协议，故永久禁止重放这 5 个 slot。
- **p05 预算已用尽**（3/3），协议上不可再发任何 POST。
- 7 POST 通过独立的 `D3Q_BUDGET_RECONCILIATION.json` 入账（不污染只含逐 POST 已验证事件的冻结 ledger 格式），driver 的 provider 预算检查把 reconciliation 计入有效用量。
- r1-small 最终只有 7/12 prompts 有完整结果（p00 seed + p06..p11）。最终报告必须披露该 attrition 及其对 r1 统计功效的影响；任何依赖"r1-small 全 12 prompt"的结论均不得声称。

## 6. 修复项（本 incident 后实施）

1. launcher：移除 dispatch loop 中错误的 post-slot `_enforce_budget_before_slot(after, spec)` 调用；保留真正的超限检查（actual > limit）与下一 slot 的 pre-check。
2. launcher：per-slot 产物在该 slot 完成后立即收集（增量收集），后续任何中途失败不再丢失已完成证据。
3. driver：新增 reconciliation 读取与预算联动（slot 级禁发 + provider 级计数），fail-closed 校验 reconciliation 证据哈希。
4. 新增回归测试：slot 恰好使用 3 POST 必须 PASS；中途失败必须保留已完成 slot 产物。

## 7. 边界确认

- 未发生真实预算超限（slot 最大 3/3，provider 7/108）。
- GPU0/1/3 未触碰；GPU2 仅通过门禁读取；无密钥读取/输出；远端 exec root 已清理并验证。
- 冻结文件（runner/budget/cpu_validate/manifest/repair_template）未修改，hash 绑定不变。
