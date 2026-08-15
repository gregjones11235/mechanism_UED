# D3Q Phase 0 对账与冻结报告

- 角色: 实现 worker（director 已完成只读勘察并委派本 worker 执行 Phase 0）
- 记录 UTC: 2026-08-15T01:11:26Z
- Worktree: `C:\Users\Lenovo\Desktop\dicode-codex-director\skill_preflight_ued_d3_worktree`
- 分支: `perf/llm-small-large-quality-cost-d3`
- HEAD: `88adf4c3b9ef4135897697a6d7c455c80861a221`
- Base: `62b7d115b6de6506cb955733beaf1f5b8e79d521`（merge-base 即 base，为 HEAD 祖先）
- 工作区: 除本 Phase 0 新增目录外无任何其他改动（`git status --porcelain` 仅列出本目录）

## 交付内容
1. `EVIDENCE_INTEGRITY.json` — 36 个受保护文件的 path/sha256/bytes、git 状态、全部 cross_check 逐项 verified 标记。
2. `rerun_gate_SHA256SUMS` — rerun gate 两个 JSON 的标准 sha256sum 清单（补强；未改动原目录）。
3. `REMOTE_PRECHECK_EVIDENCE.json` — director 2026-08-15 远端只读事实原样转录（method=read-only ssh；仅记录环境变量名，无任何值）。
4. `D3Q_MATRIX_BINDING.json` — 72 slot 矩阵绑定、slot id 方案、arm 顺序、预算、preflight、主指标、no-secret 与 artifact 规则。
5. `D3Q_FROZEN_REPAIR_TEMPLATE.json` — 生产 repair 模板逐字节冻结（含模板 sha256 与组装规则）。
6. `TOOL_BINDING.json` — 12 个工具/测试文件当前 sha256。
7. `d3q_budget.py` + `test_d3q_budget.py` — D3Q 共享 POST 预算状态机与强反例测试（10 项全过）。

## Cross-check 结果
| 检查项 | 结果 |
| --- | --- |
| FROZEN_MANIFEST.json `manifest_sha256` 重算（llm_replay_manifest.canonical/fingerprint） | verified=true（2066515499... 一致） |
| `load_manifest(FROZEN_MANIFEST.json)` | PASS |
| rerun gate: launcher `artifact_sha256` == canonical(gate artifact 去 `artifact_sha256` 字段)（任务字面要求） | **verified=false**（见“异常”一节） |
| rerun gate: launcher `artifact_sha256` == launcher 结果自身 canonical self-seal | verified=true（652e5962...） |
| rerun gate: launcher `artifact_internal_sha256` == canonical(gate artifact 去字段) | verified=true（390f9f3b...） |
| rerun gate: artifact 自身 `artifact_sha256` 字段重算 | verified=true |
| rerun gate: launcher `local_artifact_sha256` == artifact 原始文件 sha256 | verified=true（e16f44e0...） |
| rerun gate: pre_execution_sha256 / post_execution_sha256 与 manifest_sha256 一致 | verified=true（provider 94f5de68..., tool 80ac2d92...） |
| rerun gate: manifest sha 与 artifact provenance 绑定 | verified=true |
| 旧 gate (052903Z): launcher self-seal | verified=true（c6e25019...） |
| 旧 gate SHA256SUMS 条目与重算 raw sha256 一致 | verified=true |
| D3Q_FROZEN_REPAIR_TEMPLATE 模板文本与生产模块 `context` 逐字节一致 | verified=true（sha256 beff6ea4...，48686 字节） |

## 异常
- **唯一异常（已澄清，非篡改）**：任务说明中“launcher result 的 `artifact_sha256` == 对 gate artifact 去 `artifact_sha256` 字段后的 canonical sha256”字面不成立。核对 `d3_deepseek_gate_launcher.py`（`_seal_result` / `verify_launcher_result`，初始 result 模板不含 `artifact_sha256` 键）后确认：launcher 结果的 `artifact_sha256` 是 **launcher 结果自身**的 canonical self-seal；gate artifact 的 canonical 内部哈希记录在 `artifact_internal_sha256`。两者独立验证均通过，pre/post_execution_sha256 与 manifest_sha256 一致，不存在完整性破坏。EVIDENCE_INTEGRITY.json 中按任务字面要求记录 verified=false 并附解释，同时记录正确的 self-seal 语义为 verified=true。
- 受保护文件（含两个旧 artifact 目录）哈希与字节数均已记录，未改动任何既有文件字节。

## 测试摘要
命令（工作目录 = D 目录）：`python -m pytest test_d3q_budget.py -p no:cacheprovider -v`
环境: Python 3.12.4 / pytest 9.1.1

- 结果: `10 passed in 0.31s`
- 覆盖: slot 第 4 次 POST 被拒（即使 provider 预算充足）；provider 第 109 次被拒；kind 混合（initial+transport_retry+semantic_repair=3 后拒绝）；ledger resume 后预算状态一致；非法 kind 拒绝且不落盘；被拒 reserve 不写入 ledger；损坏 ledger fail-closed；超限 ledger fail-closed。

## 结论
Phase 0 交付物齐备。除上述已澄清的 launcher self-seal 语义外无任何异常。未开始 Phase 1 实现。