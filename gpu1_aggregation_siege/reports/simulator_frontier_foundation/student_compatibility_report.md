# Student Compatibility Report · Stage 3/4 只读证据汇总

> 本轮全部为**只读、零参数更新**证据。forward smoke ≠ 性能评估；契约/探针 PASS ≠ 真实闭环。
> 机器可读证据：`student_adapter_matrix.json`、`sota_launch_gate.json`、各候选 JSON。

## 1. 主挂载：PERSISTENT_RMT16_ORIGINAL_VTRACE_98304（R4b PASS）

证据：`student_compatibility_PERSISTENT_RMT16_ORIGINAL_VTRACE_98304.json`（驱动 `scripts/run_student_mount_smoke.py`，exit 0）。

**REAL_CHECKPOINT_LOADED=true**（身份门禁逐项 fail-closed 通过）：
- 文件 SHA256：`2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723`（21,741,728 B，与 Stage 0 复算一致）；
- params 树 SHA256（运行时重算）：`aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d` == profile 期望；fresh-process 子进程复算同值；
- 驱动源 LF-SHA256 门控：`453bd1ecc8d9671c741c4462214bd7699c74611a52ec157ff30cd68653b4bafc`（`_cc2_stage/train_rmt16_p2replay.py`，冻结 cfg 经 AST 恢复，从不执行驱动源）；
- 身份：global_step=98304、total_env_steps=98304、arm=RMT16-Persistent-OrigVtrace、identity_hash=`822f5789619bf9a16b2eea20fefe3c95cd701906399c49d88c20467c67cb14bc`；
- 结构门禁：101 个 param leaves；encoder kernel `(8335,256)`；actor 输出头 `(256,43)`；环境枚举 `len(Action)==43` 交叉验证；memory_spec_hash=`ab03915d783d3a47cdf40335938054e00691f29a1e9ea1c5551de52d4668f69c`。

**REAL_FORWARD_SMOKE_PASS=true**（零更新）：
- batch=1 与 batch=4、deterministic + stochastic（seeded rng 777）全可复现；actions ∈ [0,43)，logits/value 有限；
- memory 推进符合 RMT16 契约：seg_count == min(steps,128)；128 步段边界探针确认 **persistent carry**（seg_count→0 而 mem_tokens 非零）；
- 前后 params 逐位一致（零更新证明）；
- **obs 诚实标注**：真实 MiniCraftaxTrain reset obs = 8268 维 ≠ 训练契约 8335（8268 环境 + 67 multitask embedding 由训练 wrapper 组装）；smoke 使用 STRUCTURED_SYNTHETIC_SEEDED_NORMAL（seed 20260803 正态），报告中记录真实尺寸，从不伪装为真实环境 obs。

**诚实缺口**：optimizer / train RNG / policy memory 在该 CC2 pkl 中 **ABSENT_IN_CHECKPOINT**（pkl 只含 params+manifest）→ R4c 联合证明对这些组件不完整；`COMBINED_FRESH_PROCESS_RESTORE=false`。R4a PASS ∧ R4b PASS ≠ R4c 联合证明。

## 2. 次级探针（20 项只读检查，`probes/student_compatibility.py`）

| candidate | 产物 | 结果 | exit |
|---|---|---|---|
| RESET128_RMT16_ORIGINAL_VTRACE_98304 | 本地 pkl（21,741,720 B） | **PASS 20/20**（同一只读 RMT16 adapter 链；params sha 前缀 `78a14cc6…`，与 PERSISTENT 臂不同身份） | 0 |
| CONTROL_CONTINUOUS_98304 | 本地 orbax TrainState（目录） | ADAPTER_PENDING（4/20 身份+清点，其余 NOT_APPLICABLE；无 GTrXL128 adapter，绝不伪造 PASS） | 5 |
| BASELINE_TEACHER_CKPT17500 | 本地 orbax（目录）；canonical pkl MISSING | ADAPTER_PENDING；仅参照，不入 98304 排名 | 5 |
| SLOWGRU_PERSISTENT_24576 | 仅服务器 | ARTIFACT_HANDOFF_REQUIRED（如实记录交接路径） | 5 |
| BASE_GTRXL_ORIGINAL_VTRACE_98304 | 仅服务器（本地仅 SHA 合同） | ARTIFACT_HANDOFF_REQUIRED（未跑本地探针；无产物永不 PASS） | — |

证据：`probes/student_compatibility_probe_<CANDIDATE>.json`（commit `784762a`）。

## 3. 语义边界声明

- 以上全部为 **R4b checkpoint 侧 / 只读兼容性**证据；与 Stage 1 的 R4a env 侧 PASS 合并**仍不构成** R4c 联合 fresh-process 证明。
- forward smoke 不构成任何性能结论；20 项探针零参数更新。
- 缺失 adapter/产物一律记 ADAPTER_PENDING / ARTIFACT_HANDOFF_REQUIRED，fail-closed，不以 PASS 顶替。
