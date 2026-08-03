# Matched Control Plan · 98304 高能力 Student 配对（方向三）

> 状态：本轮为零执行的**接口级计划**。所有候选仅做只读探针/挂载，未跑任何对照 rollout，未做任何训练更新。
> 任何真实对照运行都需要：总控授权 + R4c 联合门禁放行 + 对应 adapter 就位。

## 1. 目的

在**同一环境步预算（98304）**下，对高能力 Student 做配对受控比较，回答「在等预算条件下，记忆机制（persistent vs reset-128 vs 连续对照）对前沿可达性的影响」。对照必须 matched：除被研究的机制变量外，其余条件逐项对齐。

## 2. 配对矩阵

| 角色 | candidate_id | 架构 | 记忆机制 | 本地产物 | 本轮状态 |
|---|---|---|---|---|---|
| 主研究臂（persistent） | PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | RMT16 | persistent carry | 本地 pkl（21,741,728 B） | 真实只读挂载 PASS（R4b） |
| 次研究臂（reset-128） | RESET128_RMT16_ORIGINAL_VTRACE_98304 | RMT16 | 每 128 步 reset | 本地 pkl（21,741,720 B） | 探针 20/20 PASS |
| 匹配对照（等预算） | CONTROL_CONTINUOUS_98304 | GTrXL128 | continuous | 本地 orbax TrainState | ADAPTER_PENDING（无 GTrXL128 adapter） |
| 原始基线 | BASE_GTRXL_ORIGINAL_VTRACE_98304 | GTrXL128 | continuous | 仅服务器 | ARTIFACT_HANDOFF_REQUIRED |
| 教师参照 | BASELINE_TEACHER_CKPT17500 | TEACHER_REFERENCE | — | orbax 在本地；canonical pkl MISSING | ADAPTER_PENDING；**不入 98304 排名** |
| 兼容性候选 | SLOWGRU_PERSISTENT_24576 | SLOWGRU | persistent | 仅服务器 | ARTIFACT_HANDOFF_REQUIRED |

## 3. 匹配维度（对齐项）

- **环境步预算**：三主候选均为 98304 步训练产物（teacher ckpt17500 仅作参照）。
- **环境**：同一 MiniCraftaxTrain（survive 任务族）与同一 EnvParams 构造约定（构造与步进共用同一 `EnvParams(max_timesteps=K)`）。
- **reset 协议**：standard-reset 采集一律走 `STANDARD_RESET` 协议字段，锚点协议由总控共享 manifest 冻结。
- **评估接口**：统一经 `StudentAdapter` 协议（obs=8335 / action=43 身份门禁 + memory_spec），不各自私设评估口径。
- **provenance**：所有 Frontier 采集 rollout 携带 `TRAINING_DISCOVERY`，与冻结正式评估 bank/worlds 结构性隔离。

**被研究的机制变量（唯一不对齐项）**：记忆机制（persistent carry / reset-128 / continuous GTrXL128）。

## 4. 对照如何执行（未来轮次；本轮不执行）

1. **前置**：CONTROL_CONTINUOUS_98304 的 GTrXL128 只读 adapter 就位并通过同一 20 项探针；BASE_GTRXL 完成 server→local artifact handoff。
2. **R4c 联合门禁**：每次对照 rollout 起点必须由单一 fresh process 联合恢复（params/optimizer/step/RNG + EnvState + env RNG + wrapper state + memory/history）并交叉核验；env-only ∨ ckpt-only 单边 PASS 不放行。
3. **采集**：R1 standard-reset rollout（TRAINING_DISCOVERY）→ R2 关键状态捕获 → R3 Frontier Archive（StateBundle 全字段 + 身份绑定）。
4. **搜索与统计**：R5 actual-N 多分支（实测 N_actual，禁 best-of-N 外推）→ R6 Feasibility Statistics（success rate(N_actual)/progress/Wilson CI/cost/failure category）。
5. **报告**：配对结果只报聚合统计量；成功动作/路线/waypoint/logits 永不进入报告的课程侧输入。

## 5. 诚实缺口（本轮）

- CONTROL/TEACHER 无 adapter：探针只记录 ADAPTER_PENDING，绝不伪造 PASS。
- SLOWGRU / BASE_GTRXL 无本地产物：ARTIFACT_HANDOFF_REQUIRED。
- TEACHER canonical pkl 本地 MISSING：如实记录，不寻找「最像的」替代。
- R4c 联合执行待 Phase 2（optimizer/train RNG/policy memory/history 还原）→ `COMBINED_FRESH_PROCESS_RESTORE=false`。

证据路径见 `student_adapter_matrix.json` 与 `probes/` 下各候选 JSON。
