# 正式全局性能评估 Runbook（V2_DYNAMIC_TOPOLOGY，6 student + teacher reference）

日期：2026-07-31　责任人：CC4（统一评测）　分支：`henry/tier3-scaffolded-evaluation`　worktree：`D:/cc4tmp`

## 0. 授权与硬约束

二级审计结论逐字：`SECONDARY_AUDIT_VERDICT=PASS_FOR_FORMAL_GLOBAL_PERFORMANCE_EVALUATION_START`。
依据 8 条：COMMON_EVALUATOR_V2_READY=true；STUDENT_COMMON_BINDING_PASS_COUNT=6/6；TEACHER_REFERENCE_BINDING=PASS；gate_failures=[]；verify_v2dt_closing_evidence.py 164/164 PASS；CHECKPOINTS_MODIFIED=false；CONTROL_RETRAINED=false；CANDIDATE_EXCEPTION_USED=false。

总控硬约束（逐字执行）：
1. 使用 V2_DYNAMIC_TOPOLOGY common evaluator；
2. 使用同一套 FRONT/BACK/FULL frozen banks；
3. 评估 6 个 student，teacher 只做 reference，不进入 student ranking；
4. 不改 checkpoint；5. 不重训；
6. 不把 interface smoke 当性能结果；
7. 输出正式 evaluation certificates、ranking summary、per-student metrics；
8. 启动前记录 SECONDARY_AUDIT_PASS marker 或等价审计记录。

## 0.1 GPU 解禁记录（总控裁定，2026-07-31）

彩排外推超单队列一夜窗口后，总控显式裁定：**解禁 GPU0/GPU1 用于本次正式评估，四卡并行（墙钟约 8–15h）**。执行约束：
- 每 GPU 仍为**单进程顺序队列**（同卡并发 jax 仍是已证实的 CUDA OOM 类，禁止）；
- GPU0（`GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6`）/ GPU1（`GPU-3c7a2864-755b-7045-b293-6f80e748283f`）正式运行前**必须先过跨 GPU 确定性预检**（metric_schema bit_agreement_policy 的 canonical 字段逐位相同：action_sequence / terminal_label / timesteps / valid_start / primary 与 dense 载荷值 / episode_record_sha256 / episode_records_sha256 / bank_content_sha256 / checkpoint_file_sha256 / params_sha256）；
- 注册表 `CC4_GPU_ALLOWED_UUIDS` 已扩为 4 卡（提交可审计，历史禁令注释保留）；本记录与最终报告如实披露。
- 其余 standing 纪律不变：cc1/cc1_retrain/cc2/cc3 只读；孤儿 PID 106885 不触碰；密钥纪律不变。

## 1. 设计偏差记录（重要）

原计划：marker 工具 = 写 marker + flip READY(`FORMAL_RANKING_STARTED=true`) → 正式运行 → 排名。
实测冲突：冻结的 V2 公共验证器 `smokev2.verify_engine_and_common_v2`（步骤 1e，已随 b736c8c 证据冻结，**不可改**）要求 `FORMAL_RANKING_STARTED is False`；若启动前 flip，每个正式运行都会在 stage 1 即 fail-close。
采用方案（单一 READY 写入者、零竞争）：
- **marker 工具（`tier3_formal_start_marker_v2dt.py`）只写 marker**（`<pool>/cc4/SECONDARY_AUDIT_PASS.json` + `.sha256`，拒覆盖）；
- **正式驱动 stage 1b 校验 marker**（文件存在、sidecar SHA 一致、verdict 逐字相等、绑定门 SHA == `cec16711…`、READY 仍为 `FORMAL_RANKING_STARTED=false`）；marker + 逐候选正式证书即启动记录；
- **排名工具（`tier3_formal_ranking_v2dt.py`）在收口时执行唯一一次 READY 白名单 RMW**（`FORMAL_RANKING_STARTED=true`、`FORMAL_RANKING_PUBLISHED=true`、started-at 取自 marker、summary/gate SHA、marker 引用、pending gate 退役）。flip 之后驱动对任何重跑 fail-close（正式运行不可静默重复）。

## 2. 新工具清单（全部 `tools/tier3_scaffolded_evaluation/`，未改任何冻结文件）

| 工具 | 职责 | 自检 |
|---|---|---|
| `tier3_evaluation_certificate_v2dt.py` | 注册表形正式证书（值绑定，rank=null，诚实标签，禁夸扫描） | `--self-test`（JAX-free，111 checks） |
| `tier3_formal_evaluation_v2dt.py` | 单候选正式驱动：stage 0–6 同 smoke + stage 7 = 冻结 performance schedule（FULL 64 / FRONT 8 / BACK 8，max_steps=4096，FULL→FRONT→BACK） | `--self-test`（server venv，含 JAX） |
| `tier3_formal_ranking_v2dt.py` | 7 份 bundle 验证 + 冻结规则排名（tol 1e-12，四级全平→INCONCLUSIVE，teacher rank=null）+ 收口门 + READY flip | `--self-test`（JAX-free） |
| `tier3_formal_start_marker_v2dt.py` | 启动前审计 marker（只写 marker，拒覆盖） | `--self-test`（JAX-free） |

## 3. 前置清单（服务器，逐项核验后再动手）

- [ ] 服务器 repo HEAD == bundle 同步后的 C4 HEAD（`git -C /home/oseasy/cc4_tier3_eval_20260730/repo rev-parse HEAD`）；
- [ ] venv 可用：`/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python -V`；
- [ ] 四工具 `--self-test` 全绿（驱动自检需 venv）；
- [ ] `nvidia-smi`：GPU2 `GPU-8df11537-ab79-722d-606f-411966196c4c` / GPU3 `GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd` 空闲；GPU0/GPU1 禁用；
- [ ] `<pool>/common_v2/COMMON_EVALUATOR_V2_READY.json`：`COMMON_EVALUATOR_V2_READY=true` 且 `FORMAL_RANKING_STARTED=false`；
- [ ] `<pool>/cc4/POOL_BINDING_GATE_V2DT.json` SHA == `cec167117a7aa8e67a3d5eb60839e711e72d950135553e4035a87e6c9859a352`；
- [ ] 孤儿进程 PID 106885 不触碰；cc1/cc1_retrain/cc2/cc3 目录只读。

## 4. 彩排（marker 之前，最轻候选 CONTROL_CONTINUOUS）

```bash
cd /home/oseasy/cc4_tier3_eval_20260730/repo
TS=$(date -u +%Y%m%dT%H%M%SZ)
CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python \
  tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v2dt.py \
  --candidate-id CONTROL_CONTINUOUS_98304 \
  --rehearsal-scratch /home/oseasy/student_pool_v1/cc4/_rehearsal_$TS \
  --limit-full 2 --limit-front 2 --limit-back 2 < /dev/null
```

验收：退出码 0；READY_FORMAL_V2DT.json 中 `rehearsal=true`、`evaluation_status=REHEARSAL_NOT_FORMAL`、门 G4_REHEARSAL_SCHEDULE_EXECUTED / G11_REHEARSAL_LIMITS_RESPECTED 为真；证书 status `REHEARSAL_NOT_FORMAL`。记录单 episode 墙钟、VRAM（外部 nvidia-smi）、RSS 平稳性；按 64/8/8 外推单队列总墙钟。**若外推超过单队列一夜 → 升级总控（仅调调度，不改范围）**。彩排目录为诊断用，**不提交**；验收后 `mv` 到 `_failed_*` 或直接删除。

彩排不得触碰正式输出目录 `<pool>/cc4/<ID>/formal_evaluation_v2dt`（驱动对无 scratch 的 limit 参数 fail-close）。

## 5. 记录审计 marker（彩排验收后、正式启动前）

```bash
cd /home/oseasy/cc4_tier3_eval_20260730/repo
/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python \
  tools/tier3_scaffolded_evaluation/tier3_formal_start_marker_v2dt.py \
  --pool-cc4-dir /home/oseasy/student_pool_v1/cc4 \
  --common-dir /home/oseasy/student_pool_v1/common_v2 < /dev/null
```

立即回拉本地证据副本：`SECONDARY_AUDIT_PASS.json` + `.sha256`。marker 拒覆盖——写错只能换名重审。

## 5.1 跨 GPU 确定性预检（GPU0/GPU1 首跑前必过）

CONTROL_CONTINUOUS 以 1/1/1 受限彩排分别在 GPU0、GPU1 上跑（新 scratch 目录），与既有 GPU2 彩排记录（`_rehearsal_20260731T092931Z`）比对 canonical 字段：`full-seed200000`、`front_l2-bank0`、`back_l2-bank0` 三条 episode 的 action_sequence / terminal_label / timesteps / episode_record_sha256 必须逐位相同。任一不符 → CROSS_GPU_DETERMINISM_PREFLIGHT=FAIL，GPU0/1 不启用（回退双卡队列），不得重铸银行规避差异。

## 6. 正式启动：四 GPU 顺序队列（总控 2026-07-31 裁定；开跑后不改）

同 GPU 上并发 jax 进程已证实导致 CUDA stream/cuSolver/CUBIN-OOM，故每 GPU 一条顺序队列、nohup、`< /dev/null`、CWD=repo root。RMT16 对拆分（GPU2/GPU3）、SlowGRU 对拆分（GPU3/GPU0）、重者先行、teacher 独占 GPU1（reference 同协议全量）。

| GPU2（`GPU-8df11537…`） | GPU3（`GPU-f56a59b4…`） | GPU0（`GPU-e8c08612…`） | GPU1（`GPU-3c7a2864…`） |
|---|---|---|---|
| PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | RESET128_RMT16_ORIGINAL_VTRACE_98304 | SLOWGRU_RESET128_CANONICAL_98304 | BASELINE_TEACHER_CKPT17500（reference） |
| BASE_GTRXL_ORIGINAL_VTRACE_98304 | SLOWGRU_PERSISTENT_CANONICAL_98304 | CONTROL_CONTINUOUS_98304 | |

```bash
mkdir -p /home/oseasy/student_pool_v1/cc4/formal_eval_logs
L=/home/oseasy/student_pool_v1/cc4/formal_eval_logs

nohup bash -c '
  cd /home/oseasy/cc4_tier3_eval_20260730/repo
  PY=/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python
  DR=tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v2dt.py
  L=/home/oseasy/student_pool_v1/cc4/formal_eval_logs
  for C in PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 \
           BASE_GTRXL_ORIGINAL_VTRACE_98304; do
    CUDA_VISIBLE_DEVICES=GPU-8df11537-ab79-722d-606f-411966196c4c \
      $PY $DR --candidate-id $C < /dev/null > $L/$C.log 2>&1
  done' > $L/_queue_gpu2.nohup 2>&1 &

nohup bash -c '
  cd /home/oseasy/cc4_tier3_eval_20260730/repo
  PY=/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python
  DR=tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v2dt.py
  L=/home/oseasy/student_pool_v1/cc4/formal_eval_logs
  for C in RESET128_RMT16_ORIGINAL_VTRACE_98304 \
           SLOWGRU_PERSISTENT_CANONICAL_98304; do
    CUDA_VISIBLE_DEVICES=GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd \
      $PY $DR --candidate-id $C < /dev/null > $L/$C.log 2>&1
  done' > $L/_queue_gpu3.nohup 2>&1 &

nohup bash -c '
  cd /home/oseasy/cc4_tier3_eval_20260730/repo
  PY=/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python
  DR=tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v2dt.py
  L=/home/oseasy/student_pool_v1/cc4/formal_eval_logs
  for C in SLOWGRU_RESET128_CANONICAL_98304 \
           CONTROL_CONTINUOUS_98304; do
    CUDA_VISIBLE_DEVICES=GPU-e8c08612-c22a-c8a4-6df5-affb2dd1f9a6 \
      $PY $DR --candidate-id $C < /dev/null > $L/$C.log 2>&1
  done' > $L/_queue_gpu0.nohup 2>&1 &

nohup bash -c '
  cd /home/oseasy/cc4_tier3_eval_20260730/repo
  PY=/home/oseasy/cc4_tier3_eval_20260730/venv/bin/python
  DR=tools/tier3_scaffolded_evaluation/tier3_formal_evaluation_v2dt.py
  L=/home/oseasy/student_pool_v1/cc4/formal_eval_logs
  CUDA_VISIBLE_DEVICES=GPU-3c7a2864-755b-7045-b293-6f80e748283f \
    $PY $DR --candidate-id BASELINE_TEACHER_CKPT17500 < /dev/null \
    > $L/BASELINE_TEACHER_CKPT17500.log 2>&1' > $L/_queue_gpu1.nohup 2>&1 &
```

## 7. 监控

- 逐候选日志：`<pool>/cc4/formal_eval_logs/<ID>.log`（驱动逐 episode 打印 steps/defeat/died/transition/progress/wall）；
- 每候选 READY/证书落盘于 `<pool>/cc4/<ID>/formal_evaluation_v2dt/`；
- VRAM/RSS：外部 `nvidia-smi`（间隔采样）+ 驱动 provenance 的 peak_rss_kb / 逐 episode wall；
- 异常判据：VRAM 单调增长、RSS 漂移、单 episode 墙钟突增 >3σ、进程消失 → 停该队列诊断。

## 8. 失败恢复（不升级即自动执行）

- **基础设施失败（GPU/CUDA/OOM/进程崩溃）**：`mv <pool>/cc4/<ID>/formal_evaluation_v2dt <pool>/cc4/<ID>/_failed_formal_evaluation_v2dt_<UTC>`（诊断用，**不提交**），同候选重跑（`assert_output_dir_fresh` 新鲜门因目录移走而满足）。
- **引擎 FailClosed 中止 = 永久 BLOCKED**（V2 下合法挖矿不会触发；剩余 FailClosed 仅为腐败类）：驱动写结构化 `formal_abort`，候选证书/READY 记 `BLOCKED`；**禁重训、禁候选级豁免、禁记为成绩**。排名仅对 ELIGIBLE_COMPLETE；<6 完成 → 汇总 `INCONCLUSIVE_PARTICIPATION` 并升级总控。
- 任何情况不伪造 PASS。

## 9. 收口（7 份全部落盘后）

```bash
# 先干跑核验（不写任何文件、不 flip）
$PY tools/tier3_scaffolded_evaluation/tier3_formal_ranking_v2dt.py \
    --pool-cc4-dir /home/oseasy/student_pool_v1/cc4 \
    --common-dir /home/oseasy/student_pool_v1/common_v2 --dry-run < /dev/null
# 全绿后正式收口（写 summary+gate，唯一一次 READY flip）
$PY tools/tier3_scaffolded_evaluation/tier3_formal_ranking_v2dt.py \
    --pool-cc4-dir /home/oseasy/student_pool_v1/cc4 \
    --common-dir /home/oseasy/student_pool_v1/common_v2 < /dev/null
```

随后：证据回传（7 份 formal 目录 + summary/gate + READY + marker + 日志摘要；npz/pkl 永不出服务器）→ 离线复验器（独立重算排名）全绿 → C5 证据提交（路径限定；tgz/tmp_* 保持未跟踪）→ 固定格式中文汇报（`scientific_claim_authorized=false`、scaffolded≠full task、teacher reference 不入排名）。

## 10. Git 纪律

- 提交路径限定，`git status --porcelain` 仪式逐次执行；`reports/**/*.tgz`、`tmp_*` 目录保持 `??`，push 前逐项确认归档范围；
- 禁 merge/rebase/amend/force push/reset --hard/git clean/`git add .`；
- 提交尾行：`Co-Authored-By: Claude <noreply@anthropic.com>`。
