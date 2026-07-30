# CC4 收口报告(Phase 2)— common evaluator 收口 + 6-student 池可审计化(2026-07-30)

合同:CC4 common evaluator 收口任务 — 把 6-student formal ranking 从 INCOMPLETE
推进到**可审计的 READY 现状**,**不启动正式性能评估**。最终裁定:**INCOMPLETE(2/6 binding PASS)**。

## 边界遵守(§四)

未启动训练 / 正式 ranking / 正式 SR 评估 / 多 episode 性能运行 / 消融 / D052 / GPU 长任务;
未修改任何候选训练产物;未覆盖已有有效胶囊(两个 RMT16 capsule 与 common/ 均保留原样);
未重复生成 artifact;teacher 未纳入 student ranking;无 force push/rebase/amend/merge。
本轮服务器写入仅为:READY marker 追加字段(sums 排除文件,57/57 不变)+ 5 个候选的
pending 审计记录目录(每个 5 文件,各自 sums 3/3)。

## 分支与提交

- 分支:`henry/tier3-scaffolded-evaluation`;本地/服务器 HEAD 一致 `ddd21f883d6194afbd4195b4de1b997cc250b59d`(ff-only 同步)。
- push 仍 BLOCKED_NETWORK(127.0.0.1:443 代理不可达;非代码失败)。

## 三段工作

- **§一 READY marker 补齐**:追加 `negative_test_report_sha256=bc42ee6a…`、`assembly_manifest_sha256=52d1fa0f…`、`full_profile_sha256=2eceb288…`、`FULL_PROFILE_STATUS=FROZEN`、`common_sha256sums_self_check=PASS(57/57)`、`FORMAL_RANKING_AUTHORIZED=false`。追加前全部既有顶层 SHA 在线复验一致,漂移即 fail closed。
- **§二 两个 RMT16 capsule**:上一合同已 READY=true(8/8),本轮只核验保留 — full64 checkpoint/params SHA 全部已复算(`2866b5de…/aa6ba440…`、`de3a159f…/78a14cc6…`),budget_class=MATCHED_98304,steps=98304,seed=42,candidate_class=STUDENT。
- **§三 5 个 pending binding**(4 非-RMT16 student + teacher):每个候选写 `candidate_manifest.json`(审计投影,immutable)+ `common_evaluator_binding_result.json`(§三 全 23 字段)+ `environment_lock.json`(字节拷贝)+ `SHA256SUMS` + `READY.json`(READY=false,8 诚实门)。判定阶梯:PENDING_COMMON_READY → PENDING_FULL_PROFILE → MISSING_EVIDENCE(PASS 构造上不可达:runtime family 未注册 ⇒ 无法做接口 smoke ⇒ 无法按 owner 协议复算 params_sha256)。4/4 单文件 pkl 的 checkpoint_file_sha256 已由 CC4 独立流式复算且与 owner 声明一致;CONTROL orbax 目录哈希按 CC1 协议 BLOCKED,不伪造竞争哈希;teacher `reference_only=true / formal_student_ranking_eligible=false / student_rank=null / budget=UNMATCHED_REFERENCE / steps=17500`。

## 证据

- 服务器侧:common sums 57/57 成功;5 个新目录各自 3/3 成功。
- 本目录:服务器回传 27 文件(排除 npz/pkl)+ `verify_pool_readiness_evidence.py`
  纯门本地复验(与上一轮 closing_evidence 交叉核验:所有 binding 内 common SHA == 上一轮 manifest 恒等;
  supplement 字段从上一轮文件再现):**POOL_READINESS_LOCAL_REVERIFY_PASS**(105 项)。

## 缺口(为什么 INCOMPLETE)

四个 student(base_gtrxl / control / slowgru×2)+ teacher 的 runtime family 均不在
CC4 common ABI 注册集合(目前恰为 `('rmt16_gtrxl_cc2',)`,ABI 文档规定注册权归 owner)。
补齐路径(均需 owner 或授权后进行,不在本轮边界内):owner 注册 runtime family →
CC4 经 common evaluator 跑接口 smoke → 按 owner 协议复算 params_sha256 →
binding 方可转 PASS;CONTROL 另需 CC1 冻结 orbax 文件级 SHA 定义。
`FORMAL_RANKING_AUTHORIZED=false`,直至 STUDENT_COMMON_BINDING_PASS_COUNT=6/6。
