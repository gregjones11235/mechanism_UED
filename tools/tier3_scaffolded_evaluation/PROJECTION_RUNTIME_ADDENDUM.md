# PROJECTION RUNTIME ADDENDUM — CC4 common ABI 非-RMT 投影族登记(条件 C3)

合同:NON_RMT_RUNTIME_ABI_BINDING_CLOSURE §一(C3)。本文件是 `candidate_runtime_abi.md`
注册制的**补充登记**(不改 ABI 文档原文、不改 `common_runner.py`、不改任何冻结文件):
ABI 文档 §1.2 的"Base GTrXL / Control / SlowGRU / Teacher 由各自 owner 注册(本轮不实现)"
是上一轮的分工声明;本合同 §一 显式授权 CC4 在**不改 owner runtime 语义**的前提下
直接进行最小注册。注册权裁定的完整审计证据见
`reports/tier3_scaffolded_evaluation/non_rmt_abi_binding_closure_20260731/NON_RMT_RUNTIME_REGISTRATION_AUTHORITY_AUDIT.md`
(机器可读版:同目录 `non_rmt_registration_authority.json`)。

```
NON_RMT_RUNTIME_REGISTRATION_AUTHORITY = CC4_CAN_REGISTER_PROJECTIONS (conditional)
OWNER_ACTION_REQUIRED = false
```

## 登记的五族(代码正本:`tier3_projection_runtime.py` PROJECTION_REGISTRY)

| candidate_id | runtime_family | owner | 类别 | 绑定面(owner 文件,只读,SHA-固定) |
|---|---|---|---|---|
| BASE_GTRXL_ORIGINAL_VTRACE_98304 | `base_gtrxl_cc2_projection` | CC2 | STUDENT | cc2 capsule `candidate_runtime.py` `31e28eb6…` + frozen_modules `network_rmt16.py b5c37d7a…` / `rmt_memory_anchor.py 4ff54fb4…` / `rmt16_memory.py 17e1a614…` |
| CONTROL_CONTINUOUS_98304 | `gtrxl128_cc1_control_projection` | CC1 | STUDENT | cc1 capsule `candidate_runtime.py` `ed1a5c3f…` + `gtrxl128_reference_runtime.py d3d4e552…` + `dicode/network.py 172e1cd4…`(== CC1 声明 policy_source) |
| SLOWGRU_RESET128_CANONICAL_98304 | `slowgru_reset128_cc3_projection` | CC3 | STUDENT | cc3 capsule `candidate_runtime.py` `e3fcd9a6…` + `slowgru_runtime.py d3b74d2e…` + arm `slowgru_network.py b2652105…` |
| SLOWGRU_PERSISTENT_CANONICAL_98304 | `slowgru_persistent_cc3_projection` | CC3 | STUDENT | cc3 capsule `candidate_runtime.py` `a450029c…` + `slowgru_runtime.py d3b74d2e…` + arm `slowgru_network.py b2652105…` |
| BASELINE_TEACHER_CKPT17500 | `gtrxl128_cc1_teacher_reference_projection` | CC1 | TEACHER_REFERENCE | cc1 capsule `candidate_runtime.py` `ed1a5c3f…`(与 control 字节同一)+ `gtrxl128_reference_runtime.py d3d4e552…` |

全部 full64 及胶囊四文件 SHA(`candidate_runtime.py` / `candidate_manifest.json` /
`checkpoint_contract.json` / `evaluate_candidate.py`)以 PROJECTION_REGISTRY 为准,
均来自 owner 胶囊 SHA256SUMS / 合同 / READY / interface-smoke 记录(只读审计于
2026-07-31,服务器在盘复验一致)。

## 条件 C1–C5(每份 binding 强制)

- **C1 冻结零改动**:common/ 57 文件 + `tier3_evaluator.py` `54ae18db…` +
  `tier3_candidate_runtime.py` `6af09be4…` + `common_runner.py` + ABI 文档
  `61e52af6…` 字节不变;注册物 = CC4 新增文件(本目录
  `tier3_projection_runtime.py` / `tier3_projection_binding_smoke.py` + 服务器
  `cc4/<ID>/projection_binding_v2/` 证据)。驱动在任一 binding 前复验 common
  sums 57/57 与全部引擎模块 LF-SHA,漂移即 fail closed。
- **C2 零重实现**:projection 只写三件事——(a) 引擎 policy 协议壳
  (`reset()` / `__call__(obs, env_state) -> int`,与 `tier3_cc2_policy_adapter.py`
  同协议);(b) greedy_argmax 取用;(c) boundary 调度与证据。网络前向、内存机构、
  checkpoint 装载、哈希定义**全部调用 owner 自己的模块**:
  - CC2:`Candidate.init_memory/policy_step/reset_memory`(policy_step 内置
    argmax,greedy 来自 owner);params 复算 = `candidate_runtime.canonical_params_sha`。
  - CC1:模块级 `load_candidate/policy_step(greedy=True → pi.mode())`;params 复算 =
    `gtrxl128_reference_runtime.params_sha256`;CONTROL orbax 目录哈希 =
    `R.dir_sha256`(CC1 协议,CC4 不定义任何竞争目录哈希)。
  - CC3:`load_candidate`(三重 fail-closed 门 + `CARRY_MODE_MISMATCH`)/
    `policy_step` / `on_segment_boundary`;params 复算 = `slowgru_runtime.params_sha`。
- **C3 本文档**。
- **C4 诚实标签**:所有 v2 binding `run_class=INTERFACE_SMOKE`、
  `performance_claim_authorized=false`;正式冻结尺度 FRONT/BACK/FULL = **8/8/64**
  由驱动**从 evaluation_profile.json 现场提取**(`scenarios.front_l2.n` /
  `scenarios.back_l2.n` / `scenarios.full.world_seed_set.count`),与已执行的 smoke
  episode 数(默认 2/scenario × 32 步)显式分离为两组字段。
- **C5 owner 产物只读**:checkpoint / 胶囊只读;params/checkpoint SHA 按 owner
  协议只读复算(full64),与 owner 声明值不符即 fail closed。

## 协议映射要点

- **greedy**:CC2 用 owner policy_step 返回的 argmax action;CC1 用 owner
  `greedy=True`(`pi.mode()`);CC3 的 owner policy_step 恒采样,但**内存更新与
  action 无关**(mem_out 来自同一前向),故取 `argmax(extras["logits"])` 是该同一
  前向的忠实 greedy 读出——不改行为、不重实现。
- **done/true_done**:引擎在 env done 时**停止** episode(不越过 done 再步进),
  故 done_mask / true_done 恒 False,与引擎对 CC2 adapter 的约定一致。
- **§四 语义分立**:两 SlowGRU 族绝不统一。smoke rollout 每 128 步调度
  `on_segment_boundary`;32 步 smoke 到不了边界,故另跑**直接 boundary 单元核验**
  (longstate 叶 +1.0 扰动 → `on_segment_boundary` → RESET128 必须复原 init
  (`LONGSTATE_RESET_TO_INIT`,fast memories `CARRIED`),PERSISTENT 必须原样保留
  (`FULL_CARRY_NO_CLEAR`)),两条证据都写入 projection 记录。
- **dicode 解析钉定**:驱动在 **canonical env 构建与 owner 装载之前**就把源码根
  钉到仓库审计的 `dicode_src/src`——`dicode` 与 `minicraftax` **都在该源根下**
  (均不在 site-packages),canonical env 的 `minicraftax` 导入与随后 bank treedef
  反序列化都必须解析到仓库字节。`dicode/network.py` 字节 SHA `172e1cd4…` == CC1
  声明 policy_source,与 CC1 V7fix58 树字节恒等,2026-07-31 服务器复验;两树唯一
  差异 `wrappers_cl.py` 仅用于 owner 侧 eval_env 包装,不在引擎 canonical env /
  rollout 路径上。
- **wandb stub(仅 CONTROL orbax 路径)**:`dicode.utils.general` 包链在 import 期
  引用 wandb,但 `train_state_utils.py`(SHA `cbd091f9…`)本体零处 wandb
  (grep 验证)。驱动仅在 `import wandb` 失败时安装最小 PEP-562 no-op 模块壳,
  范围与理由写入每份 binding 的 `wandb_stub` 字段。

## GPU 与边界

- **启动合同(CWD = 仓库根)**:冻结引擎 `tier3_source_audit.SOURCE_FILES` 把审计
  原始数据抽取路径记为 `D:/Projects/…`,POSIX 下按 **CWD 相对**解析;驱动必须在
  仓库根启动,使解析命中仓库内 SHA-核验过的抽取副本(`<repo>/D:/…`,
  `s4_task_code.py` `45fdd17c…` == 审计期望 == 服务器原文件,2026-07-31 复验)。
  这与历史 RMT16 bank 铸造 / binding smoke 的启动环境一致;CWD 不符即 fail closed。
- 仅 GPU2 `GPU-8df11537-ab79-722d-606f-411966196c4c` / GPU3
`GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd`(驱动强制;GPU0/GPU1 fail closed);
锁定 CC4 venv(craftax 1.4.5 / jax 0.4.30 / flax 0.8.5 / orbax-checkpoint 0.6.4);
dicode310 不承载 rollout。不启动正式 ranking / 性能评估 / 训练;teacher binding
可 PASS 但 `counts_toward_student_binding_count=false`,不计入
STUDENT_COMMON_BINDING_PASS_COUNT。

## 旧记录关系

服务器 `cc4/<ID>/` 既有 5 份 v1 pending 记录(pool-readiness 轮,2026-07-30)
**原样保留为历史**,不修改;v2 证据写入同目录 `projection_binding_v2/` 子目录,
各自 `SHA256SUMS_V2` 覆盖三份新证据文件,`READY_V2.json` 沿用 sums-excluded 约定。
v2 binding 的 `supersedes` 字段指回对应 v1 文件。
