# §一 审计记录 — NON_RMT runtime 注册权归属裁定(2026-07-31)

合同:NON_RMT_RUNTIME_ABI_BINDING_CLOSURE §一。审计对象:
`/home/oseasy/student_pool_v1/common/`(candidate_runtime_abi.md、common_runner.py、
common_evaluator.py、assembly_manifest.json、SHA256SUMS,本地证据副本见
`../closing_evidence_20260730/common/`,服务器侧 57/57 sums 本轮复验通过)。

## 裁定

```
NON_RMT_RUNTIME_REGISTRATION_AUTHORITY = CC4_CAN_REGISTER_PROJECTIONS (conditional)
```

CC4 有权为 Base GTrXL / Control / SlowGRU(两族)/ Teacher 注册**最小 projection
(投影/适配)entry**,无需等待 owner 亲自注册;条件是下列 C1–C5 全部成立并写入
binding 证据。这不是改写 ABI 注册制,而是沿用 ABI 既已存在的唯一注册先例。

## 证据(原文引用)

### 表面指向 owner 注册的两处文字

1. `candidate_runtime_abi.md` §1.2(本地副本 line 对应段落):
   > "runtime 族注册制。runner 按 runtime_family 分派,
   > `RUNTIME_FAMILIES = ("rmt16_gtrxl_cc2",)`;未知/缺失族 fail closed。
   > common runner 不硬编码 RMT16 — Base GTrXL / Control / SlowGRU / Teacher
   > **由各自 owner 注册(本轮不实现)**"
2. `common_runner.py` 文件头 docstring:
   > "Base GTrXL / Control / SlowGRU / Teacher runtimes are registered by their
   > OWN owners; unknown or missing families fail closed in the engine."

### 为什么这两处不构成阻断

**(a) 唯一先例就是 CC4 写的 projection adapter。** 注册集合中唯一在籍的族
`rmt16_gtrxl_cc2` 并不是 CC2 写进 common/ 的代码,而是 CC4 自己的
`tier3_cc2_policy_adapter.py`(仓库 tools/ 下,CC4 署名),它按 LF-SHA 绑定 CC2
审计过的冻结源码(`network_rmt16.py b5c37d7a…`、`rmt_memory_anchor.py 4ff54fb4…`、
`rmt16_memory.py 17e1a614…`,本轮服务器在盘复验一致),并在每一步调用 CC2 自己的
`rmt_step_forward`。CC2 的胶囊甚至声明 `requires_consumer_adapter=false`,但正式
binding 仍然走 CC4 adapter 完成。因此 ABI 实践中的"注册"一直意指
**"CC4 编写绑定 owner SHA-固定源码的投影/适配模块"**,而非 owner 向 common/ 写码。

**(b) 该句是分工声明且被本合同显式覆盖。** "(本轮不实现)"把其余四族的注册限定
在上一轮工期之外;本合同 §一原文授权:"如果 common ABI 允许 CC4 注册
projection/adapter entry,而不改 owner runtime 语义,则不要等待 owner,直接进行
最小注册。"

**(c) ABI 边界原则 §1.1–§1.7 不仅不禁止、反而强制投影适配。** §1.3:"内存语义
只复用、不重实现"——projection 正是唯一合规形态;§1.1 科学语义评估器独占;
§1.2 未知族 fail closed(投影族注册后即已知族)。没有任何一条原则要求注册代码
必须由 owner 执笔。

**(d) 结构性约束(最强证据)。** 引擎与 runner 文件被 SHA 冻结并被多方引用:
`tier3_candidate_runtime.py` LF-SHA `6af09be4efdb…3f681f`(common_runner 内嵌常量
fail-closed 校验)、`tier3_evaluator.py` LF-SHA `54ae18db24c6…da715`、common/
57 个文件 sums(`sha256sums_sha256=14892443…`)被 assembly_manifest、READY marker、
两个 RMT16 capsule、5 份 pending binding 共同引用。**任何"owner 向 common/ 或
引擎写码"的路径都会打破 57/57 sums 与全部 binding 引用**——即 ABI 在结构上只允许
"新增 CC4 projection 文件"这一种注册方式。

## 条件(写入每份 binding)

- **C1 冻结文件零改动**:`common/` 全部 57 文件、`tier3_evaluator.py`、
  `tier3_candidate_runtime.py`、`common_runner.py`、ABI 文档字节不变;注册物全部
  为 CC4 新增文件(`tools/tier3_scaffolded_evaluation/tier3_projection_runtime.py`
  等 + 服务器 `cc4/<ID>/` 侧 projection/binding 记录)。binding 运行前复验
  common sums 57/57,漂移即 fail closed。
- **C2 零重实现**:每个 projection 直接调用 owner 自己的 SHA-固定运行时模块
  (owner 的 `load_candidate` fail-closed 门 / `policy_step` / SHA 协议函数);
  CC4 侧只写协议映射(policy 对象壳)、greedy_argmax 取用、boundary 调度与证据记录。
- **C3 投影登记文档**:本目录 `PROJECTION_RUNTIME_ADDENDUM.md`(正本随代码提交于
  `tools/tier3_scaffolded_evaluation/`)登记 5 个投影族、绑定面、owner 文件 SHA 来源。
- **C4 诚实标签**:所有 binding `run_class=INTERFACE_SMOKE`、
  `performance_claim_authorized=false`、smoke episode 数 ≠ 正式尺度(正式冻结尺度
  FRONT/BACK/FULL = 8/8/64,取自 evaluation_profile.json)。
- **C5 owner 产物只读**:checkpoint / capsule 文件只读;params/checkpoint SHA
  按 **owner 文件内定义的协议**只读复算(full64),不符即 fail closed,绝不伪造、
  绝不定义竞争哈希。

## 每 owner 协议复算可行性(本轮审计结论,全部 owner-defined 且 owner-enforced)

| 候选 | params 协议(owner 定义) | checkpoint 文件 SHA 协议 | 复算所需运行时 | CC4 venv(锁定环境)可导入? |
|---|---|---|---|---|
| BASE_GTRXL | `canonical_params_sha`:逐 leaf `ascontiguousarray(np.asarray(leaf)).tobytes()`(tree_leaves 序);`load_candidate` 内置复算门 | pkl 字节 sha256 | cc2 candidate_runtime + frozen_modules(`b5c37d7a…/4ff54fb4…/17e1a614…`) | **是**(本轮探测 network_rmt16 IMPORT_OK) |
| CONTROL | `R.params_sha256`:`sha256(b"".join(np.asarray(l).tobytes() for l in tree_leaves))`(eval_bakeoff 约定);CC1 已算 `4c313c58…` | `R.dir_sha256`:排序 (relpath,file_sha256),`rel.encode(); b"\0"; sha.encode("ascii"); b"\n"`;CC1 已算 `34819d77…` | cc1 candidate_runtime → gtrxl128_reference_runtime(`d3d4e552…` 在盘一致)+ dicode `load_weights_only`(orbax)+ stage4 env bundle | **是**(orbax.checkpoint 0.6.4 / dicode.network / minicraftax / wrappers_cl 全部 IMPORT_OK;`train_state_utils.py` 本体 0 处 wandb,仅包链需最小 stub) |
| SLOWGRU_RESET128 | pkl 预存 `(leaves, treedef)`;`params_sha_packed` 逐 leaf `ascontiguousarray.tobytes()`;loader 三重 fail-closed 门(file SHA + network 源 SHA + params SHA) | pkl 字节 sha256 | cc3 candidate_runtime → slowgru_runtime(`d3b74d2e…`)+ arm src slowgru_network(`b2652105…`) | **是**(两 arm slowgru_network IMPORT_OK;SLOW_INTERVAL=32/SLOW_DIM=256 一致;无 orbax/craftax 依赖) |
| SLOWGRU_PERSISTENT | 同上(carry_mode=PERSISTENT;`load_candidate` CARRY_MODE_MISMATCH fail closed) | pkl 字节 sha256 | 同上 | **是** |
| TEACHER | `R.params_sha256`(同 CONTROL);合同 `expected_params_sha256=d4e85af5…`(== canonical base) | pkl 字节 sha256(合同 `file_sha256_method="sha256 of file bytes"`;CC4 上轮已复算 `a87924a3…` 一致) | cc1 candidate_runtime(pickle loader,不需要 orbax) | **是** |

结论:5/5 均可在**锁定环境 CC4 venv**(craftax 1.4.5 / jax 0.4.30 / flax 0.8.5)
内完成 owner 协议复算与投影 smoke,无需环境分裂;dicode310 仅作对照,不承载
rollout(environment_lock 要求 craftax 1.4.5,引擎必须在 CC4 venv)。

## §四 语义分立性(审计确认)

两 SlowGRU 族行为差异在 owner `on_segment_boundary`(slowgru_runtime.py L251–267):
RESET128 → `longstate=init_longstate(B)`、info `boundary_action=LONGSTATE_RESET_TO_INIT`、
`fast_memories=CARRIED`;PERSISTENT → 原样返回、info `boundary_action=FULL_CARRY_NO_CLEAR`。
`load_candidate` 以 `carry_mode` fail closed,两族绝不统一。projection 侧:
(a) smoke rollout 每 128 步调度一次 `on_segment_boundary`(owner 文档边界);
(b) 32 步 smoke 到不了边界,故另跑**直接 boundary 单元核验**(init→步进→
boundary→断言 info 串与 longstate 叶相等性),两条证据都写入 projection 记录。

## 不做的事

不改 common/、不改引擎、不改 owner capsule/checkpoint;不启动正式 ranking /
性能评估 / 训练;不定义任何新哈希;teacher 不计入 STUDENT_COMMON_BINDING_PASS_COUNT;
无 force push/rebase/amend/merge。
