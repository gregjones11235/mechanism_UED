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
- **batch-1 协议壳(仅 CC1 GTrXL128 两族:CONTROL + TEACHER)**:owner 的 dicode
  `transformerXL.forward_eval` 每层后执行 `x = x.squeeze()`,**B=1 时把 batch 维
  一并挤掉**,第 2 层 `jnp.concatenate([memories[:, :, i], x[:, None]])` 即形状
  失配(服务器实测 `(1,128,256)` vs `(256,1)` TypeError)。owner 自有评测**从不
  在 B=1 运行**(build_stage4_env smoke_batch_size≥2;eval_bakeoff NUM_ENVS=256,
  恒向量化)。`forward_eval` 行间完全独立(逐行 encoder / 逐行 attention,无任何
  跨 batch 运算),故 adapter **原样调用 owner policy_step**,仅以 batch 2 复制
  行运行、读 row-0 action / row-0 memory——与 B=1 语义**数值恒等**。这是协议壳
  的 batching 选择(projection 的本职),**未改 owner 任何代码**
  (`owner_code_modified=false`);两份 binding 以 `batch1_workaround` 字段完整
  公开。CC2 BASE 的 frozen 网络 B=1 路径正常(实测通过),不受此影响。
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
- **import-chain 补全(锁定 venv,全部公开)**:CC1 owner `build_stage4_env` 的
  导入链(`dicode.utils.general` 包链、`dicode.task_utils → dicode.dreaming.
  gen_manager → {llm, utils}`)会触及锁定 venv 未装的包。处置分两类,每份 binding
  的 `wandb_stub` / `import_stubs` 字段逐一公开:
  - **真装(纯依赖,零数值/零网络语义)**:`networkx 3.6.1`、`hydra-core 1.3.4`、
    `omegaconf 2.3.1`、`antlr4-python3-runtime 4.9.3`,经 `pip install --no-index
    --no-deps` 从 SHA-记录 wheel/sdist 装入(networkx wheel
    `d47fbf30…`,hydra_core wheel `e5868369…`,omegaconf wheel `3d701d14…`)。
    装后复探锁定版本**全数不变**(python 3.11.15 / jax 0.4.30 / jaxlib 0.4.30 /
    craftax 1.4.5 / flax 0.8.5 / orbax-checkpoint 0.6.4 / distrax 0.1.5 /
    optax 0.2.5 / numpy 1.26.4 / scipy 1.17.1,2026-07-31 服务器复验)。
  - **stub(仅满足 import)**:`wandb`(PEP-562 no-op 壳;`train_state_utils.py`
    本体零处 wandb,grep 验证);`openai`(**import-only 壳,`AsyncOpenAI` 可被
    import 但实例化即 raise**,其余属性访问亦 raise——合同禁止 CC4 任何新 LLM
    调用,stub 从结构上使调用不可能;本路径实际只用 `task_utils.
    get_achievement_multi_hot` 的纯 numpy/craftax 常数数学,LLM 类从不实例化)。
    壳设 `__path__=[]`(无子模块包壳:CPython from-import 机制在 C 层探查
    `__path__`,须为真实属性;`import openai.X` 仍 fail closed);驱动重试
    循环只对 `ModuleNotFoundError.name` **恰为** `wandb`/`openai` 的顶层缺失
    装壳,任何子模块级缺失一律 fail closed,不扩大 stub 面。
  - 任何**其他**缺失模块一律 fail closed,不猜、不静默 stub。

- **引擎 FRONT corridor predicate 裁定记录(引擎设计级 fail-closed,不放宽)**:
  `tier3_evaluator.rollout_episode` 在 FRONT 场景对每一仍在 floor-1 的步调用
  `tier3_event_predicates.normalized_corridor_progress(state, walkable, …)`,其中
  `walkable = _front_walkable_grid(start_state, view)` 是**静态初始网格**——取初始
  `map[FRONT_FLOOR]` 的 BlockType 陆生可走集(与 `game_logic.move_player` 碰撞
  一致,排除 SOLID_BLOCK/WATER/LAVA),唯一例外是把两枚梯子转运 tile OR 入网格
  (LADDER_TILE_TRANSIT)。引擎 docstring 明示该度量是 "graph distance over map
  topology"(**初始**地图拓扑):**挖掘(mining)出初始可走网格、站到被挖开的
  原 SOLID_BLOCK 格**在该度量域内为非法,引擎按设计 fail-closed 中止该 rollout
  (引擎注释:"FAIL CLOSED (no swallowing) … never silently skipped … STOPS
  permanently")。predicate 由 predicate_code_sha256 绑定、引擎模块 LF-SHA 冻结,
  CC4 无权放宽、跳过或重实现(C1/C2);正式评估跑**同一引擎代码路径**,对同一
  候选会得到**同一裁定**,故 binding 必须如实记录而非掩盖。驱动处置:仅捕获
  `tier3_event_predicates.FailClosed`(引擎设计级裁定),记录为结构化最小阻断
  证据(`smoke_abort` 字段:exception_type / engine_message / scenario /
  episode_index / entry_id / seed / verdict=ENGINE_PREDICATE_REJECTED_ROLLOUT /
  formal_evaluation_consequence),保留中止前已完成 episode 的部分证据
  (`partial=true`),写 `binding_status=BLOCKED`、`interface_smoke_status=
  FAIL_CLOSED_ENGINE_PREDICATE`、`READY_V2=false`(G4 失败),并附每候选一条
  minimum owner prompt(§七 诚实 BLOCKED 纪律);**任何其他异常**(含 evaluator
  自身 require 的 FailClosed)仍令驱动 fail-closed 崩溃。实测触发者:
  CONTROL_CONTINUOUS_98304(训练后的 greedy policy 挖墙推进;FULL 种子已完成、
  FRONT 被拒)。BASE / TEACHER 的 FRONT 行为不挖墙,未触发。

## GPU 与边界

- **启动合同(CWD = 仓库根)**:冻结引擎 `tier3_source_audit.SOURCE_FILES` 把审计
  原始数据抽取路径记为 `D:/Projects/…`,POSIX 下按 **CWD 相对**解析;驱动必须在
  仓库根启动,使解析命中仓库内 SHA-核验过的抽取副本(`<repo>/D:/…`,
  `s4_task_code.py` `45fdd17c…` == 审计期望 == 服务器原文件,2026-07-31 复验)。
  这与历史 RMT16 bank 铸造 / binding smoke 的启动环境一致;CWD 不符即 fail closed。
- 仅 GPU2 `GPU-8df11537-ab79-722d-606f-411966196c4c` / GPU3
`GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd`(驱动强制;GPU0/GPU1 fail closed);
锁定 CC4 venv(craftax 1.4.5 / jax 0.4.30 / flax 0.8.5 / orbax-checkpoint 0.6.4;
补装 import-chain 纯依赖后锁定版本复验全数不变,见上条);
dicode310 不承载 rollout。不启动正式 ranking / 性能评估 / 训练;teacher binding
可 PASS 但 `counts_toward_student_binding_count=false`,不计入
STUDENT_COMMON_BINDING_PASS_COUNT。

## 旧记录关系

服务器 `cc4/<ID>/` 既有 5 份 v1 pending 记录(pool-readiness 轮,2026-07-30)
**原样保留为历史**,不修改;v2 证据写入同目录 `projection_binding_v2/` 子目录,
各自 `SHA256SUMS_V2` 覆盖三份新证据文件,`READY_V2.json` 沿用 sums-excluded 约定。
v2 binding 的 `supersedes` 字段指回对应 v1 文件。
