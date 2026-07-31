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
- **batch-1 协议壳(CC1 GTrXL128 两族 CONTROL + TEACHER,CC3 SlowGRU 两族
  RESET128 + PERSISTENT——共四族同一崩溃点)**:owner 的 dicode
  `transformerXL.forward_eval` 每层后执行 `x = x.squeeze()`,**B=1 时把 batch 维
  一并挤掉**,第 2 层 `jnp.concatenate([memories[:, :, i], x[:, None]])` 即形状
  失配(服务器实测 `(1,128,256)` vs `(256,1)` TypeError,崩溃点
  `transformerXL.py:194`)。CC3 的 `slowgru_network.forward_eval`(arm
  `b2652105…`)第一行即委托**同一** dicode `transformerXL.forward_eval`(字节
  同一模块,2026-07-31 服务器实测同一行同一形状报错),故同一协议壳同样适用。
  各 owner 自有评测**从不在 B=1 运行**:CC1 build_stage4_env
  smoke_batch_size≥2、eval_bakeoff NUM_ENVS=256;CC3 trainer PPO `_env_step`
  恒在 E envs 上向量化(slowgru_runtime docstring:"replicating the trainer
  `_env_step` memory mechanics verbatim")。**行间独立性证据**:
  `transformerXL.forward_eval` 逐行 encoder / 逐行 attention,无任何跨 batch
  运算;owner 自有 `_slow_update` 头部注释明示 "vectorised over env axis; no
  cross-env mixing"(逐行 buffer 写入 / 逐行 attention pooling / 逐行
  GRUCell);fast memory `jnp.roll(..., axis=1)` 逐行;mask 机构逐行;
  `on_segment_boundary` 逐行(RESET128:longstate → `init_longstate(B)` 全批;
  PERSISTENT:恒等)。故 adapter **原样调用 owner policy_step /
  on_segment_boundary**,仅以 batch 2 复制行运行、读 row-0 action——与 B=1
  语义**数值恒等**(§四 boundary 语义在壳的有效批上同样忠实:row-0 所受逐行
  运算与 B=1 完全相同;两 SlowGRU 族的行为分立仍**唯一**来自 owner 的
  mode-dependent `on_segment_boundary`,绝不统一)。这是协议壳的 batching
  选择(projection 的本职),**未改 owner 任何代码**
  (`owner_code_modified=false`);每份 binding 以 `batch1_workaround` 字段完整
  公开。adapter 复制内存状态时 `step_idx`(python int,非数组叶)保持 int 不升维。
  CC2 BASE 的 frozen 网络 B=1 路径正常(实测通过),不受此影响。
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
- **numpy2 pickle 兼容(仅 CC3 两族:RESET128 + PERSISTENT)**:CC3 两份
  checkpoint 在 **numpy>=2** 下 pickle,ndarray 归约在流中引用
  `numpy._core.numeric._frombuffer`(protocol-5 in-band 重建)。锁定 CC4
  venv 钉 numpy 1.26.4(jax 0.4.30 钉/environment_lock,**不得升级
  numpy**)。该 venv 的 numpy 1.26.4 **自带官方 numpy2-pickle 兼容 shim
  包** `site-packages/numpy/_core/`(自带 docstring:"stubs for
  interoperability with NumPy 2.0 pickled arrays"),覆盖 `_dtype` /
  `_internal` / `multiarray` / `_multiarray_umath` / `umath`,**唯独缺
  `numeric` 这一叶**——正是 CC3 流所引用者。处置:驱动在 owner
  `load_candidate()` **之前**为该官方 shim **补齐唯一缺叶**:
  `sys.modules["numpy._core.numeric"] = numpy.core.numeric`(numpy1 同一
  `_frombuffer`,语义恒等);门为"`numpy._core.numeric` 能否原生
  import",能则不做任何事。注意 `hasattr(np, "_core")` **不是**合法门:
  on-disk shim 包导入后即设该属性,但不提供 numeric 叶(实测驱动进程
  在 stage3 jax 导入链后 hasattr=True 而 pickle 仍失败,据此定位)。
  作用域经两份 pkl 的**只读字节扫描**核验:每份恰一条 numpy2 路径引用
  (`numpy._core.numeric`),别无其他;任何其他缺失模块仍 fail closed,
  不扩面。保真见证 = owner params_sha_packed 门(G3)+ pkl 文件 SHA 门
  (G2):别名若改动任何数值,声明 SHA 必不符,binding 即 fail closed。
  owner 代码零改动、pkl 字节只读。每份 binding 以 `numpy_pickle_compat`
  字段完整公开。

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

## V2_DYNAMIC_TOPOLOGY 修正(任务 CC4_FIX_FRONT_DYNAMIC_TOPOLOGY_METRIC_AND_REBIND_FORMAL_POOL_V2)

```
COMMON_EVALUATOR_PROTOCOL_VERSION = V2_DYNAMIC_TOPOLOGY
COMMON_EVALUATOR_V1_DRIVER        = d0d05ff26ffd1ea0bfd80e4c0364edfe6f5616d4
COMMON_EVALUATOR_V1_STATUS        = SUPERSEDED_PRE_RANKING(字节原样保留,不覆盖、不伪装)
```

总控裁定根因:`COMMON_EVALUATOR_METRIC_DOMAIN_NOT_CLOSED_UNDER_LEGAL_TOPOLOGY_MUTATION`——
CONTROL 的 checkpoint 身份/参数身份/运行时装载/策略执行**全 PASS**;旧 FRONT corridor
dense metric 在**初始**地图可行图(INITIAL_MAP_TOPOLOGY)上算图距离,合法挖墙推进站到
初始图之外的 tile 即被旧引擎以 `invalid_position_policy: player position non-walkable`
中止。**不是候选损坏,不重训候选,不给候选级豁免**;修的是 COMMON FRONT dense metric 本身。

**修正面(恰好三处,§二/§三/§四)**:
1. **图来源**:`BFS_GRAPH_SOURCE=CURRENT_ENVIRONMENT_STATE_TOPOLOGY`——每步按**当前**
   环境状态 `state.map[FRONT_FLOOR]` 重建合法可行图(构造规则与 V1 完全相同:BlockType
   陆生可走集 + LADDER_TILE_TRANSIT 两枚梯子 tile OR 入;被合法挖开的 tile 不再是
   SOLID_BLOCK,自然可走)。归一化分母 d(start, exit) **固定**为 episode 起始在初始图上
   一次性算得的基线。
2. **合法位置域**:当前图成员资格。站到合法挖开的当前 tile / 初始图之外但当前状态确认
   合法的 tile = 有效,绝不因初始图成员资格中止。继续 fail-closed:坐标越界;非有限/
   不可解码坐标(状态腐败);玩家状态与**当前**地图矛盾(当前图仍为 solid——合法移动
   不可能造成,因为挖开即变可走)。
3. **不可达处置**(两类,均不中止):目标在当前动态图中暂不可达 → primary 保持 false、
   dense progress **保守冻结不增加**(返回上一步 progress 原值)、episode 继续;
   **基线** d(start, exit) 在初始图上不存在 → 同处置(分母未定义,progress 冻结于
   前值)。合法 floor2→floor3 跃迁 → primary=true(V1 primary predicate 原对象复用,
   未改)。
   **实证订正(2026-07-31)**:冻结 FRONT 银行(内容 SHA `21aeb7dc…` 装载时复验)实测
   含**合法**的挖掘必需 scaffold——state 7 / seed 10007 初始图 start→exit 不可达
   (d_start=None)而 `valid_front_scaffold_start=True`。故基线不可达是**合法边界**而非
   银行 payload 腐败;V1 在此以 NEG18 中止即 §一 根因同族。V2 第一版曾误将其归为腐败类
   FailClosed,经 8 状态实景探针实测后订正为保守冻结(位置域 fail-closed 不变)。

**未动(结构保证,非转写)**:progress 公式 `clip(1 - d_t/max(d_start,1), 0, 1)`、
NEG17、bfs_distance、全部有效性/事件/primary predicate——`tier3_event_predicates_v2.py`
从冻结 V1 模块**直接 import 复用**(同一对象);`tier3_evaluator_v2.py` 对冻结
`tier3_evaluator.py` 做全表面 re-export,**仅覆盖 rollout_episode 的 FRONT dense 块**;
BACK / FULL 语义、episode 记录字段、running-max 聚合、调度、冻结合同(greedy_argmax /
4096 / 8-8-64)、全部候选 checkpoint、冻结状态银行、FULL 种子、BACK predicate 一律未动。

**新文件(仅新增;V1 字节零改动)**:
- `tier3_event_predicates_v2.py` — V2 dense predicate + 自检(31 项);
- `tier3_evaluator_v2.py` — V2 rollout(FRONT dense 块三处替换)+ V1 全表面 re-export;
- `tier3_projection_binding_smoke_v2.py` — V2 binding 驱动(common_v2/ 钉定 + V1 冻结
  保全复验 57/57 + 装配清单 4 引擎模块 LF-SHA 钉定:`tier3_evaluator.py` /
  `tier3_candidate_runtime.py` 两枚 V1 冻结钉不变 + `tier3_evaluator_v2.py` /
  `tier3_event_predicates_v2.py` 两枚新 V2 钉;V2 薄壳另行钉 4 枚引擎 LF-SHA 含
  `tier3_event_predicates.py`;证据写 `cc4/<ID>/projection_binding_v2dt/`,文件
  `*_v2dt.json` / `SHA256SUMS_V2DT` / `READY_V2DT.json`);
- `tier3_v2_dynamic_topology_regression.py` — 回归 A–F(纯逻辑 `--self-test` +
  服务器实景 `--server-suite`)。

**回归测试(§五,不得当性能结论)**:A STATIC_TOPOLOGY_PARITY(固定不变图轨迹 V1≡V2:
primary/dense/terminal/episode 规范化载荷全同;合成 + 8 FRONT bank NOOP 实景——初始可达
状态逐字节同;挖掘必需状态按**预定分歧**见证:V1 复现 NEG18 中止、V2 完成且 progress
冻结 0.0 无假 primary);B LEGAL_DIG_NO_ABORT(合法挖到初始图外不中止;合成 + CONTROL
实景);C DYNAMIC_DISTANCE_UPDATE(挖开后 BFS 用当前图;合成 + 实景见证);D
UNREACHABLE_CONTINUES(不可达不中止/不假 primary/progress 冻结;合成权威);E
TRUE_INVALID_FAIL_CLOSED(越界/非有限/不可解码/与当前图矛盾仍 fail-closed,基线不可达
则合法冻结——位置域校验在基线冻结下依旧 fail-closed;合成权威);F CONTROL_REPRODUCTION
(原 CONTROL checkpoint + 原阻断起点 front_l2-bank0/seed=10000:V1 仍复现中止,V2 完成)。

**服务器 common_v2/ 装配**:V2 薄壳 `common_evaluator.py`(钉 `tier3_evaluator_v2.py` +
`tier3_event_predicates_v2.py` LF-SHA,委托 V2 模块)、V2 时间戳壳 `common_runner.py`
(引擎钉定不变:`tier3_candidate_runtime.py` `6af09be4…`——runner 不含 FRONT dense 语义,
语义零改动,仅装配 provenance 时间戳更新)、`evaluation_profile.json` /
`metric_schema.json`(V1 字段逐字不动 + 追加 `common_evaluator_protocol_version` 与修正
块;历史自引用 SHA 字段原样保留)、`environment_lock.json` 与银行 artifact 字节不变
(内容 SHA 与 V1 同一)、`candidate_runtime_abi.md` 字节复制(SHA 同 V1 `61e52af6…`)、
新 `assembly_manifest_v2.json` + `SHA256SUMS` + `COMMON_EVALUATOR_V2_READY.json`
(§七 门:6/6 + teacher + sums + A–F 全 PASS 前 READY=false)。

**门禁(§七)**:6/6 Student V2 binding PASS + Teacher V2 binding PASS(reference_only,
`counts_toward_student_binding_count=false`)+ 全部 SHA256SUMS PASS + A–F 全 PASS →
`COMMON_EVALUATOR_V2_READY=true`,`STUDENT_COMMON_BINDING_PASS_COUNT=6/6`。**本任务仍不
启动正式性能 ranking,必须等待独立二次审计**(`FORMAL_RANKING_STARTED=false`)。

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
