# TASK B1 — HO Reinjection + Floor2→Floor3 Probe (tier3-integrated)

日期: 2026-08-16。下发者: 总控(director)。执行者: deepseekv4flash_worker。
状态: 待执行。本文件即完整实现规范；实现中若与仓库证据冲突，停止并报告，禁止自行改设计。

## 0. 背景（已核实事实）

- Memory Study 协议 Step 1(G0) 已完成: docs/memory_study/MEMORY_STUDY_CODE_MAP.md。结论: HO reinjection 与 Floor2→Floor3 probe 在现役树不存在（NOT_FOUND），用户已批准新建。
- tier3 评估器位于未合并分支 origin/henry/tier3-scaffolded-evaluation 的 tools/tier3_scaffolded_evaluation/（48 文件，32314 行；Henry-branch 无 tools/ 目录，提取为纯增量）。
- tier3 FRONT_L2 场景即 Floor2→Floor3 探针: 起点=floor-2 黑暗走廊，PRIMARY event=FRONT_FLOOR_TRANSITION_REACHED（player level 2->3），primary metric=P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START。状态库跨臂字节级共享、结果盲选择（NEG26）、双进程验证。
- tier3_projection_runtime.py 有四架构投影策略: BaseGtrxlProjectionPolicy / GTrXL128ProjectionPolicy / SlowGRUProjectionPolicy / Rmt16CapsuleProjectionPolicy + build_policy(spec, ctx)，以及四族 checkpoint 装载器（cc1/cc2/cc3/cc4）。
- tier3 代码为冻结产物: 提取后禁止修改任何字节。
- 本地环境: Windows + PowerShell, Python 3.12.4 (Anaconda), 无 jax/craftax。tier3 状态库 REAL 模式需 JAX+craftax==1.4.5（仅服务器有）; SYNTHETIC 模式无 JAX 可跑 —— 本地测试必须全部在无 jax 下通过。
- 服务器: oseasy@172.25.14.221（SSH 已通），tier3 V3 正式运行纪律: 锁定 CC4 venv, GPU2/GPU3, CWD=repo root（本任务不在服务器运行，仅写 RUNBOOK）。
- 工作树: C:\Users\Lenovo\Desktop\dicode-codex-director\mechanism_UED_Henry_worktree（分支 Henry-branch, HEAD 0dd9de5b, 与 origin 同步）。
- 禁碰清单（已有他人改动）:
  M gpu1_aggregation_siege/src/dicode/student_adapters/slowgru_adapter.py
  M gpu1_aggregation_siege/tests/simulator_frontier/test_slowgru_adapter.py
  ?? gpu1_aggregation_siege/scripts/evaluate_e3_slowgru_original_task.py
  ?? gpu1_aggregation_siege/src/dicode/simulator_frontier/e3_slowgru_original_eval.py
  ?? gpu1_aggregation_siege/tests/simulator_frontier/test_e3_slowgru_original_eval.py
  ?? gpu1_aggregation_siege/src/dicode/e3_litesim/（并行会话在制品）
  ?? gpu1_aggregation_siege/docs/e3_litesim/

## 1. Part 0 — tier3 冻结工具提取（git 写操作仅限此步）

1.1 git checkout origin/henry/tier3-scaffolded-evaluation -- tools/
1.2 验证: git diff origin/henry/tier3-scaffolded-evaluation -- tools/ 为空（逐字节一致）; git status 只显示 tools/ 新增。
1.3 本地运行 python tools/tier3_scaffolded_evaluation/tier3_self_test.py。预期无 jax 也可通过（模块顶部 JAX-free; SYNTHETIC 模式）。若失败: 记录完整输出，判断是否环境缺失（如实报告，不修改冻结代码）。
1.4 提交（原子提交 1）: feat(eval): import frozen tier3 scaffolded evaluation tooling verbatim from henry/tier3-scaffolded-evaluation
1.5 禁止 push。禁止改任何提取文件。

## 2. Part 1 — 新 memory_study 包（唯一代码写范围）

写范围（全部新建）:
- gpu1_aggregation_siege/src/dicode/memory_study/__init__.py
- gpu1_aggregation_siege/src/dicode/memory_study/ho_contract.py
- gpu1_aggregation_siege/src/dicode/memory_study/ho_burnin.py
- gpu1_aggregation_siege/src/dicode/memory_study/ho_capture_bank.py
- gpu1_aggregation_siege/src/dicode/memory_study/floor23_probe.py
- gpu1_aggregation_siege/scripts/run_memory_study_floor23.py
- gpu1_aggregation_siege/tests/memory_study/*.py（测试）
- docs/memory_study/HO_FLOOR23_DESIGN.md（设计与 RUNBOOK）

### 2.1 ho_contract.py
- HOMode 枚举: BASE / HO_ZERO / HO_REAL。
- HistoryCapture: obs_segment (T, obs_dim) + provenance（source_seed, capture_policy_id, bank_hash, payload_sha256, floor_context）。obs_dim=8335 常量引用 d052/legacy/canonical_constants.py（若可导入则复用，否则常量+注释出处）。
- IsolationReceipt: 逐项记录 G2 隔离断言结果（params_sha_before/after, env_state_payload_hash_before/after, rng_stream_id, task_embedding_hash, timestep, inventory_hash, position_hash, entities_hash），任何一项不符 -> FailClosed 异常。
- 序列化约定与 tier3_state_serializer 的 payload hash 风格一致（canonical JSON + sha256），可直接 import tier3 模块作库使用（sys.path 方式，与 tier3 内部做法一致）。

### 2.2 ho_burnin.py
- burnin_history(step_fn, params, memory, capture, mode) -> (memory_out, receipt)
  - step_fn 最小协议: (params, memory, obs) -> new_memory（由适配层包装 tier3 投影策略或 StudentTrainingBackend.policy_forward_eval 得到；包装器也放在本包内）。
  - HO_REAL: 逐步把 capture.obs_segment 喂给 step_fn，累积 memory（纯 student 前向，无 env 交互，无梯度）。
  - HO_ZERO: 喂全零段（形状相同）——结构性对照。
  - BASE: 不做 burn-in，memory 原样返回（receipt 仍生成，便于 G1 对比）。
  - receipt 断言: params 哈希前后一致; 无 env state 参与（函数签名层面隔离）; RNG 使用独立 stream（burn-in 不消耗 env/rollout RNG）。
- 确定性要求: 同输入两次调用 memory 逐叶相等（测试用 mock 验证）。

### 2.3 ho_capture_bank.py
- 结果盲 capture 生成: 从 FRONT_L2 状态库起点出发，用固定声明的 capture policy（默认 uniform-random，seed schedule 固定且与被测 student 无关）跑 K 步，记录 obs 序列 + metadata。
- 输出 capture bank manifest（JSON + SHA256SUMS 风格），每条 capture 有 payload hash；禁止引用被测 student 的任何结果（NEG26 同构纪律）。
- 无 jax 环境提供 SYNTHETIC 生成路径（生成明确标注 SYNTHETIC_TEST_ONLY 的假 obs 段），REAL 路径在服务器 RUNBOOK 中描述。

### 2.4 floor23_probe.py
- 探针主循环: for state in front_l2 bank: for student in candidates: for mode in (BASE, HO_ZERO, HO_REAL): memory = burnin(...); rollout 用 tier3 投影策略从该 state 起步（冻结代码作库调用，不改）; 收集 primary event（tier3_event_predicates 的 front_floor_transition_reached）+ dense metrics（tier3_metrics.summarize 作库调用）。
- 每个 (state, student, mode) 结果写 JSON: 完整 provenance（state payload hash, params sha, capture bank hash, ho_mode, seed, schema id）。
- SYNTHETIC 模式下全链路可跑（mock policy + synthetic bank），用于本地测试与协议演练；REAL 模式在服务器执行（RUNBOOK）。

### 2.5 CLI 脚本 scripts/run_memory_study_floor23.py
- argparse: --mode {synthetic,real} --candidate-id ... --ho-modes base,ho_zero,ho_real --bank-manifest PATH --capture-bank PATH --out-root PATH --max-states N
- fail-closed: 缺资产/哈希不符 -> 非零退出 + 结构化 BLOCKED 输出。

### 2.6 测试（全部无 jax 本地通过）
- test_ho_contract.py: 枚举/序列化/哈希稳定性。
- test_ho_burnin_isolation.py: mock step_fn 验证确定性、HO_ZERO 结构路径、receipt 断言触发（篡改 params -> FailClosed）。
- test_capture_bank.py: 结果盲纪律（manifest 不含 student 标识）、哈希复算一致。
- test_floor23_probe_synthetic.py: synthetic 端到端，结果 schema 校验，fail-closed 路径。

### 2.7 HO_FLOOR23_DESIGN.md
- 架构图（文字版）: capture bank -> burn-in -> FRONT_L2 rollout -> 指标。
- G1/G2/G3 映射: G1=同 checkpoint BASE vs HO_ZERO 等价（本地 mock 结构等价 + 服务器数值容差复验）; G2=隔离 receipt 全项通过; G3=HO_REAL 在已知案例上显著优于 BASE/HO_ZERO（服务器 REAL 运行，样本量与判据在 RUNBOOK 预先冻结）。
- 服务器 RUNBOOK: CC4 venv、GPU2/3、REAL 状态库物化（tier3_state_bank_materializer 两进程协议）、capture bank 生成、G1-G3 执行顺序、结果落盘路径约定、失败即 BLOCKED 不重试纪律。
- 六组 student 对比计划: GTRXL_BASE/GTRXL_HO/RMT16_RESET/RMT16_PERSISTENT/SLOWGRU_RESET/SLOWGRU_PERSISTENT 在 FRONT_L2 上的 primary metric + dense metrics 矩阵；与 tier3 V3 既有结论（INCONCLUSIVE_FULL_TIE）的关系声明。

## 3. 验收标准（director 将逐项独立复核）

A1. tier3 提取逐字节一致（git diff 空）且 self_test 结果如实报告。
A2. 新测试全绿: python -m pytest gpu1_aggregation_siege/tests/memory_study -q（无 jax 环境）。
A3. 隔离断言为机械强制（receipt + FailClosed），不是注释约定。
A4. 禁碰清单零触碰; 冻结 tier3 文件零修改; 无 git push; 无全局 git/ssh 配置改动。
A5. 三个原子提交: (1) tier3 import; (2) memory_study 包+测试; (3) 设计文档。提交信息清晰。
A6. 汇报包含: 文件清单、测试命令与结果、self_test 输出摘要、任何 BLOCKED 项及原因。

## 4. 执行纪律

- 你是实现 worker，直接用自己的工具执行；禁止 spawn 子代理；禁止扩大写范围；禁止 git push。
- 开始执行前先回复一行: SCOPE_RECEIVED B1 + 写范围一句话复述。
- 与仓库证据冲突或遇阻断: 停止并报告，不要猜。
