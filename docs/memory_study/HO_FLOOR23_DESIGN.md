# HO Reinjection + Floor2→Floor3 Probe — 设计与 RUNBOOK（TASK B1）

日期: 2026-08-16 · 分支: Henry-branch · 状态: B1 本地实现完成（SYNTHETIC 全绿），REAL 待服务器 G1–G3。

## 1. 目标

回答一个机制问题：**在学生（Student）的持久记忆被一段"历史观测（HO, history-observation）"
预热（burn-in）之后，其从 Floor-2 黑暗走廊走到 Floor-3 的能力是否可测地改变？**
探针完全复用冻结的 tier3 FRONT_L2 评估契约（场景、事件谓词、指标、状态库纪律），
tier3 代码零字节修改、仅作库调用。

## 2. 架构（文字版）

```
结果盲 capture bank（共享、字节级同一、含 payload hash）
        │  assign_capture(key=state_id)  ← 只依赖状态身份，绝不依赖任何被测结果
        ▼
HO burn-in（ho_burnin.burnin_history）
  ├─ BASE    : 不预热，memory 原样（对照）
  ├─ HO_ZERO : 全零段预热（结构性对照，形状与 REAL 相同）
  └─ HO_REAL : 真实 capture 段预热（纯前向、无 env、无梯度、无 RNG 消耗）
        │  IsolationReceipt（机械 G2：任何一项不符即 FailClosed）
        ▼
FRONT_L2 rollout（候选策略从冻结起点状态出发；tier3 投影策略作库调用）
        ▼
tier3 计分（库调用，零修改）
  ├─ tier3_event_predicates.front_floor_transition_reached(from_level, to_level)
  │    = (from_level == 2) and (to_level >= 3)          ← PRIMARY 事件交叉校验
  └─ tier3_metrics.summarize / compute_primary_metric
       primary = P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START
       dense   = GRAPH_DISTANCE_PROGRESS ∈ [0,1]
        ▼
每 (state, candidate, ho_mode) 一份全溯源结果 JSON + 聚合 summary.json
```

### 2.1 模块清单（gpu1_aggregation_siege/src/dicode/memory_study/）

| 模块 | 职责 |
|---|---|
| ho_contract.py | HOMode / HistoryCapture / IsolationContext / IsolationReceipt；canonical JSON+sha256；hash_pytree；OBS_DIM=8335 锚定 d052/legacy/canonical_constants.py（文件式导入，漂移即 FailClosed） |
| ho_burnin.py | burnin_history（step_fn 最小协议 (params, memory, obs)->memory）+ 两个适配器：wrap_tier3_projection_policy（冻结投影策略，burn-in 期间动作丢弃、只推进 policy.ms）、wrap_backend_policy_forward_eval（StudentTrainingBackend 后端的 (pi,value,mem_out,new_memory) 四元组，只取 new_memory） |
| ho_capture_bank.py | 结果盲 capture bank：SYNTHETIC 生成器（stdlib-only，SYNTHETIC_TEST_ONLY 标签）、写盘（manifest.json + captures/*.json + SHA256SUMS）、加载复算全部哈希、assign_capture 确定性盲分配 |
| floor23_probe.py | 探针主循环 + 结果 schema + SYNTHETIC 状态/候选构造器 + load_tier3_library（sys.path 库化冻结 tier3） |
| scripts/run_memory_study_floor23.py | CLI：--mode synthetic 本地端到端；--mode real 缺资产/无 jax 一律结构化 BLOCKED + 退出码 2 |

### 2.2 隔离断言是机械的（G2）

- burnin_history 的函数签名在模块导入期即被断言不含 env 参数（结构隔离，非注释约定）；
- IsolationContext.env_state_payload_hash 非 None → 立即 FailClosed；
- params 哈希：调用前快照必须与调用方声明一致（PARAMS_SNAPSHOT_DISAGREEMENT），
  前后再复算一次，任何变化 → ISOLATION_VIOLATION: params_invariant；
- receipt 七项检查全过才允许 verdict=PASS，任一失败抛 FailClosed，不存在"FAIL receipt"。

### 2.3 结果盲纪律（NEG26 同构）

- capture bank manifest 禁止出现 candidate/student 字样（生成与加载双向强制扫描）；
- 所有臂共享同一 bank（字节级）；capture 分配只哈希 state_id；
- capture 由声明式策略（SYNTHETIC: UNIFORM_RANDOM_SYNTHETIC_V1；REAL: RUNBOOK 固定
  capture policy + 固定 seed schedule）生成，与被测学生无关。

## 3. G1/G2/G3 门禁映射

| 门禁 | 内容 | 本地（本文档交付） | 服务器（RUNBOOK） |
|---|---|---|---|
| G1 | 同 checkpoint 下 BASE vs HO_ZERO 等价性（burn-in 结构不改变无信息输入下的行为） | mock 结构等价测试通过（test_ho_burnin_isolation.py） | 真实 checkpoint 数值容差复验：同一 GTrXL checkpoint 两臂 primary/dense 差 ≤ 预先冻结容差 |
| G2 | 隔离 receipt 全项 PASS（params 不变、env 缺席、RNG/task/inventory/position/entities 声明齐全） | 机械强制已演示（篡改即 FailClosed，47 测试全绿） | REAL 运行中每个 (state,cand,mode) 均须 PASS receipt 落盘 |
| G3 | HO_REAL 在已知案例上显著优于 BASE/HO_ZERO（正向控制） | 不做性能声称（SYNTHETIC 仅为管线演练） | 预先冻结样本量与判据（见 RUNBOOK §5.4），配对检验 over 共享状态库 |

## 4. 与冻结 tier3 的关系

- 提取来源: origin/henry/tier3-scaffolded-evaluation（tools/ 45 tier3 文件 + 3 global_evaluation 文件；
  另按依赖提取 schemas/ 7 件、configs/ 5 件——tier3_metrics 以 LF-SHA 绑定 schemas/tier3_metric_schema_v1.json，
  checkpoint 契约在 configs/）。提取后 git diff 对该分支路径为 0 行。
- tier3_self_test.py 本地（无 jax）结果: **TIER3_AGGREGATE_SELF_TEST_PASS (modules=11, negative_tests=FAIL0)**。
- 探针仅 import tier3_metrics / tier3_event_predicates 作库；REAL 运行额外使用
  tier3_state_bank_materializer（两进程物化协议）与 tier3_projection_runtime（build_policy + 四族装载器）。
- 冻结文件禁止任何字节修改；任何适配一律发生在本包适配器层。

## 5. 服务器 RUNBOOK（REAL 路径）

前置: oseasy@172.25.14.221，锁定 CC4 venv，GPU2/GPU3，CWD=repo root；失败即 BLOCKED，不自动重试。

1. **状态库物化**: `tier3_state_bank_materializer.py` 两进程协议物化 FRONT_L2 状态库
   （craftax==1.4.5；SYNTHETIC→REAL 切换仅在服务器）；记录 FRONT_SCAFFOLD_STATE_BANK_HASH 与两进程一致证据。
2. **capture bank 生成（REAL）**: 从状态库起点出发，声明式 capture policy（固定 seed schedule，
   与被测学生无关）跑 K 步记录 obs 段（obs_dim=8335），写 manifest + SHA256SUMS；
   manifest 结果盲扫描同本地纪律。
3. **候选绑定**: 按 tier3_projection_runtime.build_policy(spec, ctx) 绑定六组候选
   （cc1 GTrXL128 / cc2 BaseGtrxl / cc3 SlowGRU reset+persistent / cc4 Rmt16 capsule reset+persistent），
   checkpoint 装载器校验 contract SHA，漂移即 FailClosed。
4. **执行顺序**: G1（BASE vs HO_ZERO 同 checkpoint 等价，数值容差先冻结）→ G2（全量 receipt 复核）
   → G3（HO_REAL 正向控制）。任一门禁失败即停止并 BLOCKED 报告。
5. **判据冻结**: G3 样本量 = 状态库全体有效起点；配对比较（同状态跨模式）；
   显著性判据与容差数值须在 G3 运行前写入本文件（当前留白=未冻结，禁止先跑后定）。
6. **结果落盘**: 每 (state,cand,mode) 一份 result JSON + summary.json，路径约定
   reports/memory_study/<date>_{g1,g2,g3}/（服务器），证书引用 bank hash、params SHA、receipt verdict。

## 6. 六组 Student 对比计划（B3 阶段）

| 臂 | 架构 | 记忆携带模式 | HO 处理 |
|---|---|---|---|
| GTRXL_BASE | GTrXL (cc2) | window_mem=128 | BASE / HO_REAL |
| GTRXL_HO | GTrXL (cc1 128) | window_mem=128 | HO_REAL 主臂 |
| RMT16_RESET128 | RMT16 (cc4) | carry_mode=reset128 | BASE / HO_REAL |
| RMT16_PERSISTENT | RMT16 (cc4) | carry_mode=persistent | BASE / HO_REAL |
| SLOWGRU_RESET128 | SlowGRU (cc3) | longstate 每 128 重置 | BASE / HO_REAL |
| SLOWGRU_PERSISTENT | SlowGRU (cc3) | longstate 持久 | BASE / HO_REAL |

对比在 FRONT_L2 上进行：primary metric + dense progress 矩阵，共享同一状态库与 capture bank。
**与 tier3 V3 既有结论的关系**: V3 结论为 INCONCLUSIVE_FULL_TIE（BASE_GTRXL ≡ RESET128，
六组全平）。本研究不推翻该结论；HO 预热是新增干预维度：若 G3 显示 HO_REAL 在特定携带模式上
产生可测差异，则说明"记忆内容注入"是被 V3 无干预对照掩盖的机制变量；若无差异，则 FULL_TIE
在干预意义下得到加强。两种结果都是信息。

## 7. 偏差与纪律记录（披露）

- **D1 执行者偏差**: 本任务原计划委派 deepseekv4flash_worker 执行；本会话委派通道被证实损坏
  （旧实例携带无关监控上下文、新实例未激活、12 分钟文件监视零产出），由总控（director）直接执行，
  与既往两次披露先例同性质。
- **D2 提取范围**: 规范字面为 tools/；实际一并提取 schemas/（7）与 configs/（5），
  原因是冻结代码以 LF-SHA 绑定 schema 文档且引用 checkpoint 契约；三者对源分支 diff 均为 0 行。
  audit_outputs/（45 件审计证据）未提取（当前失败项不依赖；留给服务器阶段按需处理）。
- 纪律: 无 git push；无全局 git/ssh 配置改动；禁碰清单（E3 脏文件与 e3_litesim 在制品）零触碰；
  冻结 tier3/schemas/configs 提取后零修改。

## 8. 本地证据索引

- tier3 提取提交: `feat(eval): import frozen tier3 scaffolded evaluation tooling verbatim ...`（60 文件, 33451 行, diff=0）
- tier3 self_test: TIER3_AGGREGATE_SELF_TEST_PASS (modules=11, negative_tests=FAIL0)，无 jax
- memory_study 测试: `python -m pytest gpu1_aggregation_siege/tests/memory_study -q` → **47 passed**（无 jax）
- CLI 端到端（SYNTHETIC）: `python gpu1_aggregation_siege/scripts/run_memory_study_floor23.py --mode synthetic --out-root <tmp>` 退出码 0；`--mode real` 本地结构化 BLOCKED 退出码 2