# Student Candidate Registry v1 —— 性能优先候选盘点报告

- 任务：CC3_ALL_EXISTING_STUDENT_CANDIDATE_REGISTRY_AND_PERFORMANCE_PRIORITY（只做候选盘点、身份归一化与优先级排序；**未开始 D052 v2 实验，未授权任何训练**）
- 日期：2026-07-30 ｜ 分支：`henry/d052-canonical-refactor` @ 12ac24d7 → 1b52b47（盘点轮单提交）→ 本轮修正 follow-up 提交（见 §9；不 amend 1b52b47）
- 产物：`student_candidate_registry_v1.json`（机器可读，23 候选 = 21 盘点 + 2 修正轮新增，见 §9）+ `student_candidate_registry_v1_schema.json`（JSON Schema draft-07，修正轮扩展 4 个候选字段 + 4 个顶层字段）+ 本报告（§9 为修正轮）
- 证据来源：`experiments/henry_dicode_student_upgrade/` 归档（13 阶段目录 + inventory + MANIFEST.sha256 + README/TERMINOLOGY/SCIENTIFIC_STATUS/EXPERIMENT_TIMELINE/EVALUATION_PROTOCOL/ARTIFACT_MANIFEST 六份治理文档）与 `gpu1_aggregation_siege/reports/` 冻结标签；CC3 四路并行取证 agent（Base/GTrXL+D052、RMT16、W512+P2、SlowGRU/长记忆 bakeoff/P7-P9）

## 1. 关键前置事实（决定一切 readiness 判定）

1. **所有 checkpoint 实体均不在本仓库**。归档策略（ARTIFACT_MANIFEST.md）明确排除 checkpoint 实体、Orbax 权重、replay buffer、>5MB 文件；本地仅存 manifest.json / report / 源码镜像与极少数 orbax `_CHECKPOINT_METADATA` 存根。实体在服务器 `oseasy@172.25.14.221` 的 `/home/oseasy/experiments/...`。→ `checkpoint_file_sha256`/`params_sha256` 一律为**报告级转录**，本地不可复算（除 MANIFEST.sha256 所登在库小文件）。
2. **canonical 身份锚**：`STUDENT_OBS_DIM=8335`（= base 8268 + 67 achievement multi-hot，冻结于 `gpu1_aggregation_siege/d052/legacy/canonical_constants.py`，判定 `D052_STUDENT_OBS_DIM_8335=PASS`）；`action_dim=43`（运行时 `env.action_space(env_params).n`，被 `run_p9_authentic_98304.py:198` assert 固化）。所有 21 候选 obs/action 均与此一致（D052 以小模板探针载入验证）。
3. **共享基座 ckpt17500**（`base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500`，params `d4e85af5…`，被 6+ 独立文件交叉印证）是一切谱系根。注意三种加载态参数树哈希不同：纯 GTrXL `5dfe67dd…`、Henry 原生 TrainState `d4e85af5…`、W512 合并初始化 `59425263…`——同一基座、不同参数树表示，不是不同 checkpoint。
4. **科学现状**：`GLOBAL_LONG_MEMORY_WINNER=NONE`——没有任何长记忆机制被证明击败 Control；所有正向 SR 要么瞬态（SlowGRU）、要么负结果（P8/P9/W512/RMT16 架构净负/P7 无效）。最强持续表现仍属纯 base GTrXL（教师 ckpt17500 与 Control 线）。
5. **评估口径不可混用**（EVALUATION_PROTOCOL.md）：CC1 冻结口径（256 世界，seed42，随机策略，spawn_floor2，max 4096 步，DK ever-set SR）与 exploratory FULL-style 口径（seed_base 100000）数字不同；Stage4-native DK SR ≠ Official FULL Tier3 SR。**全部候选均无 OFFICIAL_FULL/FRONT/BACK 记录（NONE_RECORDED）**，排序只能基于 Stage4 DK SR + 可持续性 + 证据链。

## 2. 盘点范围与排除（§一）

已扫描并登记 21 个真实候选，覆盖任务列举的全部族：Base/GTrXL（教师 + Control 连续重训线 + P9）、RMT16（bakeoff 双臂 + Phase4A smoke）、P2 Replay（P2-Full-A-v1、P2-v1-lite AMAGO-style）、历史 bakeoff（W512 四臂、RMT16、SlowGRU、EventMem32）、AMAGO-style/长上下文（P2-v1-lite、W512 long-context、P7 EgoMap）、其他 canonical（P8 Summary、SlowGRU 归因双臂）、D052 矩阵 BEST cell。

**按任务 §一 强制排除、未登记为 ELIGIBLE 的对象**：
- D052 25-cell 矩阵：`INVALID_DATA_CODE_ONLY`、数据按请求删除、小模板非 canonical → 仅登记 BEST cell 一个条目，`BLOCKED_IDENTITY`/`INVALIDATED`；
- RMT16 Phase4A smoke：仅 step0、update 0 崩溃 → `ENGINEERING_ONLY`/`BLOCKED_IDENTITY`；
- W512-Persistent-PPO、RMT16-Persistent：checkpoint SHA 未存档（REPORT_ONLY）→ `BLOCKED_IDENTITY`；
- P7 EgoMap 臂：无 SHA + 训练锚点校验 FAIL + 超参漂移 → `INVALIDATED`/`BLOCKED_IDENTITY`；
- P2-v1-lite：run manifest 无 params_sha256、base 谱系表述张力、无评估 → `BLOCKED_IDENTITY`；
- SlowGRU Detach/MatchedMLP：纯归因臂 → `ABLATION_ONLY`/`BLOCKED_IDENTITY`（§四 明确不得进 CC2 推荐）；
- **无任何 synthetic / random-init / NOOP / 纯配置目录被登记为 ELIGIBLE**（P2-v1-lite 的 scratch-init 因身份不可核验而 BLOCKED，未获任何 ELIGIBLE 级）。

## 3. 优先级排序（§三，性能优先，8 条规则应用）

`ROUND_1_CANDIDATE_ORDER`（8 个）：

| 序 | candidate_id | 依据（规则号） |
|---|---|---|
| 1 | CONTROL_CONTINUOUS_98304 | 真实 ckpt(1) + canonical 身份(2) + DK 任务(3)；FULL/FRONT/BACK 双方均无记录 → 规则 4 平手 → 规则 5 预算最大（matched 98304）；规则 8 证据链最完整（哈希链 + 本地 orbax 存根 + 双口径互证）。canonical 权威对照 |
| 2 | BASELINE_TEACHER_CKPT17500 | 规则 4 最强单点之一（39.45%/38.28%）；规则 8 谱系根、SHA 六文件互证；预算 17500 不匹配（规则 5 弱于 Control），故列第 2 |
| 3 | SLOWGRU_RESET128_LONGRUN_98304 | 规则 4 全项目最强峰值（44.53%，+8.20pp p=0.038）；规则 5 matched 98304 长训存在；**但 @98304 反转 −8.59pp（TRANSIENT_SIGNAL）、carry 因果为负**——性能优先入选，风险如实标注 |
| 4 | SLOWGRU_PERSISTENT_24576 | 规则 4 次强（42.97%）；真实 SHA；无匹配 98304（规则 5 弱） |
| 5 | EVENTMEM32_RESET128_24576 | 规则 4（40.23%）；真实 SHA；carry 因果为负（REGULARIZATION_ONLY） |
| 6 | EVENTMEM32_PERSISTENT_24576 | 规则 4（37.50%）；真实 SHA |
| 7 | W512_RESET128_P2REPLAY_24576 | 规则 4（37.11% ns）+ 规则 6 replay 机制完整且工程健康（accepted 11/11）；manifest 本地可验 |
| 8 | P2_FULL_A_V1_98304 | 规则 4 偏弱（@98304 30.08%）但规则 5（matched 98304）+ 规则 6（V-trace+hindsight+AWR 机制最完整、ENGINEERING_PASS）+ 规则 8（谱系哈希链自洽）共同支撑；且为 §四 默认 PRIMARY 谱系 |

候补（9–13，未入 ROUND_1）：W512_PERSISTENT_P2REPLAY（35.16%）、RMT16_PERSISTENT（27.34%，证据链不全）、RMT16_RESET128（11.33%）、W512_RESET128_PPO（2.73%，collapse 证据）、P8（−14.06pp，FAILED_HARMFUL）、P9（−4.30pp，NO_POSITIVE_SIGNAL）。W512/RMT16 PPO 与 P8/P9 虽为真实 checkpoint，但性能为负/崩溃，按性能优先不入前 8；P7/P2-v1-lite/D052/两归因臂因身份或资格问题不参与排序。

## 4. 给 CC2 的推荐（§四）

- （**本节已被 §9 修正轮覆写**：PRIMARY_NEW_TRAINING_CANDIDATE=NONE_PENDING_FAST_EXISTING_STUDENT_SCREEN；P2_FULL_A_V1_98304=DEPRIORITIZED_KNOWN_UNDERPERFORMER；新增 PRIMARY_ARTIFACT_RECOVERY=CONTROL_CONTINUOUS_98304 / SECONDARY_ARTIFACT_RECOVERY=BASELINE_TEACHER_CKPT17500。以下为盘点轮原文，保留留痕。）
- **PRIMARY_NEW_TRAINING_CANDIDATE = P2_FULL_A_V1_98304**（Base GTrXL + Original V-trace Replay 谱系）。按任务默认条款：除非发现**已存在可验证的匹配 98304 checkpoint**，否则该谱系默认 PRIMARY。盘点结论：P2-Full-A Resume RUN1@98304（params `67689592…`）仅**报告级**存在（manifest 已镜像但实体在服务器、本地不可复算 → `REAL_CHECKPOINT_REPORTED` 而非 `VERIFIED`），且性能未超 Control（@98304 30.08% vs 34.38%，`NO_DELAYED_ONSET_WITHIN_98304=true`）。故 `matched_98304_already_exists=false`，默认条款生效。推荐理由：工程健康（ENGINEERING_PASS、any_nan false、KL 门生效）、replay/hindsight/AWR 机制最完整、谱系哈希链自洽（LevelB@24576 `bd084220…` → resume@98304）。**张力如实上报**：该谱系已有一次报告级 98304 负结果；是否值得在 canonical 冻结 provenance 下重训，由总监/CC2 裁定——若总监认为重跑价值低，可将 Control 线（第 1 位）视为事实最强基线。本 registry 不授权训练。
- **SECONDARY_NEW_TRAINING_CANDIDATE = SLOWGRU_PERSISTENT_24576**。尚无匹配 98304 的高性能候选中性能最强（42.97%），非纯消融。风险：其 Reset128 同胞长训 @98304 已反转有害（−8.59pp），carry 因果为负（−1.56pp ns）——新匹配训练恰为检验 persistent-carry 变体在匹配预算下是否可持续。候补顺位：W512_RESET128_P2REPLAY_24576（37.11%，replay 机制完整、工程健康）。
- 两者均属高性能候选而非纯消融；SlowGRU Detach/MatchedMLP 两归因臂依规排除。

## 5. 给 CC4 的薄 adapter 路由（§五）

**原则：只允许薄 adapter，不得为每个 checkpoint 复制完整 evaluator。** 所有候选 obs/action 一致（8335/43），greedy 推理统一走 canonical 评估入口（`gpu0_training_mechanisms/src/dicode/craftax_evaluation.py`，action_space(ctor).n=43）；差异只在 loader + network constructor + memory state 初始化。按 adapter_family 分 6 族（覆盖 13 个 ELIGIBLE_AFTER_* 候选）：

| adapter_family | 覆盖候选 | checkpoint loader | network constructor | memory state 初始化 | required source files（SHA 见 registry 各条 driver/policy_source_sha256） | manifest 必需字段 | 缺失的最小 adapter 工作 |
|---|---|---|---|---|---|---|---|
| THIN_GTRXL128_PICKLE_CONTROL | CONTROL_CONTINUOUS_98304、P9_AUTHENTIC_RESET_98304 | pickle `full_state.pkl` → 取 params pytree（P9 与 Control 无网络改动，loader 完全共享） | ActorCriticTransformer（embed256/heads8/layers2/win128/gating_bias2.0，80 leaves） | GTrXL window-128 零初始化（mem (B,128,2,256)、mask (B,8,1,129)） | ppo_tr.py `faa561c0…`、network.py（源 SHA 未记录→**需服务器补 SHA**）、craftax_evaluation.py | step、params_sha256、format | 仅 pkl→params 解包 + greedy apply；无新网络代码 |
| THIN_GTRXL128_ORBAX_BASE | BASELINE_TEACHER_CKPT17500 | orbax `load_weights_only`（基座为 orbax 目录） | 同上（大 GTrXL-128） | 同上 | s175_verify_probe.py（载入参考）、network.py | params_sha256 | orbax restore 封装（注意 Phase4A 曾报 Orbax restore warnings，需静默路径核验） |
| THIN_GTRXL128_P2REPLAY_PICKLE | P2_FULL_A_V1_98304 | `p2_full_a_pure_pickle_v1`（纯 params pickle；**replay buffer 未持久化，adapter 不恢复 buffer**） | ActorCriticTransformer + sparse memory anchors（ANCHOR_INTERVAL=128，burn-in≤128，stop_gradient） | window-128 零初始化 + anchor burn-in | full_p2_learner.py `c374f0aa…`、full_p2_core.py `2e0fa3c6…`、memory_anchor.py `49ac6241…`、checkpointing.py `93a52648…` | format、params_sha256、anchor 计数 | anchor burn-in 复现（评估前 ≤128 步预热）；resume manifest 缺 driver SHA → **需服务器补** |
| THIN_GTRXL128_SLOWGRU_PICKLE | SLOWGRU_RESET128_LONGRUN_98304、SLOWGRU_PERSISTENT_24576（两归因臂若解禁亦共用） | pickle `full_state.pkl`（含 memories/mask/midx） | ActorCriticTransformer + SlowGRU（SLOW_INTERVAL=32 pool、GRUCell 256、zero-init slow_to_actor） | slow state 零初始化 + full_state.pkl 内 memories 还原（roundtrip/EXACT_RESUME 已验证） | slowgru_network.py `b2652105…`、launcher `6585f4b0…`/`86bb12c4…` | step、params_sha256、file sha | slow pool 状态解包（full_state.pkl 已含 memories，直接还原） |
| THIN_GTRXL128_EVENTMEM32_PICKLE | EVENTMEM32_RESET128_24576、EVENTMEM32_PERSISTENT_24576 | pickle `full_state.pkl` | ActorCriticTransformer + EventMemory32（32-slot event buffer） | event buffer 零初始化 + pkl 内 memories 还原 | eventmem network `6a5cd695…`、launcher `fbca81d3…`/`a4662efb…` | step、params_sha256 | event-slot 写入规则在推理期复现（32-slot 环形） |
| THIN_W512_P2REPLAY_PICKLE | W512_RESET128_P2REPLAY_24576、W512_PERSISTENT_P2REPLAY_24576、W512_RESET128_PPO_24576（W512_PERSISTENT_PPO 解禁后亦共用） | pickle `params.pkl` + 旁置 manifest.json（本地可验） | ActorCriticTransformerW512（5,268,013 params；window128 + W512 long buffer 384 + sparse anchors；gate 零初始化 → init 与 ckpt17500 bit-exact） | window-128 + long buffer 384 零初始化 | network_w512.py `8d1824d2…`、w512_memory.py `ee89fd0b…`、run_w512_p2_levelB.py `016ccb59…` | step、params_sha256、updates、replay_size | long buffer + gate 前向复现；**replay buffer 组成未持久化，评估期不恢复**（冻结审计已列此限制） |
| （未就绪族）THIN_RMT16_PICKLE / THIN_GTRXL128_SUMMARYMEM_PICKLE / THIN_GTRXL_EGOMAP_PICKLE / THIN_GTRXL_LONGCONTEXT_PICKLE | RMT16_RESET128（ELIGIBLE_AFTER_MATCHED_98304）；RMT16_PERSISTENT / RMT16_PHASE4A_SMOKE / P8 / P7 / P2-v1-lite（当前 BLOCKED，不在首轮路由） | 各为 pickle（RMT16 params.pkl；P8 full_state.pkl；P7 params+carry pkl→orbax 转换；P2-lite manifest 无哈希） | RMT16（rmt_num_tokens=16）/ Summary ring buffer（K64×N16，actor-only attn）/ EgoMap（map_bank[9,32,32,9]）/ long-context GTrXL | RMT memory tokens 零初始化（gate=0）/ summary buffer 零初始化 / EgoMap zero-init fusion | network_rmt16.py `b5c37d7a…` + rmt16_memory.py `17e1a614…` / p8_network `5fd11efd…` / — | step、params_sha256 | RMT16：仅 RMT16_RESET128 可路由（manifest 完整）；P8 虽 ELIGIBLE_AFTER_THIN_ADAPTER 但 FAILED_HARMFUL，首轮不建议；P7/P2-lite/RMT16-Persistent 需先补 SHA/manifest 解禁 |

每个 adapter 的共同最小契约：(a) 从 manifest/summary 读 `params_sha256` 并与加载后参数叶 `np.tobytes()` 串接复算一致（算法见 `run_w512_p2_levelB.py` L217-231）；(b) greedy 推理统一入口，不复制 evaluator；(c) 记录 adapter 自身 source SHA 与所消费 manifest SHA 进评估证书。

## 6. D052 边界（§六）

```
D052_REAL_POOL        = NOT_STARTED
D052_STUDENT_PROFILE  = NOT_FINALIZED
D052_CURRICULUM       = NOT_GENERATED
winner_candidate_id   = null（预留；真正 winner 产生前停止 D052）
winner_profile_input_schema = null（预留）
```

本轮未生成任何 D052 cell、未调 LLM、未训练；registry 中 D052 历史 BEST cell 仅作 INVALIDATED 盘点条目，不得作为 winner 输入。

## 7. 本地可验证 vs 报告记录（核验记录）

本轮实际复算并命中 `MANIFEST.sha256` 的在库文件（5 件，全部 MATCH）：
- `12_w512_phase4a/.../w512_persistent_p2replay_24576/checkpoints/24576/manifest.json` → `60ed1cf9…`
- `12_w512_phase4a/.../w512_reset128_p2replay_24576/checkpoints/24576/manifest.json` → `4f2a154d…`
- `10_long_memory_carry_phase2/.../gpu1_rmt16_reset128_training/train_24576/checkpoints/24576/manifest.json` → `91bb681b…`（内文 params_sha256=`b4ef22bc…`，与 cc1 报告逐字一致）
- `10_long_memory_carry_phase2/.../gpu0_w512_reset128_training/train_24576/checkpoints/24576/manifest.json` → `a43c0c16…`（内文 params=`51388e21…`，与 cc1 逐字一致）
- `04_p2_full_a_v1/.../eval_prep/control_orbax/98304/_CHECKPOINT_METADATA` → `e6e9dc10…`

其余一切 checkpoint 实体哈希（full_state.pkl / params.pkl / orbax 权重 blob）均为报告级，本地不可复算；跨文件互证未发现篡改/不一致（除报告中本就声明的 P7 训练锚点失配与 P8 空 teacher_init 字段，均已写入对应 blockers）。D052 逐 cell 哈希与 SR 因 2026-07-26 按请求删除而永久 NONE_RECORDED。

## 8. 约束合规

- 只读盘点：未改训练代码、未改 evaluator、未改迁移包原件、未覆盖任何历史产物；
- 未登记任何 synthetic/random-init/NOOP/纯配置/损坏 checkpoint 为 ELIGIBLE；身份不可核验者一律 BLOCKED_IDENTITY；
- 未授权训练（TRAINING_TO_98304_AUTHORIZED=false 继承）、未调 LLM、未 push；
- 单提交 `docs(student): register performance-screen candidates`，仅含 registry JSON + schema + 本报告；
- 未从总结重新生成任何 JSON/JSONL——所有 SHA 与性能数字均逐字转录自在库 manifest/report 原件。

## 9. 修正轮（2026-07-30，最小修正；不重新盘点全部项目）

吸收 CC2→CC4 direct98304 交接真实证据，修正训练推荐与 adapter 优先级。仅改 registry JSON / schema / 本报告三件；盘点轮提交 1b52b47 不动（不 amend）。

### 9.1 新增两个 ELIGIBLE_NOW 候选（CC4 外部已验证 artifact）

| candidate_id | final_params_sha @98304 | base 匹配 | 工程门禁 | checkpoint（服务器，CC4 持有） |
|---|---|---|---|---|
| PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 | `aa6ba44040a0742b`（16 字符前缀，addendum 逐字） | d4e85af5… PASS；step0 `2f8cd875993ae103` bit-exact | 全 PASS（rc=0，证书 PASS） | `/home/oseasy/cc2_data/cc2_runs_76b294b/runs/RMT16-LONG98304-PERSISTENT/ckpt/98304/full_state.pkl` |
| RESET128_RMT16_ORIGINAL_VTRACE_98304 | `78a14cc6e9ccdeb2`（16 字符前缀，addendum 逐字） | 同上 | 同上 | `.../RMT16-LONG98304-RESET128/ckpt/98304/full_state.pkl` |

- 新增字段：`evidence_location_class=EXTERNAL_VERIFIED_ARTIFACT`、`artifact_owner=CC4`、`artifact_handoff_required=true`；`evidence_class=REAL_CHECKPOINT_VERIFIED`、`budget_class=MATCHED_98304`。
- **readiness 策略修正（按总监指令）**："不得因为 checkpoint 未提交到 Git 而标记为不可运行"。两者已具备：真实 final98304 checkpoint（每臂 13 个、both_arms_complete=true）、checkpoint/params SHA（final 98304 pkl 全文件 SHA 交接未录——如实记 NONE_RECORDED_FOR_FINAL_98304_PKL；身份证据落在 final_params_sha + per-arm scientific_config/certificate/config_file 全 SHA 链）、CC4 RMT16 adapter、真实前向与 interface smoke、可访问的 CC4 artifact handoff → 登记 **ELIGIBLE_NOW**。盘点轮的保守 BLOCKED 策略仅继续适用于身份不可核验者。
- 证据链（本地控制平面 `D:/Projects/dicode-codex-director/orchestration/control/`，不在本仓库，已在条目内标注）：CC2→CC4 交接 MD（RUN_COMPLETE_ALL_GATES_PASS）+ run_completion_addendum.json（pair_pass=true，total_env_steps=98304，total_updates=48，elapsed 7986s，GPU2/GPU3 only）+ 20260729T210800 发射报告 + 20260729T195618 postjax hotfix 报告 + tournament_v1 录取记录（budget_tagging/blocker_ledger/readiness_status，逻辑名与 SHA/budget-tag 由本 registry 权威提供）。
- `performance=null`：CC2 交接 §10 明确不交付成功率/科学解释；性能由 ROUND_1 fast-screen（CC4 统一评估器 + 冻结 FRONT/BACK 状态集）测定。
- 跨阶段锚点复验：base inner-params `d4e85af5…` == 本 registry 教师/Control 锚点；step0 `2f8cd875993ae103` == bakeoff RMT16 与 Phase4A smoke step0 全文 `2f8cd875993ae10385dbb5dae530a557a0eb1008541b98de416cc7ae7ba2d93b` 前缀（前缀匹配）→ matched-init 链成立。

### 9.2 训练推荐修正（覆写 §4）

- `PRIMARY_NEW_TRAINING_CANDIDATE = NONE_PENDING_FAST_EXISTING_STUDENT_SCREEN`：`matched_98304_already_exists` 由 false 升为 true（已有可验证匹配 98304 真实 artifact），默认条款被推翻；当下不建议新训练，先筛既有 artifact。
- `P2_FULL_A_V1_98304.priority = DEPRIORITIZED_KNOWN_UNDERPERFORMER`（@98304 30.08% < Control 34.38%，NO_DELAYED_ONSET_WITHIN_98304=true），退出 ROUND_1。
- 新增顶层字段：`PRIMARY_ARTIFACT_RECOVERY = CONTROL_CONTINUOUS_98304`、`SECONDARY_ARTIFACT_RECOVERY = BASELINE_TEACHER_CKPT17500`。
- `SECONDARY_NEW_TRAINING_CANDIDATE` 保留 `SLOWGRU_PERSISTENT_24576`（指令未撤销；仅在 fast-screen 证明既有 artifact 全部不足时进入考虑）。

### 9.3 adapter 优先级（覆写 §5 路由次序）

```
FAST_ADAPTER_PRIORITY = [EXISTING_RMT16_ADAPTER, THIN_GTRXL128_PICKLE_CONTROL, THIN_GTRXL128_ORBAX_BASE, THIN_GTRXL128_SLOWGRU_PICKLE]
DEFERRED_AFTER_FAST_SCREEN = [THIN_GTRXL128_P2REPLAY_PICKLE, THIN_GTRXL128_EVENTMEM32_PICKLE, THIN_W512_P2REPLAY_PICKLE, THIN_RMT16_PICKLE, THIN_GTRXL128_SUMMARYMEM_PICKLE, THIN_GTRXL_EGOMAP_PICKLE, THIN_GTRXL_LONGCONTEXT_PICKLE]
```

### 9.4 ROUND_1 fast-lane（重排，6 条；schema maxItems=8，下限 4）

`PERSISTENT_RMT16_ORIGINAL_VTRACE_98304 → RESET128_RMT16_ORIGINAL_VTRACE_98304 → CONTROL_CONTINUOUS_98304 → BASELINE_TEACHER_CKPT17500 → SLOWGRU_RESET128_LONGRUN_98304 → SLOWGRU_PERSISTENT_24576`

候补 `round_1_alternates`（7–9 位）：`EVENTMEM32_RESET128_24576 → EVENTMEM32_PERSISTENT_24576 → W512_RESET128_P2REPLAY_24576`。

- 移出 fast-lane：`P2_FULL_A_V1_98304`（降权 DEPRIORITIZED_KNOWN_UNDERPERFORMER）；`W512_RESET128_P2REPLAY_24576`、`EVENTMEM32_*_24576` 两名（其 adapter 族 THIN_W512_P2REPLAY_PICKLE / THIN_GTRXL128_EVENTMEM32_PICKLE 均标记 DEFERRED_AFTER_FAST_SCREEN → 不得占 fast-lane 席位，降为候补）。
- 可执行性（§9.5 验证脚本复核）：6 个席位逐条被 FAST_ADAPTER_PRIORITY 覆盖——前 2 位 ELIGIBLE_NOW × EXISTING_RMT16_ADAPTER（CC4 已持有，零新 adapter 工程）；CONTROL_CONTINUOUS_98304 → THIN_GTRXL128_PICKLE_CONTROL；BASELINE_TEACHER_CKPT17500 → THIN_GTRXL128_ORBAX_BASE；SLOWGRU 两名 → THIN_GTRXL128_SLOWGRU_PICKLE。无 DEFERRED 族成员占位。P9_AUTHENTIC_RESET_98304 虽属 PICKLE_CONTROL 族且 matched 98304，但性能 −4.30pp@98304（NO_POSITIVE_SIGNAL），按性能优先不入 fast-lane（如 CC4 需要第 7 席可由总监裁定补入）。

### 9.5 计数自洽

```
total_candidates_found = 23 = ELIGIBLE_NOW 2 + ELIGIBLE_AFTER_THIN_ADAPTER 5 + ELIGIBLE_AFTER_MATCHED_98304 8 + BLOCKED_IDENTITY 8 + REJECTED 0 ✓
```

（jsonschema draft-07 验证、直方图自洽、ROUND_1 可执行性三项复核结果见修正轮提交说明与验证脚本输出。）

### 9.6 合规

不重新盘点全部项目；不 amend 1b52b47；未授权训练、未调 LLM、未触碰任何 checkpoint 实体；两新候选 evidence_refs 为本地控制平面路径（非本仓库），已显式标注；单 follow-up 提交 `fix(student): bind verified RMT16 artifacts and correct SOTA priorities`，随后非强制 fast-forward push 两个提交（盘点轮 + 修正轮）。

**两轮均到此停止，等待总监复核。真正 winner 产生前，D052 保持 NOT_STARTED。**
