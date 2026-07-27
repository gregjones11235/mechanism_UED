# RMT16 Phase4A-v2 — 代码审计报告

**基线**：`henry/reviewed-rmt16-l512-probe` @ `d3c8c7d6`
**分支**：`henry/rmt16-phase4a-v2-original-vtrace`
**审计范围**：本轮（§二–§十三）所有新增/修改文件。所有路径相对
`gpu2_rmt16_phase4a_snapshot/`。

> 授权说明： standing 规则原为"不改 frozen 模块"。本轮 Phase4A-v2 任务书**明确授权**为达成
> §三/§五/§七/§八 所需的**加性**修改。本报告对每处触及 frozen 模块的改动单独列出并论证其加性。

---

## A. 新增文件（全部加性，不改任何既有行为）

| 文件 | 作用 | 依赖 | py_compile |
|---|---|---|---|
| `runtime/experiment_src/phase4a_v2_counters.py` | §二/§三 拆分计数器 + 精确 resolved-step 公式 + §八 防火墙计数 | 纯 Python（无 JAX/numpy） | OK |
| `runtime/experiment_src/launch_phase4a_v2.sh` | §二 未来 launcher：真实退出码/PID/完成时间捕获 | bash | `bash -n` OK |
| `tests/config_diff_validator.py` | §九/GATE14 单变量 config diff | yaml | OK（本地实跑 PASS） |
| `configs/rmt16_phase4a_v2_persistent.yaml` | §六/§九 预注册正式 config（Persistent 臂） | — | — |
| `configs/rmt16_phase4a_v2_reset128.yaml` | §六/§九 预注册正式 config（Reset128 臂） | — | — |
| `tests/test_phase4a_v2_gates.py` | §十 15 门禁测试 | numpy / 可选 jax | （#22 实跑） |
| `tests/recompute_probe_step.py` | §二/GATE1 离线重算 first_ge512 精确 step | numpy | （#22 实跑） |
| `reports/rmt16_phase4a_v2_design.md` | 设计报告 | — | — |
| `reports/rmt16_phase4a_v2_code_audit.md` | 本审计 | — | — |
| `reports/rmt16_phase4a_v2_known_limitations.md` | 已知局限 | — | — |
| `reports/rmt16_phase4a_v2_test_report.md` | 测试报告（#22 填实测） | — | — |
| `reports/rmt16_l512_probe_step_correction.md` | §二 step 重算前后对照（#22 填实测） | — | — |

---

## B. 修改的 experiment_src 文件

### B1. `runtime/experiment_src/rmt_collect.py`（§二/§三）
- **新增 import**：`from phase4a_v2_counters import completion_resolved_env_step`（纯函数）。
- **签名新增两个 keyword 参数**（均默认 `None`）：`outer_update_index`、`policy_version`。
  `None` 时分别回退到 `collected_update_count` / `outer_update_index`，故旧调用方（不传新参）行为不变。
- **episode_records 新增键**：`completion_resolved_env_step`（权威）、`outer_update_index`、
  `policy_version`、`completion_global_step_deprecated=True`；保留旧 `completion_global_step`
  （仅历史对照）。`update_index` 现取自 `outer_update_index`（off 模式与旧值相等）。
- **trajectory 构造**：`collected_update_count` 现写 `outer_update_index`（legacy schema 兼容），
  新增 `outer_update_index` / `policy_version_at_collection`。
- **`pending.reset_slot`**：`policy_version` 现取**接受的** policy version（off 模式 == 旧值 → 逐位一致）。
- **加性论证**：所有改动为新增 dict 键 / 新增旁路取值；进入 PPO/rollout/RNG 的数组与随机流**未改**。
  off 模式下 `outer_update_index == policy_version == 旧 collected_update_count`，故 GATE 13 逐位不变。

### B2. `runtime/experiment_src/train_rmt16_p2replay.py`（§三/§四/§六/§七/§八）
- **CLI**：删 `--replay {on,off}`；新增**必填无默认** `--replay_mode {off,original_vtrace,full_p2_legacy}`、
  `--allow-full-p2-legacy`、`--sequence_length`（默认 129）。`full_p2_legacy` 缺授权 → `ap.error`（退出码 2）。
- **常量**：`SEQUENCE_LENGTH`、`SEGMENT_LEN=128`、`ENGINEERING_LONG_WINDOW_MODE=512`；
  `original_vtrace` 强制 `sequence_length>128`。
- **计数器**：实例化 `Phase4ACounters`；PPO 后 `on_outer_update + on_ppo_accepted`；replay 按
  committed/kl-rejected 分别 `on_replay_policy_committed`/`on_replay_kl_rejected`；legacy 路径递增
  防火墙计数。新增 `replay_sample_rng=np.random.RandomState(seed+7)`、`replay_sequences_consumed`。
- **collect 调用**：传 `outer_update_index=u, policy_version=counters.policy_version`。
- **replay 块重写为模式分发**：
  - `original_vtrace` → `sample_eligible(SEQUENCE_LENGTH, replay_sample_rng, K_BATCH)` +
    `RL.original_vtrace_update_rmt(...)`；NOT_READY 显式跳过；每次更新后 `assert_hindsight_awr_disabled()`。
  - `full_p2_legacy` → 保留原 K_BATCH relabel + `full_p2_update_rmt` 路径（仅审计；需授权）。
  - EMA：`if not did_replay_update: ema_update(...)`（off 模式每迭代 EMA == 旧 `if not REPLAY_ON`）。
- **manifest**（`_phase4a_v2_manifest_fields`）：`sequence_length/segment_len/crosses_boundary/
  replay_mode/hindsight/awr/w_original_vtrace/allow_full_p2_legacy`，写入每个 checkpoint 与 summary。
- **train_state 计数器**：新增 `replay_sequences_consumed`、`replay_sample_rng_state`、`phase4a_v2`
  快照（GATE 12）。
- **终局防火墙**：`replay_mode in {off,original_vtrace}` → `assert_hindsight_awr_disabled()`。
- **summary**：新增 `phase4a_v2_counters / replay_update_count / accepted_replay_policy_update_count /
  replay_attempt_count / replay_sequences_consumed / policy_version / outer_update_index /
  global_env_steps / matched_replay_protocol_ready`。
- **probe**：`first_ge512` 增 `first_ge512_resolved_env_step`（权威）+ 标 deprecated。
- **修复所有残留 `args.replay`**（已删参数）→ `REPLAY_MODE`；`grep` 确认零残留。
- py_compile：OK。

---

## C. 修改的 frozen 模块（**已授权**；逐处论证加性）

### C1. `runtime/frozen_modules/rmt_replay_buffer.py`
- **RMTTrajectory 新增两个带默认值字段**：`outer_update_index: int = 0`、
  `policy_version_at_collection: int = 0`。加性：默认 0，旧构造/旧 pickle 不受影响；不改任何
  采样/anchor/conservation 逻辑。
- **RMTReplaySample 新增字段**：`policy_version_at_collection: int = 0`；`sample()` 填充之。
- **新增 `EligibleSampleBatch` dataclass + `RMTReplayBuffer.sample_eligible(...)`**：纯新增方法，
  复用既有 `self.sample(trajectory_id=..., start_step=...)` 切片逻辑；**不**改 `sample()`/`insert()`/
  `state_dict()`/`hash_digest()` 的既有行为。`sample_eligible` 传显式 id+offset，故不消耗 `self._rng`。
- **冻结系数全部未动**：capacity=64、ANCHOR_INTERVAL=128、MIN_SEQUENCE_LENGTH=129、确定性采样、
  无成功/质量/TD-error 优先。py_compile：OK。

### C2. `runtime/frozen_modules/rmt_replay_learner.py`
- **仅追加**模块常量 `W_ORIGINAL_VTRACE=1.0` 与两个新函数 `compute_loss_original_vtrace_rmt` /
  `original_vtrace_update_rmt`。**未改** `compute_loss_rmt` / `full_p2_update_rmt` /
  `reconstruct_rmt_batch` / `_make_scan_rmt` / `_target_scan_rmt` / `_window_log_softmax_rmt` 等
  既有函数（legacy 路径逐位不变）。新函数复用既有 helper（V-trace、KL gate、EMA、grad-clip）。
- **结构防火墙**：新函数体仅引用 `V`(vtrace)、`FPL`(full_p2_learner)，**无** `A.`(awr)、
  `relabel`、`samples_rel`、`recon_r`、`target_vals_r`（已 grep 验证：命中全在注释/文档字符串）。
- py_compile：OK。

### C3. 未触及的 frozen 模块（保持基线 SHA）
`rmt_ppo.py`、`rmt_hindsight.py`、`full_p2_learner.py`、`replay_buffer.py`、`vtrace.py`、`awr.py`、
`pending_episodes.py`、`network_rmt16.py`、`rmt16_memory.py`、`rmt_memory_anchor.py`、`memory_anchor.py`、
`hindsight.py`、`rng_utils.py` 均**未修改**。

---

## D. 静态审计结论
- 全部新增/修改 `.py` 通过 `python -m py_compile`；`launch_phase4a_v2.sh` 通过 `bash -n`。
- `args.replay` 零残留（已删参数）。
- off-path 加性 → GATE 13 预期逐位不变（#22 服务器 CPU 实测确认）。
- original_vtrace 结构隔离 → GATE 4/5/6 预期通过（#22 实测 + monkeypatch 确认）。
- config 单变量契约 → GATE 14 本地已 PASS（仅 `carry_mode` 差异）。

---

## E. Phase4A-v2.1 加固审计增补

- `rmt_collect.collect_rollout_rmt`：start 版本读取点位于 `pending.reset_slot(e, ...)` **之前**
  （GATE16 静态索引序检查）；end=当前 policy_version；两条断言在构造 `RMTTrajectory` 前。
- `rmt_replay_buffer.sample()`：`policy_version_start/end/span` 逐字传播，别名绑定 start
  （GATE18 行为验证）。
- `phase4a_v2_contract.py`（新增，纯 Python）：policy-lag 身份 + fail-closed 配置校验 +
  四标签 + exposure 证书规格/比较/门禁；被 launcher、gates、validator 共用。
- `train_rmt16_p2replay.py`：manifest 合入 `policy_lag_runtime_manifest(REPLAY_MODE)` +
  `replay_protocol_labels(...)`；summary 删除 `matched_replay_protocol_ready=`，改输出
  `exposure_certificate`（14 字段）+ 运行时 fail-closed lag 一致性守卫；新增每 outer-update
  exposure 记录（attempt mask / update indices / batch sizes / seq lengths / eligible counts /
  内部 sample_ids & start_offsets）。
- `phase4a_v2_counters.py`：新增 `kl_rejected_replay_update_count`（计数不改 policy 状态；
  off-path 计数等价不变，GATE13 仍 PASS）。
- configs：活动 `max_policy_lag:16` 移除；`policy_lag`/`exposure_contract`/`legacy_full_p2_only`
  两臂逐字相同（GATE14 复验仅 carry_mode 差异；runtime_assignment 仅 gpu_uuid/out_dir 差异）。
- 全部 v2.1 新增/修改 `.py` 通过 `py_compile` 与 `compileall`（本地 + 服务器 CPU）。

---

## Phase4A-v2.2 代码审计补遗（本轮改动文件）

本轮（V2.2）改动严格限定在 `gpu2_rmt16_phase4a_snapshot/` 内；CC3 / CC4 / Henry-branch 未触碰；
`evidence/raw_probe/` 8 个冻结证据文件逐字节未改（`git diff --exit-code 87d1e55 -- evidence/raw_probe/`
为空）。无阈值／网络／任务／评估器／种子／预算改动。

**修改的源码（`.py`，均通过 py_compile + compileall）：**
- `runtime/experiment_src/phase4a_v2_contract.py`：删除 `PROTOCOL_MATCH_FIELDS`；新增
  `REQUIRED_PROTOCOL_FIELDS` / `canonical_protocol_json` / `protocol_definition_sha256` /
  `missing_required_protocol_fields` / `compare_protocols`（全字典身份 + canonical SHA）；
  `replay_protocol_labels` 增补 rng 细分键与 `learner`/`rng_rule`；Level 1 改走 `compare_protocols`；
  新增 `active_replay_config_manifest` / `legacy_full_p2_manifest` / `assert_no_active_policy_lag_leak`
  （活动域 lag 泄漏扫描，fail-closed）。
- `tests/phase4a_v2_exposure_validator.py`：Level-1 spec 文案更新为“全 canonical 协议身份”；
  report 增补 `PROTOCOL_MISSING_FIELDS_*` / `PROTOCOL_KEYSET_MISMATCH` /
  `PROTOCOL_DEFINITION_SHA256_*`；`_synthetic_summary` 增补 learner/rng_rule/drop/extra 钩子；
  self-test 11→18（协议 SHA 相等、learner/rng_rule 差异、缺字段、多键 keyset、键序不变）。
- `runtime/experiment_src/phase4a_v2_runtime_config.py`（**新增**，纯 Python，无 JAX）：
  `load_formal_config` / `canonical_scientific_config` / `scientific_config_sha256` /
  `build_runtime_scientific_config` / `deep_diff` / `preflight_require_formal_config`（pre-JAX）/
  `validate_arm_binding` / `build_checkpoint_identity` / `verify_checkpoint_params_sha`（冻结期望
  SHA `d4e85af5…`）/ `validate_runtime_against_formal_config`（certificate）/
  `write_runtime_config_certificate` / `certificate_shas_record`。self-test 29/29，负例 28（≥19）。
- `runtime/experiment_src/train_rmt16_p2replay.py`：`--formal_config` 参数；pre-JAX preflight +
  arm 绑定（line 84 < `import jax` line 102）；从真实常量构造 runtime scientific config +
  certificate（binding line 237 < env line 320 < ckpt line 341）；certificate_status≠PASS →
  `SystemExit(FORMAL_CONFIG_RUNTIME_MISMATCH)`（env/ckpt 之前）；base params SHA 加载后比对 +
  certificate 重写；manifest 与 summary 内嵌 `runtime_config_certificate`；`p2_frozen` 改
  `max_policy_lag=None`/`policy_lag_gate_active=False`；summary 增补 active_replay_config /
  legacy_full_p2_only / leak scan。
- `runtime/experiment_src/rmt_collect.py`：episode 完成块构造 `episode_record`，写入
  `policy_version_start/end/span` + 旧 `policy_version`（别名=end, deprecated=True）+ 8 条
  pre-write assert；recompute provenance 键不动（冻结复跑仍 20/6、21/5、8979、BOTH）。
- `runtime/frozen_modules/rmt_replay_buffer.py`：新增 `validate_policy_version_range_fields` /
  `validate_sample_policy_version_range`（只读）/ `RMTTrajectory.validate_policy_version_range`；
  `validate_anchors()` 末尾强制区间校验（insert 即生效）；`sample()` 构造后只读校验。

**修改的配置 / 报告：**
- `configs/rmt16_phase4a_v2_persistent.yaml` & `_reset128.yaml`：`legacy_full_p2_only` 改
  `active:false`（保留文档 `max_policy_lag:16`），两臂**逐字相同**编辑；GATE14 复验仅 carry_mode 差异。
- `reports/rmt16_phase4a_v2_2_labels.json`（新增，§十四 完整标签集，分层发布状态，无范围
  PUSH_PERFORMED 已禁）。
- `reports/rmt16_phase4a_v2_2_final.md`（新增，本轮最终报告）。
- 四份 v2 文档 + exposure_contract.md 各加 V2.2 补遗。

**新增门禁 27–38**（`tests/test_phase4a_v2_gates.py`，全部非 JAX）：协议完整字段（27）、协议
fail-closed（28）、active lag 泄漏（29）、episode 区间记录（30）、轨迹区间校验（31）、样本区间
校验（32）、YAML↔runtime 绑定 PASS（33）、科学字段失配（34）、runtime_assignment 失配（35）、
`--formal_config` pre-JAX（36）、certificate SHA 一致（37）、分层发布标签（38）。门禁总数 38。
