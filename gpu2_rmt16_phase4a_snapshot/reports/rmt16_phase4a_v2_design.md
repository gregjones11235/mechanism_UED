# RMT16 Phase4A-v2 — Original-goal V-trace 实现设计报告

**任务**：`RMT16_PHASE4A_V2_ORIGINAL_GOAL_VTRACE_IMPLEMENTATION`（CC2 / RMT16 / P2 Replay）
**审核基线**：分支 `henry/reviewed-rmt16-l512-probe` @ `d3c8c7d6abd2df3d0ba69dc2c1f326f8668798e5`
**实现分支**：`henry/rmt16-phase4a-v2-original-vtrace`
**冻结结论（本轮不重审）**：`L512_REACHABILITY_BOTH = PASS`（Persistent count_ge512=6/20，Reset128=5/21；Replay=OFF，Hindsight=OFF；探针**不**做 Carry/性能断言）。

本轮**仅**允许：代码修改、CPU 测试、静态审计、不更新参数的小型前向探针、git commit+push。
本轮**禁止**：正式训练、4096 smoke、24576、98304、多 seed、Hindsight、AWR、启动正式两臂、改动 CC3/CC4 分支、merge/rebase/force push/建 PR。`NEW_TRAINING_RUNS=0`。

---

## 0. 设计总原则：严格加性（additive-only）

所有改动遵守两条不可违背的约束：

1. **off-path 逐位不变（GATE 13）**：`replay_mode=off`（含旧探针路径）的训练数值必须与基线
   `--replay off` 完全一致。所有新增字段/计数器/日志都是**加性**的（新增 dict 键、新增带默认值的
   dataclass 字段、新增旁路计数器），不改变任何进入 loss/optimizer/RNG 流的数值。
2. **original_vtrace 结构隔离（GATE 4/5/6）**：Original-goal V-trace 路径**结构上**不引用
   Hindsight/AWR/relabel 符号——不是"暂时不使用其输出"，而是函数体里根本没有这些符号。

---

## 1. §二 精确 resolved-step 出处（provenance）

### 问题
`rmt_collect.py` 旧字段
`completion_global_step = update_index*(num_envs*rollout_steps) + rollout_step`
**不是**精确解析 env step：它在 `rollout_step` 项漏乘 `num_envs`、漏掉每 env 的 `env_id` 偏移、
漏掉 `+1`，因此既少计又把同一 rollout_step 的所有 env 折叠到同一整数。

### 修复
新增权威字段（`phase4a_v2_counters.completion_resolved_env_step`）：

```
completion_resolved_env_step =
    outer_update_index * num_envs * rollout_steps
  + rollout_step * num_envs
  + env_id
  + 1
```

语义：每个 outer update 消耗 `num_envs*rollout_steps` 个解析（并行）env step；update 内第 `r`
个 rollout_step 同时推进全部 `num_envs` 个 env，env `e` 是该 update 的第 `(r*num_envs+e)` 个
解析 step；`+1` 使计数 1-索引（整条 run 的第一个解析 step == 1）。

旧字段 `completion_global_step` **保留**（兼容 + 历史重算对照），并新增
`completion_global_step_deprecated: True` 标记，文档明确其**不是**精确 step。

### 离线重算（不重跑 16384 探针）
两臂的 `first_ge512` 精确 step 由**既有** episode 记录（每条含 `update_index, rollout_step,
env_id, length`，常量 `num_envs=16, rollout_steps=128`）离线重算，公式同上。可达性结论不变
（仍 BOTH=PASS）。详见 `rmt16_l512_probe_step_correction.md` 与 `tests/recompute_probe_step.py`。

### 未来 launcher 的真实退出码
新增 `experiment_src/launch_phase4a_v2.sh`：后台启动捕获真实 PID（`$!`），`wait $PID` 取**真实**
返回码 `$?`（被信号杀死时为 `128+signum`，如 137=SIGKILL/OOM），记录 PID / start / end /
elapsed / real_return_code 到 `<out>/launch_status.json`，并以该真实返回码作为脚本自身退出码。
**绝不**从日志内容推断 exit=0。（本轮不执行训练，仅交付。）

---

## 2. §三 计数器拆分

### 问题
单一 `update_count` 同时承载：episode update index、pending policy_version、replay policy-lag
参考、日志 global step。Replay 开启后 PPO 与 Replay 在同一 outer 迭代内都 `+1`，四种语义全部劈裂。

### 修复：`phase4a_v2_counters.Phase4ACounters`（纯 Python，无 JAX）
每个字段单一语义：

| 计数器 | 定义 | 递增时机 |
|---|---|---|
| `outer_update_index` | 完成的 outer rollout+PPO 迭代数（权威 episode update index） | 每 outer 迭代 +1 |
| `global_env_steps` | 精确解析 env step 累计（权威 global step） | 每 outer `+= num_envs*rollout_steps` |
| `online_ppo_update_count` | on-policy PPO 主更新次数（与 Replay 独立） | 每次 PPO +1 |
| `replay_update_count` | Replay 梯度更新**执行**次数（不论 KL 接受与否） | 每次 replay step +1 |
| `accepted_replay_policy_update_count` | Replay 更新中**通过 KL gate 并提交**策略的次数 | 仅 policy_committed 时 +1 |
| `replay_attempt_count` | Replay 序列采样尝试次数（原与 hindsight_attempts 混用） | 每次采样 +n |
| `policy_version` | **仅**在实际改变在线 policy 的已接受更新后递增 | PPO 每次 +1；Replay 仅 policy_committed 时 +1 |
| `hindsight_attempt_count` / `hindsight_eligible_count` / `awr_update_count` / `relabeled_sample_count` | 防火墙计数（§八） | **仅** full_p2_legacy 路径递增 |

**关键不变式**：被 KL rollback 拒绝的 Replay `on_replay_kl_rejected()` **不**推进 `policy_version`、
**不**推进 `accepted_replay_policy_update_count`（策略侧已回滚）。

`policy_version` 用作：pending-episode 的 `policy_version`（`pending.reset_slot`）与 replay
policy-lag 参考。RMTTrajectory 新增 `outer_update_index` 与 `policy_version_at_collection`
（加性默认 0），后者是权威 lag 参考；旧 `collected_update_count` 字段保留为 outer loop index
以维持 legacy schema 兼容。

**off-path 等价性**：off 模式下每次 outer 迭代恰好一次 PPO，故 `policy_version == 旧 update_count`
恒成立 → `reset_slot` 取值不变 → 逐位一致（GATE 13）。

---

## 3. §四 显式 Replay 模式

废弃歧义的 `--replay on/off`，改为**必填**、**无默认**的
`--replay_mode {off, original_vtrace, full_p2_legacy}`（缺失即 argparse 失败退出码 2，
不做任何旧参数自动推断）：

- **off**：在线 PPO，无 replay learner / hindsight / AWR（== 旧 `--replay off`，逐位一致）。
- **original_vtrace**：在线 PPO + Original-goal Replay V-trace。Hindsight 调用**必须**为 0，
  AWR 调用**必须**为 0，无 relabeled sample，无第二次 relabeled RMT scan。
- **full_p2_legacy**：保留 V-trace+AWR 路径，**仅**审计/legacy，正式科学**默认禁止**，
  需显式 `--allow-full-p2-legacy`（GATE 15；缺失则 `ap.error` 退出码 2）。

---

## 4. §五 Original-goal V-trace 专用 Learner

新增**独立函数**（`rmt_replay_learner.py`）：

- `compute_loss_original_vtrace_rmt(params, scan_fn, po, obs_o_ext, don_o_ext, target_vals_o, recon_o, cfg)`
- `original_vtrace_update_rmt(network, params, target_params, opt_state, optimizer, apply_eval_rmt, scan_fn, samples_orig, cfg, rmt_cfg, carry_mode)`

**不是**把 `full_p2_update_rmt` 的 relabeled 分支"关掉"，而是单独函数。其梯度图**结构上只含**：
original obs/goal/rewards/dones/behavior log_probs；RMT/GTrXL anchor 重建（`reconstruct_rmt_batch`）；
**一次** online RMT scan（V-trace log_pi/value）+ **一次**对应 target-network scan；V-trace value loss +
actor loss + entropy；与 `full_p2_update_rmt` 相同的事务 KL gate / actor step-scale 重试 / rollback /
EMA / grad-clip / critic-only 分区。

**结构上不含**（无任何符号引用）：`relabel_sample_rmt` / 任何 relabeled obs/goal/reward；`awr.*` /
`w_awr`；第二次 online 重建（`recon_r`）；第二次 target scan（`target_vals_r`）。
`original_vtrace_update_rmt` 甚至**没有** `samples_rel` 形参——relabeled sample 根本无法传入。
V-trace 不需要 policy-lag（存取的 behavior log_prob 即真实行为策略），故也**无** `update_count/lag` 参数。

**冻结权重**：`W_ORIGINAL_VTRACE = 1.0`（模块常量），`loss = 1.0 * (vt_aloss + vf_coef*vt_vloss)`。
**刻意不取** combined loss 里那个说不清的 0.5——那个 0.5 只为平衡 V-trace 项与 AWR 项而存在；
没有 AWR 项就 nothing to balance。

---

## 5. §六 正式序列长度

新增显式 `--sequence_length`（默认/预注册 **129**）。Phase4A-v2 正式干净 Carry 实验预注册
`sequence_length=129`：恰好跨越**一个** 128-step RMT 段边界——Persistent 在第 129 步读取跨段 token，
Reset128 在边界清零；比 512 更高的 replay 可达性；无需 Hindsight；避免把"512 长 episode 稀缺"与
Carry 效应混为一谈。512 仅作为 `ENGINEERING_LONG_WINDOW_MODE` 常量保留。

launcher 对 `replay_mode=original_vtrace` 强制 `sequence_length > 128`（否则 SystemExit）。
manifest（每个 checkpoint + summary）记录：`sequence_length, segment_len=128,
crosses_boundary=(sequence_length>128), replay_mode, hindsight=false, awr=false,
w_original_vtrace=1.0`。

---

## 6. §七 Eligible-only 确定性采样

新增 `RMTReplayBuffer.sample_eligible(sequence_length, rng, batch_size)` → `EligibleSampleBatch`：

- **预筛**：仅从 `length >= sequence_length` 的轨迹集合抽取；
- **绝不**随机抽短再重试（消除旧 K_BATCH try/except `continue` 造成的两臂 replay 成功数漂移）；
- 空 eligible → 显式 `status="NOT_READY"`，零样本、无异常、无静默重抽、不以短轨迹顶替；
- OK 时**每次固定** `batch_size` 个样本；
- 记录 `sample_ids / start_offsets / sequence_lengths / eligible_count`；
- **位可复现**：给定相同 buffer 状态 + 相同 RNG（专用 `np.random.RandomState(seed+7)`，独立于
  JAX rollout/action RNG 流），产出的 sample_ids/start_offsets/sequence_lengths 逐位相同
  （每次 `self.sample()` 都传入显式 trajectory_id+start_step，故不消耗隐藏的 `self._rng`）。

**MATCHED_REPLAY_EXPOSURE**：正式两臂解释前**必须**满足
`persistent_replay_update_count == reset128_replay_update_count` 且
`persistent_replay_sequences_consumed == reset128_replay_sequences_consumed`；否则
`MATCHED_REPLAY_EXPOSURE=FAIL`，不得做 Carry 因果结论。每臂在 summary 记录自身
`(replay_update_count, replay_sequences_consumed)` 与 `matched_replay_protocol_ready`，
跨臂相等由 host 侧从两份 summary 裁定。

---

## 7. §八 Hindsight 防火墙

`original_vtrace`（与 off）模式下硬断言四个防火墙计数 `== 0`：
`hindsight_attempt_count, hindsight_eligible_count, awr_update_count, relabeled_sample_count`
（`Phase4ACounters.assert_hindsight_awr_disabled()`）。launcher 在每次 original_vtrace replay
更新后**及** run 终局各断言一次。测试中 monkeypatch `RH.relabel_sample_rmt` 使其被调用即 raise，
所有 original_vtrace 测试仍须通过——证明是**结构不进入**而非"暂不使用输出"。

---

## 8. §九 RMT 单变量契约

Persistent vs Reset128 **唯一**允许差异 = `carry_mode`：
- Persistent：128-step 边界 `mem_tokens <- 更新后的 tokens`（跨边界携带，仅 true done 清零）。
- Reset128：128-step 边界 `mem_tokens <- zero`（单窗口读写，边界清零，true done 也清零）。

两臂在以下**完全相同**：step0 参数、optimizer、target 参数、env、task、rollout、
`sequence_length=129`、replay mode、batch size、replay schedule、PPO、V-trace、KL、EMA、seed、
training budget、evaluator。`tests/config_diff_validator.py`（GATE 14）递归 diff 两份 YAML 的
`scientific_config`，任何 `carry_mode` 以外的叶差异即失败。GPU 设备与输出路径属硬件/日志放置
（`runtime_assignment`，明确排除于科学 diff 之外，见 GPU map：GPU2=persistent，GPU3=reset128，
GPU0/GPU1 严禁）。

---

## 9. §十 测试门禁（15 项）概览

GATE1 旧 L512 可达性可由原始 episode 记录重算且结论仍 BOTH；GATE2 resolved-step 公式对各
env_id/rollout_step 正确；GATE3 outer/PPO/Replay/policy_version 计数不混用；GATE4 original_vtrace
不调 Hindsight；GATE5 original_vtrace 不算 AWR；GATE6 original_vtrace loss 仅一次 original RMT scan
+ 对应 target scan；GATE7 sequence_length=129 跨 128 边界；GATE8 Persistent 第 129 步进入 token
非零路径 / Reset128 第 129 步 token 为零；GATE9 eligible-only 采样器绝不抽短轨迹；GATE10 相同
buffer+RNG → 相同 sample IDs & offsets；GATE11 Critic/actor KL rollback → policy_version 语义正确；
GATE12 checkpoint 含 params/PPO opt/Replay opt/EMA/RNG/action RNG/buffer/pending episodes/GTrXL state/
RMT state/全部计数器；GATE13 旧探针 off-path 逐位不变；GATE14 Persistent/Reset config diff 仅
carry_mode；GATE15 full_p2_legacy 需显式授权。**任一门禁失败立即停止汇报，不自动修码重跑。**
实现与结果见 `tests/test_phase4a_v2_gates.py` 与 `rmt16_phase4a_v2_test_report.md`。

---

## Phase4A-v2.1 加固增补（PROVENANCE_AND_EXPOSURE_HARDENING）

1. **Episode policy-version 区间出处（§二）**：`RMTTrajectory` / `RMTReplaySample` 新增
   `policy_version_start/end/span`。完成 episode 时，start 取自 `pending.policy_version[e]`
   （**reset_slot 覆写之前**读取），end = 当前已接受 `policy_version`，span = end−start，
   断言 `end>=start`、`span>=0`。`policy_version_at_collection` 降级为
   **DEPRECATED_ALIAS_OF_POLICY_VERSION_START**（不再是 end/当前版本——那正是被修的 bug）。
   科学边界：`PER_TRANSITION_POLICY_VERSION=NOT_RECORDED`，
   `EPISODE_POLICY_VERSION_RANGE=RECORDED`；V-trace 用每 transition 存储的 behavior log_probs，
   不需要逐 transition 策略版本。`reset_slot(e, policy_version=current)` 语义保留（新 episode
   余下步仍由当前 rollout 策略生成，PPO 更新在 rollout 之后）。
2. **policy-lag 门禁身份（§三）**：original_vtrace 的 policy-lag gate = **NOT_APPLICABLE**。
   config 的活动 `max_policy_lag:16` 替换为
   `policy_lag:{active:false, mode:not_applicable_original_vtrace, max_policy_lag:null,
   correction:{method:vtrace_importance_sampling, rho_bar:1.0, c_bar:1.0}}`；
   `original_vtrace + active=true`（或顶层残留 max_policy_lag）fail-closed
   `ORIGINAL_VTRACE_POLICY_LAG_CONFIG_CONFLICT`（config 校验器 + launcher 运行时守卫）。
   遗留 lag 仅存于 `legacy_full_p2_only.max_policy_lag:16`。manifest 记录
   `policy_lag_gate_active=false / max_policy_lag=null / off_policy_correction=vtrace_importance_sampling`。
3. **四标签拆分（§四/§五）**：删除单一 `matched_replay_protocol_ready`。四标签：
   `SAME_REPLAY_PROTOCOL=READY`、`MATCHED_REPLAY_EXPOSURE=NOT_RUN`、
   `MATCHED_REPLAY_CONTENT=NOT_CLAIMED`、`ENDOGENOUS_REPLAY_SCREENING=READY_AFTER_SMOKE`。
   每臂 summary 输出 14 字段 `exposure_certificate`；两级跨臂比较 + fail-closed 门禁见
   `rmt16_phase4a_v2_exposure_contract.md` 与 `tests/phase4a_v2_exposure_validator.py`。
4. **原始 probe JSONL 入 Git（§六）**：`evidence/raw_probe/`（6 证据文件 + SHA256SUMS + README），
   服务器源 SHA 先对清单校验再拷贝；`tests/recompute_probe_step.py --persistent/--reset128/--out`
   全量重算（无硬编码）→ `rmt16_l512_probe_recomputed.json`（8979/BOTH）。
5. **GATE13 统一（§七）**：两个标签——`GATE13_STRUCTURAL_OFF_PATH_EQUIVALENCE=PASS`
   （加性 + 计数等价 + 静态守卫）与 `GATE13_NUMERIC_PARAMETER_UPDATE_HASH_RERUN=NOT_RUN`
   （本轮无参数更新训练；合成 CPU 单测不作为真实 rollout 复跑声明）。
6. **新增门禁 16–26（§十）**：provenance（16–18）、policy-lag（19–21）、标签拆分（22）、
   exposure fail-closed（23–24）、冻结证据（25–26）。本地 25 PASS/1 SKIP(JAX)；服务器 CPU
   26 PASS/0 SKIP。
