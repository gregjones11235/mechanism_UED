# RMT16 Phase4A-v2 — 已知局限（Known Limitations）

本报告诚实记录本轮实现的边界与未覆盖项，避免被误读为"已完成正式科学"。

## 1. 本轮不产生任何训练证据（NEW_TRAINING_RUNS=0）
- 本轮**仅**：代码修改、CPU 测试、静态审计、不更新参数的前向探针、git commit+push。
- **未**启动正式两臂；**未**跑 4096 smoke / 24576 / 98304 / 多 seed；**未**用 GPU 训练。
- 因此本轮**不**给出任何 Carry 性能/有效性结论，也**不**重审 `L512_REACHABILITY_BOTH=PASS`。
- `MATCHED_REPLAY_EXPOSURE` 的跨臂相等**尚未被任何真实 run 满足**（尚无 original_vtrace 正式 run）；
  协议与计数器已就绪，但 `matched_replay_protocol_ready=true` 仅表示"协议可用"，**不**表示"已匹配"。

## 2. 冻结结论的承继边界
- `L512_REACHABILITY_BOTH=PASS` 来自基线 `d3c8c7d6` 已吸收的探针证据（Persistent 6/20，Reset128 5/21；
  Replay=OFF，Hindsight=OFF）。本轮**只**对其 step 出处做离线精化（§二），**不**改变可达性判定。
- 探针**不**做 Carry/performance 断言——这一限制原样继承。

## 3. §二 离线 step 重算的依赖
- `first_ge512` 精确 step 由**服务器**上既有 episode 记录（`*_probe_episodes.jsonl`）离线重算，
  依赖常量 `num_envs=16, rollout_steps=128` 与每条记录的 `(update_index, rollout_step, env_id)`。
  这些记录**不在** Git 仓库内（运行输出），故重算须在服务器执行（#22）。重算前，仓库内报告以
  方法论 + 待填实测数呈现。

## 4. full_p2_legacy 路径的语义保留
- `full_p2_legacy` 保留旧 K_BATCH "随机抽短再重试" 采样，以**逐位保留** legacy 审计语义；§七 的
  eligible-only 修复**只**应用于正式 `original_vtrace` 路径。full_p2_legacy 正式科学**默认禁止**，
  需显式 `--allow-full-p2-legacy`（GATE 15）。
- 一处有意的边缘差异：统一后的 EMA 逻辑 `if not did_replay_update: ema` 在 legacy "can_sample 但
  len(so)<2" 情形会做一次 PPO-only EMA（旧代码此情形既不 replay-update 也不 PPO-EMA）。该路径
  非正式、本轮不运行；off 与 original_vtrace 不受影响（off 每迭代 EMA 与旧 `if not REPLAY_ON` 一致）。

## 5. 测试的 CPU/JAX 分层
- 纯 Python 门禁（GATE 2/3/7/9/10/11 的计数与确定性部分/14/15）可本地 CPU 跑（numpy/yaml）。
- 依赖 JAX 前向的门禁（GATE 8 的 token 非零/清零路径、GATE 4/5/6 的部分、GATE 13 的逐位 hash）
  须在**服务器 CPU**（dicode310 env，JAX 0.6.0）跑。本地无 GPU/可能无 jax，故不在本地强行跑这些。
- 不为追求测试通过而篡改阈值；任一门禁失败 → 立即停止汇报，不自动修码重跑。

## 6. 加性保证的验证范围
- GATE 13（off-path 逐位不变）通过**构造**论证（所有改动加性、off 下计数器与旧值恒等）+ #22
  服务器 CPU 小步前向 hash 对照**实证**。在 #22 实证前，它是"高置信构造论证"而非"已实测"。

## 7. GPU / 分支 / 推送边界（standing 约束，本轮持续生效）
- GPU0/GPU1 严禁；GPU2=Persistent，GPU3=Reset128。本轮不训练，仅交付 config 中的硬件分配。
- 推送**仅**到 `henry/rmt16-phase4a-v2-original-vtrace`；**不**推 main / Henry-branch / reviewed
  分支 / CC3 / CC4；**不**改 CC3/CC4 分支；禁 force push / reset --hard / rebase / merge / 改写历史。
- 服务器**不**直接 push；GitHub 推送统一在本地 Windows worktree 经 GCM 完成。

## 8. 未实现/超出本轮范围
- 正式两臂 run、评估、多 seed、统计检验、Carry 因果裁定：**均不在本轮**，须后续授权。
- Hindsight / AWR 的正式启用：**本轮结构禁用于 original_vtrace**；仅 full_p2_legacy 保留（禁默认）。

## 9. Phase4A-v2.1 加固轮的限制（本轮新增）

- `PER_TRANSITION_POLICY_VERSION=NOT_RECORDED`：长 episode 仅记录**区间** start/end/span；
  episode 内逐 transition 策略版本不可得（V-trace 用存储的 behavior log_probs，不需要它）。
- `policy_version_at_collection` 是弃用别名（= start），仅为 schema 兼容保留；新代码应读
  `policy_version_start/end/span`。
- `MATCHED_REPLAY_EXPOSURE=NOT_RUN`：本轮 `NEW_TRAINING_RUNS=0`，无正式两臂 run，exposure
  证书规格与 validator 已就绪但**未**产生 PASS 裁定；`SAME_REPLAY_PROTOCOL=READY` **不**蕴含
  exposure 匹配（GATE22）。
- `MATCHED_REPLAY_CONTENT=NOT_CLAIMED` 且**不可**升级为 PASS：endogenous 按臂 buffer 无共享
  轨迹身份（fail-closed `ENDOGENOUS_REPLAY_CONTENT_CANNOT_BE_CLAIMED_MATCHED`）。
- `GATE13_NUMERIC_PARAMETER_UPDATE_HASH_RERUN=NOT_RUN`：结构等价 PASS **不**等于数值逐位复跑
  PASS；合成 CPU 单测仅为构造证据。
- 原始 probe 证据冻结于提交时的 SHA256；若后续服务器源文件改变，repo 内副本**不**随之更新
  （它是历史证据快照）。
- 本轮仅本地 commit，`PUSH_PERFORMED=false`，等待总控复审。

---

## Phase4A-v2.2 已知局限补遗

- **绑定是“静态 + pre-JAX”门禁，不是训练验证**：`--formal_config` 绑定证明 YAML 预登记与真实
  runtime scientific config **逐字段一致并产出 certificate**，但它**不**运行 4096 smoke、不启动
  两臂、不更新参数（`SMOKE_4096=NOT_RUN`，`FORMAL_TWO_ARM_LAUNCH=NOT_AUTHORIZED`，
  `NEW_TRAINING_RUNS=0`，`GPU_TRAINING_RUNS=0`）。certificate PASS 是“配置绑定正确”的证据，
  **不是**“训练结果正确”的证据。
- **base checkpoint SHA 比对依赖冻结期望**：`d4e85af5…` 取自两臂冻结 probe summary 的
  `base_sha256`。若未来换用不同 base checkpoint，期望值需显式更新；缺期望时 fail-safe 标
  `NOT_FROZEN`，**从不**伪造 PASS。
- **out_dir 比较允许后缀匹配**：YAML 相对路径（如 `runs/RMT16-PERSISTENT-ORIGVTRACE-129`）与
  可能的绝对 `args.out` 之间用“精确相等 OR realpath 后缀”判定，避免把路径前缀差异误判为失配；
  这放宽了“字符串严格相等”，但 gpu_uuid 仍要求精确相等。
- **certificate 两段式**：结构 + 科学 + assignment 校验在 env build **之前**完成并落盘；base
  params SHA 在 ckpt load **之后**二次校验并**重写** certificate。中间窗口的 certificate 尚未含
  base SHA 结论——属设计内的两阶段，非缺陷。
- **协议身份是“定义级”而非“执行级”**：`PROTOCOL_MATCH` 比较两臂的协议**定义**（learner/rng_rule/
  各字段 + canonical SHA）；它不比较两臂实际 replay 的**内容**（Level 3 恒
  `NOT_APPLICABLE_ENDOGENOUS_BUFFERS`）。`MATCHED_REPLAY_EXPOSURE=NOT_RUN`、
  `MATCHED_REPLAY_CONTENT=NOT_CLAIMED` 仍成立。
- **本轮仅本地 commit**：`IMPLEMENTATION_ROUND_PUSH_PERFORMED=false`、
  `V2_2_REMOTE_PUBLICATION_STATUS=NOT_PUSHED`、`V2_2_PUSH_PERFORMED=false`。无时间范围的
  `PUSH_PERFORMED` 键已禁用；等待总控复审。
