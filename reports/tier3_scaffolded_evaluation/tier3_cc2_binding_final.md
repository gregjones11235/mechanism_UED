# Tier3 真实 CC2 绑定轮最终报告 (CC4_REAL_CC2_POLICY_ADAPTER_AND_INTERFACE_SMOKE)

- 分支: `henry/tier3-scaffolded-evaluation`
- 上轮远端 HEAD: `4f4db50e0a935441aa02c71d23a8f4ce45052c19`
- 本轮提交: 单个 `fix(eval): bind real CC2 checkpoint and policy adapter`
- 语义状态: **Tier3 state banks 与评测语义已冻结,本轮零重物化、零语义修改**;仅绑定真实 CC2 checkpoint/policy 与值级证据。

## 1. checkpoint 格式绑定(PA-1)

CC2 `train_rmt16_p2replay.py::save_ckpt` 的 `full_state.pkl` 实际格式为
`{"params": <DIRECT pytree(numpy 叶)>, "manifest": {params_sha256, step, arm, carry_mode, replay, gpu_uuid, seed, config, tag}}` —— **不是** `(leaves, treedef)`。

- `load_full_params_readonly(path)` 改为 `params = d["params"]`,要求 dict 同时含 `params` 与 `manifest`,`manifest["params_sha256"]` 必须存在。
- `cc2_params_sha256(params)`:与 CC2 `_params_sha` **逐字节同算法** —— `hashlib.sha256(); for leaf in jax.tree_util.tree_leaves(params): h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())`。
- NEG21:重算 SHA ≠ manifest 声明值 → fail closed;同时记录 checkpoint 文件 SHA(`checkpoint_file_sha256`)。
- 兼容性证据:用 CC2 自有 `network.init(..., method=network.init_all)` 构造**合成 CC2 格式** checkpoint(`arm=RMT16-SYNTHETIC-NOT-A-STUDENT`,随机初始化参数,明确不是 Student),写入真实 `save_ckpt` 格式;`load_full_params_readonly` 读回并通过 SHA 重算核验(双解释器 self-test PASS;NEG21/22/23 guards live)。
- 合成 checkpoint `params_sha256 = 9a9e69f21803fc1c729ca8edc724b61c61d599f943e02bda27172ee873c8fbb7`。

## 2. 真实 CC2 policy adapter(PA-2,新模块 `tier3_cc2_policy_adapter.py`)

- 从 `--cc2_snapshot_root` 导入 CC2 **当前实际模块**:`ActorCriticTransformerRMT16`、`RMT16Config`、`rmt16_init`、`make_apply_eval_rmt`、`make_update_fn`、`rmt_step_forward`。**CC4 零重实现 RMT/GTrXL 状态转移。**
- 源码字节绑定 `cc2_policy_source_sha256`(逐文件 LF 归一化 SHA 的有序聚合)= `31c1092c037577c56ba0eba9d51ea40cc6a97210bbcbc98fe047762daed2f46f`;模块 `__file__` 必须 realpath 落在声明根内(sys.modules 缓存投毒/错根 → fail closed)。
- 网络与 `RMT16Config` 从 `manifest["config"]` 按 CC2 原样重建(encoder_size=embed_size 等 11 个必填配置字段,缺一 fail closed);`carry_mode` 只从 manifest 读取,必须 ∈ {persistent, reset128}。
- 每 episode:以 CC2 driver 约定初始化真实 GTrXL+RMT16 态(memories zeros (1,128,2,256)、mem_mask zeros (1,8,1,129) bool、mem_idx=full((1,),128,int32)、`rmt16_init(1, rmt_cfg)`);params 只读;每步调 CC2 `rmt_step_forward`;greedy_argmax 选动;**不调用 optimizer/replay learner**。
- NEG23:评测前后 `cc2_params_sha256(params)` 必须不变。
- **carry 语义实证**(adapter self-test,非零确定性观测 `np.random.default_rng(20260729)`;关键发现:全零观测使无偏置 GTrXL 输出 h_t≡0,段边界更新不可见):
  - 128 步段边界**之前**(64 步):persistent/reset128 的 mem_tokens 均保持 0(`rmt_advance_tokens` 仅在 seg_count>=segment_len 时触发);
  - 段边界(128 步):persistent 携带**无 gate** 的 cross-attention 更新(`mem_tokens + LN(attn)`,max|tok|≈2.85),reset128 清零为 0;两模式 tokens 严格分叉。
  - 这是 CC2 代码中 Persistent 与 Reset128 的**唯一**差异路径。

## 3. 冻结 bank identity 值绑定(PA-3)

真实评测前 `verify_frozen_bank_identity(scenario)` 内存重铸(PROCESS_A)并逐项核验,任一不符 fail closed;**不写盘、不修改冻结 bank**:

| 绑定项 | 冻结值 |
|---|---|
| FRONT state_bank_hash | `21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687` |
| BACK state_bank_hash | `c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566` |
| field_manifest_sha256 | `615d4be4df22115e4ac520718076860bf9def636a46806f5a2948be21456ee07` |
| predicate_code_sha256 | `a4fba86b054d20412fc1df2c79e7000d66b0525decb1801fa474ee7fb0d25b4c` |
| canonical_task_sha256 | `45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d` |
| seeds | seed_base=10_000, stride=1, n=8(BACK 偏移 +1_000_000) |

核验含:hash_label 与值双绑定、序敏感的 ordered payload hash → bank hash 重算、逐条 REAL-only + field manifest 冻结、source_shas、predicate code SHA、canonical task SHA,再叠加 PROCESS_B 独立复验。纯比较层 `check_frozen_manifest_bindings` 无 JAX/craftax 依赖,NEG28 在任意主机可跑。

## 4. FRONT progress fail-closed(PA-3)

- **删除** `rollout_episode` 中的 `except pred.FailClosed: pass`。NEG18(start→exit 不可达)、off-grid、non-walkable 一律向上传播、中止评测,**不得吞掉**。
- 进度计算仅在 `player_level == FRONT_FLOOR` 且 `max_level < CORRIDOR_EXIT_FLOOR` 时进行:已到 floor3 后 floor2 图距离无定义,dense metric **永久冻结**于转移时刻。
- ladder tile 采用显式、可审计的 **LADDER_TILE_TRANSIT** 规则:floor2 `down_ladders`(走廊出口)与 `up_ladders` 位置从归一化视图取出,OR 入静态可走掩码;越界 → fail closed(broken scaffold start)。

## 5. 证书真实值绑定(PA-4,NEG27)

`evaluation_certificate.json` 的 `eval_binding` 必须实际绑定 15 个字段(全为真实值,不得只记 hash label):
`state_bank_hash` / `state_payload_hashes`(有序) / `checkpoint_file_sha256` / `cc2_params_sha256` / `checkpoint_step` / `carry_mode` / `run_class` / `episode_records_sha256` / `cc2_policy_source_sha256` / `evaluator_source_sha256` / `predicate_code_sha256` / `observation_shape=[8335]` / `action_dim=43` / `params_unchanged=true` / `performance_claim_authorized`。

- SHA 字段必须 64-hex **值**(以标签冒充 → fail closed);`INTERFACE_SMOKE` ⇒ `performance_claim_authorized` 必须为 false;`run_class` ∈ {INTERFACE_SMOKE, FORMAL_EVALUATION}。
- NEG27 覆盖:label 冒充、缺字段、空 payload hashes、错 obs shape、错 action dim、params_changed、坏 run_class、smoke 声称性能、坏 carry_mode,全部拒绝;FORMAL_EVALUATION+authorized 合法路径放行。

## 6. 真实 CLI 与接口 smoke(PA-4)

```
python tier3_evaluator.py --checkpoint <full_state.pkl> --cc2_snapshot_root <PATH> \
    --scenario {front_l2|back_l2|full|all} --out <DIR> --interface-smoke \
    [--episodes 2] [--max-steps 32]
```

- run_class=INTERFACE_SMOKE,`performance_claim_authorized=false`;FULL/FRONT_L2/BACK_L2 各少量 episode、max_steps=32。
- 输出:`episode_records.jsonl`、`evaluation_result.json`、`evaluation_certificate.json`、`SHA256SUMS`(LF、确定性)。
- **合成 CC2 格式 e2e 链路验证**(本地无真实 step4096 checkpoint):以 §1 的合成 CC2 格式 pkl 跑 `--scenario all --interface-smoke --episodes 2 --max-steps 32`,全程通过(冻结 bank 双核验 + 真实 CC2 policy 前向 + 证书值绑定 + NEG23 params 不变)。产物核验(`.tier3_ft_scratch/cli_smoke_out/`,`sha256sum -c SHA256SUMS` 全 **OK**):

```
episode_records.jsonl      6 条(full/front_l2/back_l2 各 2),valid_start 6/6,timesteps=32
evaluation_result.json     front_l2: {TIMEOUT_NO_TRANSITION: 2}
                           back_l2 : {DIED_AFTER_ENGAGEMENT: 1, TIMEOUT_COMBAT_NOT_WON: 1}
                           full    : {TIMEOUT_NO_KOBOLD: 2}   (随机初始化 policy 的合理终端分布)
                           frozen_bank_bindings: front_l2/back_l2 verified=true
                             (冻结哈希命中,各 8 条有序载荷)
                           checkpoint: params_sha256=9a9e69f2… (评测前后不变) step=4096 trainable=false
                           cc2_policy_source_sha256=31c1092c…
evaluation_certificate.json  三场景 run_class=INTERFACE_SMOKE;performance_claim_authorized=false;
                           observation_shape=[8335];action_dim=43;params_unchanged=true;
                           state_bank_hash: front=21aeb7dc…d687 / back=c632e30d…2566
                                            / full=CANONICAL_RESET_SEEDS 45fdd17c…824d
SHA256SUMS                 3 条目全部校验通过(LF,确定性)
```

- 注:后台进程在产物与 SHA256SUMS 写完后被外部停止,stdout 摘要行未落盘(无 traceback);以**产物完整 + SHA256SUMS 自洽**(SHA256SUMS 为 run 最后写入项,三文件校验全 OK)为链路通过证据。

- 真实 CC2 step4096 checkpoint 到位后,以同一 CLI 直接运行接口 smoke(checkpoint 格式已绑定,无需改码)。

## 7. 测试矩阵(双解释器)

| 套件 | base(D:/Anaconda,无 JAX) | venv(jax 0.4.30 + craftax 1.4.5) |
|---|---|---|
| tier3_negative_tests.py | PASS 28/28(NEG28 纯比较层) | PASS 28/28 |
| tier3_evaluation_certificate.py --self-test | PASS(NEG24/25/27 live) | PASS |
| tier3_checkpoint_adapter.py --self-test | PASS(无 JAX 分支) | PASS(CC2 格式绑定 + NEG21/22/23) |
| tier3_cc2_policy_adapter.py --self-test | PASS(纯门禁分支) | PASS(carry 分叉实证) |
| tier3_evaluator.py --self-test | PASS | PASS(REAL_ENV_INTERFACE_READY) |
| CLI interface-smoke e2e(合成 CC2 格式) | — | PASS(见 §6) |

NEG 电池 26 → 28(仅负向测试扩张;事件词汇表仍 10,schema 文件数不变)。

## 8. 禁止项遵守情况

未做 Student 训练;未做正式性能评估;未做 Persistent/Reset128 性能比较(仅 carry 语义机制实证);未重物化/修改冻结 state bank;未新增场景;单提交;scaffold hash 未冒充 GLOBAL_WORLD_SET_HASH;无 arm-specific scaffold;未按 checkpoint/结果筛选 scaffold。

## 9. 最终输出字段

- CC4_NEW_COMMIT=__NEW_COMMIT__
- CC4_REMOTE_HEAD=__REMOTE_HEAD__
- CC2_CHECKPOINT_FORMAT_COMPATIBILITY=REAL_FORMAT_BOUND(load_full_params_readonly 读 `{"params": <direct pytree>, "manifest"}`;CC2 同算法 params SHA 重算 = manifest["params_sha256"],NEG21 守护;合成 CC2 格式 roundtrip + CLI e2e 验证;真实 step4096 格式同源,到位即跑)
- CC2_POLICY_ADAPTER_SOURCE_BOUND=true(cc2_policy_source_sha256=31c1092c037577c56ba0eba9d51ea40cc6a97210bbcbc98fe047762daed2f46f;import ActorCriticTransformerRMT16/make_apply_eval_rmt/make_update_fn/rmt_step_forward/RMT16Config/rmt16_init,CC4 零重实现)
- PERSISTENT_EVAL_CARRY_SEMANTICS=VERIFIED_CARRIES_CROSS_ATTENTION_UPDATE(128 步段边界 mem_tokens ← mem_tokens + LN(cross-attn(seg_buf)),无 gate;边界前不动;self-test 实证 max|tok|≈2.85)
- RESET128_EVAL_CARRY_SEMANTICS=VERIFIED_CLEARS_TO_ZERO_AT_BOUNDARY(同一 rmt_advance_tokens 分叉:边界处 mem_tokens ← 0;self-test 实证 all==0)
- FRONT_PROGRESS_FAIL_CLOSED=true(删 except FailClosed: NEG18/off-grid/non-walkable 传播中止;floor3 后进度永久冻结;LADDER_TILE_TRANSIT 显式规则)
- STATE_BANK_VALUE_BINDING=FROZEN_AND_RUNTIME_VERIFIED(FRONT 21aeb7dc…d687 / BACK c632e30d…2566 / field manifest 615d4be4…ee07 / predicate a4fba86b…5b4c / canonical task 45fdd17c…824d;序敏感重算 + PROCESS_B;NEG28 三路篡改拒绝)
- CERTIFICATE_CHECKPOINT_BINDING=REAL_VALUE_BINDING_15_FIELDS(eval_binding: checkpoint_file_sha256/cc2_params_sha256/checkpoint_step/carry_mode/state_bank_hash/state_payload_hashes/episode_records_sha256/cc2_policy_source_sha256/evaluator_source_sha256/predicate_code_sha256/observation_shape=[8335]/action_dim=43/params_unchanged=true/run_class/performance_claim_authorized;NEG27 拒绝 label 冒充)
- CC2_CC4_REAL_INTERFACE_SMOKE=SYNTHETIC_CC2_FORMAT_E2E_PASS(合成 CC2 格式 CLI `--scenario all --interface-smoke` 全链路通过);真实 step4096 checkpoint=WAITING_CC2_CHECKPOINT(本地未到位;格式/adapter/CLI 已就绪,到位即跑,无需改码)
