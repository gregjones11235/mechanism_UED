# Tier3 真实 98,304-step CC2 checkpoint 绑定与接口闭环最终报告

- 任务: **CC4_REAL_CC2_98304_CHECKPOINT_BINDING_AND_INTERFACE_SMOKE**
- 分支: `henry/tier3-scaffolded-evaluation`(worktree `D:/cc4tmp`)
- 基线提交: `d1cdec4cedda5b99395ba935430b44cb59c90ca2`(上一轮真实 CC2 格式/policy adapter 绑定)
- 本轮提交: 单个(见 §10)
- 唯一规范本地目录: `D:\Projects\dicode-codex-director\`(交接材料只从这里读取)
- 日期: 2026-07-30(UTC+8)

## 0. 结论

**verdict = PASS**。服务器真实 98,304-step Student 双臂最终 checkpoint 已完成:只读定位 → 逐位传回本地 → SHA 与 manifest 全门核验 → SHA 绑定 driver 源非猜测重建 CC2 policy → 真实前向 → 冻结 bank 双核验 → 两份独立证书 → 输出 SHA256SUMS 校验。全程零训练、零 LLM 调用、零冻结物改动、零续跑。smoke 的 SR/reward/achievement **不作为性能**(run_class=INTERFACE_SMOKE,performance_claim_authorized=false)。

## 1. 与合成轮证据的明确区分(任务书第 7 条)

| | 合成接口测试(历史,Commit d1cdec4) | 真实 98,304-step 测试(本轮) |
|---|---|---|
| checkpoint | CC4 本地 `network.init` 随机初始化,`arm=RMT16-SYNTHETIC-NOT-A-STUDENT` | 服务器真实双臂长跑产物,`arm=RMT16-Persistent/Reset128-OrigVtrace`,step=98304 |
| 证明内容 | 格式理论兼容、链路可跑 | 真实训练参数加载/重算/前向/证书闭环 |
| 状态 | 保留为历史证据(`tier3_cc2_binding_final.md`) | 本报告为最终真实闭环证据,取代其"最终"地位 |

旧报告中的 `SYNTHETIC_CC2_FORMAT_E2E_PASS` / `WAITING_CC2_CHECKPOINT` / `__NEW_COMMIT__` / `__REMOTE_HEAD__` 占位与合成 smoke 结果**不得**作为本轮最终证据。

## 2. 服务器端只读核验(SSH 项目专用密钥,IdentitiesOnly+BatchMode)

- RUN_ROOT `/home/oseasy/cc2_data/cc2_runs_76b294b/runs/` 下 `RMT16-LONG98304-PERSISTENT` / `RMT16-LONG98304-RESET128`,每臂 `ckpt/<step>/full_state.pkl` **13 个**,step ∈ {0, 8192, …, 98304}(与交接文档 §2 一致);普通文件、只读 stat/SHA,**不从中间 checkpoint 恢复、不重训、不写服务器**。
- 服务器端 26 条 checkpoint 只读审计(`.tier3_ft_scratch/real_98304/server_audit.jsonl`,26/26 PASS):每条 `params` 用 CC2 原算法重算 == `manifest["params_sha256"]`;manifest step/arm/carry_mode 与目录一致;交接文档 §3 前缀全部命中(step0 `2f8cd875…` 两臂相同;final persistent `aa6ba440…` / reset128 `78a14cc6…`)。
- 交接文档与三份 CC2 JSON(handover SHA `0d90b232…`/9289B = 总控给定值)已全部读取,证据链交叉一致。
- 陈旧归档 `cc2_direct98304_stale_20260729T205051`(ENOSPC 失败轮)按交接 §8 **未读、未用**。

## 3. 最终 checkpoint 逐位传回与本地核验

`ssh cat` 纯读传输(服务器端零写入),本地落 `.tier3_ft_scratch/real_98304/`:

| 臂 | 文件 | 字节数 | 文件 SHA256(= 服务器审计值) |
|---|---|---|---|
| persistent | `RMT16-LONG98304-PERSISTENT_step98304_full_state.pkl` | 21,741,728 | `2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723` |
| reset128 | `RMT16-LONG98304-RESET128_step98304_full_state.pkl` | 21,741,720 | `de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638` |

## 4. 真实 manifest 契约与非猜测重建(本轮代码核心)

- **实证发现**(26 个真实 checkpoint):`manifest["config"] == {}` —— CC2 `Cfg`(driver 303-309 行)是类属性配置类,`vars(Cfg())` 按设计为空。网络超参**冻结在 driver 源码**,不在 pickle。真实 manifest 键为 `replay_mode`(非 `replay`)并携带 `phase4a_v2` provenance(run_class=`long_run_98304`、sequence_length=129、segment_len=128、crosses_boundary=true、base_checkpoint_params_sha256=`d4e85af5…`)。
- **非猜测重建**:`load_cfg_from_driver_source()` 对 SHA 绑定 driver 源(`453bd1ecc8d9671c741c4462214bd7699c74611a52ec157ff30cd68653b4bafc`,五方一致:交接 §3 / addendum / launch report / 本地 `_cc2_stage` / 服务器部署)做 **AST 字面解析**(ast.parse + ast.literal_eval;**不执行、不猜测、不默认值**)恢复 `class Cfg` 26 个属性,11 个必需超参齐全(embed_size=256 / num_heads=8 / num_layers=2 / rmt_num_tokens=16 / window_mem=128 / num_steps=128 / gating=True / gating_bias=2.0 / activation=relu / hidden_layers=256 / qkv_features=256)。
- **一致性门**(fail closed):非空 `manifest["config"]` 必须与 driver Cfg 逐键一致;`phase4a_v2.segment_len` 必须 == cfg num_steps;carry_mode 只从 manifest 读取。`driver_source_sha256` 作为新独立绑定进入 checkpoint record 与证书;`cc2_policy_source_sha256=31c1092c…` 保持不变。
- **证书进程 provenance**(NEG29,任务 §5):`eval_binding` 新增必填 `driver_source_sha256` / `process_pid`(正整数)/ `process_argv`(非空列表)/ `run_start_utc` / `run_end_utc`(可解析 ISO-8601)/ `run_exit_code`(必须 0)。
- **反污染门**(交接 §7):`run_interface_smoke` 在任何绑定前检查 `RMT16_POSTJAX_BINDING_SELFTEST`,非空且非 "0" → fail closed。

## 5. 双臂逐臂绑定核验(verify_real_98304,单 JAX 进程顺序执行)

证据: `.tier3_ft_scratch/real_98304/verify_real_98304.log`(VERIFY_REAL_98304_PASS,exit 0)。

| 门 | persistent | reset128 |
|---|---|---|
| 文件 SHA == 服务器证据 | ✅ `2866b5de…` | ✅ `de3a159f…` |
| params 重算 == manifest == 服务器证据 | ✅ `aa6ba440…` | ✅ `78a14cc6…` |
| step == 98304 | ✅ | ✅ |
| arm | ✅ `RMT16-Persistent-OrigVtrace` | ✅ `RMT16-Reset128-OrigVtrace` |
| carry_mode | ✅ persistent | ✅ reset128 |
| replay_mode / seed / run_class | ✅ original_vtrace / 42 / long_run_98304 | ✅ 同左 |
| phase4a_v2 segment_len / base SHA | ✅ 128 / `d4e85af5…` | ✅ 同左 |
| config == {} (实测) | ✅ | ✅ |
| driver Cfg 重建 + 一致性门 | ✅ | ✅ |
| 真实前向 130 步(跨 128 段边界,greedy_argmax) | ✅ 动作 ∈ [0,43) | ✅ 同左 |
| carry 分叉实证(训练后真实参数) | mem_tokens max\|·\| ≈ **2.70**(携带 cross-attn 更新) | mem_tokens max\|·\| = **0.0**(边界清零) |
| NEG23 params 前向后不变 | ✅ | ✅ |
| GPU uuid(来自 manifest,仅证据) | GPU-8df11537…(CC2 GPU2) | GPU-f56a59b4…(CC2 GPU3) |

## 6. 双臂真实 INTERFACE_SMOKE(严格顺序,绝不并行 JAX)

每臂一条 CLI:`tier3_evaluator.py --checkpoint <real pkl> --scenario all --episodes 2 --max-steps 32 --out <dir> --interface-smoke`(driver/snapshot 根取规范默认值)。

| | persistent | reset128 |
|---|---|---|
| 输出目录 | `.tier3_ft_scratch/real_smoke_persistent` | `.tier3_ft_scratch/real_smoke_reset128` |
| exit / run_class / perf_auth | 0 / INTERFACE_SMOKE / false | 0 / INTERFACE_SMOKE / false |
| 实际 child PID | **125628** | **46440** |
| run_start → end (UTC) | 2026-07-29T17:04:58 → 17:18:56 | 2026-07-29T17:19:19 → 17:33:18(顺序无重叠) |
| argv | 完整绑定(含真实 checkpoint 路径) | 完整绑定 |
| valid_start | 6/6(full/front_l2/back_l2 各 2) | 6/6 |
| params_unchanged (NEG23) | ✅ true | ✅ true |
| 冻结 bank 核验 | ✅ front_l2/back_l2 verified=true | ✅ 同左 |
| SHA256SUMS(`sha256sum -c`) | ✅ 3/3 OK | ✅ 3/3 OK |
| 证书绑定 | checkpoint_file / cc2_params / step=98304 / carry / driver 453bd1ec… / policy_src 31c1092c… / predicate a4fba86b… / obs=[8335] / adim=43 / state_bank FRONT 21aeb7dc… BACK c632e30d… FULL canonical 45fdd17c… / 8 条有序 payload hash / episode_records_sha256 | 同左(params/file 为 reset128 值) |

终端标签分布(**仅链路证据,不是性能**,不做臂间比较,交接 §10):
- persistent: full {TIMEOUT_NO_KOBOLD: 2};front_l2 {FRONT_FLOOR_TRANSITION_REACHED: 1, TIMEOUT_NO_TRANSITION: 1};back_l2 {SUCCESS_DEFEAT_KOBOLD: 1, TIMEOUT_COMBAT_NOT_WON: 1}
- reset128: full {TIMEOUT_NO_KOBOLD: 2};front_l2 {TIMEOUT_NO_TRANSITION: 2};back_l2 {SUCCESS_DEFEAT_KOBOLD: 1, TIMEOUT_COMBAT_NOT_WON: 1}

## 7. 测试矩阵(双解释器,顺序执行)

| 套件 | base(D:/Anaconda,无 JAX) | venv(jax 0.4.30 + craftax 1.4.5) |
|---|---|---|
| tier3_negative_tests.py | PASS **29/29** | PASS **29/29** |
| tier3_evaluation_certificate.py --self-test | PASS(NEG24/25/27/29) | PASS |
| tier3_checkpoint_adapter.py --self-test | PASS | PASS(真实格式 roundtrip) |
| tier3_cc2_policy_adapter.py --self-test | PASS(driver Cfg 纯门禁 + 错 SHA/缺源拒绝) | PASS(carry 分叉 + 一致性门拒绝用例) |
| tier3_evaluator.py --self-test | PASS(反污染门 fail-closed) | PASS(REAL_ENV_INTERFACE_READY) |
| 双臂真实绑定核验(§5) | — | PASS |
| 双臂真实 CLI smoke(§6) | — | PASS × 2 |

NEG 电池 28 → 29(仅负向测试扩张;事件词汇表仍 10,schema 文件数不变)。

## 8. 禁止项遵守情况

未训练、未 resume、未调 LLM/API;未碰 GPU0/GPU1(本地 CPU 运行,本机无 GPU 干扰问题;服务器端仅只读审计);未改冻结 bank/checkpoint/原始 SHA256SUMS;未用合成 checkpoint 替代、未降级"预期失败即 PASS";scaffold hash 未冒充 GLOBAL_WORLD_SET_HASH;无 arm-specific scaffold;未按结果筛选 scaffold;SSH 仅项目专用密钥 + IdentitiesOnly + BatchMode,未输出任何凭据;路径限定 git add;单提交。

## 9. 最终输出字段(任务书 §8,共 17 项)

- CC4_REAL_PERSISTENT_98304_BINDING=REAL_LOADED_AND_VERIFIED(只读 load + params 重算==manifest `aa6ba440…` + 文件==`2866b5de…` + step 98304 + arm/carry/run_class 一致 + driver Cfg 非猜测重建 + 130 步真实前向 + NEG23 不变)
- CC4_REAL_RESET128_98304_BINDING=REAL_LOADED_AND_VERIFIED(同上,params `78a14cc6…` / 文件 `de3a159f…` / arm `RMT16-Reset128-OrigVtrace` / carry reset128)
- PERSISTENT_CHECKPOINT_COUNT=13(服务器只读核验,steps 0,8192,…,98304)
- RESET128_CHECKPOINT_COUNT=13(同上)
- PERSISTENT_FINAL_CHECKPOINT_SHA256=2866b5defc356b57345ca47b2f4f44f19f63e618aa62bc8e03c8f751f005c723
- RESET128_FINAL_CHECKPOINT_SHA256=de3a159f58f904c4ed0bce17bcb87e4b39b21b4ffd0cea557ce61b860727b638
- PERSISTENT_PARAMS_SHA256=aa6ba44040a0742bd709ebe6299acde6242e4faf748159dd0765ee2428addd0d
- RESET128_PARAMS_SHA256=78a14cc6e9ccdeb2c9c3d827ff9e21366d7ceb73ca6c9e557fb1a3733fe6b3f2
- PARAMS_UNCHANGED_BOTH=true(绑定核验 130 步前向 + 各 6 rollout smoke,前后 SHA 全不变,NEG23)
- REAL_INTERFACE_SMOKE_PERSISTENT=PASS_INTERFACE_ONLY(pid 125628,exit 0,valid_start 6/6,冻结 bank verified,证书+SHA256SUMS OK;NOT_A_PERFORMANCE_CLAIM)
- REAL_INTERFACE_SMOKE_RESET128=PASS_INTERFACE_ONLY(pid 46440,exit 0,同上;NOT_A_PERFORMANCE_CLAIM)
- FROZEN_BANK_IDENTITY_VERIFIED=true(两臂 run 均内存重铸+PROCESS_B 复验:FRONT `21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687` / BACK `c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566` / predicate `a4fba86b…5b4c` / canonical task `45fdd17c…824d`)
- CERTIFICATE_SHA256SUMS_VERIFIED=true(两输出目录 `sha256sum -c` 各 3/3 OK)
- PERFORMANCE_CLAIM_AUTHORIZED=false
- NEW_TRAINING_RUNS=0
- NEW_LLM_CALLS=0
- verdict=**PASS**

### 附加(提交/推送)

- CC4_NEW_COMMIT: 本轮提交为 `henry/tier3-scaffolded-evaluation` 上紧接基线 `d1cdec4cedda5b99395ba935430b44cb59c90ca2` 的唯一新提交(提交标题 `feat(eval): real 98304-step CC2 checkpoint binding + interface smoke closure`;`git log d1cdec4..HEAD` 恰好 1 条即为本报告所属提交;单提交纪律下报告不自引 SHA)。
- CC4_PUSH_STATUS: 预期 BLOCKED_NETWORK —— github.com:443 经本地代理(127.0.0.1)此前两轮共三次探测均连接拒绝;本轮再做一次非强制 ff 推送尝试,结果如实记入 CC4 汇报消息。本地提交即本轮闭环证据,待联网补推。

## 10. 提交与推送

- 路径限定 `git add`(仅本轮 8 个文件:5 个 tier3 模块 + schema + 设计文档 §7.4 + 本报告),单提交,附 `Co-Authored-By: Claude <noreply@anthropic.com>`。
- 推送为非强制 ff;若网络仍封锁则如实报 BLOCKED_NETWORK,不降级、不伪报 PASS。
