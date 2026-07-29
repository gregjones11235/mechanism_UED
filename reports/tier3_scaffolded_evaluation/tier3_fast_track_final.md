# Tier3 真实评测环境 fast-track 最终报告

- 任务: CC4_TIER3_REAL_STATE_BANK_AND_EVALUATOR_FAST_TRACK
- 分支: `henry/tier3-scaffolded-evaluation`
- 提交: Commit 1 `e6362b5`(语义收口 + 真实物化接口 + evaluator 接口修正);Commit 2 = 本文件 + 两份证据 JSON
- 主机: JAX + craftax==1.4.5(venv `.tier3_ft_venv`,设备 TFRT_CPU_0);基础解释器无 JAX(纯逻辑门禁全部通过)

## 最终输出字段

| 字段 | 值 |
|---|---|
| FRONT_EVAL_FORM | PASS(FRONT 评测**形式/链路**完备且验证通过;**不是** Student 性能 PASS —— 本轮无任何性能结论) |
| BACK_EVAL_FORM | BOSS_COMBAT_SCAFFOLDED(只评 Kobold combat;boss 区域搜索 N/A) |
| REAL_FRONT_STATE_BANK | `21aeb7dcdcb4ffccfb1eedc80f2c6daec1995c242822a0efbec9b947e275d687`(label=FRONT_SCAFFOLD_STATE_BANK_HASH,n=8,REAL_ENVSTATE;绝非 GLOBAL_WORLD_SET_HASH) |
| REAL_BACK_STATE_BANK | `c632e30dcabea7ff812fa07ce855809ec54e7cbca87747fa2d5ab775431f2566`(label=BACK_SCAFFOLD_STATE_BANK_HASH,n=8,REAL_ENVSTATE,resolved_kobold_type_id=3) |
| TWO_PROCESS_FRONT_AGREEMENT | true(两个独立 OS 进程:ordered IDs、per-state payload hash、field manifest、state-bank hash 全一致;双方均通过 PROCESS_B 重铸核验) |
| TWO_PROCESS_BACK_AGREEMENT | true(同上) |
| OBSERVATION_SCHEMA_UNCHANGED | true(observation shape (8335,) == canonical S4;FRONT bank-state reset 与 canonical reset 在同 rng 下叶级 + obs 完全一致) |
| ACTION_SPACE_UNCHANGED | true(action_count=43 == canonical 绑定;Achievement=67,DEFEAT_KOBOLD index=41) |
| NO_PRIVILEGED_INFORMATION | true(obs 由**未改动的 canonical `envns.get_obs`** 生成;bank state 不含 boss 方向/最短路径/隐藏字段;无额外观测通道) |
| CC2_CC4_INTERFACE_SMOKE | WAITING_CC2_CHECKPOINT(本机所有 worktree 均无 `full_state.pkl`;CC2 训练在远端服务器)。链路就绪已由**合成链路 smoke** 证明(见下),待真实 checkpoint 到位后按同一链路运行 |
| GIT_COMMITS_CREATED | `e6362b5`(Commit 1);Commit 2(本提交) |

## 证据文件

- `reports/tier3_scaffolded_evaluation/tier3_real_materialization_evidence_v1.json`
  —— 双进程真实物化、canonical 环境合同、FRONT reset 等价、无特权信息论证、arm 独立性(NEG07/08/26)、NEG 全套 FAIL=0、predicate_code_sha256=`a4fba86b054d...`。
- `reports/tier3_scaffolded_evaluation/tier3_synthetic_chain_smoke_v1.json`
  —— 合成(全零参数,显式标注 NOT_A_STUDENT)链路 smoke:full_state.pkl → `load_full_params_readonly` → NEG21 身份绑定 → NEG22 obs 形状绑定 → greedy_argmax → `rollout_episode`(FULL/FRONT_L2/BACK_L2 各 2 episode,max_steps=32 仅链路)→ NEG19/20 分类 → metrics → certificate(NEG24/25)→ NEG23 参数未变。结果:三场景 valid=2/2,rollout_status=REAL_ENV_INTERFACE_READY;FRONT=TIMEOUT_NO_TRANSITION×2,BACK=DIED_AFTER_ENGAGEMENT×1 + TIMEOUT_COMBAT_NOT_WON×1(kobold_engaged 检测真实触发),FULL=TIMEOUT_NO_KOBOLD×2。零参数 NOOP 的结果即预期,**不构成任何性能数字**。

## 语义收口摘要(Commit 1)

- FRONT_L2: 主事件 `FRONT_FLOOR_TRANSITION_REACHED`(player level 2→3);primary=`P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START`;dense=`GRAPH_DISTANCE_PROGRESS`;`CORRIDOR_EXIT_REACHED` 降为 `PENDING_EQUIVALENCE_ALIAS`(真实地图证明楼层转移必经目标走廊前不定义成功;transition=True 且 alias 显式 False → FailClosed)。
- BACK_L2: identity=`BOSS_COMBAT_SCAFFOLDED`;Kobold 绑定 RANGED type_id 3 / HP 8.0(craftax==1.4.5 解析);`boss_area_reached`/`time_to_boss_area`/`BACK_BOSS_NOT_FOUND` 全部 N/A;不再声称评估 Boss 区域搜索。
- 事件词汇表保持 10、NEG 保持 26、schema 文件数不变:只原地收口,无 schema 扩张。

## 测试门禁(双解释器)

- 基础 python(无 JAX):11 模块 self-test 全 PASS;NEG FAIL=0(26/26);聚合 `TIER3_AGGREGATE_SELF_TEST_PASS`。
- venv(JAX+craftax):builder/materializer/evaluator self-test 全 PASS(env=JAX_CRAFTAX_AVAILABLE;rollout=REAL_ENV_INTERFACE_READY;materializer hash_status=MATERIALIZED);NEG FAIL=0(26/26,其中 NEG06/07/24/26 走真实 mint 路径)。

## 诚实边界

- 本轮**无 Student 性能数据、无正式 evaluation run、无性能比较**;`scaffolded_results_can_replace_full_task=false`;scaffold bank hash 永不冒充 GLOBAL_WORLD_SET_HASH(NEG24)。
- CC2 真实 checkpoint 到位后,CC4 将以同一冻结合同(greedy_argmax、obs (8335,)、43 actions、max_timesteps 4096)运行 Persistent/Reset128 × FULL/FRONT_L2/BACK_L2 接口 episode(先链路、后由授权任务决定正式评测)。
