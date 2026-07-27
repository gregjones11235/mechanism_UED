# Tier3 已知限制 (known limitations)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1
- 分支: `henry/tier3-scaffolded-evaluation`(基线 `7443aec`)

## 1. 环境能力缺失(诚实 BLOCKED,不伪装)

- 本机**无 JAX、无 craftax**(`importlib.find_spec` 均 False)。因此:
  - 真实 EnvState 物化 / 真实 scaffold 铸造 = **BLOCKED_ENVIRONMENT**;
  - 真实 Student 评测(rollout)= **NOT_RUN**;
  - `Achievement.DEFEAT_KOBOLD` 整数索引、Kobold `type_id`、`ItemType.NONE`、`BlockType` 可行走集、`get_distance_map`、`Inventory` 字段表 = **BLOCKED_SOURCE_SEMANTICS**(符号化引用,craftax==1.4.5 主机运行时绑定)。
- 自检用 `SYNTHETIC_KOBOLD_TYPE_ID=7` 仅作协议演练,**不声称**是 craftax 值。

## 2. 证据层级(不可越级)

- 本轮至多可达:`IMPLEMENTED_STATIC` / `TESTED_SYNTHETIC`(若在 JAX 主机 reset 自检可达 `TESTED_REAL_ENV_RESET`)。
- **不可**达:`REAL_CRAFTAX_SCAFFOLD_TEST`(BLOCKED_ENVIRONMENT)、`REAL_STUDENT_EVALUATION`(NOT_RUN)。
- 因此**禁止**写出 `FRONT_SCAFFOLD_EVALUATION=PASS` / `TIER3_FRONT_HALF_BREAKTHROUGH` / SOTA / Persistent>Reset128 / Replay 科学增益。

## 3. 哈希作用域(不可冒充)

- `FRONT_/BACK_SCAFFOLD_STATE_BANK_HASH` = **NOT_MATERIALIZED**,且**永不**等于 `GLOBAL_WORLD_SET_HASH`(后者属 seed42 canonical world materializer,CC4 V3,当前 `BLOCKED_SOURCE_UNVERIFIED`)。
- 合成自检 bank 哈希仅为协议演示(manifest 标 `SYNTHETIC_TEST_ONLY`)。

## 4. 范围限制

- FRONT_L1 / BACK_L1 接口**预留但本轮不实现**;本轮仅 FULL + FRONT_L2 + BACK_L2。
- scaffold 结果为**机制诊断专用**,`scaffolded_results_can_replace_full_task=false`。
- 未做任何 Student 训练、多 seed 性能实验、正式 evaluation run、性能比较、新 LLM 调用、D052 candidate pool。

## 5. 复用与不破坏

- 加法式复用 V3(runtime source identity / EnvState serializer / field manifest / 双进程 / fail-closed),**未重新实现 V3**,未触碰冻结 54 文件 / 原始 SHA256SUMS / CC2 / CC3 / Henry-branch。
- 未 push、未 merge、未 rebase、未 amend、未 force push、未 reset --hard、未 git clean。
