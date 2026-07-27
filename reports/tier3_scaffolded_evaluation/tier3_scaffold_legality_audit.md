# Tier3 Scaffold 合法性审计 (scaffold legality audit)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1
- 分支: `henry/tier3-scaffolded-evaluation`(基线 `7443aec`)
- 机读文件: `schemas/tier3_scaffold_spec_v1.json`(由 `tier3_scaffold_builder.py --json` 生成)
- 模块: `tier3_scaffold_builder.py`(spec/合法性/源身份)、`tier3_negative_tests.py`(主动攻击每条合法性)
- 状态: **PASS**(`tier3_scaffold_builder.py --self-test` exit 0;负向测试 FAIL=0)

## 1. 唯一合法支架机制 = WorldBuilder

支架不新造任何机制,只复用仓库自身的 `minicraftax.world_builder.WorldBuilder`(SHA `96536bbf...`)。仓库内 `tasks/seed_tasks/combat.py`(SHA `d9ede709...`)源码注释即 `ADDED SCAFFOLDING`,证明 WorldBuilder 支架是**仓库原生合法手段**。合法 API 集冻结于 `tier3_source_audit.LEGAL_BUILDER_API`。

## 2. 两条诊断支架(冻结 V1)

| 支架 | 身份类 | 起点 | 移除 | 保留 | 主指标 |
|---|---|---|---|---|---|
| FRONT_L2 | TIER3_FRONT_DIAGNOSTIC_SCAFFOLD | floor2(黑暗走廊,canonical 入场) | 上游资源准备 + 地牢入口(floor0-1) | 走廊导航、多怪生存、出口搜索(下行至 floor3) | P_CORRIDOR_EXIT_REACHED_GIVEN_VALID_START |
| BACK_L2 | TIER3_BACK_DIAGNOSTIC_SCAFFOLD | floor3 + **活 Kobold** | floor2 走廊瓶颈 | Kobold 搜索、接触、战斗、生存、DEFEAT_KOBOLD | P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START |

两条支架的初始装备/击杀计数均来自 canonical 任务事实(`set_player_inventory({wood7,stone27,coal3,iron3,sapling1,pickaxe3,sword3,bow1,arrows7,torches10})`、`set_monsters_killed(2,8)`、floor2 up-ladder 移除),**未发明字段**。

## 3. 合法性不变式(9 条,全部强制为 True)

`no_privileged_information` / `no_extra_observation_channel` / `no_hidden_boss_direction` / `no_shortest_path_hint` / `no_future_monster_action` / `no_arm_specific_state` / `same_action_space` / `same_observation_schema` / `common_state_bank_for_all_arms`。

另有两条结构性约束:
- observation_schema 与 action_space 字符串必须含 `UNCHANGED`(否则 NEG09/NEG10 fail-closed)。
- `scaffolded_results_can_replace_full_task` 恒为 `false`。

## 4. 负向攻击覆盖(本节每条都有对应 NEG,FAIL=0)

| 不变式 | 负向测试 | 结果 |
|---|---|---|
| 源 SHA 绑定(builder realpath/SHA) | NEG02 | 注入异文件 → FailClosed |
| canonical 任务源 | NEG03 | 错 SHA → FailClosed |
| 必需 EnvState 字段 | NEG04 | 删字段 → FailClosed |
| 状态哈希完整性 | NEG05 | 改哈希 → FailClosed |
| 库存合法(NEG12) | NEG12 | 负值/浮点/空键 → FailClosed |
| 玩家位置合法(NEG13) | NEG13 | 负值/越界 → FailClosed |
| 无臂特异元数据(NEG08) | NEG08 | 注入 arm_id → FailClosed |
| 无额外观测通道(NEG09) | NEG09 | 改 observation_schema → FailClosed |
| 动作空间不变(NEG10) | NEG10 | 改 action_space → FailClosed |
| 无隐藏 Boss 方向(NEG11) | NEG11 | no_hidden_boss_direction=False → FailClosed |
| 前段起点不过出口(NEG14) | NEG14 | player_level=3 → invalid |
| 后段起点未 DEFEAT(NEG15) | NEG15 | achieved 含 DEFEAT → invalid |
| 后段有活 Kobold(NEG16) | NEG16 | mobs=[] → invalid |

## 5. 禁止发明的字段(再次声明)

源码无 `darkness_level / corridor_length / monster_density_mode / boss_distance / boss_area_id`。支架不引入任何此类字段;“走廊出口/Boss 区”仅由 `player_level` 转移与 `down_ladders` 位置表达(见边界设计报告)。

## 6. 物化状态

`materialize_start()` 在无 JAX/craftax 的本机 **fail-closed(BLOCKED_ENVIRONMENT)**,不产生任何状态;JAX 主机路径已按 spec 写好 WorldBuilder 调用序列(front: canonical floor2 入场;back: floor3 入场 + `add_mob(3,'melee',KOBOLD_TYPE_ID,...)`)。真实 Kobold type_id 为 BLOCKED_SOURCE_SEMANTICS(主机从 craftax 常量绑定),自检用 `SYNTHETIC_KOBOLD_TYPE_ID=7` 仅作协议演练,**不声称**是 craftax 值。
