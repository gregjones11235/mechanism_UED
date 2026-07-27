# Tier3 真实源码审计报告 (tier3_source_audit)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1
- 分支: `henry/tier3-scaffolded-evaluation`(基线 `7443aec`)
- 模块: `tools/tier3_scaffolded_evaluation/tier3_source_audit.py`(单一 SHA/事实真理源)
- 状态: **PASS**(在库 12 个被审计文件 SHA 全部 MATCH;`tier3_source_audit.py --self-test` exit 0)
- 原则: **先审计,后定义**。本报告所有字段/语义均来自真实源码逐行阅读;**未发明任何字段**。无法在本机严格定义者一律标 `BLOCKED_SOURCE_SEMANTICS`(因外部 `craftax` 包未安装)。

## 1. 被审计真实源文件(realpath + 完整 SHA256 + 导入路径)

权威在库树: `dicode_src/src/`(CC4 V3 materializer 即锚定于此)。

| 角色 | 仓库相对路径 | SHA256 | 导入路径 |
|---|---|---|---|
| EnvState 定义 | dicode_src/src/minicraftax/craftax_state.py | `7ed6eed02495fa6f0992ebe3e7a2c89b56d2c8d0798915fed76c60e3a5be770b` | minicraftax.craftax_state |
| multitask 环境 | dicode_src/src/minicraftax/envs/multitask.py | `c8f2d5c3c23476c92ab3897f47bef4df7f202a3bd57360fc1bd4cb92b9498bae` | minicraftax.envs.multitask |
| craftax 环境 | dicode_src/src/minicraftax/envs/craftax.py | `be90ee9c9cb4977f07ba52b58676166dea3446f9f40e87a26ef552fbec54104a` | minicraftax.envs.craftax |
| env base | dicode_src/src/minicraftax/envs/base.py | `34e9e3392e8fe73069389387f022e4adf32b1ece1ce0d55730560434920db572` | minicraftax.envs.base |
| WorldBuilder | dicode_src/src/minicraftax/world_builder.py | `96536bbf955376b75c44208d80f452e1907d976cd49685a0e97e3a752679b50d` | minicraftax.world_builder |
| game_mechanics | dicode_src/src/minicraftax/game_mechanics.py | `1bb9a4a64fde852c970b32dc3e049d472856490eaf68d67f64dd241319d3a65a` | minicraftax.game_mechanics |
| base_task | dicode_src/src/minicraftax/tasks/base_task.py | `9b2cb995a807c625fde933a5edf8266dfbf32af3aac9c767f40e41e50586b1fa` | minicraftax.tasks.base_task |
| combat seed task | dicode_src/src/minicraftax/tasks/seed_tasks/combat.py | `d9ede70921dc96e14a974efb20481b5eb225df4828793604b172e6a964e3fae5` | minicraftax.tasks.seed_tasks.combat |
| optimistic wrapper | dicode_src/src/dicode/wrappers_cl.py | `2ded41d81a98c712620dc1633262f2d185ce7dd22e7cc447db22a6ad04b0ddd8` | dicode.wrappers_cl |

外部(冻结 raw-data extract 内,记录 SHA;`--self-test` 确认本机存在且 MATCH):

| 角色 | SHA256 | 说明 |
|---|---|---|
| canonical S4 任务 | `45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d` | p2_v1_20260722/evidence/s4_task_code.py(== CC4 V3 TASK_SHA256) |
| canonical evaluator | `224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1` | eval_phase2_unified.py(STATIC_PROTOCOL_ANCHOR_NOT_EXECUTED;== V3 EVALUATOR_SHA256) |
| **REJECTED** P2-v0 任务 | `df7cde78bc4ce1067a543063d0a23037046ea9a5975ca953076e59f93e29e6f5` | P2-v0-exploratory-invalid-for-attribution;**禁用**(REJECTED_ALTERNATIVE) |

wrapper / multitask 的 SHA 与 CC4 V3 冻结锚点(WRAPPER_SHA256 / ENV_SOURCE_SHA256)逐一相等 → 复用 V3 身份绑定,无第二套。

## 2. MiniCraftax EnvState 顶层字段(53,逐行摘自 craftax_state.py 7ed6eed0)

`task_id, map, item_map, mob_map, light_map, down_ladders, up_ladders, chests_opened, monsters_killed, player_position, player_level, player_direction, player_health, player_food, player_drink, player_energy, player_mana, is_sleeping, is_resting, player_recover, player_hunger, player_thirst, player_fatigue, player_recover_mana, player_xp, player_dexterity, player_strength, player_intelligence, inventory, melee_mobs, passive_mobs, ranged_mobs, mob_projectiles, mob_projectile_directions, player_projectiles, player_projectile_directions, growing_plants_positions, growing_plants_age, growing_plants_mask, potion_mapping, learned_spells, sword_enchantment, bow_enchantment, armour_enchantments, boss_progress, boss_timesteps_to_spawn_this_round, light_level, achievements, state_rng, timestep, fractal_noise_angles, running_original_return, task_params`

- 地图为 **3D**:`map[player_level, row, col]`,`mob_map[player_level, row, col]`。
- **floor 身份 = `player_level`**;`num_levels == 9`(world_builder build 拼接 9 层)。
- `achievements` 为 `jnp.zeros((len(Achievement),), bool)`,按 `Achievement.X.value` 索引。
- `inventory` 与 `melee/passive/ranged_mobs` 类型为 **craftax.Inventory / craftax.Mobs**(`from craftax.craftax.craftax_state import Inventory, Mobs`)。

## 3. Mobs 字段(摘自 WorldBuilder._generate_empty_mobs / build,world_builder.py 96536bbf)

`position[num_levels,max_mobs,2]int, health[num_levels,max_mobs]float, mask[num_levels,max_mobs]bool, attack_cooldown[num_levels,max_mobs]int, type_id[num_levels,max_mobs]int`

真实访问模式(game_mechanics.py):`state.melee_mobs.position[player_level, idx, 0/1]`、`.mask[player_level, idx]`、`.replace(...)`。

## 4. 合法 builder API(scaffold 唯一合法机制)

`WorldBuilder(rng, static_params, params)` → `set_starting_floor(level)` / `set_player_stats` / `set_player_inventory(dict)`(经 `inventory.replace`)/ `set_weapon_enchantments` / `set_armour_enchantments` / `set_learned_spells` / `set_monsters_killed(level,count)` / `place_block` / `fill_area` / `add_mob(level,mob_name,type_id,position,health)` / `add_mobs_randomly_near(...)` / `place_randomly* ` / `build(rng) -> EnvState`(health=9.0、food/drink/energy/mana=9、achievements=zeros、boss_progress=0、timestep=0)。

combat.py 源码注释即 “ADDED SCAFFOLDING”,证明 **WorldBuilder 支架是仓库原生合法机制**(非我们新造)。

## 5. canonical Stage4 DEFEAT_KOBOLD 任务事实(摘自 s4_task_code.py 45fdd17c)

- `relevant_achievements=[Achievement.DEFEAT_KOBOLD]`,`label="DEFEAT_KOBOLD"`,`completed_achievements=[]`
- `get_task_params() = TaskParams(needs_depletion_multiplier=0.3)`
- `generate_world`:`WorldBuilder → set_starting_floor(2) → set_monsters_killed(2,8) → set_player_inventory({wood7,stone27,coal3,iron3,sapling1,pickaxe3,sword3,bow1,arrows7,torches10}) → build → 移除 floor-2 up-ladder(item_map.at[2,up].set(ItemType.NONE.value))`
- docstring:`S4_dark(全黑,无半径辅助)`、**kobold 实际在 floor 3,必须下行击杀**、achievement embedding = 67 维。

**据此冻结的前后半段边界(源码可证)**:FRONT_FLOOR=2(黑暗走廊)、BACK_FLOOR=3(kobold/目标层)、CORRIDOR_EXIT_FLOOR=3(下行至 floor3 即出走廊)。

## 6. 必须 BLOCKED_SOURCE_SEMANTICS / BLOCKED_ENVIRONMENT 的项(本机无 craftax,**不猜测**)

- `Achievement.DEFEAT_KOBOLD` 的整数索引(谓词代码**符号化引用**,主机运行时绑定)。
- Kobold 的 `MeleeMobType/MobType` type_id 值(`add_mob` 参数,主机绑定)。
- `ItemType.NONE` 值、`BlockType` 可行走集(图距离 traversability)。
- `craftax.Inventory` 完整字段表、`get_distance_map` 原语、`static_params.num_levels/map_size` 精确值。

这些在真实 JAX/craftax==1.4.5 主机上绑定;在此之前相关真实运行标 `BLOCKED_ENVIRONMENT`,**绝不**以硬编码整数/截图/印象替代。

## 7. 不存在字段的明确否定(禁止发明)

源码中**不存在** `darkness_level / corridor_length / monster_density_mode / boss_distance / boss_area_id`。黑暗由 `light_map/light_level` + dungeon `BlockType.DARKNESS` 表达;“走廊出口/ Boss 区”由 `player_level` 转移与 `down_ladders` 位置表达(见边界设计报告与 REJECTED_ALTERNATIVE)。

## 8. 自检

`python tools/tier3_scaffolded_evaluation/tier3_source_audit.py --self-test`
→ `TIER3_SOURCE_AUDIT_SELF_TEST_PASS (in-repo MATCH=12, recorded-external=0, envstate_fields=53, mobs_fields=5)`,exit 0。
