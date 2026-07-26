# World-generation path audit + world-identity schema (CC4 round-6 -- three+four)

- UTC: `2026-07-26T15:04:30Z`
- **WORLD_GENERATION_SOURCE_PATH = FOUND**
- **WORLD_MATERIALIZER_IMPLEMENTATION = PROCEEDS_STATIC** (real run NOT_RUN on this JAX-less host)
- Located from ACTUAL repo source (`dicode_src/`/`dicode_v6/`) + read-only tar extract; two independent Explore agents cross-confirmed; CC4 hand-verified the eval lines and every SHA. NOT a guessed path.

## Canonical evaluator
- `D:/Projects/dicode-codex-director/audit_outputs/global_raw_data_extract_20260726T110032Z/home/oseasy/experiments/student_upgrade_wave1_4gpu/eval_phase2_unified.py`
- SHA256 `224514026aefd273a6647e055fc2e1a434760dc5f4b6b9acd0624bbba57035a1`  (anchor `224514026aefd273` match = **True**)
- constants: {"DK": "Achievement.DEFEAT_KOBOLD (:73)", "SEWERS": "Achievement.ENTER_SEWERS (:74)", "NUM_ENVS": "256 (:75)", "NUM_STEPS": "4096 (:76)", "EVAL_SEED": "42 hardcoded (:77)", "EVAL_SHA256_self_hash": "(:79-80)"}

## Reset-RNG derivation -- PURE POSITIONAL split (NO fold_in)

```
EVAL_SEED = 42                                      eval:77 (hardcoded module const)
rng = jax.random.PRNGKey(EVAL_SEED)                 eval:169
rng, reset_rng = jax.random.split(rng)              eval:170  -> take [1]
obsv, log_state = env.reset(reset_rng, ctor)        eval:171  (wrapper splits internally)
  key, _rng = jax.random.split(key)                 wrappers_cl:228  -> take [1]
  reset_rngs = jax.random.split(_rng, num_envs=256) wrappers_cl:229
  obs, env_state = vmapped_reset(reset_rngs, params, task_ids, task_embeddings)  wrappers_cl:231
  world i receives reset_rngs[i]
  reset_env: rng, world_rng = jax.random.split(rng) multitask.py:129  -> take [1]
  state = lax.switch(task_id, world_gen_fns, world_rng)  multitask.py:132
  S4 generate_world: rng,_r = split(rng)            s4_task_code:39; WorldBuilder(_r); s=b.build(rng) s4:47
  WorldBuilder: generate_full_base_world split(rng,7)+split(rng,4); build split(rng,3)->state_rng,potion_rng; potion_mapping=permutation(potion_rng,arange(6))  world_builder:785,793,1186,1254
```

- closed form (outer): `world_key[i] = split( split( split( PRNGKey(42) )[1] )[1], 256 )[i]`
- consequence: a single world's key is NOT independently derivable from (seed, world_index); it depends on the WHOLE 256-way batch and ordering. The materializer MUST reproduce the entire batch (call env.reset(reset_rng) once and index [i]); feeding a pre-split single key back into env.reset would split it AGAIN and yield WRONG worlds.
- fold_in count in dicode_src = **0**; in evaluators = **0**
- **evaluation_seed enters the real world-gen RNG**: PRNGKey(EVAL_SEED=42) at eval:169 is the ROOT of the world-generation split chain. evaluation_seed GENUINELY enters the world-generation RNG (not merely action sampling). Changing the numeric seed changes PRNGKey -> the entire split tree -> every world. (The action stream uses a separate rng split inside the scan, eval:134/146, also seeded from EVAL_SEED.)
- **world_index enters positionally**: POSITIONALLY: world_index selects reset_rngs[i] = split(_rng, 256)[i] (wrappers_cl:229). It is NOT fold_in(key, world_index) and NOT (seed_base + world_index).
- **PRNGKey(0)**: ONLY jax.random.permutation(rng, task_ids) at :218; identity when num_tasks=1; HARDCODED 0 independent of any seed; does NOT enter the per-world reset key

## Construction functions (all SHA-pinned)
| component | function | source | sha256 |
|---|---|---|---|
| env | MultiTaskMiniCraftaxEnv | dicode_src/src/minicraftax/envs/multitask.py:57-120 | `c8f2d5c3c23476c92ab3897f47bef4df7f202a3bd57360fc1bd4cb92b9498bae` |
| wrapper | DistributedMultiTaskOptimisticLogWrapper | dicode_src/src/dicode/wrappers_cl.py:169-223 | `2ded41d81a98c712620dc1633262f2d185ce7dd22e7cc447db22a6ad04b0ddd8` |
| task | exec(s4_task_code.py) | D:/Projects/dicode-codex-director/audit_outputs/global_raw_data_extract_20260726T110032Z/home/oseasy/experiments/p2_v1_20260722/evidence/s4_task_code.py | `45fdd17c5b34b9f32a7f85b8030437f74d63d16bed2d6f2c683d80454e4d824d` |
| base_task | BaseTask | dicode_src/src/minicraftax/tasks/base_task.py:10-81 | `9b2cb995a807c625fde933a5edf8266dfbf32af3aac9c767f40e41e50586b1fa` |
| EnvState/TaskParams | dataclass defs | minicraftax/craftax_state.py:42/:9 | `7ed6eed02495fa6f0992ebe3e7a2c89b56d2c8d0798915fed76c60e3a5be770b` |

- wrapper byte-identical copies found: **6** (dicode_src == dicode_v6 == dicode_v7fix58_armB (the copy the evaluator loads via V7_SRC sys.path[0]) == gpu0/gpu1 mirrors; all sha256 prefix 2ded41d8)
- **task trap**: the canonical task is the 49-line p2_v1 copy (`45fdd17c...`); the 34-line P2-v0 copy (`df7cde78bc4ce106`) is INVALID-for-attribution and must NOT be used.
- the cc4tmp raw_sources mirror of the canonical task (0df86b26...) is the SAME content CRLF-mangled (CR=49); extract LF copy 45fdd17c... (CR=0) is canonical; identical after CRLF normalization

## World-identity schema -- mechanism_UED.craftax_materialized_world/v1

- **Decision**: serialize the COMPLETE initial EnvState snapshot (all **53** fields at timestep=0); per-world hash then bit-equals the evaluator's `log_state.env_state[i]`.
- **Why not only the 6 immutable fields**: only 6 fields are strictly immutable, but the INITIAL VALUES of the runtime-mutated fields (map terrain, item_map incl. removed floor-2 up-ladder, starting inventory, monsters_killed[2]=8, mob placements, pre-populated achievements, player_position/level) ARE the world's identity. Dropping them would discard result-affecting initial state (forbidden).
- strictly-immutable identity fields (6): `task_id`, `down_ladders`, `up_ladders`, `potion_mapping`, `fractal_noise_angles`, `task_params`
- initial-value-is-identity fields: **46** (runtime-mutated, but reset value is world identity)
- decorative: `state_rng`
- excluded NON-identity: `LogEnvState.episode_returns`; `LogEnvState.episode_lengths`; `LogEnvState.running_original_return`; `obs`
- MUST NOT serialize only: a PRNG key, a seed label, a path, a description, a recipe JSON

### Field table (53)

| field | type | set at reset by | runtime-mutated | identity class |
|---|---|---|---|---|
| `task_id` | int | reset_env state.replace (multitask:165) | no | **STRICT_IMMUTABLE_IDENTITY** |
| `map` | jnp.ndarray(9,48,48) | WorldBuilder terrain | yes(mining/placing) | **INITIAL_VALUE_IS_IDENTITY** |
| `item_map` | jnp.ndarray | WorldBuilder(ladders/torches/chests)+s4 floor-2 up-ladder removed | yes(place/open) | **INITIAL_VALUE_IS_IDENTITY** |
| `mob_map` | jnp.ndarray | WorldBuilder/add_mobs_randomly_near | yes(update/spawn) | **INITIAL_VALUE_IS_IDENTITY** |
| `light_map` | jnp.ndarray | WorldBuilder | yes(torch/light) | **INITIAL_VALUE_IS_IDENTITY** |
| `down_ladders` | jnp.ndarray | WorldBuilder | no(read only) | **STRICT_IMMUTABLE_IDENTITY** |
| `up_ladders` | jnp.ndarray | WorldBuilder | no(read only) | **STRICT_IMMUTABLE_IDENTITY** |
| `chests_opened` | jnp.ndarray | build=zeros(:1249) | yes(counter) | **INITIAL_VALUE_IS_IDENTITY** |
| `monsters_killed` | jnp.ndarray | build at[0].set(10) + s4 set_monsters_killed(2,8) | yes(kill counter) | **INITIAL_VALUE_IS_IDENTITY** |
| `player_position` | jnp.ndarray | WorldBuilder(center/ladder) | yes(move/change_floor) | **INITIAL_VALUE_IS_IDENTITY** |
| `player_level` | int | set_starting_floor(2) | yes(floor change) | **INITIAL_VALUE_IS_IDENTITY** |
| `player_direction` | int | WorldBuilder(Action.UP) | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_health` | float | build=9.0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_food` | int | build=9 | yes(intrinsics) | **INITIAL_VALUE_IS_IDENTITY** |
| `player_drink` | int | build=9 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_energy` | int | build=9 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_mana` | int | build=9 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `is_sleeping` | bool | build=False | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `is_resting` | bool | build=False | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_recover` | float | build=0.0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_hunger` | float | build=0.0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_thirst` | float | build=0.0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_fatigue` | float | build=0.0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_recover_mana` | float | build=0.0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_xp` | int | build=0 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_dexterity` | int | set_player_stats(default 1) | yes(level_up) | **INITIAL_VALUE_IS_IDENTITY** |
| `player_strength` | int | set_player_stats(default 1) | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_intelligence` | int | set_player_stats(default 1) | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `inventory` | Inventory(upstream dataclass) | WorldBuilder + s4 set_player_inventory{wood7,stone27,coal3,iron3,sapling1,pickaxe3,sword3,bow1,arrows7,torches10} | yes(core gameplay) | **INITIAL_VALUE_IS_IDENTITY** |
| `melee_mobs` | Mobs | WorldBuilder(add_mob*) | yes(spawn/update) | **INITIAL_VALUE_IS_IDENTITY** |
| `passive_mobs` | Mobs | WorldBuilder | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `ranged_mobs` | Mobs | WorldBuilder | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `mob_projectiles` | Mobs | build=empty | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `mob_projectile_directions` | jnp.ndarray | build=empty | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_projectiles` | Mobs | build=empty | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `player_projectile_directions` | jnp.ndarray | build=empty | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `growing_plants_positions` | jnp.ndarray | WorldBuilder=zeros | yes(update_plants) | **INITIAL_VALUE_IS_IDENTITY** |
| `growing_plants_age` | jnp.ndarray | WorldBuilder=zeros | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `growing_plants_mask` | jnp.ndarray | WorldBuilder=zeros | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `potion_mapping` | jnp.ndarray | build=permutation(potion_rng, arange(6)) :1254 | no | **STRICT_IMMUTABLE_IDENTITY** |
| `learned_spells` | jnp.ndarray | set_learned_spells | yes(read_book) | **INITIAL_VALUE_IS_IDENTITY** |
| `sword_enchantment` | int | builder setter | yes(enchant) | **INITIAL_VALUE_IS_IDENTITY** |
| `bow_enchantment` | int | builder setter | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `armour_enchantments` | jnp.ndarray | builder setter | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `boss_progress` | int | build=0 | yes(boss_logic) | **INITIAL_VALUE_IS_IDENTITY** |
| `boss_timesteps_to_spawn_this_round` | int | build=50 | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `light_level` | float | build=calculate_light_level(0,params) | yes(recomputed each step) | **INITIAL_VALUE_IS_IDENTITY** |
| `achievements` | jnp.ndarray(bool,67) | build=zeros -> pre-populated in reset_env(multitask:139-161) | yes | **INITIAL_VALUE_IS_IDENTITY** |
| `state_rng` | PRNGKey | build(:1259) from split(rng,3) | yes(overwritten each step; never read by reset) | **DECORATIVE** |
| `timestep` | int | build=0 | yes(+1 each step) | **INITIAL_VALUE_IS_IDENTITY** |
| `fractal_noise_angles` | tuple[int x4] | build copies params.fractal_noise_angles | no | **STRICT_IMMUTABLE_IDENTITY** |
| `running_original_return` | float | build=0.0; wrapper sets per-env after reset | yes(dynamic bonus) | **INITIAL_VALUE_IS_IDENTITY** |
| `task_params` | TaskParams(dataclass) | reset_env state.replace(multitask:166) | no | **STRICT_IMMUTABLE_IDENTITY** |

## Shared-builder status (GATE19)
- strict shared builder = **BLOCKED**
- reason: the real evaluators keep env construction + the three seed lines INLINE (copy-pasted into each eval script) and load the task via exec() of a server-absolute path; there is NO importable shared world-builder. The canonical evaluator file is READ-ONLY and must NOT be modified, so a literal 'evaluator and materializer call the same builder function' cannot be achieved without altering canonical files / breaking the canonical SHA.
- honest substitute: every constant and derivation step embedded in materialize_craftax_world_set_twice.py is asserted EQUAL to the literal value parsed from the canonical source files (static_anchor_check). Against the real sources this round: 12/12 anchors PASS, 0 mismatches. The wrapper the materializer imports (dicode_src) is byte-identical (2ded41d8...) to the copy the evaluator loads; the task exec'd is the canonical 45fdd17c....

## Host environment
- jax importable = **False** ; craftax importable = **False**
- implication: real materialization NOT_RUN on this host; the materializer FAILS CLOSED before emitting any world_set_hash

## Discipline
- no training, no formal evaluation, no real Exact Resume, no matched Replay run
- no fabricated world_set_hash; key-only hash never called a world hash
- 54 frozen files unmodified; SHA256SUMS not rewritten
- did NOT invent a plausible path -- everything line-anchored + SHA-pinned from real source
- evaluator left read-only; strict shared-builder honestly reported BLOCKED with a static-anchor substitute


---

## Round 7 (V3) addendum — GLOBAL_WORLD_MATERIALIZER_RUNTIME_IDENTITY_HARDENING_V3

本轮（V3）在 V2 之上补充四点，V2 结论全部保留、未降级：

1. **运行时执行源码身份绑定（§二/三/四）**：物化器在真实 `import dicode.wrappers_cl` /
   `import minicraftax.envs.multitask` 之后，捕获 `module.__file__` 与
   `inspect.getsourcefile(实际调用的类)`，做 `abspath`+`realpath`（解析 symlink），重新计算
   SHA256，并**要求**与命令行传入的 source 相等，否则 `FailClosed(EXECUTED_SOURCE_IDENTITY_MISMATCH)`。
   字节相同但 realpath 不同的副本会被**拒绝**——不再依赖"多份副本碰巧 byte-identical"。canonical
   S4 task 的 exec 路径同样绑定（完整 SHA + `Env` 类名 + `generate_world`/`get_task_params` 接口），
   不一致即 `FailClosed(TASK_EXECUTED_SOURCE_IDENTITY_MISMATCH)`，无静默回退。
2. **executed vs protocol-anchor 拆分（§三）**：wrapper / environment / task 记入 `executed_sources`
   （物化器确实 import/exec/call）；`eval_phase2_unified.py` 记入 `protocol_anchor_sources`，
   `executed_by_materializer = false`（物化器**复现**其构建+reset 逻辑，但**不执行** evaluator 主程序）。
   绝不写 "evaluator source executed"。
3. **seed-free world payload hash（§七）**：`serialize_world_payload` / `state_payload_hash` 只对
   canonical 序列化后的初始 EnvState payload 取 SHA256，**不含** evaluation_seed / seed_id / source SHA /
   版本。这是"数值 seed 是否真的改变世界"的唯一诚实载体；header-tagged 的 per_world_hash 差异**不是**证据
   （header 本就带 seed 标签）。
4. **world field manifest 持久化（§十一/十二）**：真实 run 输出 `world_field_manifests.json`
   （schema `mechanism_UED.craftax_world_field_manifests/v1`，256 world，每数组字段 path/dtype/shape/nbytes）
   + `world_field_schema_summary.json`（结构差异被**记录**而非静默覆盖），并把 `world_field_manifests_sha256`
   绑定进 `world_hashes.json`；`assert_materialized` 拒绝缺少该证据的结果。

**seed100000 独立 evaluator（§九）**：仓库中确有一个真实、冻结、使用 seed100000 的 evaluator ——
`eval_p7_egomap_paired_256.py`（P7_PAIRED_256，`--seed_base` 默认 100000，`EVAL_SEED=int(args.seed_base)`，
`rng=PRNGKey(EVAL_SEED)` :190，同样的 `DistributedMultiTaskOptimisticLogWrapper(s4_base, PRNGKey(0), ...)` :136）。
raw（CRLF）SHA256 = `f9c864359cfffe7726d93870fd17e52e18a7e49aa9a468471abf59088799a1a9`；LF SHA256 =
`c082db8b82e86b971d8943bd9275ba8b709ffdc0da198fb236c52ccd56c08325`。它以**独立**身份（自己的完整 SHA）记录，
绝不复用 seed42 evaluator 的身份；seed100000 世界集仍是 PARAMETERIZED variant，**不得**冒充 seed42 精确世界集。

**新增门禁**：GATE20 RUNTIME_EXECUTED_SOURCE_IDENTITY（PASS_STATIC_CODE / REAL_RUNTIME_NOT_RUN）；
GATE21 REAL_SEED_WORLD_PAYLOAD_DIFFERENCE（BLOCKED_ENVIRONMENT）；GATE22 SEED_IDENTITY_CLASSIFICATION（PASS）；
GATE23 WORLD_FIELD_MANIFEST_PERSISTED（PASS_STATIC_IMPLEMENTATION，非 PASS_REAL_OUTPUT）。

### V3 冻结标签

- CC4_RUNTIME_SOURCE_IDENTITY_CODE = PASS
- CC4_RUNTIME_SOURCE_IDENTITY_REAL_RUN = NOT_RUN
- EXECUTED_WRAPPER_SOURCE_BINDING = PASS_STATIC
- EXECUTED_ENV_SOURCE_BINDING = PASS_STATIC
- EXECUTED_TASK_SOURCE_BINDING = PASS_STATIC
- EVALUATOR_SOURCE_ROLE = STATIC_PROTOCOL_ANCHOR_NOT_EXECUTED
- SEED42_IDENTITY_CLASS = CANONICAL_EVALUATOR_EXACT_WORLD_SET
- SEED100000_IDENTITY_CLASS = PARAMETERIZED_WORLD_GENERATION_PROTOCOL_VARIANT
- EVALUATION_SEED_STATIC_RNG_BINDING = PASS
- EVALUATION_SEED_REAL_WORLD_PAYLOAD_EFFECT = BLOCKED_ENVIRONMENT
- NEG02_FALSE_PASS_REMOVED = PASS
- WORLD_STATE_PAYLOAD_HASH = IMPLEMENTED
- WORLD_FIELD_MANIFEST_CODE = IMPLEMENTED
- WORLD_FIELD_MANIFEST_REAL_OUTPUT = NOT_RUN
- MATERIALIZER_EVALUATOR_SHARED_BUILDER = BLOCKED_EVALUATOR_INLINE_READ_ONLY
- STATIC_ANCHOR_EQUIVALENCE = PASS
- GLOBAL_WORLD_SET_HASH = BLOCKED_SOURCE_UNVERIFIED (unchanged)
