# Tier3 边界语义设计 (boundary design)

- 任务: CC4_TIER3_SCAFFOLDED_EVALUATION_ENVIRONMENT_V1
- 分支: `henry/tier3-scaffolded-evaluation`(基线 `7443aec`)
- 权威定义机读文件: `schemas/tier3_boundary_schema_v1.json`(由 `tier3_boundary_schema.py --emit` 生成)
- 谓词实现: `tools/tier3_scaffolded_evaluation/tier3_event_predicates.py`(`predicate_code_sha256` 绑定)
- 状态: **PASS**(`tier3_boundary_schema.py --self-test` exit 0;events=10;谓词 self-test checks=29)

## 1. 设计原则

1. **源码可证**:每条边界事件都记录其读取的真实 EnvState 字段与来源文件 SHA(见 `tier3_source_audit.py`),不引用任何源码不存在的字段。
2. **评测器侧私有**:所有边界判定 `visible_to_student=false`,不进入 Student observation(无特权信息、无隐藏方向、无最短路径提示)。
3. **单一事件词汇表**:`mechanism_UED.tier3_boundary_schema/v1`,CC2/CC3 不得另造第二套走廊/Boss 区谓词。
4. **冻结 + 拒绝记录**:对语义模糊的边界(走廊出口、Boss 区)冻结唯一 V1 定义,并记录 REJECTED_ALTERNATIVE 及拒绝理由。
5. **符号化**:凡依赖 craftax 枚举整数(achievement 索引、Kobold type_id)者,代码以符号引用,主机运行时绑定;本机标 BLOCKED_ENVIRONMENT,绝不硬编码。

## 2. 楼层锚点(由 canonical 任务 45fdd17c 直接导出)

canonical Stage4 `generate_world`:`set_starting_floor(2)` + `set_monsters_killed(2,8)` + winner-median 装备;docstring 明示 **kobold 实际在 floor 3,必须下行击杀**;floor-2 up-ladder 被移除(只能向前/向下)。floor 身份 = `player_level`(地图 3D `map[player_level,row,col]`)。

| 锚点 | 值 | 来源证据 |
|---|---|---|
| FRONT_FLOOR(黑暗走廊层) | 2 | `set_starting_floor(2)` |
| CORRIDOR_EXIT_FLOOR | 3 | “kobold on floor 3,必须下行” → 下行即出走廊 |
| BACK_FLOOR(kobold/目标层) | 3 | 同上 |

## 3. 冻结 V1 谓词

| 事件/谓词 | 冻结定义 | 主字段 |
|---|---|---|
| valid_full_start | floor2 入场 & 存活 & timestep==0 & 未 DEFEAT_KOBOLD | player_level/health/timestep/achievements/inventory/item_map |
| valid_front_scaffold_start (NEG14) | player_level==2 & 存活 & timestep==0 & 未过出口(<3) & 未 DEFEAT | player_level/health/timestep/achievements |
| valid_back_scaffold_start (NEG15/16) | player_level==3 & 存活 & timestep==0 & **floor3 有活 Kobold**(RANGED type_id 3,HP 8.0) & 未 DEFEAT | + ranged_mobs(position/health/mask/type_id) |
| front_half_entered | player_level==2 | player_level |
| **front_floor_transition_reached**(FRONT 主事件) | **from_level==2 & to_level>=3**(player level 2→3 转移) | player_level 转移 |
| **corridor_exit_reached** | **player_level >= 3**;状态 = **PENDING_EQUIVALENCE_ALIAS**(真实地图证明楼层转移必经目标走廊之前,不定义成功;与主事件矛盾→FailClosed) | player_level |
| back_half_entered | player_level==3 | player_level |
| **boss_area_reached** | **player_level == 3**;BACK_L2 中为 **N/A**(仅保留于冻结词汇表,不作为 BACK 指标) | player_level |
| kobold_engaged | floor 上 Kobold(RANGED 类)有 active attack_cooldown / 受伤记录(每态代理) | ranged_mobs |
| **defeat_kobold** | **achievements[Achievement.DEFEAT_KOBOLD.value]==True**(符号解析,index=41) | achievements |
| front_half_progress | GRAPH_DISTANCE_PROGRESS ∈[0,1],GRAPH_DISTANCE | player_position/map/down_ladders |

## 4. FRONT 进度:dense metric = GRAPH_DISTANCE_PROGRESS

- 方法:**GRAPH_DISTANCE** — 评测器私有 traversability mask(由 `map[player_level]` + `BlockType` 可行走集导出)上做 BFS 最短路。
- 出口 = floor-2 `down_ladders` 位置。
- 归一化:`progress = clip(1 - d(current,exit) / max(d(start,exit), 1), 0, 1)`。
- 范围 `[0,1]`(越界→FailClosed,NEG17)。
- **单调性不保证**:死胡同/往复可使 `d_t` 增大 → 进度回落;暂态死胡同记 0.0。
- Fail-closed 策略:出口不可达且无显式 blocked 标签→FailClosed(NEG18);当前位置越界/不可走→FailClosed。
- traversability mask / 地图拓扑为评测器私有,不进 observation。REGION_PHASE 为文档化的备用降级方案(本轮不实现)。

## 5. REJECTED_ALTERNATIVE(已记录,冻结不改)

| 边界 | 被拒方案 | 拒绝理由 |
|---|---|---|
| CORRIDOR_EXIT_REACHED | 站在 floor-2 down_ladder 瓦片上 | 真实游戏事件是楼层转移(player_level);瓦片级定义徒增脆弱,不改诊断能力 |
| BOSS_AREA_REACHED | boss_progress > 0 | boss_progress 驱动 Necromancer 生成机制(另一系统),非 DEFEAT_KOBOLD 目标 |
| DEFEAT_KOBOLD 检测 | 硬编码 achievement 整数索引 | 必须从 craftax constants 符号解析 Achievement.DEFEAT_KOBOLD |
| FRONT 进度 | 曼哈顿距离 / 屏幕像素 / 瓦片颜色 | 非源码可证;忽略墙/门;会高估离廊位置 |
| BACK_L2 身份 | 声称同时评估 Boss 区域搜索 | canonical FULL reset 在 floor 3 天然零 mobs,scaffold 必须显式加一只 Kobold;故 BACK_L2 只评 combat,`boss_area_reached`/`time_to_boss_area`/`BACK_BOSS_NOT_FOUND` 全部 N/A |

## 6. 科学边界(写入 schema 与所有 config)

- 三个场景共享同一冻结评测合同:`action_mode=greedy_argmax`、`observation_schema=canonical_craftax_symbolic`、`action_space=canonical_craftax_action_set`、`max_timesteps=4096`。
- **前后半段 scaffold 仅用于机制诊断**,`scaffolded_results_can_replace_full_task=false`;scaffold state-bank hash(`FRONT_/BACK_SCAFFOLD_STATE_BANK_HASH`)**永远不得**冒充 `GLOBAL_WORLD_SET_HASH`(后者属 seed42 canonical world materializer,当前 BLOCKED_SOURCE_UNVERIFIED)。
- 本轮无 Student 性能数据,任何结果标签不得为 FRONT_SCAFFOLD_EVALUATION=PASS / TIER3_FRONT_HALF_BREAKTHROUGH。

## 7. 自检

`python tools/tier3_scaffolded_evaluation/tier3_boundary_schema.py --self-test`
→ `TIER3_BOUNDARY_SCHEMA_SELF_TEST_PASS (events=10, predicate_code_sha256=a4fba86b054d...)`,exit 0;并与已提交 `schemas/tier3_boundary_schema_v1.json` 做谓词 SHA 漂移校验。

### 7.1 `predicate_code_sha256` 绑定基准(漂移修复,静态证据)

- 冻结值(JSON 顶层 + 10 条事件,共 11 处):`05ac6edcb7baecc5bd4fd25138da86de934f76d023dad2313d755f3b85b6b3d1`。
- **基准 = LF 归一化后的源码内容 SHA256**(EOL-independent),等于 git blob(`d20ead15...`)所存内容的 SHA256(已验证 `git cat-file blob | sha256` = 上值)。
- **根因与修复**:总控复审发现绑定漂移 —— 原 binder 对工作区**原始字节**取 SHA,而 `core.autocrlf=true` 下同一 clean 文件的工作区字节可为 LF 或 CRLF 两种合法形式(blob 恒为 LF):LF 形式 = `05ac6edc...`,CRLF 形式 = `d66fe614fb99278544865a87098c62caaa222c9fd8c47e4b97ca7d45429d5568`(即总控本机 CRLF 工作区复跑所测“当前 SHA”)。两者是**同一源码**(blob 未变,`a4075f8..HEAD` 谓词源码零改动)的行尾别名。修复:`predicate_code_sha256()` 在哈希前做 CRLF→LF 归一化,冻结值与任一 checkout 形式一致,漂移类问题收敛。
- 验证:CRLF 字节归一化后的 SHA256 = `05ac6edc...` = 冻结值(即 CRLF 工作区上的自检重算必然命中冻结绑定)。
- 注意:这是**绑定基准的确定性修复**,不改任何谓词语义、不改事件集、不改冻结边界定义;scaffold 评测科学边界(§6)不变。
- **后续**:fast-track 语义收口对谓词源码做了合法编辑(新增 `front_floor_transition_reached` 纯函数、`corridor_exit_reached` 降为 PENDING_EQUIVALENCE_ALIAS、Kobold 绑定改 RANGED type_id 3),冻结值随之**合法更新**为 `a4fba86b054d20412fc1df2c79e7000d66b0525decb1801fa474ee7fb0d25b4c`(仍是 LF 归一化基准,11 处一致;事件集保持 10 个,无 schema 扩张)。`05ac6edc...` 作废。

### 7.2 fast-track 语义收口记录(本轮)

- **FRONT_L2**:主事件 = `FRONT_FLOOR_TRANSITION_REACHED`(player level 2→3);起点 = floor 2 合法 scaffold;primary metric = `P_FRONT_FLOOR_TRANSITION_REACHED_GIVEN_VALID_START`;dense metric = `GRAPH_DISTANCE_PROGRESS`;`CORRIDOR_EXIT_REACHED` 仅为 **PENDING_EQUIVALENCE_ALIAS**,真实地图证明楼层转移必经目标走廊之前不定义成功;transition=True 且 alias 显式 False → FailClosed。
- **BACK_L2**:identity = `BOSS_COMBAT_SCAFFOLDED`;起点 = floor 3 + 必须有 live Kobold(RANGED type_id 3,HP 8.0)+ t0 `DEFEAT_KOBOLD` 必须为 false;primary metric = `P_DEFEAT_KOBOLD_GIVEN_VALID_BACK_START`;`boss_area_reached`/`time_to_boss_area`/`BACK_BOSS_NOT_FOUND` 标记 **N/A**;不再声称 BACK_L2 评估 Boss 区域搜索。
- **真实物化接口**(JAX + craftax==1.4.5 主机已就绪,证据见 Commit 2):canonical-rng 真实 FRONT/BACK state bank 物化、V3 payload hash、per-state + bank 级 field manifest、双独立 OS 进程一致性、canonical 环境合同断言(obs `(8335,)`、actions 43)、bank-state reset 与 canonical reset 叶级+obs 等价。
- 事件词汇表、schema 文件数量、NEG 测试数量均**不变**(10 事件 / 26 NEG):只原地收口,禁止 schema 扩张。
