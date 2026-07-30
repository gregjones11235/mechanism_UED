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

### 7.3 CC2 真实绑定轮记录(CC4_REAL_CC2_POLICY_ADAPTER_AND_INTERFACE_SMOKE)

- **checkpoint 格式绑定**:`load_full_params_readonly` 改按 CC2 `save_ckpt` 真实格式读取 —— `{"params": <DIRECT pytree(numpy 叶)>,"manifest": {params_sha256, step, arm, carry_mode, replay, …, config, tag}}`(不是 `(leaves, treedef)`)。params SHA 用与 CC2 **逐字节相同**的 `_params_sha` 算法(`tree_leaves` 序 + `np.ascontiguousarray(np.asarray(leaf)).tobytes()`)重算,必须等于 `manifest["params_sha256"]`(NEG21),同时记录 checkpoint 文件 SHA。
- **真实 CC2 policy adapter**(`tier3_cc2_policy_adapter.py`,新模块):从 `--cc2_snapshot_root` 导入 CC2 **实际**模块(`ActorCriticTransformerRMT16` / `RMT16Config` / `rmt16_init` / `make_apply_eval_rmt` / `make_update_fn` / `rmt_step_forward`),CC4 零重实现 RMT/GTrXL 状态转移。源码字节绑定 `cc2_policy_source_sha256`(逐文件 LF 归一化 SHA 的有序聚合);模块必须恰从声明根解析(缓存/错根 → fail closed);网络与 `RMT16Config` 从 manifest["config"] 按 CC2 原样重建;carry_mode 只从 manifest 读取。每 episode 以 CC2 driver 约定初始化真实 GTrXL+RMT16 态(memories/mem_mask/mem_idx=window_mem/rmt16_init),greedy_argmax 选动。自检证明 CC2 单一模式分叉(`rmt_advance_tokens`):128 步段边界前两模式 tokens 均不动;边界处 persistent 携带 cross-attention 更新、reset128 清零(更新路径无 gate,非零观测下即可观测)。
- **冻结 bank identity 值绑定**:`FROZEN_FRONT_STATE_BANK_HASH=21aeb7dc…d687` / `FROZEN_BACK_STATE_BANK_HASH=c632e30d…2566` / `FROZEN_FIELD_MANIFEST_SHA256=615d4be4…ee07` / `FROZEN_PREDICATE_CODE_SHA256=a4fba86b…5b4c` / `FROZEN_CANONICAL_TASK_SHA256=45fdd17c…824d`,n=8、seed_base=10_000、stride=1。`verify_frozen_bank_identity` 在真实评测前内存重铸并逐项核验(hash_label/值绑定/seeds/逐条 field manifest/REAL-only/序敏感的 bank hash 重算/source_shas/predicate code SHA/canonical task SHA)+ PROCESS_B 独立复验,任一不符 fail closed;**不写盘、不修改冻结 bank**。纯比较层 `check_frozen_manifest_bindings` 无环境依赖(NEG28 在任意主机可跑)。
- **FRONT progress fail-closed**:`rollout_episode` 删除 `except pred.FailClosed: pass`。start→exit 不可达(NEG18)、off-grid、non-walkable 一律向上传播、中止评测,不得吞掉。进度计算仅在 `player_level==FRONT_FLOOR` 且 `max_level<CORRIDOR_EXIT_FLOOR` 时进行:到 floor3 后 floor2 图距离无定义,dense metric 冻结于转移时刻。ladder tile 用显式 **LADDER_TILE_TRANSIT** 规则处理:floor2 down_ladder(走廊出口)与 up_ladder 位置从归一化视图取出、OR 入静态可走掩码(越界 fail closed),规则写入 docstring,可审计。
- **证书真实值绑定**(NEG27):`eval_binding` 要求 15 个字段全为**真实值** —— state_bank_hash / state_payload_hashes(有序)/ checkpoint_file_sha256 / cc2_params_sha256 / checkpoint_step / carry_mode / run_class / episode_records_sha256 / cc2_policy_source_sha256 / evaluator_source_sha256 / predicate_code_sha256 / observation_shape=[8335] / action_dim=43 / params_unchanged=true / performance_claim_authorized。SHA 字段必须 64-hex **值**(以标签冒充 → fail closed);`INTERFACE_SMOKE` 永不得授权性能声明。
- **真实 CLI**:`tier3_evaluator.py --checkpoint <full_state.pkl> --cc2_snapshot_root <PATH> --scenario {front_l2,back_l2,full,all} --out <DIR> --interface-smoke [--episodes 2] [--max-steps 32]`,输出 `episode_records.jsonl` / `evaluation_result.json` / `evaluation_certificate.json` / `SHA256SUMS`;run_class=INTERFACE_SMOKE,performance_claim_authorized=false。
- **NEG 电池 26 → 28**(仅负向测试扩张;事件词汇表仍 10,schema 文件数不变):NEG27 证书值绑定、NEG28 冻结 manifest 篡改(载荷/哈希/种子三路)。双解释器 FAIL=0。

### 7.4 真实 98,304-step checkpoint 绑定轮记录(CC4_REAL_CC2_98304_CHECKPOINT_BINDING_AND_INTERFACE_SMOKE)

- **真实 manifest 契约实证**(服务器 26/26 checkpoint 只读审计 + 双臂最终 pkl 逐位传回本地):真实 `full_state.pkl` 的 `manifest["config"] == {}` —— CC2 `Cfg`(driver 303-309 行)是类属性配置类,`vars(Cfg())` 按设计为空,`save_ckpt` 写入 `config={k: v for k, v in vars(cfg).items()}` 即 `{}`。网络超参**冻结在 driver 源码**,不在 pickle。真实 manifest 携带 `replay_mode`(原键名,非 `replay`)与 `phase4a_v2` provenance(run_class=`long_run_98304`、sequence_length=129、segment_len=128、base_checkpoint_params_sha256=d4e85af5…)。
- **无猜测重建路径**:`load_cfg_from_driver_source()` 对 SHA 绑定(LF-SHA `453bd1ec…`,五方一致)的 driver 源文件做 **AST 字面解析**(ast.parse + ast.literal_eval,绝不执行、绝不猜测、绝不默认值)恢复 `class Cfg` 全部 11 个必需超参;`build_network_from_manifest(modules, manifest, action_dim, cfg)` 新签名,cfg 来自 driver 源,并加两道一致性门:非空 `manifest["config"]` 与 driver Cfg 逐键一致(冲突 fail closed;`{}` 为真实观测态放行)、`phase4a_v2.segment_len == cfg["num_steps"]`。`driver_source_sha256` 作为新独立绑定进入 checkpoint record 与证书;`cc2_policy_source_sha256=31c1092c…` 保持不变。
- **证书进程 provenance 绑定**(NEG29):`eval_binding` 必填新增 `driver_source_sha256`(64-hex)/ `process_pid`(正整数)/ `process_argv`(非空字符串列表)/ `run_start_utc` / `run_end_utc`(可解析 ISO-8601)/ `run_exit_code`(必须 0)。evaluator 在 run 起始捕获真实 `os.getpid()` / `sys.argv` / UTC 起始时刻,全部 NEG23 门通过后盖终止时刻与 exit_code=0 写入每份证书。
- **反污染强制门**(handover §7):`run_interface_smoke` 在任何 checkpoint 绑定之前检查 `RMT16_POSTJAX_BINDING_SELFTEST`,取值非空且非 "0" → fail closed(该 hook 使 CC2 driver 训练前 rc=0 提前退出 = 假成功);门在 JAX import 之前触发,任意主机可验。
- **NEG 电池 28 → 29**(仅负向测试扩张;事件词汇表仍 10,schema 文件数不变):NEG29 证书 provenance 缺失/非法(10 条篡改路径全部拒绝,完整绑定放行)。双解释器自检 5 套件 FAIL=0(base 纯门禁 + venv JAX 全链)。

### 7.5 冻结 98,304 性能评估 + provisional 选择轮记录(CC4_FROZEN_98304_PERFORMANCE_EVALUATION_AND_PROVISIONAL_STUDENT_SELECTION)

- **机读 checkpoint contract**(任务 §一):`configs/tier3_cc2_final98304_checkpoint_contract_v1.json` 冻结双臂最终 checkpoint 身份(persistent:file `2866b5de…`/params `aa6ba440…`;reset128:file `de3a159f…`/params `78a14cc6…`;step=98304)与公共身份(replay_mode=original_vtrace、seed=42、run_class=long_run_98304、sequence_length=129、segment_len=128、crosses_boundary=true、base `d4e85af5…`、driver `453bd1ec…`、policy `31c1092c…`)。`checkpoint_contract_sha256` = 去掉该字段自身的规范 JSON(sort_keys、紧凑分隔符)SHA256 = `7dda2bc7517342b189a1f1ba949d620eb4d1c978e252b74f4e2bdeb61363f2e5`,文件自校验;`tier3_checkpoint_contract.py` 再加冻结值纵深防御(即使重新封缄、自校验通过,内容漂移仍 fail closed)。评估 CLI 必填 `--checkpoint-contract <PATH> --arm {persistent|reset128}`,对**实际加载字节**(文件 SHA、重算 params SHA、manifest 全字段、driver/policy 源 SHA)逐项核验,任一不符 → 稳定 ID `FINAL_98304_CHECKPOINT_CONTRACT_MISMATCH`;绝非拷贝字段进证书。
- **两段式证书绑定**(任务 §二):评估器引擎**不得自书** exit code。引擎写 ENGINE 阶段证书(无进程 provenance;若携带任何 exit 字段或旧式 `run_exit_code`/`process_pid`/`process_argv`/`run_start_utc`/`run_end_utc` → NEG37 fail closed),不写 SHA256SUMS。父进程 `tier3_evaluation_runner.py`:spawn 引擎子进程 → 捕获 pid/argv/started_at → `wait()` 取**字面** exit code → rc≠0:只在最终目录写 `run_status.json`(ENGINE_FAILED),临时目录**永不提升**,不存在任何 PASS 证书;rc=0:先重验 ENGINE 绑定干净、注入 runner provenance(`child_process_pid`/`child_process_argv`/`actual_started_at_utc`/`actual_finished_at_utc`/`literal_exit_code=0`/`exit_source=wait_pid`/`inferred_from_log=false`/`evaluation_runner_source_sha256`),跑完整 finalize 验证(assert_eval_binding_complete + NEG24/25),再写 run_status.json(FINALIZED_PASS)+ SHA256SUMS,fsync,`os.replace` 原子改名。
- **Student 状态四分**(任务 §三):删除歧义 `has_student_data`。证书携带 `student_state` 四元组(student_checkpoint_loaded / student_policy_rollout_executed / performance_evaluation_executed / scientific_claim_authorized,末项恒 false);状态标签拆为 REAL_STUDENT_INTERFACE_SMOKE / REAL_STUDENT_PERFORMANCE_EVALUATION(按 mode 置 EXECUTED)+ FORMAL_SCIENTIFIC_CLAIM=NOT_AUTHORIZED_SINGLE_TRAINING_SEED;旧键 REAL_STUDENT_EVALUATION 从一切 mode 移除,真实跑过 Student rollout 永不再标 NOT_RUN。smoke=(true,true,false,false),正式选择=(true,true,true,false)。
- **输出新鲜门 + 原子写**(任务 §四):最终目录必须不存在或为空,否则 `EVALUATION_OUTPUT_DIRECTORY_NOT_FRESH`;`.inprogress` 临时目录已存在 → fail closed;禁止 rm -rf / 覆盖 / 追加旧 JSONL / 自动改名。全部 JSON/JSONL/证书/SHA256SUMS 经临时目录 → fsync → 校验 → 原子 rename。
- **性能评估入口**(任务 §六):`--performance-evaluation` 与 `--interface-smoke` 互斥;run_class=PROVISIONAL_STRONG_STUDENT_SELECTION,action_mode=greedy_argmax,max_timesteps=4096,performance_evaluation_executed=true,scientific_claim_authorized=false,provisional_selection_only=true;禁止 SOTA / FORMAL_SCIENTIFIC_PASS / *_PROVEN_BETTER / TIER3_SOLVED 任何字样(比较器对全文档字符串扫描,fail closed)。
- **冻结评估起点**(任务 §七):FULL = 64 个留出 canonical reset 种子 **200000..200063**;FRONT_L2/BACK_L2 = 全部 8 个冻结 bank 状态各恰好一次(`evaluator.performance_start_schedule()` 纯函数,**双臂同一调度**,严格复算一致);greedy 下绝不重复 scaffold 状态凑样本。证书对 PROVISIONAL 强校验三场景齐备 + 种子精确复现 + 64/8/8 条目计数;smoke 证书仅按已跑场景做结构校验。
- **指标补全**(任务 §八,仅用现有 schema 字段,不扩 scenario 语义):FRONT dense 增加 median 与逐状态配对 `per_state_progress`;BACK 增加 `diagnostics`(kobold_engaged_count、survival{died/defeat/mean_timesteps/max_observed}、failure_taxonomy);`time_to_first_engagement`/`time_to_kill`/`damage` 非冻结 episode schema 字段 → 诚实记 null + schema_note,不扩 schema。
- **冻结比较规则(范围修正后)**(任务 §九 + 总控范围修正):本轮只比较 **Persistent RMT16 + Original V-trace Replay** 与 **Reset128 RMT16 + Original V-trace Replay** 两个承载方式,是机制对照,**不是**总体强 Student 选拔。`tier3_provisional_selection.py`(文件名保留)字典序固定规则,运行前冻结、机读常量 `SELECTION_RULE`(rule_version=`tier3_rmt16_carry_mode_comparison_rule/v1`,output_field=`RMT16_CARRY_MODE_WINNER`)逐字嵌入比较输出:① FULL success_count;② FRONT transition_count;③ FRONT 均值进度(容差 1e-12;若均值亦同,规则允许任意偏向 —— 本实现确定性地继续第 ④ 级);④ BACK defeat_count;⑤ 全同 → INCONCLUSIVE。输出 **RMT16_CARRY_MODE_WINNER=PERSISTENT|RESET128|INCONCLUSIVE** + 固定范围字段(CANDIDATE_SCOPE=RMT16_ORIGINAL_VTRACE_PAIR_ONLY、MECHANISM_QUESTION=PERSISTENT_CROSS_SEGMENT_CARRY_VS_RESET128、OVERALL_STRONG_STUDENT_SELECTION_AUTHORIZED=false、STRONG_STUDENT_V1=NOT_SELECTED、EXISTING_STUDENT_BAKEOFF_REQUIRED=true、SINGLE_TRAINING_SEED=true、SCIENTIFIC_SUPERIORITY_CLAIM=false、REQUIRES_MULTI_SEED_CONFIRMATION=true),赢家可从最终指标严格复算(自测含对称性/确定性/全五级平局路径)。overclaim 门禁扩展:PROVISIONAL_STRONG_STUDENT_RECOMMENDATION / STRONG_STUDENT_V1(=PERSISTENT|=RESET128)/ BEST_OVERALL_STUDENT / ALL_STUDENT_BAKEOFF_WINNER 任一字样(值扫描)一律 fail closed(沿用现有门禁,不新增 NEG 编号)。comparator 先验双臂 SHA256SUMS + 全证书 finalize 验证 + 同一 contract SHA + 64/8/8 计数,再比较;绝不按 smoke 结果、事后检查、其他 Student 或 bakeoff 结果改规则。
- **run_status 完整 provenance 绑定**(总控 §三):每个 `run_status.json`(PASS 与失败路径皆然)绑定 `local_commit_sha`(`git rev-parse HEAD`,失败记 UNAVAILABLE,绝不让元数据探测弄崩运行)、`local_tree_sha`(`HEAD^{tree}`)、`push_status`(CLI `--push-status`,默认 NOT_PUSHED_AT_RUN_TIME;非强制 push 网络失败记 BLOCKED_NETWORK,**不是代码失败**)、`evaluator_source_sha256`、`evaluation_runner_source_sha256`;证书侧 `evaluator_git_commit` + `evaluator_source_sha256`(引擎绑定)+ `evaluation_runner_source_sha256`(runner 注入)构成同源证据链。
- **运行顺序与证据**(任务 §十/§十一):先代码提交(题 `fix(eval): finalize checkpoint contract and performance evaluation provenance`)非强制 ff push;再严格顺序 persistent → reset128 → comparator,绝不并行两个 JAX 评估器,不训练/不复跑/不更参/不新 LLM。证据提交只含小件可审计产物(两臂 run_status / 证书 / 结果 / episode_records / SHA256SUMS + cross_arm_comparison.json + 短报告),**绝不提交 checkpoint 文件**;产物不含占位符、不以本地绝对路径为唯一身份(身份全为 SHA)。
- **NEG 电池 29 → 42**(仅负向测试扩张;事件词汇表仍 10):NEG30-36 checkpoint contract 七路篡改(file/params SHA、step8192 冒充、arm/carry、replay_mode、seed/run_class、base SHA,稳定 MISMATCH id);NEG37 自书 exit code 全形态拒绝;NEG38 引擎证书无 runner provenance 永不能 finalize;NEG39 输出新鲜门;NEG40 状态标签四分 + 旧键移除;NEG41 双臂调度同一 + 漂移拒绝;NEG42 比较规则固定可复算 + 固定范围字段 + 禁选词汇拒绝(RMT16_CARRY_MODE_WINNER 语义,不新增 NEG 编号)。双解释器自检 8 套件 FAIL=0。
