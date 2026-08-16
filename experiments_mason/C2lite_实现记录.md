# C-2-lite 实现记录（队列第 1 项：三层实现 + 单测）

> 2026-07-11。对应 `C2lite_设计稿.md` §5.1。状态 = **实现完成,51/51 测试通过(本地,无 GPU 依赖)**,待 pod 上复跑测试 → 短跑验证(队列第 2 项)。

---

## 1. 交付清单

| 层 | 文件 | 内容 |
|---|---|---|
| 数据 | `src/dicode/skill_preflight/prereq_graph.py`(新) | 67 成就直接前置图 `DIRECT_PREREQS` + 楼层阶梯 `floor_grants` + 库存替代映射 `inventory_grants`;导入时自检(全覆盖/无环/前置不深于自身) |
| §1 调度 | `src/dicode/skill_preflight/skill_scheduler.py`(改) | `pick_target(frontier_mode="prereq", prereq_threshold=0.3)`:候选 ⟺ 自身 SR < mastery_threshold 且**每个直接前置**单独 ≥ prereq_threshold;排序 = 最低 SR 优先、深层 tier 破平;全过线时回退 tier 模式 consolidate。`SchedulerTarget` 新增 `mode` / `sr_snapshot` 字段(默认参数下与旧行为逐位一致,回归测试钉死) |
| §2 生成 | 同上 + `gen_manager.py`(改) | `format_target_for_prompt_one_step`(设计段:mastery 快照三档 + 四条 one-step 契约,走 [Curriculum focus] 注入)与 `format_scaffold_rules_for_coder`(编码段:追加在 gen_env user prompt 尾部,明文声明**压过 few-shot 示例与 docstring 冲突文本**——种子任务示例是泄漏的最强教师,必须显式盖过)。挂载点:`run_dicode.py` [A] hook 每 session 设置/清空 `env_generator.scaffold_rules_block` |
| §3 验证 | `src/dicode/skill_preflight/scaffold_gate.py`(新)+ `evolution_efficient.py`(改)+ `gen_manager.py`(改) | C-0 静态闸 `check_code`:R1 预标已掌握 / R2 预标 relevant / R3 焦点直接前置被脚手架(预标 ∪ 库存授予 ∪ 起始层跳过)。挂在 worker 编译校验后:违规 → 罪证进现有 reflection 模板(`repair_scaffold_violations`,复用 `user_prompt_reflection_not_compilation_error`)重生成 ≤`scaffold_gate_retries`(默认 2)次,仍违规才丢弃(`error="scaffold_gate: R…"`,可事后统计) |
| 单测 | `tests/test_prereq_graph.py` / `test_prereq_frontier.py` / `test_scaffold_gate.py`(新) | 24 个新用例;含设计稿 §5.1 指定的两种验证:**末点 JSON 回放**(eval_PROBET02_seed0.json@2400 真数据过判据)与 **task_19 复刻件**过闸;另含与根目录 `scaffold_audit.audit_code` 的**平价测试**(两套 AST 抽取不许静默漂移) |

全部 flag 默认关;`pick_target` 无新参调用逐位复现旧行为 → 旧 run 可复现性保持。

## 2. 前置图溯源(重要:两处反直觉地面真相)

对照 craftax==1.4.5 wheel 源码逐条核实(挖矿镐级/合成配方/LEVEL_ACHIEVEMENT_MAP/FLOOR_MOB_MAPPING×贴图索引/宝箱掉落/附魔台位置),要点:

1. **兽人在 1 层(dungeon),侏儒在 2 层(gnomish mines)** —— FLOOR_MOB_MAPPING 实测,与直觉相反;`defeat_orc_*` 前置 = `enter_dungeon`(不是 sewers)。
2. **附魔台**:ICE 台在 sewers(3 层,耗 sapphire)、FIRE 台在 vaults(4 层,耗 ruby)→ enchant 的最廉路径 = sewers + sapphire。
3. 书/药水/宝石走宝箱掉落线 → `learn_*`/`drink_potion`/`collect_sapphire|ruby` 的正则前置 = `open_chest`(挖宝石的钻镐路线是替代路径,记录在 `_ALTERNATIVE_PATHS`,不入边——连词语义下入贵边会错误封锁调度)。

## 3. ★ 回放发现(短跑观察点,非 bug)

用 probe 末点真数据回放 prereq 判据:**合格池 > 默认上限 6**。若干真 0.0% 且前置齐备的技能(learn_fireball/learn_iceball 走 open_chest 62%、enter_gnomish_mines 走 enter_dungeon 66%、make_iron_armour、defeat_orc_mage)在"最难优先"排序下排在 make_iron_pickaxe(2.7%)之前,把断裂带锚点挤到**第 7 名**(cap=8 即进入;make_iron_sword 2.6% 在榜内)。

- 判据核心不变量在真数据上成立:**发出的每个目标零断前置;钻石族全部拦下**(测试钉死);
- 但"火力集中在铁镐"依赖 cap/排序的相互作用——短跑读数时看调度日志里 iron 族出现频率;若被 0.0% 长尾稀释,候选调法:`max_target_achievements` 提到 8-10(一行),或排序改"前置强度优先"(min prereq SR 降序)——**先看数据再动**,不预改。

## 4. 语义决策记录(实现时的三个拍板)

1. **S3(近距刷怪)不入闸**:终端技能重复练习正是机制臂 +12 分的来源(probe 文档 §3.1 反问的答案),bare_reverify 也保留刷怪——闸只打忠实性违规,不打练习强度。
2. **"immediate prereq 也 relevant"会收紧一跳**:任务把 enter_sewers 也列进 relevant,则 sewers 自己的直接前置(gnomish mines)也须裸做,floor 2 起步即违规;只列 defeat_lizard 时 floor 2 合法。测试各钉一例。
3. **修复在编码层,docstring 不回改**:闸的罪证走 reflection 重生成代码;docstring 的 Prerequisites 段若与规则冲突,coder 规则块明文优先。代价 = docstring 与代码可能轻微失配;收益 = 忠实性轴以代码为准(审计/裸复验都在代码层)。

已知共享状态注记:`scaffold_rules_block` 与 `current_skill_target` 同款"主线程写、worker 线程读"模式(跨 session 竞态窗口与 [A-2] 原有暴露完全一致,未新增风险面)。

## 5. 短跑命令(队列第 2 项,session 10 ≈ 137M steps)

在 checklist 第 0 节环境检查通过后,+A+B 模板上叠加三个新 flag(scaffold_prompt 默认 auto → prereq 模式自动启用 one-step 文本):

```bash
python experiments/training/run_dicode.py \
  seed=1 \
  use_wandb=true wandb_entity=mechanism_UED wandb_project=Skill_Preflight_UED \
  training.total_timesteps=140000000 \
  gen_manager/llm@gen_manager.task_generator=local_qwen14b \
  gen_manager/llm@gen_manager.env_generator=local_qwen14b \
  gen_manager.embedding_model.model=nomic-embed-text \
  gen_manager.task_generator.max_tokens=8192 \
  gen_manager.env_generator.max_tokens=8192 \
  +skill_preflight.use_scheduler=true \
  +skill_preflight.use_preflight=true \
  +skill_preflight.mastery_threshold=0.2 \
  +skill_preflight.frontier_mode=prereq \
  +skill_preflight.prereq_threshold=0.3 \
  +skill_preflight.use_scaffold_gate=true
```

(140M = 13.7M/session × 10 折算;实跑 9 sessions 打满善终。注意:**不要加 `+validation=default`** —— conf/config.yaml defaults 已内置 validation,重复声明 hydra 启动即死,2026-07-12 首次点火实测。tmux 包装/日志路径照抄 checklist。)

三判据读数位置:
1. **泄漏语义**:跑完用 `scaffold_audit.py` 复扫新 task_graph → "预标已掌握技能"比例应显著 ↓(闸+prompt 双重作用);
2. **tier-3 裸指标**:官方协议 eval 对比 probe 同期(session 10 ≈ update 1200 档)iron/diamond 族 held-out SR;
3. **拒绝率**:日志 grep `[Preflight] kept` 与 `[ScaffoldGate] checked ... dropped` —— preflight 拒绝率 <30%,且 ScaffoldGate 的 repaired/dropped 比给出 14B 对 one-step prompt 的服从率首个实测(dropped 高 → 触发队列第 5 项 ★C-1 接入条件)。

## 6. 单测复跑(pod)

```bash
cd /workspace/mechanism_UED/dicode_src
uv run pytest src/dicode/skill_preflight/tests/ -v   # 期望 51 passed
```
