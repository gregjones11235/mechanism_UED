# BA-BAGR-UED v2 fix1 — 审计 BLOCKER 与硬化修复摘要

基线提交：`1efcceb9660a5c4f33edbf2175983ce5bc22b7f0`
目标分支：`henry/ba-bagr-ued-review-board-v2`（专用干净 worktree，快进推送）
运行类别：**ENGINEERING_FIX_VALIDATION**（real_llm_calls=0，
real_environment_rollouts=0，training_started=false，
formal_evaluation_started=false，formal_banks_used=false）

## 1. BLOCKER — UnsafeRestNearHostileDetector（§1/§2）

检测器升级到 v2，finding 合同改为**基于条件**：

> finding 存在 ⇔ rest_or_sleep=true ∧ hostile_nearby=true ∧
> environment_confirmed_safe=false

后续 damage_taken / chased / died **只能**：提高 severity（base 0.5 →
可审计 HARM_SEVERITY_LEVELS 最大值，复用 v1 数值）、提高 confidence
（base 0.6 + 0.2 加成）、补充 supporting_events、置 realized_harm=true。
**绝不**决定 finding 是否存在。输出明确区分
`unsafe_condition_observed=true` 与 `realized_harm=true/false`。
无新增写死科学数值（BASE_SEVERITY / BASE_CONFIDENCE /
HARM_CONFIDENCE_BONUS 均为类上可审计常量）。
回归测试 A–F 全部通过（test_bagr_ued_unsafe_rest_contract.py，8 项）。

## 2. PREREQUISITE — Controller shortfall 发射门（§3/§4）

最终批次/发射决策处增加硬门 `_launch_gate`：BATCH_PLAN_READY 与
TRAINING_LAUNCH_AUTHORIZED 同时要求 budget_plan.status=OK、
selected_ued_slots=12、canonical_anchor_slots=4（固定全局锚点）、
total_envs=16、rollout_length=128、transitions_per_update=2048、
所有入选 descriptor 合法、无未解决 shortfall、无未解决 guard 违规。
任一失败 → 两标志均 false + 结构化 launch_block_reasons + shortfall 计数。
禁止复制候选凑 12、禁止缩减槽位/锚点/k/批次/transitions、禁止静默继续；
诊断 dry-run 仍可运行，未就绪时证书 run_class 改标 BLOCKED_DRY_RUN。
归档 `commit` 增加 ACTIVE_ARCHIVE_COMMIT_BLOCKED 门（门禁 false 即
fail-closed）。测试 A–F + e2e 全部通过（test_bagr_ued_launch_gate.py，8 项）。

## 3. HARDENING — Guard 硬化（§5/§6/§7）

- **§5 序列化字符串解析**：trim 后形如 JSON object/array（外加 JSON 字符串
  字面量，堵双重编码绕过）的字符串在
  MAX_SERIALIZED_PARSE_DEPTH=12 /
  MAX_SERIALIZED_STRING_LENGTH=65536 /
  MAX_SERIALIZED_CONTAINER_ITEMS=4096
  限制下解析并对解析结果**重跑完整 guard**；解析失败 → 继续走纯文本 NL
  模式（绝非宽松跳过）；超限 → fail-closed
  SERIALIZED_GUARD_LIMIT_EXCEEDED。
- **§6 别名**：14 个别名
  （suggested_action(s) / recommended_action(s) / recommended_move /
  recommended_policy / route / navigation_route / path_to_follow /
  expert_plan / bank_blob / formal_state_blob / formal_state_payload /
  state_payload）并入归一化键表（保留 casefold + 分隔符剥离 + 嵌套
  mapping/sequence 扫描）；四个 payload 别名同步镜像进 Guard B。
- **§7 中英动作建议**：新增 go/head/move-toward 方向命令式与
  攻击<目标>/往<方向>走/向<方向>走/朝<目标>移动 等模式；规范要求的 12 条
  建议文本全部 REJECT，行为描述对照句（"The student repeatedly attacks
  without effect." / "智能体重复攻击但没有效果。" 等）全部放行，无误杀。
- 测试：test_bagr_ued_guard_hardening.py（46 项）全部通过。

## 4. HARDENING — alpha_front 结构性 <1（§8）

schema `alpha_front: Field(ge=0.0, lt=1.0)`（原 le）；Controller 运行时
三断言：0≤alpha_front<1、0≤alpha_min≤alpha_max<1、1−alpha_front>0；
新增可审计常量 ALPHA_FRONT_MIN=0.0 / ALPHA_FRONT_MAX=0.75。
global 分量 (1−alpha_front)×weight 在任何 schema 合法 alpha 下恒严格为正。
测试：test_bagr_ued_alpha_front.py（13 项）全部通过。

## 5. 测试与回归（§10）

- bagr_ued 全套：**143/143 通过**（61 旧 + 82 新，0 失败）。
- d052 全量（fix worktree）：449 通过，6 失败。
- 6 个失败 = test_real_bundle_reconciliation.py 环境性失败；在父提交
  worktree（@1efcceb）同测试对照运行：**失败集合逐条相同**
  （6 项，根因 outputs/ 数据目录
  不在 worktree，agg_source_sha_matches_bundle=False）→ 非回归。
- **NEW_REGRESSIONS_INTRODUCED = 0**；无 skip/xfail/删除隐藏失败。

## 6. 科学边界保持（§9）

六角色固定顺序、RoleEnvelope provenance、竞争性因果假设合同、Tutor
环境级边界、Explorer 差异要求、Critic 拒绝/选择分离、Reconciler 规则、
front/global regret 分离、behavioral_gap 定义、Soft Copeland 成对语义、
12 UED + 4 锚点、2048 transitions/update、8192/review、formal 泄漏源策略、
两条 REAL_CANONICAL critic 规则保持 PENDING — 全部未改动。
无真实 LLM、无真实 TaskParams 接入。
