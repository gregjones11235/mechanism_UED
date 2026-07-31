# BA-BAGR-UED V2 — CC3 fix2 修复报告（工程验证，非科研结论）

基线：`b781f38fee2dc33ef429690962e6a98d89f27905`（fix1 提交），专用分支
`henry/ba-bagr-ued-review-board-v2`。本轮只修静态审计发现的五个问题，
不引入新科学主张、不跑真实实验。

## 五个修复

### P0-1 Soft Copeland 逐准则化（§1-§3，soft_copeland.py 全重写）
- 8 个 RAW 维度**独立** min-max 归一化，归一化**之前不乘 alpha**（v1 缺陷：
  alpha 乘在原始值上被归一化抵消，排名对 alpha 不敏感）；
- 逐准则软偏好 `p_k(i>j) = sigmoid(sign_k·(s_i_k − s_j_k)/T_k)`（数值稳定实现），
  critic_penalty 为 lower-is-better（sign=−1）；
- 维度权重作用于**成对比较层**：front_weight=alpha_front，
  global_weight=1−alpha_front（schema 拒绝 alpha=1，恒正），其余来自版本化权重表；
  **不存在预聚合 strength**；
- `copeland_i = mean_{j≠i} P(i>j)`（等价于 0.5 + 平均 margin，两者均记录）；
- 完整审计链：归一化 provenance（min/max/constant/normalized）、权重、温度、
  n×n 偏好矩阵与 margin 矩阵、tie-break 规则，全部绑定 ranking_hash。
- §2 alpha 有效性实测：A front-strong / B global-strong，alpha=0.75 → A 胜
  （P(A>B)=0.519918843 > 0.5），alpha=0.25 → B 胜
  （P(A>B)=0.480081157 < 0.5）；归一化 provenance 与
  alpha 无关而排名随 alpha 翻转。ALPHA_SENSITIVITY_TEST=PASS。
- §3 边界 A–J 全部 fail-closed（成对冲突 / alpha 翻转 / 常数列中性 / 并列稳定 /
  排列不变 / NaN·Inf / 重复 ID / 缺失准则 / 非法高分不可选 / critic penalty 非合法性门）。

### P0-2 强类型不可绕过 LaunchGate（§4-§8，launch_gate.py 新增 + archive.py 重写）
- `@dataclass(frozen=True)` LaunchGate：structural_batch_ready /
  director_training_authorized / final_training_launch_authorized（= 前两者 AND）
  + batch_plan_hash / selected_descriptor_hash / guard_report_hash /
  legality_report_hash + reasons + gate_version；构造期即校验
  final == structural AND director、版本匹配。
- **禁用**同名双义字段 `training_launch_authorized`（实测 gate 与证书中均不存在）。
- director_training_authorized 本轮恒 false → final 恒 false。
- structural_batch_ready 要求全部：status OK / 12 UED 槽 / 4 锚点 / 16 环境 /
  128 rollout / 2048 transitions / 选中全合法 / ID 唯一 / 与 batch_plan_hash 一致 /
  无 shortfall / 无 guard 违规 / 选中提案无 critic 硬拒 / 全部 provenance 哈希。
- archive.commit **必须**携带 gate（关键字必填、isinstance 校验、版本校验），
  并对四个哈希与当前状态逐项重算比对 → 不符即 ARCHIVE_COMMIT_REJECTED；
  None / dict 门 / 缺参全部 fail-closed；refresh(dry_run=False) 无门即
  REFRESH_GATE_REQUIRED；永不内部重建默认 PASS 门。
- §8 A–J 实测全部通过。

### P1-1 符号行为片段（§9-§14，symbolic_behavior_clip.py 新增）
- SymbolicBehaviorStep（step_offset / action_semantic_classes /
  hostile_distance_band / safety_status / health_delta_band /
  resource_delta_bands / progress_delta_band / event_semantics /
  terminal_category）+ SymbolicBehaviorClipPayload（clip_id / episode_id /
  source / start·end_step / steps / 6 个 provenance 哈希 /
  clip_payload_sha256 / truncation_applied / schema_version）。
- 数据边界：只允许带宽/语义类/事件；原始动作整数、原始状态、观测、轨迹、
  logits、建议全部拒绝；双 guard + 来源白名单 + 哈希重算 + 限制校验。
- 有界限制（MAX_CLIP_STEPS=24 等 6 项）：截断且
  truncation_applied=true，或 fail-closed；4096 步完整轨迹永不入板。
- 六角色上下文携带逐步骤符号片段；证书
  BEHAVIOR_REVIEW_HAS_SYMBOLIC_CLIPS=true / RAW_ACTION_INTEGER_EXPOSED=false /
  RAW_STATE_EXPOSED=false / FORMAL_TRAJECTORY_EXPOSED=false。
- 暂定性超分类假设（provisional=true / requires_deterministic_validation=true）
  仅展示，禁止进入 selector/budget/archive（证书 id 不相交实测=0）。
- §14 A–H 实测全部通过（含原始动作整数/原始状态/FORMAL_FRONT/超限/哈希伪造/
  provenance 不符 fail-closed）。

### P1-2 合法性语义（§15-§16）
- 原始提案**可以**含非法候选；LegalityGate 拒绝并记录；只有合法进入评分/选择；
  最终门**只查选中侧**（选中全合法 / 选中∩拒绝=∅ / 合法数≥12）；
  未选中非法只记录不阻断合法批次。实测：12 合法 + 1 未选中非法 →
  structural_batch_ready=true；11 合法 → shortfall 阻断；selector 引用被拒候选 →
  fail；选中无合法性证据 → fail。A–E 全部通过。

### P1-3 自然语言 guard 误报（§17-§18）
- 结构化键保持最严；祈使句/第二人称/建议语境检测取代裸关键词：
  10 条 MUST REJECT（中英祈使/应该/建议）全部拒绝，8 条 MUST ALLOW
  （中英过去式描述/检测器发现/反事实环境描述）全部放行；歧义中文框架
  fail-closed。§17 全表 + §18 ≥10 类别实测通过。

## 契约驱动的测试更新（透明记录，非弱化）
- test_bagr_ued_scoring.py：2 个测试改为逐准则契约（8 RAW 维度 component 键 +
  copeland_score 比较，strength 断言移除）；
- test_bagr_ued_alpha_front.py：3 个 layer-3 结构证明测试改为成对层权重
  （global 权重 = 1−alpha > 0、常数列归一化 0.5）；
- test_bagr_ued_launch_gate.py：整体重写为强类型 LaunchGate（dict 门 API 已删除）。
无删除测试、无断言弱化、无 skip/xfail。

## 测试结果（实测）
- bagr_ued 套件：**223 passed**（fix1 基线 143，+80），0 warnings；
- 全 d052（除 real_bundle）：**506 passed**，0 failed
  （2 个 pre-existing UserWarning 来自 d052/legacy/protocol_version.py，基线即存在）；
- test_real_bundle_reconciliation.py：6 failed / 23 passed —
  与基线 b781f38（= 1efcceb）失败集合**完全一致**（6 个环境性失败：缺少
  orchestration/experiments/d052_modeler_shadow_v1/outputs/ + agg_source_sha
  不匹配；fix2 未触碰任何 orchestration/数据路径）；
- **NEW_REGRESSIONS_INTRODUCED = 0**。

## 证书（本轮性质）
run_class=ENGINEERING_FIX_VALIDATION；real_llm_calls=0；
real_environment_rollouts=0；training_started=false；
formal_evaluation_started=false；formal_banks_used=false；
checkpoints_modified=false；real_integration_authorized_by_CC3=false；
independent_reaudit_required=true。

## 不可变契约保持（§19）
GLOBAL_UED_CONTROLLER / 六角色顺序 / 12+4 / 16×128=2048 / 8192 /
两条 REAL_CANONICAL critic 规则 PENDING / alpha_front 结构 <1 — 全部保持。
