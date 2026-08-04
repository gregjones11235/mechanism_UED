# 测试报告（test_report）— feedback_llm_ued（六角色 + 双窗口状态机版）

全部命令在 worktree
`C:/Users/Lenovo/Desktop/dicode-codex-director/mechanism_UED_bagr_ued_fix1_worktree/gpu1_aggregation_siege`
下执行，Python 为 `/d/Anaconda/python`（3.12.4，无 jax/flax/orbax），
`PYTHONPATH=.`。结果为真实运行输出，未修饰。

## 1. 测试命令

```bash
cd /c/Users/Lenovo/Desktop/dicode-codex-director/mechanism_UED_bagr_ued_fix1_worktree/gpu1_aggregation_siege

# 方向二全部测试（17 个文件，含 C9 门禁定向测试）
PYTHONPATH=. /d/Anaconda/python -m pytest d052/tests/test_feedback_llm_ued_*.py -q

# d052 全量套件
PYTHONPATH=. /d/Anaconda/python -m pytest d052/tests -q
```

## 2. 方向二测试结果：388 passed / 0 failed

```
388 passed（CC4 C9 门禁第二轮后：既有 370 + 门禁文件 18
= 旁路/滞后 10 + 再识别负测 3 + 全 prompt 字节奇偶 3
+ static 独立性/结构负测 2）
```

| 文件 | 用例数 | 覆盖内容（摘要） |
|---|---|---|
| `test_feedback_llm_ued_controller.py` | 58 | 授权姿态（全部 REAL_* 旗标 False、C16 工程旗标与 CC4 门禁后复 True 的两个 C9 旗标、任一 never-true 旗标为 True 即拒绝构造）；双窗口状态机：同窗 apply verdict/改计划→SAME_WINDOW_REVISION_FORBIDDEN、STALE/UNKNOWN/DUPLICATE_FEEDBACK_ID（旧/当前/未来引用均 fail-closed）、P0-6 绑定守卫、phase 迁移；三模式端到端（7 调用/窗、61440 transitions/探针窗、revision **恰好**滞后反馈一窗）；C10 cooldown/重开（CC3 重定基线）；C11 REQUEST_CONTROL 停环 + artifact + final_batch final=False；确定性逐字节；shuffled 决策分布按 CC4 数值侧信道加固重定基线 |
| `test_feedback_llm_ued_review_board.py` | 32 | 六角色固定顺序与每窗完整 6 次调用；verdict 显式引用 feedback_id/hypothesis_id/prediction_signature；新假设 PENDING+预测签名；AxisDirective 合法性；Critic 升级规则（HIGH_SEVERITY_FLOOR/OPPOSITE_MATCH_FLOOR/WIDE_CI）与 endorsed；REQUEST_CONTROL 两触发路径；错误窗口视图/引用→STALE_FEEDBACK_ID（CC3 门禁） |
| `test_feedback_llm_ued_c9_gate.py` | 18 | **C9 门禁定向测试（CC3 旁路/滞后 10 + CC4 再识别/字节奇偶 6 + CC4 static 独立性 2）**：static 满 store 下 board 上下文结构性空（证据/SR/CI/候选 id/历史全空，序列化扫描无真实 id）；shuffled 上下文+证据层仅匿名 id（证据与 payload 匿名 id 逐位一致、candidate id 掩码、置换恰好覆盖诚实记录集、resolve_citation 唯一还原路径）；混合窗口视图构造 fail-closed；旧/当前/未来引用→STALE_FEEDBACK_ID、恰好 k−1 通过；build_board_prompt_context 双重窗口校验；三模式端到端逐引用恰好滞后一窗。**CC4 新增**：两层数值全部等于公开可复算的族级窗口聚合（payload 与证据层一致、expected_signature 置空）；store 联接对手不得把任何条目收窄到单例（唯一性负测，含数值联接）；序列化上下文全 float 扫描无任何精确逐记录指标；两次独立运行全 prompt 上下文（载荷+BoardContext+假设，canonical JSON）逐字节一致（shuffled 与 normal 各一）+ 同运行内重组逐字节一致；**static 独立性负测**（director 点名的静态泄漏）：仅反馈记录不同的两个 store（运行前注入外来 fb-junk 记录）下，static 全 prompt 上下文与全部六个 board prompt 逐字节一致，且 `_retirement_state` 对 static 结构性 fail-closed（STATIC_MODE_HAS_NO_RETIREMENT_LIFECYCLE） |
| `test_feedback_llm_ued_persistence.py` | 31 | C15：冻结点快照→恢复→续跑与不间断运行 summary 逐字节一致（normal+shuffled）；fresh subprocess 恢复哈希一致；篡改矩阵（顶层字段+重签名深篡改）→HASH_CHAIN_BROKEN；逐假设 revision 链校验单元负测；原子写无 .tmp 残留；停环恢复保持停止且零新增调用 |
| `test_feedback_llm_ued_selection_anchors.py` | 26 | C12+C13：八准则 clamp 与 schema 合法性；共享 soft_copeland_rank 等价（ranking_hash 逐字节一致，不分叉）；constant 维度 provenance；族多样性贪心（penalty 0 vs 0.5 对照）；AnchorManifestSource（缺失/未冻结/哈希不一致 fail-closed）；controller 锚位绑定（占位标签、冻结 manifest 绑定、三模式预算相等） |
| `test_feedback_llm_ued_compare_gate_reconcile.py` | 28 | 比较器阈值与 MAJORITY；reconciler 预算/探索预留/悬空引用/伪装禁止/RETIRE_REQUIRES_FEEDBACK/REQUEST_CONTROL 零预算 |
| `test_feedback_llm_ued_probe.py` | 21 | 漏斗形状 64→64→24→12+4=16；61440 transitions 精确核算；stage-2 Copeland 审计字段（score/copeland_rank/criteria/selected）；锚位重复/数量非法 fail-closed |
| `test_feedback_llm_ued_hash_recompute.py` | 20 | C14：七类对象空哈希计算、携带哈希逐字比较（mismatch→CONTENT_HASH_MISMATCH）、自 dump 往返、构造后 tamper→rehash 暴露、Envelope 三哈希与 replay key 一致 |
| `test_feedback_llm_ued_axis_directive.py` | 19 | 轴/层级/方向/角色合法性、control 必须 hold、treatment 方向-层级一致、held axes 约束、批次唯一性与每窗每轴至多一个 treatment |
| `test_feedback_llm_ued_backends.py` | 19 | Mock/Replay/Real 后端：usage 计数、未授权构造 Blocked、录制-回放等价、assert_no_real_llm_usage |
| `test_feedback_llm_ued_view_isolation.py` | 18 | C9：static board 上下文零反馈载荷（结构级）；shuffled 冻结置换可复算、匿名化 id、身份侧信道不可还原（CC4：数值只发布族级窗口聚合、family-grain 预测签名移除、prompt 层与证据层一致）；三模式角色/EnvCoder/seed/预算一致 |
| `test_feedback_llm_ued_evidence.py` | 19 | 行为失败证据提取（return shortfall/早停/行为激活缺口/reference gap）+ 确定性 CI 半宽；BoardContext 只经视图装配（原始 store→BOARD_CONTEXT_STORE_FORBIDDEN、视图窗口≠证据窗→BOARD_CONTEXT_WINDOW_MISMATCH，CC3 门禁） |
| `test_feedback_llm_ued_ledger_store.py` | 18 | 账本生命周期与哈希链衔接；store 白名单/formal 源拒绝/bind_match 重哈希 |
| `test_feedback_llm_ued_real_probe.py` | 16 | ExecutableCandidate 哈希重算、RealTaskParamsAdapter、FakeStepEnv fake-real、未授权 Blocked |
| `test_feedback_llm_ued_contracts.py` | 16 | 候选哈希稳定性、计划签名顺序无关、诚实性错误码、上下文块往返 |
| `test_feedback_llm_ued_student_binding.py` | 15 | 固定身份 PERSISTENT_RMT16_ORIGINAL_VTRACE_98304、CC4 缺失 fail-closed、训练接缝 no-op 记账、诚实姿态 |
| `test_feedback_llm_ued_env_coder.py` | 14 | SpecEnvCoder 确定性、compile/reset/step 三级门禁 fail-closed、LLM 接缝 Blocked |

## 3. d052 全量套件：942 passed / 6 failed（均为既有环境性失败）

```
6 failed, 942 passed, 2 warnings in 10.42s

FAILED d052/tests/test_real_bundle_reconciliation.py::test_bundle_integrity_13_of_13
FAILED d052/tests/test_real_bundle_reconciliation.py::test_r4_replay_reproduces_all_historical_anchors
FAILED d052/tests/test_real_bundle_reconciliation.py::test_frozen_labels - As...
FAILED d052/tests/test_real_bundle_reconciliation.py::test_ccv2_replay_overlap_jaccard_and_anchors_unchanged
FAILED d052/tests/test_real_bundle_reconciliation.py::test_frozen_output_allowlist_matches_git
FAILED d052/tests/test_real_bundle_reconciliation.py::test_historical_replay_unchanged
```

对照说明（诚实性）：

- 本方向开工前基线为 554 passed + 同样 6 个环境性失败；C1–C16 过程中基线
  演进 554→749→889→920，CC3 C9 门禁后 920→934，CC4 C9 门禁第二轮后
  934→942（新增再识别负测、字节奇偶测试与 static 独立性负测），**6 个失败
  名单全程未变**。
- 6 个失败全部位于 `test_real_bundle_reconciliation.py`，依赖本 worktree
  不具备的历史 real-bundle 数据，与本方向无关。
- 无新增失败、无跳过掩盖；CC4 门禁两个 C9 旗标翻转（Commit C 置 False →
  数值侧信道加固 + static phase-A 存储读取修复 + 再识别负测/字节奇偶/static
  独立性测试全绿后 Commit D 复 True）后全量复跑确认无回归。

## 4. Smoke（三模式闭环 6 窗，新架构真实输出）

命令（真实执行）：

```bash
PYTHONPATH=. /d/Anaconda/python -c "
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
sums = {}
for mode in (C.MODE_STATIC_LLM, C.MODE_NORMAL_FEEDBACK, C.MODE_SHUFFLED_FEEDBACK):
    sums[mode] = FeedbackUEDController(mode).run(max_windows=6)
print('SMOKE OK: modes=3 windows=6')
for m, s in sums.items():
    print(f'  {m}: n_windows={s.n_windows} llm_calls={s.n_llm_calls} revision_rate={s.revision_rate} decisions={s.decision_distribution} citation_cov={round(s.feedback_citation_coverage,4)} supp_retain={s.supported_retention_rate} ref_retire={s.refuted_retirement_rate} transitions={s.total_simulator_transitions}')
print('  comparison:', FeedbackUEDController.compare_summaries(
    sums[C.MODE_NORMAL_FEEDBACK], sums[C.MODE_SHUFFLED_FEEDBACK], sums[C.MODE_STATIC_LLM]))
"
```

真实输出（CC4 C9 门禁第二轮后重定基线，恰好 k−1 滞后 + 数值侧信道加固）：

```
SMOKE OK: modes=3 windows=6
  static_llm: n_windows=6 llm_calls=42 revision_rate=1.0 decisions={'MUTATE': 6} citation_cov=0.0 supp_retain=0.0 ref_retire=0.0 transitions=368640
  normal_feedback: n_windows=6 llm_calls=42 revision_rate=1.0 decisions={'MUTATE': 7, 'RETIRE': 4, 'RETAIN': 3} citation_cov=0.8047 supp_retain=1.0 ref_retire=1.0 transitions=368640
  shuffled_feedback: n_windows=6 llm_calls=42 revision_rate=1.0 decisions={'MUTATE': 9, 'RETIRE': 3, 'RETAIN': 1} citation_cov=0.8047 supp_retain=1.0 ref_retire=1.0 transitions=368640
  comparison: plan_difference_windows=4 plan_identical_windows=2
    feedback_binding_matters=True static_plan_difference_vs_normal=5
```

口径说明：新架构下 static 基线也执行完整六角色+EnvCoder（compute-matched，
每窗 7 次 LLM 族调用、6 窗共 42 次），与旧版 0 调用基线不同；其反馈视图是
结构性空的 NullFeedbackView，故 citation_cov=0、无 verdict 驱动的
retain/retire，计划修订全部为 EXPLORATION。CC3 门禁重定基线说明：恰好 k−1
语义下每窗 board 只看到上一窗 64 条记录，REFUTED 判定下一窗即携新证据触发
退休，故 RETIRE 数上升（normal 2→4、shuffled 2→3）、ref_retire 升至 1.0。
CC4 门禁重定基线说明：shuffled 视图移除数值侧信道后（精确逐候选率/缺口是
候选哈希的确定性指纹，改为只发布族级窗口聚合，prompt 层与证据层一致），
mock 角色读到的是更粗的数值，shuffled 决策分布由 {MUTATE:7, RETIRE:3,
RETAIN:3} 变为 {MUTATE:9, RETIRE:3, RETAIN:1}；normal 与 static 未受影响
（normal 视图本就是诚实全精度）。normal 与 shuffled 退休族集合仍不同
（feedback_binding_matters=True 保持）。CC4 static 泄漏修复说明：phase A
不再为 static 调用读取 store 的退休生命周期查询（改用冻结空生命周期 +
查询本身 fail-closed），该修复只改变结构性保证、不改变任何数值——上方
smoke 在修复落地后原样重跑，输出逐字节一致。

## 5. 结论

- 388 个方向二测试全绿；全量套件 942 通过、基线失败名单不变。
- 双窗口时序（恰好 k−1）、对照隔离（BoardContext 只经视图）、哈希重算、
  持久化/跨窗恢复等价（含 fresh-process）均有专门正负测试锁定；两次独立
  运行 summary 逐字节一致（确定性），两次独立运行**全 prompt 上下文**亦
  逐字节一致（CC4 字节奇偶测试）。
- C9 门禁定向测试（`test_feedback_llm_ued_c9_gate.py`，18 用例）锁定：
  两类旁路（static 满 store 零载荷、shuffled 证据层无身份侧信道）；四层
  滞后防线（视图构造、视图选择、两级引用校验、端到端逐引用断言）；
  **CC4 再识别负测**（两层数值=公开族级聚合、store 联接不可收窄到单例、
  序列化上下文无精确逐记录指标）；**全 prompt 字节奇偶**；**static 独立性
  负测**（外来 store 污染下 static 上下文与六个 board prompt 逐字节一致 +
  退休查询对 static fail-closed）。
- 闭环数值（引用覆盖 0.8047、normal≠shuffled、feedback_binding_matters=True）
  由端到端测试与 smoke 双重锁定。
