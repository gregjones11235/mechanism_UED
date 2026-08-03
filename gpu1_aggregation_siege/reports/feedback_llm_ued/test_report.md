# 测试报告（test_report）— feedback_llm_ued

全部命令在 worktree
`C:/Users/Lenovo/Desktop/dicode-codex-director/mechanism_UED_bagr_ued_fix1_worktree/gpu1_aggregation_siege`
下执行，Python 为 `/d/Anaconda/python`（3.12.4）。结果为真实运行输出，未修饰。

## 1. 测试命令

```bash
cd /c/Users/Lenovo/Desktop/dicode-codex-director/mechanism_UED_bagr_ued_fix1_worktree/gpu1_aggregation_siege

# 新方向测试（6 个文件）
/d/Anaconda/python -m pytest \
  d052/tests/test_feedback_llm_ued_contracts.py \
  d052/tests/test_feedback_llm_ued_ledger_store.py \
  d052/tests/test_feedback_llm_ued_compare_gate_reconcile.py \
  d052/tests/test_feedback_llm_ued_roles.py \
  d052/tests/test_feedback_llm_ued_probe.py \
  d052/tests/test_feedback_llm_ued_controller.py -q

# d052 全量套件
/d/Anaconda/python -m pytest d052/tests -q
```

## 2. 新方向测试结果：109 passed / 0 failed

```
109 passed in 0.60s
```

| 文件 | 用例数 | 覆盖内容 |
|---|---|---|
| `test_feedback_llm_ued_contracts.py` | 16 | 候选哈希稳定性、族/轴合法性、真实适配器状态诚实、ProbeMetrics 正式源拒绝、计划签名与顺序无关、PlanModification/PlanRevisionRecord 全部诚实性错误码（EXPLORATION_LABEL_REQUIRED / EXPLORATION_DECISION_ONLY / MASQUERADE_FORBIDDEN / REVISION_LABEL_FORCED / FEEDBACK_ID_MISMATCH / UNKNOWN_FEEDBACK_ID）、prompt 上下文块往返 |
| `test_feedback_llm_ued_ledger_store.py` | 18 | 假设生命周期（PENDING→各 verdict）、修订历史哈希链（previous_record_hash 逐条衔接）、bind_feedback 重哈希、非法状态/置信度拒绝；store 重复 id、REFERENCE_FIELD_FORBIDDEN、REFERENCE_CARRIER_FORBIDDEN、FORMAL_SOURCE_FORBIDDEN（构造时即 raise）、候选哈希 sha256 校验、bind_match 重哈希与 graded 过滤 |
| `test_feedback_llm_ued_compare_gate_reconcile.py` | 27 | 比较器 agree/neutral/opposite 阈值、counter 语义、MAJORITY 总体判定、无重叠→neutral、NO_PROBE_METRICS；gate 8 条 must-invoke 条件逐条独立触发 + 阈值严格性 + 顺序确定性；reconciler 预算恰好 12、探索预留不被饿死、探索上限、悬空引用、RETIRE_REQUIRES_FEEDBACK、伪装禁止、retire-overrides-active、REQUEST_CONTROL 零预算、重复去重、空提案 fail-closed、确定性 plan_id |
| `test_feedback_llm_ued_roles.py` | 17 | Diagnostician verdict 由反馈绑定派生（SUPPORTED/REFUTED/INCONCLUSIVE/STALE + 置信度漂移 + 风险升级 HIGH/MEDIUM/LOW）；Designer verdict→决策映射、STALE→诚实探索、高风险→REQUEST_CONTROL、输出诚实性校验；Reviewer 7 条触发器逐条 + 过度自信关注 + 族冲突强制退休 + 探索越界；mock 后端 real_calls=0、未知角色拒绝、三角色全部分派、envelope 哈希绑定与确定性 |
| `test_feedback_llm_ued_probe.py` | 15 | 静态合法性 3 分支、符号探针确定性与 Reference 载荷干净、非法 stage/episode 预算全部 raise、真实 Craftax 接缝 ProbeRunnerBlocked（BLOCKED_NO_LOCAL_CRAFTAX）、漏斗形状 64→64→24→24→12+4=16、转移数 61440 精确核算、跨 runner 确定性、raw 上限、哈希去重、静态拒绝记录、生成器 64 候选合法性与假设绑定 |
| `test_feedback_llm_ued_controller.py` | 16 | 端到端三模式（6 窗/模式）：授权旗标全 False + 任一为 True 即拒绝构造、未知模式拒绝；static 基线 0 次 LLM/计划恒定/仅窗口 0 探针；normal 15 次调用、revision_rate=0.8333、supported_retention=1.0、refuted_retirement=1.0、引用覆盖>0.5、账本被绑定反馈驱动、所有引用均存在于 store、envelope 哈希绑定、模拟成本 368640；shuffled 绑定标记、计划与决策分布偏离 normal、feedback_binding_matters=True；两次运行逐字节确定性 |

## 3. d052 全量套件：663 passed / 6 failed（均为既有环境性失败）

```
6 failed, 663 passed, 2 warnings in 5.45s

FAILED d052/tests/test_real_bundle_reconciliation.py::test_bundle_integrity_13_of_13
FAILED d052/tests/test_real_bundle_reconciliation.py::test_r4_replay_reproduces_all_historical_anchors
FAILED d052/tests/test_real_bundle_reconciliation.py::test_frozen_labels - As...
FAILED d052/tests/test_real_bundle_reconciliation.py::test_ccv2_replay_overlap_jaccard_and_anchors_unchanged
FAILED d052/tests/test_real_bundle_reconciliation.py::test_frozen_output_allowlist_matches_git
FAILED d052/tests/test_real_bundle_reconciliation.py::test_historical_replay_unchanged
```

对照说明（诚实性）：

- 本方向改动前的基线为 **554 passed + 同样 6 个环境性失败**（任务 #86 记录）。
- 现在 663 = 554 + 109（本方向新增），**6 个失败与本方向无关**：全部位于
  `test_real_bundle_reconciliation.py`，依赖本 worktree 不具备的历史 real-bundle
  数据，在改动前即失败，失败名单完全一致。
- 无任何新增失败、无跳过掩盖。

## 4. Smoke（三模式闭环 6 窗）

命令（真实执行，输出见 `final_implementation_report.md` §5）：

```bash
/d/Anaconda/python -c "
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.controller import FeedbackUEDController
sums = {}
for mode in (C.MODE_STATIC_LLM, C.MODE_NORMAL_FEEDBACK, C.MODE_SHUFFLED_FEEDBACK):
    sums[mode] = FeedbackUEDController(mode).run(max_windows=6)
cmp_ = FeedbackUEDController.compare_summaries(sums[C.MODE_NORMAL_FEEDBACK], sums[C.MODE_SHUFFLED_FEEDBACK], sums[C.MODE_STATIC_LLM])
print('SMOKE OK: modes=3 windows=6')
for m, s in sums.items():
    print(f'  {m}: llm_calls={s.n_llm_calls} revision_rate={s.revision_rate} supp_retain={s.supported_retention_rate} ref_retire={s.refuted_retirement_rate} transitions={s.total_simulator_transitions}')
print('  comparison: plan_difference_windows=%s feedback_binding_matters=%s static_llm_calls=%s' % (cmp_['plan_difference_windows'], cmp_['feedback_binding_matters'], cmp_['static_llm_calls']))
"
```

## 5. 结论

- 109 个新测试全绿；全量套件无回归（基线失败名单不变）。
- 闭环数值（revision_rate、retain/retire 分布、引用覆盖、normal≠shuffled）
  由端到端测试与 smoke 双重锁定，且两次独立运行逐字节一致。
