# E1 十一标志语义与本轮取值

> 总控澄清的两条语义红线记录在案：
> `E1_FORMAL_PLAN_ALIGNED` **仅工程计划对齐，不等于真实闭环**；
> `E1S_STATIC_ABLATION_PRESERVED` **仅工件保留，不等于 E1-S 可运行**。
> 三个 `REAL_*` 标志在本实现中**默认 False 且无置 true 的代码路径**
> （`tests/e1_formal/test_flag_manifest.py` + grep 审计）。

| # | 标志 | 为 true 当且仅当 | 本轮值 | 证据指针 |
|---|---|---|---|---|
| 1 | E1_FORMAL_PLAN_ALIGNED | 九阶段皆有已提交代码/配置对应且测试绿；**不表示真实闭环** | **true** | 提交链 `7c6bc88`…`1fa41ab`+C12；`tests/e1_formal` 全绿；九阶段映射见 `e1_formal_plan.md` §一 |
| 2 | E1S_STATIC_ABLATION_PRESERVED | static_llm 工件未删未覆盖、被复用、降级成文；**不表示 E1-S 可运行** | **true** | `src/dicode/teachers/static_llm/{guards.py,schemas.py,student_init_contract.py}` 原样存在；`e1s_demotion.md`；tests/static_llm 全绿 |
| 3 | SIX_ROLE_BOARD_IMPLEMENTED | 每执行窗恰 6 命名角色（固定顺序）、fail-closed、无 2 角色/条件路径、测试绿 | **true** | `teachers/e1_formal/board.py`；`INCOMPLETE_REVIEW_WINDOW`→REUSE；`test_board.py` |
| 4 | REAL_ENVCODER_USED | EnvCoder 由真实 LLM provider 应答 | **false** | replay-only（`llm_client.py`：miss⇒HARD FAIL，record 本轮禁用）；`status_report` 断言 `test_integration_smoke.py::test_all_real_flags_false` |
| 5 | REAL_STUDENT_REFERENCE_EVAL | 经 CC4 adapter 的真实 Student+Reference rollout | **false** | G1 契约身份值未冻结 + 无 adapter；seam 阻断码测试 `test_candidate_evaluation_seam.py` |
| 6 | REAL_TRAINING_UPDATE_EXECUTED | 真实 PPO update 跑在 12+4 batch | **false** | 本轮离线，未训练；replay store 为空⇒开窗 HARD FAIL by design |
| 7 | REFERENCE_CONTRACT_READY | G1 契约 schema+fail-closed+配置 seam 实现且测试绿，无任何猜测/默认路径（机制级；身份值冻结另计） | **true（机制级）** | `teachers/e1_formal/reference_contract.py`（8 类字段无默认；TODO/latest/auto/空⇒拒）；`test_reference_contract.py`；身份值待总控冻结 |
| 8 | CANDIDATE_LEARNABILITY_REAL | 本候选真实双 probe+Wilson CI 产生三态分类（无 0.25 替代） | **false** | 本轮无 probe⇒`LEARNABILITY_UNAVAILABLE`⇒`SELECTION_BLOCKED_NO_REAL_EVIDENCE`；分类器+fixture 绿（`test_learnability.py`）；0.25 已删（grep 审计） |
| 9 | ANCHOR_RETENTION_REAL | 同一 Student 在冻结 manifest 绑定的 4 anchor 上更新前/后真实评价 | **false** | manifest=DRAFT_UNFROZEN⇒`BLOCKED_SHARED_ANCHOR_MANIFEST`（`test_anchor_manifest.py`）；成就数替代已删 |
| 10 | SOFT_COPELAND_PARITY | pin canonical_v2+源码 SHA 的跨实现 fixture 门禁四项全等且绿（无 skip） | **true（对本分支 d052 canonical）** | `test_copeland_parity.py`；pin：copeland.py `80a60829…`、canonical_constants.py `32c7a1c9…`、base.py `c9d08585…`；CC3 bagr_ued 源不在本 worktree，跨 CC3 比对待总控提供源/SHA |
| 11 | LLM_ACCOUNTING_CORRECT | N1=6·G1+T1+K1+F1 账本实现+公式核对测试绿；T1≡0、K1 按唯一 artifact、F1 独立；无"固定 7 次"表述 | **true** | `teachers/e1_formal/accounting.py`；`test_llm_accounting.py`（1 窗+10 spec×2 变体⇒board=6、K1=20、T1=0、F1=0）；"每窗 7 次"表述已从全部源码删除 |

## 汇总

- true（6）：#1、#2、#3、#7（机制级）、#10（对 d052 canonical）、#11
- false（5）：#4、#5、#6、#8、#9 —— 全部因真实证据/冻结件缺位，
  按 D5 降级链诚实阻断，未伪造任何数值。

**INDEPENDENT_AUDIT_REQUIRED = true**：以上全部取值请总控独立审计。
