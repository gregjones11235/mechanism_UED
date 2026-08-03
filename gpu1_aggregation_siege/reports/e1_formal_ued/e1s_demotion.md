# E1-S（Static-LLM UED）降级记录

> **`E1S_STATIC_ABLATION_PRESERVED = true` 的准确含义**：Static-LLM
> 工件**未删除、未覆盖、且被 E1 复用**——仅此而已。
> **不等于 E1-S 实验本轮可运行**。本轮没有 static 控制器，
> 也没有为其排期；E1-S 的补完与调度是总控决定。

## 一、保留的工件（原样，未删未覆盖）

| 工件 | 位置 | 状态 |
|---|---|---|
| 确定性输出 guards | `src/dicode/teachers/static_llm/guards.py` | 保留；原两处正则缺陷（`\breward` 不跨下划线漏检 `total_reward += bonus`；`\bwaypoints?\b` 漏检 dict 键 `waypoint_list`）已在 C1（`7c6bc88`）修复并收编提交——收编前先审查，修复仅针对这两处确诊缺陷，其余逻辑一字未动 |
| guards 测试 | `tests/static_llm/test_guards.py` | 保留并随 C1 收编；tests/static_llm 全绿 |
| 共享 schema | `src/dicode/teachers/static_llm/schemas.py` | 保留（`edf10cb` 提交）；E1 `e1_formal/schemas.py` import 复用，未重复定义 |
| StudentInitContract 薄消费端 | `src/dicode/teachers/static_llm/student_init_contract.py` | C2（`100488e`）新增——这是 static_llm 包内本轮**唯一增量**；身份最小集，无文件 I/O，非 loader 非 registry |

## 二、E1 对 E1-S 工件的复用

- `e1_formal/schemas.py` 直接 import `static_llm.schemas`（共享
  fail-closed 错误类型族，不另立平行 schema）。
- E1 复用 guards 扫描六角色输出（board.py 在解析前先过 guards）；
  guards 的拒绝语义对两条方向线一致。
- StudentInitContract（C2）是强 Student 身份消费的薄端，E1 与
  未来 E1-S 共用同一 pinned 身份 `PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`。

## 三、本轮不运行的部分与原因

1. **无 static 控制器**：计划 D6 明确本轮不写 static 控制器、不建
   其 yaml。E1-S 作为弱消融的完整实验留待总控排期。
2. **`setup.py::_resolve_teacher` 对 `teacher_type=static_llm` 显式
   抛 `NotImplementedError`**，消息指向本文档与降级事实；绝不静默
   回退到任何假控制器（防止伪造 E1-S 运行记录）。
3. 因此任何声称"E1-S 已运行/已对比"的表述在本轮都不成立；
   `E1S_STATIC_ABLATION_PRESERVED` 只证明上表的工件事实。

## 四、若总控未来排期 E1-S

需要（至少）：static 控制器实现（在既有 guards/schema/契约之上）、
其专属 yaml、与 E1 相同口径的降级链诚实标志，以及对
`setup.py::_resolve_teacher` 该分支的替换——该替换同样受"默认路径
字节不变"与路径限定提交约束。
