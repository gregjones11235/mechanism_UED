# E3 双 Student 交接阻断(导演核验前)

阶段:**E3_DUAL_STUDENT_CONSUMER_READY** —— 双 Student 可选择接入已贯穿 Runtime Bundle、
Frontier Archive、memory restore、actual-N 和训练交接契约,并由 12 个专用测试文件钉住。

保持诚实:`REAL_ACTUAL_N_EXECUTED=false / REAL_TWO_LLM_EXECUTED=false /
REAL_ONE_UPDATE_EXECUTED=false / CHECKPOINT_RELOAD=false /
E3_REAL_SMOKE_AUTHORIZED=false / FORMAL_EXPERIMENT_AUTHORIZED=false`。

## 已交付

1. **冻结双 Student 允许集合**:`ALLOWED_PRIMARY_STUDENT_IDS` 恰为
   `PERSISTENT_RMT16_ORIGINAL_VTRACE_98304` 与 `RESET128_RMT16_ORIGINAL_VTRACE_98304`;
   未知候选 fail closed,绝无默认第一名。
2. **Runtime Bundle 显式选择 Student**:student section 携带
   selected_candidate_id / profile_name+hash / checkpoint path+file sha / params sha /
   source_commit / adapter entrypoint+implementation+identity hash / memory_mode /
   memory_spec_hash / carry_mode / driver source+sha;验证
   selected == profile.candidate_id == adapter.identity().candidate_id == checkpoint/carry 语义。
3. **Frontier Archive 绑定选定 Student**:entry 增加 source_memory_mode + runtime_bundle_hash;
   同一运行 capture==search==train==selected;新增三个 E3_FRONTIER_* 错误。
4. **双记忆语义分别验证**:PERSISTENT_CARRY(段边界保留 mem_tokens)与
   RESET128_CARRY(段边界清零)在构造上不同并分别执行;跨臂记忆恢复被拒。
5. **actual-N / mixed-start 连续性**:Student 分支与训练交接绑定同一选定 Student;
   前两名 Student 是两套独立实验起点,不是 Student/Reference 二元组
   (第二 Student 作 Reference 被拒)。
6. **只读 Adapter 与训练 Runtime 分离**:RMT16StudentAdapter 只读挂载;
   save/restore 为 NotImplementedError;继续训练由总监 CanonicalDiCodeOneUpdateRuntime
   (已绑定 selected_candidate_id)承担;未绑定时只读=就绪、训练=false。
7. **入口**:`--runtime-bundle / --student-candidate-id / --check-only / --report-out`;
   候选缺失/不匹配一律 FAIL,check-only 不执行 actual-N/LLM/mixed-start/update/写 checkpoint。
8. **测试**:12 个新测试文件(见 `e3_dual_student_binding_audit.json`)。

## 剩余阻断(等待导演)

| 阻断 | 含义 | 解除条件 |
|------|------|----------|
| `E3_STUDENT_TRAINING_RUNTIME_READY=false` | 未见绑定选定 candidate 的 CanonicalDiCodeOneUpdateRuntime | 总监提供并为选中 Student 绑定训练运行时 |
| `E3_REAL_SMOKE_READY=false` | 未授权真实 Smoke | 总监授权 E3_REAL_SMOKE |
| `REAL_TWO_LLM_EXECUTED=false` | 双 LLM 未执行 | 总监授权真实 LLM 客户端并铸造 TwoLLMRuntimeDescriptor |
| `E3_REAL_SMOKE_AUTHORIZED=false` | Smoke 未授权 | 导演审核本报告后批准 |
| `FORMAL_EXPERIMENT_AUTHORIZED=false` | 正式实验未授权 | 人工批准(另见 98304 不再作为正式预算) |

## 下一步(单一最高依赖)

导演核验双 Student check-only 绑定(两个 Profile + 真实 checkpoint 的对象级绑定),
并在批准后执行 Smoke。
