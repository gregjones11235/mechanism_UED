# E3 剩余 Smoke 阻断(总监复审前)

阶段最高: **E3_OBJECT_LEVEL_CONSUMER_READY**(未伪造 DIRECTOR_SMOKE_HANDOFF_READY)。

## 本轮已修复(P0)
1. **可信 Bundle Verifier**: `bundle_verifier.py` — TrustedSignerRegistry / DirectorBundleVerifier / 注入槽;
   未注入时生产 bundle 一律 `E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND` BLOCKED。不自行发明密码学。
2. **DiCode ABI**: `DiCodeOneUpdateContext` + `execute_one_update` 使用真实
   `run_session_training(config, rng, rl_train_state, gen_manager, global_update_step,
   global_env_steps, current_session_idx, sampled_task_ids, original_return_prev_session)`;
   旧 `session(plan, adapter, params, run_state, budget)` 已删除。
3. **Anchor 显式身份**: 非位置 `anchors[:3]/[3]`;显式 non_target_anchor_ids(3)+
   original_task_anchor_id(1)+original_task_id(original_craftax);OriginalTask 绝不进 sampled_task_ids。
4. **记忆边界**: 真实 129 步执行需本地 CC2 checkpoint —— 当前本地无真实文件(在服务器),诚实 BLOCKED,
   不以字符串冒充真实边界证据。

## 仍存阻断(等待总监)
| 阻断 | 解除 |
|------|------|
| `E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND` | 总监注入共享 DirectorBundleVerifier + TrustedSignerRegistry |
| 真实 CC2 checkpoint 本地缺位 | 提供两个 Student 的真实 checkpoint 文件 |
| `E3_STUDENT_TRAINING_RUNTIME_READY=false` | 总监为选中 Student 绑定 CanonicalDiCodeOneUpdateRuntime |
| `E3_REAL_SMOKE_AUTHORIZED=false / FORMAL_EXPERIMENT_AUTHORIZED=false` | 总监人工批准 |

仅当两个 Student 对象级 check-only 均通过且训练 ABI 正确后,才允许 `DIRECTOR_SMOKE_HANDOFF_READY=true`。
