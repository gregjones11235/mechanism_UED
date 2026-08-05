# E3 Persistent 剩余 Smoke 阻断(复审前)

阶段: **E3_PERSISTENT_OBJECT_CONSUMER_IMPLEMENTED**(head 750d17b68e4288454023d868c4e6c646ad89b55b)。

## 本轮落地
- controller_identity 进入 exact schema(非空、参与 manifest_hash);controller_signature_ref 仅作签名引用。
- manifest_hash 唯一 canonical(排除自身),schema validator 与 verifier 调用同一 manifest_canonical_hash。
- 伪密码学已删;DirectorBundleVerifier 注入 verify_signature 必须返回 True。
- checkpoint_file_sha256 统一;checkpoint_sha256 被 exact schema 拒绝。
- E3_WINDOW_STEPS 改为 STEP09_COMPILE_CANONICAL_DICODE_PLAN .. STEP13_FRESH_PROCESS_RESTORE_AND_EQUIVALENCE。
- py_compile(5 文件)OK;24 bundle 定向测试 + 既有 530 全绿。

## 仍存阻断(下一提交/总监侧)
| 阻断 | 解除 |
|------|------|
| E3WindowConfig canonical 字段 + 入口构造 Canonical 对象 | 下一提交实现 |
| execute_one_update 8 元组解析 + OneUpdateReceipt | 下一提交实现 |
| TaskArchive 真实 API 注册 + load_tasks_from_env_codes 验证 | 下一提交实现 |
| Smoke config 副本(deepcopy,max_updates_per_session=1) | 下一提交实现 |
| CanonicalDiCodeRunStateCheckpoint + fresh-process 恢复 | 下一提交实现 |
| `E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND`(真实 verifier 未注入) | 总监注入 |
| Persistent 真实 CC2 checkpoint 本地缺位 | 总监提供 |
| `E3_REAL_SMOKE_AUTHORIZED=false / FORMAL_EXPERIMENT_AUTHORIZED=false` | 总监批准 |

真实 Persistent 对象级 check-only 通过后才允许 E3_PERSISTENT_OBJECT_CHECK_ONLY_OK / DIRECTOR_SMOKE_HANDOFF_READY=true。
