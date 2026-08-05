# E2 Persistent Smoke 剩余阻断（REQUEST_CHANGES 修复后）

分支 `henry/ba-bagr-ued-review-board-v2`。本轮把新增修复真正接入生产入口：
main() 接受总监 Verifier + FormalAssetRegistry；调用链固定为 load →
require_trusted_verifier → assert_runtime_bundle_hash_cross_bound →
build_shared_bundle → resolve_director_runtime_objects → resolve_shared_runtime →
discover_blockers → check-only/执行。生产 Controller 永不自设 test_only
（缺 dicode 绑定即 REAL_DICODE_BATCH_PLAN_REQUIRED）；CLI 四旋钮全部锁死
（一致性确认或拒绝）；`--transport` 动态导入禁止；`sign_director_runtime_bundle`
残留导出已删；`smoke_only_origin` 错误语义已删（98304 仅是 checkpoint 身份，
非正式预算；该 Student 人工批准后可为正式起点）。

## 1. Persistent 对象级 check-only（本轮焦点）

`E2 + PERSISTENT_RMT16_ORIGINAL_VTRACE_98304` 的对象级 check-only 在真实
对象注入前**诚实 BLOCKED**（OBJECT_LEVEL_CHECK_BLOCKED / PRODUCTION_BUNDLE_
VERIFIER_UNBOUND / FORMAL_ASSET_REGISTRY_UNBOUND），不报告 PASS，不启动
LLM/Probe/训练。

## 2. 达成 DIRECTOR_SMOKE_HANDOFF_READY 的唯一路径

1. 总监注入共享 DirectorBundleVerifier（signer 注册 + payload 哈希 +
   source commit + implementation hash）→ 通过；
2. 总监注入共享 FormalAssetRegistry → 解析全部真实对象（identity==manifest）；
3. 本地 jax/craftax 可导入；
4. transport closure 由 Registry 解析（非 CLI module.attr）；
5. Persistent Student 真实挂载，全链（window/feedback/probe/training）一致。

## 3. 本轮已修复（全绿验证）

| 项 | 状态 |
|---|---|
| 生产入口调用链 | 已接线（§一） |
| 对象注入 API（main kwargs + run_e2_object_level_check） | 已建立（§二） |
| 完整对象解析 | 已扩展（§三，11 类对象 + Adapter） |
| 生产 Controller 自设 test_only | 已移除（§五） |
| CLI 覆盖 Bundle | 已锁死（§六） |
| __all__ 残留 + smoke_only_origin | 已删除（§七） |

测试：全量 `d052/tests` **1552 passed**（`cc3_persistent_runtime_repair.log`）。
