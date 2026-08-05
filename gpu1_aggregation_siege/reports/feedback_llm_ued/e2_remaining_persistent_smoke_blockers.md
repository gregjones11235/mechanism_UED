# E2 Persistent Smoke 剩余阻断（完整接线后）

分支 `henry/ba-bagr-ued-review-board-v2`。本轮把已写出的修复（可信 Verifier、
bundle 哈希交叉绑定、Registry 对象解析、三态槽位、Canonical DiCode 15+1）真正接入
生产入口：`main()` 显式依赖注入；`run_e2_entrypoint` / `run_e2_object_level_check` /
`run_e2_production_two_window` 构成可静态追踪的调用链；Controller 固定
`execution_mode="PRODUCTION"`，永不自设 test_only；CLI 仅 `--director-runtime-bundle /
--student-candidate-id / --check-only / --report-out`，`--transport` 动态导入禁止。

## 1. Persistent 对象级 check-only（本轮焦点）

`E2 + PERSISTENT_RMT16_ORIGINAL_VTRACE_98304`：**OBJECT_LEVEL_CHECK_BLOCKED（诚实）**。
真实 DirectorBundleVerifier / FormalAssetRegistry / Persistent checkpoint 对象未注入，
`run_e2_object_level_check` 在 require_trusted_verifier 处 fail closed
（PRODUCTION_BUNDLE_VERIFIER_UNBOUND），不执行任何 LLM/Probe/Simulator/optimizer。

## 2. 达成 DIRECTOR_SMOKE_HANDOFF_READY 的唯一路径

1. 总监注入共享 DirectorBundleVerifier（signer 注册 + payload 哈希 + source commit
   + implementation hash）→ 通过；
2. 总监注入共享 FormalAssetRegistry（registry_identity/registry_hash/
   resolve_asset/verify_implementation）→ 解析 13 类真实对象（identity==manifest、
   implementation hash 经注册表验证）；
3. Persistent checkpoint 文件在场且 SHA256 一致、params hash 一致、memory/carry 匹配；
4. 本地 jax/craftax 可导入；
5. transport closure 由 Registry 解析；
6. 人工 Smoke 批准（E2_REAL_SMOKE_AUTHORIZED=true）。

## 3. 本轮已修复（全绿）

| 项 | 状态 |
|---|---|
| 生产入口真实调用链 | 已接线（§一/§二/§三/§九） |
| 完整对象解析（13 类） | 已扩展（§四） |
| 三态真实语义（BOUND_OBJECT 必有对象） | 已强化（§五） |
| 最强 Student 完整身份绑定 | 已绑定（§六） |
| 生产 Controller 自设 test_only | 已移除（§七 P0，execution_mode 契约） |
| 生产仅 Canonical DiCode | 已强制（§八） |
| CLI 锁死 | 已锁死（§十） |
| 残留导出 + smoke_only 语义 | 已清理（§十一） |

测试：全量 `d052/tests` **1552 passed** + 9 项调用追踪测试（`cc3_persistent_full_repair.log`）。
