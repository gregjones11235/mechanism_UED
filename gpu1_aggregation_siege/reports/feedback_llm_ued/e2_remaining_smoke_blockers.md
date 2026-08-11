# E2 剩余 Smoke 阻断（REQUEST_CHANGES 修复后）

分支 `henry/ba-bagr-ued-review-board-v2`。本轮完成了真实对象注入边界的修复：
三态槽位（DECLARED_NOT_RESOLVED / BOUND_OBJECT / EMPTY）、可信 Bundle 验证
边界（共享 DirectorBundleVerifier，生产移除自签入口）、对象解析器
（resolve_director_runtime_objects）、CLI 身份锁死、runtime_bundle_hash 交叉
绑定、生产路径仅 Canonical DiCode（REAL_DICODE_BATCH_PLAN_REQUIRED；legacy 仅
TEST_ONLY_LEGACY_ADAPTER）。最高状态 `E2_OBJECT_LEVEL_CONSUMER_READY`。

## 1. 对象未注入时的诚实阻断（两个 Student 一致）

当前工作树只有 manifest 身份，真实对象未从 FormalAssetRegistry 解析，本地
无 jax/craftax —— `--check-only` 对 PERSISTENT 与 RESET128 均诚实 BLOCKED：

* 对象槽位（probe_runner / training）为 DECLARED_NOT_RESOLVED →
  BLOCKED_WAITING_SHARED_RUNTIME；
* jax / craftax 本地不可导入 → LOCAL_RUNTIME_MODULE_MISSING × 2；
* 不报告 DIRECTOR_SMOKE_HANDOFF_READY。

## 2. 达成 OBJECT_LEVEL_CHECK / Smoke 交接的唯一路径

1. 注入共享 DirectorBundleVerifier（signer 注册 + payload 哈希 + source
   commit + implementation hash 校验）—— 未注入即 PRODUCTION_BUNDLE_VERIFIER_UNBOUND；
2. 由 FormalAssetRegistry 解析全部 11 个真实对象（identity == manifest）；
3. 本地 jax / craftax 可导入；
4. （真实 Smoke）注入 transport closure 对象；
5. 两个 Student 之一真实挂载。

## 3. 本轮修复清单（已测试）

| 项 | 状态 |
|---|---|
| BOUND 但对象 None 的错误状态 | 已消除（三态模型 + resolve 机械检查） |
| 生产自签入口 | 已移除（sign 辅助仅在 d052/tests/） |
| CLI 覆盖 Bundle 身份 | 已锁死（冲突即拒绝） |
| runtime_bundle_hash 交叉绑定 | 已校验（E2_RUNTIME_BUNDLE_HASH_MISMATCH） |
| 生产回退旧 optimizer | 已禁止（REAL_DICODE_BATCH_PLAN_REQUIRED） |

## 4. 未运行 / 未验证（诚实标注）

| 项 | 状态 |
|---|---|
| 真实 Smoke（任一 Student） | 未启动 |
| 正式实验 | 未启动（FORMAL_EXPERIMENT_AUTHORIZED=false） |
| 冻结 D052 历史证据 | 未修改 |
| 全部 REAL_* 旗标 | 恒 False |
