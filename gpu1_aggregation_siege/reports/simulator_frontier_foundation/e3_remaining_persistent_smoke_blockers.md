# E3 Persistent 剩余 Smoke 阻断(复审前)

阶段最高: **E3_PERSISTENT_OBJECT_CONSUMER_READY**(未伪造 DIRECTOR_SMOKE_HANDOFF_READY)。

## 本轮已修复
1. **manifest_hash 统一**: REQUIRED_TOP_KEYS 要求 manifest_hash;schema 校验/verifier/入口
   验证同一 canonical hash(排除自身),单一 payload hash;夹具已补。
2. **伪签名删除**: `signature_ref.split(":",1)[0]` 字符串推断已删;DirectorBundleVerifier 现携带
   总监注入的 `verify_signature(signer_id, payload_hash, signature_ref) -> bool`,生产验证顺序为
   registry 信任 -> payload hash 重算 -> 注入 verify_signature 必须返回 True -> allowlist -> 实现 hash。
   未注入: `E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND` BLOCKED。
3. **字段统一**: checkpoint_file_sha256 用于 student/reference 段与入口 restore cross-binding;
   bundle schema 中残留 checkpoint_sha256 已清除。
4. **自定义训练路径**: OriginalTrainingRuntime 仅保留为 TEST_ONLY_LEGACY_ADAPTER;
   生产训练委托 CanonicalDiCodeOneUpdateRuntime(STEP09-13 完整接线与 E3WindowConfig
   canonical 字段作为下一提交追踪)。

## 仍存阻断
| 阻断 | 解除 |
|------|------|
| `E3_PRODUCTION_BUNDLE_VERIFIER_UNBOUND` | 总监注入共享 DirectorBundleVerifier |
| Persistent 真实 CC2 checkpoint 本地缺位 | 提供真实 checkpoint 文件 |
| STEP09-13 canonical 全接线(E3WindowConfig 字段、8 元组解析、TaskArchive 真实 API、RunState checkpoint、fresh-process 恢复) | 下一提交实现 |
| `E3_REAL_SMOKE_AUTHORIZED=false / FORMAL_EXPERIMENT_AUTHORIZED=false` | 总监人工批准 |

Persistent 真实对象级 check-only 通过后才允许 DIRECTOR_SMOKE_HANDOFF_READY=true。
