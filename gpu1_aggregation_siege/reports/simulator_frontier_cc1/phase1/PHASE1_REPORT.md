# Phase 1 真实 Craftax EnvState 恢复/动力学平价报告（R4a env 侧）

- verdict：**PASS**
- pin 集：proven（主机已验证组合）
- 核心环境：MiniCraftaxTrain(survive.Env)，动作维度 43
- seeds：{'reset': 20260803, 'runner': 777, 'action': 0}
- key 约定：r, step_key = jax.random.split(r); step_env(step_key, ...)
- state_rng 说明：EnvState.state_rng is overwritten by the engine every step and never consumed by step_env; restored byte-for-byte as an ordinary leaf, never used as a step key

## 门禁结果

| 检查 | 结果 |
|---|---|
| bootstrap | PASS |
| restore_roundtrip | PASS |
| dynamics_parity | PASS |
| terminal_restore | PASS |
| autoreset_evidence | PASS |
| corrupted_payload | PASS |
| version_mismatch | PASS |
| batch_parity | PASS |
| multitask_secondary | PASS |

## 范围声明

PASS here proves R4a (env-side restore/parity) ONLY; it is not the R4c combined fresh-process proof.

本报告仅为 R4a（env 侧 restore/parity）证据；不构成 R4c 联合 fresh-process 证明，
也不构成任何性能评估。
