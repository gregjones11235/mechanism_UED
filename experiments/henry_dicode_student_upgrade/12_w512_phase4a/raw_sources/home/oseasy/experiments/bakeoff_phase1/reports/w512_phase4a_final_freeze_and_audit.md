# W512 Phase4A Final Freeze and Audit Report

**Overall Status: PASS_WITH_LIMITATIONS**
**Phase4A Verdict: REPLAY_ELIMINATES_CARRY**
Generated: 2026-07-26T01:46:52Z

---

## 1. Frozen Results

### Six-Arm DK Success Rate

| Arm | DK SR | n/256 | floor3 | death | eplen |
|-----|-------|-------|--------|-------|-------|
| Baseline (ckpt17500) | 39.45% | 101 | 43.36% | 147 | 986 |
| Control @24576 | 36.33% | 93 | 43.36% | 156 | 862 |
| W512-Persistent (PPO) | 10.94% | 28 | 17.58% | 226 | 622 |
| W512-Reset128 (PPO) | 2.73% | 7 | 6.64% | 249 | 575 |
| W512-Persistent-P2Replay | 35.16% | 90 | 41.02% | 162 | 954 |
| W512-Reset128-P2Replay | 37.11% | 95 | 44.92% | 152 | 1025 |

### Core Causal Quantities

| Quantity | Value | p | 95% CI |
|----------|-------|---|--------|
| CARRY_NO_REPLAY (PPO) | +8.20pp | 6.3e-05 | [+4.69, +12.11] |
| CARRY_WITH_REPLAY | -1.95pp | 0.42 | [-5.86, +1.95] |
| REPLAY_EFFECT_PERSISTENT | +24.22pp | <1e-6 | [+18.75, +29.69] |
| REPLAY_EFFECT_RESET | +34.38pp | <1e-6 | [+28.52, +40.23] |
| MEMORY_REPLAY_INTERACTION | -10.16pp | — | — |

### Frozen Labels

- W512_PPO_TRAINING_HEALTH = FAILED
- W512_REPLAY_STABILIZATION_SIGNAL = true
- W512_LONG_MEMORY_CAUSAL_CANDIDATE = false
- W512_PERFORMANCE_UPGRADE = false
- W512_PHASE4A_VERDICT = REPLAY_ELIMINATES_CARRY

---

## 2. Audit Results Summary

| Audit | Verdict | Checks |
|-------|---------|--------|
| Evaluation Audit | PASS | 15/15 |
| Training Match Audit | PASS | 20/20 |
| P2Replay Package Relation | FUNCTIONAL_MATCH_WITH_LISTED_DIFFERENCES | — |
| Replay Diagnostics | COMPLETE | 20/20 items |
| Artifact Manifest | GENERATED | 44 artifacts |

---

## 3. Limitations

1. 两个不同评估器脚本（SHA不同），但协议功能等价。核心因果量CARRY_WITH_REPLAY使用同一评估器评估，不受影响。
2. Control臂使用PPO训练（非Replay），不能作为P2Replay臂的严格同方法参考。Control比较测量的是(W512+Replay) vs (base GTrXL+PPO)的组合效应。
3. Replay buffer内容（per-episode outcome构成）未持久化到checkpoint，无法事后审计buffer组成。

---

## 4. Frozen Interpretation Rules

1. Persistent相对Reset128的+8.20pp只表示"少退化"，不能解释为健康长期记忆能力。
2. Replay修复的是W512训练健康，而不是跨128步Carry能力。
3. W512不再是长期记忆候选。
4. 不得称W512+Replay实现了Student性能升级。
5. 推荐表述：「Replay stabilized W512 training and recovered control-level performance, but did not produce positive cross-rollout carry value.」

---

## 5. GPU Discipline

- GPU0: 空闲，本阶段未使用
- GPU1: 空闲，本阶段未使用
- GPU2/GPU3: 禁触
- 本阶段未启动任何新训练

---

## 6. Authorizations (all unchanged)

- TRAINING_TO_98304_AUTHORIZED = false
- W512_SECOND_SEED_AUTHORIZED = false
- UPDATE_HORIZON_ON_W512_AUTHORIZED = false
- P2_FULL_B_AUTHORIZED = false

---

## 7. Reports Generated

| # | File |
|---|------|
| 1 | reports/w512_phase4a_evaluation_audit.md / .json |
| 2 | reports/w512_phase4a_training_match_audit.md / .json |
| 3 | reports/w512_phase4a_config_diff.txt |
| 4 | reports/w512_phase4a_p2replay_package_relation.json |
| 5 | reports/w512_phase4a_replay_diagnostics_audit.md / .json |
| 6 | reports/w512_phase4a_artifact_manifest.md / .json |
| 7 | reports/post_rmt_decision_tree.md |
| 8 | reports/w512_phase4a_final_freeze_and_audit.md / .json |

---

## 8. Conclusion

W512 Phase4A审计完成。所有审计检查通过（含两项已记录限制）。
W512的+8.20pp carry效应被确认为PPO训练不稳定的artifact，而非真实长期记忆收益。
P2 Replay作为训练稳定化机制有效，恢复了control-level性能，但未产生正向carry值。
W512不再是长期记忆候选架构。所有结果已冻结，所有科学产物标记只读。
