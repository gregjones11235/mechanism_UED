# Base GTrXL Matched Replay Control — diff vs P2-Full-A

This is the **missing strict control** (MISS-1). It is NOT a performance bug; it is a design gap that
prevents isolating the P2 Replay effect from architecture/training-bundle effects. Status:
`BASE_GTRXL_MATCHED_REPLAY_CONTROL = READY_NOT_AUTHORIZED` (config built; **training not launched**).

## Single difference
| Field | P2-Full-A | BASE_GTRXL_MATCHED_REPLAY |
|---|---|---|
| **network** | GTrXL + long-memory/replay-augmented trunk (e.g. W512/RMT/SlowGRU/EventMem) | **Base GTrXL `ActorCriticTransformer`, NO long-memory module** |

Everything else is held identical:
- init: teacher17500 (`d4e85af58b7f87d6`)
- Replay bundle: capacity 64, L_SEQ (see conflict), K_BATCH 4, V-trace, AWR/hindsight, EMA tau 0.995,
  policy-lag gate (16), transactional KL gate, anchor_interval 128
- PPO: Adam lr 2e-5 eps 1e-5, gamma 0.999, grad-clip 1.0, num_envs 16, rollout 128, 2048/update, 24576 total
- eval: CANONICAL_EVALUATOR_V1, 256 worlds, seed42, stochastic, max4096, DEFEAT_KOBOLD

## Open item that MUST be frozen before any run
- **L_SEQ conflict (MISS-6):** W512 repro = 129; RMT16 / P2-Full-A v2.1 = 512. The matched arm must use the
  SAME L_SEQ as the P2-Full-A arm it controls for. Freeze canonical L_SEQ on the server and record it in
  provenance before launching.

## Comparisons this unlocks
1. `BASE_GTRXL_MATCHED_REPLAY` vs `P2-Full-A` → isolates the **architecture** contribution under identical Replay.
2. `BASE_GTRXL_MATCHED_REPLAY` vs `CONTROL24576_BASELINE` (93/256) → isolates the **Replay-bundle** contribution
   under identical Base GTrXL — the comparison that is currently impossible (GATE15 verifies the config match).

## GATE15
The regression gate asserts this config equals the P2 bundle on every field EXCEPT `network.class` /
`long_memory_module`. Implemented in `tools/regression_gates.py`.
