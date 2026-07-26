# Experiment Timeline

## D052

Terminal 5×5 native end-to-end evaluation over aggregation selectors and student training mechanisms. Frozen status: `D052_STATUS=TERMINAL`, `NATIVE_END_TO_END_EVAL=25/25_COMPLETE`, `D052_BEST=original_x_clpa_seed0`, `ENHANCED_VARIANTS=WEAK`.

## P2-v0 invalid

Initial P2 attempt, frozen invalid due to severe implementation bugs and regression. Archived as provenance and negative engineering evidence, not performance evidence.

## P2-v1-lite

Pipeline-fixed version. It is useful engineering progress but is not formal Henry P2 because it does not fully implement genuine long-context Student, whole-episode replay, hindsight relabel, and off-policy TD Actor/Value.

## P2-Full-A-v1

Full-A implements native GTrXL-128, whole-episode replay, replay length 512, V-trace original-goal Actor/Value, hindsight AWR Actor/Value, sparse memory anchors, re-burn-in, EMA target, policy-lag, importance sampling, ESS, transactional KL gate, optimizer rollback, sequence sampler, checkpoint state, and exact resume. Engineering passed, performance did not show sustained improvement within 98,304 steps.

## P2 Posthoc Attribution

Posthoc analyses identified a structural history gap: policy behavior depends strongly on recent GTrXL memory, zero-memory induces large KL/action flips, recent128 is close to 384 burn-in, the original architecture discards information beyond 128 steps, and many failures exceed 128/256/512 steps. This does not prove long memory will improve performance.

## P7/P8/P9

P7 EgoMap reached engineering evidence but evaluation validity remains unresolved. P8 Summary LongMemory was used by the policy but harmful. P9 Authentic Reset showed no sustained positive signal.

## Long-Memory Bakeoff and Carry Matched Ablation

Phase1 did not identify a long-memory winner. Phase2 matched carry/reset ablations showed no causal carry signal for SlowGRU or EventMemory; observed effects are best treated as training regularization rather than long-memory benefit.

## SlowGRU Reset128 Phase3

SlowGRU Reset128 showed transient early gain sourced from within-rollout recurrence, not long-term carry. The gain did not sustain to 98,304 steps.

## W512 Phase4A

Replay repaired W512 training collapse but eliminated the apparent carry advantage. W512 is not a long-memory causal candidate and is not a performance upgrade.

## RMT16 Phase4A

RMT16 has matched initialization and unit-level engineering evidence, but current scientific result is not reached. The known current blocker is a repairable collector `env_params` wiring error in the 4096 smoke path unless later evidence supersedes it.

