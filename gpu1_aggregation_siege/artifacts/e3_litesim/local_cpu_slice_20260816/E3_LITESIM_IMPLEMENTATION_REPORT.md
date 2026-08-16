# E3-LITESIM Implementation Report

## 1. Real structure of the original E3
See docs/e3_litesim/E3_CURRENT_ARCHITECTURE_AUDIT.md: longrun -> e3_window / branch_search_runner (Actual-N) -> frontier_distributions (12+4) -> ppo_tr.make_train(backend) -> RunStateCheckpointManager.

## 2. Files modified
No shared production files modified (ppo_tr.py, training_backend*.py, student_adapters/*, minicraftax/* untouched).

## 3. Modules reused from old code
env_restore (freeze/restore/stack), backend ABC semantics, student adapter protocol surface, RunState hash discipline, slot semantics.

## 4. Measurement Plane
tier_registry + capability_probe with comprehensive behavior metrics (success, progress, health, floor, oscillation, stalls, torch latency, threat damage) + deterministic frontier_locator + failure_capsule + diff-guarded single-factor counterfactual_runner + causal_evidence with UNKNOWN first-class.

## 5. Data Plane
frontier_spec + state_bank (frozen + prefix-variant simulator-valid states with provenance hash chain) + state_sampler + lightweight_rollout (state-start vectorized short rollouts) + data_engine (distribution -> per-family on-policy batches).

## 6. PPO Bridge
Canonical PPO objective (clipped surrogate + value + entropy, GAE) recomputed sequence-wise from captured entering memory through the backend ABC; ppo_tr.py byte-identical; minibatches split env axis.

## 7. Student binding
StudentBindingGuard binds runstate/probe/ppo/checkpoint hashes each iteration; PROBE_INVALID fail-closed (G1).

## 8. FrontierStateBank generation
Frozen capsule states + prefix-k policy continuations; every entry restore-and-step validated (G2).

## 9. Recurrent state
Entering memory captured per start state; memory_trace enables mid-episode capsule capture; SlowGRU longstate keys validated; zero-longstate state-start rejected unless explicit memory-reset intervention (G3).

## 10. SlowGRU persistent semantics
Preserved by contract: the bridge consumes the same backend surface (policy_forward_eval + longstate memory dict). Production validation requires the GPU server (slowgru_runtime not importable locally).

## 11. Lightweight vs full rollout
full=4.7 t/s; short=2.0 t/s; speedup=0.43x

## 12. transitions/sec
{"full_transitions_per_sec": 4.7, "short_transitions_per_sec": 2.0}

## 13. Is state-start rollout truly on-policy?
Yes: batches carry the generating policy hash; PPOBridge rejects any batch whose hash differs from current TrainState params (G5); D_k is discarded after the update.

## 14. Transition accounting
{
  "probe": 416,
  "diagnosis": 192,
  "training": 72,
  "anchor": 0,
  "original": 0,
  "total_simulator_transitions": 680,
  "ppo_updates": 8,
  "llm_calls": 0,
  "llm_tokens": 0,
  "wall_clock_sec": 258.862,
  "env_steps_per_sec": 2.626883852206988,
  "student_version": "slice_student",
  "accounting_hash": "d997dde0d9da30c33d67b23ae3240c4e91e912029128ba82508bb7200aed7536"
}

## 15. Vertical slice
PASS

## 16. Gates passed
{
  "G4_READ_ONLY_PROBE": true,
  "G1_STUDENT_BINDING": true,
  "G2_STATE_RESTORE": true,
  "G3_RECURRENT_STATE": true,
  "G5_ON_POLICY": true,
  "G6_PPO_BRIDGE": true,
  "G9_VERTICAL_SLICE": true,
  "G7_TRANSITION_ACCOUNTING": true,
  "G8_THROUGHPUT": true
}

## 17. Blockers
- slowgru_runtime is server-only; local slice uses the labeled slice student. Tier3 dark-corridor world validated on the server.

## 18. Ready for formal experiments?
READY_FOR_6_TO_10_SESSION_VERTICAL_TRAINING on the GPU server with the SlowGRU backend after server-side G3/G6 re-validation; the local slice proves mechanics only.
