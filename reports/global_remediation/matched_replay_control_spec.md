> GLOBAL_EVALUATION_REMEDIATION (CC4) · repo mechanism_UED @ Henry-branch · HEAD=UNVERIFIED (git access BLOCKED)
> Environment: JAX/Craftax ABSENT, experiment server UNREACHABLE, GPU idle (not used), no CC2/CC3 processes.
> Discipline: read-only w.r.t. CC2/CC3 & old audit; NEW_TRAINING_RUNS=0; MISSING/UNVERIFIED never relabeled FAIL.

# Matched-Replay Control Spec (BASE_GTRXL_MATCHED_REPLAY)

Status: **READY_NOT_AUTHORIZED** — config built; training NOT launched (NEW_TRAINING_RUNS=0). Fills MISS-1.

## Single difference vs P2-Full-A
network = Base GTrXL `ActorCriticTransformer`, long_memory_module = NONE (NOT W512/RMT/SlowGRU/EventMemory).
Everything else identical: init teacher17500 (d4e85af58b7f87d6); replay capacity 64 / K_BATCH 4 / V-trace /
AWR / EMA tau 0.995 / policy-lag 16 / transactional KL gate / anchor 128; PPO Adam lr 2e-5 eps 1e-5 gamma
0.999 grad-clip 1.0 num_envs 16 rollout 128 2048/update 24576 total; eval CANONICAL_EVALUATOR_V1 256 worlds
seed42 stochastic max4096 DEFEAT_KOBOLD. L_SEQ pinned 512 with CONFLICT NOTE (MISS-6 must freeze first).

Comparisons unlocked: (1) vs P2-Full-A → isolate ARCHITECTURE under identical Replay; (2) vs
CONTROL24576_BASELINE (93/256) → isolate REPLAY-bundle under identical Base GTrXL. GATE15 verifies the config
equals the P2 bundle on every field except network.
