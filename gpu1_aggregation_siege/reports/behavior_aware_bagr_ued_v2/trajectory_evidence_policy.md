# Trajectory evidence policy

ALLOWED evidence: current Student generative-training trajectories — action
semantics (resolved via EXTERNAL symbolic adapter; no hardcoded Craftax
action integers, no raw state leaf indices), state/resource change summaries,
threat/damage/achievement/progress/death/timeout events, limited windows
around anomalies.

FORBIDDEN: formal FRONT/BACK/FULL state payloads; formal evaluation per-state
trajectories; formal map/ladder positions; expert trajectories; Reference
action sequences as demonstration; hidden policy state as supervision; manual
correct-action labels.

Guards: A. TrajectorySupervisionGuard — rejects in ANY output the keys
recommended_actions / action_sequence_to_follow / waypoints /
expert_demonstration / policy_override / hidden_state_override / reward_delta
/ reward_shaping, and any direct action-advice text (bilingual patterns).
B. FormalEvaluationLeakageGuard — rejects FORMAL_FRONT / FORMAL_BACK /
FORMAL_FULL / FROZEN_BANK / FORMAL_EVALUATION_CERTIFICATE_PRIVATE_STATE
provenance anywhere in the input. Both fail closed with greppable codes.
