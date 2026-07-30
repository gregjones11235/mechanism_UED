"""D052-v2 BA-BAGR-UED: Behavior-Aware Bottleneck-Aware Global Regret UED.

A GLOBAL UED controller + Tier3 FRONT bottleneck signal + multi-role Student
failure-behavior review board + counterfactual environment induction, added as
a d052 subpackage reusing the canonical_v2 conventions (CanonicalModel,
fail-closed greppable codes, deterministic hashing, deterministic selection).

What this package IS:
  * deterministic trajectory evidence extraction (plugin detectors, symbolic
    adapter, no hardcoded Craftax action ints / state leaf indices);
  * a six-role review board (StudentModeler / BehaviorAuditor /
    CausalFailureAnalyst / InterventionTutor / Explorer / Critic-Skeptic)
    behind a mock LLM backend (REAL_LLM_CALLS_AUTHORIZED=false);
  * a rule-based Reconciler binding every decision to role output hashes,
    evidence span hashes, prompt versions and backend identity;
  * counterfactual environment builders + mock TaskParams descriptors
    (REAL_TASKPARAMS_ADAPTER=BLOCKED_EXTERNAL_DEPENDENCY);
  * THREE separate scoring dimensions (front_regret / global_regret /
    behavioral_gap) + Soft Copeland (>=8 inputs) + 12 UED + 4 anchor budget.

What this package is NOT (task section 0):
  * not a Tier3-only trainer, not trajectory imitation, not expert demos,
    not action guidance, not reward shaping, not a hand-crafted curriculum.

This round: ENGINEERING_DRY_RUN only. TRAINING_AUTHORIZED=false,
FORMAL_EVALUATION_AUTHORIZED=false, REAL_LLM_CALLS_AUTHORIZED=false,
GPU_USED=false, PUSH_PERFORMED=false.
"""
from d052.bagr_ued import constants as C
from d052.bagr_ued.controller import (
    BAGRUEdController,
    DryRunResult,
    assert_round_authorization,
)

__all__ = [
    "C",
    "BAGRUEdController",
    "DryRunResult",
    "assert_round_authorization",
]
