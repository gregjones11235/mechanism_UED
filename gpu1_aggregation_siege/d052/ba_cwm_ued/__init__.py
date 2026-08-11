"""BA-CWM-UED V1 — Behavior-Aware Counterfactual World Model for UED.

An INDEPENDENT, trainable, testable but SHADOW-MODE-ONLY world-model layer that
sits between the Counterfactual Generator and the real Craftax rollout. It learns
the short-horizon dynamics of *generative training* environments, simulates
counterfactual TaskParams interventions under a FIXED real action sequence, and
predicts Student outcomes (death / damage / unsafe_rest / front_transition /
defeat_kobold / progress_delta / resource_delta / entity persistence) with an
ensemble uncertainty estimate — so UED candidates can be PRE-SCREENED in shadow
mode and compared against real rollouts.

HARD frozen authorization state (this round; every flag is re-asserted at
startup and the package REFUSES to behave as though any were True):

    WORLD_MODEL_MODE                     = SHADOW_COUNTERFACTUAL
    WORLD_MODEL_CAN_CHANGE_BATCH         = false
    IMAGINED_ROLLOUT_CAN_TRAIN_STUDENT   = false
    REAL_TRAINING_AUTHORIZED             = false
    POLICY_IN_LOOP_ENABLED               = false
    REAL_CANDIDATE_PREFILTER_AUTHORIZED  = false
    REAL_IMAGINED_TRAINING_AUTHORIZED    = false
    COURSE_VALUE_CLAIM_AUTHORIZED        = false
    COUNTERFACTUAL_SEMANTICS             = FIXED_ACTION_CONSEQUENCE

First-phase counterfactual semantics answer ONLY:

    "Holding the REAL semantic action sequence fixed, how does an environment
     intervention change state, events and consequences?"

It does NOT model the Student's policy response to a changed environment,
adaptation behavior, full course value, Student-vs-Reference performance, or
final learnability (all NOT_MODELED / NOT_AVAILABLE this phase).

It does NOT change Soft Copeland / Budget Allocator / LaunchGate /
ProposalArchive / the training batch, does NOT bypass the real rollout into
final curriculum selection, does NOT connect a real LLM, does NOT emit Student
or expert actions, does NOT do reward shaping, and does NOT read formal
FRONT/BACK/FULL/FROZEN_BANK/certificate-private state.
"""
from __future__ import annotations

BA_CWM_UED_VERSION = "d052_v2.ba_cwm_ued.v1"

__all__ = ["BA_CWM_UED_VERSION"]
