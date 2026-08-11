"""BA-CWM-UED V1 frozen constants.

Every authorization flag below is a compile-time constant the controller
re-asserts at startup; the package REFUSES to operate as though any flag were
True. Flipping any of them is a director decision OUTSIDE this package (and is
refused here regardless).

This module also fixes the source-policy split (CWM allow/deny is a strict
SUPERSET of the BA-BAGR split: the same formal-evaluation sources stay forbidden
and two extra allowed training-replay sources are added) and the shared
counterfactual / semantic vocabulary NAME constants. Integer ids for the model
training path live in ``vocabularies.py`` (the training path never sees strings).
"""
from __future__ import annotations

from d052.bagr_ued import constants as BAGR

# ---------------------------------------------------------------------------
# FROZEN authorization state (THIS ROUND). NEVER permissive in this package.
# ---------------------------------------------------------------------------
WORLD_MODEL_MODE = "SHADOW_COUNTERFACTUAL"

#: The world model may NOT replace / reorder / filter the real training batch.
WORLD_MODEL_CAN_CHANGE_BATCH = False
#: Imagined rollouts may NOT train the Student (no imagined PPO, no latent ->
#: Student, no reward shaping).
IMAGINED_ROLLOUT_CAN_TRAIN_STUDENT = False
#: No real Student training is started by this package.
REAL_TRAINING_AUTHORIZED = False
#: The world model does NOT start its own real training on real environments.
WORLD_MODEL_REAL_TRAINING_AUTHORIZED = False
#: Policy-in-the-loop (closing the Student through the imagined world model) is
#: an interface-only stub this phase; it is NOT authorized.
POLICY_IN_LOOP_ENABLED = False
#: Shadow screening may NOT be used as a real candidate pre-filter.
REAL_CANDIDATE_PREFILTER_AUTHORIZED = False
#: No real imagined-training loop is authorized.
REAL_IMAGINED_TRAINING_AUTHORIZED = False
#: No full-course-value / SOTA / curriculum-improvement claim is authorized.
COURSE_VALUE_CLAIM_AUTHORIZED = False
#: No real LLM calls (mock / structured adapters only).
REAL_LLM_CALLS_AUTHORIZED = False
#: Final launch authorization may NOT be set by the shadow controller.
TRAINING_LAUNCH_AUTHORIZED = False

#: First-phase counterfactual semantics (amendment §1): action replay answers
#: ONLY "holding the real semantic action sequence fixed, how does an
#: environment intervention change state / events / consequences?".
COUNTERFACTUAL_SEMANTICS = "FIXED_ACTION_CONSEQUENCE"
#: The Student's POLICY RESPONSE to a changed environment is NOT modeled.
POLICY_RESPONSE_MODE = "NOT_MODELED"

#: Reference prediction is NOT available this phase (amendment §3) — there is no
#: same-environment Reference action sequence and no verified Reference
#: policy-in-loop adapter.
PREDICTED_REFERENCE_FAILURE = "NOT_AVAILABLE"
PREDICTED_BEHAVIOR_GAP = "NOT_AVAILABLE"

#: Action registry status (amendment §6): no real Craftax Action enum is
#: installed/read; the documented 43-action map is a compatibility FIXTURE only.
REAL_CRAFTAX_ACTION_REGISTRY_READY = False
DOCUMENTED_COMPATIBILITY_FIXTURE = "DOCUMENTED_COMPATIBILITY_FIXTURE"

#: The entity correspondence matcher (Hungarian) is OFFLINE / non-differentiable
#: (amendment §7): it builds supervision targets and evaluates, but does NOT
#: enter the JAX JIT training graph.
ENTITY_MATCHER_DIFFERENTIABLE = False

# Inherited BA-BAGR audit residual status (amendment §10): this task MUST NOT
# modify BA-BAGR modules and does NOT fix their residual blockers.
INHERITED_BA_BGR_BLOCKERS_PRESENT = True
BA_CWM_FIXES_INHERITED_BLOCKERS = False
REAL_BA_BGR_INTEGRATION_AUTHORIZED = False
REAL_BA_CWM_INTEGRATION_AUTHORIZED = False

#: Independent audit is required before any real use.
INDEPENDENT_AUDIT_REQUIRED = True

# ---------------------------------------------------------------------------
# Source policy (task §七 / §23.1, amendment: strict superset of BA-BAGR).
#
# ALLOWED = BA-BAGR allowed (generative training env + synthetic test trace)
#           PLUS historical-training-replay + UED-training-rollout.
# FORBIDDEN = BA-BAGR forbidden (FORMAL_FRONT/BACK/FULL, FROZEN_BANK,
#             certificate-private-state) PLUS FORMAL_EVALUATOR_TRACE +
#             TEACHER_PRIVATE_EVALUATION_TRACE.
# Anything else is UNKNOWN and fails closed.
# ---------------------------------------------------------------------------
SOURCE_HISTORICAL_TRAINING_REPLAY = "HISTORICAL_TRAINING_REPLAY"
SOURCE_UED_TRAINING_ROLLOUT = "UED_TRAINING_ROLLOUT"
SOURCE_FORMAL_EVALUATOR_TRACE = "FORMAL_EVALUATOR_TRACE"
SOURCE_TEACHER_PRIVATE_EVALUATION_TRACE = "TEACHER_PRIVATE_EVALUATION_TRACE"

CWM_ALLOWED_EVIDENCE_SOURCES = frozenset(set(BAGR.ALLOWED_EVIDENCE_SOURCES) | {
    SOURCE_HISTORICAL_TRAINING_REPLAY,
    SOURCE_UED_TRAINING_ROLLOUT,
})
CWM_FORBIDDEN_EVIDENCE_SOURCES = frozenset(set(BAGR.FORBIDDEN_EVIDENCE_SOURCES) | {
    SOURCE_FORMAL_EVALUATOR_TRACE,
    SOURCE_TEACHER_PRIVATE_EVALUATION_TRACE,
})

# Sanity: allowed and forbidden must be disjoint.
assert not (CWM_ALLOWED_EVIDENCE_SOURCES & CWM_FORBIDDEN_EVIDENCE_SOURCES), \
    "CWM source policy: allowed and forbidden sets overlap"

# ---------------------------------------------------------------------------
# Counterfactual mutation axes (reuse BA-BAGR's legal environment-induction
# axes — environment knobs ONLY, never action / reward / policy knobs).
# ---------------------------------------------------------------------------
MUTATION_AXES = BAGR.MUTATION_AXES

#: Axes whose mutation REQUIRES rebuilding the initial symbolic world state
#: (amendment §2) rather than a dynamics-only intervention.
STATE_REBUILD_AXES = frozenset({
    "threat_distance_grading",      # enemy distance / location
    "threat_count",                 # entity population
    "safe_rest_area_availability",  # safe-rest geometry
    "view_occlusion",               # terrain / occlusion
    "visibility",                   # terrain / visibility geometry
    "resource_pressure",            # resource location / pressure
})

# ---------------------------------------------------------------------------
# Structured world-state schema constants (task §八).
# ---------------------------------------------------------------------------
LOCAL_GRID_SIZE = 9            # 9x9 local terrain grid
MAX_ENTITY_SLOTS = 16          # bounded entity token slots

# ---------------------------------------------------------------------------
# Semantic action vocabulary NAMES (task §八 / §十). Only these semantic classes
# may appear in a SemanticAction; bare action integers are rejected at the schema
# boundary. The integer ids used by the training path are in vocabularies.py.
# ---------------------------------------------------------------------------
SEMANTIC_ACTION_CLASSES = (
    "MOVE",
    "ATTACK",
    "REST",
    "SLEEP",
    "DIG",
    "CRAFT",
    "USE_RESOURCE",
    "INTERACT",
    "NO_OP",
    "UNKNOWN_SEMANTIC",
)

#: Visibility status vocabulary (task §八). missing/None is NEVER silently
#: coerced to VISIBLE / NOT_VISIBLE — it maps to UNKNOWN_OR_OUT_OF_VIEW.
VISIBILITY_CLASSES = (
    "VISIBLE",
    "RECENTLY_VISIBLE",
    "UNKNOWN_OR_OUT_OF_VIEW",
)

#: Aggression status vocabulary.
AGGRESSION_CLASSES = (
    "HOSTILE",
    "NEUTRAL",
    "PASSIVE",
    "UNKNOWN",
)

#: Entity correspondence labels (task §十四, amendment §7). TEMPORARILY_UNOBSERVED
#: is distinct from REMOVED: an entity that is merely out of view is NOT removed.
ENTITY_CORRESPONDENCE_LABELS = (
    "PERSISTED",
    "MOVED",
    "SPAWNED",
    "REMOVED",
    "TEMPORARILY_UNOBSERVED",
)

#: Counterfactual intervention classification (amendment §2).
INTERVENTION_CLASSES = (
    "DYNAMICS_ONLY_INTERVENTION",
    "STATE_REBUILD_REQUIRED",
    "INVALID_INTERVENTION",
)

#: Screening buckets (task §十九).
SCREENING_BUCKETS = (
    "EXPLOITATION",
    "CAUSAL_DISCRIMINATION",
    "UNCERTAINTY_EXPLORATION",
    "GLOBAL_RETENTION",
    "REJECTED",
)

#: Prediction trust labels (task §十三).
TRUST_LABELS = (
    "TRUSTED_PREDICTION",
    "UNCERTAIN_ACTIVE_SAMPLING",
    "REJECTED_DEGENERATE",
    "INVALID_PROVENANCE",
)
