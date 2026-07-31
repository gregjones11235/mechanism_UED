"""BA-BAGR-UED (D052-v2 Behavior-Aware Bottleneck-Aware Global Regret UED) constants.

Hard authorization state for THIS round. Every flag below is a compile-time
constant the controller re-asserts at startup; the package REFUSES to run if any
caller tries to operate as though a flag were True. Flipping any of them is a
director decision OUTSIDE this package (and is refused here regardless):

    TRAINING_AUTHORIZED            = false   (no real training this round)
    FORMAL_EVALUATION_AUTHORIZED   = false   (no formal evaluation this round)
    REAL_LLM_CALLS_AUTHORIZED      = false   (mock backend only; real_calls == 0)
    REAL_TASKPARAMS_ADAPTER        = BLOCKED_EXTERNAL_DEPENDENCY (mock adapter only)

Two critic-policy canonical rules are deliberately NOT frozen this round; both
stay PENDING for a director decision before any real run:

    REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE = PENDING
    REAL_CANONICAL_CRITIC_SELECTION_POLICY       = PENDING
"""
from __future__ import annotations

BA_BAGR_UED_VERSION = "d052_v2.bagr_ued.v1"

# ---------------------------------------------------------------------------
# Authorization state (THIS ROUND). NEVER True in this package.
# ---------------------------------------------------------------------------
TRAINING_AUTHORIZED = False
FORMAL_EVALUATION_AUTHORIZED = False
REAL_LLM_CALLS_AUTHORIZED = False

#: The real Global TaskParams adapter is an external dependency CC3 does not own;
#: architecture tests use a mock adapter and MUST NOT guess real field names.
REAL_TASKPARAMS_ADAPTER = "BLOCKED_EXTERNAL_DEPENDENCY"

#: Both canonical critic rules remain PENDING (task: must NOT be secretly frozen).
REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE = "PENDING"
REAL_CANONICAL_CRITIC_SELECTION_POLICY = "PENDING"

# ---------------------------------------------------------------------------
# Global-not-Tier3-only scope (task section 13).
# ---------------------------------------------------------------------------
TRAINING_SCOPE = "GLOBAL"
TIER3_ONLY_TRAINING = False
GLOBAL_SIGNAL_REQUIRED = True
GLOBAL_UED_SLOTS_MINIMUM = 1

# ---------------------------------------------------------------------------
# Soft Copeland alpha_front structural bounds (CC1 audit fix1, task §8).
# alpha_front MUST be structurally < 1 so the global-regret component
# (1 - alpha_front) is ALWAYS strictly positive — a global-scope method can
# never degenerate into a front-only scorer.
# ---------------------------------------------------------------------------
ALPHA_FRONT_MIN = 0.0
ALPHA_FRONT_MAX = 0.75

# ---------------------------------------------------------------------------
# 2048-transition dry-run plan (task section 14).
# ---------------------------------------------------------------------------
NUM_ENVS = 16
ROLLOUT_LENGTH = 128
TRANSITIONS_PER_UPDATE = 2048          # NUM_ENVS * ROLLOUT_LENGTH
UED_ACTIVE_SLOTS = 12
GLOBAL_CANONICAL_ANCHORS = 4
REVIEW_INTERVAL_UPDATES = 4
REVIEW_INTERVAL_TRANSITIONS = REVIEW_INTERVAL_UPDATES * TRANSITIONS_PER_UPDATE  # 8192

# ---------------------------------------------------------------------------
# Trajectory evidence policy (task section 3): allowed vs forbidden sources.
# ---------------------------------------------------------------------------
SOURCE_GENERATIVE_TRAINING_ENV = "GENERATIVE_TRAINING_ENV"
SOURCE_SYNTHETIC_TEST_TRACE = "SYNTHETIC_TEST_TRACE"
SOURCE_FORMAL_FRONT = "FORMAL_FRONT"
SOURCE_FORMAL_BACK = "FORMAL_BACK"
SOURCE_FORMAL_FULL = "FORMAL_FULL"
SOURCE_FROZEN_BANK = "FROZEN_BANK"
SOURCE_FORMAL_CERT_PRIVATE_STATE = "FORMAL_EVALUATION_CERTIFICATE_PRIVATE_STATE"

ALLOWED_EVIDENCE_SOURCES = frozenset({
    SOURCE_GENERATIVE_TRAINING_ENV,
    SOURCE_SYNTHETIC_TEST_TRACE,
})
FORBIDDEN_EVIDENCE_SOURCES = frozenset({
    SOURCE_FORMAL_FRONT,
    SOURCE_FORMAL_BACK,
    SOURCE_FORMAL_FULL,
    SOURCE_FROZEN_BANK,
    SOURCE_FORMAL_CERT_PRIVATE_STATE,
})

#: Guard A (TrajectorySupervisionGuard): keys that, if present in ANY output,
#: turn read-only trajectory evidence into forbidden Student supervision.
FORBIDDEN_SUPERVISION_KEYS = frozenset({
    "recommended_actions",
    "action_sequence_to_follow",
    "waypoints",
    "expert_demonstration",
    "policy_override",
    "hidden_state_override",
    "reward_delta",
    "reward_shaping",
})

#: Guard A alias hardening (CC1 audit fix1, task §6): renames / smuggled
#: spellings of the same forbidden concepts. Matched AFTER normalization
#: (casefold + separator stripping) on EVERY key of every nested mapping.
FORBIDDEN_SUPERVISION_KEY_ALIASES = frozenset({
    # action-advice key aliases
    "suggested_action",
    "suggested_actions",
    "recommended_action",
    "recommended_actions",
    "recommended_move",
    "recommended_policy",
    "route",
    "navigation_route",
    "path_to_follow",
    "expert_plan",
    # formal-state / bank payload aliases (also mirrored into Guard B)
    "bank_blob",
    "formal_state_blob",
    "formal_state_payload",
    "state_payload",
    # reward-shaping aliases (reward_delta / reward_shaping covered above)
})

#: Guard A serialized-string parsing limits (CC1 audit fix1, task §5): a
#: string that looks like JSON is parsed and RE-SCANNED under these bounds;
#: exceeding any bound is a fail-closed finding (never a lenient skip).
MAX_SERIALIZED_PARSE_DEPTH = 12
MAX_SERIALIZED_STRING_LENGTH = 65536
MAX_SERIALIZED_CONTAINER_ITEMS = 4096

# ---------------------------------------------------------------------------
# Review board identity (mock backend; prompt versions are pinned, not frozen
# canonical rules — the two REAL_CANONICAL_* rules above stay PENDING).
# ---------------------------------------------------------------------------
MOCK_BACKEND_ID = "mock.bagr_ued.deterministic.v1"
MOCK_MODEL_ID = "deterministic-rule-synth.v1"
ROLE_PROMPT_VERSION = "bagr_ued.roles.v1"
RECONCILIATION_RULE_VERSION = "bagr_ued.reconcile.v1"

ROLE_STUDENT_MODELER = "student_modeler"
ROLE_BEHAVIOR_AUDITOR = "behavior_auditor"
ROLE_CAUSAL_FAILURE_ANALYST = "causal_failure_analyst"
ROLE_INTERVENTION_TUTOR = "intervention_tutor"
ROLE_EXPLORER = "explorer"
ROLE_CRITIC_SKEPTIC = "critic_skeptic"

REVIEW_BOARD_ROLES = (
    ROLE_STUDENT_MODELER,
    ROLE_BEHAVIOR_AUDITOR,
    ROLE_CAUSAL_FAILURE_ANALYST,
    ROLE_INTERVENTION_TUTOR,
    ROLE_EXPLORER,
    ROLE_CRITIC_SKEPTIC,
)

# ---------------------------------------------------------------------------
# Causal vocabulary (task section 6): the ONLY allowed cause categories.
# ---------------------------------------------------------------------------
CAUSE_CATEGORIES = (
    "perception_or_observability",
    "memory_or_context_retention",
    "value_or_risk_misestimation",
    "resource_planning_failure",
    "exploration_noise",
    "action_semantics_confusion",
    "distribution_shift",
    "environment_ambiguity",
    "implementation_or_adapter_bug",
    "unknown",
)

# ---------------------------------------------------------------------------
# Legal mutation axes (task sections 7/11): environment-induction knobs ONLY.
# No action, reward, or policy knob may ever appear here.
# ---------------------------------------------------------------------------
MUTATION_AXES = (
    "threat_distance_grading",
    "safe_rest_area_availability",
    "rest_need_pressure",
    "threat_count",
    "view_occlusion",
    "resource_pressure",
    "day_night_rest_need",
    "visibility",
    "multi_threat_interference",
    "long_term_memory_requirement",
    "global_task_conflict",
)

# ---------------------------------------------------------------------------
# Global environment families (task sections 8/13): deliberately broader than
# the floor2->floor3 threat axis; the board MUST NOT collapse onto Tier3-only.
# ---------------------------------------------------------------------------
ENVIRONMENT_FAMILIES = (
    "threat_distance_family",
    "resource_pressure_family",
    "day_night_rest_need_family",
    "visibility_family",
    "multi_threat_interference_family",
    "long_term_memory_family",
    "global_task_conflict_family",
)

# ---------------------------------------------------------------------------
# Global canonical anchors (task sections 13/14): 4 fixed GLOBAL anchors the
# budget allocator always reserves. Synthetic ids this round — the REAL
# canonical pool is out of scope (dry run; no real pool access).
# ---------------------------------------------------------------------------
GLOBAL_CANONICAL_ANCHOR_IDS = (
    "GLOBAL_ANCHOR_EARLY_SURVIVAL",
    "GLOBAL_ANCHOR_RESOURCE_CHAIN",
    "GLOBAL_ANCHOR_THREAT_ENGAGEMENT",
    "GLOBAL_ANCHOR_LONG_HORIZON_PLANNING",
)

#: Mock TaskParams adapter field whitelist (legality gate). Real fields are
#: UNKNOWN (REAL_TASKPARAMS_ADAPTER=BLOCKED_EXTERNAL_DEPENDENCY) and MUST NOT
#: be guessed; every descriptor field is mock-namespaced. The ONE non-mock
#: entry, protocol_version, is the d052 canonical_v2 framework identity field
#: inherited from CanonicalModel — framework provenance, not a guessed real
#: environment field.
MOCK_TASKPARAMS_FIELD_WHITELIST = frozenset({
    "protocol_version",
    "descriptor_id",
    "descriptor_hash",
    "mock_env_family",
    "mock_axis_values",
    "mock_control_values",
    "mock_variant_id",
    "mock_variant_kind",
    "mutation_axes",
    "distinguishes_hypothesis_ids",
    "provenance",
    "real_adapter_status",
    "legality_hint",
})
