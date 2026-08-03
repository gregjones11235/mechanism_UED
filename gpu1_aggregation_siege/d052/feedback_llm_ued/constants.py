"""Constants + hard authorization state for the feedback-adaptive LLM-UED loop.

Authorization for THIS round: every real-world capability flag is a
compile-time constant the controller re-asserts at startup; the package
REFUSES to run as though any of them were True. The closed loop therefore
executes against a deterministic symbolic probe runner + a deterministic mock
LLM backend, and nothing here silently pretends otherwise.

    TRAINING_AUTHORIZED        = false   (no optimizer step this round)
    FORMAL_EVALUATION_AUTHORIZED = false (no formal FRONT/BACK/FULL run)
    REAL_LLM_CALLS_AUTHORIZED    = false (mock backend only; real_calls == 0)
    REAL_SIMULATOR_PROBE_AUTHORIZED = false (no real Craftax rollout locally)

The last flag encodes the honest local state: there is no JAX/Craftax in this
environment, so probes run on a DETERMINISTIC SYMBOLIC runner. A real-Craftax
adapter seam exists (``simulator_probe.CraftaxPreflightProbeRunner``) but is
BLOCKED until run on the training host; flipping the flag is a director
decision outside this package and is refused here regardless.
"""
from __future__ import annotations

from d052.bagr_ued.hashing import canonical_sha256

# ---------------------------------------------------------------------------
# Authorization state (THIS ROUND). NEVER True inside this package.
# ---------------------------------------------------------------------------
TRAINING_AUTHORIZED = False
FORMAL_EVALUATION_AUTHORIZED = False
REAL_LLM_CALLS_AUTHORIZED = False

#: No real Craftax rollout is available locally (no JAX/Craftax interpreter).
#: Probes run on the deterministic symbolic runner; the real adapter is a seam.
REAL_SIMULATOR_PROBE_AUTHORIZED = False
REAL_SIMULATOR_PROBE_STATUS = "BLOCKED_NO_LOCAL_CRAFTAX"

#: probe seed policies. The symbolic runner is seedless by construction; the
#: real Craftax seam must use explicitly banked seeds (never implicit global
#: RNG state) so every episode is reproducible and auditable.
SEED_POLICY_NONE_SYMBOLIC = "NONE_SYMBOLIC"
SEED_POLICY_JAX_PRNG_SEEDED = "JAX_PRNG_SEEDED"
SEED_POLICIES = frozenset({SEED_POLICY_NONE_SYMBOLIC,
                           SEED_POLICY_JAX_PRNG_SEEDED})

#: The shuffled-mode feedback permutation is FROZEN and recomputable: it is
#: derived ONLY from (mode, board window, SEED_SCHEDULE_HASH) plus the record
#: set — never from runtime randomness. This round is seedless (deterministic
#: symbolic probe + mock backend), so the seed schedule is a fixed declaration
#: rather than a random draw, and its hash is a compile-time constant. Any
#: future real-seed round must replace this with the real frozen schedule hash.
SEED_SCHEDULE_HASH = canonical_sha256(dict(
    schedule="seedless_symbolic_deterministic",
    probe_seed_policy=SEED_POLICY_NONE_SYMBOLIC,
    shuffle_derivation="feedback_llm_ued.permutation.v1",
))

# ---------------------------------------------------------------------------
# Round status flags (director review board). Everything whose truth would
# mean a REAL capability was exercised stays False until the corresponding
# real integration lands; the current implementation is ENGINEERING_SCAFFOLD.
# ---------------------------------------------------------------------------
ENGINEERING_SCAFFOLD = "ENGINEERING_SCAFFOLD"

#: SOTA integration readiness (false until the real minimal closed loop with
#: real Craftax + real Student + real feedback + plan revision passes).
SOTA_INTEGRATION_READY = False
#: A real high-capability Student checkpoint was loaded via the CC4 shared
#: StudentAdapter (false: CC4 contract not present in this worktree yet).
REAL_CHECKPOINT_LOADED = False
#: A real optimizer step on the high-capability Student was executed.
REAL_TRAINING_UPDATE_EXECUTED = False
#: A real LLM EnvCoder generated candidate environment code that compiled.
REAL_ENVCODER_USED = False
#: A real Craftax probe (not the symbolic runner) produced feedback.
REAL_SIMULATOR_PROBE = False
#: The four standard-reset anchors were bound to the cross-direction shared
#: FROZEN manifest (false: no such frozen manifest exists in this worktree).
SHARED_ANCHOR_MANIFEST_BOUND = False
BLOCKED_SHARED_ANCHOR_MANIFEST = "BLOCKED_SHARED_ANCHOR_MANIFEST"

#: engineering-progress flags (may flip True as scaffold pieces land):
#: C16 earned: the direction-two formal plan (six-role Review Board +
#: independent EnvCoder + double-window state machine + three-mode
#: structural isolation + shared Soft Copeland + anchor-manifest seam +
#: hash recomputation + persistence/restore equivalence) is fully
#: implemented and tested — at ENGINEERING_SCAFFOLD evidence level (mock
#: backend + symbolic probe; every REAL_* flag below remains False).
E2_FORMAL_PLAN_ALIGNED = True
#: C6 earned: six-role Review Board implemented (mock rules,
#: ENGINEERING_SCAFFOLD evidence — no real LLM calls).
SIX_ROLE_BOARD_IMPLEMENTED = True
#: C16 earned: plan revisions are bound to Simulator feedback — window k+1's
#: six roles cite window-k feedback ids/hypothesis ids/prediction signatures
#: explicitly, the revision record's label is forced by its citation union,
#: and uncited modifications can only be EXPLORATION (hard validators).
FEEDBACK_REVISION_BOUND = True
#: C8 earned (tightened by the CC3 C9 gate, 2026-08-04): the double-window
#: state machine guarantees a plan revision at window k may cite ONLY
#: feedback from EXACTLY window k-1 — older, current and future records all
#: fail closed as STALE_FEEDBACK_ID (the window-k board's six roles are the
#: sole producers of window-k revisions).
NEXT_WINDOW_REVISION_ONLY = True
#: C8 earned: any attempt to apply a verdict or modify window k's plan after
#: feedback_k is staged raises SAME_WINDOW_REVISION_FORBIDDEN (fail closed);
#: negative tests prove the refusal.
SAME_WINDOW_REVISION_REJECTED = True
#: C9 earned again (CC3 gate, 2026-08-04): the static-no-feedback mode binds
#: the board to the structurally empty NullFeedbackView, and the BoardContext
#: is built ONLY from that view — empty behavior evidence, zero pooled
#: episodes/SR, maximal-uncertainty CI, no candidate ids, no history — never
#: from the raw SimulatorFeedbackStore (assemble_board_context refuses a raw
#: store with BOARD_CONTEXT_STORE_FORBIDDEN). The CC3 gate found the previous
#: store-fed assembly path leaking evidence into the static context; the fix
#: plus the targeted bypass tests (test_feedback_llm_ued_c9_gate.py) earn
#: this flag back.
STATIC_FEEDBACK_STRUCTURALLY_HIDDEN = True
#: C9 earned again (CC3 gate, 2026-08-04): the shuffled permutation is a
#: frozen recomputable anonymization with NO identity side channel — at the
#: prompt layer AND at the BoardContext evidence layer (anonymized evidence
#: ids consistent with the payload, candidate id masked) — and the window lag
#: is EXACTLY one window (rec.window == window-1; older/current/future fail
#: closed as STALE_FEEDBACK_ID). Re-earned by the targeted bypass and lag
#: tests (test_feedback_llm_ued_c9_gate.py).
SHUFFLE_PERMUTATION_FROZEN = True

#: flags that must NEVER be True this round (authorization posture re-asserts)
NEVER_TRUE_REAL_CAPABILITY_FLAGS = (
    "TRAINING_AUTHORIZED",
    "FORMAL_EVALUATION_AUTHORIZED",
    "REAL_LLM_CALLS_AUTHORIZED",
    "REAL_SIMULATOR_PROBE_AUTHORIZED",
    "SOTA_INTEGRATION_READY",
    "REAL_CHECKPOINT_LOADED",
    "REAL_TRAINING_UPDATE_EXECUTED",
    "REAL_ENVCODER_USED",
    "REAL_SIMULATOR_PROBE",
)

# ---------------------------------------------------------------------------
# Loop + role identity
# ---------------------------------------------------------------------------
FEEDBACK_LOOP_VERSION = "feedback_llm_ued.loop.v1"
MOCK_BACKEND_ID = "mock.feedback_llm_ued.deterministic.v1"
MOCK_MODEL_ID = "deterministic-rule-synth.v1"
REPLAY_BACKEND_ID = "replay.feedback_llm_ued.v1"
REPLAY_MODEL_ID = "replayed-corpus.v1"

#: backend kinds (P0-1 abstraction; the launch gate decides which are allowed)
BACKEND_KIND_MOCK = "mock"
BACKEND_KIND_REPLAY = "replay"
BACKEND_KIND_REAL = "real"
BACKEND_KINDS = frozenset({BACKEND_KIND_MOCK, BACKEND_KIND_REPLAY,
                           BACKEND_KIND_REAL})
ROLE_PROMPT_VERSION = "feedback_llm_ued.roles.v1"
RECONCILE_RULE_VERSION = "feedback_llm_ued.reconcile.v1"

# ---------------------------------------------------------------------------
# Six-role Review Board (director-approved formal architecture). Every review
# window runs ALL SIX roles unconditionally — the legacy
# Diagnostician+Designer+conditional-Reviewer 2/3-call pattern was abolished
# and its modules removed in C8.
# ---------------------------------------------------------------------------
ROLE_STUDENT_MODELER = "student_modeler"
ROLE_BEHAVIOR_AUDITOR = "behavior_auditor"
ROLE_CAUSAL_FAILURE_ANALYST = "causal_failure_analyst"
ROLE_INTERVENTION_TUTOR = "intervention_tutor"
ROLE_EXPLORER = "explorer"
ROLE_CRITIC_SKEPTIC = "critic_skeptic"
BOARD_ROLES = (
    ROLE_STUDENT_MODELER,
    ROLE_BEHAVIOR_AUDITOR,
    ROLE_CAUSAL_FAILURE_ANALYST,
    ROLE_INTERVENTION_TUTOR,
    ROLE_EXPLORER,
    ROLE_CRITIC_SKEPTIC,
)
BOARD_CALLS_PER_WINDOW = len(BOARD_ROLES)   # 6 LLM-family calls, every window

#: the independent EnvCoder is the 7th LLM-family call of every window: it
#: consumes the board's AxisDirectives and emits candidate environment code
#: (symbolic this round) which the compile/reset/step gates then check.
ROLE_ENV_CODER = "env_coder"
LLM_CALLS_PER_WINDOW = BOARD_CALLS_PER_WINDOW + 1     # 6 board + 1 EnvCoder

# ---------------------------------------------------------------------------
# Hypothesis lifecycle (task: ledger statuses). A hypothesis is a claim about
# the Student's behavior that an environment family is meant to test.
# ---------------------------------------------------------------------------
HYPOTHESIS_SUPPORTED = "SUPPORTED"
HYPOTHESIS_REFUTED = "REFUTED"
HYPOTHESIS_INCONCLUSIVE = "INCONCLUSIVE"
HYPOTHESIS_STALE = "STALE"
HYPOTHESIS_PENDING = "PENDING"              # not yet probed this line
HYPOTHESIS_STATUSES = frozenset({
    HYPOTHESIS_SUPPORTED, HYPOTHESIS_REFUTED, HYPOTHESIS_INCONCLUSIVE,
    HYPOTHESIS_STALE, HYPOTHESIS_PENDING,
})
#: statuses that count as a terminal verdict for retention/retirement metrics
HYPOTHESIS_TERMINAL_VERDICTS = frozenset({HYPOTHESIS_SUPPORTED,
                                          HYPOTHESIS_REFUTED})

# ---------------------------------------------------------------------------
# Curriculum decision vocabulary (task §3). Environment-level actions ONLY —
# never an action/reward/policy knob. The board's InterventionTutor proposes
# them; the DeterministicReconciler disposes.
# ---------------------------------------------------------------------------
DECISION_RETAIN = "RETAIN"
DECISION_MUTATE = "MUTATE"
DECISION_RETIRE = "RETIRE"
DECISION_EXPAND_BUDGET = "EXPAND_BUDGET"
DECISION_REDUCE_BUDGET = "REDUCE_BUDGET"
DECISION_REQUEST_CONTROL = "REQUEST_CONTROL"
DESIGNER_DECISIONS = frozenset({
    DECISION_RETAIN, DECISION_MUTATE, DECISION_RETIRE,
    DECISION_EXPAND_BUDGET, DECISION_REDUCE_BUDGET, DECISION_REQUEST_CONTROL,
})

#: a plan modification with no cited feedback id may ONLY be one of these
EXPLORATION_DECISIONS = frozenset({DECISION_MUTATE, DECISION_EXPAND_BUDGET})
EXPLORATION_LABEL = "EXPLORATION"

#: C11: window-level revision label for a window STOPPED by REQUEST_CONTROL.
#: No PlanRevisionRecord exists for such a window (no verdicts are applied,
#: no plan is produced) — the label only documents the halt in the
#: WindowRecord / RunSummary audit trail.
REVISION_LABEL_REQUEST_CONTROL_STOPPED = "REQUEST_CONTROL_STOPPED"

#: C10 RETIRE lifecycle: a family retired at window w is in COOLDOWN for the
#: next RETIRE_COOLDOWN_WINDOWS windows (w+1 .. w+RETIRE_COOLDOWN_WINDOWS);
#: any proposal of any decision targeting it during cooldown fails closed
#: (FAMILY_IN_COOLDOWN). Past the cooldown the family STAYS retired until
#: explicitly reopened — either human authorization (human_reopen_families)
#: or ALL distinguishing probe evidence postdating the retirement window.
#: A STALE verdict can therefore never resurrect a retired family.
RETIRE_COOLDOWN_WINDOWS = 3

# ---------------------------------------------------------------------------
# Three comparison modes (task §5)
# ---------------------------------------------------------------------------
MODE_STATIC_LLM = "static_llm"
MODE_NORMAL_FEEDBACK = "normal_feedback"
MODE_SHUFFLED_FEEDBACK = "shuffled_feedback"
FEEDBACK_MODES = frozenset({MODE_STATIC_LLM, MODE_NORMAL_FEEDBACK,
                            MODE_SHUFFLED_FEEDBACK})

# ---------------------------------------------------------------------------
# Staged probe funnel (task §4)
# ---------------------------------------------------------------------------
RAW_CANDIDATES = 64                    # Stage-0 generated candidates
STAGE1_KEEP = 24                       # fast probe keeps ~24
STAGE2_KEEP = 12                       # full probe keeps ~12 dynamic
DYNAMIC_UED_SLOTS = 12
GLOBAL_ANCHOR_SLOTS = 4
FINAL_BATCH = DYNAMIC_UED_SLOTS + GLOBAL_ANCHOR_SLOTS      # 12 + 4 = 16

#: C12: run-level front/global regret weighting for the Stage-2 Soft
#: Copeland (consumed by d052.bagr_ued.soft_copeland). Structurally < 1 so
#: the global pairwise weight (1 - alpha) stays strictly positive.
ALPHA_FRONT_STAGE2 = 0.5
#: C12: per-family greedy diversity penalty applied to the Copeland score
#: when picking the 12 dynamic slots
STAGE2_FAMILY_PENALTY = 0.10

STAGE1_STUDENT_EPISODES = 2
STAGE1_REFERENCE_EPISODES = 1
STAGE2_STUDENT_EPISODES_MIN = 4
STAGE2_STUDENT_EPISODES_MAX = 8
STAGE2_REFERENCE_EPISODES_MIN = 2
STAGE2_REFERENCE_EPISODES_MAX = 4

ROLLOUT_LENGTH = 128                   # transitions per episode (symbolic)

#: preflight routing thresholds (selectively reused from skill_preflight.route)
PREFLIGHT_LEARNABLE_LOW = 0.05
PREFLIGHT_TOO_EASY = 0.85

# ---------------------------------------------------------------------------
# Expected-vs-observed comparison
# ---------------------------------------------------------------------------
MATCH_DIRECTION_AGREE = "agree"
MATCH_DIRECTION_OPPOSITE = "opposite"
MATCH_DIRECTION_NEUTRAL = "neutral"
#: relative gap above which a predicted-vs-observed metric counts as a mismatch
COMPARATOR_RELATIVE_TOLERANCE = 0.25

# ---------------------------------------------------------------------------
# Formal evaluation isolation (task §6). The formal data domains may NEVER
# enter the ledger, the LLM roles, the generator, the selector, or the
# optimizer. Training + candidate probes use a SEPARATE source enum.
# ---------------------------------------------------------------------------
SOURCE_GENERATIVE_TRAINING_ENV = "GENERATIVE_TRAINING_ENV"
SOURCE_CANDIDATE_PROBE = "CANDIDATE_PROBE"          # independent probe source
SOURCE_SYNTHETIC_TEST_TRACE = "SYNTHETIC_TEST_TRACE"
SOURCE_FORMAL_FRONT = "FORMAL_FRONT"
SOURCE_FORMAL_BACK = "FORMAL_BACK"
SOURCE_FORMAL_FULL = "FORMAL_FULL"

ALLOWED_LOOP_SOURCES = frozenset({
    SOURCE_GENERATIVE_TRAINING_ENV,
    SOURCE_CANDIDATE_PROBE,
    SOURCE_SYNTHETIC_TEST_TRACE,
})
FORMAL_FORBIDDEN_SOURCES = frozenset({
    SOURCE_FORMAL_FRONT, SOURCE_FORMAL_BACK, SOURCE_FORMAL_FULL,
})

# ---------------------------------------------------------------------------
# Reference-output restriction (task §4). Reference probe results may expose
# ONLY coarse episode-level statistics to the Student / LLM. These forbidden
# carriers must never cross into Student supervision or an LLM prompt.
# ---------------------------------------------------------------------------
REFERENCE_ALLOWED_FIELDS = frozenset({
    "episode_success_rate", "mean_progress", "achievement_count",
    "behavior_activation_rate", "mean_episode_length",
})
REFERENCE_FORBIDDEN_CARRIERS = frozenset({
    "action_sequence", "trajectory", "waypoints", "hidden_state", "logits",
    "expert_action_sequence", "policy_logits", "state_trajectory",
})

# ---------------------------------------------------------------------------
# Environment-level TaskParams vocabulary (method-specific; deliberately
# environment-induction knobs ONLY — no action/reward/policy knob).
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

ENVIRONMENT_FAMILIES = (
    "threat_distance_family",
    "resource_pressure_family",
    "day_night_rest_need_family",
    "visibility_family",
    "multi_threat_interference_family",
    "long_term_memory_family",
    "global_task_conflict_family",
)

GLOBAL_CANONICAL_ANCHOR_IDS = (
    "GLOBAL_ANCHOR_EARLY_SURVIVAL",
    "GLOBAL_ANCHOR_RESOURCE_CHAIN",
    "GLOBAL_ANCHOR_THREAT_ENGAGEMENT",
    "GLOBAL_ANCHOR_LONG_HORIZON_PLANNING",
)

# ---------------------------------------------------------------------------
# Strong Student identity (director: fixed candidate). Direction two only
# CONSUMES the CC4 shared StudentAdapter/StudentInitContract — it never builds
# a second loader/registry/codec. The vocabulary below is the direction's own
# bookkeeping; candidate id reused from
# experiments/henry_dicode_student_upgrade/student_candidate_registry_v1.json.
# ---------------------------------------------------------------------------
STRONG_STUDENT_CANDIDATE_ID = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"

#: how a Student checkpoint is used inside a window (recorded per feedback)
STUDENT_ROLE_CAPTURE = "capture"
STUDENT_ROLE_SEARCH = "search"
STUDENT_ROLE_TRAIN = "train"
STUDENT_ROLES = frozenset({STUDENT_ROLE_CAPTURE, STUDENT_ROLE_SEARCH,
                           STUDENT_ROLE_TRAIN})

#: memory-compatibility status stamped on feedback records while no real
#: Student weights / memory state exists locally (CC4 adapter absent).
MEMORY_COMPATIBILITY_NOT_APPLICABLE = "NOT_APPLICABLE_LOCAL"

#: candidate descriptor field whitelist (legality gate; mock-namespaced —
#: the real TaskParams adapter is BLOCKED_EXTERNAL_DEPENDENCY and its real
#: field names MUST NOT be guessed).
MOCK_TASKPARAMS_FIELD_WHITELIST = frozenset({
    "protocol_version",
    "candidate_id",
    "candidate_hash",
    "environment_family",
    "axis_values",
    "held_constant_axes",
    "variant_id",
    "variant_kind",
    "mutation_axes",
    "distinguishes_hypothesis_ids",
    "provenance",
    "real_adapter_status",
    "legality_hint",
})

# ---------------------------------------------------------------------------
# Output caps (精简版 spec, carried forward): keep LLM outputs bounded.
# ---------------------------------------------------------------------------
MAX_DIAGNOSED_HYPOTHESES_PER_WINDOW = 8
MAX_WEAKNESSES = 3
MAX_HYPOTHESES = 6
MAX_INTERVENTIONS = 8
MAX_EXPLORATION_PROPOSALS = 2
MAX_AXES_PER_INTERVENTION = 3

MAX_WINDOWS = 8                        # bounded loop horizon for dry runs
