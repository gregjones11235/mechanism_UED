"""Frozen canonical_v2 configuration constants.

These values are FIXED for protocol_version == canonical_v2. They are NOT
overridable by a run config: a canonical_v2 run that disagrees with any of them
is a configuration error, not a variant. The source-of-truth evidence for each
constant is recorded in
audit_outputs/d052_legacy_source_freeze_20260726T060626Z/source_inventory.json
and reports/d052_legacy_source_freeze.md.

Evidence anchors (baseline a2726e3):
  * 67 achievements   -> dicode_src/auction/craftax_achievements.py (blob 5bb881a6)
  * multi-hot dim 67  -> dicode_src/src/dicode/task_utils.py (blob 1df5caf3)
  * obs_dim 8335      -> dicode_src/src/minicraftax/envs/multitask.py (blob 8c38bc5c)
                         + run_p9_authentic_98304.py assert obs_dim==8335 & EMB==67
  * eval mapping      -> dicode_src/src/dicode/evaluation/online_evaluation.py:229
                         Achievement[name.upper()].value
"""
from __future__ import annotations

from typing import Any, Dict

# --- protocol versioning ---------------------------------------------------
CANONICAL_PROTOCOL_VERSION = "canonical_v2"
LEGACY_PROTOCOL_VERSION = "legacy"

#: The ONLY two legal values for a config's protocol_version field.
LEGAL_PROTOCOL_VERSIONS = frozenset({LEGACY_PROTOCOL_VERSION, CANONICAL_PROTOCOL_VERSION})

# --- achievement / goal conditioning (FROZEN) ------------------------------
ACHIEVEMENT_SCHEMA = "craftax_67_v1"
NUM_ACHIEVEMENTS = 67
MAX_ACHIEVEMENT_VALUE = 66  # 0..66 inclusive

#: canonical_id == goal_vector_index == craftax Achievement enum .value.
#: CONFIRMED identical (not separate) via task_utils.py embedding[ach.value]=1.0.
CONDITIONING_TYPE = "achievement_multi_hot"
CONDITIONING_DIMENSION = 67  # == NUM_ACHIEVEMENTS

#: Student observation dimensionality: base symbolic (8268) + 67 multi-hot = 8335.
#: CONFIRMED by in-repo assert obs_dim==8335 & EMB==67. NOTE: the legacy
#: "one-hot 32-slot / obs_dim 8300" interface is NOT present in current HEAD and
#: is explicitly banned; the +9 bonus scalars inside the 8268 base are not fully
#: verifiable in-repo (craftax not installed) -- see known-limitations report.
STUDENT_OBS_DIM = 8335
BASE_OBS_DIM = STUDENT_OBS_DIM - CONDITIONING_DIMENSION  # 8268

# --- candidate pool (FROZEN) -----------------------------------------------
CANDIDATE_POOL_MODE = "shared_frozen"

# --- error policies (FROZEN; all hard-fail, never silent) ------------------
UNKNOWN_TARGET_POLICY = "error"   # illegal/unknown achievement name -> raise
EMPTY_GOAL_POLICY = "error"       # empty target set -> raise
FALLBACK_POLICY = "error"         # no default-goal / provider / k-reduction fallback

# --- score normalization (FROZEN) ------------------------------------------
SCORE_NORMALIZATION = "rank_percentile_v1"  # per-role, output in [0,1], ties deterministic


def _frozen_config() -> Dict[str, Any]:
    return {
        "protocol_version": CANONICAL_PROTOCOL_VERSION,
        "achievement_schema": ACHIEVEMENT_SCHEMA,
        "num_achievements": NUM_ACHIEVEMENTS,
        "max_achievement_value": MAX_ACHIEVEMENT_VALUE,
        "conditioning_type": CONDITIONING_TYPE,
        "conditioning_dimension": CONDITIONING_DIMENSION,
        "student_obs_dim": STUDENT_OBS_DIM,
        "base_obs_dim": BASE_OBS_DIM,
        "candidate_pool_mode": CANDIDATE_POOL_MODE,
        "unknown_target_policy": UNKNOWN_TARGET_POLICY,
        "empty_goal_policy": EMPTY_GOAL_POLICY,
        "fallback_policy": FALLBACK_POLICY,
        "score_normalization": SCORE_NORMALIZATION,
    }


#: Immutable canonical_v2 fixed config. Treat as read-only.
CANONICAL_V2_FIXED_CONFIG: Dict[str, Any] = _frozen_config()


def canonical_v2_config() -> Dict[str, Any]:
    """Return a fresh copy of the frozen canonical_v2 fixed config.

    A copy is returned so callers cannot mutate the module-level constant.
    """
    return _frozen_config()


def assert_canonical_invariants() -> None:
    """Internal self-check that the frozen constants are mutually consistent.

    Called at import time of the schemas/achievements layers and by GATE tests.
    Raises AssertionError if any invariant is violated.
    """
    assert NUM_ACHIEVEMENTS == 67, NUM_ACHIEVEMENTS
    assert MAX_ACHIEVEMENT_VALUE == NUM_ACHIEVEMENTS - 1, MAX_ACHIEVEMENT_VALUE
    assert CONDITIONING_DIMENSION == NUM_ACHIEVEMENTS, CONDITIONING_DIMENSION
    assert STUDENT_OBS_DIM == BASE_OBS_DIM + CONDITIONING_DIMENSION, (
        STUDENT_OBS_DIM, BASE_OBS_DIM, CONDITIONING_DIMENSION)
    assert CONDITIONING_TYPE == "achievement_multi_hot", CONDITIONING_TYPE
    assert CANDIDATE_POOL_MODE == "shared_frozen", CANDIDATE_POOL_MODE
    assert UNKNOWN_TARGET_POLICY == EMPTY_GOAL_POLICY == FALLBACK_POLICY == "error"
    assert SCORE_NORMALIZATION == "rank_percentile_v1", SCORE_NORMALIZATION


# Fail fast if this module is ever edited into an inconsistent state.
assert_canonical_invariants()
