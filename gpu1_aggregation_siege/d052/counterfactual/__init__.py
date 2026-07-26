"""canonical_v2 Phase-2.5 counterfactual package.

A matched B/C counterfactual protocol for D052 selection, built entirely on the
canonical_v2 framework (official-67 targets, shared frozen pool, deterministic
selectors, execution-mapping certificates, per-cell authorization). Capabilities:

  * firewall              -- salted / hash-modulo / unknown / empty target guard
                             with SPECIFIC codes (gate 3); banned schemes fail closed.
  * prompts               -- canonical B/C prompt specs + deterministic prompt_hash
                             + the SHARED role-judgment prompt hash.
  * student_modeler_channel -- StudentProfile -> Modeler context (SR series only;
                             tier labels stripped -- the modeler firewall) + the
                             deterministic modeler_bonus.
  * judgment_cache        -- content-addressed judgment read + deterministic replay
                             (gate 2: bit-identical selection_hash).
  * protocol              -- CounterfactualArm + verify_matched_bc strict-match
                             verifier (gate 1: identical except modeler conditioning).
  * ablation              -- the canonical modeler OFF/ON role-ablation arms.
  * manifest              -- MatchedCounterfactualManifest binding gates 1-4 + 8.
  * pipeline              -- offline B=S1 / C=S2 harness (zero training).

This phase performs NO training (D052_LONG_TRAINING_RUNS=0).
"""
from d052.counterfactual.ablation import (
    ABLATION_SCORING_ROLES,
    modeler_ablation_arms,
)
from d052.counterfactual.firewall import (
    TargetFirewallError,
    assert_execution_mapping_rejects,
    assert_target_firewall,
    classify_target,
)
from d052.counterfactual.judgment_cache import (
    JudgmentCache,
    JudgmentCacheError,
    cache_key_hash,
)
from d052.counterfactual.manifest import (
    MatchedCounterfactualManifest,
    build_manifest,
    selection_change,
)
from d052.counterfactual.pipeline import (
    PHASE25_K,
    PHASE25_POOL_ID,
    PHASE25_SEED,
    Phase25Result,
    build_phase25_cellspecs,
    build_phase25_judgments,
    build_phase25_modeler_judgment,
    build_phase25_pool,
    build_phase25_student_profile,
    compute_phase25,
    emit_phase25_artifacts,
    register_phase25_cells,
)
from d052.counterfactual.prompts import (
    PROMPT_CONTRACT_VERSION,
    PromptSet,
    PromptSpec,
    build_prompt_set,
    compute_prompt_set_hash,
    role_judgment_prompt_hash,
)
from d052.counterfactual.protocol import (
    CounterfactualArm,
    CounterfactualProtocolError,
    MatchedVerification,
    verify_matched_bc,
)
from d052.counterfactual.student_modeler_channel import (
    MODELER_BONUS_WEIGHT,
    ModelerContext,
    ModelerFirewallError,
    assert_modeler_firewall,
    build_modeler_context,
    modeler_bonus_for,
    student_profile_hash,
)

__all__ = [
    "ABLATION_SCORING_ROLES", "modeler_ablation_arms",
    "TargetFirewallError", "assert_execution_mapping_rejects",
    "assert_target_firewall", "classify_target",
    "JudgmentCache", "JudgmentCacheError", "cache_key_hash",
    "MatchedCounterfactualManifest", "build_manifest", "selection_change",
    "PHASE25_K", "PHASE25_POOL_ID", "PHASE25_SEED", "Phase25Result",
    "build_phase25_cellspecs", "build_phase25_judgments",
    "build_phase25_modeler_judgment", "build_phase25_pool",
    "build_phase25_student_profile", "compute_phase25",
    "emit_phase25_artifacts", "register_phase25_cells",
    "PROMPT_CONTRACT_VERSION", "PromptSet", "PromptSpec", "build_prompt_set",
    "compute_prompt_set_hash", "role_judgment_prompt_hash",
    "CounterfactualArm", "CounterfactualProtocolError", "MatchedVerification",
    "verify_matched_bc",
    "MODELER_BONUS_WEIGHT", "ModelerContext", "ModelerFirewallError",
    "assert_modeler_firewall", "build_modeler_context", "modeler_bonus_for",
    "student_profile_hash",
]
