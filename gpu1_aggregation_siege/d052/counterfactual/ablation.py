"""Role-ablation protocol (Phase 2.5).

Task §角色消融协议. An ablation holds everything constant and varies ONE role
condition, producing a pair of matched arms whose selection delta is attributable
to that role alone. The canonical Phase-2.5 contrast ablates the MODELER -- the
only non-scoring, conditioning role:

    arm B = S1_THREE_ROLE          (modeler OFF: tutor/critic/explorer only)
    arm C = S2_FOUR_ROLE_MODELER   (modeler ON : + deterministic modeler_bonus)

Because S2's composite is exactly S1's composite PLUS modeler_bonus (see
selectors/baseline.py), and the bonus is 0 for every candidate when the modeler is
off, the B->C selection delta is precisely the modeler's causal contribution -- a
clean matched counterfactual. ``modeler_ablation_arms`` builds the pair with every
shared field identical (pool, judgments, k, seed, critic policy, roles, weight) so
``verify_matched_bc`` passes.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from d052.counterfactual.protocol import CounterfactualArm
from d052.counterfactual.prompts import PromptSet
from d052.counterfactual.student_modeler_channel import MODELER_BONUS_WEIGHT
from d052.schemas.roles import ScoringRole
from d052.schemas.selector import CriticPolicy, SelectorConfig, SelectorType

#: the canonical scoring roles held constant across the modeler ablation
ABLATION_SCORING_ROLES: List[ScoringRole] = [
    ScoringRole.TUTOR, ScoringRole.CRITIC, ScoringRole.EXPLORER,
]


def modeler_ablation_arms(
    *,
    pool_hash: str,
    judgment_cache_hash: str,
    prompt_set_b: PromptSet,
    prompt_set_c: PromptSet,
    k: int,
    seed: int,
    critic_policy: CriticPolicy = CriticPolicy.HARD_VETO,
    roles: Optional[List[ScoringRole]] = None,
    modeler_bonus_weight: float = MODELER_BONUS_WEIGHT,
    student_profile_hash: Optional[str] = None,
    modeler_context_hash: Optional[str] = None,
) -> Tuple[CounterfactualArm, CounterfactualArm]:
    """Build the matched (B, C) modeler-ablation arms.

    B = S1_THREE_ROLE, modeler OFF. C = S2_FOUR_ROLE_MODELER, modeler ON. Every
    shared selector field (k/seed/critic_policy/roles/budget) and every shared
    binding (pool_hash, judgment_cache_hash, modeler_bonus_weight) is identical;
    only the modeler conditioning differs.
    """
    use_roles = list(roles) if roles else list(ABLATION_SCORING_ROLES)
    if prompt_set_b.arm != "B" or prompt_set_c.arm != "C":
        raise ValueError("ARMS_MISLABELED: prompt_set_b must be arm B, prompt_set_c arm C")

    selector_b = SelectorConfig(
        selector=SelectorType.S1_THREE_ROLE, critic_policy=critic_policy,
        k=k, seed=seed, roles=use_roles)
    selector_c = SelectorConfig(
        selector=SelectorType.S2_FOUR_ROLE_MODELER, critic_policy=critic_policy,
        k=k, seed=seed, roles=use_roles)

    arm_b = CounterfactualArm(
        arm_label="B", selector=selector_b, pool_hash=pool_hash,
        judgment_cache_hash=judgment_cache_hash,
        prompt_set_hash=prompt_set_b.prompt_set_hash,
        modeler_enabled=False, student_profile_hash=None,
        modeler_context_hash=None, modeler_bonus_weight=modeler_bonus_weight)
    arm_c = CounterfactualArm(
        arm_label="C", selector=selector_c, pool_hash=pool_hash,
        judgment_cache_hash=judgment_cache_hash,
        prompt_set_hash=prompt_set_c.prompt_set_hash,
        modeler_enabled=True, student_profile_hash=student_profile_hash,
        modeler_context_hash=modeler_context_hash,
        modeler_bonus_weight=modeler_bonus_weight)
    return arm_b, arm_c
