"""Tier-C fail-closed gate for the TWO independent critic decision dimensions.

D052_PREMERGE_SEMANTIC_CLEANUP_V3 splits "critic policy" into two dimensions
that must NEVER substitute for each other:

  A. critic_reject_derivation_rule
     how the canonical critic_reject boolean is DERIVED from a legacy/LLM
     judgment (handled by judgment_adapter.py). Candidates: decision_reject |
     flags_too_hard. Status: UNDECIDED.

  B. critic_selection_policy
     how the SELECTOR consumes critic_reject / the critic score (handled by
     schemas/selector.py + selectors/). Candidates: hard_veto | soft_penalty |
     score_only. Status: UNDECIDED.

Any REAL canonical Tier-C path must have BOTH dimensions explicitly frozen;
if either is missing/unknown/pending, the gate fails closed. This module never
chooses a rule or a policy, never runs training, and never touches the
historical legacy replay (which consumes raw critic_penalty only).
"""
from __future__ import annotations

from typing import Optional

from d052.reconciliation.judgment_adapter import CRITIC_REJECT_DERIVATION_RULES

#: Dimension B candidates (mirrors schemas.selector.CriticPolicy values; the
#: selector layer owns the enum, this tuple exists only for Tier-C gating).
LEGAL_CRITIC_SELECTION_POLICIES = ("hard_veto", "soft_penalty", "score_only")

#: sentinel values that mean "not frozen" and must fail closed
_UNFROZEN = (None, "", "UNDECIDED", "NONE", "PENDING",
             "PENDING_DIRECTOR_DECISION")


class TierCPolicyError(Exception):
    CRITIC_DERIVATION_RULE_REQUIRED = "CRITIC_DERIVATION_RULE_REQUIRED"
    CRITIC_SELECTION_POLICY_REQUIRED = "CRITIC_SELECTION_POLICY_REQUIRED"

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


def require_frozen_critic_fields(
        critic_reject_derivation_rule: Optional[str],
        critic_selection_policy: Optional[str]) -> dict:
    """Fail closed unless BOTH dimensions are explicitly frozen and legal.

    Returns the two-field policy record that a real canonical run must attach
    to its execution/selection evidence (each dimension's status recorded
    separately). Raises TierCPolicyError naming the missing dimension.
    """
    if (critic_reject_derivation_rule in _UNFROZEN
            or critic_reject_derivation_rule not in CRITIC_REJECT_DERIVATION_RULES):
        raise TierCPolicyError(
            TierCPolicyError.CRITIC_DERIVATION_RULE_REQUIRED,
            f"critic_reject_derivation_rule must be frozen to one of "
            f"{sorted(CRITIC_REJECT_DERIVATION_RULES)}; got "
            f"{critic_reject_derivation_rule!r} "
            f"(REAL_CANONICAL_CRITIC_REJECT_DERIVATION_RULE=UNDECIDED)")
    if (critic_selection_policy in _UNFROZEN
            or critic_selection_policy not in LEGAL_CRITIC_SELECTION_POLICIES):
        raise TierCPolicyError(
            TierCPolicyError.CRITIC_SELECTION_POLICY_REQUIRED,
            f"critic_selection_policy must be frozen to one of "
            f"{list(LEGAL_CRITIC_SELECTION_POLICIES)}; got "
            f"{critic_selection_policy!r} "
            f"(REAL_CANONICAL_CRITIC_SELECTION_POLICY=UNDECIDED)")
    return {"critic_reject_derivation_rule": critic_reject_derivation_rule,
            "critic_selection_policy": critic_selection_policy,
            "both_frozen": True}


def validate_template_critic_fields(template: dict) -> dict:
    """Verify a real-canonical cell template keeps BOTH dimensions split,
    PENDING and blocking (the template must NOT be promotable to DRAFT)."""
    sel = template.get("fields_PENDING_real_values", {}).get("selector", {})
    problems = []
    if "critic_policy" in sel:
        problems.append("ambiguous single 'critic_policy' field must be split")
    deriv = sel.get("critic_reject_derivation_rule")
    policy = sel.get("critic_selection_policy")
    if deriv != "PENDING_DIRECTOR_DECISION":
        problems.append(f"critic_reject_derivation_rule must be "
                        f"PENDING_DIRECTOR_DECISION, got {deriv!r}")
    if policy != "PENDING_DIRECTOR_DECISION":
        problems.append(f"critic_selection_policy must be "
                        f"PENDING_DIRECTOR_DECISION, got {policy!r}")
    blockers = template.get("blockers", [])
    needs = ("critic_reject derivation rule not frozen",
             "critic selector consumption policy not frozen")
    for n in needs:
        if not any(n in b for b in blockers):
            problems.append(f"missing separate blocker: {n!r}")
    # and the gate itself must fail closed on the template's pending values
    try:
        require_frozen_critic_fields(deriv, policy)
        problems.append("gate did NOT fail closed on PENDING values")
    except TierCPolicyError:
        pass
    return {"ok": not problems, "problems": problems}
