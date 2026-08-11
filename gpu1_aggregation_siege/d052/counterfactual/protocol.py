"""B/C strict-match counterfactual protocol + verifier (Phase 2.5, gate 1).

Task §B/C 严格匹配反事实协议 + gate 1: arms B and C must be IDENTICAL in every field
EXCEPT the StudentProfile / Modeler conditioning. This module defines one
``CounterfactualArm`` (a full selection configuration) and ``verify_matched_bc``
which fails closed if ANY non-permitted field differs between the arms.

Permitted B/C deltas (the counterfactual itself):
  * arm_label                      (B vs C)
  * modeler_enabled                (B=False, C=True)  -- the ablated condition
  * student_profile_hash           (B=None, C=set)
  * modeler_context_hash           (B=None, C=set)
  * selector.selector              (B=S1_THREE_ROLE, C=S2_FOUR_ROLE_MODELER)
  * prompt_set_hash                (conditioning block differs by arm)

Everything else -- pool_hash, judgment_cache_hash, modeler_bonus_weight, and the
selector's k / seed / critic_policy / roles / budget -- MUST match bit-for-bit.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from d052.counterfactual._hash import sha256_hex
from d052.counterfactual.student_modeler_channel import MODELER_BONUS_WEIGHT
from d052.schemas.common import CanonicalModel, validate_sha256_hex
from d052.schemas.selector import (
    SelectionResult,
    SelectorConfig,
    SelectorType,
)


class CounterfactualProtocolError(Exception):
    NOT_ARM_B = "NOT_ARM_B"
    NOT_ARM_C = "NOT_ARM_C"
    B_NOT_MODELER_OFF = "B_NOT_MODELER_OFF"
    C_NOT_MODELER_ON = "C_NOT_MODELER_ON"
    WRONG_ABLATION_SELECTORS = "WRONG_ABLATION_SELECTORS"
    POOL_MISMATCH = "POOL_MISMATCH"
    CACHE_MISMATCH = "CACHE_MISMATCH"
    WEIGHT_MISMATCH = "WEIGHT_MISMATCH"
    K_MISMATCH = "K_MISMATCH"
    SEED_MISMATCH = "SEED_MISMATCH"
    CRITIC_POLICY_MISMATCH = "CRITIC_POLICY_MISMATCH"
    ROLES_MISMATCH = "ROLES_MISMATCH"
    BUDGET_MISMATCH = "BUDGET_MISMATCH"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class CounterfactualArm(CanonicalModel):
    """One arm's full, hash-bound selection configuration."""

    arm_label: Literal["B", "C"]
    selector: SelectorConfig
    pool_hash: str
    judgment_cache_hash: str
    prompt_set_hash: str
    modeler_enabled: bool
    student_profile_hash: Optional[str] = None
    modeler_context_hash: Optional[str] = None
    modeler_bonus_weight: float = MODELER_BONUS_WEIGHT
    #: filled once the arm is run (kept out of the strict-match comparison except
    #: for replay binding); optional so arms can be built before selection.
    selection_result: Optional[SelectionResult] = None

    @model_validator(mode="after")
    def _hashes(self) -> "CounterfactualArm":
        validate_sha256_hex(self.pool_hash, "pool_hash")
        validate_sha256_hex(self.judgment_cache_hash, "judgment_cache_hash")
        validate_sha256_hex(self.prompt_set_hash, "prompt_set_hash")
        if self.student_profile_hash is not None:
            validate_sha256_hex(self.student_profile_hash, "student_profile_hash")
        if self.modeler_context_hash is not None:
            validate_sha256_hex(self.modeler_context_hash, "modeler_context_hash")
        return self


#: fields permitted to differ between matched B and C arms (the counterfactual)
PERMITTED_DELTA_FIELDS = (
    "arm_label",
    "modeler_enabled",
    "student_profile_hash",
    "modeler_context_hash",
    "selector.selector",
    "prompt_set_hash",
)

#: fields that MUST be identical (recorded in the verification for audit)
IDENTICAL_FIELDS = (
    "pool_hash",
    "judgment_cache_hash",
    "modeler_bonus_weight",
    "selector.k",
    "selector.seed",
    "selector.critic_policy",
    "selector.roles",
    "selector.budget",
)


class MatchedVerification(CanonicalModel):
    """Audit-grade record that B and C are matched except for the conditioning."""

    passed: Literal[True] = True
    identical_fields: List[str] = Field(default_factory=list)
    permitted_delta_fields: List[str] = Field(default_factory=list)
    selector_b: str
    selector_c: str
    modeler_enabled_b: bool = False
    modeler_enabled_c: bool = True
    verification_hash: str = ""

    @model_validator(mode="after")
    def _hash(self) -> "MatchedVerification":
        payload = {
            "identical_fields": sorted(self.identical_fields),
            "permitted_delta_fields": sorted(self.permitted_delta_fields),
            "selector_b": self.selector_b,
            "selector_c": self.selector_c,
            "modeler_enabled_b": self.modeler_enabled_b,
            "modeler_enabled_c": self.modeler_enabled_c,
        }
        expected = sha256_hex(payload)
        if self.verification_hash and self.verification_hash != expected:
            raise ValueError("MATCHED_VERIFICATION_HASH_MISMATCH")
        object.__setattr__(self, "verification_hash", expected)
        return self


def _roles_sorted(cfg: SelectorConfig) -> List[str]:
    return sorted(r.value for r in cfg.roles)


def verify_matched_bc(b: CounterfactualArm, c: CounterfactualArm) -> MatchedVerification:
    """Assert B and C are a valid matched counterfactual pair (gate 1).

    Raises CounterfactualProtocolError with a specific code on ANY non-permitted
    difference. Returns the audit record on success.
    """
    if b.arm_label != "B":
        raise CounterfactualProtocolError(
            CounterfactualProtocolError.NOT_ARM_B, f"first arm label={b.arm_label}")
    if c.arm_label != "C":
        raise CounterfactualProtocolError(
            CounterfactualProtocolError.NOT_ARM_C, f"second arm label={c.arm_label}")
    if b.modeler_enabled is not False:
        raise CounterfactualProtocolError(
            CounterfactualProtocolError.B_NOT_MODELER_OFF,
            "arm B must be modeler-OFF (the ablation baseline)")
    if c.modeler_enabled is not True:
        raise CounterfactualProtocolError(
            CounterfactualProtocolError.C_NOT_MODELER_ON,
            "arm C must be modeler-ON (the modeler condition)")
    if not (b.selector.selector is SelectorType.S1_THREE_ROLE
            and c.selector.selector is SelectorType.S2_FOUR_ROLE_MODELER):
        raise CounterfactualProtocolError(
            CounterfactualProtocolError.WRONG_ABLATION_SELECTORS,
            f"canonical modeler ablation requires B=S1_THREE_ROLE and "
            f"C=S2_FOUR_ROLE_MODELER; got B={b.selector.selector.value}, "
            f"C={c.selector.selector.value}")

    def _eq(field: str, vb, vc, code: str) -> None:
        if vb != vc:
            raise CounterfactualProtocolError(
                code, f"matched B/C field {field!r} differs: B={vb!r} C={vc!r}")

    _eq("pool_hash", b.pool_hash, c.pool_hash,
        CounterfactualProtocolError.POOL_MISMATCH)
    _eq("judgment_cache_hash", b.judgment_cache_hash, c.judgment_cache_hash,
        CounterfactualProtocolError.CACHE_MISMATCH)
    _eq("modeler_bonus_weight", b.modeler_bonus_weight, c.modeler_bonus_weight,
        CounterfactualProtocolError.WEIGHT_MISMATCH)
    _eq("selector.k", b.selector.k, c.selector.k,
        CounterfactualProtocolError.K_MISMATCH)
    _eq("selector.seed", b.selector.seed, c.selector.seed,
        CounterfactualProtocolError.SEED_MISMATCH)
    _eq("selector.critic_policy", b.selector.critic_policy, c.selector.critic_policy,
        CounterfactualProtocolError.CRITIC_POLICY_MISMATCH)
    _eq("selector.roles", _roles_sorted(b.selector), _roles_sorted(c.selector),
        CounterfactualProtocolError.ROLES_MISMATCH)
    _eq("selector.budget", b.selector.budget, c.selector.budget,
        CounterfactualProtocolError.BUDGET_MISMATCH)

    return MatchedVerification(
        identical_fields=list(IDENTICAL_FIELDS),
        permitted_delta_fields=list(PERMITTED_DELTA_FIELDS),
        selector_b=b.selector.selector.value,
        selector_c=c.selector.selector.value,
        modeler_enabled_b=b.modeler_enabled,
        modeler_enabled_c=c.modeler_enabled,
    )
