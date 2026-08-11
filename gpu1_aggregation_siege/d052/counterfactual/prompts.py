"""Canonical B/C prompt specifications + deterministic prompt hashes (Phase 2.5).

Task §B/C Prompt 与 Prompt hash. This defines canonical_v2's OWN prompt contract
for the two counterfactual arms and computes a deterministic ``prompt_hash`` over
it, so a selection is bound to the exact prompt definitions that produced its
judgments.

HONESTY NOTE (NO_RAW_DATA_NO_STRONG_CLAIM / NO_SILENT_FALLBACK): the real
Modeler CC Phase-2.5 migration package (its live prompt templates + judgment
cache) is NOT yet on disk. These PromptSpecs are canonical_v2's framework-level
definition -- built ONLY from the pinned ROLE_REGISTRY (provider / exact_model_id
/ prompt_version / output_schema) plus an explicit per-arm conditioning block --
NOT a reproduction of the Modeler CC's private prompt text. The ``prompt_hash``
therefore identifies canonical_v2's prompt CONTRACT; the Modeler CC's real prompt
hashes remain PENDING and must be reconciled when the package arrives. We do not
fabricate or claim parity with the Modeler CC's prompts.

Arm conditioning blocks:
  B (modeler OFF): "achievement_multi_hot:67:modeler=off"
  C (modeler ON) : "achievement_multi_hot:67:modeler=on:student_profile_channel=<id>"
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from d052.counterfactual._hash import sha256_hex
from d052.roles.protocol import ROLE_REGISTRY, RoleName
from d052.schemas.common import CanonicalModel, validate_sha256_hex

PROMPT_CONTRACT_VERSION = "canonical_v2.counterfactual.prompts.v1"

#: The two arm conditioning blocks (the ONLY permitted prompt-level B/C delta).
CONDITIONING_BLOCK_B = "achievement_multi_hot:67:modeler=off"
CONDITIONING_BLOCK_C_PREFIX = "achievement_multi_hot:67:modeler=on:student_profile_channel="


class PromptSpec(CanonicalModel):
    """One role's pinned prompt contract within one arm."""

    arm: Literal["B", "C"]
    role: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    exact_model_id: str = Field(min_length=1)
    role_prompt_version: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    conditioning_block: str = Field(min_length=1)
    prompt_contract_version: str = PROMPT_CONTRACT_VERSION

    def hash_payload(self) -> dict:
        return {
            "arm": self.arm,
            "role": self.role,
            "provider": self.provider,
            "exact_model_id": self.exact_model_id,
            "role_prompt_version": self.role_prompt_version,
            "output_schema": self.output_schema,
            "conditioning_block": self.conditioning_block,
            "prompt_contract_version": self.prompt_contract_version,
        }


class PromptSet(CanonicalModel):
    """The full, ordered prompt contract for one arm (bound by prompt_set_hash)."""

    arm: Literal["B", "C"]
    modeler_enabled: bool
    student_profile_channel_id: Optional[str] = None
    specs: List[PromptSpec] = Field(min_length=1)
    prompt_set_hash: str

    @model_validator(mode="after")
    def _verify_hash(self) -> "PromptSet":
        validate_sha256_hex(self.prompt_set_hash, "prompt_set_hash")
        expected = compute_prompt_set_hash(
            self.arm, self.modeler_enabled, self.specs,
            self.student_profile_channel_id)
        if self.prompt_set_hash != expected:
            raise ValueError(
                f"PROMPT_SET_HASH_MISMATCH: expected {expected}, "
                f"got {self.prompt_set_hash}")
        return self


def conditioning_block(arm: str, modeler_enabled: bool,
                       student_profile_channel_id: Optional[str]) -> str:
    """The per-arm conditioning block (the only prompt-level B/C difference)."""
    if arm == "B":
        if modeler_enabled:
            raise ValueError("ARM_B_MODELER_ON: arm B must be modeler-OFF")
        return CONDITIONING_BLOCK_B
    # arm C
    if not modeler_enabled:
        raise ValueError("ARM_C_MODELER_OFF: arm C must be modeler-ON")
    if not student_profile_channel_id:
        raise ValueError(
            "ARM_C_MISSING_CHANNEL: arm C conditioning requires the "
            "student_profile_channel id")
    return CONDITIONING_BLOCK_C_PREFIX + student_profile_channel_id


def compute_prompt_set_hash(arm: str, modeler_enabled: bool,
                            specs: List[PromptSpec],
                            student_profile_channel_id: Optional[str]) -> str:
    """Deterministic sha256 binding an arm to its exact prompt contract."""
    payload = {
        "arm": arm,
        "modeler_enabled": modeler_enabled,
        "student_profile_channel_id": student_profile_channel_id,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "specs": [s.hash_payload() for s in sorted(specs, key=lambda s: s.role)],
    }
    return sha256_hex(payload)


def role_judgment_prompt_hash(roles: List[RoleName]) -> str:
    """Deterministic hash of the SHARED role-scoring prompt contract.

    This is the prompt identity the judgment cache binds to. It covers ONLY the
    per-role pins (provider / exact_model_id / prompt_version / output_schema) and
    DELIBERATELY EXCLUDES the per-arm conditioning block, so it is IDENTICAL for
    arms B and C: the three scoring roles are prompted the same way in both arms
    (only the modeler conditioning differs). Binding the cache to this shared hash
    is what makes the matched B/C judgment_cache_hash identical (gate 1) while the
    per-arm prompt_set_hash still differs (the legitimate conditioning delta).
    """
    specs = []
    for role in sorted(roles, key=lambda r: r.value):
        d = ROLE_REGISTRY[role]
        specs.append({
            "role": role.value,
            "provider": d.provider,
            "exact_model_id": d.exact_model_id,
            "role_prompt_version": d.prompt_version,
            "output_schema": d.output_schema,
        })
    return sha256_hex({
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "scope": "shared_role_scoring_judgments",
        "specs": specs,
    })


def build_prompt_set(arm: Literal["B", "C"], roles: List[RoleName],
                     modeler_enabled: bool,
                     student_profile_channel_id: Optional[str] = None) -> PromptSet:
    """Build one arm's PromptSet from the pinned ROLE_REGISTRY.

    Every provider/model/prompt_version/output_schema comes from ROLE_REGISTRY
    (never hardcoded), so the prompt contract cannot drift from the role pins.
    """
    block = conditioning_block(arm, modeler_enabled, student_profile_channel_id)
    specs: List[PromptSpec] = []
    for role in sorted(roles, key=lambda r: r.value):
        d = ROLE_REGISTRY[role]
        specs.append(PromptSpec(
            arm=arm, role=role.value, provider=d.provider,
            exact_model_id=d.exact_model_id, role_prompt_version=d.prompt_version,
            output_schema=d.output_schema, conditioning_block=block))
    return PromptSet(
        arm=arm, modeler_enabled=modeler_enabled,
        student_profile_channel_id=student_profile_channel_id,
        specs=specs,
        prompt_set_hash=compute_prompt_set_hash(
            arm, modeler_enabled, specs, student_profile_channel_id))
