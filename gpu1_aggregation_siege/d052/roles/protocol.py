"""Role protocol for the four D052 roles: Tutor / Critic / Explorer / Modeler.

Fixes the legacy defect where llm_roles.py prompt templates hardcoded model names
(qwen-turbo / deepseek-chat) that CONFLICTED with model_manifest.py pins. Here the
role -> (provider, exact_model_id, prompt_version, output_schema) binding is defined
in ONE pinned, versioned registry; a consistency check guarantees the prompt layer
cannot disagree with it (NO_SILENT_FALLBACK at the role boundary).

The three SCORING roles (tutor/critic/explorer) emit per-candidate RoleJudgments;
the Modeler runs once per session over the student state (ModelerJudgment, in
d052/profiling) and is NOT a per-candidate scorer.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

from pydantic import Field

from d052.schemas.common import CanonicalModel
from d052.schemas.roles import RoleJudgment, RoleName, ScoringRole

#: canonical prompt/output schema versions for this protocol
ROLE_PROMPT_VERSION = "canonical_v2.roles.v1"
ROLE_OUTPUT_SCHEMA = "role_judgment_v2"


class RoleDefinition(CanonicalModel):
    """Pinned, versioned binding of a role to its provider/model/prompt."""

    role: RoleName
    provider: str = Field(min_length=1)
    exact_model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    is_scoring_role: bool
    description: str = ""


def _registry() -> Dict[RoleName, RoleDefinition]:
    # Reconciled from gpu1 mechanisms/llm_roles.py ROLE_PROVIDER_MAP
    # (tutor=qwen, critic=deepseek, explorer=glm) + auction/modeler.py (GLM).
    # exact_model_id values are the canonical pins; prompt templates MUST reference
    # these via the registry, never hardcode a different model (the legacy bug).
    return {
        RoleName.TUTOR: RoleDefinition(
            role=RoleName.TUTOR, provider="dashscope",
            exact_model_id="qwen-turbo", prompt_version=ROLE_PROMPT_VERSION,
            output_schema=ROLE_OUTPUT_SCHEMA, is_scoring_role=True,
            description="Progression/learnability scorer (headline: progression_score)."),
        RoleName.CRITIC: RoleDefinition(
            role=RoleName.CRITIC, provider="deepseek",
            exact_model_id="deepseek-chat", prompt_version=ROLE_PROMPT_VERSION,
            output_schema=ROLE_OUTPUT_SCHEMA, is_scoring_role=True,
            description="Validity/hacking critic (headline: critic_penalty + critic_reject)."),
        RoleName.EXPLORER: RoleDefinition(
            role=RoleName.EXPLORER, provider="zhipu",
            exact_model_id="glm-4.5-air", prompt_version=ROLE_PROMPT_VERSION,
            output_schema=ROLE_OUTPUT_SCHEMA, is_scoring_role=True,
            description="Novelty/coverage scorer (headline: novelty_score)."),
        RoleName.MODELER: RoleDefinition(
            role=RoleName.MODELER, provider="zhipu",
            exact_model_id="glm-4.5", prompt_version=ROLE_PROMPT_VERSION,
            output_schema="modeler_judgment_v1", is_scoring_role=False,
            description="Student-state modeler; runs once/session, not per-candidate."),
    }


ROLE_REGISTRY: Dict[RoleName, RoleDefinition] = _registry()

#: the three per-candidate scoring roles
SCORING_ROLES: tuple = (RoleName.TUTOR, RoleName.CRITIC, RoleName.EXPLORER)


def role_definition(role: RoleName) -> RoleDefinition:
    return ROLE_REGISTRY[role]


def assert_registry_consistency() -> None:
    """Internal self-check: exactly the 4 roles, 3 scoring, pins non-empty.

    This is the structural guarantee that replaces the legacy hardcoded-model
    conflict: every role's prompt_version/output_schema come from this registry.
    """
    assert set(ROLE_REGISTRY) == set(RoleName), set(ROLE_REGISTRY)
    scoring = {r for r, d in ROLE_REGISTRY.items() if d.is_scoring_role}
    assert scoring == set(SCORING_ROLES), scoring
    for d in ROLE_REGISTRY.values():
        assert d.provider and d.exact_model_id and d.prompt_version
        assert d.prompt_version == ROLE_PROMPT_VERSION


assert_registry_consistency()


class RoleProtocolError(Exception):
    DUPLICATE_ROLE = "DUPLICATE_ROLE"
    MISSING_ROLE = "MISSING_ROLE"
    CANDIDATE_MISMATCH = "CANDIDATE_MISMATCH"
    NON_SCORING_ROLE = "NON_SCORING_ROLE"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


def _coerce(j: object) -> RoleJudgment:
    return j if isinstance(j, RoleJudgment) else RoleJudgment.model_validate(j)


def validate_judgment_batch(
    candidate_id: str,
    judgments: Iterable[object],
    *,
    required_roles: Sequence[RoleName] = SCORING_ROLES,
) -> Dict[RoleName, RoleJudgment]:
    """Validate one candidate's judgment batch (one judgment per required role).

    Fail-closed: duplicate role, missing required role, a non-scoring role in the
    batch, or a judgment whose candidate_id disagrees -> error. Returns a
    role->RoleJudgment map.
    """
    out: Dict[RoleName, RoleJudgment] = {}
    for raw in judgments:
        j = _coerce(raw)
        if j.candidate_id != candidate_id:
            raise RoleProtocolError(
                RoleProtocolError.CANDIDATE_MISMATCH,
                f"judgment candidate_id {j.candidate_id!r} != batch "
                f"candidate_id {candidate_id!r}")
        role = RoleName(j.role.value)
        if role not in set(SCORING_ROLES):
            raise RoleProtocolError(
                RoleProtocolError.NON_SCORING_ROLE,
                f"role {role.value} is not a per-candidate scoring role")
        if role in out:
            raise RoleProtocolError(
                RoleProtocolError.DUPLICATE_ROLE,
                f"duplicate judgment for role {role.value} on {candidate_id}")
        out[role] = j
    missing = [r.value for r in required_roles if r not in out]
    if missing:
        raise RoleProtocolError(
            RoleProtocolError.MISSING_ROLE,
            f"missing judgments for required roles {missing} on {candidate_id}")
    return out


def headline_scores(batch: Mapping[RoleName, RoleJudgment]) -> Dict[str, float]:
    """role value -> headline score (for normalization/selection)."""
    return {role.value: j.headline_score for role, j in batch.items()}


def critic_vetoed(batch: Mapping[RoleName, RoleJudgment]) -> bool:
    """True iff the critic (present) set critic_reject=True (hard-veto bit)."""
    j = batch.get(RoleName.CRITIC)
    return bool(j is not None and j.critic_reject is True)
