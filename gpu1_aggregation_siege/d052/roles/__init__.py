"""Role protocol: Tutor / Critic / Explorer / Modeler."""
from d052.roles.protocol import (
    ROLE_OUTPUT_SCHEMA,
    ROLE_PROMPT_VERSION,
    ROLE_REGISTRY,
    SCORING_ROLES,
    RoleDefinition,
    RoleProtocolError,
    assert_registry_consistency,
    critic_vetoed,
    headline_scores,
    role_definition,
    validate_judgment_batch,
)

__all__ = [
    "ROLE_OUTPUT_SCHEMA",
    "ROLE_PROMPT_VERSION",
    "ROLE_REGISTRY",
    "SCORING_ROLES",
    "RoleDefinition",
    "RoleProtocolError",
    "assert_registry_consistency",
    "critic_vetoed",
    "headline_scores",
    "role_definition",
    "validate_judgment_batch",
]
