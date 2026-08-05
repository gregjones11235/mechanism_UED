"""CC2 follow-up P0-7: the authorized EnvCoder validation runtime.

The real EnvCoder backend used to block outright with no controlled
execution surface. This module is that surface — an authorization
carrier plus the 13-stage validation priority ladder::

    AUTHORIZATION -> SOURCE_HASH -> SYNTAX -> GUARDS -> STRUCTURE
    -> IMPORT_ISOLATION -> INSTANTIATE -> RESET -> STEP
    -> TERMINAL_AUTORESET -> DETERMINISM -> RESOURCE_ENVELOPE
    -> ATTESTATION

Discipline this round:

* PRODUCTION authorization needs a signer on the supervisor-owned
  whitelist (EMPTY — no real validation is authorized yet; honest
  ``ENV_VALIDATION_PROVIDER_UNAUTHORIZED``);
* the TEST_ONLY contract proves the ladder's shape and order with
  conspicuously-marked synthetic stages — it NEVER executes real code,
  never flips ``real_envcoder_used``, and never substitutes for the
  real backend;
* NO LLM code is ever exec'd/imported inside the main training
  process through this surface — validation consumes an EnvCoder
  artifact's identity only; execution belongs to the shared runtime;
* the outcome is an immutable record; a partial ladder is never
  attested as complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from .canonical import canonical_sha256
from .schemas import E1SchemaError

#: the 13-stage validation priority ladder (fixed order)
ENVCODER_VALIDATION_STAGES = (
    "AUTHORIZATION",
    "SOURCE_HASH",
    "SYNTAX",
    "GUARDS",
    "STRUCTURE",
    "IMPORT_ISOLATION",
    "INSTANTIATE",
    "RESET",
    "STEP",
    "TERMINAL_AUTORESET",
    "DETERMINISM",
    "RESOURCE_ENVELOPE",
    "ATTESTATION",
)

#: authorization modes
ENV_VALIDATION_MODE_PRODUCTION = "PRODUCTION"
ENV_VALIDATION_MODE_TEST_ONLY = "TEST_ONLY"

#: synthetic TEST_ONLY identities (greppable)
SYNTHETIC_TEST_ONLY_VALIDATOR = "SYNTHETIC_TEST_ONLY_ENV_VALIDATOR"
SYNTHETIC_TEST_ONLY_VALIDATION_SIGNER = (
    "SYNTHETIC_TEST_ONLY_VALIDATION_SIGNER"
)

#: supervisor-owned production whitelist — EMPTY this round
AUTHORIZED_ENV_VALIDATION_PROVIDERS: Tuple[str, ...] = ()

#: binding version
ENV_VALIDATION_VERSION = "e1-envcoder-validation-v1"

# fail-closed codes (greppable)
ENV_VALIDATION_BAD_TYPE = "ENV_VALIDATION_BAD_TYPE"
ENV_VALIDATION_GRANT_BAD = "ENV_VALIDATION_GRANT_BAD"
ENV_VALIDATION_PROVIDER_UNAUTHORIZED = (
    "ENV_VALIDATION_PROVIDER_UNAUTHORIZED"
)
ENV_VALIDATION_PROVIDER_FORBIDDEN = "ENV_VALIDATION_PROVIDER_FORBIDDEN"
ENV_VALIDATION_TEST_ONLY_REJECTED = "ENV_VALIDATION_TEST_ONLY_REJECTED"
ENV_VALIDATION_LADDER_INCOMPLETE = "ENV_VALIDATION_LADDER_INCOMPLETE"

#: forbidden providers (never pose as the real validation surface)
FORBIDDEN_ENV_VALIDATION_PROVIDERS = frozenset({"replay", "mock"})


class EnvValidationError(E1SchemaError):
    """Fail-closed EnvCoder-validation violation; ``code`` is
    greppable."""


@dataclass(frozen=True)
class AuthorizedEnvironmentValidationRuntime:
    """The explicit authorization for EnvCoder validation (immutable)."""

    mode: str
    authorization_grant_hash: str
    validator_id: str
    stages: Tuple[str, ...]
    source_commit: str
    runtime_hash: str


@dataclass(frozen=True)
class EnvironmentValidationOutcome:
    """The immutable record of one validation run."""

    artifact_id: str
    artifact_hash: str
    validator_id: str
    stages_passed: Tuple[str, ...]
    ladder_complete: bool
    test_only: bool
    outcome_hash: str


def authorize_environment_validation_runtime(
    *,
    mode: str,
    authorization_grant_hash: str,
    validator_id: str,
    source_commit: str,
) -> AuthorizedEnvironmentValidationRuntime:
    """Authorize the validation runtime fail-closed on EVERY field."""
    ctx = "envcoder_validation.authorize"
    if mode not in (
        ENV_VALIDATION_MODE_PRODUCTION,
        ENV_VALIDATION_MODE_TEST_ONLY,
    ):
        raise EnvValidationError(
            ENV_VALIDATION_BAD_TYPE,
            f"{ctx}: mode must be one of "
            f"{[ENV_VALIDATION_MODE_PRODUCTION, ENV_VALIDATION_MODE_TEST_ONLY]}, got {mode!r}",
        )
    if (
        not isinstance(authorization_grant_hash, str)
        or len(authorization_grant_hash) != 64
    ):
        raise EnvValidationError(
            ENV_VALIDATION_GRANT_BAD,
            f"{ctx}: authorization_grant_hash must be 64-hex, got "
            f"{authorization_grant_hash!r}",
        )
    if not isinstance(validator_id, str) or not validator_id.strip():
        raise EnvValidationError(
            ENV_VALIDATION_BAD_TYPE,
            f"{ctx}: validator_id must be a non-empty str, got "
            f"{validator_id!r}",
        )
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise EnvValidationError(
            ENV_VALIDATION_BAD_TYPE,
            f"{ctx}: source_commit must be a non-empty str, got "
            f"{source_commit!r}",
        )
    if mode == ENV_VALIDATION_MODE_PRODUCTION:
        if validator_id in FORBIDDEN_ENV_VALIDATION_PROVIDERS:
            raise EnvValidationError(
                ENV_VALIDATION_PROVIDER_FORBIDDEN,
                f"{ctx}: validator {validator_id!r} is a replay/mock "
                "identity; it may never serve the production "
                "validation surface",
            )
        if validator_id not in AUTHORIZED_ENV_VALIDATION_PROVIDERS:
            raise EnvValidationError(
                ENV_VALIDATION_PROVIDER_UNAUTHORIZED,
                f"{ctx}: validator {validator_id!r} is not on the "
                "supervisor-owned whitelist (EMPTY this round); no "
                "real EnvCoder validation is authorized",
            )
    else:
        if validator_id != SYNTHETIC_TEST_ONLY_VALIDATOR:
            raise EnvValidationError(
                ENV_VALIDATION_PROVIDER_FORBIDDEN,
                f"{ctx}: TEST_ONLY validations must use "
                f"{SYNTHETIC_TEST_ONLY_VALIDATOR!r}, got "
                f"{validator_id!r}",
            )
    runtime_hash = canonical_sha256(
        {
            "validation_version": ENV_VALIDATION_VERSION,
            "mode": mode,
            "authorization_grant_hash": authorization_grant_hash,
            "validator_id": validator_id,
            "stages": list(ENVCODER_VALIDATION_STAGES),
            "source_commit": source_commit,
        }
    )
    return AuthorizedEnvironmentValidationRuntime(
        mode=mode,
        authorization_grant_hash=authorization_grant_hash,
        validator_id=validator_id,
        stages=ENVCODER_VALIDATION_STAGES,
        source_commit=source_commit,
        runtime_hash=runtime_hash,
    )


def run_authorized_validation(
    envcoder_artifact: Any,
    *,
    authorization: Any,
) -> EnvironmentValidationOutcome:
    """Run the 13-stage ladder under the authorization (fail-closed).

    PRODUCTION this round: impossible (the whitelist is empty, so no
    authorization can exist). TEST_ONLY contract: the ladder's shape
    and order are recorded with synthetic stage results — NO real
    execution happens, and the outcome is conspicuously marked
    test_only. A partial ladder is never marked complete.
    """
    ctx = "envcoder_validation.run"
    if not isinstance(
        authorization, AuthorizedEnvironmentValidationRuntime
    ):
        raise EnvValidationError(
            ENV_VALIDATION_BAD_TYPE,
            f"{ctx}: authorization must be an "
            f"AuthorizedEnvironmentValidationRuntime, got "
            f"{type(authorization).__name__}",
        )
    artifact_id = getattr(envcoder_artifact, "artifact_id", None)
    if not isinstance(artifact_id, str) or not artifact_id:
        raise EnvValidationError(
            ENV_VALIDATION_BAD_TYPE,
            f"{ctx}: envcoder_artifact must expose a non-empty "
            f"artifact_id, got {artifact_id!r}",
        )
    env_code = getattr(envcoder_artifact, "env_code", "")
    artifact_hash = canonical_sha256(
        {
            "artifact_id": artifact_id,
            "env_code": env_code,
            "template_hash": getattr(
                envcoder_artifact, "template_hash", ""
            ),
        }
    )
    if authorization.mode == ENV_VALIDATION_MODE_PRODUCTION:
        # defense in depth: even a forged production authorization
        # cannot validate while the whitelist is empty
        if authorization.validator_id not in (
            AUTHORIZED_ENV_VALIDATION_PROVIDERS
        ):
            raise EnvValidationError(
                ENV_VALIDATION_PROVIDER_UNAUTHORIZED,
                f"{ctx}: production validation is unauthorized this "
                f"round (validator {authorization.validator_id!r})",
            )
        raise EnvValidationError(
            ENV_VALIDATION_LADDER_INCOMPLETE,
            f"{ctx}: real 13-stage execution requires the shared "
            "runtime execution surface (absent); a production ladder "
            "never attests without executing",
        )
    if authorization.validator_id != SYNTHETIC_TEST_ONLY_VALIDATOR:
        raise EnvValidationError(
            ENV_VALIDATION_TEST_ONLY_REJECTED,
            f"{ctx}: TEST_ONLY validation requires the synthetic "
            f"validator, got {authorization.validator_id!r}",
        )
    # TEST_ONLY contract: the ladder runs as SHAPE ONLY — every stage
    # records its name in order; nothing executes
    stages_passed = tuple(authorization.stages)
    if stages_passed != ENVCODER_VALIDATION_STAGES:
        raise EnvValidationError(
            ENV_VALIDATION_LADDER_INCOMPLETE,
            f"{ctx}: the ladder must run all "
            f"{len(ENVCODER_VALIDATION_STAGES)} stages in the fixed "
            "order; a partial ladder never attests",
        )
    outcome_hash = canonical_sha256(
        {
            "validation_version": ENV_VALIDATION_VERSION,
            "artifact_id": artifact_id,
            "artifact_hash": artifact_hash,
            "validator_id": authorization.validator_id,
            "stages_passed": list(stages_passed),
            "test_only": True,
        }
    )
    return EnvironmentValidationOutcome(
        artifact_id=artifact_id,
        artifact_hash=artifact_hash,
        validator_id=authorization.validator_id,
        stages_passed=stages_passed,
        ladder_complete=True,
        test_only=True,
        outcome_hash=outcome_hash,
    )
