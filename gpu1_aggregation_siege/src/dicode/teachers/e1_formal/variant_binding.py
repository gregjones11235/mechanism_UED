"""CC2 follow-up P0-6: variant task parameters ENTER execution.

Mode A (parameterized template): one template's Env Code + per-variant
deterministic TaskParams => one executable candidate. Commit 4 bound
the variant parameters into the CANDIDATE HASH; this module is the
ONLY surface that may clear the conspicuous
``VARIANT_PARAMETER_NOT_EXECUTED`` marker — and only on mechanical
evidence that those exact parameters actually entered a REAL
validation run:

* the validation record must bind the SAME candidate hash and the
  SAME task-params hash;
* the executed parameter sequence must EQUAL the candidate's variant
  parameters (byte-for-byte — parameters that never reached the
  environment never clear the marker);
* the backend must be the REAL ladder backend with the FULL 8-stage
  ladder passed; replay/mock backends are refused outright;
* TEST_ONLY validation records never clear a production marker.

Also here: the driver-level pool binder
(``bind_executable_pool_from_materials``) that turns the window's
compile+EnvCoder materials into the executable candidate pool, and
the execution-surface resolution contract (ABI / reward / reset /
seed surfaces come ONLY from bundle-bound shared objects — absent
shared runtime => fail closed).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Tuple

from . import envcoder_backends as EB
from .canonical import canonical_sha256
from .executable_candidates import (
    ExecutableCandidate,
    ExecutableEnvironmentArtifact,
    VARIANT_PARAMETER_NOT_EXECUTED,
    bind_executable_candidate_pool,
    build_executable_environment_artifact,
)
from .schemas import E1SchemaError
from .task_specs import CompileResult

#: binding provenance label (versioned)
VARIANT_EXECUTION_VERSION = "e1-variant-execution-v1"

#: the execution surfaces every executable artifact needs, and the
#: bundle capability contract responsible for each (fail-closed when
#: the shared objects are absent — this round: always)
EXECUTION_SURFACE_CONTRACTS = {
    "observation_action_abi_hash": "student_adapter",
    "reward_contract_hash": "formal_asset_registry",
    "reset_protocol_hash": "probe_runner",
    "seed_policy_hash": "probe_runner",
}

#: capability attribute exposing each surface on the shared object
_EXECUTION_SURFACE_ATTRIBUTES = {
    "observation_action_abi_hash": "observation_action_abi_hash",
    "reward_contract_hash": "reward_contract_hash",
    "reset_protocol_hash": "reset_protocol_hash",
    "seed_policy_hash": "seed_bank_hash",
}

# fail-closed codes (greppable)
VARIANT_BAD_TYPE = "VARIANT_BAD_TYPE"
VARIANT_HASH_MISMATCH = "VARIANT_HASH_MISMATCH"
VARIANT_PARAMS_MISMATCH = "VARIANT_PARAMS_MISMATCH"
VARIANT_BACKEND_FORBIDDEN = "VARIANT_BACKEND_FORBIDDEN"
VARIANT_STAGES_INCOMPLETE = "VARIANT_STAGES_INCOMPLETE"
VARIANT_TEST_ONLY_REJECTED = "VARIANT_TEST_ONLY_REJECTED"
VARIANT_ALREADY_EXECUTED = "VARIANT_ALREADY_EXECUTED"
VARIANT_EXECUTION_SURFACE_UNBOUND = "VARIANT_EXECUTION_SURFACE_UNBOUND"


class VariantBindingError(E1SchemaError):
    """Fail-closed variant-execution violation; ``code`` is
    greppable."""


@dataclass(frozen=True)
class VariantExecutionValidation:
    """One REAL validation run's report about one candidate.

    Minted only by the authorized real validation path; every field is
    mechanically compared against the candidate before the marker may
    clear. ``test_only`` records are refused on the production
    surface.
    """

    candidate_hash: str
    task_params_hash: str
    executed_variant_params: Tuple[Tuple[str, str], ...]
    backend_name: str
    stages_passed: Tuple[str, ...]
    validation_record_hash: str
    test_only: bool = False


def execute_variant_parameters(
    candidate: Any,
    validation: Any,
    ctx: str,
) -> ExecutableCandidate:
    """Clear ``VARIANT_PARAMETER_NOT_EXECUTED`` — with evidence ONLY.

    Returns the executed candidate (``variant_params_executed=True``,
    marker cleared). Any mismatch, any non-real backend, any
    incomplete stage ladder, any TEST_ONLY record fails closed and
    the marker STAYS.
    """
    if not isinstance(candidate, ExecutableCandidate):
        raise VariantBindingError(
            VARIANT_BAD_TYPE,
            f"{ctx}: candidate must be an ExecutableCandidate, got "
            f"{type(candidate).__name__}",
        )
    if not isinstance(validation, VariantExecutionValidation):
        raise VariantBindingError(
            VARIANT_BAD_TYPE,
            f"{ctx}: validation must be a VariantExecutionValidation, "
            f"got {type(validation).__name__}",
        )
    if candidate.variant_params_executed:
        raise VariantBindingError(
            VARIANT_ALREADY_EXECUTED,
            f"{ctx}: candidate {candidate.candidate_id} is already "
            "marked executed; execution evidence is never stacked",
        )
    if validation.test_only:
        raise VariantBindingError(
            VARIANT_TEST_ONLY_REJECTED,
            f"{ctx}: TEST_ONLY validation records never clear the "
            "production execution marker",
        )
    if validation.candidate_hash != candidate.candidate_hash:
        raise VariantBindingError(
            VARIANT_HASH_MISMATCH,
            f"{ctx}: validation binds candidate "
            f"{validation.candidate_hash!r} but the candidate is "
            f"{candidate.candidate_hash!r}",
        )
    if validation.task_params_hash != candidate.task_params_hash:
        raise VariantBindingError(
            VARIANT_HASH_MISMATCH,
            f"{ctx}: validation task_params_hash "
            f"{validation.task_params_hash!r} != candidate "
            f"{candidate.task_params_hash!r}",
        )
    executed = tuple(
        (str(name), str(value))
        for name, value in validation.executed_variant_params
    )
    if executed != candidate.variant_params:
        raise VariantBindingError(
            VARIANT_PARAMS_MISMATCH,
            f"{ctx}: executed parameters {executed} != candidate "
            f"variant parameters {candidate.variant_params}; "
            "parameters that never entered execution never clear the "
            "marker",
        )
    if validation.backend_name != EB.BACKEND_REAL:
        raise VariantBindingError(
            VARIANT_BACKEND_FORBIDDEN,
            f"{ctx}: backend {validation.backend_name!r} is not the "
            f"real validation backend {EB.BACKEND_REAL!r}; replay/mock "
            "validation never counts as execution",
        )
    if tuple(validation.stages_passed) != tuple(EB.STAGES):
        raise VariantBindingError(
            VARIANT_STAGES_INCOMPLETE,
            f"{ctx}: stages_passed {list(validation.stages_passed)} "
            f"!= the full ladder {list(EB.STAGES)}; a partially "
            "validated candidate is not executed",
        )
    return replace(
        candidate,
        variant_params_executed=True,
        execution_marker="",
    )


def execution_surfaces_from_bundle_resolutions(
    resolutions: Mapping[str, Any], ctx: str
) -> Dict[str, str]:
    """Extract the four execution-surface hashes from bundle-bound
    shared objects (mechanical; fail-closed on any gap).

    ``resolutions`` maps bundle capability contract -> the seam's
    bound SharedRuntimeResolution. Each surface's responsible object
    must expose the surface hash as a 64-hex string attribute; absent
    shared objects therefore fail closed (this round: always).
    """
    surfaces: Dict[str, str] = {}
    for surface, contract in EXECUTION_SURFACE_CONTRACTS.items():
        resolution = resolutions.get(contract)
        obj = getattr(resolution, "object_ref", None)
        if not getattr(resolution, "bound", False) or obj is None:
            raise VariantBindingError(
                VARIANT_EXECUTION_SURFACE_UNBOUND,
                f"{ctx}: execution surface {surface!r} needs the "
                f"bundle-bound {contract!r} object; the contract is "
                "unbound (the shared runtime is absent — fail closed, "
                "never a default surface)",
            )
        attribute = _EXECUTION_SURFACE_ATTRIBUTES[surface]
        value = getattr(obj, attribute, None)
        if not isinstance(value, str) or len(value) != 64:
            raise VariantBindingError(
                VARIANT_EXECUTION_SURFACE_UNBOUND,
                f"{ctx}: the {contract!r} object exposes no 64-hex "
                f"{attribute!r} surface for {surface!r} (got "
                f"{value!r}); refusing a guessed surface",
            )
        try:
            int(value, 16)
        except ValueError:
            raise VariantBindingError(
                VARIANT_EXECUTION_SURFACE_UNBOUND,
                f"{ctx}: {contract!r}.{attribute} is not hexadecimal: "
                f"{value!r}",
            )
        surfaces[surface] = value
    return surfaces


def bind_executable_pool_from_materials(
    *,
    window: Any,
    compile_result: Any,
    template_artifacts: Tuple[Any, ...],
    execution_surfaces: Mapping[str, str],
    backend_name: str,
    stages_passed: Tuple[str, ...],
) -> Tuple[ExecutableCandidate, ...]:
    """Turn compile+EnvCoder materials into the executable pool.

    ``template_artifacts`` is the materials' (template_hash,
    EnvCoderArtifact, repairs) tuples; each template's family id comes
    from the compile result. Every execution surface hash must be
    present (64-hex) — absent shared surfaces fail closed upstream.
    """
    ctx = "variant_binding.bind_pool"
    if not isinstance(compile_result, CompileResult):
        raise VariantBindingError(
            VARIANT_BAD_TYPE,
            f"{ctx}: compile_result must be a CompileResult, got "
            f"{type(compile_result).__name__}",
        )
    if not isinstance(execution_surfaces, Mapping) or set(
        execution_surfaces
    ) != set(EXECUTION_SURFACE_CONTRACTS):
        raise VariantBindingError(
            VARIANT_EXECUTION_SURFACE_UNBOUND,
            f"{ctx}: execution_surfaces must carry exactly "
            f"{sorted(EXECUTION_SURFACE_CONTRACTS)}, got "
            f"{sorted(execution_surfaces) if isinstance(execution_surfaces, Mapping) else execution_surfaces!r}",
        )
    family_by_template = {
        spec.template_hash: spec.family_id
        for spec in compile_result.specs
    }
    artifacts_by_template: Dict[str, ExecutableEnvironmentArtifact] = {}
    for entry in template_artifacts:
        if not isinstance(entry, (tuple, list)) or len(entry) != 3:
            raise VariantBindingError(
                VARIANT_BAD_TYPE,
                f"{ctx}: template_artifacts entries must be "
                f"(template_hash, EnvCoderArtifact, repairs), got "
                f"{entry!r}",
            )
        template_hash, envcoder_artifact, _repairs = entry
        family_id = family_by_template.get(template_hash)
        if family_id is None:
            raise VariantBindingError(
                VARIANT_BAD_TYPE,
                f"{ctx}: template {template_hash!r} has an EnvCoder "
                "artifact but no compiled spec — the materials are "
                "inconsistent",
            )
        artifacts_by_template[template_hash] = (
            build_executable_environment_artifact(
                envcoder_artifact=envcoder_artifact,
                family_id=family_id,
                observation_action_abi_hash=execution_surfaces[
                    "observation_action_abi_hash"
                ],
                reward_contract_hash=execution_surfaces[
                    "reward_contract_hash"
                ],
                reset_protocol_hash=execution_surfaces[
                    "reset_protocol_hash"
                ],
                seed_policy_hash=execution_surfaces["seed_policy_hash"],
                backend_name=backend_name,
                stages_passed=stages_passed,
            )
        )
    return bind_executable_candidate_pool(
        window=window,
        compile_result=compile_result,
        artifacts_by_template=artifacts_by_template,
    )
