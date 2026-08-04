"""CC2 follow-up P0-5/P0-6: executable candidate binding.

The one-window pipeline binds every candidate to the FULL artifact
chain — nothing stands alone, nothing is summarized away::

    ReviewWindow -> CompileResult -> TaskTemplate -> TaskSpec
        -> EnvCoderArtifact -> ExecutableEnvironmentArtifact
        -> ExecutableCandidate -> CandidateProbeRunner

``ExecutableEnvironmentArtifact`` is the TEMPLATE-level executable
environment (the EnvCoder artifact plus its execution-surface
identities: observation/action ABI, reward contract, reset protocol,
seed policy — each a REAL 64-hex hash, never a default).

``ExecutableCandidate`` is the VARIANT-level executable record: the
template's executable artifact + the variant's deterministic
TaskParams. Its hash binds the variant parameters, so two variants
of one template are two DIFFERENT executable candidates (Mode A:
parameterized template — different variant_params => different
candidate hash). Until the variant parameters are actually executed
by a real backend, the candidate carries the conspicuous marker
``VARIANT_PARAMETER_NOT_EXECUTED`` — never silently cleared.

Production discipline: every execution-surface hash must come from
the shared runtime (absent this round), so a production candidate
pool cannot bind yet — fail closed, honest. TEST_ONLY bindings carry
conspicuously-marked synthetic hashes and prove the chain only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .board import ReviewWindow, WINDOW_STATUS_COMPLETE
from .canonical import canonical_sha256
from .envcoder import EnvCoderArtifact
from .schemas import E1SchemaError
from .task_specs import CompileResult, TaskSpec

#: conspicuous marker: variant parameters are bound into the candidate
#: hash but have NOT been executed by a real backend yet (P0-6). Only
#: the authorized real validation path may clear it (later commit).
VARIANT_PARAMETER_NOT_EXECUTED = "VARIANT_PARAMETER_NOT_EXECUTED"

#: binding provenance label (versioned)
EXECUTABLE_CANDIDATE_BINDING_VERSION = "e1-executable-candidate-binding-v1"

# fail-closed codes (greppable)
EXEC_BAD_TYPE = "EXEC_BAD_TYPE"
EXEC_HASH_BAD = "EXEC_HASH_BAD"
EXEC_VOID_WINDOW = "EXEC_VOID_WINDOW"
EXEC_SPEC_WINDOW_MISMATCH = "EXEC_SPEC_WINDOW_MISMATCH"
EXEC_TEMPLATE_MISMATCH = "EXEC_TEMPLATE_MISMATCH"
EXEC_FAMILY_MISMATCH = "EXEC_FAMILY_MISMATCH"
EXEC_TEMPLATE_UNBOUND = "EXEC_TEMPLATE_UNBOUND"
EXEC_DUPLICATE_CANDIDATE = "EXEC_DUPLICATE_CANDIDATE"
EXEC_CHAIN_MISMATCH = "EXEC_CHAIN_MISMATCH"


class ExecutableCandidateError(E1SchemaError):
    """Fail-closed executable-candidate violation; ``code`` is
    greppable."""


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ExecutableCandidateError(
            EXEC_HASH_BAD,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    try:
        int(value, 16)
    except ValueError:
        raise ExecutableCandidateError(
            EXEC_HASH_BAD,
            f"{ctx}: {name} is not hexadecimal: {value!r}",
        )
    return value


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: {name} must be a non-empty str, got {value!r}",
        )
    return value.strip()


def envcoder_artifact_identity_hash(artifact: Any) -> str:
    """Canonical identity of one EnvCoderArtifact (deterministic)."""
    if not isinstance(artifact, EnvCoderArtifact):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"executable.binding: expected an EnvCoderArtifact, got "
            f"{type(artifact).__name__}",
        )
    return canonical_sha256(
        {
            "template_hash": artifact.template_hash,
            "artifact_id": artifact.artifact_id,
            "env_code": artifact.env_code,
            "prompt_envelope_hash": artifact.prompt_envelope_hash,
        }
    )


@dataclass(frozen=True)
class ExecutableEnvironmentArtifact:
    """The TEMPLATE-level executable environment (immutable).

    Every execution-surface identity is a real 64-hex hash bound at
    construction; the seam that supplies them (shared runtime) is
    fail-closed — no defaults, no guesses.
    """

    template_hash: str
    template_artifact_id: str
    family_id: str
    envcoder_artifact_id: str
    envcoder_artifact_hash: str
    env_code: str
    environment_family: str
    executable_artifact_id: str
    executable_artifact_hash: str
    observation_action_abi_hash: str
    reward_contract_hash: str
    reset_protocol_hash: str
    seed_policy_hash: str
    backend_name: str
    stages_passed: Tuple[str, ...]
    provenance_hash: str


@dataclass(frozen=True)
class ExecutableCandidate:
    """The VARIANT-level executable candidate (immutable, hash-bound).

    ``candidate_hash`` binds the window, the spec (whose hash already
    binds template + variant + variant_params), the executable
    artifact AND the task-params hash: different variant parameters
    => different candidate hash, mechanically.
    ``variant_params_executed`` stays False with the conspicuous
    marker until a real backend actually executes those parameters.
    """

    candidate_id: str
    family_id: str
    window_id: str
    window_hash: str
    task_spec_id: str
    task_spec_hash: str
    template_id: str
    template_hash: str
    envcoder_artifact_id: str
    envcoder_artifact_hash: str
    executable_artifact_id: str
    executable_artifact_hash: str
    environment_family: str
    variant_index: int
    variant_params: Tuple[Tuple[str, str], ...]
    task_params_hash: str
    seed_policy_hash: str
    reset_protocol_hash: str
    observation_action_abi_hash: str
    reward_contract_hash: str
    candidate_hash: str
    provenance_hash: str
    variant_params_executed: bool
    execution_marker: str


def build_executable_environment_artifact(
    *,
    envcoder_artifact: Any,
    family_id: str,
    observation_action_abi_hash: str,
    reward_contract_hash: str,
    reset_protocol_hash: str,
    seed_policy_hash: str,
    backend_name: str,
    stages_passed: Tuple[str, ...],
) -> ExecutableEnvironmentArtifact:
    """Bind ONE EnvCoderArtifact into its executable environment.

    Every execution-surface hash is REQUIRED and validated (64-hex);
    absent shared surfaces therefore fail closed here — a production
    executable artifact cannot exist on defaults.
    """
    ctx = "executable.build_artifact"
    if not isinstance(envcoder_artifact, EnvCoderArtifact):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: envcoder_artifact must be an EnvCoderArtifact, "
            f"got {type(envcoder_artifact).__name__}",
        )
    family_id = _require_non_empty_str(family_id, "family_id", ctx)
    abi_hash = _require_sha64(
        observation_action_abi_hash, "observation_action_abi_hash", ctx
    )
    reward_hash = _require_sha64(
        reward_contract_hash, "reward_contract_hash", ctx
    )
    reset_hash = _require_sha64(
        reset_protocol_hash, "reset_protocol_hash", ctx
    )
    seed_hash = _require_sha64(seed_policy_hash, "seed_policy_hash", ctx)
    backend_name = _require_non_empty_str(backend_name, "backend_name", ctx)
    if not isinstance(stages_passed, (tuple, list)) or not all(
        isinstance(stage, str) and stage for stage in stages_passed
    ):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: stages_passed must be a sequence of non-empty "
            f"stage names, got {stages_passed!r}",
        )
    envcoder_hash = envcoder_artifact_identity_hash(envcoder_artifact)
    executable_artifact_hash = canonical_sha256(
        {
            "binding_version": EXECUTABLE_CANDIDATE_BINDING_VERSION,
            "envcoder_artifact_hash": envcoder_hash,
            "template_hash": envcoder_artifact.template_hash,
            "family_id": family_id,
            "observation_action_abi_hash": abi_hash,
            "reward_contract_hash": reward_hash,
            "reset_protocol_hash": reset_hash,
            "seed_policy_hash": seed_hash,
            "backend_name": backend_name,
            "stages_passed": list(stages_passed),
        }
    )
    provenance_hash = canonical_sha256(
        {
            "binding_version": EXECUTABLE_CANDIDATE_BINDING_VERSION,
            "executable_artifact_hash": executable_artifact_hash,
            "kind": "executable_environment_artifact",
        }
    )
    return ExecutableEnvironmentArtifact(
        template_hash=envcoder_artifact.template_hash,
        template_artifact_id=envcoder_artifact.artifact_id,
        family_id=family_id,
        envcoder_artifact_id=envcoder_artifact.artifact_id,
        envcoder_artifact_hash=envcoder_hash,
        env_code=envcoder_artifact.env_code,
        environment_family=family_id,
        executable_artifact_id=f"{executable_artifact_hash}::env",
        executable_artifact_hash=executable_artifact_hash,
        observation_action_abi_hash=abi_hash,
        reward_contract_hash=reward_hash,
        reset_protocol_hash=reset_hash,
        seed_policy_hash=seed_hash,
        backend_name=backend_name,
        stages_passed=tuple(stages_passed),
        provenance_hash=provenance_hash,
    )


def compute_task_params_hash(spec: Any, ctx: str) -> str:
    """Canonical hash of the variant's deterministic TaskParams."""
    if not isinstance(spec, TaskSpec):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: spec must be a TaskSpec, got {type(spec).__name__}",
        )
    return canonical_sha256(
        {
            "task_spec_hash": spec.spec_hash,
            "variant": spec.variant,
            "variant_params": [
                [name, value] for name, value in spec.variant_params
            ],
        }
    )


def compute_candidate_hash(
    *,
    window_hash: str,
    spec: TaskSpec,
    executable_artifact: ExecutableEnvironmentArtifact,
    task_params_hash: str,
) -> str:
    """The candidate identity: window x spec(variant params) x
    executable artifact x execution surfaces."""
    return canonical_sha256(
        {
            "binding_version": EXECUTABLE_CANDIDATE_BINDING_VERSION,
            "window_hash": window_hash,
            "task_spec_hash": spec.spec_hash,
            "template_hash": spec.template_hash,
            "variant": spec.variant,
            "variant_params": [
                [name, value] for name, value in spec.variant_params
            ],
            "task_params_hash": task_params_hash,
            "executable_artifact_hash": (
                executable_artifact.executable_artifact_hash
            ),
            "seed_policy_hash": executable_artifact.seed_policy_hash,
            "reset_protocol_hash": executable_artifact.reset_protocol_hash,
            "observation_action_abi_hash": (
                executable_artifact.observation_action_abi_hash
            ),
            "reward_contract_hash": (
                executable_artifact.reward_contract_hash
            ),
        }
    )


def bind_executable_candidate(
    *,
    window: Any,
    spec: Any,
    executable_artifact: Any,
) -> ExecutableCandidate:
    """Bind ONE (window, spec, executable artifact) triple fail-closed.

    The candidate carries the conspicuous
    ``VARIANT_PARAMETER_NOT_EXECUTED`` marker: binding alone never
    executes the variant parameters.
    """
    ctx = "executable.bind_candidate"
    if not isinstance(window, ReviewWindow):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: window must be a ReviewWindow, got "
            f"{type(window).__name__}",
        )
    if not isinstance(spec, TaskSpec):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: spec must be a TaskSpec, got {type(spec).__name__}",
        )
    if not isinstance(executable_artifact, ExecutableEnvironmentArtifact):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: executable_artifact must be an "
            f"ExecutableEnvironmentArtifact, got "
            f"{type(executable_artifact).__name__}",
        )
    if window.status != WINDOW_STATUS_COMPLETE:
        raise ExecutableCandidateError(
            EXEC_VOID_WINDOW,
            f"{ctx}: window {window.window_id} is {window.status} "
            f"({window.void_code}); executable candidates bind ONLY to "
            "COMPLETE windows",
        )
    if spec.window_hash != window.window_hash:
        raise ExecutableCandidateError(
            EXEC_SPEC_WINDOW_MISMATCH,
            f"{ctx}: spec {spec.spec_id} binds window hash "
            f"{spec.window_hash!r} but the window carries "
            f"{window.window_hash!r}",
        )
    if spec.template_hash != executable_artifact.template_hash:
        raise ExecutableCandidateError(
            EXEC_TEMPLATE_MISMATCH,
            f"{ctx}: spec template {spec.template_hash!r} != "
            f"executable artifact template "
            f"{executable_artifact.template_hash!r}",
        )
    if spec.family_id != executable_artifact.family_id:
        raise ExecutableCandidateError(
            EXEC_FAMILY_MISMATCH,
            f"{ctx}: spec family {spec.family_id!r} != executable "
            f"artifact family {executable_artifact.family_id!r}",
        )
    task_params_hash = compute_task_params_hash(spec, ctx)
    candidate_hash = compute_candidate_hash(
        window_hash=window.window_hash,
        spec=spec,
        executable_artifact=executable_artifact,
        task_params_hash=task_params_hash,
    )
    provenance_hash = canonical_sha256(
        {
            "binding_version": EXECUTABLE_CANDIDATE_BINDING_VERSION,
            "kind": "executable_candidate",
            "candidate_hash": candidate_hash,
            "window_id": window.window_id,
        }
    )
    return ExecutableCandidate(
        candidate_id=f"{candidate_hash}::cand",
        family_id=spec.family_id,
        window_id=window.window_id,
        window_hash=window.window_hash,
        task_spec_id=spec.spec_id,
        task_spec_hash=spec.spec_hash,
        template_id=spec.template_artifact_id,
        template_hash=spec.template_hash,
        envcoder_artifact_id=executable_artifact.envcoder_artifact_id,
        envcoder_artifact_hash=executable_artifact.envcoder_artifact_hash,
        executable_artifact_id=(
            executable_artifact.executable_artifact_id
        ),
        executable_artifact_hash=(
            executable_artifact.executable_artifact_hash
        ),
        environment_family=executable_artifact.environment_family,
        variant_index=spec.variant,
        variant_params=spec.variant_params,
        task_params_hash=task_params_hash,
        seed_policy_hash=executable_artifact.seed_policy_hash,
        reset_protocol_hash=executable_artifact.reset_protocol_hash,
        observation_action_abi_hash=(
            executable_artifact.observation_action_abi_hash
        ),
        reward_contract_hash=executable_artifact.reward_contract_hash,
        candidate_hash=candidate_hash,
        provenance_hash=provenance_hash,
        variant_params_executed=False,
        execution_marker=VARIANT_PARAMETER_NOT_EXECUTED,
    )


def bind_executable_candidate_pool(
    *,
    window: Any,
    compile_result: Any,
    artifacts_by_template: Mapping[str, Any],
) -> Tuple[ExecutableCandidate, ...]:
    """Bind the window's WHOLE compiled pool (deterministic order).

    Every compiled spec's template must carry a bound executable
    artifact (no silent skips); duplicate candidate hashes fail
    closed. The pool is the exact compile order of the window.
    """
    ctx = "executable.bind_pool"
    if not isinstance(compile_result, CompileResult):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: compile_result must be a CompileResult, got "
            f"{type(compile_result).__name__}",
        )
    if not isinstance(artifacts_by_template, Mapping):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: artifacts_by_template must be a mapping, got "
            f"{type(artifacts_by_template).__name__}",
        )
    pool = []
    seen_hashes = set()
    for spec in compile_result.specs:
        artifact = artifacts_by_template.get(spec.template_hash)
        if artifact is None:
            raise ExecutableCandidateError(
                EXEC_TEMPLATE_UNBOUND,
                f"{ctx}: template {spec.template_hash!r} (family "
                f"{spec.family_id!r}) has no bound executable artifact; "
                "the pool is refused rather than padded",
            )
        candidate = bind_executable_candidate(
            window=window, spec=spec, executable_artifact=artifact
        )
        if candidate.candidate_hash in seen_hashes:
            raise ExecutableCandidateError(
                EXEC_DUPLICATE_CANDIDATE,
                f"{ctx}: duplicate candidate hash "
                f"{candidate.candidate_hash!r}; every variant must be a "
                "distinct executable candidate",
            )
        seen_hashes.add(candidate.candidate_hash)
        pool.append(candidate)
    return tuple(pool)


def verify_candidate_chain(
    candidate: Any,
    *,
    window: Any,
    spec: Any,
    executable_artifact: Any,
    ctx: str = "executable.verify_chain",
) -> None:
    """Re-derive EVERY link of one candidate against its sources.

    ReviewWindow -> TaskSpec -> ExecutableEnvironmentArtifact ->
    ExecutableCandidate: any mismatch fails closed. This is the
    mechanical chain check the probe stage consumes before trusting a
    candidate hash.
    """
    if not isinstance(candidate, ExecutableCandidate):
        raise ExecutableCandidateError(
            EXEC_BAD_TYPE,
            f"{ctx}: candidate must be an ExecutableCandidate, got "
            f"{type(candidate).__name__}",
        )
    if candidate.window_hash != window.window_hash:
        raise ExecutableCandidateError(
            EXEC_CHAIN_MISMATCH,
            f"{ctx}: candidate window_hash {candidate.window_hash!r} "
            f"!= window {window.window_hash!r}",
        )
    if candidate.task_spec_hash != spec.spec_hash:
        raise ExecutableCandidateError(
            EXEC_CHAIN_MISMATCH,
            f"{ctx}: candidate task_spec_hash "
            f"{candidate.task_spec_hash!r} != spec {spec.spec_hash!r}",
        )
    if candidate.template_hash != spec.template_hash:
        raise ExecutableCandidateError(
            EXEC_CHAIN_MISMATCH,
            f"{ctx}: candidate template_hash "
            f"{candidate.template_hash!r} != spec template "
            f"{spec.template_hash!r}",
        )
    if candidate.executable_artifact_hash != (
        executable_artifact.executable_artifact_hash
    ):
        raise ExecutableCandidateError(
            EXEC_CHAIN_MISMATCH,
            f"{ctx}: candidate executable_artifact_hash "
            f"{candidate.executable_artifact_hash!r} != artifact "
            f"{executable_artifact.executable_artifact_hash!r}",
        )
    task_params_hash = compute_task_params_hash(spec, ctx)
    if candidate.task_params_hash != task_params_hash:
        raise ExecutableCandidateError(
            EXEC_CHAIN_MISMATCH,
            f"{ctx}: candidate task_params_hash "
            f"{candidate.task_params_hash!r} != re-derived "
            f"{task_params_hash!r}",
        )
    recomputed = compute_candidate_hash(
        window_hash=window.window_hash,
        spec=spec,
        executable_artifact=executable_artifact,
        task_params_hash=task_params_hash,
    )
    if candidate.candidate_hash != recomputed:
        raise ExecutableCandidateError(
            EXEC_CHAIN_MISMATCH,
            f"{ctx}: candidate_hash {candidate.candidate_hash!r} != "
            f"re-derived {recomputed!r} (tampered candidate)",
        )
