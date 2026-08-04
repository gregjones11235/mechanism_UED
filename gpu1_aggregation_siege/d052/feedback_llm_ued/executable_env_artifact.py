"""P0-2 (CC3 follow-up audit): the ExecutableEnvironmentArtifact chain.

The production environment path is a single fail-closed chain::

    AxisDirective batch -> CanonicalTaskSpec -> RealEnvCoderArtifact
        -> ExecutableEnvironmentArtifact (this module, immutable contract)
        -> parameterized candidate instances (bound copies)
        -> SharedCandidateProbeRunner (real probes)
        -> feedback records carrying the SAME artifact hash

An ``ExecutableEnvironmentArtifact`` is the ONLY object a production probe
may execute. It is immutable (extra=forbid CanonicalModel; every carried
hash is recomputed and compared verbatim), it is derived ONLY from a
PASSED ``RealEnvCoderArtifact`` (all four verification links green for
every directive), and it binds — by canonical hash — the task spec, the
directive batch, the per-directive Python sources, the runtime adapter and
the five ABI/protocol hashes declared by the shared runtime's owner.

Missing artifact, unbound candidate, id / hash / family mismatch at the
probe boundary ALL fail closed; there is no symbolic stand-in.

Honesty: locally the shared runtime assets are absent, so this chain is
READY and contract-tested with TEST_ONLY fixtures only — never executed.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from pydantic import Field, model_validator

from d052.bagr_ued.hashing import canonical_sha256, verify_content_hash
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.real_env_coder import (
    STATUS_PASSED,
    CanonicalTaskSpec,
    RealEnvCoderArtifact,
    RealEnvCoderOutput,
)
from d052.feedback_llm_ued.axis_directive import AxisDirective
from d052.feedback_llm_ued.feedback_contracts import CandidateEnvironment
from d052.schemas.common import CanonicalModel, is_sha256_hex

#: the seed policy every bound candidate declares: probe seeds come from
#: the probe's banked schedule (derive_seed_bank over the candidate hash),
#: never from the EnvCoder smoke seed
SEED_POLICY_BANKED_SCHEDULE = "feedback_llm_ued.banked_seed_schedule.v1"

#: canonical id prefix of a derived executable artifact
ARTIFACT_ID_PREFIX = "e2_exec_env_art_"


class ExecutableArtifactBlocked(RuntimeError):
    """The executable-artifact chain failed closed (derivation or probe
    binding)."""


class ExecutableEnvironmentArtifact(CanonicalModel):
    """The immutable executable-environment contract (audit-grade).

    One artifact groups the verified directive bindings of ONE environment
    family within one window. Every carried hash is recomputed and compared
    verbatim (CONTENT_HASH_MISMATCH fails closed); ``artifact_hash`` binds
    the whole payload and ``provenance_hash`` binds the source
    ``RealEnvCoderArtifact``.
    """

    artifact_id: str = Field(min_length=1)
    template_id: str = C.ENVCODER_UNIQUE_TEMPLATE_ID
    source_window: int = Field(ge=0)
    source_plan_id: str = Field(min_length=1)
    canonical_task_spec_hash: str = Field(min_length=1)
    directive_batch_hash: str = Field(min_length=1)
    environment_family: str = Field(min_length=1)
    directive_ids: List[str] = Field(default_factory=list)
    directive_hashes: List[str] = Field(default_factory=list)
    changed_axes: Dict[str, str] = Field(default_factory=dict)
    held_constant_axes: Dict[str, str] = Field(default_factory=dict)
    python_source_hash: str = Field(min_length=1)
    compiled_artifact_hash: str = Field(min_length=1)
    runtime_adapter_id: str = Field(min_length=1)
    observation_abi_hash: str = Field(min_length=1)
    action_abi_hash: str = Field(min_length=1)
    reward_contract_hash: str = Field(min_length=1)
    reset_protocol_hash: str = Field(min_length=1)
    step_protocol_hash: str = Field(min_length=1)
    validation_report_hash: str = Field(min_length=1)
    artifact_hash: str = ""
    provenance_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_and_hash(self) -> "ExecutableEnvironmentArtifact":
        if self.template_id != C.ENVCODER_UNIQUE_TEMPLATE_ID:
            raise ValueError(
                f"ENVCODER_TEMPLATE_NOT_UNIQUE: {self.template_id!r}")
        if self.environment_family not in C.ENVIRONMENT_FAMILIES:
            raise ValueError(
                f"UNKNOWN_ENVIRONMENT_FAMILY: {self.environment_family!r}")
        if not self.directive_ids:
            raise ValueError(
                "EXECUTABLE_ARTIFACT_WITHOUT_DIRECTIVES: at least one "
                "verified directive binding is required")
        if len(set(self.directive_ids)) != len(self.directive_ids):
            raise ValueError(
                f"DUPLICATE_EXECUTABLE_DIRECTIVE_ID: {self.directive_ids}")
        if len(self.directive_hashes) != len(self.directive_ids):
            raise ValueError(
                "EXECUTABLE_ARTIFACT_HASH_COUNT_MISMATCH: "
                f"{len(self.directive_ids)} directive id(s) vs "
                f"{len(self.directive_hashes)} directive hash(es)")
        sha_fields = ["canonical_task_spec_hash", "directive_batch_hash",
                      "python_source_hash", "compiled_artifact_hash",
                      "observation_abi_hash", "action_abi_hash",
                      "reward_contract_hash", "reset_protocol_hash",
                      "step_protocol_hash", "validation_report_hash",
                      "provenance_hash"]
        for field_name in sha_fields:
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise ValueError(
                    f"EXECUTABLE_ARTIFACT_HASH_NOT_SHA256: {field_name}="
                    f"{value!r}")
        for dhash in self.directive_hashes:
            if not is_sha256_hex(dhash):
                raise ValueError(
                    f"EXECUTABLE_ARTIFACT_HASH_NOT_SHA256: directive_hash="
                    f"{dhash!r}")
        computed = verify_content_hash(self.model_dump(),
                                       hash_field="artifact_hash",
                                       carried=self.artifact_hash,
                                       kind="ExecutableEnvironmentArtifact")
        object.__setattr__(self, "artifact_hash", computed)
        return self


# ---------------------------------------------------------------------------
# derivation (PASSED real EnvCoder run -> per-family immutable artifacts)
# ---------------------------------------------------------------------------
def derive_executable_artifacts(*,
                                spec: CanonicalTaskSpec,
                                parsed: RealEnvCoderOutput,
                                source_artifact: RealEnvCoderArtifact,
                                directives: Sequence[AxisDirective],
                                runtime_adapter_id: str,
                                observation_abi_hash: str,
                                action_abi_hash: str,
                                reward_contract_hash: str,
                                reset_protocol_hash: str,
                                step_protocol_hash: str
                                ) -> List[ExecutableEnvironmentArtifact]:
    """Derive the per-family executable artifacts of a PASSED run.

    Fail-closed ladder (nothing is defaulted, nothing is coerced):

    * source artifact not PASSED            -> EXECUTABLE_ARTIFACT_SOURCE_NOT_PASSED
    * any directive link not green          -> EXECUTABLE_ARTIFACT_VERIFICATION_INCOMPLETE
    * spec/window/plan disagreements        -> EXECUTABLE_ARTIFACT_SPEC_MISMATCH /
                                               ..._WINDOW_MISMATCH / ..._PLAN_MISMATCH
    * binding coverage != directive batch   -> EXECUTABLE_ARTIFACT_COVERAGE_MISMATCH
    * directive hash disagreements          -> EXECUTABLE_ARTIFACT_DIRECTIVE_HASH_MISMATCH
    * family disagreements                  -> EXECUTABLE_ARTIFACT_FAMILY_MISMATCH
    * conflicting held axes                 -> EXECUTABLE_ARTIFACT_HELD_AXIS_CONFLICT
    * undeclared runtime adapter            -> EXECUTABLE_ARTIFACT_ADAPTER_UNDECLARED
    """
    if source_artifact.overall_status != STATUS_PASSED:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_SOURCE_NOT_PASSED: source artifact "
            f"overall_status={source_artifact.overall_status!r} — only a "
            "PASSED real EnvCoder run can mint executable artifacts")
    if any(not a.passed for a in source_artifact.directive_artifacts):
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_VERIFICATION_INCOMPLETE: every directive "
            "artifact must have passed all four verification links "
            "(compile/import/reset/step)")
    if source_artifact.spec_hash != spec.spec_hash:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_SPEC_MISMATCH: source artifact binds "
            f"spec_hash={source_artifact.spec_hash!r} but the carried "
            f"spec recomputes to {spec.spec_hash!r}")
    if source_artifact.window != spec.window:
        raise ExecutableArtifactBlocked(
            f"EXECUTABLE_ARTIFACT_WINDOW_MISMATCH: source window="
            f"{source_artifact.window} vs spec window={spec.window}")
    if source_artifact.plan_id != spec.plan_id:
        raise ExecutableArtifactBlocked(
            f"EXECUTABLE_ARTIFACT_PLAN_MISMATCH: source plan="
            f"{source_artifact.plan_id!r} vs spec plan={spec.plan_id!r}")
    if parsed.window != spec.window or parsed.plan_id != spec.plan_id:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_OUTPUT_MISMATCH: the parsed EnvCoder "
            f"output is window={parsed.window} plan={parsed.plan_id!r}, "
            f"the spec is window={spec.window} plan={spec.plan_id!r}")
    if not isinstance(runtime_adapter_id, str) or not runtime_adapter_id:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_ADAPTER_UNDECLARED: the shared runtime "
            "adapter id must be declared explicitly")

    directive_by_id = {d.directive_id: d for d in directives}
    binding_by_id = {b.directive_id: b for b in parsed.directive_bindings}
    artifact_by_id = {a.directive_id: a
                      for a in source_artifact.directive_artifacts}
    spec_ids = set(directive_by_id)
    if set(binding_by_id) != spec_ids or set(artifact_by_id) != spec_ids:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_COVERAGE_MISMATCH: directive batch "
            f"{sorted(spec_ids)} vs parsed bindings "
            f"{sorted(binding_by_id)} vs verified artifacts "
            f"{sorted(artifact_by_id)}")
    for did in sorted(spec_ids):
        directive = directive_by_id[did]
        if binding_by_id[did].directive_hash != directive.directive_hash \
                or artifact_by_id[did].directive_hash \
                != directive.directive_hash:
            raise ExecutableArtifactBlocked(
                f"EXECUTABLE_ARTIFACT_DIRECTIVE_HASH_MISMATCH: directive "
                f"{did!r} hash disagrees between the board directive, the "
                "coder binding and the verified artifact")
        if binding_by_id[did].environment_family \
                != directive.environment_family:
            raise ExecutableArtifactBlocked(
                f"EXECUTABLE_ARTIFACT_FAMILY_MISMATCH: binding for {did!r} "
                f"claims family "
                f"{binding_by_id[did].environment_family!r}, the directive "
                f"belongs to {directive.environment_family!r}")

    #: group the directives by family, preserving batch order
    by_family: Dict[str, List[AxisDirective]] = {}
    for directive in directives:
        by_family.setdefault(directive.environment_family,
                             []).append(directive)

    artifacts: List[ExecutableEnvironmentArtifact] = []
    for family in sorted(by_family):
        dirs = by_family[family]
        ids = [d.directive_id for d in dirs]
        changed_axes = {d.axis: d.new_level for d in dirs}
        held: Dict[str, str] = {}
        for d in dirs:
            for held_axis, held_level in d.held_constant_axes.items():
                if held_axis in changed_axes:
                    raise ExecutableArtifactBlocked(
                        f"EXECUTABLE_ARTIFACT_HELD_AXIS_CONFLICT: axis "
                        f"{held_axis!r} is held constant by {d.directive_id!r} "
                        "but changed within the same family batch")
                if held_axis in held and held[held_axis] != held_level:
                    raise ExecutableArtifactBlocked(
                        f"EXECUTABLE_ARTIFACT_HELD_AXIS_CONFLICT: axis "
                        f"{held_axis!r} held at {held[held_axis]!r} and "
                        f"{held_level!r} within the same family batch")
                held[held_axis] = held_level
        darts = [artifact_by_id[i] for i in ids]
        python_source_hash = canonical_sha256(
            {a.directive_id: a.python_source_hash for a in darts})
        compiled_artifact_hash = canonical_sha256(dict(
            directive_ids=ids,
            compile_links={a.directive_id: [a.compile_status,
                                            a.import_status,
                                            a.reset_status,
                                            a.step_status]
                           for a in darts}))
        validation_report_hash = canonical_sha256(
            [a.model_dump() for a in darts])
        payload = dict(
            artifact_id="",
            template_id=C.ENVCODER_UNIQUE_TEMPLATE_ID,
            source_window=source_artifact.window,
            source_plan_id=source_artifact.plan_id,
            canonical_task_spec_hash=spec.spec_hash,
            directive_batch_hash=spec.directive_batch_hash,
            environment_family=family,
            directive_ids=ids,
            directive_hashes=[directive_by_id[i].directive_hash
                              for i in ids],
            changed_axes=changed_axes,
            held_constant_axes=held,
            python_source_hash=python_source_hash,
            compiled_artifact_hash=compiled_artifact_hash,
            runtime_adapter_id=runtime_adapter_id,
            observation_abi_hash=observation_abi_hash,
            action_abi_hash=action_abi_hash,
            reward_contract_hash=reward_contract_hash,
            reset_protocol_hash=reset_protocol_hash,
            step_protocol_hash=step_protocol_hash,
            validation_report_hash=validation_report_hash,
            artifact_hash="",
            provenance_hash=source_artifact.artifact_hash)
        content_hash = canonical_sha256(payload)
        artifacts.append(ExecutableEnvironmentArtifact(
            **dict(payload,
                   artifact_id=f"{ARTIFACT_ID_PREFIX}{content_hash[:16]}")))
    if not artifacts:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_DERIVATION_EMPTY: the directive batch "
            "produced no family group")
    return artifacts


# ---------------------------------------------------------------------------
# candidate binding (parameterized instances entering the probe)
# ---------------------------------------------------------------------------
def parameter_variant_hash_of(candidate: CandidateEnvironment) -> str:
    """Canonical hash of the candidate's OWN parameterization (the axes it
    moves, the axes it holds, its variant identity)."""
    return canonical_sha256(dict(
        axis_values=dict(candidate.axis_values),
        held_constant_axes=dict(candidate.held_constant_axes),
        variant_id=candidate.variant_id,
        variant_kind=candidate.variant_kind))


def seed_policy_hash_of(candidate: CandidateEnvironment) -> str:
    """The candidate's declared seed policy: probe seeds come from the
    banked schedule derived over the candidate hash — never from the
    EnvCoder smoke seed."""
    return canonical_sha256(dict(
        policy=SEED_POLICY_BANKED_SCHEDULE,
        candidate_hash=candidate.candidate_hash))


def bind_candidate_to_artifact(candidate: CandidateEnvironment,
                               artifact: ExecutableEnvironmentArtifact
                               ) -> CandidateEnvironment:
    """Return the BOUND parameterized candidate (a NEW object; the
    candidate_hash recomputes over the bound content)."""
    if candidate.environment_family != artifact.environment_family:
        raise ExecutableArtifactBlocked(
            f"EXECUTABLE_ARTIFACT_FAMILY_MISMATCH: candidate "
            f"{candidate.candidate_id!r} is family "
            f"{candidate.environment_family!r} but the artifact realizes "
            f"{artifact.environment_family!r}")
    dump = candidate.model_dump()
    dump.update(executable_artifact_id=artifact.artifact_id,
                executable_artifact_hash=artifact.artifact_hash,
                parameter_variant_hash=parameter_variant_hash_of(candidate),
                seed_policy_hash=seed_policy_hash_of(candidate),
                #: un-carry the old hash: the bound candidate is a NEW
                #: object and recomputes it
                candidate_hash="")
    return CandidateEnvironment(**dump)


def assert_candidate_artifact_binding(candidate: CandidateEnvironment,
                                      artifact: ExecutableEnvironmentArtifact
                                      ) -> None:
    """Probe-boundary check: the candidate must be bound to EXACTLY this
    artifact (id, content hash, family) and declare both hashes."""
    if not candidate.executable_artifact_id:
        raise ExecutableArtifactBlocked(
            f"EXECUTABLE_ARTIFACT_UNBOUND: candidate "
            f"{candidate.candidate_id!r} carries no executable artifact "
            "binding; production probes refuse unbound candidates")
    if candidate.executable_artifact_id != artifact.artifact_id:
        raise ExecutableArtifactBlocked(
            f"EXECUTABLE_ARTIFACT_ID_MISMATCH: candidate binds "
            f"{candidate.executable_artifact_id!r} but the probe holds "
            f"{artifact.artifact_id!r}")
    if candidate.executable_artifact_hash != artifact.artifact_hash:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_HASH_MISMATCH: candidate binds artifact "
            f"hash {candidate.executable_artifact_hash!r} but the artifact "
            f"recomputes to {artifact.artifact_hash!r}")
    if candidate.environment_family != artifact.environment_family:
        raise ExecutableArtifactBlocked(
            f"EXECUTABLE_ARTIFACT_FAMILY_MISMATCH: candidate family "
            f"{candidate.environment_family!r} vs artifact family "
            f"{artifact.environment_family!r}")
    if not candidate.parameter_variant_hash \
            or not candidate.seed_policy_hash:
        raise ExecutableArtifactBlocked(
            "EXECUTABLE_ARTIFACT_BINDING_INCOMPLETE: a bound candidate "
            "must declare BOTH the parameter-variant hash and the "
            "seed-policy hash")


__all__ = [
    "SEED_POLICY_BANKED_SCHEDULE", "ARTIFACT_ID_PREFIX",
    "ExecutableArtifactBlocked", "ExecutableEnvironmentArtifact",
    "derive_executable_artifacts", "parameter_variant_hash_of",
    "seed_policy_hash_of", "bind_candidate_to_artifact",
    "assert_candidate_artifact_binding",
]
