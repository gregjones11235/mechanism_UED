"""CC2 follow-up P0-5 tests: executable candidate binding chain.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
every execution-surface hash below is a conspicuously-marked
SYNTHETIC 64-hex placeholder (the real ABI / reward / reset / seed
surfaces belong to the absent shared runtime). These tests prove the
binding CHAIN and hash discipline — never that anything executed.

Covered negative matrix:
* non-64-hex execution-surface hashes      -> EXEC_HASH_BAD
* non-EnvCoderArtifact / bad stages        -> EXEC_BAD_TYPE
* VOID window                              -> EXEC_VOID_WINDOW
* spec window hash swap                    -> EXEC_SPEC_WINDOW_MISMATCH
* template swap                            -> EXEC_TEMPLATE_MISMATCH
* family swap                              -> EXEC_FAMILY_MISMATCH
* unbound template in pool                 -> EXEC_TEMPLATE_UNBOUND
* duplicate candidate hash                 -> EXEC_DUPLICATE_CANDIDATE
* tampered candidate / chain swap          -> EXEC_CHAIN_MISMATCH
* variant-parameter distinctness (Mode A)  -> different hashes
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import board
from dicode.teachers.e1_formal import executable_candidates as EX
from dicode.teachers.e1_formal.canonical import canonical_sha256
from dicode.teachers.e1_formal.envcoder import EnvCoderArtifact
from dicode.teachers.e1_formal.manifest import BOARD_ROLE_ORDER
from dicode.teachers.e1_formal.task_specs import (
    CompileResult,
    TaskSpec,
    TaskTemplate,
)

# SYNTHETIC execution-surface hashes — TEST_ONLY placeholders
_ABI_HASH = "a1" * 32
_REWARD_HASH = "b2" * 32
_RESET_HASH = "c3" * 32
_SEED_HASH = "d4" * 32


# ---------------------------------------------------------------------------
# fixtures (all TEST_ONLY / SYNTHETIC)
# ---------------------------------------------------------------------------
def _window(status="COMPLETE", void_code="", window_id="e1-w000001"):
    role_results = tuple(
        (role, {"role": role, "note": "TEST_ONLY"})
        for role in BOARD_ROLE_ORDER
    )
    payload = board._window_payload(
        window_id,
        1,
        "FIRST_WINDOW",
        "e" * 64,
        status,
        void_code,
        role_results,
        (),
        (),
    )
    return board.ReviewWindow(
        window_id=window_id,
        session_idx=1,
        trigger_code="FIRST_WINDOW",
        evidence_hash="e" * 64,
        status=status,
        void_code=void_code,
        role_results=role_results,
        surviving_families=(),
        ignored_vetoes=(),
        window_hash=board._compute_window_hash(payload),
    )


def _spec(window, *, family_id, template_hash, variant, variant_params):
    spec_hash = canonical_sha256(
        {
            "window_hash": window.window_hash,
            "template_hash": template_hash,
            "variant": variant,
            "variant_params": [list(pair) for pair in variant_params],
        }
    )
    return TaskSpec(
        spec_id=f"spec-{template_hash[:8]}-v{variant}",
        window_id=window.window_id,
        window_hash=window.window_hash,
        family_id=family_id,
        variant=variant,
        description="TEST_ONLY synthetic spec",
        target_achievements=("test_only:target",),
        axis_changes=(),
        constant_axes=("test_only_axis",),
        scaffolding="TEST_ONLY scaffolding",
        student_must_do="TEST_ONLY student objective",
        template_hash=template_hash,
        template_artifact_id=f"{template_hash}::tpl",
        variant_params=tuple(variant_params),
        spec_hash=spec_hash,
        artifact_id=f"{spec_hash}::v{variant}",
    )


def _envcoder_artifact(template_hash):
    return EnvCoderArtifact(
        template_hash=template_hash,
        artifact_id=f"{template_hash}::tpl",
        env_code="# TEST_ONLY env code\ndef make_env(params):\n    return None\n",
        prompt_envelope_hash="ab" * 32,
    )


def _executable_artifact(template_hash, family_id):
    return EX.build_executable_environment_artifact(
        envcoder_artifact=_envcoder_artifact(template_hash),
        family_id=family_id,
        observation_action_abi_hash=_ABI_HASH,
        reward_contract_hash=_REWARD_HASH,
        reset_protocol_hash=_RESET_HASH,
        seed_policy_hash=_SEED_HASH,
        backend_name="test-only-backend",
        stages_passed=("SYNTAX", "GUARDS"),
    )


def _variant_params(variant):
    return (("difficulty:level", str(variant)), ("axis_a:level", "1/2"))


# ---------------------------------------------------------------------------
# template-level executable artifact
# ---------------------------------------------------------------------------
class TestExecutableEnvironmentArtifact:
    def test_binds_all_execution_surfaces(self):
        template_hash = "f" * 64
        artifact = _executable_artifact(template_hash, "fam_a")
        assert artifact.template_hash == template_hash
        assert artifact.family_id == "fam_a"
        assert artifact.environment_family == "fam_a"
        assert artifact.observation_action_abi_hash == _ABI_HASH
        assert artifact.reward_contract_hash == _REWARD_HASH
        assert artifact.reset_protocol_hash == _RESET_HASH
        assert artifact.seed_policy_hash == _SEED_HASH
        assert len(artifact.envcoder_artifact_hash) == 64
        assert len(artifact.executable_artifact_hash) == 64
        assert artifact.executable_artifact_id.endswith("::env")
        assert len(artifact.provenance_hash) == 64
        # deterministic
        again = _executable_artifact(template_hash, "fam_a")
        assert again.executable_artifact_hash == (
            artifact.executable_artifact_hash
        )

    def test_different_env_code_changes_the_identity(self):
        template_hash = "f" * 64
        first = EX.envcoder_artifact_identity_hash(
            _envcoder_artifact(template_hash)
        )
        other = EnvCoderArtifact(
            template_hash=template_hash,
            artifact_id=f"{template_hash}::tpl",
            env_code="# TEST_ONLY env code (DIFFERENT)\n",
            prompt_envelope_hash="ab" * 32,
        )
        assert EX.envcoder_artifact_identity_hash(other) != first

    def test_bad_execution_surface_hashes_refused(self):
        template_hash = "f" * 64
        for field in (
            "observation_action_abi_hash",
            "reward_contract_hash",
            "reset_protocol_hash",
            "seed_policy_hash",
        ):
            kwargs = dict(
                envcoder_artifact=_envcoder_artifact(template_hash),
                family_id="fam_a",
                observation_action_abi_hash=_ABI_HASH,
                reward_contract_hash=_REWARD_HASH,
                reset_protocol_hash=_RESET_HASH,
                seed_policy_hash=_SEED_HASH,
                backend_name="test-only-backend",
                stages_passed=("SYNTAX",),
            )
            kwargs[field] = "short"
            with pytest.raises(EX.ExecutableCandidateError) as excinfo:
                EX.build_executable_environment_artifact(**kwargs)
            assert excinfo.value.code == EX.EXEC_HASH_BAD

    def test_non_hex_hash_refused(self):
        template_hash = "f" * 64
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.build_executable_environment_artifact(
                envcoder_artifact=_envcoder_artifact(template_hash),
                family_id="fam_a",
                observation_action_abi_hash="z" * 64,
                reward_contract_hash=_REWARD_HASH,
                reset_protocol_hash=_RESET_HASH,
                seed_policy_hash=_SEED_HASH,
                backend_name="test-only-backend",
                stages_passed=("SYNTAX",),
            )
        assert excinfo.value.code == EX.EXEC_HASH_BAD

    def test_non_envcoder_artifact_refused(self):
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.build_executable_environment_artifact(
                envcoder_artifact={"env_code": "summary"},
                family_id="fam_a",
                observation_action_abi_hash=_ABI_HASH,
                reward_contract_hash=_REWARD_HASH,
                reset_protocol_hash=_RESET_HASH,
                seed_policy_hash=_SEED_HASH,
                backend_name="test-only-backend",
                stages_passed=("SYNTAX",),
            )
        assert excinfo.value.code == EX.EXEC_BAD_TYPE

    def test_bad_stages_refused(self):
        template_hash = "f" * 64
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.build_executable_environment_artifact(
                envcoder_artifact=_envcoder_artifact(template_hash),
                family_id="fam_a",
                observation_action_abi_hash=_ABI_HASH,
                reward_contract_hash=_REWARD_HASH,
                reset_protocol_hash=_RESET_HASH,
                seed_policy_hash=_SEED_HASH,
                backend_name="test-only-backend",
                stages_passed=("SYNTAX", ""),
            )
        assert excinfo.value.code == EX.EXEC_BAD_TYPE

    def test_envcoder_identity_hash_requires_real_artifact(self):
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.envcoder_artifact_identity_hash({"artifact": "summary"})
        assert excinfo.value.code == EX.EXEC_BAD_TYPE


# ---------------------------------------------------------------------------
# variant-level executable candidate
# ---------------------------------------------------------------------------
class TestExecutableCandidateBinding:
    def test_binds_the_full_chain(self):
        window = _window()
        template_hash = "f" * 64
        spec = _spec(
            window,
            family_id="fam_a",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        artifact = _executable_artifact(template_hash, "fam_a")
        candidate = EX.bind_executable_candidate(
            window=window, spec=spec, executable_artifact=artifact
        )
        assert candidate.window_hash == window.window_hash
        assert candidate.task_spec_hash == spec.spec_hash
        assert candidate.template_hash == template_hash
        assert candidate.executable_artifact_hash == (
            artifact.executable_artifact_hash
        )
        assert candidate.candidate_id.endswith("::cand")
        assert len(candidate.candidate_hash) == 64
        assert len(candidate.task_params_hash) == 64
        assert len(candidate.provenance_hash) == 64
        # binding alone NEVER executes the variant parameters
        assert candidate.variant_params_executed is False
        assert candidate.execution_marker == (
            EX.VARIANT_PARAMETER_NOT_EXECUTED
        )

    def test_two_variants_are_two_distinct_candidates(self):
        window = _window()
        template_hash = "f" * 64
        artifact = _executable_artifact(template_hash, "fam_a")
        c0 = EX.bind_executable_candidate(
            window=window,
            spec=_spec(
                window,
                family_id="fam_a",
                template_hash=template_hash,
                variant=0,
                variant_params=_variant_params(0),
            ),
            executable_artifact=artifact,
        )
        c1 = EX.bind_executable_candidate(
            window=window,
            spec=_spec(
                window,
                family_id="fam_a",
                template_hash=template_hash,
                variant=1,
                variant_params=_variant_params(1),
            ),
            executable_artifact=artifact,
        )
        assert c0.candidate_hash != c1.candidate_hash
        assert c0.task_params_hash != c1.task_params_hash
        # same executable artifact underneath (Mode A: one template env)
        assert c0.executable_artifact_hash == c1.executable_artifact_hash

    def test_binding_is_deterministic(self):
        window = _window()
        template_hash = "f" * 64
        artifact = _executable_artifact(template_hash, "fam_a")
        spec = _spec(
            window,
            family_id="fam_a",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        first = EX.bind_executable_candidate(
            window=window, spec=spec, executable_artifact=artifact
        )
        again = EX.bind_executable_candidate(
            window=window, spec=spec, executable_artifact=artifact
        )
        assert first.candidate_hash == again.candidate_hash

    def test_void_window_refused(self):
        window = _window(status="VOID", void_code="INCOMPLETE_REVIEW_WINDOW")
        template_hash = "f" * 64
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.bind_executable_candidate(
                window=window,
                spec=_spec(
                    window,
                    family_id="fam_a",
                    template_hash=template_hash,
                    variant=0,
                    variant_params=_variant_params(0),
                ),
                executable_artifact=_executable_artifact(
                    template_hash, "fam_a"
                ),
            )
        assert excinfo.value.code == EX.EXEC_VOID_WINDOW

    def test_spec_window_mismatch_refused(self):
        window = _window()
        other_window = _window(window_id="e1-w000002")
        template_hash = "f" * 64
        spec = _spec(
            other_window,  # bound to a DIFFERENT window
            family_id="fam_a",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.bind_executable_candidate(
                window=window,
                spec=spec,
                executable_artifact=_executable_artifact(
                    template_hash, "fam_a"
                ),
            )
        assert excinfo.value.code == EX.EXEC_SPEC_WINDOW_MISMATCH

    def test_template_mismatch_refused(self):
        window = _window()
        template_hash = "f" * 64
        spec = _spec(
            window,
            family_id="fam_a",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        wrong = _executable_artifact("0" * 64, "fam_a")
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.bind_executable_candidate(
                window=window, spec=spec, executable_artifact=wrong
            )
        assert excinfo.value.code == EX.EXEC_TEMPLATE_MISMATCH

    def test_family_mismatch_refused(self):
        window = _window()
        template_hash = "f" * 64
        spec = _spec(
            window,
            family_id="fam_a",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        wrong = _executable_artifact(template_hash, "fam_other")
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.bind_executable_candidate(
                window=window, spec=spec, executable_artifact=wrong
            )
        assert excinfo.value.code == EX.EXEC_FAMILY_MISMATCH

    def test_chain_verification_detects_tamper(self):
        window = _window()
        template_hash = "f" * 64
        spec = _spec(
            window,
            family_id="fam_a",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        artifact = _executable_artifact(template_hash, "fam_a")
        candidate = EX.bind_executable_candidate(
            window=window, spec=spec, executable_artifact=artifact
        )
        # untampered chain verifies
        EX.verify_candidate_chain(
            candidate,
            window=window,
            spec=spec,
            executable_artifact=artifact,
        )
        # candidate hash tamper
        tampered = replace(candidate, candidate_hash="e" * 64)
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.verify_candidate_chain(
                tampered,
                window=window,
                spec=spec,
                executable_artifact=artifact,
            )
        assert excinfo.value.code == EX.EXEC_CHAIN_MISMATCH
        # source-spec swap (wrong spec for this candidate)
        other_spec = _spec(
            window,
            family_id="fam_a",
            template_hash=template_hash,
            variant=1,
            variant_params=_variant_params(1),
        )
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.verify_candidate_chain(
                candidate,
                window=window,
                spec=other_spec,
                executable_artifact=artifact,
            )
        assert excinfo.value.code == EX.EXEC_CHAIN_MISMATCH


# ---------------------------------------------------------------------------
# pool binding
# ---------------------------------------------------------------------------
class TestPoolBinding:
    def _compile_result(self, window):
        templates = []
        specs = []
        for family_index in range(2):
            template_hash = canonical_sha256(
                {"family": f"fam_{family_index}"}
            )
            templates.append(
                TaskTemplate(
                    family_id=f"fam_{family_index}",
                    template_hash=template_hash,
                    template_artifact_id=f"{template_hash}::tpl",
                )
            )
            for variant in range(2):
                specs.append(
                    _spec(
                        window,
                        family_id=f"fam_{family_index}",
                        template_hash=template_hash,
                        variant=variant,
                        variant_params=_variant_params(variant),
                    )
                )
        return CompileResult(
            templates=tuple(templates), specs=tuple(specs), notes=()
        )

    def test_pool_binds_every_spec_in_compile_order(self):
        window = _window()
        compile_result = self._compile_result(window)
        artifacts = {
            template.template_hash: _executable_artifact(
                template.template_hash, template.family_id
            )
            for template in compile_result.templates
        }
        pool = EX.bind_executable_candidate_pool(
            window=window,
            compile_result=compile_result,
            artifacts_by_template=artifacts,
        )
        assert len(pool) == 4
        assert [c.task_spec_id for c in pool] == [
            spec.spec_id for spec in compile_result.specs
        ]
        hashes = {c.candidate_hash for c in pool}
        assert len(hashes) == 4  # all distinct (variant-sensitive)
        for candidate in pool:
            assert candidate.execution_marker == (
                EX.VARIANT_PARAMETER_NOT_EXECUTED
            )

    def test_unbound_template_refuses_the_pool(self):
        window = _window()
        compile_result = self._compile_result(window)
        artifacts = {
            compile_result.templates[0].template_hash: _executable_artifact(
                compile_result.templates[0].template_hash,
                compile_result.templates[0].family_id,
            )
            # second template deliberately unbound
        }
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.bind_executable_candidate_pool(
                window=window,
                compile_result=compile_result,
                artifacts_by_template=artifacts,
            )
        assert excinfo.value.code == EX.EXEC_TEMPLATE_UNBOUND

    def test_duplicate_candidate_hash_refused(self):
        window = _window()
        template_hash = canonical_sha256({"family": "fam_dup"})
        spec = _spec(
            window,
            family_id="fam_dup",
            template_hash=template_hash,
            variant=0,
            variant_params=_variant_params(0),
        )
        compile_result = CompileResult(
            templates=(
                TaskTemplate(
                    family_id="fam_dup",
                    template_hash=template_hash,
                    template_artifact_id=f"{template_hash}::tpl",
                ),
            ),
            specs=(spec, spec),  # the SAME spec twice
            notes=(),
        )
        artifacts = {
            template_hash: _executable_artifact(template_hash, "fam_dup")
        }
        with pytest.raises(EX.ExecutableCandidateError) as excinfo:
            EX.bind_executable_candidate_pool(
                window=window,
                compile_result=compile_result,
                artifacts_by_template=artifacts,
            )
        assert excinfo.value.code == EX.EXEC_DUPLICATE_CANDIDATE
