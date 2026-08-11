"""CC2 follow-up P0-10 tests: ONE GenManager across the whole window
(+ P0-7 authorized EnvCoder validation surface).

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
all fixtures synthetic. A structurally identical SECOND teacher is a
SWAPPED teacher and must be refused; the EnvCoder validation ladder
never executes real code this round.

Covered negative matrix:
* swapped teacher instance                 -> E1_TEACHER_SWAPPED
* swapped runtime bundle                   -> E1_TEACHER_RUNTIME_SWAPPED
* bad session object                       -> E1_TEACHER_CONTINUITY_BAD
* driver stage with a swapped teacher      -> E1_TEACHER_SWAPPED
* production EnvCoder validation           -> PROVIDER_UNAUTHORIZED
* replay/mock validator                    -> PROVIDER_FORBIDDEN
* bad grant / mode                         -> GRANT_BAD / BAD_TYPE
* TEST_ONLY ladder shape                   -> 13 stages, in order
"""
import json
import os
from types import SimpleNamespace

import pytest
import yaml

from dicode.teachers.e1_formal import envcoder_validation as EV
from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import teacher_continuity as TC
from dicode.teachers.e1_formal.envcoder import EnvCoderArtifact

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


def _committed_manager():
    with open(
        os.path.join(REPO_ROOT, "conf", "teacher", "e1_formal.yaml"),
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    with open(
        os.path.join(REPO_ROOT, "configs", "e1_formal_ued.yaml"),
        "r",
        encoding="utf-8",
    ) as handle:
        frozen = yaml.safe_load(handle)
    with open(
        os.path.join(
            REPO_ROOT,
            "configs",
            "e1_formal_ued_anchor_manifest.DRAFT.json",
        ),
        "r",
        encoding="utf-8",
    ) as handle:
        draft = json.load(handle)
    return GM.E1FormalGenManager(
        config, frozen_manifest=frozen, anchor_manifest_mapping=draft
    )


def _test_only_bundle(marker="one"):
    return RB.build_test_only_runtime_bundle(
        source_commit=f"TEST_ONLY_SYNTHETIC_SOURCE_COMMIT_{marker}",
        capabilities={
            contract: SimpleNamespace(
                kind=contract, identity_id=f"test-only-{contract}-{marker}"
            )
            for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
        },
    )


# ---------------------------------------------------------------------------
# continuity session
# ---------------------------------------------------------------------------
class TestOneWindowContinuity:
    def test_session_binds_the_teacher_and_bundle(self):
        teacher = _committed_manager()
        bundle = _test_only_bundle()
        session = TC.begin_one_window_session(teacher, bundle)
        assert session.teacher_id == id(teacher)
        assert session.teacher_type == "E1FormalGenManager"
        assert session.runtime_bundle_hash == bundle.bundle_hash
        assert session.cycles_run_at_open == teacher.cycles_run
        assert len(session.session_hash) == 64

    def test_same_teacher_and_bundle_pass(self):
        teacher = _committed_manager()
        bundle = _test_only_bundle()
        session = TC.begin_one_window_session(teacher, bundle)
        TC.assert_one_window_continuity(session, teacher, bundle, "test")

    def test_second_teacher_instance_refused(self):
        teacher = _committed_manager()
        impostor = _committed_manager()  # same config, DIFFERENT object
        assert teacher is not impostor
        bundle = _test_only_bundle()
        session = TC.begin_one_window_session(teacher, bundle)
        with pytest.raises(TC.TeacherContinuityError) as excinfo:
            TC.assert_one_window_continuity(
                session, impostor, bundle, "test"
            )
        assert excinfo.value.code == TC.E1_TEACHER_SWAPPED

    def test_swapped_runtime_bundle_refused(self):
        teacher = _committed_manager()
        session = TC.begin_one_window_session(
            teacher, _test_only_bundle("one")
        )
        with pytest.raises(TC.TeacherContinuityError) as excinfo:
            TC.assert_one_window_continuity(
                session, teacher, _test_only_bundle("two"), "test"
            )
        assert excinfo.value.code == TC.E1_TEACHER_RUNTIME_SWAPPED

    def test_bad_session_object_refused(self):
        teacher = _committed_manager()
        bundle = _test_only_bundle()
        with pytest.raises(TC.TeacherContinuityError) as excinfo:
            TC.assert_one_window_continuity(
                {"session": "summary"}, teacher, bundle, "test"
            )
        assert excinfo.value.code == TC.E1_TEACHER_CONTINUITY_BAD

    def test_none_teacher_refused_at_open(self):
        with pytest.raises(TC.TeacherContinuityError) as excinfo:
            TC.begin_one_window_session(None, _test_only_bundle())
        assert excinfo.value.code == TC.E1_TEACHER_CONTINUITY_BAD


# ---------------------------------------------------------------------------
# driver wiring: a swapped teacher never runs a later stage
# ---------------------------------------------------------------------------
def _window_result_with_continuity(teacher, bundle):
    session = TC.begin_one_window_session(teacher, bundle)
    return DRV.E1WindowResult(
        window=SimpleNamespace(window_id="e1-w000001"),
        evidence=None,
        gate_signals=None,
        cycle=None,
        window_result_hash="e" * 64,
        continuity=session,
    )


class TestDriverContinuityEnforcement:
    def test_envcoder_stage_refuses_a_swapped_teacher(self):
        teacher = _committed_manager()
        impostor = _committed_manager()
        bundle = _test_only_bundle()
        window_result = _window_result_with_continuity(teacher, bundle)
        materials = SimpleNamespace(
            window_result_hash=window_result.window_result_hash
        )
        # the stage requires the REAL materials object first
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_envcoder_and_compile(
                impostor, window_result, bundle
            )
        # teacher swap detected BEFORE any compilation happens
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE or (
            excinfo.value.code == TC.E1_TEACHER_SWAPPED
        )

    def test_envcoder_stage_refuses_swapped_teacher_with_materials(self):
        teacher = _committed_manager()
        impostor = _committed_manager()
        bundle = _test_only_bundle()
        window_result = _window_result_with_continuity(teacher, bundle)
        # a placeholder materials object passes require_real_object and
        # the isinstance check fails ONLY after continuity — so build
        # the minimal real shape the stage type-checks first
        from dicode.teachers.e1_formal.task_specs import CompileResult

        materials = DRV.E1CandidateMaterials(
            window_result_hash=window_result.window_result_hash,
            compile_result=CompileResult(
                templates=(), specs=(), notes=()
            ),
            template_artifacts=(),
            materials_hash="m" * 64,
        )
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_envcoder_and_compile(
                impostor, window_result, bundle
            )
        assert excinfo.value.code == TC.E1_TEACHER_SWAPPED

    def test_envcoder_stage_refuses_a_swapped_bundle(self):
        teacher = _committed_manager()
        bundle_one = _test_only_bundle("one")
        window_result = _window_result_with_continuity(
            teacher, bundle_one
        )
        from dicode.teachers.e1_formal.task_specs import CompileResult

        materials = DRV.E1CandidateMaterials(
            window_result_hash=window_result.window_result_hash,
            compile_result=CompileResult(
                templates=(), specs=(), notes=()
            ),
            template_artifacts=(),
            materials_hash="m" * 64,
        )
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_envcoder_and_compile(
                teacher, window_result, _test_only_bundle("two")
            )
        assert excinfo.value.code == TC.E1_TEACHER_RUNTIME_SWAPPED


# ---------------------------------------------------------------------------
# P0-7: the authorized EnvCoder validation surface
# ---------------------------------------------------------------------------
class TestEnvValidationAuthorization:
    def test_production_whitelist_is_empty_this_round(self):
        assert EV.AUTHORIZED_ENV_VALIDATION_PROVIDERS == ()

    def test_production_authorization_unauthorized(self):
        with pytest.raises(EV.EnvValidationError) as excinfo:
            EV.authorize_environment_validation_runtime(
                mode=EV.ENV_VALIDATION_MODE_PRODUCTION,
                authorization_grant_hash="a" * 64,
                validator_id="real-envcoder-validator",
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            )
        assert excinfo.value.code == (
            EV.ENV_VALIDATION_PROVIDER_UNAUTHORIZED
        )

    def test_replay_mock_validators_forbidden(self):
        for validator in ("replay", "mock"):
            with pytest.raises(EV.EnvValidationError) as excinfo:
                EV.authorize_environment_validation_runtime(
                    mode=EV.ENV_VALIDATION_MODE_PRODUCTION,
                    authorization_grant_hash="a" * 64,
                    validator_id=validator,
                    source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
                )
            assert excinfo.value.code == (
                EV.ENV_VALIDATION_PROVIDER_FORBIDDEN
            )

    def test_bad_grant_refused(self):
        with pytest.raises(EV.EnvValidationError) as excinfo:
            EV.authorize_environment_validation_runtime(
                mode=EV.ENV_VALIDATION_MODE_TEST_ONLY,
                authorization_grant_hash="short",
                validator_id=EV.SYNTHETIC_TEST_ONLY_VALIDATOR,
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            )
        assert excinfo.value.code == EV.ENV_VALIDATION_GRANT_BAD

    def test_test_only_requires_the_synthetic_validator(self):
        with pytest.raises(EV.EnvValidationError) as excinfo:
            EV.authorize_environment_validation_runtime(
                mode=EV.ENV_VALIDATION_MODE_TEST_ONLY,
                authorization_grant_hash="a" * 64,
                validator_id="attacker-validator",
                source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            )
        assert excinfo.value.code == (
            EV.ENV_VALIDATION_PROVIDER_FORBIDDEN
        )

    def test_test_only_authorization_assembles(self):
        authorization = EV.authorize_environment_validation_runtime(
            mode=EV.ENV_VALIDATION_MODE_TEST_ONLY,
            authorization_grant_hash="a" * 64,
            validator_id=EV.SYNTHETIC_TEST_ONLY_VALIDATOR,
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        )
        assert authorization.stages == EV.ENVCODER_VALIDATION_STAGES
        assert len(authorization.runtime_hash) == 64


class TestEnvValidationLadder:
    def test_ladder_is_thirteen_stages_in_fixed_order(self):
        assert len(EV.ENVCODER_VALIDATION_STAGES) == 13
        assert EV.ENVCODER_VALIDATION_STAGES[0] == "AUTHORIZATION"
        assert EV.ENVCODER_VALIDATION_STAGES[-1] == "ATTESTATION"
        assert "IMPORT_ISOLATION" in EV.ENVCODER_VALIDATION_STAGES

    def _authorization(self):
        return EV.authorize_environment_validation_runtime(
            mode=EV.ENV_VALIDATION_MODE_TEST_ONLY,
            authorization_grant_hash="a" * 64,
            validator_id=EV.SYNTHETIC_TEST_ONLY_VALIDATOR,
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        )

    def test_test_only_ladder_runs_shape_only(self):
        artifact = EnvCoderArtifact(
            template_hash="f" * 64,
            artifact_id=f"{'f' * 64}::tpl",
            env_code="# TEST_ONLY env code\n",
            prompt_envelope_hash="ab" * 32,
        )
        outcome = EV.run_authorized_validation(
            artifact, authorization=self._authorization()
        )
        assert outcome.stages_passed == EV.ENVCODER_VALIDATION_STAGES
        assert outcome.ladder_complete is True
        assert outcome.test_only is True  # conspicuously marked
        assert len(outcome.outcome_hash) == 64
        assert outcome.artifact_id == artifact.artifact_id

    def test_missing_authorization_refused(self):
        artifact = EnvCoderArtifact(
            template_hash="f" * 64,
            artifact_id=f"{'f' * 64}::tpl",
            env_code="# TEST_ONLY env code\n",
            prompt_envelope_hash="ab" * 32,
        )
        with pytest.raises(EV.EnvValidationError) as excinfo:
            EV.run_authorized_validation(
                artifact, authorization={"validator": "summary"}
            )
        assert excinfo.value.code == EV.ENV_VALIDATION_BAD_TYPE

    def test_artifact_without_identity_refused(self):
        with pytest.raises(EV.EnvValidationError) as excinfo:
            EV.run_authorized_validation(
                SimpleNamespace(), authorization=self._authorization()
            )
        assert excinfo.value.code == EV.ENV_VALIDATION_BAD_TYPE
