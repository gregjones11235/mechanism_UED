"""CC2 follow-up P0-6 tests: variant parameters ENTER execution.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
no real backend runs here. The ONLY marker-clearing surface
(``execute_variant_parameters``) is exercised against synthetic
validation records to prove it fails closed without real execution
evidence; the readiness split pins that behavioral distinctness stays
FALSE this round.

Covered negative matrix:
* candidate / task-params hash swap        -> VARIANT_HASH_MISMATCH
* executed params != candidate params      -> VARIANT_PARAMS_MISMATCH
* replay / mock backend                    -> VARIANT_BACKEND_FORBIDDEN
* incomplete stage ladder                  -> VARIANT_STAGES_INCOMPLETE
* TEST_ONLY validation record              -> VARIANT_TEST_ONLY_REJECTED
* double execution claim                   -> VARIANT_ALREADY_EXECUTED
* bad types                                -> VARIANT_BAD_TYPE
* unbound / attribute-less surfaces        -> EXECUTION_SURFACE_UNBOUND
* driver stage placeholder inputs          -> MISSING_OBJECT / BAD_TYPE
* readiness split: distinctness false => readiness false
"""
import os
import sys
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import board
from dicode.teachers.e1_formal import executable_candidates as EX
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal import variant_binding as VB
from dicode.teachers.e1_formal.canonical import canonical_sha256
from dicode.teachers.e1_formal.envcoder import EnvCoderArtifact
from dicode.teachers.e1_formal.manifest import BOARD_ROLE_ORDER
from dicode.teachers.e1_formal.task_specs import (
    CompileResult,
    TaskSpec,
    TaskTemplate,
)
from dicode.teachers.e1_formal import envcoder_backends as EB

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import e1_formal_readiness as RD  # noqa: E402

# SYNTHETIC execution-surface hashes — TEST_ONLY placeholders
_ABI_HASH = "a1" * 32
_REWARD_HASH = "b2" * 32
_RESET_HASH = "c3" * 32
_SEED_HASH = "d4" * 32
_SURFACES = {
    "observation_action_abi_hash": _ABI_HASH,
    "reward_contract_hash": _REWARD_HASH,
    "reset_protocol_hash": _RESET_HASH,
    "seed_policy_hash": _SEED_HASH,
}


# ---------------------------------------------------------------------------
# fixtures (all TEST_ONLY / SYNTHETIC)
# ---------------------------------------------------------------------------
def _window(window_id="e1-w000001"):
    role_results = tuple(
        (role, {"role": role, "note": "TEST_ONLY"})
        for role in BOARD_ROLE_ORDER
    )
    payload = board._window_payload(
        window_id,
        1,
        "FIRST_WINDOW",
        "e" * 64,
        "COMPLETE",
        "",
        role_results,
        (),
        (),
    )
    return board.ReviewWindow(
        window_id=window_id,
        session_idx=1,
        trigger_code="FIRST_WINDOW",
        evidence_hash="e" * 64,
        status="COMPLETE",
        void_code="",
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


def _variant_params(variant):
    return (("difficulty:level", str(variant)), ("axis_a:level", "1/2"))


def _bound_candidate(window=None, variant=0):
    window = window or _window()
    template_hash = "f" * 64
    spec = _spec(
        window,
        family_id="fam_a",
        template_hash=template_hash,
        variant=variant,
        variant_params=_variant_params(variant),
    )
    artifact = EX.build_executable_environment_artifact(
        envcoder_artifact=EnvCoderArtifact(
            template_hash=template_hash,
            artifact_id=f"{template_hash}::tpl",
            env_code="# TEST_ONLY env code\n",
            prompt_envelope_hash="ab" * 32,
        ),
        family_id="fam_a",
        observation_action_abi_hash=_ABI_HASH,
        reward_contract_hash=_REWARD_HASH,
        reset_protocol_hash=_RESET_HASH,
        seed_policy_hash=_SEED_HASH,
        backend_name="test-only-backend",
        stages_passed=("SYNTAX", "GUARDS"),
    )
    return EX.bind_executable_candidate(
        window=window, spec=spec, executable_artifact=artifact
    )


def _validation(candidate, **overrides):
    kwargs = dict(
        candidate_hash=candidate.candidate_hash,
        task_params_hash=candidate.task_params_hash,
        executed_variant_params=candidate.variant_params,
        backend_name=EB.BACKEND_REAL,
        stages_passed=tuple(EB.STAGES),
        validation_record_hash="c9" * 32,
        test_only=False,
    )
    kwargs.update(overrides)
    return VB.VariantExecutionValidation(**kwargs)


# ---------------------------------------------------------------------------
# the ONLY marker-clearing surface (real execution evidence only)
# ---------------------------------------------------------------------------
class TestExecuteVariantParameters:
    def test_real_validation_clears_the_marker(self):
        candidate = _bound_candidate()
        assert candidate.execution_marker == (
            EX.VARIANT_PARAMETER_NOT_EXECUTED
        )
        executed = VB.execute_variant_parameters(
            candidate, _validation(candidate), "test"
        )
        assert executed.variant_params_executed is True
        assert executed.execution_marker == ""
        # every other field stays identical (immutability discipline)
        assert executed.candidate_hash == candidate.candidate_hash
        assert executed.task_params_hash == candidate.task_params_hash

    def test_candidate_hash_swap_refused(self):
        candidate = _bound_candidate()
        other = _bound_candidate(variant=1)
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(
                candidate, _validation(other), "test"
            )
        assert excinfo.value.code == VB.VARIANT_HASH_MISMATCH

    def test_task_params_hash_swap_refused(self):
        candidate = _bound_candidate()
        validation = _validation(candidate, task_params_hash="d" * 64)
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(candidate, validation, "test")
        assert excinfo.value.code == VB.VARIANT_HASH_MISMATCH

    def test_executed_params_must_equal_candidate_params(self):
        candidate = _bound_candidate()
        validation = _validation(
            candidate,
            executed_variant_params=(("difficulty:level", "999"),),
        )
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(candidate, validation, "test")
        assert excinfo.value.code == VB.VARIANT_PARAMS_MISMATCH

    def test_replay_and_mock_backends_never_count_as_execution(self):
        candidate = _bound_candidate()
        for backend in (EB.BACKEND_REPLAY, EB.BACKEND_MOCK, "fixture"):
            validation = _validation(candidate, backend_name=backend)
            with pytest.raises(VB.VariantBindingError) as excinfo:
                VB.execute_variant_parameters(
                    candidate, validation, "test"
                )
            assert excinfo.value.code == VB.VARIANT_BACKEND_FORBIDDEN

    def test_incomplete_stage_ladder_refused(self):
        candidate = _bound_candidate()
        validation = _validation(
            candidate, stages_passed=("SYNTAX", "GUARDS")
        )
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(candidate, validation, "test")
        assert excinfo.value.code == VB.VARIANT_STAGES_INCOMPLETE

    def test_test_only_validation_record_refused(self):
        candidate = _bound_candidate()
        validation = _validation(candidate, test_only=True)
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(candidate, validation, "test")
        assert excinfo.value.code == VB.VARIANT_TEST_ONLY_REJECTED

    def test_execution_evidence_never_stacked(self):
        candidate = _bound_candidate()
        executed = VB.execute_variant_parameters(
            candidate, _validation(candidate), "test"
        )
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(
                executed, _validation(executed), "test"
            )
        assert excinfo.value.code == VB.VARIANT_ALREADY_EXECUTED

    def test_bad_types_refused(self):
        candidate = _bound_candidate()
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(
                {"candidate": "summary"},
                _validation(candidate),
                "test",
            )
        assert excinfo.value.code == VB.VARIANT_BAD_TYPE
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execute_variant_parameters(
                candidate, {"validation": "summary"}, "test"
            )
        assert excinfo.value.code == VB.VARIANT_BAD_TYPE


# ---------------------------------------------------------------------------
# execution-surface resolution from bundle-bound shared objects
# ---------------------------------------------------------------------------
def _resolution(obj, bound=True):
    return SimpleNamespace(object_ref=obj, bound=bound)


def _surface_objects():
    return {
        "student_adapter": SimpleNamespace(
            observation_action_abi_hash=_ABI_HASH
        ),
        "formal_asset_registry": SimpleNamespace(
            reward_contract_hash=_REWARD_HASH
        ),
        "probe_runner": SimpleNamespace(
            reset_protocol_hash=_RESET_HASH,
            seed_bank_hash=_SEED_HASH,
        ),
    }


class TestExecutionSurfaceResolution:
    def test_all_four_surfaces_bind_from_shared_objects(self):
        objects = _surface_objects()
        resolutions = {
            "student_adapter": _resolution(objects["student_adapter"]),
            "formal_asset_registry": _resolution(
                objects["formal_asset_registry"]
            ),
            "probe_runner": _resolution(objects["probe_runner"]),
        }
        surfaces = VB.execution_surfaces_from_bundle_resolutions(
            resolutions, "test"
        )
        assert surfaces == _SURFACES

    def test_unbound_contract_refused(self):
        resolutions = {
            "student_adapter": _resolution(None, bound=False),
            "formal_asset_registry": _resolution(
                _surface_objects()["formal_asset_registry"]
            ),
            "probe_runner": _resolution(_surface_objects()["probe_runner"]),
        }
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execution_surfaces_from_bundle_resolutions(
                resolutions, "test"
            )
        assert excinfo.value.code == VB.VARIANT_EXECUTION_SURFACE_UNBOUND

    def test_missing_contract_refused(self):
        objects = _surface_objects()
        resolutions = {
            "student_adapter": _resolution(objects["student_adapter"]),
            # formal_asset_registry deliberately absent
            "probe_runner": _resolution(objects["probe_runner"]),
        }
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execution_surfaces_from_bundle_resolutions(
                resolutions, "test"
            )
        assert excinfo.value.code == VB.VARIANT_EXECUTION_SURFACE_UNBOUND

    def test_attribute_without_64hex_surface_refused(self):
        objects = _surface_objects()
        bad_adapter = SimpleNamespace(
            observation_action_abi_hash="not-a-hash"
        )
        resolutions = {
            "student_adapter": _resolution(bad_adapter),
            "formal_asset_registry": _resolution(
                objects["formal_asset_registry"]
            ),
            "probe_runner": _resolution(objects["probe_runner"]),
        }
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.execution_surfaces_from_bundle_resolutions(
                resolutions, "test"
            )
        assert excinfo.value.code == VB.VARIANT_EXECUTION_SURFACE_UNBOUND


# ---------------------------------------------------------------------------
# pool binding from compile+EnvCoder materials
# ---------------------------------------------------------------------------
class TestPoolFromMaterials:
    def _materials(self, window):
        entries = []
        specs = []
        templates = []
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
            entries.append(
                (
                    template_hash,
                    EnvCoderArtifact(
                        template_hash=template_hash,
                        artifact_id=f"{template_hash}::tpl",
                        env_code="# TEST_ONLY env code\n",
                        prompt_envelope_hash="ab" * 32,
                    ),
                    (),
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
        compile_result = CompileResult(
            templates=tuple(templates), specs=tuple(specs), notes=()
        )
        return compile_result, tuple(entries)

    def test_binds_the_pool_with_markers(self):
        window = _window()
        compile_result, entries = self._materials(window)
        pool = VB.bind_executable_pool_from_materials(
            window=window,
            compile_result=compile_result,
            template_artifacts=entries,
            execution_surfaces=_SURFACES,
            backend_name="test-only-backend",
            stages_passed=("SYNTAX", "GUARDS"),
        )
        assert len(pool) == 4
        assert len({c.candidate_hash for c in pool}) == 4
        for candidate in pool:
            assert candidate.variant_params_executed is False
            assert candidate.execution_marker == (
                EX.VARIANT_PARAMETER_NOT_EXECUTED
            )

    def test_incomplete_surface_set_refused(self):
        window = _window()
        compile_result, entries = self._materials(window)
        partial = dict(_SURFACES)
        del partial["seed_policy_hash"]
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.bind_executable_pool_from_materials(
                window=window,
                compile_result=compile_result,
                template_artifacts=entries,
                execution_surfaces=partial,
                backend_name="test-only-backend",
                stages_passed=("SYNTAX",),
            )
        assert excinfo.value.code == VB.VARIANT_EXECUTION_SURFACE_UNBOUND

    def test_malformed_materials_entry_refused(self):
        window = _window()
        compile_result, _entries = self._materials(window)
        with pytest.raises(VB.VariantBindingError) as excinfo:
            VB.bind_executable_pool_from_materials(
                window=window,
                compile_result=compile_result,
                template_artifacts=(("only-two-fields", "x"),),
                execution_surfaces=_SURFACES,
                backend_name="test-only-backend",
                stages_passed=("SYNTAX",),
            )
        assert excinfo.value.code == VB.VARIANT_BAD_TYPE


# ---------------------------------------------------------------------------
# driver stage: placeholders fail closed before any binding
# ---------------------------------------------------------------------------
class _SyntheticCapability:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION placeholder."""

    def __init__(self, kind):
        self.kind = kind
        self.identity_id = f"test-only-{kind}"


def _test_only_bundle():
    from dicode.teachers.e1_formal import runtime_bundle as RB

    return RB.build_test_only_runtime_bundle(
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        capabilities={
            contract: _SyntheticCapability(contract)
            for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
        },
    )


class TestDriverStageFailClosed:
    def test_none_window_result_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_candidate_binding(
                object(), None, object(), _test_only_bundle()
            )
        # runtime verifies first, then the teacher type check fires
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE

    def test_summary_dict_window_refused_with_real_teacher_type(self):
        # even with a wrong-type teacher caught first, a string
        # runtime placeholder is refused at the boundary
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_candidate_binding(
                object(), object(), object(), "shared_runtime_bundle"
            )
        assert excinfo.value.code == DRV.E1_DRIVER_SUMMARY_REJECTED


# ---------------------------------------------------------------------------
# readiness split (P0-6): the third gate stays false this round
# ---------------------------------------------------------------------------
class TestReadinessSplit:
    def test_logical_specs_reachable_computes_true(self):
        assert RD._compute_dynamic_12_logical_specs_reachable() is True

    def test_executable_candidates_reachable_computes_true(self):
        assert (
            RD._compute_dynamic_12_executable_candidates_reachable()
            is True
        )

    def test_behavioral_distinctness_stays_false_this_round(self):
        # no signed probe evidence exists => fail-closed constant
        assert (
            RD._compute_dynamic_12_behaviorally_distinct_verified()
            is False
        )

    def test_readiness_false_when_distinctness_unverified(self):
        # EVERYTHING else passes — the unverified third gate alone
        # keeps readiness false
        assert RD.decide_real_smoke_ready(
            sequential=True,
            dynamic_12_logical_specs_reachable=True,
            dynamic_12_executable_candidates_reachable=True,
            dynamic_12_behaviorally_distinct_verified=False,
            criterionwise=True,
            bounded_repair=True,
            student_adapter_bound=True,
            reference_adapter_bound=True,
            anchor_manifest_bound=True,
            probe_executed=True,
            update_executed=True,
            blockers=[],
        ) is False
