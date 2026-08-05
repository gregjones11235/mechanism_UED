"""CC2 follow-up P0-1/P0-21 tests: the one-window driver's REAL
object flow + the refactored single-update entrypoint.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
every bundle/capability fixture here is synthetic; the entrypoint is
only ever exercised in ``--check-only`` or honestly-BLOCKED full-run
mode (no LLM, no probe, no training, no EXECUTED). These tests prove
the stage boundaries carry real objects and fail closed on None /
string / summary placeholders — never that anything real executed.

Covered negative matrix:
* stage input None                     -> E1_DRIVER_MISSING_OBJECT
* stage input bare string              -> E1_DRIVER_SUMMARY_DICT_REJECTED
* stage input wrong type               -> E1_DRIVER_BAD_TYPE
* manifest-level bundle (unbound)      -> E1_DRIVER_RUNTIME_UNBOUND
* teacher without evidence             -> E1_DRIVER_NO_ADMISSIBLE_EVIDENCE
* artifacts field None / bad pool / empty str / wrong stage type
* entrypoint with no bundle            -> CHECK_ONLY_BLOCKED / BLOCKED
* entrypoint full run TEST_ONLY bundle -> TEST_ONLY_REJECTED blocker
* forbidden hardcodes removed          -> source-level guard
"""
import json
import os
import sys
from dataclasses import dataclass

import pytest
import yaml

from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal import runtime_bundle as RB

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
TEACHER_YAML = os.path.join(REPO_ROOT, "conf", "teacher", "e1_formal.yaml")
FROZEN_YAML = os.path.join(REPO_ROOT, "configs", "e1_formal_ued.yaml")
DRAFT_JSON = os.path.join(
    REPO_ROOT, "configs", "e1_formal_ued_anchor_manifest.DRAFT.json"
)
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "run_e1_real_one_update.py")

# the entrypoint lives in scripts/ (not a package); same bootstrap
# convention as conftest.py / test_flag_manifest.py
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import e1_production_runtime as RT  # noqa: E402
import run_e1_real_one_update as ENT  # noqa: E402


# ---------------------------------------------------------------------------
# SYNTHETIC fixtures — TEST_ONLY, never real shared runtime objects.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SyntheticCapability:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION placeholder."""

    kind: str
    identity_id: str


def _synthetic_capabilities():
    return {
        contract: _SyntheticCapability(
            kind=contract, identity_id=f"test-only-{contract}"
        )
        for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
    }


def _test_only_bundle():
    return RB.build_test_only_runtime_bundle(
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        capabilities=_synthetic_capabilities(),
    )


def _test_only_manifest() -> dict:
    bundle = _test_only_bundle()
    return {
        "bundle_id": bundle.bundle_id,
        "mode": bundle.mode,
        "source_commit": bundle.source_commit,
        "signer_id": bundle.signer_id,
        "authorization_grant_hash": bundle.authorization_grant_hash,
        "object_identity_hashes": dict(bundle.object_identity_hashes),
        "student_selection": bundle.student_selection_mapping,
        "bundle_hash": bundle.bundle_hash,
    }


def _write_test_only_manifest(tmp_path) -> str:
    path = tmp_path / "test_only_bundle_manifest.json"
    path.write_text(
        json.dumps(_test_only_manifest(), sort_keys=True), encoding="utf-8"
    )
    return str(path)


def _committed_manager():
    """The real teacher from the committed config files (degraded,
    honest state: no evidence, no frozen reference contract)."""
    with open(TEACHER_YAML, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(FROZEN_YAML, "r", encoding="utf-8") as handle:
        frozen = yaml.safe_load(handle)
    with open(DRAFT_JSON, "r", encoding="utf-8") as handle:
        draft = json.load(handle)
    return GM.E1FormalGenManager(
        config,
        frozen_manifest=frozen,
        anchor_manifest_mapping=draft,
    )


def _placeholder_artifacts(**overrides):
    defaults = dict(
        runtime_bundle_hash="a" * 64,
        student_identity=object(),
        reference_identity=object(),
        student_adapter=object(),
        reference_adapter=object(),
        student_checkpoint_identity="b" * 64,
        reference_checkpoint_identity="c" * 64,
        gen_manager=object(),
        review_window=object(),
        candidate_materials=object(),
        executable_candidate_pool=(object(),),
        probe_result_pool=(object(),),
        criterion_signals_pool=(object(),),
        selection_outcome=object(),
        verified_batch=object(),
        update_attestation=object(),
        roundtrip_attestation=object(),
        run_id="test-only-run",
        source_commit="d" * 40,
    )
    defaults.update(overrides)
    return DRV.E1OneWindowArtifacts(**defaults)


# ---------------------------------------------------------------------------
# stage boundary guards: real objects only
# ---------------------------------------------------------------------------
class TestRequireRealObject:
    def test_none_placeholder_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.require_real_object(None, "stage_input", "test")
        assert excinfo.value.code == DRV.E1_DRIVER_MISSING_OBJECT

    def test_bare_string_placeholder_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.require_real_object("student_adapter", "stage_input", "test")
        assert excinfo.value.code == DRV.E1_DRIVER_SUMMARY_REJECTED

    def test_real_object_passes(self):
        obj = _SyntheticCapability("probe_runner", "test-only")
        assert DRV.require_real_object(obj, "stage_input", "test") is obj


class TestValidateRuntimeSurface:
    def test_none_runtime_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_runtime_surface(None, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_MISSING_OBJECT

    def test_string_runtime_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_runtime_surface("shared_runtime_bundle", "test")
        assert excinfo.value.code == DRV.E1_DRIVER_SUMMARY_REJECTED

    def test_non_bundle_object_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_runtime_surface(object(), "test")
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE

    def test_manifest_level_bundle_is_unbound_for_execution(self):
        # a manifest carries identity hashes only — the objects must be
        # bound at the seam before any stage may consume the bundle
        manifest_bundle = RB.load_verified_runtime_bundle(
            _test_only_manifest(), "test"
        )
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_runtime_surface(manifest_bundle, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_RUNTIME_UNBOUND

    def test_fully_bound_test_only_bundle_accepted(self):
        bundle = _test_only_bundle()
        assert DRV.validate_runtime_surface(bundle, "test") is bundle


class TestValidateOneWindowArtifacts:
    def test_wrong_record_type_refused(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_one_window_artifacts({"window": "summary"}, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE

    def test_none_field_refused(self):
        artifacts = _placeholder_artifacts(student_identity=None)
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_one_window_artifacts(artifacts, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_ARTIFACTS_INCOMPLETE

    def test_pool_must_be_tuple(self):
        artifacts = _placeholder_artifacts(
            executable_candidate_pool=[object()]
        )
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_one_window_artifacts(artifacts, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE

    def test_pool_item_string_placeholder_refused(self):
        artifacts = _placeholder_artifacts(
            probe_result_pool=(object(), "candidate_probe_result")
        )
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_one_window_artifacts(artifacts, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_SUMMARY_REJECTED

    def test_empty_str_identity_field_refused(self):
        artifacts = _placeholder_artifacts(run_id="")
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_one_window_artifacts(artifacts, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_ARTIFACTS_INCOMPLETE

    def test_review_window_must_be_the_stage_object(self):
        # object() placeholders stand in until the owning commits land
        artifacts = _placeholder_artifacts(review_window=object())
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.validate_one_window_artifacts(artifacts, "test")
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE
        assert "E1WindowResult" in str(excinfo.value)


# ---------------------------------------------------------------------------
# driver stages against the REAL committed teacher
# ---------------------------------------------------------------------------
class TestDriverStagesFailClosed:
    def test_review_window_stage_requires_the_gen_manager(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_review_window(object(), _test_only_bundle())
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE

    def test_review_window_stage_requires_the_runtime_bundle(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_review_window(_committed_manager(), None)
        assert excinfo.value.code == DRV.E1_DRIVER_MISSING_OBJECT

    def test_review_window_stage_refuses_string_runtime(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_review_window(
                _committed_manager(), "student_adapter"
            )
        assert excinfo.value.code == DRV.E1_DRIVER_SUMMARY_REJECTED

    def test_review_window_stage_honestly_stops_without_evidence(self):
        manager = _committed_manager()
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_review_window(manager, _test_only_bundle())
        assert excinfo.value.code == DRV.E1_DRIVER_NO_EVIDENCE
        # no window may be fabricated: teacher bookkeeping untouched
        assert manager.cycles_run == 0
        assert manager.last_review_window is None
        assert manager.consecutive_reuses == 0

    def test_envcoder_stage_requires_the_window_result_object(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_envcoder_and_compile(
                _committed_manager(), None, _test_only_bundle()
            )
        assert excinfo.value.code == DRV.E1_DRIVER_MISSING_OBJECT

    def test_envcoder_stage_refuses_summary_dict_window(self):
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_envcoder_and_compile(
                _committed_manager(),
                {"window_id": "summary"},
                _test_only_bundle(),
            )
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE


# ---------------------------------------------------------------------------
# entrypoint: --check-only and honestly-BLOCKED full runs
# ---------------------------------------------------------------------------
def _read_report(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestEntrypointCheckOnly:
    def test_check_only_without_bundle_blocks(self, tmp_path):
        report_path = str(tmp_path / "check_only_blocked.json")
        rc = ENT.main(["--check-only", "--report-out", report_path])
        assert rc == 2
        report = _read_report(report_path)
        assert report["status"] == ENT.E1_CHECK_ONLY_BLOCKED
        assert report["check_only"] is True
        assert report["executed"] is False  # check-only NEVER executes
        assert report["checks"]["bundle_manifest_verified"] is False
        codes = [b["code"] for b in report["production_blockers"]]
        assert ENT.E1_DIRECTOR_RUNTIME_BUNDLE_REQUIRED in codes
        # no real LLM provider is authorized this round
        assert RT.E1_REAL_LLM_NOT_AUTHORIZED in codes

    def test_check_only_with_missing_bundle_file_blocks(self, tmp_path):
        report_path = str(tmp_path / "check_only_missing_file.json")
        rc = ENT.main(
            [
                "--check-only",
                "--director-runtime-bundle",
                str(tmp_path / "no_such_manifest.json"),
                "--report-out",
                report_path,
            ]
        )
        assert rc == 2
        report = _read_report(report_path)
        assert report["status"] == ENT.E1_CHECK_ONLY_BLOCKED
        assert report["executed"] is False

    def test_check_only_test_only_bundle_proves_connectivity(self, tmp_path):
        manifest_path = _write_test_only_manifest(tmp_path)
        report_path = str(tmp_path / "check_only_ok.json")
        rc = ENT.main(
            [
                "--check-only",
                "--director-runtime-bundle",
                manifest_path,
                "--report-out",
                report_path,
            ]
        )
        assert rc == 0
        report = _read_report(report_path)
        assert report["status"] == ENT.E1_TEST_ONLY_CONTRACT_OK
        assert report["check_only"] is True
        assert report["executed"] is False  # NEVER EXECUTED
        checks = report["checks"]
        assert checks["bundle_manifest_verified"] is True
        assert checks["bundle_mode"] == RB.BUNDLE_MODE_TEST_ONLY
        assert checks["capability_contracts_declared"] is True
        assert checks["driver_dataflow_constructible"] is True
        assert checks["driver_dataflow_fields_complete"] is True
        # the shared runtime is absent on this host: every contract
        # stays honestly unbound (connectivity != binding)
        assert all(
            bound is False
            for bound in checks["shared_runtime_objects_bound"].values()
        )
        # TEST_ONLY connectivity NEVER clears the production blockers
        codes = [b["code"] for b in report["production_blockers"]]
        assert RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED in codes


class TestEntrypointFullRunStaysBlocked:
    def test_full_run_without_bundle_is_blocked(self, tmp_path):
        report_path = str(tmp_path / "full_run_blocked.json")
        rc = ENT.main(["--report-out", report_path])
        assert rc == 2
        report = _read_report(report_path)
        assert report["status"] == "BLOCKED"
        assert report["real_one_update_executed"] is False
        assert report["flags"] == {
            "real_envcoder_used": False,
            "real_student_reference_eval": False,
            "real_training_update_executed": False,
        }
        codes = [b["code"] for b in report["blockers"]]
        assert ENT.E1_DIRECTOR_RUNTIME_BUNDLE_REQUIRED in codes

    def test_full_run_with_test_only_bundle_refused(self, tmp_path):
        manifest_path = _write_test_only_manifest(tmp_path)
        report_path = str(tmp_path / "full_run_test_only.json")
        rc = ENT.main(
            [
                "--director-runtime-bundle",
                manifest_path,
                "--report-out",
                report_path,
            ]
        )
        assert rc == 2
        report = _read_report(report_path)
        assert report["status"] == "BLOCKED"
        assert report["real_one_update_executed"] is False
        codes = [b["code"] for b in report["blockers"]]
        # TEST_ONLY bundles never enter a production path
        assert RB.RUNTIME_BUNDLE_TEST_ONLY_REJECTED in codes
        # the shared runtime stays unbound (honest per-contract codes)
        assert any(
            code.startswith("BLOCKED_WAITING_SHARED_RUNTIME")
            for code in codes
        )


class TestForbiddenHardcodesRemoved:
    """CC2 follow-up P0-21: the former placeholder hardcodes must be
    structurally gone from the single-update entrypoint."""

    def test_entrypoint_source_carries_no_forbidden_hardcodes(self):
        import ast

        with open(SCRIPT_PATH, "r", encoding="utf-8") as handle:
            source = handle.read()
        # the module docstring documents the REMOVED defects by name;
        # the guard covers EXECUTABLE code: strip the docstring via AST
        # line span, then the literals must be structurally absent
        tree = ast.parse(source)
        code_source = source
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines = source.splitlines(keepends=True)
            code_source = "".join(lines[first.end_lineno :])
        forbidden = (
            "teacher.evolve(",        # nonexistent method (P0-1)
            "gen_manager=None",       # P0-5 placeholder
            "rl_train_state=None",    # P0-10 placeholder
            "stage_real_probe(())",   # P0-3 empty candidate set
            "llm_client=None",        # P0-2 null-client fallback
            "global_step=0",          # faked reinit claim
        )
        for token in forbidden:
            assert token not in code_source, (
                f"forbidden hardcode {token!r} re-appeared in code"
            )

    def test_entrypoint_requires_the_runtime_bundle_flag(self):
        assert "--director-runtime-bundle" in ENT.__doc__
        assert "--check-only" in ENT.__doc__
