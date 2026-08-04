"""C10 tests: committed teacher configs vs frozen manifest consistency.

Loads the REAL committed artifacts — conf/teacher/e1_formal.yaml,
configs/e1_formal_ued.yaml, configs/e1_formal_ued_anchor_manifest.
DRAFT.json — and verifies them against the code pins. Nothing here is
re-derived from the test itself; a mismatch is a real config bug.
"""
import json
import os
import sys

import pytest
import yaml

from dicode.teachers.e1_formal import anchor_manifest as AM
from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import layout
from dicode.teachers.e1_formal import metrics as MT
from dicode.teachers.e1_formal import reference_contract as RC
from dicode.teachers.e1_formal import selector as S
from dicode.teachers.e1_formal.flags import (
    assert_flags_match_manifest,
    parse_flags,
)
from dicode.teachers.e1_formal.student_contract import (
    PINNED_STUDENT_CANDIDATE_ID,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEACHER_YAML = os.path.join(REPO_ROOT, "conf", "teacher", "e1_formal.yaml")
FROZEN_YAML = os.path.join(REPO_ROOT, "configs", "e1_formal_ued.yaml")
DRAFT_JSON = os.path.join(
    REPO_ROOT, "configs", "e1_formal_ued_anchor_manifest.DRAFT.json"
)

# fix(e1): require real probe and update for readiness — the
# readiness gate lives in scripts/ (not a package); same bootstrap
# convention as conftest.py's src/ insert
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import e1_formal_readiness as RD  # noqa: E402


def _teacher_config():
    with open(TEACHER_YAML, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _frozen_manifest():
    with open(FROZEN_YAML, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _draft_manifest():
    with open(DRAFT_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestCommittedConfigConsistency:
    def test_flags_agree_between_config_and_manifest(self):
        config = _teacher_config()
        manifest = _frozen_manifest()
        flags = parse_flags(config["teacher"]["flags"], "test")
        assert_flags_match_manifest(flags, manifest, "test")
        for name in (
            "real_envcoder_used",
            "real_student_reference_eval",
            "real_training_update_executed",
        ):
            assert getattr(flags, name) is False
            assert manifest["flags"][name] is False

    def test_copeland_manifest_pins_equal_code_pins(self):
        copeland = _frozen_manifest()["copeland"]
        assert copeland["protocol_version"] == S.COPELAND_PROTOCOL_VERSION
        assert copeland["source_sha256"] == S.COPELAND_SOURCE_SHA256
        assert copeland["constants_sha256"] == S.COPELAND_CONSTANTS_SHA256
        assert copeland["base_sha256"] == S.COPELAND_BASE_SHA256

    def test_replay_identity_agrees(self):
        config = _teacher_config()
        manifest = _frozen_manifest()
        assert config["teacher"]["replay"]["model_id"] == GM.REPLAY_MODEL_ID
        assert manifest["replay"]["model_id"] == GM.REPLAY_MODEL_ID
        assert config["teacher"]["replay"]["provider"] == "replay"
        assert config["teacher"]["replay"]["record"] == "disabled"

    def test_anchor_ids_agree_everywhere(self):
        config = _teacher_config()
        manifest = _frozen_manifest()
        assert (
            tuple(config["teacher"]["anchors"]["task_ids"])
            == layout.ANCHOR_TASK_IDS
        )
        assert tuple(manifest["anchors"]["task_ids"]) == layout.ANCHOR_TASK_IDS

    def test_strong_student_identity_agrees(self):
        config = _teacher_config()
        manifest = _frozen_manifest()
        assert (
            config["teacher"]["strong_student"]["candidate_id"]
            == PINNED_STUDENT_CANDIDATE_ID
        )
        assert (
            manifest["strong_student"]["candidate_id"]
            == PINNED_STUDENT_CANDIDATE_ID
        )

    def test_selection_pins_match_the_dynamic_slot_count(self):
        selection = _teacher_config()["teacher"]["selection"]
        assert selection["k"] == layout.NUM_DYNAMIC_SLOTS == 12
        assert selection["critic_policy"] == S.CRITIC_HARD_VETO
        assert isinstance(selection["seed"], int)

    def test_hydra_output_dir_is_e1_scoped(self):
        config = _teacher_config()
        assert config["hydra"]["run"]["dir"].startswith(
            "outputs/e1_formal_ued/"
        )


class TestReferenceContractBlockStaysUnfrozen:
    def test_yaml_block_is_explicitly_unfrozen_with_no_defaults(self):
        block = _teacher_config()["teacher"]["reference_contract"]
        assert block["frozen"] is False
        # every identity field is present but null => NO hidden default
        for name, value in block.items():
            if name == "frozen":
                continue
            assert value is None, f"field {name!r} carries a value"

    def test_consuming_the_yaml_block_fails_closed_unfrozen(self):
        block = _teacher_config()["teacher"]["reference_contract"]
        with pytest.raises(RC.ReferenceContractError) as excinfo:
            RC.consume_reference_identity_contract(block, "test")
        assert excinfo.value.code == "REFERENCE_CONTRACT_UNFROZEN"


class TestLearnabilityBlockStaysEmpty:
    def test_yaml_block_has_every_field_null(self):
        block = _teacher_config()["teacher"]["learnability"]
        expected = {
            "tau_saturated",
            "tau_reachable",
            "tau_unreachable",
            "delta_min",
            "min_episodes",
            "ci_level",
        }
        assert set(block) == expected
        assert all(value is None for value in block.values())

    def test_consuming_the_yaml_block_fails_closed_missing(self):
        block = _teacher_config()["teacher"]["learnability"]
        with pytest.raises(MT.MetricsError) as excinfo:
            MT.consume_learnability_thresholds(block, "test")
        assert excinfo.value.code == MT.LEARNABILITY_THRESHOLD_MISSING


class TestDraftAnchorManifestArtifact:
    def test_draft_consumes_and_is_unfrozen(self):
        manifest = AM.consume_anchor_manifest(_draft_manifest(), "test")
        assert manifest.status == AM.STATUS_DRAFT_UNFROZEN
        assert manifest.is_frozen is False
        assert manifest.anchor_ids == (
            "anchor_task_1",
            "anchor_task_2",
            "anchor_task_3",
            "anchor_original_craftax",
        )
        assert len(manifest.manifest_sha256) == 64

    def test_draft_anchors_are_unsigned(self):
        for anchor in _draft_manifest()["anchors"]:
            assert anchor["frozen_by"] == ""
            assert anchor["frozen_at"] == ""

    def test_draft_source_tasks_are_the_four_anchors(self):
        sources = [a["source_task_id"] for a in _draft_manifest()["anchors"]]
        assert tuple(sources) == layout.ANCHOR_TASK_IDS

    def test_retention_stays_blocked_on_the_committed_draft(self):
        manifest = AM.consume_anchor_manifest(_draft_manifest(), "test")
        scores = {aid: 0.5 for aid in manifest.anchor_ids}
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.evaluate_retention(manifest, scores, scores)
        assert excinfo.value.code == AM.BLOCKED_SHARED_ANCHOR_MANIFEST

    def test_assert_frozen_raises_on_the_committed_draft(self):
        manifest = AM.consume_anchor_manifest(_draft_manifest(), "test")
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.assert_manifest_frozen(manifest, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_NOT_FROZEN

    def test_hash_tamper_detection_on_disk_artifact(self):
        mapping = _draft_manifest()
        mapping["anchors"][0]["task_params_hash"] = "TAMPERED"
        with pytest.raises(AM.AnchorManifestError) as excinfo:
            AM.consume_anchor_manifest(mapping, "test")
        assert excinfo.value.code == AM.ANCHOR_MANIFEST_HASH_MISMATCH


class TestTeacherFromCommittedFiles:
    """The teacher must construct from the committed files in a fully
    degraded, honest state (no defaults, no guesses)."""

    def test_init_from_committed_files_records_all_blocks(self):
        manager = GM.E1FormalGenManager(
            _teacher_config(),
            frozen_manifest=_frozen_manifest(),
            anchor_manifest_mapping=_draft_manifest(),
        )
        blocked = manager.current_blocked_codes()
        assert "REFERENCE_CONTRACT_UNFROZEN" in blocked
        assert "LEARNABILITY_THRESHOLD_MISSING" in blocked
        assert AM.BLOCKED_SHARED_ANCHOR_MANIFEST in blocked
        assert "SELECTION_BLOCKED_NO_REAL_EVIDENCE" in blocked
        report = manager.status_report()
        assert report["flags"]["real_envcoder_used"] is False
        assert report["flags"]["real_student_reference_eval"] is False
        assert report["flags"]["real_training_update_executed"] is False

    def test_degraded_batch_from_committed_files(self):
        manager = GM.E1FormalGenManager(
            _teacher_config(),
            frozen_manifest=_frozen_manifest(),
            anchor_manifest_mapping=_draft_manifest(),
        )
        batch = manager.build_training_batch()
        # C13: blocked => ZERO trainable tasks (no anchors-only sneak)
        assert batch["task_ids"] == []
        assert batch["training_permitted"] is False
        assert batch["provenance"] == "BLOCKED"
        assert batch["reuse_only"] is True
        workers = manager.evolve_tasks()
        assert len(workers) == 12
        assert all(w["compiled"] is False for w in workers)
        assert manager.ledger.counts()["N1"] == 0


# ----------------------------------------------------------------------
# fix(e1): require real probe and update for readiness (CC2 P0).
# The final e1_real_smoke_ready conjunction must demand REAL
# EXECUTION evidence from the one-update entrypoint's own status
# report — structural capability gates alone NEVER grant readiness.
# Hosted here (round-3 forbids NEW test files); static/CPU only, no
# API, no training.
# ----------------------------------------------------------------------


def _structural_pass() -> dict:
    """Every structural capability gate true, zero blockers.

    CC2 follow-up P0-6: the single dynamic_12 gate is split into the
    three-way reachability/verification split. The unit-level
    conjunction tests pass all three mechanically; the PRODUCTION
    computed value of dynamic_12_behaviorally_distinct_verified stays
    FALSE this round (no signed probe evidence exists — see
    scripts/e1_formal_readiness.py).
    """
    return dict(
        sequential=True,
        dynamic_12_logical_specs_reachable=True,
        dynamic_12_executable_candidates_reachable=True,
        dynamic_12_behaviorally_distinct_verified=True,
        criterionwise=True,
        bounded_repair=True,
        student_adapter_bound=True,
        reference_adapter_bound=True,
        anchor_manifest_bound=True,
        blockers=[],
    )


def _executed_report(probe: bool = True, update: bool = True) -> dict:
    """The shape run_e1_real_one_update.py writes on COMPLETE success."""
    return {
        "entrypoint": "scripts/run_e1_real_one_update.py",
        "status": "EXECUTED",
        "blockers": [],
        "flags": {
            "real_envcoder_used": True,
            "real_student_reference_eval": probe,
            "real_training_update_executed": True,
        },
        "real_one_update_executed": update,
    }


def _write_status(tmp_path, report: dict) -> str:
    path = tmp_path / "real_one_update_status.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


class TestRealSmokeReadinessGate:
    def test_structural_gates_alone_never_grant_readiness(self, tmp_path):
        # all structural gates pass but NO real execution report on disk
        missing = str(tmp_path / "no_such_report.json")
        probe, update = RD._compute_real_execution_flags(missing)
        assert (probe, update) == (False, False)
        assert RD.decide_real_smoke_ready(
            probe_executed=probe,
            update_executed=update,
            **_structural_pass(),
        ) is False

    def test_status_missing_stays_false(self, tmp_path):
        report = _executed_report()  # flags forged true...
        del report["status"]         # ...but no EXECUTED status
        probe, update = RD._compute_real_execution_flags(
            _write_status(tmp_path, report)
        )
        assert (probe, update) == (False, False)
        assert RD.decide_real_smoke_ready(
            probe_executed=probe,
            update_executed=update,
            **_structural_pass(),
        ) is False

    def test_status_blocked_stays_false_even_with_forged_flags(self, tmp_path):
        report = _executed_report()  # flags stay forged true
        report["status"] = "BLOCKED"
        probe, update = RD._compute_real_execution_flags(
            _write_status(tmp_path, report)
        )
        assert (probe, update) == (False, False)
        assert RD.decide_real_smoke_ready(
            probe_executed=probe,
            update_executed=update,
            **_structural_pass(),
        ) is False

    def test_probe_true_but_update_false_stays_false(self, tmp_path):
        probe, update = RD._compute_real_execution_flags(
            _write_status(tmp_path, _executed_report(update=False))
        )
        assert (probe, update) == (True, False)
        assert RD.decide_real_smoke_ready(
            probe_executed=probe,
            update_executed=update,
            **_structural_pass(),
        ) is False

    def test_executed_probe_and_update_grant_readiness(self, tmp_path):
        # the ONLY true path: real EXECUTED evidence + all gates pass
        probe, update = RD._compute_real_execution_flags(
            _write_status(tmp_path, _executed_report())
        )
        assert (probe, update) == (True, True)
        assert RD.decide_real_smoke_ready(
            probe_executed=probe,
            update_executed=update,
            **_structural_pass(),
        ) is True

    def test_blockers_still_refuse_even_with_execution_evidence(self, tmp_path):
        probe, update = RD._compute_real_execution_flags(
            _write_status(tmp_path, _executed_report())
        )
        kwargs = _structural_pass()
        kwargs["blockers"] = [
            {
                "stage": "shared_runtime_resolution",
                "code": "BLOCKED_WAITING_SHARED_RUNTIME_STUDENT_ADAPTER",
                "detail": "unbound",
            }
        ]
        assert RD.decide_real_smoke_ready(
            probe_executed=probe,
            update_executed=update,
            **kwargs,
        ) is False
