"""C10 tests: committed teacher configs vs frozen manifest consistency.

Loads the REAL committed artifacts — conf/teacher/e1_formal.yaml,
configs/e1_formal_ued.yaml, configs/e1_formal_ued_anchor_manifest.
DRAFT.json — and verifies them against the code pins. Nothing here is
re-derived from the test itself; a mismatch is a real config bug.
"""
import json
import os

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
        assert batch["task_ids"] == list(layout.ANCHOR_TASK_IDS)
        assert batch["reuse_only"] is True
        workers = manager.evolve_tasks()
        assert len(workers) == 12
        assert all(w["compiled"] is False for w in workers)
        assert manager.ledger.counts()["N1"] == 0
