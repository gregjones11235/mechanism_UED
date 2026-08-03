"""C13 tests: fail-closed training gate (supervisor REQUEST_CHANGES fix).

Proves, offline, that:

* while the shared anchor manifest is DRAFT, while real dual probes
  are absent, or while no legitimate previous-window batch exists, the
  teacher batch NEVER permits training — ``enforce_training_gate``
  raises BEFORE any ``run_session_training``-equivalent can run
  (zero updates, zero step progress; no anchors-only sneak, no legacy
  distribution sneak);
* a legitimate REUSE batch is ONLY the previous window's FULLY
  VERIFIED 12 dynamic + 4 frozen shared anchors, bound to
  source/window/hash evidence; then (and only then) training proceeds.

Every snapshot used here is an explicitly labeled FIXTURE: no real
Student/Reference evaluation happened this round; the tests exercise
the mechanism, never claim real evidence.
"""
import hashlib

import pytest

from dicode.teachers.e1_formal import anchor_manifest as AM
from dicode.teachers.e1_formal import gen_manager as GM
from dicode.teachers.e1_formal import layout as L
from dicode.teachers.e1_formal import training_gate as TG
from dicode.teachers.e1_formal.canonical import canonical_sha256

from test_gen_manager_duck import _frozen_manifest, _manager, _teacher_config
from test_reference_contract import _block

WINDOW = "e1-w000007"
DYNAMIC_IDS = [f"{WINDOW}::fam_a::v{i:02d}" for i in range(1, 13)]
CODE_TEMPLATE = "class Env{i}:\n    pass\n"


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _frozen_anchor_mapping():
    """A supervisor-signed FROZEN manifest FIXTURE (test-only; the
    committed DRAFT artifact stays untouched on disk)."""
    anchors = []
    for source in L.ANCHOR_TASK_IDS:
        anchors.append(
            {
                "anchor_id": f"anchor_{source}",
                "source_task_id": source,
                "task_params_hash": "f1" * 32,
                "seed_protocol": "fixture-frozen-seed-protocol-v1",
                "code_hash": "f2" * 32,
                "reset_protocol": "standard-reset-v1",
                "frozen_by": "supervisor-fixture",
                "frozen_at": "2026-08-04T00:00:00Z",
            }
        )
    payload = {"status": AM.STATUS_FROZEN, "anchors": anchors}
    return {
        "status": AM.STATUS_FROZEN,
        "anchors": anchors,
        "manifest_sha256": canonical_sha256(payload),
    }


def _unblocked_config():
    """Teacher config with frozen G1 contract + frozen thresholds.

    FIXTURE identity values (from test_reference_contract._block) —
    the supervisor's real Reference freeze is still pending; nothing
    here claims a real Reference exists.
    """
    config = _teacher_config()
    config["teacher"]["reference_contract"] = _block()
    config["teacher"]["learnability"] = {
        "tau_saturated": 0.8,
        "tau_reachable": 0.5,
        "tau_unreachable": 0.2,
        "delta_min": 0.05,
        "min_episodes": 8,
        "ci_level": 0.95,
    }
    return config


def _unblocked_manager(**kwargs):
    return GM.E1FormalGenManager(
        _unblocked_config(),
        frozen_manifest=_frozen_manifest(),
        anchor_manifest_mapping=_frozen_anchor_mapping(),
        **kwargs,
    )


def _consume_window_artifacts(manager, task_ids=None, window_id=WINDOW):
    ids = list(task_ids or DYNAMIC_IDS)
    workers = []
    for i, task_id in enumerate(ids):
        code = CODE_TEMPLATE.format(i=i)
        workers.append(
            {
                "task_id": task_id,
                "generated_task_id": task_id,
                "compiled": True,
                "code": code,
                "code_string": code,
                "reasoning": "",
                "e1_status": {
                    "reuse": False,
                    "artifact_id": f"{task_id}::a1",
                    "spec_hash": f"{i:02d}" + "ab" * 31,
                    "window_id": window_id,
                    "compiled": True,
                    "compile_note": "",
                },
            }
        )
    manager.consume_worker_results(workers)
    return ids


def _valid_snapshot(manager, window_id=WINDOW):
    """Build a structurally honest snapshot from the teacher's OWN
    registry evidence (never from thin air)."""
    dynamic = []
    for task_id, record in manager.artifact_registry.items():
        dynamic.append(
            {
                "task_id": task_id,
                "artifact_id": record["artifact_id"],
                "spec_hash": record["spec_hash"],
                "code_sha256": hashlib.sha256(
                    record["code"].encode("utf-8")
                ).hexdigest(),
            }
        )
    contract = manager.reference_contract
    return {
        "window_id": window_id,
        "provenance": "CANDIDATE_EVALUATION",
        # for blocked-certification tests the contract is None; the
        # gate rejects on the frozen-state check before it ever trusts
        # this field
        "reference_candidate_id": (
            contract.candidate_id
            if contract is not None
            else "REFERENCE_CANDIDATE_FROZEN_BY_SUPERVISOR"
        ),
        "anchor_task_ids": list(L.ANCHOR_TASK_IDS),
        "anchor_manifest_sha256": manager.anchor_manifest.manifest_sha256,
        "dynamic_tasks": dynamic,
    }


def _run_session_step(manager, train_calls):
    """Mirror of run_dicode's hook branch: the gate runs FIRST and a
    training stand-in runs ONLY on explicit permission."""
    batch = manager.build_training_batch()
    ids = TG.enforce_training_gate(batch)  # raises => no training
    train_calls.append(list(ids))
    return batch


# ----------------------------------------------------------------------
# negative: blocked teacher never trains
# ----------------------------------------------------------------------
class TestBlockedTeacherTrainsNothing:
    """Default committed state: DRAFT anchor manifest, unfrozen
    Reference contract, no thresholds, no probes."""

    def test_blocked_batch_has_zero_trainable_tasks(self):
        batch = _manager().build_training_batch()
        assert batch["task_ids"] == []
        assert batch["training_permitted"] is False
        assert batch["provenance"] == "BLOCKED"
        assert batch["layout"] is None
        assert batch["dynamic_promoted"] == 0
        assert batch["reuse_only"] is True
        assert batch["reuse_evidence"] is None
        for code in (
            "REFERENCE_CONTRACT_UNFROZEN",
            "LEARNABILITY_THRESHOLD_MISSING",
            AM.BLOCKED_SHARED_ANCHOR_MANIFEST,
            "SELECTION_BLOCKED_NO_REAL_EVIDENCE",
        ):
            assert code in batch["blocked_codes"]
        # the anchors never appear as trainable tasks
        for anchor in L.ANCHOR_TASK_IDS:
            assert anchor not in batch["task_ids"]

    def test_gate_refuses_the_blocked_batch_before_training(self):
        train_calls = []
        with pytest.raises(TG.TrainingGateError) as excinfo:
            _run_session_step(_manager(), train_calls)
        assert excinfo.value.code == TG.TRAINING_GATE_BLOCKED
        assert AM.BLOCKED_SHARED_ANCHOR_MANIFEST in excinfo.value.codes
        assert train_calls == []  # zero training updates

    def test_promotion_while_blocked_fails_closed(self):
        with pytest.raises(GM.GenManagerError) as excinfo:
            _manager().build_training_batch(
                [f"dyn_{i:02d}" for i in range(12)]
            )
        assert excinfo.value.code == GM.GEN_MANAGER_PROMOTION_BLOCKED

    def test_record_verified_batch_blocked_on_draft_manifest(self):
        # even a structurally perfect snapshot cannot be certified
        # while the anchor manifest is DRAFT and the contract unfrozen
        manager = _manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BLOCKED
        assert manager.verified_batch_snapshot is None

    def test_frozen_manifest_but_unfrozen_contract_still_blocks(self):
        config = _teacher_config()  # unfrozen contract
        config["teacher"]["learnability"] = _unblocked_config()[
            "teacher"
        ]["learnability"]
        manager = GM.E1FormalGenManager(
            config,
            frozen_manifest=_frozen_manifest(),
            anchor_manifest_mapping=_frozen_anchor_mapping(),
        )
        _consume_window_artifacts(manager)
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(_valid_snapshot(manager))
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BLOCKED
        assert "REFERENCE_CONTRACT_UNFROZEN" in str(excinfo.value)
        assert manager.build_training_batch()["training_permitted"] is False

    def test_missing_thresholds_still_block_certification(self):
        config = _unblocked_config()
        for name in config["teacher"]["learnability"]:
            config["teacher"]["learnability"][name] = None
        manager = GM.E1FormalGenManager(
            config,
            frozen_manifest=_frozen_manifest(),
            anchor_manifest_mapping=_frozen_anchor_mapping(),
        )
        _consume_window_artifacts(manager)
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(_valid_snapshot(manager))
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BLOCKED
        assert "LEARNABILITY_THRESHOLD_MISSING" in str(excinfo.value)


# ----------------------------------------------------------------------
# negative: empty / forged REUSE snapshots never certify
# ----------------------------------------------------------------------
class TestForgedSnapshotsNeverCertify:
    def _record(self, snapshot):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        return manager, snapshot

    def test_empty_snapshot_rejected(self):
        manager, _ = self._record({})
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch({})
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    def test_non_mapping_snapshot_rejected(self):
        manager = _unblocked_manager()
        for bad in (None, [], "snapshot", 42):
            with pytest.raises(GM.GenManagerError) as excinfo:
                manager.record_verified_batch(bad)
            assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_unknown_field_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["extra"] = 1
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    @pytest.mark.parametrize(
        "provenance", ["TRAINING", "NORMAL_TRAINING_FEEDBACK", "FORMAL_FRONT"]
    )
    def test_only_dual_probe_provenance_may_certify(self, provenance):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["provenance"] = provenance
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    @pytest.mark.parametrize("count", [0, 11, 13])
    def test_dynamic_count_must_be_exactly_12(self, count):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        tasks = snapshot["dynamic_tasks"][:count]
        while len(tasks) < count:  # pad to 13 with a duplicate entry
            tasks.append(dict(snapshot["dynamic_tasks"][0]))
        snapshot["dynamic_tasks"] = tasks
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_duplicate_dynamic_id_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][11] = dict(snapshot["dynamic_tasks"][0])
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_anchor_colliding_dynamic_id_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][0]["task_id"] = "task_1"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_wrong_anchor_tuple_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["anchor_task_ids"] = ["a", "b", "c", "original_craftax"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_manifest_sha_mismatch_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["anchor_manifest_sha256"] = "0" * 64
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_reference_candidate_mismatch_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["reference_candidate_id"] = "SOME_OTHER_CANDIDATE"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_task_not_in_registry_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][0]["task_id"] = f"{WINDOW}::ghost::v99"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_spec_hash_mismatch_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][0]["spec_hash"] = "f" * 64
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_code_sha_mismatch_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][0]["code_sha256"] = "e" * 64
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_window_id_mismatch_rejected(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["window_id"] = "e1-w999999"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_certification_leaves_no_trace_on_failure(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["anchor_manifest_sha256"] = "0" * 64
        with pytest.raises(GM.GenManagerError):
            manager.record_verified_batch(snapshot)
        assert manager.verified_batch_snapshot is None
        assert (
            "SELECTION_BLOCKED_NO_REAL_EVIDENCE"
            in manager.current_blocked_codes()
        )


# ----------------------------------------------------------------------
# negative: forged "permitted" batches at the gate itself
# ----------------------------------------------------------------------
class TestGateRefusesForgedPermittedBatches:
    def test_anchors_only_sneak_refused(self):
        forged = {
            "training_permitted": True,
            "task_ids": list(L.ANCHOR_TASK_IDS),
            "layout": {a: 0.25 for a in L.ANCHOR_TASK_IDS},
        }
        with pytest.raises(TG.TrainingGateError) as excinfo:
            TG.enforce_training_gate(forged)
        assert excinfo.value.code == TG.TRAINING_GATE_BAD_BATCH

    def test_wrong_anchor_order_refused(self):
        ids = [f"dyn_{i:02d}" for i in range(12)] + [
            "original_craftax", "task_1", "task_2", "task_3",
        ]
        forged = {
            "training_permitted": True,
            "task_ids": ids,
            "layout": {t: 1 / 16 for t in ids},
        }
        with pytest.raises(TG.TrainingGateError) as excinfo:
            TG.enforce_training_gate(forged)
        assert excinfo.value.code == TG.TRAINING_GATE_BAD_BATCH

    def test_duplicate_dynamic_refused(self):
        ids = ["dyn_00"] * 12 + list(L.ANCHOR_TASK_IDS)
        forged = {
            "training_permitted": True,
            "task_ids": ids,
            "layout": {t: 1 / 16 for t in set(ids)},
        }
        with pytest.raises(TG.TrainingGateError) as excinfo:
            TG.enforce_training_gate(forged)
        assert excinfo.value.code == TG.TRAINING_GATE_BAD_BATCH

    def test_missing_layout_refused(self):
        ids = [f"dyn_{i:02d}" for i in range(12)] + list(L.ANCHOR_TASK_IDS)
        forged = {"training_permitted": True, "task_ids": ids}
        with pytest.raises(TG.TrainingGateError) as excinfo:
            TG.enforce_training_gate(forged)
        assert excinfo.value.code == TG.TRAINING_GATE_BAD_BATCH

    def test_layout_not_covering_batch_refused(self):
        ids = [f"dyn_{i:02d}" for i in range(12)] + list(L.ANCHOR_TASK_IDS)
        forged = {
            "training_permitted": True,
            "task_ids": ids,
            "layout": {ids[0]: 1.0},  # legacy-style single entry
        }
        with pytest.raises(TG.TrainingGateError) as excinfo:
            TG.enforce_training_gate(forged)
        assert excinfo.value.code == TG.TRAINING_GATE_BAD_BATCH

    @pytest.mark.parametrize("permitted", ["yes", 1, None, False])
    def test_permitted_must_be_literal_true(self, permitted):
        forged = {
            "training_permitted": permitted,
            "task_ids": [],
            "blocked_codes": ["SOME_CODE"],
        }
        with pytest.raises(TG.TrainingGateError) as excinfo:
            TG.enforce_training_gate(forged)
        assert excinfo.value.code == TG.TRAINING_GATE_BLOCKED
        assert "SOME_CODE" in excinfo.value.codes

    def test_non_mapping_batch_refused(self):
        for bad in (None, [], "batch"):
            with pytest.raises(TG.TrainingGateError) as excinfo:
                TG.enforce_training_gate(bad)
            assert excinfo.value.code == TG.TRAINING_GATE_BAD_BATCH


# ----------------------------------------------------------------------
# positive: a verified 12+4 window trains (mechanism proof, FIXTURE)
# ----------------------------------------------------------------------
class TestLegitimateVerifiedReuseTrains:
    def test_record_valid_snapshot_clears_the_selection_block(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        stored = manager.record_verified_batch(_valid_snapshot(manager))
        assert stored["window_id"] == WINDOW
        assert manager.current_blocked_codes() == []
        assert manager.verified_batch_snapshot is not None

    def test_reuse_batch_is_the_full_verified_16_with_evidence(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        batch = manager.build_training_batch()
        assert batch["training_permitted"] is True
        assert batch["provenance"] == "REUSE_VERIFIED_WINDOW"
        assert batch["task_ids"][:12] == DYNAMIC_IDS
        assert batch["task_ids"][12:] == list(L.ANCHOR_TASK_IDS)
        assert batch["task_ids"][-1] == "original_craftax"
        assert batch["reuse_only"] is True
        assert batch["blocked_codes"] == []
        assert set(batch["layout"]) == set(batch["task_ids"])
        evidence = batch["reuse_evidence"]
        assert evidence["window_id"] == WINDOW
        assert evidence["provenance"] == "CANDIDATE_EVALUATION"
        assert (
            evidence["anchor_manifest_sha256"]
            == manager.anchor_manifest.manifest_sha256
        )
        assert len(evidence["dynamic_tasks"]) == 12
        for entry in evidence["dynamic_tasks"]:
            assert len(entry["spec_hash"]) == 64
            assert len(entry["code_sha256"]) == 64

    def test_gate_passes_and_the_session_trains_exactly_once(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        train_calls = []
        batch = _run_session_step(manager, train_calls)
        assert batch["training_permitted"] is True
        assert len(train_calls) == 1
        assert train_calls[0][:12] == DYNAMIC_IDS
        assert train_calls[0][12:] == list(L.ANCHOR_TASK_IDS)

    def test_promotion_builds_permitted_16_and_becomes_reuse_source(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)  # window A (verified)
        manager.record_verified_batch(_valid_snapshot(manager))
        # window B: 12 newly compiled artifacts
        new_ids = [f"e1-w000008::fam_b::v{i:02d}" for i in range(1, 13)]
        _consume_window_artifacts(manager, task_ids=new_ids,
                                    window_id="e1-w000008")
        batch = manager.build_training_batch(new_ids)
        assert batch["training_permitted"] is True
        assert batch["provenance"] == "PROMOTED_SELECTION"
        assert batch["task_ids"][:12] == new_ids
        assert batch["task_ids"][12:] == list(L.ANCHOR_TASK_IDS)
        # the REUSE source advanced to window B
        reuse = manager.build_training_batch()
        assert reuse["provenance"] == "REUSE_VERIFIED_WINDOW"
        assert reuse["task_ids"][:12] == new_ids
        assert reuse["reuse_evidence"]["window_id"] == "e1-w000008"

    def test_promotion_with_unknown_ids_fails_closed(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        ghost = [f"e1-w000008::ghost::v{i:02d}" for i in range(1, 13)]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(ghost)
        assert excinfo.value.code == GM.GEN_MANAGER_MISSING_FIELD

    def test_promotion_mixing_windows_fails_closed(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        new_ids = [f"e1-w000008::fam_b::v{i:02d}" for i in range(1, 13)]
        _consume_window_artifacts(manager, task_ids=new_ids,
                                    window_id="e1-w000008")
        mixed = DYNAMIC_IDS[:6] + new_ids[:6]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(mixed)
        assert excinfo.value.code == GM.GEN_MANAGER_BAD_DYNAMIC_SET

    def test_snapshot_copy_is_read_only(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        copy = manager.verified_batch_snapshot
        copy["dynamic_tasks"][0]["task_id"] = "HACKED"
        copy["window_id"] = "HACKED"
        fresh = manager.verified_batch_snapshot
        assert fresh["window_id"] == WINDOW
        assert fresh["dynamic_tasks"][0]["task_id"] == DYNAMIC_IDS[0]


# ----------------------------------------------------------------------
# requirement 4 matrix: every blocked scenario keeps training at zero,
# the verified 12+4 scenario trains
# ----------------------------------------------------------------------
class TestRequirement4Matrix:
    def test_draft_manifest_no_training(self):
        train_calls = []
        with pytest.raises(TG.TrainingGateError):
            _run_session_step(_manager(), train_calls)
        assert train_calls == []

    def test_no_dual_probes_no_training(self):
        # G1/G3/thresholds clear but real dual-probe selection never
        # completed => no verified window => blocked
        manager = _unblocked_manager()
        train_calls = []
        with pytest.raises(TG.TrainingGateError) as excinfo:
            _run_session_step(manager, train_calls)
        assert "SELECTION_BLOCKED_NO_REAL_EVIDENCE" in excinfo.value.codes
        assert train_calls == []

    def test_empty_reuse_snapshot_no_training(self):
        manager = _unblocked_manager()
        assert manager.verified_batch_snapshot is None
        batch = manager.build_training_batch()
        assert batch["reuse_evidence"] is None
        assert batch["training_permitted"] is False

    def test_forged_snapshot_no_training(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        snapshot = _valid_snapshot(manager)
        snapshot["anchor_manifest_sha256"] = "0" * 64  # forged evidence
        with pytest.raises(GM.GenManagerError):
            manager.record_verified_batch(snapshot)
        train_calls = []
        with pytest.raises(TG.TrainingGateError):
            _run_session_step(manager, train_calls)
        assert train_calls == []

    def test_legitimate_12_plus_4_trains(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        train_calls = []
        batch = _run_session_step(manager, train_calls)
        assert batch["training_permitted"] is True
        assert len(train_calls) == 1 and len(train_calls[0]) == 16
