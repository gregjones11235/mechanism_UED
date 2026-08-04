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
from dicode.teachers.e1_formal.reference_contract import (
    consume_reference_identity_contract,
    reference_identity_sha256,
)
from dicode.teachers.e1_formal.student_contract import (
    PINNED_STUDENT_CANDIDATE_ID,
)

from test_gen_manager_duck import _frozen_manifest, _manager, _teacher_config
from test_reference_contract import _block

WINDOW = "e1-w000007"
#: FIXTURE window hash (test-only; stands in for the canonical hash of
#: a real review window, which does not exist this round)
WINDOW_HASH = "c1" * 32
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


def _consume_window_artifacts(
    manager, task_ids=None, window_id=WINDOW, window_hash=WINDOW_HASH
):
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
                    "window_hash": window_hash,
                    "compiled": True,
                    "compile_note": "",
                },
            }
        )
    manager.consume_worker_results(workers)
    return ids


def _attestation(**overrides):
    """FIXTURE adapter-minted dual-probe attestation (C15).

    Stands in for the record the CC4 evaluation seam (shared
    StudentAdapter + frozen Reference) would mint after real dual
    probes. Explicitly a fixture: no real probe rollout happened this
    round; the adapter id is a test placeholder.
    """
    att = {
        "adapter_id": "cc4-student-adapter-fixture-v1",
        "student_candidate_id": PINNED_STUDENT_CANDIDATE_ID,
        "student_probe_id": "fixture-student-probe-0001",
        "student_probe_hash": "d1" * 32,
        "reference_candidate_id": (
            "REFERENCE_CANDIDATE_FROZEN_BY_SUPERVISOR"
        ),
        "reference_probe_id": "fixture-reference-probe-0001",
        "reference_probe_hash": "d2" * 32,
    }
    att.update(overrides)
    return att


def _dual_probe(**overrides):
    """FIXTURE structured Student/Reference dual-probe evidence block.

    Derived from the fixture attestation (C15) so the probe strings
    match a minted attestation unless a test says otherwise.
    Explicitly a fixture: no real probe rollout happened this round.
    """
    att = _attestation()
    probe = {name: att[name] for name in GM._DUAL_PROBE_FIELDS}
    probe.update(overrides)
    return probe


def _mutated_frozen_anchor_mapping():
    """The same FIXTURE manifest with one anchor's params hash changed
    — a DIFFERENT manifest sha (simulates a re-frozen manifest)."""
    mapping = _frozen_anchor_mapping()
    mapping["anchors"][0]["task_params_hash"] = "ff" * 32
    payload = {"status": mapping["status"], "anchors": mapping["anchors"]}
    mapping["manifest_sha256"] = canonical_sha256(payload)
    return mapping


def _valid_snapshot(manager, window_id=WINDOW, mint_probe=True):
    """Build a structurally honest snapshot from the teacher's OWN
    registry evidence (never from thin air).

    C15: legitimate REUSE evidence carries ADAPTER-MINTED probes, so
    the fixture mints its fixture attestation exactly as the CC4 seam
    would mint a real one (``mint_probe``); blocked managers (unfrozen
    contract) stay unminted — ``record_verified_batch`` refuses them
    on the gate check before it ever looks at the probes.
    """
    if mint_probe and manager.reference_contract is not None:
        manager.record_dual_probe_attestation(
            _attestation(
                reference_candidate_id=(
                    manager.reference_contract.candidate_id
                )
            )
        )
    dynamic = []
    window_hashes = set()
    for task_id, record in manager.artifact_registry.items():
        window_hashes.add(record["window_hash"])
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
        # the window hash is taken from the registry evidence itself —
        # every recorded artifact of the window carries the same value
        "window_hash": (
            window_hashes.pop() if len(window_hashes) == 1 else "0" * 64
        ),
        "provenance": "CANDIDATE_EVALUATION",
        # for blocked-certification tests the contract is None; the
        # gate rejects on the frozen-state check before it ever trusts
        # these fields
        "reference_candidate_id": (
            contract.candidate_id
            if contract is not None
            else "REFERENCE_CANDIDATE_FROZEN_BY_SUPERVISOR"
        ),
        "reference_identity_hash": (
            reference_identity_sha256(contract)
            if contract is not None
            else "b0" * 32
        ),
        "anchor_task_ids": list(L.ANCHOR_TASK_IDS),
        "anchor_manifest_sha256": manager.anchor_manifest.manifest_sha256,
        "candidate_set_hash": canonical_sha256(
            [entry["task_id"] for entry in dynamic]
        ),
        "dual_probe": _dual_probe(),
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
# C14: structured dual-probe evidence binding — bypass attempts
# ----------------------------------------------------------------------
class TestC14EvidenceBindingBypassAttempts:
    """Every C14 structured-evidence bypass attempt fails closed:
    artifact_id must equal the internal registry; dual-probe ids and
    hashes, Reference identity hash, window hash and candidate-set
    hash are all mandatory; a provenance string alone NEVER certifies.
    """

    def _manager_with_window(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        return manager

    # --- artifact_id must equal the internal registry ------------------
    def test_forged_artifact_id_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][0]["artifact_id"] = "forged::a9"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH
        assert manager.verified_batch_snapshot is None

    def test_swapped_artifact_ids_rejected(self):
        # quoting a REAL registry artifact id of ANOTHER task is still
        # a forgery (artifact_id must equal THIS task's registry record)
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dynamic_tasks"][0]["artifact_id"] = (
            snapshot["dynamic_tasks"][1]["artifact_id"]
        )
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    # --- structured dual-probe block ------------------------------------
    def test_missing_dual_probe_block_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        del snapshot["dual_probe"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize("bad", [None, [], "probe", 7])
    def test_dual_probe_must_be_a_mapping(self, bad):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dual_probe"] = bad
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_dual_probe_unknown_field_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dual_probe"]["notes"] = "trust me"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    @pytest.mark.parametrize("field", GM._DUAL_PROBE_FIELDS)
    def test_every_dual_probe_field_is_mandatory(self, field):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        del snapshot["dual_probe"][field]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    def test_dual_probes_must_run_on_the_pinned_strong_student(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dual_probe"]["student_candidate_id"] = "SOME_OTHER_STUDENT"
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    @pytest.mark.parametrize(
        "field", ["student_probe_id", "reference_probe_id"]
    )
    @pytest.mark.parametrize("value", ["", "   ", 42, None])
    def test_probe_ids_must_be_non_empty_strings(self, field, value):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dual_probe"][field] = value
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize(
        "field", ["student_probe_hash", "reference_probe_hash"]
    )
    @pytest.mark.parametrize(
        "value", ["not-hex", "D1" * 32, "d1" * 31, "d1" * 33, 123, None]
    )
    def test_probe_hashes_must_be_sha256_hex(self, field, value):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["dual_probe"][field] = value
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    # --- window hash ------------------------------------------------------
    def test_missing_window_hash_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        del snapshot["window_hash"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize("bad", ["", "xyz", "C1" * 32, "c1" * 31, 42])
    def test_window_hash_must_be_sha256_hex(self, bad):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["window_hash"] = bad
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_window_hash_mismatch_vs_registry_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["window_hash"] = "aa" * 32  # well-formed, wrong window
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_registry_without_window_hash_can_never_certify(self):
        # artifacts consumed without a recorded window hash are never
        # certifiable, no matter what the snapshot claims
        manager = _unblocked_manager()
        _consume_window_artifacts(manager, window_hash="")
        snapshot = _valid_snapshot(manager)
        snapshot["window_hash"] = "aa" * 32
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    # --- Reference identity hash ------------------------------------------
    def test_missing_reference_identity_hash_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        del snapshot["reference_identity_hash"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize("bad", ["", "zz", "b0" * 31, 42])
    def test_reference_identity_hash_must_be_sha256_hex(self, bad):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["reference_identity_hash"] = bad
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_wrong_reference_identity_hash_rejected(self):
        # well-formed hash of SOME OTHER Reference identity => forgery
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["reference_identity_hash"] = "ab" * 32
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    # --- candidate-set hash ------------------------------------------------
    def test_missing_candidate_set_hash_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        del snapshot["candidate_set_hash"]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize("bad", ["", "no", "e3" * 33, 42])
    def test_candidate_set_hash_must_be_sha256_hex(self, bad):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["candidate_set_hash"] = bad
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_reversed_candidate_set_hash_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["candidate_set_hash"] = canonical_sha256(
            list(reversed(DYNAMIC_IDS))
        )
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_candidate_set_hash_of_other_ids_rejected(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        snapshot["candidate_set_hash"] = canonical_sha256(
            [f"ghost::{i}" for i in range(12)]
        )
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    # --- provenance alone NEVER certifies ----------------------------------
    def test_correct_provenance_without_evidence_never_certifies(self):
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)
        stripped = {
            "window_id": snapshot["window_id"],
            "provenance": "CANDIDATE_EVALUATION",
            "reference_candidate_id": snapshot["reference_candidate_id"],
            "anchor_task_ids": list(L.ANCHOR_TASK_IDS),
            "anchor_manifest_sha256": snapshot["anchor_manifest_sha256"],
            "dynamic_tasks": snapshot["dynamic_tasks"],
        }  # window_hash / identity hash / candidate-set hash / dual_probe
        # all absent: the exact CANDIDATE_EVALUATION string alone must
        # never certify REUSE
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(stripped)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD
        assert manager.verified_batch_snapshot is None
        train_calls = []
        with pytest.raises(TG.TrainingGateError):
            _run_session_step(manager, train_calls)
        assert train_calls == []


# ----------------------------------------------------------------------
# C14: promotion-path bypass attempts (ids + provenance are not enough)
# ----------------------------------------------------------------------
class TestC14PromotionBypassAttempts:
    def _ready_manager(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        return manager

    def test_promotion_without_dual_probe_fails_closed(self):
        manager = self._ready_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(DYNAMIC_IDS)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize("bad", [[], "probe", 7])
    def test_promotion_dual_probe_must_be_a_mapping(self, bad):
        manager = self._ready_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(DYNAMIC_IDS, dual_probe=bad)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_promotion_with_wrong_student_candidate_fails_closed(self):
        manager = self._ready_manager()
        probe = _dual_probe(student_candidate_id="NOT_THE_PINNED_STUDENT")
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(DYNAMIC_IDS, dual_probe=probe)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_promotion_with_malformed_probe_hash_fails_closed(self):
        manager = self._ready_manager()
        probe = _dual_probe(reference_probe_hash="nothex")
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(DYNAMIC_IDS, dual_probe=probe)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_promotion_never_certifies_without_registry_window_hash(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)  # window A (with window hash)
        manager.record_verified_batch(_valid_snapshot(manager))
        # window B: artifacts consumed WITHOUT a recorded window hash
        new_ids = [f"e1-w000008::fam_c::v{i:02d}" for i in range(1, 13)]
        _consume_window_artifacts(manager, task_ids=new_ids,
                                    window_id="e1-w000008", window_hash="")
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(new_ids, dual_probe=_dual_probe())
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH
        # the REUSE source stays window A's verified snapshot
        assert manager.verified_batch_snapshot["window_id"] == WINDOW

    def test_failed_promotion_attempt_keeps_training_at_zero(self):
        manager = self._ready_manager()
        with pytest.raises(GM.GenManagerError):
            manager.build_training_batch(
                DYNAMIC_IDS, dual_probe=_dual_probe(reference_probe_id="")
            )
        # the REUSE source stays the honest one; zero training either way
        train_calls = []
        batch = _run_session_step(manager, train_calls)
        assert batch["provenance"] == "REUSE_VERIFIED_WINDOW"
        assert len(train_calls) == 1
        assert train_calls[0][:12] == DYNAMIC_IDS


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
        # C14 structured evidence rides with the REUSE batch
        assert evidence["window_hash"] == WINDOW_HASH
        assert evidence["reference_identity_hash"] == (
            reference_identity_sha256(manager.reference_contract)
        )
        assert evidence["candidate_set_hash"] == canonical_sha256(
            DYNAMIC_IDS
        )
        assert evidence["dual_probe"] == _dual_probe()
        assert (
            evidence["dual_probe"]["student_candidate_id"]
            == PINNED_STUDENT_CANDIDATE_ID
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
        # C14: promotion requires structured dual-probe evidence
        batch = manager.build_training_batch(new_ids, dual_probe=_dual_probe())
        assert batch["training_permitted"] is True
        assert batch["provenance"] == "PROMOTED_SELECTION"
        assert batch["task_ids"][:12] == new_ids
        assert batch["task_ids"][12:] == list(L.ANCHOR_TASK_IDS)
        # the REUSE source advanced to window B, carrying full C14
        # structured evidence
        reuse = manager.build_training_batch()
        assert reuse["provenance"] == "REUSE_VERIFIED_WINDOW"
        assert reuse["task_ids"][:12] == new_ids
        evidence = reuse["reuse_evidence"]
        assert evidence["window_id"] == "e1-w000008"
        assert evidence["dual_probe"] == _dual_probe()
        assert evidence["window_hash"] == WINDOW_HASH
        assert evidence["reference_identity_hash"] == (
            reference_identity_sha256(manager.reference_contract)
        )
        assert evidence["candidate_set_hash"] == canonical_sha256(new_ids)

    def test_promotion_with_unknown_ids_fails_closed(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        ghost = [f"e1-w000008::ghost::v{i:02d}" for i in range(1, 13)]
        # even WITH valid dual-probe evidence, ghost ids fail closed
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(ghost, dual_probe=_dual_probe())
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
            manager.build_training_batch(mixed, dual_probe=_dual_probe())
        assert excinfo.value.code == GM.GEN_MANAGER_BAD_DYNAMIC_SET

    def test_snapshot_copy_is_read_only(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        copy = manager.verified_batch_snapshot
        copy["dynamic_tasks"][0]["task_id"] = "HACKED"
        copy["window_id"] = "HACKED"
        copy["dual_probe"]["student_probe_id"] = "HACKED"
        fresh = manager.verified_batch_snapshot
        assert fresh["window_id"] == WINDOW
        assert fresh["dynamic_tasks"][0]["task_id"] == DYNAMIC_IDS[0]
        assert (
            fresh["dual_probe"]["student_probe_id"]
            == "fixture-student-probe-0001"
        )


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


# ----------------------------------------------------------------------
# C15: the adapter-minted dual-probe attestation seam (fail-closed)
# ----------------------------------------------------------------------
class TestC15DualProbeAttestationSeam:
    """Caller-supplied probe strings alone never suffice: only probes
    minted through the adapter seam may ever certify REUSE, and the
    mint seam itself is fail-closed on every field."""

    def test_blocked_teacher_cannot_mint_attestations(self):
        with pytest.raises(GM.GenManagerError) as excinfo:
            _manager().record_dual_probe_attestation(_attestation())
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BLOCKED
        assert "REFERENCE_CONTRACT_UNFROZEN" in str(excinfo.value)

    def test_minted_attestation_is_recorded_and_read_only(self):
        manager = _unblocked_manager()
        assert manager.probe_attestations == []
        minted = manager.record_dual_probe_attestation(_attestation())
        assert minted == _attestation()
        listed = manager.probe_attestations
        assert listed == [_attestation()]
        listed[0]["student_probe_id"] = "HACKED"
        listed.append({"forged": "entry"})
        assert manager.probe_attestations == [_attestation()]

    def test_duplicate_attestation_is_minted_once(self):
        manager = _unblocked_manager()
        manager.record_dual_probe_attestation(_attestation())
        manager.record_dual_probe_attestation(_attestation())
        assert len(manager.probe_attestations) == 1

    @pytest.mark.parametrize("bad", [None, [], "attestation", 3])
    def test_attestation_must_be_a_mapping(self, bad):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(bad)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_unknown_attestation_field_rejected(self):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(extra="trust me")
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    @pytest.mark.parametrize("field", GM._PROBE_ATTESTATION_FIELDS)
    def test_every_attestation_field_is_mandatory(self, field):
        manager = _unblocked_manager()
        att = _attestation()
        del att[field]
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(att)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    @pytest.mark.parametrize("value", ["", "   ", 42, None])
    def test_adapter_id_must_be_a_non_empty_string(self, value):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(adapter_id=value)
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    def test_attestation_must_name_the_pinned_strong_student(self):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(student_candidate_id="SOME_OTHER_STUDENT")
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_attestation_must_name_the_current_reference(self):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(reference_candidate_id="SOME_OTHER_REF")
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    @pytest.mark.parametrize(
        "field", ["student_probe_id", "reference_probe_id"]
    )
    @pytest.mark.parametrize("value", ["", "   ", 42, None])
    def test_attestation_probe_ids_must_be_non_empty(self, field, value):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(**{field: value})
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISSING_FIELD

    def test_identical_probe_ids_rejected_as_swapped_or_degenerate(self):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(
                    reference_probe_id="fixture-student-probe-0001"
                )
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    @pytest.mark.parametrize(
        "field", ["student_probe_hash", "reference_probe_hash"]
    )
    @pytest.mark.parametrize(
        "value", ["not-hex", "D1" * 32, "d1" * 31, "d1" * 33, 7, None]
    )
    def test_attestation_probe_hashes_sha256_hex(self, field, value):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(**{field: value})
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE

    def test_identical_probe_hashes_rejected_as_degenerate(self):
        manager = _unblocked_manager()
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_dual_probe_attestation(
                _attestation(reference_probe_hash="d1" * 32)
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH


# ----------------------------------------------------------------------
# C15: caller strings alone never suffice at certification time
# ----------------------------------------------------------------------
class TestC15CallerStringsNeverSuffice:
    """A dual_probe block is accepted ONLY if adapter-minted: forged,
    swapped or stale probe strings never certify REUSE or promotion."""

    def _manager_with_window(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        return manager

    def test_unminted_probe_strings_never_certify(self):
        # well-formed probe ids/hashes that no adapter ever minted
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager, mint_probe=False)
        snapshot["dual_probe"] = _dual_probe(
            student_probe_id="caller-claims-probe-0002",
            reference_probe_id="caller-claims-probe-0003",
            student_probe_hash="d3" * 32,
            reference_probe_hash="d4" * 32,
        )
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH
        assert "attestation" in str(excinfo.value)
        assert manager.verified_batch_snapshot is None
        train_calls = []
        with pytest.raises(TG.TrainingGateError):
            _run_session_step(manager, train_calls)
        assert train_calls == []

    def test_swapped_student_reference_probe_ids_never_certify(self):
        # every string here is REAL (minted) — but bound to the wrong
        # role, which only the attestation binding can catch
        manager = self._manager_with_window()
        snapshot = _valid_snapshot(manager)  # mints the unswapped probe
        probe = snapshot["dual_probe"]
        probe["student_probe_id"], probe["reference_probe_id"] = (
            probe["reference_probe_id"],
            probe["student_probe_id"],
        )
        probe["student_probe_hash"], probe["reference_probe_hash"] = (
            probe["reference_probe_hash"],
            probe["student_probe_hash"],
        )
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH
        assert manager.verified_batch_snapshot is None

    def test_attestation_minted_under_old_reference_never_certifies(self):
        manager = self._manager_with_window()
        manager.record_dual_probe_attestation(_attestation())
        # the supervisor re-freezes the Reference identity under a NEW
        # candidate; the attestation minted under the old one must stop
        # certifying
        manager._reference_contract = consume_reference_identity_contract(
            _block(candidate_id="REFERENCE_CANDIDATE_REFROZEN_V2"), "t"
        )
        snapshot = _valid_snapshot(manager, mint_probe=False)
        snapshot["dual_probe"] = _dual_probe()  # attested under OLD ref
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.record_verified_batch(snapshot)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH

    def test_promotion_with_unminted_probe_fails_closed(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        new_ids = [
            f"e1-w000008::fam_d::v{i:02d}" for i in range(1, 13)
        ]
        _consume_window_artifacts(
            manager, task_ids=new_ids, window_id="e1-w000008"
        )
        probe = _dual_probe(
            student_probe_id="caller-claims-probe-0009",
            reference_probe_id="caller-claims-probe-0010",
        )
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager.build_training_batch(new_ids, dual_probe=probe)
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH
        # the honest REUSE source (window A) survives the failed try
        assert manager.verified_batch_snapshot["window_id"] == WINDOW


# ----------------------------------------------------------------------
# C15: every stored REUSE binding is re-validated on EVERY reuse
# ----------------------------------------------------------------------
class TestC15ReuseRevalidationFailsClosed:
    """Stale windows, changed identities/manifests/artifacts, reordered
    candidate sets, fabricated hashes and tampered stored state all
    invalidate REUSE fail-closed — zero training either way."""

    def _reuse_manager(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        # sanity: REUSE is legitimate right now
        assert manager.build_training_batch()["training_permitted"] is True
        return manager

    def _assert_reuse_invalidated(self, manager):
        batch = manager.build_training_batch()
        assert batch["training_permitted"] is False
        assert batch["task_ids"] == []
        assert (
            TG.TRAINING_BLOCKED_NO_VERIFIED_BATCH
            in batch["blocked_codes"]
        )
        train_calls = []
        with pytest.raises(TG.TrainingGateError):
            _run_session_step(manager, train_calls)
        assert train_calls == []  # zero training updates

    # --- stale window ----------------------------------------------------
    def test_stale_window_after_full_reconsumption(self):
        manager = self._reuse_manager()
        # a new window supersedes the certified one: the same task ids
        # recompiled under a different window hash
        _consume_window_artifacts(
            manager, window_id="e1-w000008", window_hash="c9" * 32
        )
        self._assert_reuse_invalidated(manager)

    def test_stale_window_after_single_record_reconsumed(self):
        manager = self._reuse_manager()
        _consume_window_artifacts(
            manager,
            task_ids=[DYNAMIC_IDS[3]],
            window_id="e1-w000008",
            window_hash="c9" * 32,
        )
        self._assert_reuse_invalidated(manager)

    # --- unknown / changed registry entries --------------------------------
    def test_unknown_id_after_registry_removal(self):
        manager = self._reuse_manager()
        del manager._artifact_registry[DYNAMIC_IDS[3]]
        self._assert_reuse_invalidated(manager)

    def test_changed_artifact_after_record(self):
        manager = self._reuse_manager()
        # same window, same window hash — but the artifact itself
        # changed, so the stored spec/code/artifact bindings are stale
        task_id = DYNAMIC_IDS[3]
        code = "class EnvRevised:\n    pass\n"
        manager.consume_worker_results(
            [
                {
                    "task_id": task_id,
                    "generated_task_id": task_id,
                    "compiled": True,
                    "code": code,
                    "code_string": code,
                    "reasoning": "",
                    "e1_status": {
                        "reuse": False,
                        "artifact_id": f"{task_id}::a2",
                        "spec_hash": "ee" * 32,
                        "window_id": WINDOW,
                        "window_hash": WINDOW_HASH,
                        "compiled": True,
                        "compile_note": "",
                    },
                }
            ]
        )
        self._assert_reuse_invalidated(manager)

    # --- changed Reference / protocol / manifest -----------------------------
    def test_changed_reference_identity_invalidates_reuse(self):
        manager = self._reuse_manager()
        manager._reference_contract = (
            consume_reference_identity_contract(_block(seed=43), "t")
        )
        self._assert_reuse_invalidated(manager)

    def test_changed_reset_protocol_invalidates_reuse(self):
        manager = self._reuse_manager()
        manager._reference_contract = (
            consume_reference_identity_contract(
                _block(episode_reset_protocol_id="standard_reset_v2"),
                "t",
            )
        )
        self._assert_reuse_invalidated(manager)

    def test_changed_anchor_manifest_invalidates_reuse(self):
        manager = self._reuse_manager()
        manager._anchor_manifest = AM.consume_anchor_manifest(
            _mutated_frozen_anchor_mapping(), "t"
        )
        self._assert_reuse_invalidated(manager)

    # --- tampered stored snapshot -------------------------------------------
    def test_reordered_candidate_set_invalidates_reuse(self):
        manager = self._reuse_manager()
        stored = manager._verified_batch_snapshot
        stored["dynamic_tasks"] = list(reversed(stored["dynamic_tasks"]))
        self._assert_reuse_invalidated(manager)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("window_hash", "ab" * 32),
            ("candidate_set_hash", "cd" * 32),
            ("reference_identity_hash", "ef" * 32),
        ],
    )
    def test_fabricated_hash_on_stored_snapshot(self, field, value):
        manager = self._reuse_manager()
        manager._verified_batch_snapshot[field] = value
        self._assert_reuse_invalidated(manager)

    def test_swapped_probes_on_stored_snapshot(self):
        manager = self._reuse_manager()
        probe = manager._verified_batch_snapshot["dual_probe"]
        probe["student_probe_id"], probe["reference_probe_id"] = (
            probe["reference_probe_id"],
            probe["student_probe_id"],
        )
        probe["student_probe_hash"], probe["reference_probe_hash"] = (
            probe["reference_probe_hash"],
            probe["student_probe_hash"],
        )
        self._assert_reuse_invalidated(manager)

    def test_changed_student_on_stored_snapshot(self):
        manager = self._reuse_manager()
        manager._verified_batch_snapshot["dual_probe"][
            "student_candidate_id"
        ] = "SOME_OTHER_STUDENT"
        self._assert_reuse_invalidated(manager)


# ----------------------------------------------------------------------
# C15: direct private-state manipulation never produces training
# ----------------------------------------------------------------------
class TestC15DirectMethodBypassFailsClosed:
    """Bypassing the public certification API — flipping private flags,
    assigning snapshots directly, calling private certifiers — never
    yields a trainable batch: every binding is re-validated."""

    def test_direct_flag_flip_without_snapshot_still_blocks(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager._real_selection_completed = True  # bypass bookkeeping
        batch = manager.build_training_batch()
        assert batch["training_permitted"] is False
        assert batch["task_ids"] == []
        assert (
            TG.TRAINING_BLOCKED_NO_VERIFIED_BATCH
            in batch["blocked_codes"]
        )

    def test_direct_assignment_of_fabricated_snapshot_never_trains(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        # a "perfect-looking" snapshot assembled from caller strings and
        # registry lookups — but its probes were never minted
        registry = manager.artifact_registry
        dynamic = [
            {
                "task_id": task_id,
                "artifact_id": registry[task_id]["artifact_id"],
                "spec_hash": registry[task_id]["spec_hash"],
                "code_sha256": hashlib.sha256(
                    registry[task_id]["code"].encode("utf-8")
                ).hexdigest(),
            }
            for task_id in DYNAMIC_IDS
        ]
        forged = {
            "window_id": WINDOW,
            "window_hash": WINDOW_HASH,
            "provenance": "CANDIDATE_EVALUATION",
            "reference_candidate_id": (
                manager.reference_contract.candidate_id
            ),
            "reference_identity_hash": reference_identity_sha256(
                manager.reference_contract
            ),
            "anchor_task_ids": list(L.ANCHOR_TASK_IDS),
            "anchor_manifest_sha256": (
                manager.anchor_manifest.manifest_sha256
            ),
            "candidate_set_hash": canonical_sha256(DYNAMIC_IDS),
            "dual_probe": _dual_probe(),  # never minted on this manager
            "dynamic_tasks": dynamic,
        }
        manager._verified_batch_snapshot = forged
        manager._real_selection_completed = True
        batch = manager.build_training_batch()
        assert batch["training_permitted"] is False
        assert batch["task_ids"] == []
        train_calls = []
        with pytest.raises(TG.TrainingGateError):
            _run_session_step(manager, train_calls)
        assert train_calls == []

    def test_direct_assignment_with_tampered_binding_never_trains(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        manager.record_verified_batch(_valid_snapshot(manager))
        stored = manager.verified_batch_snapshot  # deep copy
        stored["dynamic_tasks"][0]["artifact_id"] = "forged::a9"
        manager._verified_batch_snapshot = stored
        batch = manager.build_training_batch()
        assert batch["training_permitted"] is False
        assert batch["task_ids"] == []

    def test_direct_certify_call_with_unminted_probe_fails_closed(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager._certify_dynamic_window(
                WINDOW,
                DYNAMIC_IDS,
                "CANDIDATE_EVALUATION",
                "e1_formal.test.direct",
                dual_probe=_dual_probe(),
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_MISMATCH
        assert manager.verified_batch_snapshot is None

    def test_direct_certify_call_with_malformed_probe_fails_closed(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        with pytest.raises(GM.GenManagerError) as excinfo:
            manager._certify_dynamic_window(
                WINDOW,
                DYNAMIC_IDS,
                "CANDIDATE_EVALUATION",
                "e1_formal.test.direct",
                dual_probe="not-a-mapping",
            )
        assert excinfo.value.code == GM.GEN_MANAGER_SNAPSHOT_BAD_TYPE
        assert manager.verified_batch_snapshot is None


# ----------------------------------------------------------------------
# C15: the ONE adapter-minted positive path (FIXTURE throughout)
# ----------------------------------------------------------------------
class TestC15AdapterMintedPositivePath:
    """Legitimate flow: the CC4 seam mints the dual-probe attestation;
    the certified window then reuses and trains. Every value is an
    explicitly labeled fixture — no real probe happened this round."""

    def test_adapter_minted_probe_certifies_reuse_and_trains_once(self):
        manager = _unblocked_manager()
        _consume_window_artifacts(manager)
        # the adapter seam mints the probe evidence BEFORE certification
        minted = manager.record_dual_probe_attestation(_attestation())
        assert minted["adapter_id"] == "cc4-student-adapter-fixture-v1"
        snapshot = _valid_snapshot(manager, mint_probe=False)
        assert snapshot["dual_probe"] == {
            name: minted[name] for name in GM._DUAL_PROBE_FIELDS
        }
        stored = manager.record_verified_batch(snapshot)
        assert stored["dual_probe"]["student_probe_id"] == (
            minted["student_probe_id"]
        )
        train_calls = []
        batch = _run_session_step(manager, train_calls)
        assert batch["training_permitted"] is True
        assert batch["provenance"] == "REUSE_VERIFIED_WINDOW"
        assert len(train_calls) == 1
        assert train_calls[0][:12] == DYNAMIC_IDS
        assert train_calls[0][12:] == list(L.ANCHOR_TASK_IDS)
        evidence = batch["reuse_evidence"]
        assert evidence["dual_probe"]["student_probe_id"] == (
            "fixture-student-probe-0001"
        )
        assert evidence["dual_probe"]["reference_probe_id"] == (
            "fixture-reference-probe-0001"
        )
        assert (
            evidence["dual_probe"]["student_candidate_id"]
            == PINNED_STUDENT_CANDIDATE_ID
        )
        # re-validated on EVERY reuse: the bindings still hold, so the
        # next session reuses the same verified window
        assert manager.build_training_batch()["training_permitted"] is True
