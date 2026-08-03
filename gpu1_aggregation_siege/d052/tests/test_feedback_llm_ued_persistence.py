"""C15 / P1-7: persistence + cross-window restore equivalence.

The double-window state machine must survive a crash/restart at a window
boundary WITHOUT changing what the run concludes:

* freeze-point snapshot -> restore -> continue reproduces the UNINTERRUPTED
  run's RunSummary byte-for-byte (normal AND shuffled mode);
* a FRESH SUBPROCESS restoring the same snapshot file reproduces the
  identical summary hash (no in-process memory can leak into the resume);
* ANY tamper — top-level payload, snapshot hash, or a re-signed deep record
  (ledger status, store field) — fails closed with HASH_CHAIN_BROKEN;
* the hypothesis revision chain itself is verified entry-by-entry
  (status linkage, monotone windows, legal previous_record_hash, final
  status == chain end);
* ``save_controller`` is atomic (tmp + ``os.replace``, no leftover tmp);
* a REQUEST_CONTROL-stopped run restores still stopped: no new windows, no
  new backend calls, identical summary including the decision artifact.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from d052.bagr_ued.hashing import canonical_json, canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import persistence as P
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.hypothesis_ledger import HypothesisRecord
from d052.feedback_llm_ued.llm_backend import DeterministicMockFeedbackBackend

#: gpu1_aggregation_siege/ (d052/tests/this_file -> parents[2])
REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOWS = 4
FREEZE_AT = 2
BROKEN = "HASH_CHAIN_BROKEN"


def _run(mode, max_windows):
    return FeedbackUEDController(mode).run(max_windows=max_windows)


def _freeze(mode=C.MODE_NORMAL_FEEDBACK, freeze_at=FREEZE_AT):
    ctl = FeedbackUEDController(mode)
    ctl.run(max_windows=freeze_at)
    return ctl


class TestCrossWindowRestoreEquivalence:
    """snapshot at the freeze point -> restore -> continue == the
    uninterrupted run, byte-for-byte over the canonical summary JSON."""

    @pytest.mark.parametrize("mode", [C.MODE_NORMAL_FEEDBACK,
                                      C.MODE_SHUFFLED_FEEDBACK])
    def test_freeze_restore_continue_matches_uninterrupted(self, mode,
                                                           tmp_path):
        reference = _run(mode, WINDOWS)
        ctl = _freeze(mode)
        path = str(tmp_path / "snapshot.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(path)
        resumed = restored.run(max_windows=WINDOWS)
        assert canonical_json(resumed.to_dict()) == \
            canonical_json(reference.to_dict())

    def test_restore_resumes_at_the_next_window(self, tmp_path):
        ctl = _freeze()
        path = str(tmp_path / "snapshot.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(path)
        assert len(restored._completed_records) == FREEZE_AT
        assert restored._seeded is True          # seeding must NOT re-run
        restored.run(max_windows=WINDOWS)
        assert [r.window for r in restored._completed_records] == \
            list(range(WINDOWS))

    def test_restore_preserves_state_machine_fields(self, tmp_path):
        ctl = _freeze()
        path = str(tmp_path / "snapshot.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(path)
        assert restored.mode == ctl.mode
        assert restored.ledger.ids() == ctl.ledger.ids()
        assert sorted(restored.store.ids()) == sorted(ctl.store.ids())
        assert set(restored.plans) == set(ctl.plans)
        assert restored._phases == ctl._phases
        assert restored._window_feedback.keys() == ctl._window_feedback.keys()
        assert restored._sequence == ctl._sequence
        assert restored._retired_at == ctl._retired_at
        assert restored.anchor_ids == ctl.anchor_ids
        assert restored.anchor_binding == ctl.anchor_binding
        assert restored.board_hashes == ctl.board_hashes
        assert restored.runner.probe_calls == ctl.runner.probe_calls
        assert restored.runner.total_transitions == \
            ctl.runner.total_transitions
        assert restored.backend.usage.total_calls == \
            ctl.backend.usage.total_calls

    def test_human_reopen_families_round_trip(self, tmp_path):
        fam = C.ENVIRONMENT_FAMILIES[0]
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                    human_reopen_families=(fam,))
        ctl.run(max_windows=FREEZE_AT)
        path = str(tmp_path / "snapshot.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(path)
        assert restored.human_reopen_families == frozenset({fam})

    def test_real_ledger_chains_pass_verification(self, tmp_path):
        """Every hypothesis of a real 2-window run carries a revision chain
        (window-1 verdicts applied to window-0 hypotheses) and the whole
        snapshot verifies — the happy path of the chain checker."""
        ctl = _freeze()
        with_history = [d for d in ctl.ledger.dump()
                        if d["revision_history"]]
        assert with_history, "window 1 must have applied verdicts"
        payload = P.snapshot_controller(ctl)
        P.verify_snapshot_integrity(payload)       # must not raise
        for dump in with_history:
            P._verify_hypothesis_chain(dump)       # must not raise


class TestFreshProcessEquivalence:
    """A brand-new Python process that loads the snapshot from disk must
    resume to the IDENTICAL summary hash — proof that the resume point is
    fully carried by the frozen state, not by process memory."""

    _SCRIPT = (
        "import sys\n"
        "from d052.bagr_ued.hashing import canonical_sha256\n"
        "from d052.feedback_llm_ued import persistence as P\n"
        "ctl = P.load_controller(sys.argv[1])\n"
        "s = ctl.run(max_windows=%d)\n"
        "sys.stdout.write(canonical_sha256(s.to_dict()))\n" % WINDOWS
    )

    def test_subprocess_restore_matches_in_process_resume(self, tmp_path):
        reference = _run(C.MODE_NORMAL_FEEDBACK, WINDOWS)
        snap = str(tmp_path / "snapshot.json")
        P.save_controller(_freeze(), snap)

        script = tmp_path / "fresh_restore.py"
        script.write_text(self._SCRIPT, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        proc = subprocess.run(
            [sys.executable, str(script), snap],
            capture_output=True, text=True, timeout=300,
            cwd=str(REPO_ROOT), env=env)
        assert proc.returncode == 0, proc.stderr
        fresh_hash = proc.stdout.strip()

        in_process = P.load_controller(snap).run(max_windows=WINDOWS)
        assert fresh_hash == canonical_sha256(in_process.to_dict())
        assert fresh_hash == canonical_sha256(reference.to_dict())


class TestTamperFailClosed:
    """Any tamper that is not re-signed breaks the top-level snapshot hash;
    the matrix below covers every major state-machine field."""

    @pytest.mark.parametrize("mutate", [
        lambda p: p.__setitem__("snapshot_hash", "0" * 64),
        lambda p: p["ledger"][0].__setitem__("status", C.HYPOTHESIS_SUPPORTED),
        lambda p: p["store"][0].__setitem__("window", 99),
        lambda p: p["revisions"][0].__setitem__("window", 99),
        lambda p: p.__setitem__("mode", C.MODE_SHUFFLED_FEEDBACK),
        lambda p: p["retired_at"].__setitem__(
            C.ENVIRONMENT_FAMILIES[0], 0),
        lambda p: p.__setitem__("sequence", p["sequence"] + 1),
        lambda p: p.__setitem__(
            "human_reopen_families", [C.ENVIRONMENT_FAMILIES[0]]),
        lambda p: p["completed_records"][0].__setitem__("n_llm_calls", 999),
        lambda p: p["backend_usage"].__setitem__("mock_calls", 1),
        lambda p: p.__setitem__("snapshot_version", "forged.v9"),
    ])
    def test_naive_tamper_detected(self, tmp_path, mutate):
        ctl = _freeze()
        payload = P.snapshot_controller(ctl)
        mutate(payload)
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P.restore_controller(payload)

    def test_tampered_file_on_disk_fails_to_load(self, tmp_path):
        path = str(tmp_path / "snapshot.json")
        P.save_controller(_freeze(), path)
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["ledger"][0]["confidence"] = 0.99
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P.load_controller(path)

    def test_resigned_ledger_status_tamper_still_detected(self, tmp_path):
        """Defense in depth: an attacker who RE-SIGNS the snapshot hash
        after mutating a hypothesis status still fails on the record-level
        content-hash recomputation (C14)."""
        payload = P.snapshot_controller(_freeze())
        payload["ledger"][0]["status"] = C.HYPOTHESIS_SUPPORTED
        payload["snapshot_hash"] = P._hash_payload(payload)
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P.restore_controller(payload)

    def test_resigned_store_field_tamper_still_detected(self, tmp_path):
        payload = P.snapshot_controller(_freeze())
        payload["store"][0]["window"] = 99
        payload["snapshot_hash"] = P._hash_payload(payload)
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P.restore_controller(payload)

    def test_resigned_plan_tamper_still_detected(self, tmp_path):
        payload = P.snapshot_controller(_freeze())
        plan_id = next(iter(payload["plans"]))
        payload["plans"][plan_id]["mode"] = C.MODE_SHUFFLED_FEEDBACK
        payload["snapshot_hash"] = P._hash_payload(payload)
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P.restore_controller(payload)


class TestHypothesisChainVerification:
    """Unit negatives of the per-hypothesis revision-chain checks. Dumps
    carry record_hash='' so reconstruction succeeds and the CHAIN checks
    (not the content hash) are what fires."""

    def _dump(self, status, history):
        rec = HypothesisRecord(
            hypothesis_id="hyp-chain", source_window=0,
            target_behavior="chain_probe",
            environment_family=C.ENVIRONMENT_FAMILIES[0],
            confidence=0.5, status=status)
        dump = rec.model_dump()
        dump["revision_history"] = history
        dump["record_hash"] = ""
        return dump

    def _entry(self, window, prev, new, *, feedback_ids=("fb-1",),
               previous_record_hash="a" * 64):
        return dict(window=window, previous_status=prev, new_status=new,
                    feedback_ids=list(feedback_ids), reason="r",
                    previous_record_hash=previous_record_hash)

    def test_legal_chain_passes(self):
        history = [self._entry(1, C.HYPOTHESIS_PENDING,
                               C.HYPOTHESIS_INCONCLUSIVE),
                   self._entry(2, C.HYPOTHESIS_INCONCLUSIVE,
                               C.HYPOTHESIS_SUPPORTED,
                               previous_record_hash="b" * 64)]
        rec = P._verify_hypothesis_chain(
            self._dump(C.HYPOTHESIS_SUPPORTED, history))
        assert rec.status == C.HYPOTHESIS_SUPPORTED

    def test_illegal_previous_record_hash(self):
        history = [self._entry(1, C.HYPOTHESIS_PENDING,
                               C.HYPOTHESIS_SUPPORTED,
                               previous_record_hash="not-a-hash")]
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P._verify_hypothesis_chain(
                self._dump(C.HYPOTHESIS_SUPPORTED, history))

    def test_broken_status_linkage(self):
        history = [self._entry(1, C.HYPOTHESIS_PENDING,
                               C.HYPOTHESIS_INCONCLUSIVE),
                   self._entry(2, C.HYPOTHESIS_REFUTED,   # must link to
                               C.HYPOTHESIS_SUPPORTED)]    # INCONCLUSIVE
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P._verify_hypothesis_chain(
                self._dump(C.HYPOTHESIS_SUPPORTED, history))

    def test_non_monotone_windows(self):
        history = [self._entry(2, C.HYPOTHESIS_PENDING,
                               C.HYPOTHESIS_INCONCLUSIVE),
                   self._entry(1, C.HYPOTHESIS_INCONCLUSIVE,
                               C.HYPOTHESIS_SUPPORTED)]
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P._verify_hypothesis_chain(
                self._dump(C.HYPOTHESIS_SUPPORTED, history))

    def test_final_status_must_equal_chain_end(self):
        history = [self._entry(1, C.HYPOTHESIS_PENDING,
                               C.HYPOTHESIS_SUPPORTED)]
        with pytest.raises(P.SnapshotCorrupted, match=BROKEN):
            P._verify_hypothesis_chain(
                self._dump(C.HYPOTHESIS_PENDING, history))


class TestAtomicSave:
    def test_no_tmp_leftover_and_file_valid(self, tmp_path):
        path = str(tmp_path / "snapshot.json")
        P.save_controller(_freeze(), path)
        assert sorted(os.listdir(tmp_path)) == ["snapshot.json"]
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        P.verify_snapshot_integrity(payload)

    def test_resave_replaces_atomically(self, tmp_path):
        path = str(tmp_path / "snapshot.json")
        first = P.save_controller(_freeze(), path)
        second = P.save_controller(_freeze(), path)
        assert sorted(os.listdir(tmp_path)) == ["snapshot.json"]
        assert first["snapshot_hash"] == second["snapshot_hash"]
        assert P.load_controller(path)._completed_records
        # determinism: two freezes of the same mode snapshot identically
        assert canonical_json(first) == canonical_json(second)


# ------------------------------------- C11 scripted REQUEST_CONTROL backend
class _EscalatingCriticBackend(DeterministicMockFeedbackBackend):
    """Mock backend whose Nth Critic/Skeptic call demands human control
    (same deterministic escalation rule as the C11 controller tests)."""

    def __init__(self, escalate_on_critic_call: int):
        super().__init__()
        self._target = escalate_on_critic_call
        self._critic_calls = 0

    def complete(self, role, prompt):
        raw = super().complete(role, prompt)
        if role != C.ROLE_CRITIC_SKEPTIC:
            return raw
        self._critic_calls += 1
        if self._critic_calls != self._target:
            return raw
        dump = json.loads(raw)
        dump["request_control"] = True
        dump["endorsed"] = False
        dump["critique_summary"] += " | C15 test: human control requested"
        return json.dumps(dump, sort_keys=True, ensure_ascii=False)


class TestStoppedRunRestore:
    """A REQUEST_CONTROL-stopped run restores STILL STOPPED: run() must not
    open new windows, must not issue any backend call, and the summary
    (including the HumanDecisionArtifact) is byte-identical."""

    def test_stopped_restore_stays_stopped(self, tmp_path):
        ctl = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            backend=_EscalatingCriticBackend(escalate_on_critic_call=3))
        stopped = ctl.run(max_windows=6)
        assert stopped.request_control_stopped is True
        assert stopped.stopped_window == 2

        path = str(tmp_path / "stopped.json")
        P.save_controller(ctl, path)
        restored = P.load_controller(path)     # default fresh backend
        again = restored.run(max_windows=6)

        assert canonical_json(again.to_dict()) == \
            canonical_json(stopped.to_dict())
        assert again.request_control_stopped is True
        assert again.n_windows == stopped.n_windows
        # no new windows were executed after restore
        assert len(restored._completed_records) == stopped.n_windows
        # no backend calls were issued by the resumed run: usage counters
        # equal the snapshot values exactly
        assert restored.backend.usage.mock_calls == \
            ctl.backend.usage.mock_calls
        assert restored.backend.usage.total_calls == \
            ctl.backend.usage.total_calls
        assert restored.runner.total_transitions == \
            ctl.runner.total_transitions
        # the decision artifact survived the round trip hash-identical
        assert len(restored.human_decision_artifacts) == 1
        assert restored.human_decision_artifacts[0].artifact_hash == \
            ctl.human_decision_artifacts[0].artifact_hash

    def test_stopped_snapshot_round_trips_through_disk(self, tmp_path):
        ctl = FeedbackUEDController(
            C.MODE_NORMAL_FEEDBACK,
            backend=_EscalatingCriticBackend(escalate_on_critic_call=3))
        ctl.run(max_windows=6)
        path = str(tmp_path / "stopped.json")
        P.save_controller(ctl, path)
        payload = json.load(open(path, encoding="utf-8"))
        assert payload["completed_records"][-1]["request_control"] is True
        P.verify_snapshot_integrity(payload)
