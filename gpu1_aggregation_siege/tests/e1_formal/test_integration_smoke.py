"""C11 integration smoke (plan §十.13).

Asserts the HONEST degraded state of the wired teacher end-to-end:

* double-run equality of ``evolve_tasks`` + zero ledger calls;
* the four disclaimers and all-REAL_*-false flags in the status
  report;
* the G1/G2/G3 block CODES appear where real evidence is missing —
  never fabricated values standing in for probes/retention;
* the evaluation seam's first degradation-chain link emits exactly
  the G1 block code while the contract is unfrozen.

Nothing here trains, calls an API, or invents evidence.
"""
from dicode.teachers.e1_formal import anchor_manifest as AM
from dicode.teachers.e1_formal import layout
from dicode.teachers.e1_formal.metrics import LEARNABILITY_THRESHOLD_MISSING

from test_candidate_evaluation_seam import _load_seam
from test_gen_manager_duck import _manager

G1_UNFROZEN = "REFERENCE_CONTRACT_UNFROZEN"
G2_SELECTION_BLOCKED = "SELECTION_BLOCKED_NO_REAL_EVIDENCE"


class TestDoubleRunDeterminism:
    def test_evolve_tasks_double_run_equality_zero_calls(self):
        manager = _manager()
        first = manager.evolve_tasks()
        second = manager.evolve_tasks()
        assert first == second
        assert len(first) == 12
        counts = manager.ledger.counts()
        assert counts["N1"] == 0  # no evidence -> no window -> no calls
        assert counts["board_calls"] == 0

    def test_poisoned_arguments_do_not_change_the_result(self):
        manager = _manager()
        poisoned = manager.evolve_tasks(
            {"mastered": ["cheat"]}, {"reward": 999.0}
        )
        clean = manager.evolve_tasks()
        assert poisoned == clean


class TestStatusReportHonesty:
    def test_four_disclaimers_present(self):
        report = _manager().status_report()
        disclaimers = report["disclaimers"]
        assert len(disclaimers) == 4
        joined = " ".join(disclaimers)
        assert "E1_FORMAL_PLAN_ALIGNED" in joined
        assert "engineering alignment only" in joined
        assert "E1S_STATIC_ABLATION_PRESERVED" in joined
        assert "REAL_*" in joined
        assert "anchors + REUSE" in joined

    def test_all_real_flags_false(self):
        report = _manager().status_report()
        assert report["flags"] == {
            "real_envcoder_used": False,
            "real_student_reference_eval": False,
            "real_training_update_executed": False,
        }

    def test_state_fields_report_degradation_not_fabrication(self):
        report = _manager().status_report()
        assert report["reference_contract_frozen"] is False
        assert report["learnability_thresholds_present"] is False
        assert report["anchor_manifest_status"] == AM.STATUS_DRAFT_UNFROZEN
        assert report["flags"]["real_envcoder_used"] is False


class TestGateBlockCodesAppearNotValues:
    def test_all_four_gate_codes_present(self):
        codes = _manager().current_blocked_codes()
        assert G1_UNFROZEN in codes                    # G1
        assert LEARNABILITY_THRESHOLD_MISSING in codes   # G2 thresholds
        assert G2_SELECTION_BLOCKED in codes           # G2 selection
        assert AM.BLOCKED_SHARED_ANCHOR_MANIFEST in codes  # G3

    def test_blocked_batch_is_anchors_only_with_codes(self):
        manager = _manager()
        batch = manager.build_training_batch()
        # NO fabricated dynamic task id anywhere in the batch
        assert batch["task_ids"] == list(layout.ANCHOR_TASK_IDS)
        assert batch["task_ids"][-1] == "original_craftax"
        assert batch["reuse_only"] is True
        assert batch["layout"] is None
        assert batch["dynamic_promoted"] == 0
        for code in (
            G1_UNFROZEN,
            LEARNABILITY_THRESHOLD_MISSING,
            G2_SELECTION_BLOCKED,
            AM.BLOCKED_SHARED_ANCHOR_MANIFEST,
        ):
            assert code in batch["blocked_codes"]

    def test_batch_notes_cite_g2_and_g3_not_numbers(self):
        batch = _manager().build_training_batch()
        joined = " ".join(batch["notes"])
        assert "G2" in joined
        assert "G3" in joined
        # no fake learnability prior smuggled into the notes
        assert "0.25" not in joined

    def test_reuse_batch_from_evolve_carries_the_codes(self):
        manager = _manager()
        workers = manager.evolve_tasks()
        assert all(w["e1_status"]["reuse"] for w in workers)
        blocked = workers[0]["e1_status"]["blocked_codes"]
        assert G1_UNFROZEN in blocked
        assert AM.BLOCKED_SHARED_ANCHOR_MANIFEST in blocked


class TestDegradationChainFirstLink:
    def test_seam_blocks_with_g1_code_while_contract_unfrozen(self):
        CE = _load_seam()
        manager = _manager()
        assert manager.reference_contract is None  # unfrozen by config
        result = CE.evaluate_candidates_with_reference(
            {"candidate_eval": {"enabled": True}},
            None,
            ["dyn_a"],
            manager.archive,
            None,
            object(),  # even WITH states present...
            object(),
            manager.flags,
            manager.reference_contract,  # ...G1 blocks first
        )
        assert result["status"] == CE.EVAL_BLOCKED_REFERENCE_CONTRACT_UNFROZEN
        assert result["evaluated"] is False
        assert result["results"] == ()
        assert "provenance" not in result  # nothing was evaluated

    def test_chain_codes_line_up_between_teacher_and_seam(self):
        CE = _load_seam()
        manager = _manager()
        # the teacher's own blocked list names the same G1 code the
        # seam emits — one degradation chain, one vocabulary
        assert G1_UNFROZEN in manager.current_blocked_codes()
        result = CE.evaluate_candidates_with_reference(
            {}, None, [], None, None, None, None, None, None
        )
        assert result["status"].endswith(G1_UNFROZEN)
