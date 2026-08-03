"""C5 tests (G5): LLMCallLedger — N1 = 6*G1 + T1 + K1 + F1.

Counting rules under test:
* board calls belong to TRIGGERED windows (exactly 6 roles per window);
* EnvCoder calls are counted per UNIQUE artifact (deduped);
* repair calls are a separate counter (F1), never merged into K1;
* E1 has no TaskGenerator (T1 == 0; reconcile fails closed otherwise);
* no fixed "per-window total" language exists in the modules.
"""
from pathlib import Path

import pytest

from dicode.teachers.e1_formal import accounting as A
from dicode.teachers.e1_formal import llm_client as LC
from dicode.teachers.e1_formal import manifest as M
from dicode.teachers.e1_formal.manifest import BOARD_ROLE_ORDER
from dicode.teachers.e1_formal.schemas import E1SchemaError


def _open_and_fill(ledger, window_id):
    ledger.record_window_open(window_id)
    for role in BOARD_ROLE_ORDER:
        ledger.record_board_call(window_id, role)


class TestFormulaScenarios:
    def test_reuse_only_session_records_zero_calls(self):
        ledger = A.LLMCallLedger()
        counts = ledger.reconcile()
        assert counts["G1"] == 0
        assert counts["T1"] == 0
        assert counts["K1"] == 0
        assert counts["F1"] == 0
        assert counts["board_calls"] == 0
        assert counts["N1"] == 0
        assert ledger.to_records() == ()

    def test_one_window_ten_specs_two_variants(self):
        # Plan scenario: 1 triggered window + 10 specs x 2 variants.
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        for spec_i in range(10):
            for variant in (0, 1):
                ledger.record_envcoder_call(
                    "w01", f"w01::fam_{spec_i}::v{variant}"
                )
        counts = ledger.reconcile()
        assert counts["G1"] == 1
        assert counts["board_calls"] == 6
        assert counts["K1"] == 20
        assert counts["T1"] == 0
        assert counts["F1"] == 0
        assert counts["N1"] == 6 * 1 + 0 + 20 + 0 == 26

    def test_two_windows_board_calls_scale_with_g1(self):
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        _open_and_fill(ledger, "w02")
        counts = ledger.reconcile()
        assert counts["G1"] == 2
        assert counts["board_calls"] == 12
        assert counts["N1"] == 12

    def test_artifact_dedup_never_double_counts(self):
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        assert ledger.record_envcoder_call("w01", "art-1") is True
        assert ledger.record_envcoder_call("w01", "art-1") is False
        assert ledger.record_envcoder_call("w01", "art-1") is False
        assert ledger.record_envcoder_call("w01", "art-2") is True
        counts = ledger.reconcile()
        assert counts["K1"] == 2
        # duplicates remain in the audit trail but not in K1
        envcoder_records = [
            r for r in ledger.to_records() if r["kind"] == A.KIND_ENVCODER
        ]
        assert len(envcoder_records) == 4

    def test_repair_counter_is_independent_from_k1(self):
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        ledger.record_envcoder_call("w01", "art-1")
        ledger.record_repair_call("w01", "art-1")
        ledger.record_repair_call("w01", "art-1")
        counts = ledger.reconcile()
        assert counts["K1"] == 1
        assert counts["F1"] == 2
        assert counts["N1"] == 6 + 1 + 2

    def test_task_generator_nonzero_fails_closed_this_round(self):
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        ledger.record_task_generator_call("w01")
        with pytest.raises(E1SchemaError) as excinfo:
            ledger.reconcile()
        assert excinfo.value.code == A.LLM_ACCOUNTING_MISMATCH
        assert "T1" in str(excinfo.value)
        # explicit opt-out exists (counter slot is real, not deleted)
        counts = ledger.reconcile(expect_no_task_generator=False)
        assert counts["T1"] == 1
        assert counts["N1"] == 6 + 1


class TestFailClosedGuards:
    def test_incomplete_window_fails_reconcile(self):
        ledger = A.LLMCallLedger()
        ledger.record_window_open("w01")
        for role in BOARD_ROLE_ORDER[:3]:  # only 3 of 6 roles ran
            ledger.record_board_call("w01", role)
        with pytest.raises(E1SchemaError) as excinfo:
            ledger.reconcile()
        assert excinfo.value.code == A.LLM_ACCOUNTING_MISMATCH
        assert "6*G1" in str(excinfo.value)

    def test_double_window_open_rejected(self):
        ledger = A.LLMCallLedger()
        ledger.record_window_open("w01")
        with pytest.raises(E1SchemaError) as excinfo:
            ledger.record_window_open("w01")
        assert excinfo.value.code == A.LLM_ACCOUNTING_MISMATCH

    def test_board_call_for_unopened_window_rejected(self):
        ledger = A.LLMCallLedger()
        with pytest.raises(E1SchemaError) as excinfo:
            ledger.record_board_call("w-none", BOARD_ROLE_ORDER[0])
        assert excinfo.value.code == A.LLM_ACCOUNTING_MISMATCH

    def test_duplicate_board_role_in_window_rejected(self):
        ledger = A.LLMCallLedger()
        ledger.record_window_open("w01")
        ledger.record_board_call("w01", BOARD_ROLE_ORDER[0])
        with pytest.raises(E1SchemaError) as excinfo:
            ledger.record_board_call("w01", BOARD_ROLE_ORDER[0])
        assert excinfo.value.code == A.LLM_ACCOUNTING_MISMATCH

    def test_unknown_board_role_rejected(self):
        ledger = A.LLMCallLedger()
        ledger.record_window_open("w01")
        with pytest.raises(E1SchemaError) as excinfo:
            ledger.record_board_call("w01", "envcoder")
        assert excinfo.value.code == A.LLM_ACCOUNTING_MISMATCH

    def test_empty_identifiers_rejected(self):
        ledger = A.LLMCallLedger()
        with pytest.raises(ValueError):
            ledger.record_window_open("")
        ledger.record_window_open("w01")
        with pytest.raises(ValueError):
            ledger.record_envcoder_call("w01", "")


class TestAuditRecords:
    def test_records_are_jsonl_ready_and_ordered(self):
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        ledger.record_envcoder_call("w01", "art-1")
        ledger.record_repair_call("w01", "art-1")
        records = ledger.to_records()
        assert [r["seq"] for r in records] == list(range(len(records)))
        assert len(records) == 8  # 6 board + 1 envcoder + 1 repair
        for r in records:
            assert set(r) == {"seq", "kind", "role", "window_id", "artifact_id"}
        assert records[0]["kind"] == A.KIND_BOARD
        assert records[0]["role"] == BOARD_ROLE_ORDER[0]
        assert records[0]["artifact_id"] == ""
        assert records[6]["kind"] == A.KIND_ENVCODER
        assert records[6]["artifact_id"] == "art-1"
        assert records[7]["kind"] == A.KIND_REPAIR

    def test_counts_identity_always_holds(self):
        ledger = A.LLMCallLedger()
        _open_and_fill(ledger, "w01")
        ledger.record_envcoder_call("w01", "a")
        ledger.record_repair_call("w01", "a")
        c = ledger.counts()
        assert c["N1"] == 6 * c["G1"] + c["T1"] + c["K1"] + c["F1"]


class TestNoFixedPerWindowPhrasing:
    """G5 language audit: never claim a fixed per-window call total."""

    FORBIDDEN = ("每窗 7 次", "每窗7次", "第 7 次", "第7次",
                 "7 calls per window", "seventh call", "7th call")

    @pytest.mark.parametrize(
        "module", [A, LC, M], ids=["accounting", "llm_client", "manifest"]
    )
    def test_module_source_has_no_fixed_seventh_call_language(self, module):
        text = Path(module.__file__).read_text(encoding="utf-8")
        for phrase in self.FORBIDDEN:
            assert phrase not in text, (
                f"{module.__name__} contains forbidden fixed-count "
                f"phrasing {phrase!r}"
            )

    def test_manifest_documents_window_vs_artifact_units(self):
        manifest = M.build_role_manifest()
        M.assert_manifest_well_formed(manifest)
        window_roles = [
            r for r, e in manifest.items()
            if e["invocation_unit"] == M.INVOCATION_UNIT_WINDOW
        ]
        artifact_roles = [
            r for r, e in manifest.items()
            if e["invocation_unit"] == M.INVOCATION_UNIT_ARTIFACT
        ]
        assert sorted(window_roles) == sorted(BOARD_ROLE_ORDER)
        assert artifact_roles == [M.ENVCODER_ROLE]


class TestManifestAudit:
    def test_well_formed_manifest_passes(self):
        M.assert_manifest_well_formed(M.build_role_manifest())

    def test_bad_type_rejected(self):
        with pytest.raises(E1SchemaError) as excinfo:
            M.assert_manifest_well_formed(["not", "a", "dict"])
        assert excinfo.value.code == "ROLE_MANIFEST_BAD_TYPE"

    def test_role_set_mismatch_rejected(self):
        manifest = M.build_role_manifest()
        del manifest["critic"]
        with pytest.raises(E1SchemaError) as excinfo:
            M.assert_manifest_well_formed(manifest)
        assert excinfo.value.code == "ROLE_MANIFEST_ROLE_SET_MISMATCH"

    def test_missing_field_rejected(self):
        manifest = M.build_role_manifest()
        manifest["critic"]["provider"] = ""
        with pytest.raises(E1SchemaError) as excinfo:
            M.assert_manifest_well_formed(manifest)
        assert excinfo.value.code == "ROLE_MANIFEST_MISSING_FIELD"

    def test_unknown_unit_rejected(self):
        manifest = M.build_role_manifest()
        manifest["critic"]["invocation_unit"] = "session"
        with pytest.raises(E1SchemaError) as excinfo:
            M.assert_manifest_well_formed(manifest)
        assert excinfo.value.code == "ROLE_MANIFEST_BAD_UNIT"

    def test_swapped_unit_rejected(self):
        # critic is a window role; demoting it to artifact-unit breaks
        # the window-role set check.
        manifest = M.build_role_manifest()
        manifest["critic"]["invocation_unit"] = M.INVOCATION_UNIT_ARTIFACT
        with pytest.raises(E1SchemaError) as excinfo:
            M.assert_manifest_well_formed(manifest)
        assert excinfo.value.code == "ROLE_MANIFEST_ROLE_SET_MISMATCH"
