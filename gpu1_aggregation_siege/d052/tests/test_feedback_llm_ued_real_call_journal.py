"""P0-5 (§19 mandatory): the complete real-call journal contract.

Every real six-role call must write a TRANSPORT entry plus a
PARSED / SCHEMA_FAILED schema-outcome entry; PARSED closes the
logical_call_id (duplicate success refused); a repair re-call carries a
DIFFERENT prompt and therefore a NEW logical id while the original failed
entry is kept; at the end of the two-window run the COMPLETE journal (full
entries, chain head, retry cap, counts, token totals, file hash) persists
atomically and verifies fail-closed.

Every fixture in this file is TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION:
the "real" transport is a scripted closure returning deterministic
mock-rule text shaped like a provider response — NO real LLM is ever
called, NO credential exists anywhere in this file, and NO passing test
here may flip any REAL_* capability flag (asserted at the bottom).
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from d052.bagr_ued.hashing import canonical_sha256, text_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued import (
    behavior_auditor,
    causal_failure_analyst,
    critic_skeptic,
    explorer,
    intervention_tutor,
    student_modeler,
)
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.llm_backend import (
    BackendCallFailed,
    DeterministicMockFeedbackBackend,
    RealBackendAdapter,
)
from d052.feedback_llm_ued.real_call_journal import (
    ENTRY_KIND_SCHEMA_OUTCOME,
    ENTRY_KIND_TRANSPORT,
    GENESIS_ENTRY_HASH,
    JOURNAL_PERSIST_VERSION,
    OUTPUT_SCHEMA_FAILED,
    OUTPUT_SCHEMA_PARSED,
    OUTPUT_SCHEMA_PENDING,
    TOKEN_USAGE_NOT_PROVIDED,
    TOKEN_USAGE_PROVIDED,
    DuplicateSuccessfulCall,
    JournalBlocked,
    RealCallJournal,
    RealCallJournalEntry,
    RealTransportResult,
    default_logical_call_id,
    journal_counts,
    journal_role_schema_outcome,
    journal_token_totals,
    normalize_transport_result,
    persist_real_call_journal,
    verify_persisted_journal,
)

#: the six board roles in sequential-board order
ROLE_SEQUENCE = (
    student_modeler,
    behavior_auditor,
    causal_failure_analyst,
    intervention_tutor,
    explorer,
    critic_skeptic,
)
ROLE_MODULES = {module.ROLE: module for module in ROLE_SEQUENCE}

TEST_BACKEND_ID = "TEST_ONLY_BACKEND"
TEST_MODEL_ID = "TEST_ONLY_MODEL"

#: minimal board context every mock rule tolerates (TEST_ONLY)
BASE_CONTEXT = dict(window=0, feedback=[], hypotheses=[], board_context={})


def valid_scripted_transport(calls=None):
    """TEST_ONLY / SYNTHETIC transport: deterministic mock-rule text shaped
    like a real provider response (request id + token usage included)."""
    def transport(role, prompt):
        if calls is not None:
            calls.append((role, text_sha256(prompt)))
        context = extract_context(prompt)
        module = ROLE_MODULES[role]
        raw = json.dumps(module.mock_rule(context), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False,
                         default=str)
        return RealTransportResult(
            raw=raw, request_id=f"test-req-{role}",
            token_usage={"prompt_tokens": 11, "completion_tokens": 7})
    return transport


def make_backend(journal, transport=None):
    return RealBackendAdapter(
        transport if transport is not None else valid_scripted_transport(),
        backend_id=TEST_BACKEND_ID, model_id=TEST_MODEL_ID,
        authorized=True, journal=journal)


def populated_journal(*, with_usage=True) -> RealCallJournal:
    """Two completed logical calls (transport + PARSED outcome each)."""
    journal = RealCallJournal()
    for index in range(2):
        logical_call_id = f"logical-{index}"
        usage = ({"prompt_tokens": 10 + index, "completion_tokens": 5}
                 if with_usage else None)
        journal.record_transport(
            logical_call_id=logical_call_id, role="TEST_ROLE",
            backend_id=TEST_BACKEND_ID, model_id=TEST_MODEL_ID,
            request_id=f"req-{index}",
            prompt_sha256=text_sha256(f"prompt-{index}"),
            response_sha256=text_sha256(f"response-{index}"),
            token_usage=usage,
            token_usage_status=(TOKEN_USAGE_PROVIDED if with_usage
                                else TOKEN_USAGE_NOT_PROVIDED),
            retry_count=0)
        journal.record_schema_outcome(logical_call_id,
                                       status=OUTPUT_SCHEMA_PARSED,
                                       window=index, sequence=index)
    return journal


def persist_and_load(journal, tmp_path):
    path = str(tmp_path / "real_call_journal.json")
    file_hash = persist_real_call_journal(journal, path)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    return path, file_hash, document


def restamp(document: dict) -> dict:
    """Re-stamp the file hash after a tamper so deeper checks are reached."""
    body = {k: v for k, v in document.items() if k != "journal_file_hash"}
    document["journal_file_hash"] = canonical_sha256(body)
    return document


# ---------------------------------------------------------------------------
# transport + outcome flow
# ---------------------------------------------------------------------------
class TestTransportAndOutcomeFlow:
    def test_chain_links_from_genesis_to_head(self):
        journal = populated_journal()
        previous = GENESIS_ENTRY_HASH
        for entry in journal.entries:
            assert entry.previous_entry_hash == previous
            previous = entry.entry_hash
        assert journal.chain_head == previous

    def test_parsed_closes_logical_call_id(self):
        journal = populated_journal()
        assert journal.is_completed("logical-0")
        assert journal.is_completed("logical-1")
        assert journal.completed_logical_call_ids == ("logical-0",
                                                      "logical-1")

    def test_duplicate_entries_under_parsed_id_refused(self):
        journal = populated_journal()
        with pytest.raises(DuplicateSuccessfulCall,
                           match="DUPLICATE_SUCCESSFUL_CALL"):
            journal.record_transport(
                logical_call_id="logical-0", role="TEST_ROLE",
                backend_id=TEST_BACKEND_ID, model_id=TEST_MODEL_ID,
                request_id="req-dup",
                prompt_sha256=text_sha256("p"),
                response_sha256=text_sha256("r"))
        with pytest.raises(DuplicateSuccessfulCall,
                           match="DUPLICATE_SUCCESSFUL_CALL"):
            journal.record_schema_outcome("logical-0",
                                          status=OUTPUT_SCHEMA_PARSED)

    def test_schema_failed_leaves_id_open(self):
        journal = RealCallJournal()
        journal.record_transport(
            logical_call_id="logical-a", role="TEST_ROLE",
            backend_id=TEST_BACKEND_ID, model_id=TEST_MODEL_ID,
            request_id="req-a", prompt_sha256=text_sha256("p"),
            response_sha256=text_sha256("r"))
        journal.record_schema_outcome("logical-a",
                                      status=OUTPUT_SCHEMA_FAILED)
        assert not journal.is_completed("logical-a")
        assert journal.completed_logical_call_ids == ()

    def test_outcome_without_transport_refused(self):
        journal = RealCallJournal()
        with pytest.raises(JournalBlocked,
                           match="JOURNAL_NO_TRANSPORT_FOR_OUTCOME"):
            journal.record_schema_outcome("never-seen",
                                          status=OUTPUT_SCHEMA_PARSED)

    def test_illegal_outcome_status_refused(self):
        journal = populated_journal()
        with pytest.raises(ValueError,
                           match="ILLEGAL_SCHEMA_OUTCOME_STATUS"):
            journal.record_schema_outcome("logical-0",
                                          status=OUTPUT_SCHEMA_PENDING)

    def test_retry_cap_exceeded_refused(self):
        journal = RealCallJournal(retry_cap=1)
        with pytest.raises(JournalBlocked, match="JOURNAL_RETRY_CAP_EXCEEDED"):
            journal.record_transport(
                logical_call_id="logical-x", role="TEST_ROLE",
                backend_id=TEST_BACKEND_ID, model_id=TEST_MODEL_ID,
                request_id="req-x", prompt_sha256=text_sha256("p"),
                response_sha256=text_sha256("r"), retry_count=2)

    def test_negative_retry_cap_refused(self):
        with pytest.raises(ValueError, match="NEGATIVE_JOURNAL_RETRY_CAP"):
            RealCallJournal(retry_cap=-1)

    def test_counts_exact(self):
        counts = journal_counts(populated_journal())
        assert counts == dict(total_entries=4, transport_entries=2,
                              schema_outcome_entries=2, parsed_outcomes=2,
                              schema_failed_outcomes=0,
                              completed_logical_calls=2)

    def test_token_totals_count_transport_entries_only(self):
        journal = populated_journal()
        #: outcome entries repeat their transport's usage — totals must NOT
        #: double count them
        assert journal_token_totals(journal) == {"prompt_tokens": 21,
                                                 "completion_tokens": 10}


# ---------------------------------------------------------------------------
# adapter-level journaling (fake-real; scripted transport, TEST_ONLY)
# ---------------------------------------------------------------------------
class TestAdapterJournaling:
    def test_success_journals_one_transport_entry(self):
        journal = RealCallJournal()
        backend = make_backend(journal)
        prompt = student_modeler.build_prompt(BASE_CONTEXT)
        backend.complete(student_modeler.ROLE, prompt)
        assert len(journal.entries) == 1
        entry = journal.entries[0]
        assert entry.entry_kind == ENTRY_KIND_TRANSPORT
        assert entry.role == student_modeler.ROLE
        assert entry.backend_id == TEST_BACKEND_ID
        assert entry.model_id == TEST_MODEL_ID
        assert entry.prompt_sha256 == text_sha256(prompt)
        assert entry.retry_count == 0
        assert entry.output_schema_status == OUTPUT_SCHEMA_PENDING

    def test_duplicate_success_refused_before_spending_transport(self):
        journal = RealCallJournal()
        calls = []
        backend = make_backend(journal, valid_scripted_transport(calls))
        prompt = student_modeler.build_prompt(BASE_CONTEXT)
        backend.complete(student_modeler.ROLE, prompt)
        backend.record_schema_outcome(student_modeler.ROLE, prompt,
                                      status=OUTPUT_SCHEMA_PARSED)
        with pytest.raises(DuplicateSuccessfulCall,
                           match="DUPLICATE_SUCCESSFUL_CALL"):
            backend.complete(student_modeler.ROLE, prompt)
        assert len(calls) == 1          # no transport attempt was spent
        assert backend.usage.real_calls == 1

    def test_retry_then_success_records_retry_count(self):
        journal = RealCallJournal()
        attempts = {"n": 0}

        def flaky(role, prompt):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("TEST_ONLY_TRANSIENT_FAILURE")
            return "RAW_OK"

        backend = make_backend(journal, flaky)
        raw = backend.complete("TEST_ROLE", "prompt-x")
        assert raw == "RAW_OK"
        assert backend.usage.failed_calls == 2
        entry = journal.entries[0]
        assert entry.retry_count == 2
        #: a bare str transport carries no provider id / usage: labelled,
        #: never fabricated
        assert entry.request_id.startswith("derived:")
        assert entry.token_usage_status == TOKEN_USAGE_NOT_PROVIDED

    def test_exhausted_retries_journal_nothing(self):
        journal = RealCallJournal()

        def always_fails(role, prompt):
            raise ConnectionError("TEST_ONLY_PERMANENT_FAILURE")

        backend = make_backend(journal, always_fails)
        with pytest.raises(BackendCallFailed, match="REAL_LLM_CALL_FAILED"):
            backend.complete("TEST_ROLE", "prompt-x")
        assert journal.entries == ()    # no served call -> nothing journal
        assert backend.usage.failed_calls == 3
        assert backend.usage.real_calls == 0

    def test_max_retries_above_journal_cap_refused(self):
        with pytest.raises(ValueError, match="RETRY_CAP_MISMATCH"):
            RealBackendAdapter(
                valid_scripted_transport(), backend_id=TEST_BACKEND_ID,
                model_id=TEST_MODEL_ID, authorized=True, max_retries=3,
                journal=RealCallJournal(retry_cap=2))

    def test_record_schema_outcome_without_journal_is_noop(self):
        backend = RealBackendAdapter(
            valid_scripted_transport(), backend_id=TEST_BACKEND_ID,
            model_id=TEST_MODEL_ID, authorized=True, journal=None)
        assert backend.journal is None
        assert backend.record_schema_outcome(
            "TEST_ROLE", "prompt", status=OUTPUT_SCHEMA_PARSED) is None


# ---------------------------------------------------------------------------
# repair re-calls: NEW logical id, original failed entry kept
# ---------------------------------------------------------------------------
class TestRepairIdentity:
    def test_repair_gets_new_logical_id_original_failed_entry_kept(self):
        journal = RealCallJournal()
        good_raw = json.dumps(student_modeler.mock_rule(BASE_CONTEXT),
                              sort_keys=True, default=str)
        responses = iter([RealTransportResult(raw="NOT-JSON",
                                              request_id="req-bad"),
                          RealTransportResult(raw=good_raw,
                                              request_id="req-repair")])
        backend = make_backend(journal, lambda role, prompt: next(responses))

        prompt_a = student_modeler.build_prompt(BASE_CONTEXT)
        backend.complete(student_modeler.ROLE, prompt_a)
        backend.record_schema_outcome(student_modeler.ROLE, prompt_a,
                                      status=OUTPUT_SCHEMA_FAILED)
        prompt_b = prompt_a + "\nREPAIR: previous output failed schema check"
        backend.complete(student_modeler.ROLE, prompt_b)
        backend.record_schema_outcome(student_modeler.ROLE, prompt_b,
                                      status=OUTPUT_SCHEMA_PARSED)

        id_a = default_logical_call_id(student_modeler.ROLE, prompt_a,
                                       TEST_BACKEND_ID)
        id_b = default_logical_call_id(student_modeler.ROLE, prompt_b,
                                       TEST_BACKEND_ID)
        assert id_a != id_b
        assert not journal.is_completed(id_a)
        assert journal.is_completed(id_b)
        outcomes = [e for e in journal.entries
                    if e.entry_kind == ENTRY_KIND_SCHEMA_OUTCOME]
        assert [e.output_schema_status for e in outcomes] == [
            OUTPUT_SCHEMA_FAILED, OUTPUT_SCHEMA_PARSED]
        assert outcomes[0].logical_call_id == id_a   # failed entry kept
        counts = journal_counts(journal)
        assert counts["transport_entries"] == 2
        assert counts["schema_failed_outcomes"] == 1
        assert counts["parsed_outcomes"] == 1
        assert counts["completed_logical_calls"] == 1


# ---------------------------------------------------------------------------
# transport normalization honesty (labels, never fabrication)
# ---------------------------------------------------------------------------
class TestTransportNormalization:
    def test_bare_str_gets_derived_request_id_and_not_provided(self):
        raw, request_id, usage, status = normalize_transport_result("RAW")
        assert raw == "RAW"
        assert request_id == f"derived:{text_sha256('RAW')[:16]}"
        assert usage == {}
        assert status == TOKEN_USAGE_NOT_PROVIDED

    def test_result_with_usage_labelled_provided(self):
        result = RealTransportResult(raw="R", request_id="api-1",
                                     token_usage={"prompt_tokens": 3})
        raw, request_id, usage, status = normalize_transport_result(result)
        assert (raw, request_id) == ("R", "api-1")
        assert usage == {"prompt_tokens": 3}
        assert status == TOKEN_USAGE_PROVIDED

    def test_result_without_usage_labelled_not_provided(self):
        result = RealTransportResult(raw="R", request_id="api-1",
                                     token_usage=None)
        _raw, _rid, usage, status = normalize_transport_result(result)
        assert usage == {}
        assert status == TOKEN_USAGE_NOT_PROVIDED

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "3"])
    def test_illegal_token_usage_refused(self, bad):
        result = RealTransportResult(raw="R", token_usage={"k": bad})
        with pytest.raises(ValueError, match="ILLEGAL_TOKEN_USAGE"):
            normalize_transport_result(result)

    def test_unknown_result_shape_refused(self):
        with pytest.raises(ValueError,
                           match="REAL_TRANSPORT_RESULT_TYPE_UNKNOWN"):
            normalize_transport_result({"raw": "R"})

    def test_empty_raw_refused(self):
        with pytest.raises(ValueError, match="EMPTY_REAL_LLM_RESPONSE"):
            normalize_transport_result("")


# ---------------------------------------------------------------------------
# entry integrity (recompute, bound copy)
# ---------------------------------------------------------------------------
class TestEntryIntegrity:
    def test_tampered_entry_rebuild_refused(self):
        entry = populated_journal().entries[0]
        dump = entry.model_dump()
        dump["retry_count"] = 99            # tamper, keep carried hash
        with pytest.raises(ValueError, match="CONTENT_HASH_MISMATCH"):
            RealCallJournalEntry(**dump)

    def test_bound_copy_rehashes_and_journal_keeps_unbound(self):
        journal = populated_journal()
        entry = journal.entries[0]
        copy = entry.bound_copy(window=2, sequence=5,
                                artifact_binding="binding-hash")
        assert (copy.window, copy.sequence) == (2, 5)
        assert copy.artifact_binding == "binding-hash"
        assert copy.entry_hash != entry.entry_hash
        assert journal.entries[0].window == -1      # original untouched


# ---------------------------------------------------------------------------
# P0-5 persistence: round-trip + fail-closed tamper ladder
# ---------------------------------------------------------------------------
class TestPersistenceRoundTrip:
    def test_round_trip_verifies(self, tmp_path):
        journal = populated_journal()
        path, file_hash, document = persist_and_load(journal, tmp_path)
        report = verify_persisted_journal(document)
        assert report["verified"] is True
        assert report["journal_file_hash"] == file_hash
        assert report["chain_head"] == journal.chain_head
        assert report["n_entries"] == len(journal.entries)
        assert report["counts"] == journal_counts(journal)
        assert report["token_totals"] == journal_token_totals(journal)
        assert document["journal_version"] == JOURNAL_PERSIST_VERSION
        assert not (tmp_path / "real_call_journal.json.tmp").exists()

    def test_empty_journal_round_trip(self, tmp_path):
        journal = RealCallJournal()
        _path, _file_hash, document = persist_and_load(journal, tmp_path)
        report = verify_persisted_journal(document)
        assert report["verified"] is True
        assert report["chain_head"] == GENESIS_ENTRY_HASH
        assert report["n_entries"] == 0

    def test_document_never_contains_prompt_or_response_text(self, tmp_path):
        marker_context = dict(BASE_CONTEXT,
                              marker="ZQX_PROMPT_MARKER_77")
        journal = RealCallJournal()
        backend = make_backend(journal)
        student_modeler.run(marker_context, backend, window=0, sequence=0)
        path, _file_hash, _document = persist_and_load(journal, tmp_path)
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        assert "ZQX_PROMPT_MARKER_77" not in text   # hashes only, no text


class TestPersistenceTamperLadder:
    def _document(self, tmp_path):
        _path, _hash, document = persist_and_load(populated_journal(),
                                                  tmp_path)
        return document

    @pytest.mark.parametrize("bad_document", [None, [], "text"])
    def test_not_a_document_refused(self, bad_document):
        with pytest.raises(JournalBlocked,
                           match="JOURNAL_PERSIST_NOT_A_DOCUMENT"):
            verify_persisted_journal(bad_document)

    def test_file_hash_mismatch_refused(self, tmp_path):
        document = self._document(tmp_path)
        document["journal_file_hash"] = "0" * 64
        with pytest.raises(JournalBlocked,
                           match="JOURNAL_PERSIST_HASH_MISMATCH"):
            verify_persisted_journal(document)

    def test_missing_entries_refused(self, tmp_path):
        document = self._document(tmp_path)
        document.pop("entries")
        restamp(document)
        with pytest.raises(JournalBlocked,
                           match="JOURNAL_PERSIST_ENTRIES_MISSING"):
            verify_persisted_journal(document)

    def test_tampered_entry_refused(self, tmp_path):
        document = self._document(tmp_path)
        document["entries"][0]["retry_count"] = 99
        restamp(document)
        with pytest.raises(JournalBlocked, match="JOURNAL_ENTRY_TAMPERED"):
            verify_persisted_journal(document)

    def test_broken_chain_refused(self, tmp_path):
        document = self._document(tmp_path)
        document["entries"][0], document["entries"][1] = \
            document["entries"][1], document["entries"][0]
        restamp(document)
        with pytest.raises(JournalBlocked, match="JOURNAL_CHAIN_BROKEN"):
            verify_persisted_journal(document)

    def test_chain_head_mismatch_refused(self, tmp_path):
        document = self._document(tmp_path)
        document["chain_head"] = text_sha256("forged-chain-head")
        restamp(document)
        with pytest.raises(JournalBlocked,
                           match="JOURNAL_CHAIN_HEAD_MISMATCH"):
            verify_persisted_journal(document)

    def test_counts_mismatch_refused(self, tmp_path):
        document = self._document(tmp_path)
        document["counts"]["parsed_outcomes"] += 1
        restamp(document)
        with pytest.raises(JournalBlocked, match="JOURNAL_COUNTS_MISMATCH"):
            verify_persisted_journal(document)

    def test_token_totals_mismatch_refused(self, tmp_path):
        document = self._document(tmp_path)
        document["token_totals"]["prompt_tokens"] += 5
        restamp(document)
        with pytest.raises(JournalBlocked,
                           match="JOURNAL_TOKEN_TOTALS_MISMATCH"):
            verify_persisted_journal(document)


# ---------------------------------------------------------------------------
# six-role run() journaling through the real adapter seam
# ---------------------------------------------------------------------------
class TestSixRoleRunJournaling:
    @pytest.mark.parametrize("module", ROLE_SEQUENCE,
                             ids=lambda m: m.ROLE)
    def test_run_journals_transport_and_parsed(self, module):
        journal = RealCallJournal()
        backend = make_backend(journal)
        envelope = module.run(BASE_CONTEXT, backend, window=3, sequence=17)
        assert envelope.role == module.ROLE
        assert len(journal.entries) == 2
        transport, outcome = journal.entries
        assert transport.entry_kind == ENTRY_KIND_TRANSPORT
        assert outcome.entry_kind == ENTRY_KIND_SCHEMA_OUTCOME
        assert outcome.output_schema_status == OUTPUT_SCHEMA_PARSED
        assert outcome.role == module.ROLE
        assert (outcome.window, outcome.sequence) == (3, 17)
        assert outcome.logical_call_id == transport.logical_call_id
        assert journal.is_completed(transport.logical_call_id)

    @pytest.mark.parametrize("module", ROLE_SEQUENCE,
                             ids=lambda m: m.ROLE)
    def test_run_journals_schema_failed_and_reraises(self, module):
        journal = RealCallJournal()
        bad_transport = lambda role, prompt: RealTransportResult(
            raw="THIS IS NOT JSON", request_id="req-bad")
        backend = make_backend(journal, bad_transport)
        with pytest.raises(ValidationError):
            module.run(BASE_CONTEXT, backend, window=1, sequence=2)
        assert len(journal.entries) == 2
        transport, outcome = journal.entries
        assert transport.entry_kind == ENTRY_KIND_TRANSPORT
        assert outcome.output_schema_status == OUTPUT_SCHEMA_FAILED
        assert (outcome.window, outcome.sequence) == (1, 2)
        #: a failed parse leaves the id OPEN for a bounded repair re-call
        assert not journal.is_completed(transport.logical_call_id)

    def test_one_window_of_six_roles_journals_twelve_entries(self):
        journal = RealCallJournal()
        backend = make_backend(journal)
        for sequence, module in enumerate(ROLE_SEQUENCE):
            module.run(BASE_CONTEXT, backend, window=0, sequence=sequence)
        counts = journal_counts(journal)
        assert counts == dict(total_entries=12, transport_entries=6,
                              schema_outcome_entries=6, parsed_outcomes=6,
                              schema_failed_outcomes=0,
                              completed_logical_calls=6)
        ids = [e.logical_call_id for e in journal.entries
               if e.entry_kind == ENTRY_KIND_TRANSPORT]
        assert len(set(ids)) == 6       # one distinct logical id per role
        assert journal_token_totals(journal) == {"prompt_tokens": 66,
                                                 "completion_tokens": 42}


# ---------------------------------------------------------------------------
# posture: mock never journals; REAL_* flags stay False
# ---------------------------------------------------------------------------
class TestPosture:
    def test_mock_backend_never_journals(self):
        mock = DeterministicMockFeedbackBackend()
        assert not hasattr(mock, "record_schema_outcome")
        assert journal_role_schema_outcome(
            mock, role="TEST_ROLE", prompt="prompt",
            status=OUTPUT_SCHEMA_PARSED) is None

    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
