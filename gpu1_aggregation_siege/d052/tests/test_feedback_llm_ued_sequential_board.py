"""P0-1 (CC3 follow-up audit): sequential six-role context chaining.

The board is no longer six independent calls over the same base context —
the chain is

    base                              -> StudentModeler
    base + SM                         -> BehaviorAuditor
    base + SM + BA                    -> CausalFailureAnalyst
    base + first three                -> InterventionTutor
    base + first four                 -> Explorer
    base + all five                   -> Critic/Skeptic

and every upstream contribution is STRUCTURED (role name + parsed output +
canonical output hash embedded in the prompt-context JSON, plus the same
hashes in the envelope's ``context_binding``) — never natural-language
concatenation. The four audit-mandated negative tests:

* changing the StudentModeler output changes the LAST FIVE prompt hashes;
* changing the BehaviorAuditor output leaves the first TWO prompts
  unchanged but changes the last four;
* the critic missing ANY of its five upstream outputs fails closed;
* replay corpus keys are sensitive to the sequential context.

FIXTURES ARE TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: every board below
runs on the deterministic mock backend (or a TEST_ONLY patching wrapper over
it) — no real LLM is called and no REAL_* flag is flipped by any test here.
"""
import json

import pytest

from d052.bagr_ued.hashing import canonical_sha256
from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.behavior_failure import assemble_board_context
from d052.feedback_llm_ued.feedback_contracts import extract_context
from d052.feedback_llm_ued.feedback_view import NormalFeedbackView
from d052.feedback_llm_ued.hypothesis_ledger import HypothesisRecord
from d052.feedback_llm_ued.llm_backend import (
    DeterministicMockFeedbackBackend,
    RecordingBackend,
    ReplayBackend,
)
from d052.feedback_llm_ued.review_board import (
    SEQUENTIAL_PROMPT_SCHEMA_VERSION,
    SequentialChainBroken,
    build_board_prompt_context,
    make_upstream_entry,
    normalize_hypothesis_inputs,
    run_review_board,
    sequential_role_context,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.synthetic_feedback import (
    synthetic_candidate,
    synthetic_feedback_record,
)

FAM_A = "threat_distance_family"
FAM_B = "resource_pressure_family"

#: TEST_ONLY / SYNTHETIC identity hashes — never derived from real assets
STUDENT_IDENTITY_HASH = "student-identity-hash-test-only"
REFERENCE_IDENTITY_HASH = "reference-identity-hash-test-only"
PREVIOUS_PLAN_HASH = "previous-plan-hash-test-only"


def make_record(i, *, window=0, family=FAM_A, student_sr=0.4,
                distinguishes=None, feedback_id=None):
    cand = synthetic_candidate(candidate_id=f"c-sb-{window}-{i}",
                               family=family)
    return synthetic_feedback_record(
        feedback_id=feedback_id or f"fb-sb-w{window}-{i}",
        candidate=cand, plan_id=f"plan-{window}", window=window,
        student_success_rate=student_sr,
        expected_signature={"student_success_rate": 0.5},
        distinguishes_hypothesis_ids=list(distinguishes or []))


def make_hyp(hid="hyp-1", family=FAM_A):
    return HypothesisRecord(
        hypothesis_id=hid, source_window=0,
        target_behavior=f"behavior under {family}",
        predicted_signature={"student_success_rate": 0.5},
        environment_family=family, confidence=0.5)


def _case():
    """One graded opposite + one agreeing record in evidence window 0 and
    the view/context a window-1 board consumes (CC3 C9 exact lag)."""
    store = SimulatorFeedbackStore()
    store.add(make_record(0, feedback_id="fb-sb-w0-0",
                          distinguishes=["hyp-1"]))
    store.add(make_record(1, family=FAM_B, student_sr=0.6,
                          feedback_id="fb-sb-w0-1"))
    store.bind_match("fb-sb-w0-0", direction="opposite")
    store.bind_match("fb-sb-w0-1", direction="agree")
    view = NormalFeedbackView.from_store(store, evidence_window=0)
    ctx = assemble_board_context(view, window=0,
                                 mode=C.MODE_NORMAL_FEEDBACK)
    return ctx, view


def _run(backend=None, *, hypotheses=None, sequence_start=0, **kwargs):
    ctx, view = _case()
    backend = backend or DeterministicMockFeedbackBackend()
    out = run_review_board(
        window=1, mode=C.MODE_NORMAL_FEEDBACK, board_context=ctx,
        view=view,
        hypotheses=(hypotheses if hypotheses is not None
                    else [make_hyp()]),
        backend=backend, sequence_start=sequence_start, **kwargs)
    return out, backend, ctx, view


class PatchedMockBackend:
    """TEST_ONLY wrapper over the deterministic mock: serves ONE role's
    completion from a schema-legal variant of the mock output (one free
    string field replaced); every other role delegates. kind stays 'mock',
    usage is the inner backend's own. NOT_REAL_EXECUTION."""

    kind = C.BACKEND_KIND_MOCK
    backend_id = "test.patched.feedback_llm_ued.v1"
    model_id = "patched-rule-model.v1"

    def __init__(self, patch_role: str, patch_field: str,
                 patch_value: str) -> None:
        self._inner = DeterministicMockFeedbackBackend()
        self._patch_role = patch_role
        self._patch_field = patch_field
        self._patch_value = patch_value

    @property
    def usage(self):
        return self._inner.usage

    def complete(self, role, prompt):
        raw = self._inner.complete(role, prompt)
        if role == self._patch_role:
            dump = json.loads(raw)
            dump[self._patch_field] = self._patch_value
            return json.dumps(dump, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False)
        return raw


def _prompt_hashes(out):
    return [e.prompt_sha256 for e in out.envelopes]


class TestSequentialChainStructure:
    def test_each_prompt_embeds_exactly_its_upstream_roles(self):
        out, _, _, _ = _run()
        for i, env in enumerate(out.envelopes):
            ctx = extract_context(env.prompt)
            assert ctx["prompt_schema_version"] == \
                SEQUENTIAL_PROMPT_SCHEMA_VERSION
            assert ctx["upstream_roles"] == list(C.BOARD_ROLES[:i])
            outputs = ctx["upstream_outputs"]
            assert len(outputs) == i
            for entry, upstream_role in zip(outputs, C.BOARD_ROLES[:i]):
                assert set(entry) == {"role", "output", "output_hash"}
                assert entry["role"] == upstream_role
                assert entry["output_hash"] == canonical_sha256(
                    entry["output"])

    def test_critic_truly_reads_all_five_upstream_outputs(self):
        out, _, _, _ = _run()
        critic_ctx = extract_context(out.envelopes[-1].prompt)
        assert critic_ctx["upstream_roles"] == list(C.BOARD_ROLES[:5])
        for j in range(5):
            entry = critic_ctx["upstream_outputs"][j]
            #: the hash in the prompt equals the canonical hash of the
            #: upstream envelope's parsed output — one tamper flips both
            assert entry["output_hash"] == canonical_sha256(
                out.envelopes[j].parsed_json)
            assert entry["output"] == out.envelopes[j].parsed_json

    def test_upstream_output_hashes_chain_across_envelopes(self):
        out, _, _, _ = _run()
        for i, env in enumerate(out.envelopes):
            hashes = env.context_binding["upstream_output_hashes"]
            assert hashes == {
                C.BOARD_ROLES[j]: canonical_sha256(
                    out.envelopes[j].parsed_json) for j in range(i)}


class TestContextBindingFields:
    def test_binding_carries_all_canonical_hashes(self):
        out, _, ctx, view = _run(
            student_identity_hash=STUDENT_IDENTITY_HASH,
            reference_identity_hash=REFERENCE_IDENTITY_HASH,
            previous_plan_hash=PREVIOUS_PLAN_HASH)
        for i, env in enumerate(out.envelopes):
            b = env.context_binding
            assert b["window"] == 1
            assert b["feedback_view_hash"] == canonical_sha256(
                view.to_prompt_payload())
            assert b["behavior_evidence_hash"] == canonical_sha256(
                [item.model_dump() for item in ctx.behavior_evidence])
            assert b["student_identity_hash"] == STUDENT_IDENTITY_HASH
            assert b["reference_identity_hash"] == REFERENCE_IDENTITY_HASH
            assert b["previous_plan_hash"] == PREVIOUS_PLAN_HASH
            assert b["prompt_schema_version"] == \
                SEQUENTIAL_PROMPT_SCHEMA_VERSION
            assert b["backend_id"] == C.MOCK_BACKEND_ID
            assert b["model_id"] == C.MOCK_MODEL_ID
            assert b["sequence"] == env.sequence == i
            assert b["upstream_roles"] == list(C.BOARD_ROLES[:i])

    def test_identity_hashes_default_to_unbound_empty(self):
        out, _, _, _ = _run()
        for env in out.envelopes:
            b = env.context_binding
            assert b["student_identity_hash"] == ""
            assert b["reference_identity_hash"] == ""
            assert b["previous_plan_hash"] == ""

    def test_hypothesis_ledger_hash_is_content_bound(self):
        out_a, _, _, _ = _run()
        out_b, _, _, _ = _run(hypotheses=[make_hyp("hyp-other")])
        a = out_a.envelopes[0].context_binding["hypothesis_ledger_hash"]
        b = out_b.envelopes[0].context_binding["hypothesis_ledger_hash"]
        assert a != b            # a different ledger -> a different hash


class TestAuditMandatedPropagation:
    def test_student_modeler_patch_changes_last_five_prompt_hashes(self):
        """Audit test 1: changing the StudentModeler output must change the
        prompts (and therefore prompt hashes) of ALL five downstream roles,
        while the StudentModeler's own prompt stays byte-identical."""
        baseline, _, _, _ = _run()
        patched, _, _, _ = _run(backend=PatchedMockBackend(
            C.ROLE_STUDENT_MODELER, "summary", "patched-student-model"))
        assert patched.board_call_count == 6
        base_h, patch_h = _prompt_hashes(baseline), _prompt_hashes(patched)
        assert patch_h[0] == base_h[0]          # SM reads no upstream
        for i in range(1, 6):
            assert patch_h[i] != base_h[i], (
                f"role {C.BOARD_ROLES[i]} prompt did not change when the "
                f"StudentModeler output changed")

    def test_behavior_auditor_patch_leaves_first_two_changes_last_four(
            self):
        """Audit test 2: changing the BehaviorAuditor output must NOT touch
        the StudentModeler or BehaviorAuditor prompts (neither reads BA),
        but must change the last four prompts."""
        baseline, _, _, _ = _run()
        patched, _, _, _ = _run(backend=PatchedMockBackend(
            C.ROLE_BEHAVIOR_AUDITOR, "audit_summary", "patched-audit"))
        assert patched.board_call_count == 6
        base_h, patch_h = _prompt_hashes(baseline), _prompt_hashes(patched)
        assert patch_h[0] == base_h[0]
        assert patch_h[1] == base_h[1]          # BA reads only SM
        for i in range(2, 6):
            assert patch_h[i] != base_h[i], (
                f"role {C.BOARD_ROLES[i]} prompt did not change when the "
                f"BehaviorAuditor output changed")


class TestChainFailClosed:
    """Audit test 3 — the critic missing ANY upstream output fails closed
    (and the rest of the chain is just as strict). Direct seam tests of
    ``sequential_role_context`` / ``make_upstream_entry``."""

    def _base_and_entries(self):
        out, _, ctx, view = _run()
        base = build_board_prompt_context(
            window=1, mode=C.MODE_NORMAL_FEEDBACK, board_context=ctx,
            view=view, hypotheses=normalize_hypothesis_inputs([make_hyp()]))
        entries = [make_upstream_entry(role, env.parsed_json)
                   for role, env in zip(C.BOARD_ROLES, out.envelopes)]
        return base, entries[:5]                # the critic's five upstreams

    def test_critic_with_all_five_upstreams_builds(self):
        base, entries = self._base_and_entries()
        ctx = sequential_role_context(
            base, role=C.ROLE_CRITIC_SKEPTIC, upstream_entries=entries)
        assert ctx["upstream_roles"] == list(C.BOARD_ROLES[:5])
        assert len(ctx["upstream_outputs"]) == 5

    def test_critic_missing_any_single_upstream_fails_closed(self):
        base, entries = self._base_and_entries()
        for j in range(5):
            missing = entries[:j] + entries[j + 1:]
            with pytest.raises(SequentialChainBroken,
                               match="UPSTREAM_CHAIN_MISMATCH"):
                sequential_role_context(
                    base, role=C.ROLE_CRITIC_SKEPTIC,
                    upstream_entries=missing)

    def test_wrong_upstream_order_fails_closed(self):
        base, entries = self._base_and_entries()
        swapped = [entries[1], entries[0]] + entries[2:]
        with pytest.raises(SequentialChainBroken,
                           match="UPSTREAM_ROLE_MISMATCH"):
            sequential_role_context(
                base, role=C.ROLE_CRITIC_SKEPTIC, upstream_entries=swapped)

    def test_empty_upstream_output_fails_closed(self):
        base, entries = self._base_and_entries()
        hollow = list(entries)
        hollow[2] = dict(hollow[2], output={}, output_hash="")
        with pytest.raises(SequentialChainBroken,
                           match="UPSTREAM_OUTPUT_MISSING"):
            sequential_role_context(
                base, role=C.ROLE_CRITIC_SKEPTIC, upstream_entries=hollow)

    def test_unknown_role_fails_closed(self):
        base, entries = self._base_and_entries()
        with pytest.raises(ValueError, match="UNKNOWN_BOARD_ROLE"):
            sequential_role_context(
                base, role="diagnostician", upstream_entries=[])
        with pytest.raises(ValueError, match="UNKNOWN_BOARD_ROLE"):
            make_upstream_entry("diagnostician", {"window": 1})

    def test_make_upstream_entry_rejects_empty_output(self):
        with pytest.raises(SequentialChainBroken,
                           match="UPSTREAM_OUTPUT_MISSING"):
            make_upstream_entry(C.ROLE_STUDENT_MODELER, {})


class TestReplayCorpusSensitivity:
    def test_corpus_replays_cleanly_when_context_unchanged(self):
        recorder = RecordingBackend(DeterministicMockFeedbackBackend())
        out, _, _, _ = _run(backend=recorder)
        replay = ReplayBackend(recorder.to_replay_corpus())
        again, backend, _, _ = _run(backend=replay)
        assert backend.usage.replay_calls == 6
        assert _prompt_hashes(again) == _prompt_hashes(out)

    def test_corpus_keys_sensitive_to_sequential_context(self):
        """Audit test 4: a corpus recorded under one sequential context
        cannot serve a different one. Tamper ONLY the StudentModeler's
        recorded response (schema-legal variant): the SM prompt is
        unchanged so it replays, but every downstream prompt now embeds the
        changed upstream output — the BehaviorAuditor's (role, prompt-hash)
        key no longer exists and replay fails CLOSED."""
        recorder = RecordingBackend(DeterministicMockFeedbackBackend())
        out, _, _, _ = _run(backend=recorder)
        corpus = dict(recorder.to_replay_corpus())
        sm_key = (C.ROLE_STUDENT_MODELER, out.envelopes[0].prompt_sha256)
        tampered = json.loads(corpus[sm_key])
        tampered["summary"] = "replay-tampered-student-model"
        corpus[sm_key] = json.dumps(tampered, sort_keys=True)
        replay = ReplayBackend(corpus)
        with pytest.raises(KeyError, match="REPLAY_MISS"):
            _run(backend=replay)
        #: the SM call itself replayed before the downstream miss
        assert replay.usage.replay_calls == 1


class TestDeterminismAndSequence:
    def test_sequential_board_is_deterministic(self):
        out_a, _, _, _ = _run()
        out_b, _, _, _ = _run()
        assert _prompt_hashes(out_a) == _prompt_hashes(out_b)
        assert [e.context_binding for e in out_a.envelopes] == \
            [e.context_binding for e in out_b.envelopes]

    def test_sequence_start_offsets_propagate_into_binding(self):
        out, _, _, _ = _run(sequence_start=10)
        assert [e.sequence for e in out.envelopes] == [10, 11, 12, 13, 14,
                                                       15]
        for env in out.envelopes:
            assert env.context_binding["sequence"] == env.sequence
