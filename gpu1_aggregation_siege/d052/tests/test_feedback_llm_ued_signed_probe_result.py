"""P0-8 / P0-9 (§19 seam coverage): the production probe path consumes
ONLY the immutable, registry-signed CandidateProbeResult.

Contract under test:

* signed + immutable: ``result_hash`` is mandatory and recomputed-and-
  compared (unsigned / tampered results fail closed); the frozen model
  refuses post-construction mutation;
* balanced accounting: per role, requested == completed +
  failed_or_rejected; the requested counts must match what the seam
  asked for; zero requested episodes fail closed;
* CI-sample count = actually COMPLETED (valid) episodes only — failed /
  rejected episodes never count;
* checkpoint hashes are mandatory valid sha256;
* identity binding: a result signed by a different issuer / for a
  different stage is refused (identity substitution);
* feedback provenance hardening (P0-8/P0-9): changed axes must be
  declared mutation axes; mandatory checkpoint hashes; non-empty seed
  bank and positive transitions; duplicate stage-1 candidates, missing
  probe evidence and already-stored feedback ids fail closed in the
  entrypoint builder; the forbidden ``except KeyError: continue`` skip
  of unknown hypotheses is gone (fail closed instead).

Fixtures are TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION: scripted
runners sign pre-built results exactly like the shared runtime owner
would, NO real simulator episode runs, and NO passing test flips a
REAL_* flag. Retry/repair exhaustion is scoped out here (this seam has
no retry budget); snapshot/restore does not apply to immutable results
(rebuild determinism is covered instead).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.executable_env_artifact import (
    bind_candidate_to_artifact,
)
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.feedback_contracts import ProbeMetrics
from d052.feedback_llm_ued.real_probe_feedback import (
    FORBIDDEN_PRODUCTION_RUNNER_IDS,
    CandidateProbeResult,
    RealProbeBlocked,
    RealProbeFeedbackRunner,
    build_real_feedback_record,
    consume_signed_probe_result,
    sign_probe_result,
)
from d052.feedback_llm_ued.shared_runtime_binding import (
    ReferenceBindingIdentity,
)
from d052.feedback_llm_ued.simulator_feedback_store import (
    MATCH_UNGRADED,
    SimulatorFeedbackStore,
)
from d052.feedback_llm_ued.student_binding import local_symbolic_binding

from test_feedback_llm_ued_real_envcoder_binding import (
    ADAPTER_ID,
    FAM_T,
    build_chain,
    derive_ok,
    make_candidate,
    record_kwargs,
)

STUDENT_CKPT = "c1" * 32
REFERENCE_CKPT = "d1" * 32

METRICS = dict(
    student_success_rate=0.5, student_behavior_activation=0.5,
    student_front_progress=0.6, reference_success_rate=0.9,
    reference_mean_progress=0.8, reference_behavior_activation=0.7,
    global_retention=1.0, regret=0.4, learnability=0.4,
    simulator_transitions=24, too_hard=False, too_easy=False)


def signed_payload(*, stage="fast", issuer=ADAPTER_ID,
                   student_requested=2, student_completed=2,
                   student_failed=0, reference_requested=1,
                   reference_completed=1, reference_failed=0,
                   student_ckpt=STUDENT_CKPT, reference_ckpt=REFERENCE_CKPT,
                   metrics=None, transitions=24):
    return dict(
        stage=stage, metrics=dict(metrics or METRICS,
                                  simulator_transitions=transitions),
        simulator_transitions=transitions,
        student_episodes_requested=student_requested,
        student_episodes_completed=student_completed,
        student_episodes_failed_or_rejected=student_failed,
        reference_episodes_requested=reference_requested,
        reference_episodes_completed=reference_completed,
        reference_episodes_failed_or_rejected=reference_failed,
        student_checkpoint_hash=student_ckpt,
        reference_checkpoint_hash=reference_ckpt,
        issuer_runner_id=issuer)


def consume(result, *, issuer=ADAPTER_ID, stage="fast", student=2,
            reference=1):
    return consume_signed_probe_result(
        result, expected_issuer=issuer, stage=stage,
        requested_student=student, requested_reference=reference)


class ScriptedSignedRunner:
    """TEST_ONLY / SYNTHETIC shared runner returning a payload the seam
    must consume as a signed CandidateProbeResult (or refuse)."""

    runner_id = ADAPTER_ID
    real_simulator = True

    def __init__(self, factory):
        self._factory = factory

    def probe_candidate(self, *, candidate_hash, environment_family,
                        axis_values, held_constant_axes, stage,
                        student_episodes, reference_episodes, seed_bank):
        return self._factory(stage=stage,
                             student_episodes=student_episodes,
                             reference_episodes=reference_episodes)


def fam_t_artifact():
    return next(a for a in derive_ok(build_chain())
                if a.environment_family == FAM_T)


def make_adapter(monkeypatch, factory):
    monkeypatch.setattr(C, "REAL_SIMULATOR_PROBE_AUTHORIZED", True)
    gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
    adapter = RealProbeFeedbackRunner(
        shared_runner=ScriptedSignedRunner(factory), gate=gate,
        student_identity_hash=local_symbolic_binding().identity_hash)
    adapter.bind_executable_artifacts(derive_ok(build_chain()))
    return adapter


class TestSignedResultContract:
    def test_positive_sign_and_consume(self):
        result = sign_probe_result(signed_payload())
        assert result.stage == "fast"
        assert result.valid_episode_count == 3
        assert consume(result) is result
        #: registry signature is a legal sha256 content hash
        assert len(result.result_hash) == 64

    def test_signature_is_deterministic(self):
        assert (sign_probe_result(signed_payload()).result_hash
                == sign_probe_result(signed_payload()).result_hash)

    def test_frozen_model_refuses_mutation(self):
        result = sign_probe_result(signed_payload())
        with pytest.raises(ValidationError, match="frozen"):
            result.stage = "full"

    def test_unsigned_result_refused(self):
        with pytest.raises(ValidationError,
                           match="REAL_PROBE_RESULT_UNSIGNED"):
            CandidateProbeResult(**signed_payload(), result_hash="")

    def test_tampered_result_refused(self):
        result = sign_probe_result(signed_payload())
        dump = result.model_dump()
        dump["metrics"]["student_success_rate"] = 0.99
        #: keeps the stale signature -> recomputation mismatch
        dump["result_hash"] = result.result_hash
        with pytest.raises(ValidationError, match="CONTENT_HASH_MISMATCH"):
            CandidateProbeResult(**dump)

    def test_unknown_field_refused(self):
        payload = signed_payload()
        payload["episode_count"] = 3      # the retired pre-P0-8 field
        with pytest.raises(ValidationError, match="Extra inputs"):
            sign_probe_result(payload)

    @pytest.mark.parametrize("requested,completed,failed",
                             [(2, 1, 0), (2, 2, 1), (1, 0, 0), (3, 2, 2)])
    def test_student_accounting_mismatch_refused(self, requested,
                                                 completed, failed):
        with pytest.raises(
                ValidationError,
                match="REAL_PROBE_EPISODE_ACCOUNTING_MISMATCH"):
            sign_probe_result(signed_payload(
                student_requested=requested, student_completed=completed,
                student_failed=failed))

    @pytest.mark.parametrize("requested,completed,failed",
                             [(1, 0, 0), (2, 1, 2)])
    def test_reference_accounting_mismatch_refused(self, requested,
                                                   completed, failed):
        with pytest.raises(
                ValidationError,
                match="REAL_PROBE_EPISODE_ACCOUNTING_MISMATCH"):
            sign_probe_result(signed_payload(
                reference_requested=requested,
                reference_completed=completed, reference_failed=failed))

    @pytest.mark.parametrize("role", ["student", "reference"])
    def test_zero_requested_episodes_refused(self, role):
        payload = signed_payload()
        payload[f"{role}_episodes_requested"] = 0
        payload[f"{role}_episodes_completed"] = 0
        payload[f"{role}_episodes_failed_or_rejected"] = 0
        with pytest.raises(ValidationError,
                           match="REAL_PROBE_EPISODES_NOT_REQUESTED"):
            sign_probe_result(payload)

    @pytest.mark.parametrize("field", ["student_checkpoint_hash",
                                       "reference_checkpoint_hash"])
    @pytest.mark.parametrize("bad", ["z" * 64, "AB" * 32, "short"])
    def test_checkpoint_hash_must_be_sha256(self, field, bad):
        payload = signed_payload()
        payload[field] = bad
        with pytest.raises(
                ValidationError,
                match="REAL_PROBE_CHECKPOINT_HASH_NOT_SHA256"):
            sign_probe_result(payload)

    @pytest.mark.parametrize("field", ["student_checkpoint_hash",
                                       "reference_checkpoint_hash"])
    def test_checkpoint_hash_is_mandatory(self, field):
        payload = signed_payload()
        payload[field] = ""
        with pytest.raises(ValidationError):
            sign_probe_result(payload)

    def test_forbidden_issuer_refused(self):
        forbidden = sorted(FORBIDDEN_PRODUCTION_RUNNER_IDS)[0]
        with pytest.raises(ValidationError,
                           match="PRODUCTION_PATH_FORBIDDEN_RUNNER"):
            sign_probe_result(signed_payload(issuer=forbidden))

    def test_mapping_consume_path(self):
        result = sign_probe_result(signed_payload())
        consumed = consume(dict(result.model_dump()))
        assert consumed.result_hash == result.result_hash
        assert consumed.valid_episode_count == 3

    @pytest.mark.parametrize("raw", [
        SimpleNamespace(metrics=dict(METRICS), simulator_transitions=24),
        "a-signed-looking-string", None, 42, [1, 2, 3],
    ], ids=["simplenamespace", "string", "none", "int", "list"])
    def test_duck_typed_results_refused(self, raw):
        #: mock-impersonating-real angle: only the immutable signed model
        #: (or a mapping that satisfies it) is consumed
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_RESULT_NOT_SIGNED"):
            consume(raw)

    def test_wrong_stage_refused(self):
        result = sign_probe_result(signed_payload(stage="fast"))
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_STAGE_MISMATCH"):
            consume(result, stage="full")

    def test_wrong_issuer_refused(self):
        #: identity substitution: content signed by a DIFFERENT runner
        result = sign_probe_result(signed_payload())
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_RESULT_ISSUER_MISMATCH"):
            consume(result, issuer="shared.some_other_runner.v9")

    def test_requested_mismatch_refused(self):
        result = sign_probe_result(signed_payload(student_requested=2))
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_EPISODE_REQUEST_MISMATCH"):
            consume(result, student=5)
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_EPISODE_REQUEST_MISMATCH"):
            consume(result, reference=9)

    def test_zero_valid_episodes_refused(self):
        result = sign_probe_result(signed_payload(
            student_completed=0, student_failed=2,
            reference_completed=0, reference_failed=1))
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_NO_VALID_EPISODES"):
            consume(result)


class TestAdapterConsumesSignedOnly:
    def test_probe_positive_evidence_is_signed(self, monkeypatch):
        def factory(*, stage, student_episodes, reference_episodes):
            return sign_probe_result(signed_payload(
                stage=stage, student_requested=student_episodes,
                student_completed=student_episodes,
                reference_requested=reference_episodes,
                reference_completed=reference_episodes))

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        metrics = adapter.probe(cand, stage="fast", student_episodes=2,
                                reference_episodes=1)
        assert metrics.stage == "fast"
        evidence = adapter.probe_evidence[cand.candidate_id][-1]
        assert evidence["ci_sample_count"] == 3
        assert evidence["student_episodes_completed"] == 2
        assert evidence["reference_episodes_completed"] == 1
        assert evidence["student_episodes_failed_or_rejected"] == 0
        assert evidence["student_checkpoint_hash"] == STUDENT_CKPT
        assert evidence["reference_checkpoint_hash"] == REFERENCE_CKPT
        assert len(evidence["result_hash"]) == 64

    def test_adapter_refuses_duck_typed_result(self, monkeypatch):
        def factory(**kw):
            return SimpleNamespace(metrics=dict(METRICS),
                                   simulator_transitions=24)

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_RESULT_NOT_SIGNED"):
            adapter.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)
        assert adapter.probe_calls == 0
        assert adapter.probe_evidence == {}

    def test_adapter_refuses_tampered_mapping_result(self, monkeypatch):
        def factory(*, stage, student_episodes, reference_episodes):
            result = sign_probe_result(signed_payload(
                stage=stage, student_requested=student_episodes,
                student_completed=student_episodes,
                reference_requested=reference_episodes,
                reference_completed=reference_episodes))
            dump = result.model_dump()
            dump["metrics"]["student_success_rate"] = 0.99
            return dump                    # stale signature on new content

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_RESULT_ILLEGAL"):
            adapter.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)
        assert adapter.probe_calls == 0

    def test_adapter_refuses_wrong_issuer(self, monkeypatch):
        def factory(*, stage, student_episodes, reference_episodes):
            return sign_probe_result(signed_payload(
                stage=stage, issuer="shared.impostor_runner.v9",
                student_requested=student_episodes,
                student_completed=student_episodes,
                reference_requested=reference_episodes,
                reference_completed=reference_episodes))

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_RESULT_ISSUER_MISMATCH"):
            adapter.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)

    def test_adapter_refuses_episode_request_mismatch(self, monkeypatch):
        def factory(*, stage, student_episodes, reference_episodes):
            return sign_probe_result(signed_payload(
                stage=stage,
                student_requested=student_episodes + 3,
                student_completed=student_episodes + 3,
                reference_requested=reference_episodes,
                reference_completed=reference_episodes))

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_EPISODE_REQUEST_MISMATCH"):
            adapter.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)

    def test_ci_sample_counts_completed_only(self, monkeypatch):
        #: 2 student episodes requested, ONE completed + one rejected:
        #: the CI-sample count must be 2 (1 student + 1 reference), NOT 3
        def factory(*, stage, student_episodes, reference_episodes):
            return sign_probe_result(signed_payload(
                stage=stage, student_requested=student_episodes,
                student_completed=student_episodes - 1, student_failed=1,
                reference_requested=reference_episodes,
                reference_completed=reference_episodes))

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        adapter.probe(cand, stage="fast", student_episodes=2,
                      reference_episodes=1)
        evidence = adapter.probe_evidence[cand.candidate_id][-1]
        assert evidence["ci_sample_count"] == 2
        assert evidence["student_episodes_failed_or_rejected"] == 1

    def test_adapter_refuses_all_failed_episodes(self, monkeypatch):
        def factory(*, stage, student_episodes, reference_episodes):
            return sign_probe_result(signed_payload(
                stage=stage, student_requested=student_episodes,
                student_completed=0, student_failed=student_episodes,
                reference_requested=reference_episodes,
                reference_completed=0,
                reference_failed=reference_episodes))

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_NO_VALID_EPISODES"):
            adapter.probe(cand, stage="fast", student_episodes=2,
                          reference_episodes=1)
        assert adapter.probe_calls == 0


class TestFeedbackBindingHardening:
    def _bound_candidate(self):
        art = fam_t_artifact()
        return bind_candidate_to_artifact(make_candidate(FAM_T), art), art

    def test_positive_binding_with_mandatory_hashes(self):
        cand, art = self._bound_candidate()
        record, provenance = build_real_feedback_record(
            **record_kwargs(cand, art.artifact_hash))
        assert provenance.student_checkpoint_hash == STUDENT_CKPT
        assert provenance.reference_checkpoint_hash == REFERENCE_CKPT
        assert record.reference_identity_hash
        assert record.student_identity_hash

    def test_changed_axes_must_be_declared_mutation_axes(self):
        cand, art = self._bound_candidate()
        #: simulate protocol drift: an axis changed that was never
        #: declared among the candidate's mutation axes (model_copy does
        #: not revalidate — exactly the smuggled state the seam must catch)
        drifted = cand.model_copy(update=dict(
            axis_values=dict(cand.axis_values,
                             undeclared_axis="high")))
        kwargs = record_kwargs(drifted, art.artifact_hash)
        with pytest.raises(RealProbeBlocked,
                           match="CHANGED_AXES_NOT_IN_MUTATION_AXES"):
            build_real_feedback_record(**kwargs)

    @pytest.mark.parametrize("role", ["student", "reference"])
    def test_missing_checkpoint_hash_refused(self, role):
        cand, art = self._bound_candidate()
        kwargs = record_kwargs(cand, art.artifact_hash)
        kwargs[f"{role}_checkpoint_hash"] = ""
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_CHECKPOINT_HASH_MISSING"):
            build_real_feedback_record(**kwargs)

    @pytest.mark.parametrize("role", ["student", "reference"])
    def test_non_sha256_checkpoint_hash_refused(self, role):
        cand, art = self._bound_candidate()
        kwargs = record_kwargs(cand, art.artifact_hash)
        kwargs[f"{role}_checkpoint_hash"] = "zz-not-a-hash"
        with pytest.raises(RealProbeBlocked,
                           match="REAL_PROBE_CHECKPOINT_HASH_NOT_SHA256"):
            build_real_feedback_record(**kwargs)

    def test_provenance_requires_seed_bank(self):
        cand, art = self._bound_candidate()
        kwargs = record_kwargs(cand, art.artifact_hash)
        kwargs["seed_bank"] = []
        with pytest.raises(ValidationError,
                           match="PROVENANCE_WITHOUT_SEED_BANK"):
            build_real_feedback_record(**kwargs)

    def test_provenance_requires_positive_transitions(self):
        cand, art = self._bound_candidate()
        kwargs = record_kwargs(cand, art.artifact_hash)
        kwargs["stage_metrics"] = ProbeMetrics(
            stage="fast", student_success_rate=0.5,
            student_behavior_activation=0.5, student_front_progress=0.6,
            reference_success_rate=0.9, reference_mean_progress=0.8,
            reference_behavior_activation=0.7, global_retention=1.0,
            regret=0.4, learnability=0.4, simulator_transitions=0)
        with pytest.raises(ValidationError,
                           match="PROVENANCE_WITHOUT_TRANSITIONS"):
            build_real_feedback_record(**kwargs)

    def test_duplicate_feedback_id_refused_by_store(self):
        cand, art = self._bound_candidate()
        record, _p = build_real_feedback_record(
            **record_kwargs(cand, art.artifact_hash))
        store = SimulatorFeedbackStore()
        store.add(record)
        with pytest.raises(ValueError, match="DUPLICATE_FEEDBACK_ID"):
            store.add(record)


# ---------------------------------------------------------------------------
# P0-9 entrypoint guards (loaded by path — scripts/ is not a package)
# ---------------------------------------------------------------------------
def _load_entrypoint():
    path = (Path(__file__).resolve().parents[2] / "scripts"
            / "run_e2_real_two_window.py")
    spec = importlib.util.spec_from_file_location(
        "run_e2_real_two_window_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint()


class StubLedger:
    def __init__(self, records):
        self._records = dict(records)

    def get(self, hypothesis_id):
        if hypothesis_id not in self._records:
            raise KeyError(hypothesis_id)
        return self._records[hypothesis_id]

    def all(self):
        return list(self._records.values())


def _probe_metrics():
    return ProbeMetrics(
        stage="fast", student_success_rate=0.5,
        student_behavior_activation=0.5, student_front_progress=0.6,
        reference_success_rate=0.9, reference_mean_progress=0.8,
        reference_behavior_activation=0.7, global_retention=1.0,
        regret=0.4, learnability=0.4, simulator_transitions=24)


def _window_setup(store=None, ledger=None, evidence=True):
    art = fam_t_artifact()
    cand = bind_candidate_to_artifact(make_candidate(FAM_T), art)
    evidence_trail = [dict(
        stage="fast", seed_bank=[1, 2, 3], ci_sample_count=3,
        simulator_transitions=24, student_checkpoint_hash=STUDENT_CKPT,
        reference_checkpoint_hash=REFERENCE_CKPT,
        result_hash="ab" * 32, executable_artifact_id=art.artifact_id,
        executable_artifact_hash=art.artifact_hash)] if evidence else []
    adapter = SimpleNamespace(
        runner_id=ADAPTER_ID,
        probe_evidence={cand.candidate_id: evidence_trail})
    batch = SimpleNamespace(
        stage1_results=[dict(candidate_id=cand.candidate_id,
                             metrics=_probe_metrics().model_dump())],
        stage2_results=[])
    ledger = ledger if ledger is not None else StubLedger(
        {"hyp-00": SimpleNamespace(
            hypothesis_id="hyp-00",
            predicted_signature={"student_success_rate": 0.4})})
    controller = SimpleNamespace(ledger=ledger,
                                 store=store or SimulatorFeedbackStore())
    return dict(window=1, plan=SimpleNamespace(plan_id="plan-bind-01"),
                candidates=[cand], batch=batch, controller=controller,
                probe_adapter=adapter,
                student_binding=local_symbolic_binding(),
                reference_binding=ReferenceBindingIdentity(
                    candidate_id="REFERENCE_TEST_ONLY_SYNTHETIC",
                    parameter_tree_hash="e1" * 32,
                    checkpoint_global_step=0,
                    provenance_label="TEST_ONLY SYNTHETIC "
                                     "NOT_REAL_EXECUTION"),
                candidate=cand)


class TestEntrypointGuards:
    def test_positive_builds_one_provenance_bound_record(self):
        setup = _window_setup()
        records = ENTRYPOINT.build_window_real_feedback(**{
            k: v for k, v in setup.items() if k != "candidate"})
        assert len(records) == 1
        assert records[0].candidate_id == setup["candidate"].candidate_id
        assert records[0].reference_identity_hash
        assert records[0].expected_observed_match == MATCH_UNGRADED

    def test_duplicate_stage1_candidate_refused(self):
        setup = _window_setup()
        setup["batch"].stage1_results = [
            dict(setup["batch"].stage1_results[0]),
            dict(setup["batch"].stage1_results[0])]
        with pytest.raises(ENTRYPOINT.RealTwoWindowBlocked,
                           match="DUPLICATE_PROBE_FEEDBACK_RECORD"):
            ENTRYPOINT.build_window_real_feedback(**{
                k: v for k, v in setup.items() if k != "candidate"})

    def test_missing_probe_evidence_refused(self):
        setup = _window_setup(evidence=False)
        with pytest.raises(ENTRYPOINT.RealTwoWindowBlocked,
                           match="REAL_PROBE_EVIDENCE_MISSING"):
            ENTRYPOINT.build_window_real_feedback(**{
                k: v for k, v in setup.items() if k != "candidate"})

    def test_already_stored_feedback_id_refused(self):
        setup = _window_setup()
        records = ENTRYPOINT.build_window_real_feedback(**{
            k: v for k, v in setup.items() if k != "candidate"})
        setup["controller"].store.add(records[0])
        with pytest.raises(ENTRYPOINT.RealTwoWindowBlocked,
                           match="DUPLICATE_PROBE_FEEDBACK_RECORD"):
            ENTRYPOINT.build_window_real_feedback(**{
                k: v for k, v in setup.items() if k != "candidate"})

    def test_unknown_hypothesis_fails_closed(self):
        #: the forbidden ``except KeyError: continue`` skip is gone:
        #: an unknown hypothesis id refuses the predicted-signature merge
        setup = _window_setup(ledger=StubLedger({}))
        with pytest.raises(
                ENTRYPOINT.RealTwoWindowBlocked,
                match="UNKNOWN_HYPOTHESIS_ID_IN_PREDICTED_SIGNATURE"):
            ENTRYPOINT.build_window_real_feedback(**{
                k: v for k, v in setup.items() if k != "candidate"})

    def test_expected_signature_positive_merge(self):
        setup = _window_setup()
        merged = ENTRYPOINT._expected_signature(
            setup["controller"], setup["candidate"])
        assert merged == {"student_success_rate": 0.4}


class TestPosture:
    def test_real_capability_flags_stay_false(self):
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name

    def test_signed_consumption_enables_nothing(self, monkeypatch):
        #: consuming a fully signed result through the adapter keeps every
        #: capability flag False (authorization is a monkeypatched TEST
        #: fixture that reverts at teardown)
        def factory(*, stage, student_episodes, reference_episodes):
            return sign_probe_result(signed_payload(
                stage=stage, student_requested=student_episodes,
                student_completed=student_episodes,
                reference_requested=reference_episodes,
                reference_completed=reference_episodes))

        adapter = make_adapter(monkeypatch, factory)
        cand = bind_candidate_to_artifact(
            make_candidate(FAM_T), fam_t_artifact())
        adapter.probe(cand, stage="fast", student_episodes=2,
                      reference_episodes=1)
        assert C.REAL_SIMULATOR_PROBE is False
        assert C.REAL_CHECKPOINT_LOADED is False
        assert C.REAL_TRAINING_UPDATE_EXECUTED is False
