"""P0-2 (CC3 follow-up audit): executable environment artifacts bound into
the real probe path.

Contract under test::

    AxisDirective batch -> CanonicalTaskSpec -> RealEnvCoderArtifact
        -> ExecutableEnvironmentArtifact (immutable, hash-bound)
        -> bound parameterized candidates
        -> SharedCandidateProbeRunner (probes refuse unbound candidates)
        -> feedback records carrying the SAME artifact hash

Every fixture in this file is TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION:
a scripted transport stands in for the real LLM, a scripted runner stands
in for the shared CandidateProbeRunner, and the only Python sources ever
executed by the four-link verification chain are the two trivial module
literals defined below (never LLM-produced code). All REAL_* capability
flags stay False throughout; runtime authorization grants (monkeypatched
round flags + RealRuntimeAuthorization) are test fixtures, not
capabilities.
"""
import copy
from types import SimpleNamespace

import pytest

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.axis_directive import (
    DIRECTION_INCREASE,
    ROLE_TREATMENT,
    AxisDirective,
)
from d052.feedback_llm_ued.controller import FeedbackUEDController
from d052.feedback_llm_ued.env_coder import RealEnvCoderBlocked
from d052.feedback_llm_ued.executable_env_artifact import (
    ARTIFACT_ID_PREFIX,
    ExecutableArtifactBlocked,
    assert_candidate_artifact_binding,
    bind_candidate_to_artifact,
    derive_executable_artifacts,
)
from d052.feedback_llm_ued.execution_mode import (
    EXECUTION_MODE_REAL,
    FeedbackLaunchGate,
)
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    ProbeMetrics,
)
from d052.feedback_llm_ued.llm_backend import (
    DeterministicMockFeedbackBackend,
    RealBackendAdapter,
)
from d052.feedback_llm_ued.real_call_journal import (
    OUTPUT_SCHEMA_PARSED,
    RealCallJournal,
)
from d052.feedback_llm_ued.real_env_coder import (
    STATUS_FAILED,
    STATUS_PASSED,
    CanonicalTaskSpec,
    RealDirectiveArtifact,
    RealDirectiveBinding,
    RealEnvCoderArtifact,
    RealEnvCoderOutput,
    execute_real_env_coder,
    verify_directive_artifact,
)
from d052.feedback_llm_ued.real_probe_feedback import (
    FORBIDDEN_PRODUCTION_RUNNER_IDS,
    RealProbeFeedbackRunner,
    build_real_feedback_record,
)
from d052.feedback_llm_ued.real_simulator_probe import RealProbeBlocked
from d052.feedback_llm_ued.runtime_authorization import (
    RealRuntimeAuthorization,
    RuntimeAuthorizationBlocked,
)
from d052.feedback_llm_ued.shared_runtime_binding import (
    ReferenceBindingIdentity,
)
from d052.feedback_llm_ued.simulator_feedback_store import MATCH_UNGRADED
from d052.feedback_llm_ued.simulator_probe import (
    DeterministicSymbolicProbeRunner,
)
from d052.feedback_llm_ued.student_binding import local_symbolic_binding
from d052.feedback_llm_ued.synthetic_feedback import synthetic_candidate

#: TEST_ONLY SYNTHETIC NOT_REAL_EXECUTION identities
FAM_T = "threat_distance_family"
FAM_R = "resource_pressure_family"
FAM_D = "day_night_rest_need_family"
ADAPTER_ID = "shared.candidate_probe_runner.test_only.v1"
BACKEND_ID = "test.real.backend.v1"
MODEL_ID = "test-model.v1"

ABI_HASHES = dict(
    observation_abi_hash="a1" * 32,
    action_abi_hash="a2" * 32,
    reward_contract_hash="a3" * 32,
    reset_protocol_hash="a4" * 32,
    step_protocol_hash="a5" * 32,
)

#: the exact immutable contract field set (audit-grade; any drift fails).
#: ``protocol_version`` is the CanonicalModel infrastructure field shared by
#: every hash-chained object in the loop.
EXPECTED_ARTIFACT_FIELDS = frozenset({
    "artifact_id", "template_id", "source_window", "source_plan_id",
    "canonical_task_spec_hash", "directive_batch_hash",
    "environment_family", "directive_ids", "directive_hashes",
    "changed_axes", "held_constant_axes", "python_source_hash",
    "compiled_artifact_hash", "runtime_adapter_id", "observation_abi_hash",
    "action_abi_hash", "reward_contract_hash", "reset_protocol_hash",
    "step_protocol_hash", "validation_report_hash", "artifact_hash",
    "provenance_hash", "protocol_version"})

#: TEST_ONLY module literals — the ONLY sources ever executed by the
#: four-link chain in this file (never untrusted LLM code)
GOOD_MODULE_SOURCE = """\
def reset(seed):
    return {"seed": seed, "t": 0}


def step(state, action):
    nxt = dict(state, t=state["t"] + 1)
    return nxt, 0.0, nxt["t"] >= 3, {}
"""
NO_STEP_MODULE_SOURCE = """\
def reset(seed):
    return {"seed": seed}
"""


# ---------------------------------------------------------------------------
# synthetic chain fixtures (TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION)
# ---------------------------------------------------------------------------
def make_directive(directive_id, *, family, axis, old="low", new="high",
                   held=None, window=1):
    return AxisDirective(
        directive_id=directive_id, source_window=window,
        environment_family=family, axis=axis, old_level=old, new_level=new,
        direction=DIRECTION_INCREASE,
        experiment_control_role=ROLE_TREATMENT,
        held_constant_axes=dict(held or {}),
        expected_next_signature={"student_success_rate": 0.4},
        rationale="TEST_ONLY synthetic directive")


def make_directives(window=1):
    return [
        make_directive("dir-bind-t", family=FAM_T,
                       axis="threat_distance_grading",
                       held={"threat_count": "medium"}, window=window),
        make_directive("dir-bind-r", family=FAM_R,
                       axis="resource_pressure", window=window),
    ]


def build_chain(window=1, plan_id="plan-bind-01", directives=None):
    """A fully consistent spec + parsed output + PASSED source artifact."""
    directives = list(directives if directives is not None
                      else make_directives(window))
    spec = CanonicalTaskSpec(
        window=window, plan_id=plan_id,
        directives=[d.model_dump() for d in directives])
    bindings = [RealDirectiveBinding(
        directive_id=d.directive_id, directive_hash=d.directive_hash,
        environment_family=d.environment_family,
        python_source=GOOD_MODULE_SOURCE,
        reset_contract="reset(seed)->state",
        step_contract="step(state,action)->(state,reward,terminal,info)")
        for d in directives]
    parsed = RealEnvCoderOutput(
        window=window, plan_id=plan_id, directive_bindings=bindings,
        directive_batch_hash=spec.directive_batch_hash)
    dart = [verify_directive_artifact(b) for b in bindings]
    assert all(a.passed for a in dart)      # the fixture itself is green
    source = RealEnvCoderArtifact(
        window=window, plan_id=plan_id, spec_hash=spec.spec_hash,
        backend_id=BACKEND_ID, model_id=MODEL_ID, n_calls=1,
        repair_attempts=0, logical_call_ids=[f"test-lcid-w{window}"],
        envelope_request_hashes=["ab" * 32], directive_artifacts=dart,
        overall_status=STATUS_PASSED, blockers=[])
    return dict(directives=directives, spec=spec, parsed=parsed,
                source=source)


def derive_ok(chain, *, runtime_adapter_id=ADAPTER_ID, **abi_overrides):
    hashes = dict(ABI_HASHES)
    hashes.update(abi_overrides)
    return derive_executable_artifacts(
        spec=chain["spec"], parsed=chain["parsed"],
        source_artifact=chain["source"], directives=chain["directives"],
        runtime_adapter_id=runtime_adapter_id, **hashes)


def make_candidate(family, *, i=0, hypothesis_ids=("hyp-00",)):
    cand = synthetic_candidate(candidate_id=f"c-bind-{family[:3]}-{i}",
                               family=family)
    dump = cand.model_dump()
    dump["distinguishes_hypothesis_ids"] = list(hypothesis_ids)
    #: content changed -> un-carry the old hash; the rebuild recomputes it
    dump["candidate_hash"] = ""
    return CandidateEnvironment(**dump)


class FakeSharedProbeRunner:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION stand-in for the shared
    CandidateProbeRunner: scripted coarse metrics, no simulator, no jax."""

    runner_id = ADAPTER_ID
    real_simulator = True

    def __init__(self, transitions=24):
        self._transitions = transitions
        self.calls = []

    def probe_candidate(self, *, candidate_hash, environment_family,
                        axis_values, held_constant_axes, stage,
                        student_episodes, reference_episodes, seed_bank):
        self.calls.append(dict(candidate_hash=candidate_hash, stage=stage,
                               environment_family=environment_family))
        metrics = dict(
            student_success_rate=0.5, student_behavior_activation=0.5,
            student_front_progress=0.6, reference_success_rate=0.9,
            reference_mean_progress=0.8, reference_behavior_activation=0.7,
            global_retention=1.0, regret=0.4, learnability=0.4,
            simulator_transitions=self._transitions,
            too_hard=False, too_easy=False)
        return SimpleNamespace(
            metrics=metrics, simulator_transitions=self._transitions,
            episode_count=student_episodes + reference_episodes,
            student_checkpoint_hash="c1" * 32,
            reference_checkpoint_hash="d1" * 32)


@pytest.fixture
def probe_setup(monkeypatch):
    """A gate-authorized RealProbeFeedbackRunner over the fake shared
    runner, with both family artifacts bound (runtime authorization is a
    TEST fixture; capability flags stay False)."""
    monkeypatch.setattr(C, "REAL_SIMULATOR_PROBE_AUTHORIZED", True)
    gate = FeedbackLaunchGate(EXECUTION_MODE_REAL)
    shared = FakeSharedProbeRunner()
    adapter = RealProbeFeedbackRunner(
        shared_runner=shared, gate=gate,
        student_identity_hash=local_symbolic_binding().identity_hash)
    chain = build_chain()
    artifacts = derive_ok(chain)
    adapter.bind_executable_artifacts(artifacts)
    return SimpleNamespace(adapter=adapter, shared=shared, chain=chain,
                           artifacts={a.environment_family: a
                                      for a in artifacts})


def record_kwargs(candidate, artifact_hash, *, window=1,
                  plan_id="plan-bind-01", **over):
    metrics = ProbeMetrics(
        stage="fast", student_success_rate=0.5,
        student_behavior_activation=0.5, student_front_progress=0.6,
        reference_success_rate=0.9, reference_mean_progress=0.8,
        reference_behavior_activation=0.7, global_retention=1.0,
        regret=0.4, learnability=0.4, simulator_transitions=24)
    base = dict(
        feedback_id=f"fb-bind-{candidate.candidate_id}",
        candidate=candidate, source_window=window, source_plan_id=plan_id,
        known_hypothesis_ids=["hyp-00"],
        predicted_signature={"student_success_rate": 0.4},
        stage_metrics=metrics,
        reference_stats=dict(episode_success_rate=0.9, mean_progress=0.8,
                             behavior_activation_rate=0.7),
        student_binding=local_symbolic_binding(),
        reference_binding=ReferenceBindingIdentity(
            candidate_id="REFERENCE_TEST_ONLY_SYNTHETIC",
            parameter_tree_hash="e1" * 32, checkpoint_global_step=0,
            provenance_label="TEST_ONLY SYNTHETIC NOT_REAL_EXECUTION"),
        runner_id=ADAPTER_ID, seed_bank=[1, 2, 3], ci_sample_count=3,
        expected_observed_match=MATCH_UNGRADED,
        executable_artifact_hash=artifact_hash)
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# the immutable artifact contract
# ---------------------------------------------------------------------------
class TestArtifactContract:
    def test_exact_field_set_and_sha_hashes(self):
        arts = derive_ok(build_chain())
        assert len(arts) == 2                       # one per family
        for art in arts:
            assert set(art.model_dump()) == EXPECTED_ARTIFACT_FIELDS
            assert art.template_id == C.ENVCODER_UNIQUE_TEMPLATE_ID
            assert art.artifact_id.startswith(ARTIFACT_ID_PREFIX)
            assert len(art.artifact_id) == len(ARTIFACT_ID_PREFIX) + 16
            for field_name in (
                    "canonical_task_spec_hash", "directive_batch_hash",
                    "python_source_hash", "compiled_artifact_hash",
                    "observation_abi_hash", "action_abi_hash",
                    "reward_contract_hash", "reset_protocol_hash",
                    "step_protocol_hash", "validation_report_hash",
                    "artifact_hash", "provenance_hash"):
                value = getattr(art, field_name)
                assert len(value) == 64 and \
                    all(c in "0123456789abcdef" for c in value), field_name
        fams = {a.environment_family for a in arts}
        assert fams == {FAM_T, FAM_R}

    def test_changed_and_held_axes_bound_from_directives(self):
        arts = {a.environment_family: a for a in derive_ok(build_chain())}
        assert arts[FAM_T].changed_axes == {"threat_distance_grading": "high"}
        assert arts[FAM_T].held_constant_axes == {"threat_count": "medium"}
        assert arts[FAM_R].changed_axes == {"resource_pressure": "high"}
        assert arts[FAM_R].held_constant_axes == {}
        assert arts[FAM_T].directive_ids == ["dir-bind-t"]
        assert arts[FAM_R].directive_ids == ["dir-bind-r"]

    def test_tampered_artifact_fails_closed(self):
        art = derive_ok(build_chain())[0]
        dump = art.model_dump()
        dump["environment_family"] = (FAM_D if dump["environment_family"]
                                      != FAM_D else FAM_T)
        with pytest.raises(ValueError, match="CONTENT_HASH_MISMATCH"):
            type(art)(**dump)

    def test_illegal_construction_rejected(self):
        art = derive_ok(build_chain())[0]
        dump = art.model_dump()
        dump["template_id"] = "SOME_OTHER_TEMPLATE"
        with pytest.raises(ValueError, match="ENVCODER_TEMPLATE_NOT_UNIQUE"):
            type(art)(**dump)
        dump = art.model_dump()
        dump["environment_family"] = "not_a_family"
        with pytest.raises(ValueError, match="UNKNOWN_ENVIRONMENT_FAMILY"):
            type(art)(**dump)
        dump = art.model_dump()
        dump["directive_hashes"] = dump["directive_hashes"] + ["f" * 64]
        with pytest.raises(ValueError,
                           match="EXECUTABLE_ARTIFACT_HASH_COUNT_MISMATCH"):
            type(art)(**dump)
        dump = art.model_dump()
        dump["directive_ids"] = []
        dump["directive_hashes"] = []
        with pytest.raises(ValueError,
                           match="EXECUTABLE_ARTIFACT_WITHOUT_DIRECTIVES"):
            type(art)(**dump)
        dump = art.model_dump()
        dump["observation_abi_hash"] = "zz"
        with pytest.raises(ValueError,
                           match="EXECUTABLE_ARTIFACT_HASH_NOT_SHA256"):
            type(art)(**dump)
        dump = art.model_dump()
        dump["directive_ids"] = dump["directive_ids"] + \
            dump["directive_ids"]
        with pytest.raises(ValueError,
                           match="DUPLICATE_EXECUTABLE_DIRECTIVE_ID"):
            type(art)(**dump)


# ---------------------------------------------------------------------------
# derivation fail-closed ladder
# ---------------------------------------------------------------------------
class TestDeriveFailClosedLadder:
    def test_positive_derivation_is_deterministic(self):
        chain = build_chain()
        first = derive_ok(chain)
        second = derive_ok(build_chain())
        assert [a.artifact_hash for a in first] == \
            [a.artifact_hash for a in second]
        assert [a.artifact_id for a in first] == \
            [a.artifact_id for a in second]
        for art in first:
            #: provenance binds the source RealEnvCoderArtifact
            assert art.provenance_hash == chain["source"].artifact_hash
            assert art.canonical_task_spec_hash == chain["spec"].spec_hash
            assert art.directive_batch_hash == \
                chain["spec"].directive_batch_hash

    def test_source_not_passed_rejected(self):
        chain = build_chain()
        failed = RealEnvCoderArtifact(
            window=chain["source"].window,
            plan_id=chain["source"].plan_id,
            spec_hash=chain["source"].spec_hash, backend_id=BACKEND_ID,
            model_id=MODEL_ID, n_calls=1, repair_attempts=0,
            logical_call_ids=["test-lcid"], envelope_request_hashes=[],
            directive_artifacts=chain["source"].directive_artifacts,
            overall_status=STATUS_FAILED,
            blockers=["TEST_ONLY forced failure"])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_SOURCE_NOT_PASSED"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=chain["parsed"],
                source_artifact=failed, directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)

    def test_verification_incomplete_rejected(self):
        chain = build_chain()
        bad_dart = [RealDirectiveArtifact(
            directive_id=chain["source"].directive_artifacts[0].directive_id,
            directive_hash=chain[
                "source"].directive_artifacts[0].directive_hash,
            python_source_hash="ab" * 32, compile_status=STATUS_PASSED,
            import_status=STATUS_PASSED, reset_status=STATUS_PASSED,
            step_status=STATUS_FAILED, blockers=["TEST_ONLY step failure"])]
        rebuilt = RealEnvCoderArtifact(
            window=chain["source"].window,
            plan_id=chain["source"].plan_id,
            spec_hash=chain["source"].spec_hash, backend_id=BACKEND_ID,
            model_id=MODEL_ID, n_calls=1, repair_attempts=0,
            logical_call_ids=["test-lcid"], envelope_request_hashes=[],
            directive_artifacts=bad_dart +
            chain["source"].directive_artifacts[1:],
            overall_status=STATUS_PASSED, blockers=[])
        with pytest.raises(
                ExecutableArtifactBlocked,
                match="EXECUTABLE_ARTIFACT_VERIFICATION_INCOMPLETE"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=chain["parsed"],
                source_artifact=rebuilt, directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)

    def test_spec_window_plan_output_mismatches_rejected(self):
        chain = build_chain()
        other_spec = CanonicalTaskSpec(
            window=1, plan_id="plan-other",
            directives=[d.model_dump() for d in chain["directives"]])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_SPEC_MISMATCH"):
            derive_executable_artifacts(
                spec=other_spec, parsed=chain["parsed"],
                source_artifact=chain["source"],
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)
        window_shifted = RealEnvCoderArtifact(
            window=chain["source"].window + 1,
            plan_id=chain["source"].plan_id,
            spec_hash=chain["source"].spec_hash, backend_id=BACKEND_ID,
            model_id=MODEL_ID, n_calls=1, repair_attempts=0,
            logical_call_ids=["test-lcid"], envelope_request_hashes=[],
            directive_artifacts=chain["source"].directive_artifacts,
            overall_status=STATUS_PASSED, blockers=[])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_WINDOW_MISMATCH"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=chain["parsed"],
                source_artifact=window_shifted,
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)
        plan_shifted = RealEnvCoderArtifact(
            window=chain["source"].window, plan_id="plan-other",
            spec_hash=chain["source"].spec_hash, backend_id=BACKEND_ID,
            model_id=MODEL_ID, n_calls=1, repair_attempts=0,
            logical_call_ids=["test-lcid"], envelope_request_hashes=[],
            directive_artifacts=chain["source"].directive_artifacts,
            overall_status=STATUS_PASSED, blockers=[])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_PLAN_MISMATCH"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=chain["parsed"],
                source_artifact=plan_shifted,
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)
        output_shifted = RealEnvCoderOutput(
            window=chain["parsed"].window + 4,
            plan_id=chain["parsed"].plan_id,
            directive_bindings=chain["parsed"].directive_bindings,
            directive_batch_hash=chain["parsed"].directive_batch_hash)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_OUTPUT_MISMATCH"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=output_shifted,
                source_artifact=chain["source"],
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)

    def test_undeclared_adapter_rejected(self):
        chain = build_chain()
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_ADAPTER_UNDECLARED"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=chain["parsed"],
                source_artifact=chain["source"],
                directives=chain["directives"], runtime_adapter_id="",
                **ABI_HASHES)

    def test_coverage_mismatch_rejected(self):
        chain = build_chain()
        partial = RealEnvCoderOutput(
            window=chain["parsed"].window, plan_id=chain["parsed"].plan_id,
            directive_bindings=chain["parsed"].directive_bindings[:1],
            directive_batch_hash=chain["parsed"].directive_batch_hash)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_COVERAGE_MISMATCH"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=partial,
                source_artifact=chain["source"],
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)

    def test_directive_hash_mismatch_rejected(self):
        chain = build_chain()
        first = chain["parsed"].directive_bindings[0]
        forged = RealDirectiveBinding(
            directive_id=first.directive_id, directive_hash="f" * 64,
            environment_family=first.environment_family,
            python_source=first.python_source,
            reset_contract=first.reset_contract,
            step_contract=first.step_contract)
        forged_parsed = RealEnvCoderOutput(
            window=chain["parsed"].window, plan_id=chain["parsed"].plan_id,
            directive_bindings=[forged] +
            chain["parsed"].directive_bindings[1:],
            directive_batch_hash=chain["parsed"].directive_batch_hash)
        with pytest.raises(
                ExecutableArtifactBlocked,
                match="EXECUTABLE_ARTIFACT_DIRECTIVE_HASH_MISMATCH"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=forged_parsed,
                source_artifact=chain["source"],
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)

    def test_family_mismatch_rejected(self):
        chain = build_chain()
        first = chain["parsed"].directive_bindings[0]
        lying = RealDirectiveBinding(
            directive_id=first.directive_id,
            directive_hash=first.directive_hash,
            environment_family="visibility_family",
            python_source=first.python_source,
            reset_contract=first.reset_contract,
            step_contract=first.step_contract)
        lying_parsed = RealEnvCoderOutput(
            window=chain["parsed"].window, plan_id=chain["parsed"].plan_id,
            directive_bindings=[lying] +
            chain["parsed"].directive_bindings[1:],
            directive_batch_hash=chain["parsed"].directive_batch_hash)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_FAMILY_MISMATCH"):
            derive_executable_artifacts(
                spec=chain["spec"], parsed=lying_parsed,
                source_artifact=chain["source"],
                directives=chain["directives"],
                runtime_adapter_id=ADAPTER_ID, **ABI_HASHES)

    def test_held_axis_conflicts_rejected(self):
        # held axis also CHANGED within the same family batch
        changed_held = build_chain(directives=[
            make_directive("dir-h1", family=FAM_D,
                           axis="day_night_rest_need",
                           held={"rest_need_pressure": "low"}),
            make_directive("dir-h2", family=FAM_D,
                           axis="rest_need_pressure")])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_HELD_AXIS_CONFLICT"):
            derive_ok(changed_held)
        # held axis at TWO different levels within the same family batch
        two_levels = build_chain(directives=[
            make_directive("dir-d1", family=FAM_D,
                           axis="day_night_rest_need",
                           held={"rest_need_pressure": "low"}),
            make_directive("dir-d2", family=FAM_D,
                           axis="safe_rest_area_availability",
                           held={"rest_need_pressure": "high"})])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_HELD_AXIS_CONFLICT"):
            derive_ok(two_levels)

    def test_non_sha256_abi_hash_rejected(self):
        chain = build_chain()
        with pytest.raises(ValueError):
            derive_ok(chain, observation_abi_hash="not-a-sha")


# ---------------------------------------------------------------------------
# the real EnvCoder run_sink seam (scripted transport, NOT a real LLM)
# ---------------------------------------------------------------------------
class TestRunSinkSeam:
    def _backend(self, responses, journal=None):
        queue = list(responses)

        def transport(role, prompt):
            assert role == C.ROLE_ENV_CODER
            return queue.pop(0)

        return RealBackendAdapter(transport, backend_id=BACKEND_ID,
                                  model_id=MODEL_ID, authorized=True,
                                  journal=journal)

    def test_sink_called_once_with_verified_run(self):
        chain = build_chain(window=2, plan_id="plan-sink")
        sink_runs = []

        def sink(**run):
            sink_runs.append(run)

        backend = self._backend([chain["parsed"].model_dump_json()])
        artifact = execute_real_env_coder(
            window=2, plan_id="plan-sink", directives=chain["directives"],
            backend=backend,
            authorization=RealRuntimeAuthorization(
                real_llm_backend=True, real_envcoder=True),
            sequence=6, run_sink=sink)
        assert artifact.overall_status == STATUS_PASSED
        assert len(sink_runs) == 1
        run = sink_runs[0]
        #: the SAME objects that passed verification — never disk copies
        assert run["artifact"] is artifact
        assert run["spec"].spec_hash == artifact.spec_hash
        assert run["parsed"].directive_batch_hash == \
            run["spec"].directive_batch_hash
        arts = derive_executable_artifacts(
            spec=run["spec"], parsed=run["parsed"],
            source_artifact=run["artifact"],
            directives=chain["directives"], runtime_adapter_id=ADAPTER_ID,
            **ABI_HASHES)
        assert {a.environment_family for a in arts} == {FAM_T, FAM_R}

    def test_sink_failure_blocks_the_window(self):
        chain = build_chain(window=2, plan_id="plan-sink-fail")

        def exploding_sink(**run):
            raise ExecutableArtifactBlocked(
                "TEST_ONLY_SINK_FAILURE: the binding step failed closed")

        backend = self._backend([chain["parsed"].model_dump_json()])
        with pytest.raises(ExecutableArtifactBlocked,
                           match="TEST_ONLY_SINK_FAILURE"):
            execute_real_env_coder(
                window=2, plan_id="plan-sink-fail",
                directives=chain["directives"], backend=backend,
                authorization=RealRuntimeAuthorization(
                real_llm_backend=True, real_envcoder=True),
                sequence=6, run_sink=exploding_sink)

    def test_repair_then_success_calls_sink_once(self):
        chain = build_chain(window=2, plan_id="plan-repair")
        broken = RealEnvCoderOutput(
            window=2, plan_id="plan-repair",
            directive_bindings=[RealDirectiveBinding(
                directive_id=d.directive_id, directive_hash=d.directive_hash,
                environment_family=d.environment_family,
                python_source=NO_STEP_MODULE_SOURCE,
                reset_contract="reset(seed)->state",
                step_contract="step(state,action)->4-tuple")
                for d in chain["directives"]],
            directive_batch_hash=chain["spec"].directive_batch_hash)
        sink_runs = []
        backend = self._backend(
            [broken.model_dump_json(), chain["parsed"].model_dump_json()])
        artifact = execute_real_env_coder(
            window=2, plan_id="plan-repair",
            directives=chain["directives"], backend=backend,
            authorization=RealRuntimeAuthorization(
                real_llm_backend=True, real_envcoder=True),
            sequence=6, run_sink=lambda **run: sink_runs.append(run))
        assert artifact.n_calls == 2
        assert artifact.repair_attempts == 1
        assert artifact.overall_status == STATUS_PASSED
        assert len(sink_runs) == 1           # sink fires ONLY on success

    def test_journal_records_parsed_outcome(self):
        chain = build_chain(window=2, plan_id="plan-journal")
        journal = RealCallJournal()
        backend = self._backend([chain["parsed"].model_dump_json()],
                                journal=journal)
        execute_real_env_coder(
            window=2, plan_id="plan-journal",
            directives=chain["directives"], backend=backend,
            authorization=RealRuntimeAuthorization(
                real_llm_backend=True, real_envcoder=True),
            sequence=6, journal=journal)
        assert len(journal.entries) >= 1
        assert any(getattr(e, "output_schema_status", None)
                   == OUTPUT_SCHEMA_PARSED for e in journal.entries)

    def test_budget_exhaustion_and_authorization_ladder(self):
        chain = build_chain(window=2, plan_id="plan-exhaust")
        broken = RealEnvCoderOutput(
            window=2, plan_id="plan-exhaust",
            directive_bindings=[RealDirectiveBinding(
                directive_id=d.directive_id, directive_hash=d.directive_hash,
                environment_family=d.environment_family,
                python_source=NO_STEP_MODULE_SOURCE,
                reset_contract="reset(seed)->state",
                step_contract="step(state,action)->4-tuple")
                for d in chain["directives"]],
            directive_batch_hash=chain["spec"].directive_batch_hash)
        raw = broken.model_dump_json()
        backend = self._backend([raw, raw, raw])
        with pytest.raises(RealEnvCoderBlocked,
                           match="REAL_ENVCODER_REPAIR_BUDGET_EXHAUSTED"):
            execute_real_env_coder(
                window=2, plan_id="plan-exhaust",
                directives=chain["directives"], backend=backend,
                authorization=RealRuntimeAuthorization(
                real_llm_backend=True, real_envcoder=True),
                sequence=6)
        backend2 = self._backend([chain["parsed"].model_dump_json()])
        with pytest.raises(RealEnvCoderBlocked,
                           match="REAL_ENVCODER_NOT_AUTHORIZED"):
            execute_real_env_coder(
                window=2, plan_id="plan-exhaust",
                directives=chain["directives"], backend=backend2,
                authorization=RealRuntimeAuthorization(), sequence=6)
        with pytest.raises(RealEnvCoderBlocked,
                           match="REAL_ENVCODER_BACKEND_NOT_REAL"):
            execute_real_env_coder(
                window=2, plan_id="plan-exhaust",
                directives=chain["directives"],
                backend=DeterministicMockFeedbackBackend(),
                authorization=RealRuntimeAuthorization(
                real_llm_backend=True, real_envcoder=True),
                sequence=6)
        #: grant consistency is checked at construction: the EnvCoder is an
        #: LLM call, so real_envcoder without real_llm_backend is refused
        with pytest.raises(RuntimeAuthorizationBlocked,
                           match="INCONSISTENT_RUNTIME_GRANTS"):
            RealRuntimeAuthorization(real_envcoder=True)


# ---------------------------------------------------------------------------
# candidate binding (parameterized instances entering the probe)
# ---------------------------------------------------------------------------
class TestCandidateBinding:
    def test_bound_copy_recomputes_hash_and_passes_assertion(self):
        arts = derive_ok(build_chain())
        art = next(a for a in arts if a.environment_family == FAM_T)
        cand = make_candidate(FAM_T)
        bound = bind_candidate_to_artifact(cand, art)
        assert bound.executable_artifact_id == art.artifact_id
        assert bound.executable_artifact_hash == art.artifact_hash
        assert bound.parameter_variant_hash and bound.seed_policy_hash
        assert bound.candidate_hash != cand.candidate_hash
        #: binding is a NEW object; the original stays unbound
        assert cand.executable_artifact_id == ""
        assert_candidate_artifact_binding(bound, art)   # no raise
        #: deterministic — binding twice is bit-identical
        again = bind_candidate_to_artifact(cand, art)
        assert again.candidate_hash == bound.candidate_hash
        assert again.model_dump() == bound.model_dump()

    def test_family_mismatch_at_bind_rejected(self):
        arts = derive_ok(build_chain())
        art_r = next(a for a in arts if a.environment_family == FAM_R)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_FAMILY_MISMATCH"):
            bind_candidate_to_artifact(make_candidate(FAM_T), art_r)

    def test_assertion_ladder_on_tampered_bindings(self):
        arts = derive_ok(build_chain())
        art = next(a for a in arts if a.environment_family == FAM_T)
        other = next(a for a in arts if a.environment_family == FAM_R)
        cand = make_candidate(FAM_T)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_UNBOUND"):
            assert_candidate_artifact_binding(cand, art)
        bound = bind_candidate_to_artifact(cand, art)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_ID_MISMATCH"):
            assert_candidate_artifact_binding(bound, other)
        tampered = copy.copy(bound)
        object.__setattr__(tampered, "executable_artifact_hash", "f" * 64)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_HASH_MISMATCH"):
            assert_candidate_artifact_binding(tampered, art)
        incomplete = copy.copy(bound)
        object.__setattr__(incomplete, "seed_policy_hash", "")
        with pytest.raises(
                ExecutableArtifactBlocked,
                match="EXECUTABLE_ARTIFACT_BINDING_INCOMPLETE"):
            assert_candidate_artifact_binding(incomplete, art)


# ---------------------------------------------------------------------------
# probe boundary (fake shared runner; TEST_ONLY authorization)
# ---------------------------------------------------------------------------
class TestProbeBoundary:
    def test_bound_probe_positive_and_evidence_trail(self, probe_setup):
        art = probe_setup.artifacts[FAM_T]
        cand = make_candidate(FAM_T)
        bound = bind_candidate_to_artifact(cand, art)
        metrics = probe_setup.adapter.probe(
            bound, stage="fast", student_episodes=2, reference_episodes=1)
        assert metrics.stage == "fast"
        assert metrics.simulator_transitions == 24
        assert probe_setup.adapter.probe_calls == 1
        assert probe_setup.adapter.total_transitions == 24
        assert len(probe_setup.shared.calls) == 1
        evidence = probe_setup.adapter.probe_evidence[cand.candidate_id][-1]
        assert evidence["executable_artifact_id"] == art.artifact_id
        assert evidence["executable_artifact_hash"] == art.artifact_hash
        assert evidence["ci_sample_count"] == 3
        #: a scripted probe executes no real capability
        assert C.REAL_SIMULATOR_PROBE is False
        assert C.REAL_ENVCODER_USED is False

    def test_unbound_candidate_rejected_before_episodes(self, probe_setup):
        art = probe_setup.artifacts[FAM_T]
        unbound = make_candidate(FAM_T)
        with pytest.raises(ExecutableArtifactBlocked,
                           match="EXECUTABLE_ARTIFACT_UNBOUND"):
            probe_setup.adapter.probe(unbound, stage="fast",
                                      student_episodes=2,
                                      reference_episodes=1)
        assert probe_setup.shared.calls == []      # no episode ever ran

    def test_family_without_artifact_rejected(self, probe_setup):
        no_art = make_candidate(FAM_D)
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_MISSING"):
            probe_setup.adapter.probe(no_art, stage="fast",
                                      student_episodes=2,
                                      reference_episodes=1)

    def test_controller_seam_binds_or_blocks(self, probe_setup):
        cands = [make_candidate(FAM_T, i=1), make_candidate(FAM_R, i=2)]
        bound = probe_setup.adapter. \
            bind_candidates_to_executable_artifacts(cands)
        assert all(c.executable_artifact_hash for c in bound)
        assert {c.environment_family for c in bound} == {FAM_T, FAM_R}
        with_family_missing = cands + [make_candidate(FAM_D, i=3)]
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_MISSING"):
            probe_setup.adapter. \
                bind_candidates_to_executable_artifacts(with_family_missing)

    def test_rebind_rules(self, probe_setup):
        art = probe_setup.artifacts[FAM_T]
        #: idempotent rebind of the SAME artifact is allowed
        probe_setup.adapter.bind_executable_artifacts([art])
        #: conflicting hash under the same artifact_id fails closed
        evil = copy.copy(art)
        object.__setattr__(evil, "artifact_hash", "f" * 64)
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_REBIND_MISMATCH"):
            probe_setup.adapter.bind_executable_artifacts([evil])
        #: artifacts minted for a forbidden runtime adapter are refused
        forbidden = sorted(FORBIDDEN_PRODUCTION_RUNNER_IDS)[0]
        chain = build_chain(plan_id="plan-forbidden")
        tainted = derive_ok(chain, runtime_adapter_id=forbidden)
        with pytest.raises(RealProbeBlocked,
                           match="PRODUCTION_PATH_FORBIDDEN_RUNNER"):
            probe_setup.adapter.bind_executable_artifacts(tainted)

    def test_family_conflict_lookup_rejected(self, probe_setup):
        second = derive_ok(build_chain(plan_id="plan-bind-02"))
        probe_setup.adapter.bind_executable_artifacts(second)
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_FAMILY_CONFLICT"):
            probe_setup.adapter.lookup_executable_artifact(
                environment_family=FAM_T)

    def test_real_flags_stay_false_after_fake_real_probe(self, probe_setup):
        #: REAL_SIMULATOR_PROBE_AUTHORIZED is the probe_setup fixture's
        #: monkeypatched runtime grant — it reverts at teardown (the final
        #: posture test below proves it); every other flag must be False
        #: WHILE the fake-real probe is running
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            if name == "REAL_SIMULATOR_PROBE_AUTHORIZED":
                continue
            assert getattr(C, name) is False, name


# ---------------------------------------------------------------------------
# feedback record artifact-hash binding
# ---------------------------------------------------------------------------
class TestFeedbackRecordArtifactBinding:
    def _bound(self):
        arts = derive_ok(build_chain())
        art = next(a for a in arts if a.environment_family == FAM_T)
        bound = bind_candidate_to_artifact(make_candidate(FAM_T), art)
        return bound, art

    def test_positive_record_binds_same_artifact_hash(self):
        bound, art = self._bound()
        record, provenance = build_real_feedback_record(
            **record_kwargs(bound, art.artifact_hash))
        assert provenance.executable_artifact_hash == art.artifact_hash
        assert record.provenance["executable_artifact_hash"] == \
            art.artifact_hash
        assert record.provenance["production_path"] is True
        assert record.provenance["symbolic_metrics_forbidden"] is True
        assert record.reference_identity_hash == \
            record_kwargs(bound, art.artifact_hash)[
                "reference_binding"].identity_hash

    def test_missing_or_garbage_hash_rejected(self):
        bound, art = self._bound()
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_HASH_MISSING"):
            build_real_feedback_record(**record_kwargs(bound, ""))
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_HASH_NOT_SHA256"):
            build_real_feedback_record(**record_kwargs(bound, "zz" * 32))

    def test_hash_mismatch_vs_candidate_binding_rejected(self):
        bound, _art = self._bound()
        with pytest.raises(RealProbeBlocked,
                           match="EXECUTABLE_ARTIFACT_HASH_MISMATCH"):
            build_real_feedback_record(**record_kwargs(bound, "f" * 64))

    def test_provenance_tamper_fails_closed(self):
        bound, art = self._bound()
        _record, provenance = build_real_feedback_record(
            **record_kwargs(bound, art.artifact_hash))
        tampered = copy.copy(provenance)
        object.__setattr__(tampered, "source_window", 9)
        with pytest.raises(ValueError, match="CONTENT_HASH_MISMATCH"):
            type(provenance)(**tampered.model_dump())


# ---------------------------------------------------------------------------
# controller seam wiring (default symbolic path stays byte-identical)
# ---------------------------------------------------------------------------
class _SpySymbolicRunner:
    """Delegates to the deterministic symbolic runner but exposes the
    production binding seam, recording every call (TEST_ONLY)."""

    def __init__(self):
        self._inner = DeterministicSymbolicProbeRunner()
        self.bind_calls = []

    def bind_candidates_to_executable_artifacts(self, candidates):
        self.bind_calls.append(len(candidates))
        return list(candidates)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class TestControllerBindingSeam:
    def test_seam_invoked_every_window(self):
        spy = _SpySymbolicRunner()
        ctl = FeedbackUEDController(C.MODE_NORMAL_FEEDBACK,
                                    probe_runner=spy)
        summary = ctl.run(max_windows=2)
        assert len(spy.bind_calls) == 2
        assert all(n == C.RAW_CANDIDATES for n in spy.bind_calls)
        assert summary.n_llm_calls == 7 * 2
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name

    def test_default_symbolic_runner_exposes_no_seam(self):
        assert not hasattr(DeterministicSymbolicProbeRunner,
                           "bind_candidates_to_executable_artifacts")


# ---------------------------------------------------------------------------
# closing posture: NOTHING this file does flips a real capability flag
# ---------------------------------------------------------------------------
class TestPosture:
    def test_every_real_flag_is_false_with_no_fixture_active(self):
        #: runs with NO monkeypatched grant in scope — proves the fixture
        #: reverts and no test in this module flipped anything permanent
        for name in C.NEVER_TRUE_REAL_CAPABILITY_FLAGS:
            assert getattr(C, name) is False, name
        assert C.E2_PILOT_AUTHORIZED is False
