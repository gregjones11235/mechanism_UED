"""CC2 follow-up P0-4 tests: registry-signed CandidateProbeResult
consumption.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
all results are signed by the SYNTHETIC_TEST_ONLY_PROBE_SIGNER; the
production registry whitelist is EMPTY, so production consumption
must fail closed. No real probe ever runs here.

Covered negative matrix:
* empty pool / empty candidates            -> PROBE_POOL_EMPTY
* attestation tamper                       -> PROBE_HASH_MISMATCH
* TEST_ONLY result on production surface   -> TEST_ONLY_SIGNER_REJECTED
* wrong TEST_ONLY signer                   -> TEST_ONLY_SIGNER_REJECTED
* production signer (whitelist EMPTY)      -> PROBE_SIGNER_UNAUTHORIZED
* stale candidate                          -> PROBE_STALE
* wrong Student / Reference / checkpoint   -> PROBE_*_MISMATCH
* wrong seed bank / reset protocol         -> PROBE_*_MISMATCH
* wrong runner registry                    -> PROBE_CANDIDATE_MISMATCH
* duplicate result / duplicate candidate   -> PROBE_DUPLICATE
* mock/replay runner disguised             -> PROBE_MOCK_RUNNER_DISGUISED
* partial episodes at issue                -> PROBE_PARTIAL_EPISODES
* driver stage: unbound probe surfaces     -> E1_DRIVER_RUNTIME_UNBOUND
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import executable_candidates as EX
from dicode.teachers.e1_formal import probe_result_binding as PRB
from dicode.teachers.e1_formal import runtime_bundle as RB
from dicode.teachers.e1_formal import one_window_driver as DRV
from dicode.teachers.e1_formal.canonical import canonical_sha256
from dicode.teachers.e1_formal import board
from dicode.teachers.e1_formal.envcoder import EnvCoderArtifact
from dicode.teachers.e1_formal.manifest import BOARD_ROLE_ORDER
from dicode.teachers.e1_formal.task_specs import TaskSpec

# SYNTHETIC window identities — TEST_ONLY placeholders
_STUDENT_IDENTITY = "11" * 32
_STUDENT_CHECKPOINT = "12" * 32
_REFERENCE_IDENTITY = "21" * 32
_REFERENCE_CHECKPOINT = "22" * 32
_SEED_BANK = "77" * 32
_RESET_PROTOCOL = "88" * 32
_RUNNER_ID = "test-only-probe-runner-registry"
_RUNNER_HASH = "99" * 32


# ---------------------------------------------------------------------------
# fixtures (all TEST_ONLY / SYNTHETIC)
# ---------------------------------------------------------------------------
def _window(window_id="e1-w000001"):
    role_results = tuple(
        (role, {"role": role, "note": "TEST_ONLY"})
        for role in BOARD_ROLE_ORDER
    )
    payload = board._window_payload(
        window_id,
        1,
        "FIRST_WINDOW",
        "e" * 64,
        "COMPLETE",
        "",
        role_results,
        (),
        (),
    )
    return board.ReviewWindow(
        window_id=window_id,
        session_idx=1,
        trigger_code="FIRST_WINDOW",
        evidence_hash="e" * 64,
        status="COMPLETE",
        void_code="",
        role_results=role_results,
        surviving_families=(),
        ignored_vetoes=(),
        window_hash=board._compute_window_hash(payload),
    )


def _candidate(variant=0):
    window = _window()
    template_hash = canonical_sha256({"family": "fam_a"})
    variant_params = (("difficulty:level", str(variant)),)
    spec_hash = canonical_sha256(
        {
            "window_hash": window.window_hash,
            "template_hash": template_hash,
            "variant": variant,
            "variant_params": [list(p) for p in variant_params],
        }
    )
    spec = TaskSpec(
        spec_id=f"spec-v{variant}",
        window_id=window.window_id,
        window_hash=window.window_hash,
        family_id="fam_a",
        variant=variant,
        description="TEST_ONLY synthetic spec",
        target_achievements=("test_only:target",),
        axis_changes=(),
        constant_axes=("test_only_axis",),
        scaffolding="TEST_ONLY",
        student_must_do="TEST_ONLY",
        template_hash=template_hash,
        template_artifact_id=f"{template_hash}::tpl",
        variant_params=variant_params,
        spec_hash=spec_hash,
        artifact_id=f"{spec_hash}::v{variant}",
    )
    artifact = EX.build_executable_environment_artifact(
        envcoder_artifact=EnvCoderArtifact(
            template_hash=template_hash,
            artifact_id=f"{template_hash}::tpl",
            env_code="# TEST_ONLY env code\n",
            prompt_envelope_hash="ab" * 32,
        ),
        family_id="fam_a",
        observation_action_abi_hash="a1" * 32,
        reward_contract_hash="b2" * 32,
        reset_protocol_hash="c3" * 32,
        seed_policy_hash="d4" * 32,
        backend_name="test-only-backend",
        stages_passed=("SYNTAX", "GUARDS"),
    )
    return EX.bind_executable_candidate(
        window=window, spec=spec, executable_artifact=artifact
    )


def _issue(candidate, **overrides):
    kwargs = dict(
        candidate=candidate,
        student_identity_hash=_STUDENT_IDENTITY,
        student_checkpoint_hash=_STUDENT_CHECKPOINT,
        reference_identity_hash=_REFERENCE_IDENTITY,
        reference_checkpoint_hash=_REFERENCE_CHECKPOINT,
        runner_registry_id=_RUNNER_ID,
        runner_registry_hash=_RUNNER_HASH,
        seed_bank_hash=_SEED_BANK,
        reset_protocol_id="test-only-reset-protocol-v1",
        reset_protocol_hash=_RESET_PROTOCOL,
        episodes_requested=3,
        episodes_completed=3,
        episodes_failed=0,
        simulator_transitions=384,
        aggregate_metrics={"success_rate": 0.5},
        uncertainty_ci={"ci95": [0.4, 0.6]},
        terminal_event_aggregates={"terminal_events": 3},
        signer_id=PRB.SYNTHETIC_TEST_ONLY_PROBE_SIGNER,
        test_only=True,
    )
    kwargs.update(overrides)
    return PRB.issue_candidate_probe_result(**kwargs)


def _consume(pool, candidates, **overrides):
    kwargs = dict(
        candidates=candidates,
        student_identity_hash=_STUDENT_IDENTITY,
        student_checkpoint_hash=_STUDENT_CHECKPOINT,
        reference_identity_hash=_REFERENCE_IDENTITY,
        reference_checkpoint_hash=_REFERENCE_CHECKPOINT,
        seed_bank_hash=_SEED_BANK,
        reset_protocol_hash=_RESET_PROTOCOL,
        runner_registry_hash=_RUNNER_HASH,
        ctx="test",
        allow_test_only=True,
    )
    kwargs.update(overrides)
    return PRB.consume_registry_signed_probe_results(pool, **kwargs)


# ---------------------------------------------------------------------------
# issuance
# ---------------------------------------------------------------------------
class TestIssuance:
    def test_issues_a_fully_bound_result(self):
        candidate = _candidate()
        result = _issue(candidate)
        assert result.candidate_id == candidate.candidate_id
        assert result.candidate_hash == candidate.candidate_hash
        assert result.executable_artifact_id == (
            candidate.executable_artifact_id
        )
        assert result.result_id.endswith("::probe")
        assert len(result.attestation_hash) == 64
        assert result.test_only is True
        assert result.signer_id == PRB.SYNTHETIC_TEST_ONLY_PROBE_SIGNER

    def test_partial_episodes_never_issued(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _issue(
                candidate,
                episodes_requested=3,
                episodes_completed=2,
                episodes_failed=0,
            )
        assert excinfo.value.code == PRB.PROBE_PARTIAL_EPISODES
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _issue(
                candidate,
                episodes_requested=3,
                episodes_completed=0,
                episodes_failed=3,
            )
        assert excinfo.value.code == PRB.PROBE_PARTIAL_EPISODES

    def test_bad_identity_hashes_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _issue(candidate, student_identity_hash="short")
        assert excinfo.value.code == PRB.PROBE_BAD_TYPE

    def test_non_candidate_refused(self):
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _issue({"candidate": "summary"})
        assert excinfo.value.code == PRB.PROBE_BAD_TYPE


# ---------------------------------------------------------------------------
# consumption happy path
# ---------------------------------------------------------------------------
class TestConsumption:
    def test_test_only_pool_consumes_with_the_explicit_gate(self):
        c0, c1 = _candidate(0), _candidate(1)
        pool = (_issue(c0), _issue(c1))
        verified = _consume(pool, (c0, c1))
        assert verified == pool

    def test_empty_pool_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((), (candidate,))
        assert excinfo.value.code == PRB.PROBE_POOL_EMPTY

    def test_empty_candidates_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((_issue(candidate),), ())
        assert excinfo.value.code == PRB.PROBE_POOL_EMPTY

    def test_non_result_entry_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(({"result": "summary"},), (candidate,))
        assert excinfo.value.code == PRB.PROBE_BAD_TYPE


# ---------------------------------------------------------------------------
# signer gating
# ---------------------------------------------------------------------------
class TestSignerGating:
    def test_registry_whitelist_is_empty_this_round(self):
        assert PRB.AUTHORIZED_PROBE_RESULT_SIGNERS == ()

    def test_test_only_result_refused_on_production_surface(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                allow_test_only=False,
            )
        assert excinfo.value.code == (
            PRB.PROBE_TEST_ONLY_SIGNER_REJECTED
        )

    def test_wrong_test_only_signer_refused(self):
        candidate = _candidate()
        result = _issue(candidate, signer_id="attacker-test-only-signer")
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((result,), (candidate,))
        assert excinfo.value.code == (
            PRB.PROBE_TEST_ONLY_SIGNER_REJECTED
        )

    def test_production_signer_unauthorized_this_round(self):
        candidate = _candidate()
        result = _issue(
            candidate,
            signer_id="would-be-registry-signer",
            test_only=False,
        )
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((result,), (candidate,), allow_test_only=False)
        assert excinfo.value.code == PRB.PROBE_SIGNER_UNAUTHORIZED


# ---------------------------------------------------------------------------
# tamper + binding matrix
# ---------------------------------------------------------------------------
class TestBindingMatrix:
    def test_attestation_tamper_refused(self):
        candidate = _candidate()
        result = replace(_issue(candidate), attestation_hash="f" * 64)
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((result,), (candidate,))
        assert excinfo.value.code == PRB.PROBE_HASH_MISMATCH

    def test_field_tamper_breaks_the_attestation(self):
        candidate = _candidate()
        for field in (
            "episodes_completed",
            "simulator_transitions",
            "candidate_id",
            "signer_id",
        ):
            result = replace(_issue(candidate), **{field: 9999
                                                   if field in (
                                                       "episodes_completed",
                                                       "simulator_transitions",
                                                   )
                                                   else "forged"})
            with pytest.raises(PRB.ProbeResultError) as excinfo:
                _consume((result,), (candidate,))
            assert excinfo.value.code == PRB.PROBE_HASH_MISMATCH

    def test_stale_result_refused(self):
        c0, c1 = _candidate(0), _candidate(1)
        stale = _issue(c1)  # bound to a candidate NOT in the pool
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((stale,), (c0,))
        assert excinfo.value.code == PRB.PROBE_STALE

    def test_wrong_student_identity_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                student_identity_hash="f0" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_STUDENT_MISMATCH

    def test_wrong_student_checkpoint_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                student_checkpoint_hash="f1" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_CHECKPOINT_MISMATCH

    def test_wrong_reference_identity_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                reference_identity_hash="f2" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_REFERENCE_MISMATCH

    def test_wrong_reference_checkpoint_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                reference_checkpoint_hash="f3" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_CHECKPOINT_MISMATCH

    def test_wrong_seed_bank_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                seed_bank_hash="f4" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_SEED_BANK_MISMATCH

    def test_wrong_reset_protocol_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                reset_protocol_hash="f5" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_RESET_PROTOCOL_MISMATCH

    def test_wrong_runner_registry_refused(self):
        candidate = _candidate()
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume(
                (_issue(candidate),),
                (candidate,),
                runner_registry_hash="f6" * 32,
            )
        assert excinfo.value.code == PRB.PROBE_CANDIDATE_MISMATCH

    def test_duplicate_results_refused(self):
        candidate = _candidate()
        result = _issue(candidate)
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((result, result), (candidate,))
        assert excinfo.value.code == PRB.PROBE_DUPLICATE

    def test_two_results_for_one_candidate_refused(self):
        candidate = _candidate()
        first = _issue(candidate)
        second = _issue(candidate, simulator_transitions=999)
        with pytest.raises(PRB.ProbeResultError) as excinfo:
            _consume((first, second), (candidate,))
        assert excinfo.value.code == PRB.PROBE_DUPLICATE

    def test_mock_runner_disguise_refused(self):
        candidate = _candidate()
        for runner_id in ("mock-runner-01", "REPLAY-RUNNER"):
            result = _issue(candidate, runner_registry_id=runner_id)
            with pytest.raises(PRB.ProbeResultError) as excinfo:
                _consume((result,), (candidate,))
            assert excinfo.value.code == PRB.PROBE_MOCK_RUNNER_DISGUISED


# ---------------------------------------------------------------------------
# driver stage: bundle-bound probe intake
# ---------------------------------------------------------------------------
def _probe_runner_bundle():
    """TEST_ONLY bundle whose probe_runner exposes the frozen seed
    bank + reset protocol surfaces."""
    return RB.build_test_only_runtime_bundle(
        source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
        capabilities={
            contract: SimpleNamespace(
                kind=contract,
                identity_id=f"test-only-{contract}",
                **(
                    {
                        "seed_bank_hash": _SEED_BANK,
                        "reset_protocol_hash": _RESET_PROTOCOL,
                    }
                    if contract == "probe_runner"
                    else {}
                ),
            )
            for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
        },
    )


def _committed_manager():
    import json
    import os

    import yaml

    from dicode.teachers.e1_formal import gen_manager as GM

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    with open(
        os.path.join(repo_root, "conf", "teacher", "e1_formal.yaml"),
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    with open(
        os.path.join(repo_root, "configs", "e1_formal_ued.yaml"),
        "r",
        encoding="utf-8",
    ) as handle:
        frozen = yaml.safe_load(handle)
    with open(
        os.path.join(
            repo_root,
            "configs",
            "e1_formal_ued_anchor_manifest.DRAFT.json",
        ),
        "r",
        encoding="utf-8",
    ) as handle:
        draft = json.load(handle)
    return GM.E1FormalGenManager(
        config, frozen_manifest=frozen, anchor_manifest_mapping=draft
    )


class TestDriverProbeStage:
    def test_consumes_bundle_bound_test_only_pool(self):
        bundle = _probe_runner_bundle()
        teacher = _committed_manager()
        candidates = (_candidate(0), _candidate(1))
        student_identity_hash = bundle.object_identity_hash(
            "student_identity"
        )
        reference_identity_hash = bundle.object_identity_hash(
            "reference_identity"
        )
        runner_registry_hash = bundle.object_identity_hash("probe_runner")
        results = tuple(
            PRB.issue_candidate_probe_result(
                candidate=candidate,
                student_identity_hash=student_identity_hash,
                student_checkpoint_hash=_STUDENT_CHECKPOINT,
                reference_identity_hash=reference_identity_hash,
                reference_checkpoint_hash=_REFERENCE_CHECKPOINT,
                runner_registry_id="test-only-probe-runner-registry",
                runner_registry_hash=runner_registry_hash,
                seed_bank_hash=_SEED_BANK,
                reset_protocol_id="test-only-reset-protocol-v1",
                reset_protocol_hash=_RESET_PROTOCOL,
                episodes_requested=2,
                episodes_completed=2,
                episodes_failed=0,
                simulator_transitions=256,
                aggregate_metrics={"success_rate": 0.5},
                uncertainty_ci={"ci95": [0.4, 0.6]},
                terminal_event_aggregates={"terminal_events": 2},
                signer_id=PRB.SYNTHETIC_TEST_ONLY_PROBE_SIGNER,
                test_only=True,
            )
            for candidate in candidates
        )
        verified = DRV.execute_real_candidate_probes(
            teacher,
            candidates,
            bundle,
            probe_results=results,
            student_checkpoint_identity=_STUDENT_CHECKPOINT,
            reference_checkpoint_identity=_REFERENCE_CHECKPOINT,
            allow_test_only=True,
        )
        assert verified == results

    def test_probe_runner_without_seed_surface_refused(self):
        # probe_runner object lacks seed_bank_hash => unbound
        bundle = RB.build_test_only_runtime_bundle(
            source_commit="TEST_ONLY_SYNTHETIC_SOURCE_COMMIT",
            capabilities={
                contract: SimpleNamespace(
                    kind=contract, identity_id=f"test-only-{contract}"
                )
                for contract in RB.RUNTIME_CAPABILITY_CONTRACTS
            },
        )
        teacher = _committed_manager()
        candidates = (_candidate(0),)
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_candidate_probes(
                teacher,
                candidates,
                bundle,
                probe_results=(),
                student_checkpoint_identity=_STUDENT_CHECKPOINT,
                reference_checkpoint_identity=_REFERENCE_CHECKPOINT,
                allow_test_only=True,
            )
        assert excinfo.value.code == DRV.E1_DRIVER_RUNTIME_UNBOUND

    def test_empty_candidate_pool_refused(self):
        bundle = _probe_runner_bundle()
        teacher = _committed_manager()
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_candidate_probes(
                teacher,
                (),
                bundle,
                probe_results=(),
                student_checkpoint_identity=_STUDENT_CHECKPOINT,
                reference_checkpoint_identity=_REFERENCE_CHECKPOINT,
                allow_test_only=True,
            )
        assert excinfo.value.code == DRV.E1_DRIVER_MISSING_OBJECT

    def test_bad_checkpoint_identity_refused(self):
        bundle = _probe_runner_bundle()
        teacher = _committed_manager()
        with pytest.raises(DRV.DriverError) as excinfo:
            DRV.execute_real_candidate_probes(
                teacher,
                (_candidate(0),),
                bundle,
                probe_results=(),
                student_checkpoint_identity="short",
                reference_checkpoint_identity=_REFERENCE_CHECKPOINT,
                allow_test_only=True,
            )
        assert excinfo.value.code == DRV.E1_DRIVER_BAD_TYPE
