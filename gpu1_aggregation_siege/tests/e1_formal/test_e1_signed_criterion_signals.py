"""CC2 follow-up P0-8 tests: signed criterion signals from probes.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
every probe/candidate/evidence fixture is synthetic; the production
signal-signer whitelist is EMPTY, so production minting must fail
closed. No selector ever trusts a caller-shaped signal here.

Covered negative matrix:
* aggregate metrics hash tamper            -> SIGNALS_HASH_MISMATCH
* missing metric / evidence source         -> SIGNALS_MISSING_SOURCE
* retention out of range                   -> SIGNALS_OUT_OF_RANGE
* cost episodes != completed episodes      -> SIGNALS_COST_MISMATCH
* probe/candidate binding swap             -> SIGNALS_PROBE_MISMATCH
* production signer (whitelist EMPTY)      -> SIGNALS_SIGNER_UNAUTHORIZED
* wrong TEST_ONLY signer                   -> SIGNALS_TEST_ONLY_REJECTED
* signal tamper / wrong verify sources     -> SIGNALS_HASH/CANDIDATE/
                                              PROBE_MISMATCH
* non-finite values                        -> SIGNALS_BAD_TYPE
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import executable_candidates as EX
from dicode.teachers.e1_formal import probe_result_binding as PRB
from dicode.teachers.e1_formal import signed_signals as SS
from dicode.teachers.e1_formal.canonical import canonical_sha256
from dicode.teachers.e1_formal.criterion_selector import (
    CRITERIA,
)
from dicode.teachers.e1_formal.schemas import (
    PROVENANCE_CANDIDATE_EVALUATION,
)
from dicode.teachers.e1_formal import board
from dicode.teachers.e1_formal.envcoder import EnvCoderArtifact
from dicode.teachers.e1_formal.manifest import BOARD_ROLE_ORDER
from dicode.teachers.e1_formal.task_specs import TaskSpec

_METRICS = {
    "front_regret": 0.62,
    "global_regret": 0.48,
    "behavioral_gap": 0.31,
    "learnability": 0.55,
    "learning_progress": 0.18,
}
_RETENTION = {"global_retention": 0.75}
_DIVERSITY = {"axis_count": 2, "pool_axis_max": 4}
_COST = {"episodes": 3}


# ---------------------------------------------------------------------------
# fixtures (all TEST_ONLY / SYNTHETIC)
# ---------------------------------------------------------------------------
def _window(window_id="e1-w000001"):
    role_results = tuple(
        (role, {"role": role, "note": "TEST_ONLY"})
        for role in BOARD_ROLE_ORDER
    )
    payload = board._window_payload(
        window_id, 1, "FIRST_WINDOW", "e" * 64,
        "COMPLETE", "", role_results, (), (),
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
    template_hash = canonical_sha256({"family": f"fam_{variant // 2}"})
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
        family_id=f"fam_{variant // 2}",
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
        family_id=f"fam_{variant // 2}",
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


def _probe(candidate, *, metrics=None):
    return PRB.issue_candidate_probe_result(
        candidate=candidate,
        student_identity_hash="11" * 32,
        student_checkpoint_hash="12" * 32,
        reference_identity_hash="21" * 32,
        reference_checkpoint_hash="22" * 32,
        runner_registry_id="test-only-probe-runner-registry",
        runner_registry_hash="99" * 32,
        seed_bank_hash="77" * 32,
        reset_protocol_id="test-only-reset-protocol-v1",
        reset_protocol_hash="88" * 32,
        episodes_requested=3,
        episodes_completed=3,
        episodes_failed=0,
        simulator_transitions=384,
        aggregate_metrics=metrics if metrics is not None else _METRICS,
        uncertainty_ci={"ci95": [0.4, 0.6]},
        terminal_event_aggregates={"terminal_events": 3},
        signer_id=PRB.SYNTHETIC_TEST_ONLY_PROBE_SIGNER,
        test_only=True,
    )


def _derive(candidate, probe, **overrides):
    kwargs = dict(
        probe_result=probe,
        candidate=candidate,
        aggregate_metrics=_METRICS,
        retention_evidence=_RETENTION,
        diversity_evidence=_DIVERSITY,
        cost_evidence=_COST,
        signer_id=SS.SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER,
        test_only=True,
    )
    kwargs.update(overrides)
    return SS.derive_criterion_signals_from_probe_result(**kwargs)


# ---------------------------------------------------------------------------
# minting
# ---------------------------------------------------------------------------
class TestMinting:
    def test_mints_all_eight_criteria_signed(self):
        candidate = _candidate()
        probe = _probe(candidate)
        signed = _derive(candidate, probe)
        assert signed.candidate_id == candidate.candidate_id
        assert signed.candidate_hash == candidate.candidate_hash
        assert signed.family_id == candidate.family_id
        assert signed.probe_result_id == probe.result_id
        assert signed.probe_result_hash == probe.attestation_hash
        assert signed.student_identity_hash == (
            probe.student_identity_hash
        )
        assert set(dict(signed.values)) == set(CRITERIA)
        assert len(signed.signal_hash) == 64
        assert signed.derivation_version == SS.SIGNAL_DERIVATION_VERSION
        assert signed.test_only is True
        evidence_hashes = dict(signed.input_hashes)
        assert evidence_hashes["aggregate_metrics_hash"] == (
            probe.aggregate_metrics_hash
        )

    def test_selector_view_is_real_probe_backed(self):
        candidate = _candidate()
        signed = _derive(candidate, _probe(candidate))
        view = signed.to_criterion_signals()
        assert view.candidate_id == candidate.candidate_id
        assert view.has_real_probe is True
        assert view.provenance == PROVENANCE_CANDIDATE_EVALUATION
        assert view.values_dict() == signed.values_dict()

    def test_aggregate_metrics_tamper_refused(self):
        candidate = _candidate()
        probe = _probe(candidate)
        forged = dict(_METRICS)
        forged["front_regret"] = 0.99  # not what the probe signed
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(candidate, probe, aggregate_metrics=forged)
        assert excinfo.value.code == SS.SIGNALS_HASH_MISMATCH

    def test_every_metric_source_is_mandatory(self):
        candidate = _candidate()
        probe = _probe(candidate)
        for criterion in (
            "front_regret",
            "global_regret",
            "behavioral_gap",
            "learnability",
            "learning_progress",
        ):
            metrics = dict(_METRICS)
            del metrics[criterion]
            tampered_probe = _probe(candidate, metrics=metrics)
            with pytest.raises(SS.SignedSignalsError) as excinfo:
                _derive(
                    candidate,
                    tampered_probe,
                    aggregate_metrics=metrics,
                )
            assert excinfo.value.code == SS.SIGNALS_MISSING_SOURCE

    def test_retention_evidence_is_mandatory(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(candidate, _probe(candidate), retention_evidence={})
        assert excinfo.value.code == SS.SIGNALS_MISSING_SOURCE

    def test_diversity_evidence_is_mandatory(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(
                candidate,
                _probe(candidate),
                diversity_evidence={"axis_count": 1},
            )
        assert excinfo.value.code == SS.SIGNALS_MISSING_SOURCE

    def test_cost_evidence_is_mandatory(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(candidate, _probe(candidate), cost_evidence={})
        assert excinfo.value.code == SS.SIGNALS_MISSING_SOURCE

    def test_retention_out_of_range_refused(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(
                candidate,
                _probe(candidate),
                retention_evidence={"global_retention": 1.5},
            )
        assert excinfo.value.code == SS.SIGNALS_OUT_OF_RANGE

    def test_cost_must_equal_completed_episodes(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(
                candidate,
                _probe(candidate),
                cost_evidence={"episodes": 99},
            )
        assert excinfo.value.code == SS.SIGNALS_COST_MISMATCH

    def test_probe_candidate_binding_enforced(self):
        candidate = _candidate(0)
        other_probe = _probe(_candidate(1))
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(candidate, other_probe)
        assert excinfo.value.code == SS.SIGNALS_PROBE_MISMATCH

    def test_non_finite_metric_never_obtains_a_signed_probe_hash(self):
        # defense in depth: the canonical encoding refuses non-finite
        # floats at probe ISSUANCE, so a non-finite metric can never
        # reach derivation with a valid signed hash
        from dicode.teachers.e1_formal.schemas import E1Code

        candidate = _candidate()
        metrics = dict(_METRICS)
        metrics["learnability"] = float("inf")
        with pytest.raises(Exception) as excinfo:
            _probe(candidate, metrics=metrics)
        assert getattr(excinfo.value, "code", "") == (
            E1Code.CANONICAL_UNSUPPORTED_TYPE
        )


class TestSignerGating:
    def test_signal_whitelist_is_empty_this_round(self):
        assert SS.AUTHORIZED_SIGNAL_SIGNERS == ()

    def test_production_signer_unauthorized(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(
                candidate,
                _probe(candidate),
                signer_id="would-be-signal-signer",
                test_only=False,
            )
        assert excinfo.value.code == SS.SIGNALS_SIGNER_UNAUTHORIZED

    def test_wrong_test_only_signer_refused(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            _derive(
                candidate,
                _probe(candidate),
                signer_id="attacker-signal-signer",
            )
        assert excinfo.value.code == SS.SIGNALS_TEST_ONLY_REJECTED


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
class TestVerification:
    def test_untampered_signals_verify(self):
        candidate = _candidate()
        probe = _probe(candidate)
        signed = _derive(candidate, probe)
        SS.verify_signed_criterion_signals(
            signed, candidate=candidate, probe_result=probe
        )

    def test_value_tamper_detected(self):
        candidate = _candidate()
        probe = _probe(candidate)
        signed = _derive(candidate, probe)
        tampered = replace(
            signed,
            values=tuple(
                (name, value + 1.0 if name == "learnability" else value)
                for name, value in signed.values
            ),
        )
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            SS.verify_signed_criterion_signals(
                tampered, candidate=candidate, probe_result=probe
            )
        assert excinfo.value.code == SS.SIGNALS_HASH_MISMATCH

    def test_wrong_candidate_detected(self):
        candidate = _candidate(0)
        probe = _probe(candidate)
        signed = _derive(candidate, probe)
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            SS.verify_signed_criterion_signals(
                signed,
                candidate=_candidate(1),
                probe_result=probe,
            )
        assert excinfo.value.code == SS.SIGNALS_CANDIDATE_MISMATCH

    def test_wrong_probe_detected(self):
        candidate = _candidate(0)
        probe = _probe(candidate)
        signed = _derive(candidate, probe)
        other_probe = _probe(candidate, metrics={
            **_METRICS, "front_regret": 0.1,
        })
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            SS.verify_signed_criterion_signals(
                signed, candidate=candidate, probe_result=other_probe
            )
        assert excinfo.value.code == SS.SIGNALS_PROBE_MISMATCH

    def test_non_signed_object_refused(self):
        candidate = _candidate()
        with pytest.raises(SS.SignedSignalsError) as excinfo:
            SS.verify_signed_criterion_signals(
                {"signals": "summary"},
                candidate=candidate,
                probe_result=_probe(candidate),
            )
        assert excinfo.value.code == SS.SIGNALS_BAD_TYPE
