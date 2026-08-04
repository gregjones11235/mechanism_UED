"""CC2 follow-up P0-9 tests: attested criterion selection + GenManager
certification.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
every candidate/probe/signal fixture is synthetic (TEST_ONLY signers);
the production signal whitelist is EMPTY. Certification against the
committed teacher stays honestly non-trainable (real gates blocked).

Covered negative matrix:
* empty pools                              -> SELECTION_POOL_EMPTY
* caller-shaped mapping signals            -> SELECTION_BAD_TYPE
* orphan signal / missing probe            -> SIGNAL_ORPHAN / PROBE_MISSING
* signals not covering the pool            -> SELECTION_POOL_MISMATCH
* TEST_ONLY / production surface mixing    -> SELECTION_TEST_ONLY_MIXED
* attestation tamper                       -> SELECTION_ATTESTATION_TAMPERED
* pool drift at certification              -> SELECTION_POOL_MISMATCH
* wrong window                             -> SELECTION_WINDOW_MISMATCH
* selected count != 12                     -> SELECTION_BAD_COUNT
* probe checkpoint swap                    -> PROBE_BINDING_MISMATCH
"""
from dataclasses import replace

import pytest

from dicode.teachers.e1_formal import executable_candidates as EX
from dicode.teachers.e1_formal import probe_result_binding as PRB
from dicode.teachers.e1_formal import selection_attestation as SA
from dicode.teachers.e1_formal import signed_signals as SS
from dicode.teachers.e1_formal.canonical import canonical_sha256
from dicode.teachers.e1_formal.selector import CRITIC_HARD_VETO, STATUS_OK
from dicode.teachers.e1_formal import board
from dicode.teachers.e1_formal.envcoder import EnvCoderArtifact
from dicode.teachers.e1_formal.manifest import BOARD_ROLE_ORDER
from dicode.teachers.e1_formal.task_specs import TaskSpec


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


def _candidate(window, variant):
    family = f"fam_{variant // 2}"
    template_hash = canonical_sha256({"family": family})
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
        family_id=family,
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
        family_id=family,
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


def _probe(candidate):
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
        aggregate_metrics=_metrics_for(candidate.variant_index),
        uncertainty_ci={"ci95": [0.4, 0.6]},
        terminal_event_aggregates={"terminal_events": 3},
        signer_id=PRB.SYNTHETIC_TEST_ONLY_PROBE_SIGNER,
        test_only=True,
    )


def _metrics_for(variant):
    base = 0.1 * (variant + 1)
    return {
        "front_regret": min(0.95, base),
        "global_regret": min(0.95, base + 0.05),
        "behavioral_gap": min(0.95, base + 0.02),
        "learnability": min(0.95, base + 0.03),
        "learning_progress": min(0.95, base + 0.01),
    }


def _signal(candidate, probe):
    return SS.derive_criterion_signals_from_probe_result(
        probe_result=probe,
        candidate=candidate,
        aggregate_metrics=_metrics_for(candidate.variant_index),
        retention_evidence={"global_retention": 0.75},
        diversity_evidence={"axis_count": 2, "pool_axis_max": 4},
        cost_evidence={"episodes": 3},
        signer_id=SS.SYNTHETIC_TEST_ONLY_SIGNAL_SIGNER,
        test_only=True,
    )


def _pool(count=4):
    window = _window()
    candidates = tuple(_candidate(window, v) for v in range(count))
    probes = tuple(_probe(c) for c in candidates)
    signals = tuple(_signal(c, p) for c, p in zip(candidates, probes))
    return window, candidates, probes, signals


def _execute(window, cand, prb, sig, **overrides):
    kwargs = dict(
        window_id=window.window_id,
        window_hash=window.window_hash,
        candidates=cand,
        probe_results=prb,
        signed_signals=sig,
        k=len(cand),
        seed=7,
        critic_policy=CRITIC_HARD_VETO,
        family_cap=len(cand),
        allow_test_only=True,
    )
    kwargs.update(overrides)
    return SA.execute_criterion_selection(**kwargs)


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


# ---------------------------------------------------------------------------
# selection execution
# ---------------------------------------------------------------------------
class TestExecuteCriterionSelection:
    def test_selects_with_full_attestation(self):
        window, candidates, probes, signals = _pool(4)
        outcome, attestation = _execute(
            window, candidates, probes, signals, family_cap=2
        )
        assert outcome.status == STATUS_OK
        assert len(outcome.selected_ids) == 4
        assert attestation.window_id == window.window_id
        assert attestation.window_hash == window.window_hash
        assert attestation.selected_ids == outcome.selected_ids
        expected = SA.compute_pool_hashes(candidates, probes, signals)
        assert attestation.candidate_pool_hash == expected[0]
        assert attestation.probe_pool_hash == expected[1]
        assert attestation.signals_pool_hash == expected[2]
        assert len(attestation.attestation_hash) == 64
        assert attestation.selected_set_hash == canonical_sha256(
            sorted(outcome.selected_ids)
        )
        # verification passes against the same pools
        SA.verify_selection_attestation(
            attestation,
            candidates=candidates,
            probe_results=probes,
            signed_signals=signals,
            window_hash=window.window_hash,
            ctx="test",
        )

    def test_empty_pools_refused(self):
        window, candidates, probes, signals = _pool(4)
        for overrides in (
            {"candidates": ()},
            {"probe_results": ()},
            {"signed_signals": ()},
        ):
            with pytest.raises(SA.SelectionAttestationError) as excinfo:
                _execute(
                    window, candidates, probes, signals, **overrides
                )
            assert excinfo.value.code == SA.SELECTION_POOL_EMPTY

    def test_caller_shaped_mapping_signals_refused(self):
        window, candidates, probes, signals = _pool(4)
        forged = list(signals)
        forged[0] = {
            "candidate_id": candidates[0].candidate_id,
            "has_real_probe": True,
        }
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            _execute(window, candidates, probes, tuple(forged))
        assert excinfo.value.code == SA.SELECTION_BAD_TYPE

    def test_orphan_signal_refused(self):
        window, candidates, probes, signals = _pool(4)
        other_window_candidates = _candidate(
            _window(window_id="e1-w000002"), 9
        )
        orphan = _signal(
            other_window_candidates, _probe(other_window_candidates)
        )
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            _execute(
                window, candidates, probes, signals + (orphan,)
            )
        assert excinfo.value.code == SA.SELECTION_SIGNAL_ORPHAN

    def test_missing_probe_for_candidate_refused(self):
        window, candidates, probes, signals = _pool(4)
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            _execute(window, candidates, probes[:3], signals)
        assert excinfo.value.code == SA.SELECTION_PROBE_MISSING

    def test_uncovered_candidates_refused(self):
        window, candidates, probes, signals = _pool(4)
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            _execute(window, candidates, probes, signals[:3])
        assert excinfo.value.code == SA.SELECTION_POOL_MISMATCH

    def test_test_only_signals_refused_on_production_surface(self):
        window, candidates, probes, signals = _pool(4)
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            _execute(
                window,
                candidates,
                probes,
                signals,
                allow_test_only=False,
            )
        assert excinfo.value.code == SA.SELECTION_TEST_ONLY_MIXED


# ---------------------------------------------------------------------------
# attestation verification
# ---------------------------------------------------------------------------
class TestAttestationVerification:
    def test_tampered_selected_ids_detected(self):
        window, candidates, probes, signals = _pool(4)
        _outcome, attestation = _execute(
            window, candidates, probes, signals
        )
        tampered = replace(
            attestation, selected_ids=("forged-id",)
        )
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            SA.verify_selection_attestation(
                tampered,
                candidates=candidates,
                probe_results=probes,
                signed_signals=signals,
                window_hash=window.window_hash,
                ctx="test",
            )
        assert excinfo.value.code == SA.SELECTION_ATTESTATION_TAMPERED

    def test_pool_drift_detected(self):
        window, candidates, probes, signals = _pool(4)
        _outcome, attestation = _execute(
            window, candidates, probes, signals
        )
        drifted_candidates = candidates[:3]
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            SA.verify_selection_attestation(
                attestation,
                candidates=drifted_candidates,
                probe_results=probes,
                signed_signals=signals,
                window_hash=window.window_hash,
                ctx="test",
            )
        assert excinfo.value.code == SA.SELECTION_POOL_MISMATCH

    def test_wrong_window_detected(self):
        window, candidates, probes, signals = _pool(4)
        _outcome, attestation = _execute(
            window, candidates, probes, signals
        )
        with pytest.raises(SA.SelectionAttestationError) as excinfo:
            SA.verify_selection_attestation(
                attestation,
                candidates=candidates,
                probe_results=probes,
                signed_signals=signals,
                window_hash="f" * 64,
                ctx="test",
            )
        assert excinfo.value.code == SA.SELECTION_WINDOW_MISMATCH


# ---------------------------------------------------------------------------
# GenManager certification (P0-9 mechanical checks + C13-C15 gates)
# ---------------------------------------------------------------------------
class TestGenManagerCertification:
    def _certify(self, manager, *, count=12, **overrides):
        window, candidates, probes, signals = _pool(count)
        outcome, attestation = _execute(
            window,
            candidates,
            probes,
            signals,
            family_cap=count,
            **{k: v for k, v in overrides.items()
               if k in ("k", "seed", "critic_policy", "family_cap",
                        "weights", "allow_test_only")},
        )
        kwargs = dict(
            selection_attestation=attestation,
            candidate_pool=candidates,
            probe_pool=probes,
            signals_pool=signals,
            window_hash=window.window_hash,
            student_checkpoint_hash="12" * 32,
            reference_checkpoint_hash="22" * 32,
        )
        kwargs.update(
            {k: v for k, v in overrides.items()
             if k in ("selection_attestation", "candidate_pool",
                      "probe_pool", "signals_pool", "window_hash",
                      "student_checkpoint_hash",
                      "reference_checkpoint_hash", "dual_probe")}
        )
        return manager.certify_and_build_training_batch(**kwargs)

    def test_exactly_twelve_required(self):
        manager = _committed_manager()
        # a 4-candidate selection certifies mechanically but violates
        # the 12-slot rule
        from dicode.teachers.e1_formal.gen_manager import (
            GenManagerError,
        )

        with pytest.raises(GenManagerError) as excinfo:
            self._certify(manager, count=4)
        assert excinfo.value.code == SA.SELECTION_BAD_COUNT

    def test_twelve_selection_certifies_then_hits_honest_gates(self):
        from dicode.teachers.e1_formal.gen_manager import (
            GEN_MANAGER_PROMOTION_BLOCKED,
            GenManagerError,
        )

        manager = _committed_manager()
        # the attestation certifies mechanically (all six P0-9 checks
        # pass), but the committed teacher's real gates (unfrozen
        # reference contract etc.) REFUSE promotion — this round stays
        # honestly non-trainable
        with pytest.raises(GenManagerError) as excinfo:
            self._certify(manager, count=12)
        assert excinfo.value.code == GEN_MANAGER_PROMOTION_BLOCKED

    def test_tampered_attestation_refused(self):
        from dicode.teachers.e1_formal.gen_manager import (
            GenManagerError,
        )

        manager = _committed_manager()
        window, candidates, probes, signals = _pool(12)
        _outcome, attestation = _execute(
            window, candidates, probes, signals, family_cap=12
        )
        tampered = replace(attestation, seed=999)
        with pytest.raises(GenManagerError) as excinfo:
            manager.certify_and_build_training_batch(
                selection_attestation=tampered,
                candidate_pool=candidates,
                probe_pool=probes,
                signals_pool=signals,
                window_hash=window.window_hash,
                student_checkpoint_hash="12" * 32,
                reference_checkpoint_hash="22" * 32,
            )
        assert excinfo.value.code == SA.SELECTION_ATTESTATION_TAMPERED

    def test_wrong_window_refused(self):
        from dicode.teachers.e1_formal.gen_manager import (
            GenManagerError,
        )

        manager = _committed_manager()
        with pytest.raises(GenManagerError) as excinfo:
            self._certify(manager, count=12, window_hash="f" * 64)
        assert excinfo.value.code == SA.SELECTION_WINDOW_MISMATCH

    def test_student_checkpoint_swap_refused(self):
        from dicode.teachers.e1_formal.gen_manager import (
            GenManagerError,
        )

        manager = _committed_manager()
        with pytest.raises(GenManagerError) as excinfo:
            self._certify(
                manager,
                count=12,
                student_checkpoint_hash="aa" * 32,  # different checkpoint
            )
        assert excinfo.value.code == SA.SELECTION_PROBE_BINDING_MISMATCH

    def test_reference_checkpoint_swap_refused(self):
        from dicode.teachers.e1_formal.gen_manager import (
            GenManagerError,
        )

        manager = _committed_manager()
        with pytest.raises(GenManagerError) as excinfo:
            self._certify(
                manager,
                count=12,
                reference_checkpoint_hash="bb" * 32,
            )
        assert excinfo.value.code == SA.SELECTION_PROBE_BINDING_MISMATCH

    def test_pool_drift_refused(self):
        from dicode.teachers.e1_formal.gen_manager import (
            GenManagerError,
        )

        manager = _committed_manager()
        window, candidates, probes, signals = _pool(12)
        _outcome, attestation = _execute(
            window, candidates, probes, signals, family_cap=12
        )
        with pytest.raises(GenManagerError) as excinfo:
            manager.certify_and_build_training_batch(
                selection_attestation=attestation,
                candidate_pool=candidates[:11],  # drifted pool
                probe_pool=probes,
                signals_pool=signals,
                window_hash=window.window_hash,
                student_checkpoint_hash="12" * 32,
                reference_checkpoint_hash="22" * 32,
            )
        assert excinfo.value.code == SA.SELECTION_POOL_MISMATCH
