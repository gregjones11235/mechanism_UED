"""CC2 follow-up P0-14 tests: failure-pattern + curriculum-drift
producer seams.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
the producer registry is EMPTY this round (both seams honestly
UNBOUND); the data contracts and derivation are exercised with
conspicuously-marked synthetic objects.

Covered negative matrix:
* string producer registration             -> PRODUCER_BAD_TYPE
* unbound mint                             -> FAILURE_PATTERN_PRODUCER_UNBOUND /
                                              CURRICULUM_DRIFT_PRODUCER_UNBOUND
* caller-built mapping to a signal         -> PRODUCER_SIGNAL_BAD
* unfrozen threshold in derivation         -> PRODUCER_THRESHOLD_UNFROZEN
* gate states present + honest UNBOUND
"""
import pytest

from dicode.teachers.e1_formal import producer_seams as PS
from dicode.teachers.e1_formal import gate_signals as GS
from dicode.teachers.e1_formal.canonical import canonical_sha256


@pytest.fixture(autouse=True)
def _reset_producers():
    """The producer registry is module-global; reset between tests."""
    PS._FAILURE_PATTERN_PRODUCER = None
    PS._CURRICULUM_DRIFT_PRODUCER = None
    yield
    PS._FAILURE_PATTERN_PRODUCER = None
    PS._CURRICULUM_DRIFT_PRODUCER = None


def _fingerprint(novelty=0.9):
    return PS.FailurePatternFingerprint(
        behavior_clip_ids=("clip-1", "clip-2"),
        behavior_clip_hash="a1" * 32,
        detector_version=PS.FAILURE_PATTERN_DETECTOR_VERSION,
        failure_family="skill_regression",
        novelty=novelty,
        student_checkpoint_hash="b2" * 32,
        window_hash="c3" * 32,
        provenance_hash=canonical_sha256(
            {"kind": "TEST_ONLY_FINGERPRINT"}
        ),
    )


def _history(drift_metric=0.4):
    return PS.CurriculumCompositionHistory(
        prior_batch_hashes=("11" * 32, "22" * 32),
        family_composition=(("fam_a", 8), ("fam_b", 4)),
        bucket_composition=(("bucket_low", 6), ("bucket_high", 6)),
        anchor_share=0.25,
        drift_metric=drift_metric,
        student_checkpoint_progression=("33" * 32, "44" * 32),
        provenance_hash=canonical_sha256({"kind": "TEST_ONLY_HISTORY"}),
    )


class TestProducerStates:
    def test_states_exist_and_are_greppable(self):
        assert PS.INVOCATION_THRESHOLDS_UNFROZEN == (
            "INVOCATION_THRESHOLDS_UNFROZEN"
        )
        assert PS.FAILURE_PATTERN_PRODUCER_UNBOUND == (
            "FAILURE_PATTERN_PRODUCER_UNBOUND"
        )
        assert PS.CURRICULUM_DRIFT_PRODUCER_UNBOUND == (
            "CURRICULUM_DRIFT_PRODUCER_UNBOUND"
        )

    def test_gate_signals_carry_the_same_states(self):
        assert GS.FAILURE_PATTERN_PRODUCER_UNBOUND == (
            PS.FAILURE_PATTERN_PRODUCER_UNBOUND
        )
        assert GS.CURRICULUM_DRIFT_PRODUCER_UNBOUND == (
            PS.CURRICULUM_DRIFT_PRODUCER_UNBOUND
        )

    def test_both_producers_honestly_unbound_this_round(self):
        states = PS.producer_states()
        assert states["new_failure_pattern"] == (
            PS.FAILURE_PATTERN_PRODUCER_UNBOUND
        )
        assert states["curriculum_drift"] == (
            PS.CURRICULUM_DRIFT_PRODUCER_UNBOUND
        )

    def test_string_producer_registration_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.register_failure_pattern_producer(
                "dicode.shared_runtime.failure_pattern"
            )
        assert excinfo.value.code == PS.PRODUCER_BAD_TYPE
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.register_curriculum_drift_producer("producer-name")
        assert excinfo.value.code == PS.PRODUCER_BAD_TYPE


class TestMinting:
    def _fp_producer(self, **defaults):
        def producer(window_hash, student_checkpoint_hash, **inputs):
            return PS.FailurePatternFingerprint(
                behavior_clip_ids=inputs.get("behavior_clip_ids", ("c1",)),
                behavior_clip_hash=inputs.get(
                    "behavior_clip_hash", "a1" * 32
                ),
                detector_version=PS.FAILURE_PATTERN_DETECTOR_VERSION,
                failure_family=inputs.get("failure_family", "novel"),
                novelty=inputs.get("novelty", 0.9),
                student_checkpoint_hash=student_checkpoint_hash,
                window_hash=window_hash,
                provenance_hash=canonical_sha256(
                    {"kind": "TEST_ONLY_FINGERPRINT", **defaults}
                ),
            )

        return producer

    def test_unbound_failure_pattern_mint_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.mint_failure_pattern_fingerprint(
                window_hash="c3" * 32,
                student_checkpoint_hash="b2" * 32,
            )
        assert excinfo.value.code == (
            PS.FAILURE_PATTERN_PRODUCER_UNBOUND
        )

    def test_unbound_curriculum_mint_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.mint_curriculum_composition_history(
                prior_batch_hashes=("11" * 32,)
            )
        assert excinfo.value.code == (
            PS.CURRICULUM_DRIFT_PRODUCER_UNBOUND
        )

    def test_bound_failure_pattern_producer_mints(self):
        PS.register_failure_pattern_producer(self._fp_producer())
        assert PS.producer_states()["new_failure_pattern"] == "BOUND"
        fingerprint = PS.mint_failure_pattern_fingerprint(
            window_hash="c3" * 32,
            student_checkpoint_hash="b2" * 32,
            novelty=0.8,
        )
        assert isinstance(fingerprint, PS.FailurePatternFingerprint)
        assert fingerprint.novelty == 0.8
        assert fingerprint.window_hash == "c3" * 32

    def test_bound_curriculum_producer_mints(self):
        def producer(prior_batch_hashes, **inputs):
            return PS.CurriculumCompositionHistory(
                prior_batch_hashes=prior_batch_hashes,
                family_composition=inputs.get(
                    "family_composition", (("fam_a", 12),)
                ),
                bucket_composition=inputs.get(
                    "bucket_composition", (("bucket_low", 12),)
                ),
                anchor_share=inputs.get("anchor_share", 0.25),
                drift_metric=inputs.get("drift_metric", 0.3),
                student_checkpoint_progression=inputs.get(
                    "student_checkpoint_progression", ("33" * 32,)
                ),
                provenance_hash=canonical_sha256(
                    {"kind": "TEST_ONLY_HISTORY"}
                ),
            )

        PS.register_curriculum_drift_producer(producer)
        assert PS.producer_states()["curriculum_drift"] == "BOUND"
        history = PS.mint_curriculum_composition_history(
            prior_batch_hashes=("11" * 32, "22" * 32),
            drift_metric=0.55,
        )
        assert isinstance(history, PS.CurriculumCompositionHistory)
        assert history.drift_metric == 0.55
        assert history.prior_batch_hashes == ("11" * 32, "22" * 32)


class TestSignalDerivation:
    def test_failure_pattern_signal_triggered(self):
        derivation = PS.derive_failure_pattern_signal(
            _fingerprint(novelty=0.9), novelty_threshold=0.5
        )
        assert derivation["field"] == "new_failure_pattern"
        assert derivation["triggered"] is True
        assert len(derivation["fingerprint_hash"]) == 64

    def test_failure_pattern_signal_not_triggered(self):
        derivation = PS.derive_failure_pattern_signal(
            _fingerprint(novelty=0.3), novelty_threshold=0.5
        )
        assert derivation["triggered"] is False

    def test_failure_pattern_unfrozen_threshold_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.derive_failure_pattern_signal(
                _fingerprint(), novelty_threshold=None
            )
        assert excinfo.value.code == PS.PRODUCER_THRESHOLD_UNFROZEN
        assert PS.INVOCATION_THRESHOLDS_UNFROZEN in str(excinfo.value)

    def test_failure_pattern_caller_mapping_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.derive_failure_pattern_signal(
                {"novelty": 0.9}, novelty_threshold=0.5
            )
        assert excinfo.value.code == PS.PRODUCER_SIGNAL_BAD

    def test_curriculum_drift_signal_triggered(self):
        derivation = PS.derive_curriculum_drift_signal(
            _history(drift_metric=0.6), drift_threshold=0.5
        )
        assert derivation["field"] == "curriculum_drift"
        assert derivation["triggered"] is True
        assert len(derivation["history_hash"]) == 64

    def test_curriculum_drift_unfrozen_threshold_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.derive_curriculum_drift_signal(
                _history(), drift_threshold=None
            )
        assert excinfo.value.code == PS.PRODUCER_THRESHOLD_UNFROZEN

    def test_curriculum_drift_caller_mapping_refused(self):
        with pytest.raises(PS.ProducerSeamError) as excinfo:
            PS.derive_curriculum_drift_signal(
                {"drift_metric": 0.6}, drift_threshold=0.5
            )
        assert excinfo.value.code == PS.PRODUCER_SIGNAL_BAD
