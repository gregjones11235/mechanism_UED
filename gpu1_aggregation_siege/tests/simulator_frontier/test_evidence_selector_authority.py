# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-7): the evidence selector is the OFFICIAL final authority,
and the selection evidence is MINTED straight from attested branch outcomes
— every measured quantity is recomputed inside the minter, never supplied.
Unsigned/tampered evidence, unknown sources and mixed memory statuses fail
closed, and the selector refuses arbitrary mappings.
"""

import dataclasses

import pytest

from dicode.simulator_frontier.branch_search_runner import (
    SEARCH_SOURCE_STUDENT_DETERMINISTIC,
    SEARCH_SOURCE_STUDENT_STOCHASTIC,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.evidence_selector import (
    MIN_BUCKET_DIVERSITY,
    SELECTION_EVIDENCE_SCHEMA,
    SELECTOR_VERSION,
    evidence_based_select,
    mint_selection_evidence_from_outcomes,
    verify_selection_evidence,
)
from dicode.simulator_frontier.feasibility_classifier import FrontierClass
from dicode.simulator_frontier.search_statistics import BranchOutcome

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _outcome(branch_id: str, source: str, success: bool, progress: float) -> BranchOutcome:
    return BranchOutcome(
        branch_id=branch_id, state_id="s", search_source=source,
        rng_seed=0, horizon=8, transitions_used=4, success=success,
        progress=progress, terminal_event=None,
        failure_category=None if success else "HORIZON_EXHAUSTED",
        memory_mode="SAVED_POLICY_MEMORY", outcome_hash="a" * 64,
        memory_compatibility_status="SAVED_POLICY_MEMORY_VERIFIED")


_DEFAULT_OUTCOMES = (
    _outcome("b0", SEARCH_SOURCE_STUDENT_DETERMINISTIC, True, 0.7),
    _outcome("b1", SEARCH_SOURCE_STUDENT_STOCHASTIC, False, 0.2),
)


def _mint(outcomes=None):
    return mint_selection_evidence_from_outcomes(
        state_id="s",
        frontier_class=FrontierClass.LEARNABLE_FRONTIER,
        outcomes=_DEFAULT_OUTCOMES if outcomes is None else outcomes,
        retention_ok=True,
        anchor_coverage_ok=True,
        bucket_diversity=MIN_BUCKET_DIVERSITY,
    )


class TestSelectionEvidenceMint:
    def test_positive_mint_is_verified(self):
        evidence = _mint()
        assert evidence.evidence_schema == SELECTION_EVIDENCE_SCHEMA
        verify_selection_evidence(evidence)
        # Every measured quantity is recomputed by the minter.
        assert int(evidence.actual_n) == 2
        assert int(evidence.student_actual_n) == 2
        assert float(evidence.student_success_rate) == 0.5
        assert int(evidence.transition_cost) == 8

    def test_empty_outcomes_refused(self):
        with pytest.raises(InvalidEvidenceError):
            _mint(outcomes=[])

    def test_student_only_requirement(self):
        from dicode.simulator_frontier.branch_search_runner import (
            SEARCH_SOURCE_REFERENCE_POLICY,
        )
        with pytest.raises(InvalidEvidenceError):
            _mint(outcomes=(
                _outcome("r0", SEARCH_SOURCE_REFERENCE_POLICY, True, 0.9),
            ))

    def test_unknown_source_refused(self):
        with pytest.raises(InvalidEvidenceError):
            _mint(outcomes=(
                _outcome("u0", "SOME_MADE_UP_SOURCE", True, 0.5),
            ))

    def test_mixed_memory_statuses_refused(self):
        rows = list(_mint_rows())
        rows[0] = dataclasses.replace(rows[0], memory_compatibility_status="OTHER_STATUS")
        with pytest.raises(InvalidEvidenceError):
            mint_selection_evidence_from_outcomes(
                state_id="s", frontier_class=FrontierClass.LEARNABLE_FRONTIER,
                outcomes=rows, retention_ok=True, anchor_coverage_ok=True,
                bucket_diversity=MIN_BUCKET_DIVERSITY)


def _mint_rows():
    return [
        _outcome("b0", SEARCH_SOURCE_STUDENT_DETERMINISTIC, True, 0.7),
        _outcome("b1", SEARCH_SOURCE_STUDENT_STOCHASTIC, False, 0.2),
    ]


class TestVerifySelectionEvidence:
    def test_mapping_evidence_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            verify_selection_evidence({"evidence_hash": "f" * 64})

    def test_foreign_evidence_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            verify_selection_evidence("evidence")

    def test_tampered_evidence_rejected(self):
        evidence = _mint()
        verify_selection_evidence(evidence)
        tampered = dataclasses.replace(evidence)
        object.__setattr__(tampered, "student_success_rate", 0.99)
        with pytest.raises(InvalidEvidenceError):
            verify_selection_evidence(tampered)


class TestSelectorRefusesMappings:
    def test_mapping_plan_refused(self):
        evidence = _mint()
        with pytest.raises(InvalidEvidenceError):
            evidence_based_select({"plan_id": "spoofed"}, evidence=evidence)

    def test_foreign_plan_refused(self):
        evidence = _mint()
        with pytest.raises(InvalidEvidenceError):
            evidence_based_select("plan", evidence=evidence)

    def test_tampered_evidence_refused_before_decision(self):
        tampered = dataclasses.replace(_mint())
        object.__setattr__(tampered, "bucket_diversity", 99)
        with pytest.raises(InvalidEvidenceError):
            evidence_based_select("plan", evidence=tampered)
