# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-2): a VerifiedRestoreContext is minted ONLY from a fully
accepted fresh-process outcome + mechanically verified process evidence +
the controller injection slot.  The minter is controller-blocked this round,
so the honest contract to lock down in tests is its fail-closed surface:
mappings, foreign types, missing/rejected outcomes and tampered contexts are
all rejected — no self-reported restore context ever exists.
"""

import pytest

from dicode.simulator_frontier.errors import InvalidEvidenceError
from dicode.simulator_frontier.fresh_process_restore import (
    FreshProcessRestoreOutcome,
)
from dicode.simulator_frontier.verified_restore_context import (
    RESTORE_CONTEXT_DRIVER,
    VERIFIED_RESTORE_CONTEXT_SCHEMA,
    VERIFIER_HASH,
    VERIFIER_ID,
    VerifiedRestoreContext,
    compute_context_hash,
    mint_verified_restore_context,
    verify_verified_restore_context,
)
from dicode.simulator_frontier.fresh_process_restore import REQUIRED_COMPONENTS

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _rejected_outcome() -> FreshProcessRestoreOutcome:
    return FreshProcessRestoreOutcome(
        accepted=False,
        joint_proof_status="COMBINED_FRESH_PROCESS_RESTORE=false (fixture)",
        violations=("fixture-rejection",),
        child_pid=None,
        evidence=None,
    )


def _mint_kwargs(**overrides):
    kwargs = dict(
        restore_request={"schema": "fixture"},
        outcome=_rejected_outcome(),
        verdict=None,
        student_identity_hash="a" * 64,
        anchor_manifest_hash="b" * 64,
        state_id="fixture-state",
        state_hash="c" * 64,
        archive_hash="d" * 64,
        source_checkpoint_id="fixture-checkpoint",
        source_memory_spec_hash="e" * 64,
    )
    kwargs.update(overrides)
    return kwargs


class TestMintFailClosed:
    def test_mapping_restore_request_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            mint_verified_restore_context(**_mint_kwargs())

    def test_foreign_restore_request_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            mint_verified_restore_context(**_mint_kwargs(restore_request="request"))

    def test_missing_outcome_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            mint_verified_restore_context(**_mint_kwargs(outcome=None))

    def test_foreign_outcome_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            mint_verified_restore_context(**_mint_kwargs(outcome="outcome"))

    def test_rejected_outcome_can_never_mint(self):
        with pytest.raises(InvalidEvidenceError):
            mint_verified_restore_context(**_mint_kwargs())


class TestVerifyFailClosed:
    def test_mapping_context_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            verify_verified_restore_context({"context_hash": "f" * 64})

    def test_foreign_context_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            verify_verified_restore_context("context")

    def test_none_context_rejected(self):
        with pytest.raises(InvalidEvidenceError):
            verify_verified_restore_context(None)

    def test_tampered_context_rejected(self):
        # The context hash is mint-only: the minter sets it once from the
        # fields.  Mutating any field afterwards makes the recomputed hash
        # differ — verify must reject the tampered context fail closed.
        minted = _mint_fixture_context(_fixture_context())
        verify_verified_restore_context(minted)
        object.__setattr__(minted, "state_id", "tampered-state")
        with pytest.raises(InvalidEvidenceError):
            verify_verified_restore_context(minted)

    def test_wrong_state_hash_rejected_as_tamper(self):
        minted = _mint_fixture_context(_fixture_context())
        object.__setattr__(minted, "state_hash", "9" * 64)
        with pytest.raises(InvalidEvidenceError):
            verify_verified_restore_context(minted)


def _fixture_context() -> VerifiedRestoreContext:
    digests = {comp: f"{i + 1:02d}" * 32 for i, comp in enumerate(REQUIRED_COMPONENTS)}
    return VerifiedRestoreContext(
        schema_version=VERIFIED_RESTORE_CONTEXT_SCHEMA,
        restore_driver=RESTORE_CONTEXT_DRIVER,
        restore_request_hash="a" * 64,
        registry_bundle_hash="b" * 64,
        controller_signature_ref="controller-signature/fixture",
        child_pid=4242,
        child_ppid=1111,
        process_evidence_hash="c" * 64,
        verdict_hash="d" * 64,
        next_policy_step_replay_digest="e" * 64,
        component_digests=digests,
        student_identity_hash="f" * 64,
        checkpoint_manifest_hash="1" * 64,
        anchor_manifest_hash="2" * 64,
        formal_asset_registry_hash="3" * 64,
        state_id="fixture-state",
        state_hash="4" * 64,
        archive_hash="5" * 64,
        source_checkpoint_id="fixture-checkpoint",
        source_memory_spec_hash="6" * 64,
        verifier_id=VERIFIER_ID,
        verifier_hash=VERIFIER_HASH,
    )


def _mint_fixture_context(ctx: VerifiedRestoreContext) -> VerifiedRestoreContext:
    # TEST_ONLY: replicate exactly what the minter does (sets the two
    # init=False fields once, in this order).  This is a fixture, never a
    # production path.
    object.__setattr__(ctx, "production_joint_pass", True)
    object.__setattr__(ctx, "context_hash", compute_context_hash(ctx))
    return ctx
