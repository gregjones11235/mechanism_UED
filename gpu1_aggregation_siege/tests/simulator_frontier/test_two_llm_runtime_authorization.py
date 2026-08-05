# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

CC4 follow-up (P0-8): the 0-or-2 LLM decision runs through an
``AuthorizedTwoLLMRuntime`` carrying a MINT-ONLY authorization and a
tamper-evident call journal — never through a plain client factory and never
through a fake client.  The production path refuses an absent runtime, a
bare factory and any substitution.
"""

import dataclasses

import pytest

from dicode.simulator_frontier.errors import ProductionBlockedError
from dicode.simulator_frontier.invocation_gate import InvocationDecision
from dicode.simulator_frontier.llm_contracts import (
    LLMContractError,
    LLM_ROLE_SEQUENCE,
    REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT,
    TWO_LLM_CALL_CEILING,
    AuthorizedTwoLLMRuntime,
    CallJournal,
    mint_two_llm_authorization,
    run_two_llm_production,
    verify_two_llm_authorization,
)
from dicode.simulator_frontier.invocation_gate import InvocationReason

SYNTHETIC_FIXTURE_NOT_SCIENTIFIC_CONTENT = True


def _auth():
    return mint_two_llm_authorization(
        authorization_id="auth-001", authorizer_id="controller/cc4")


class TestAuthorizationMintOnly:
    def test_positive_mint_verifies(self):
        authorization = _auth()
        assert tuple(authorization.roles) == tuple(LLM_ROLE_SEQUENCE)
        assert int(authorization.max_logical_calls) == TWO_LLM_CALL_CEILING
        verify_two_llm_authorization(authorization)

    def test_mapping_authorization_rejected(self):
        with pytest.raises(LLMContractError):
            verify_two_llm_authorization({"authorization_hash": "f" * 64})

    def test_foreign_authorization_rejected(self):
        with pytest.raises(LLMContractError):
            verify_two_llm_authorization("authorization")

    def test_tampered_authorization_rejected(self):
        authorization = _auth()
        verify_two_llm_authorization(authorization)
        tampered = dataclasses.replace(authorization)
        object.__setattr__(tampered, "authorization_hash", "0" * 64)
        with pytest.raises(LLMContractError):
            verify_two_llm_authorization(tampered)


class TestAuthorizedRuntime:
    def test_runtime_requires_minted_authorization(self):
        with pytest.raises((LLMContractError, ProductionBlockedError)):
            AuthorizedTwoLLMRuntime(authorization={"spoof": 1},
                                    client_factory=lambda roles: {})

    def test_runtime_requires_a_real_client_factory(self):
        with pytest.raises(ProductionBlockedError) as exc:
            AuthorizedTwoLLMRuntime(authorization=_auth(), client_factory=None)
        assert REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT in str(exc.value)

    def test_runtime_rejects_a_fake_mapping_as_factory(self):
        with pytest.raises(ProductionBlockedError):
            AuthorizedTwoLLMRuntime(authorization=_auth(),
                                    client_factory={"diagnostician": "fake"})


class TestProductionPathNeverFallsBackToFake:
    def test_missing_runtime_blocks_with_named_reason(self):
        with pytest.raises(ProductionBlockedError) as exc:
            run_two_llm_production(None, {}, runtime=None, expected_state_id="s")
        assert REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT in str(exc.value)

    def test_bare_client_factory_is_refused(self):
        # A plain role->client mapping is NOT an AuthorizedTwoLLMRuntime: the
        # production path refuses it instead of calling it.
        def fake_factory(_roles):
            return {"diagnostician": object(), "planner": object()}
        with pytest.raises(ProductionBlockedError) as exc:
            run_two_llm_production(None, {}, runtime=fake_factory,
                                   expected_state_id="s")
        assert REAL_TWO_LLM_BLOCKED_NO_AUTHORIZED_CLIENT in str(exc.value)


class TestCallJournal:
    def test_ceiling_is_absolute(self):
        journal = CallJournal()
        journal.record(LLM_ROLE_SEQUENCE[0], input_hash="a" * 64, output_hash="b" * 64)
        journal.record(LLM_ROLE_SEQUENCE[1], input_hash="c" * 64, output_hash="d" * 64)
        with pytest.raises(LLMContractError):
            journal.record("DIAGNOSTICIAN", input_hash="e" * 64, output_hash="f" * 64)

    def test_role_order_is_binding(self):
        journal = CallJournal()
        with pytest.raises(LLMContractError):
            journal.record(LLM_ROLE_SEQUENCE[1], input_hash="a" * 64, output_hash="b" * 64)

    def test_chain_verifies_after_two_ordered_calls(self):
        journal = CallJournal()
        journal.record(LLM_ROLE_SEQUENCE[0], input_hash="a" * 64, output_hash="b" * 64)
        journal.record(LLM_ROLE_SEQUENCE[1], input_hash="c" * 64, output_hash="d" * 64)
        journal.verify()
        assert len(journal.entries) == 2


class TestZeroCallReuse:
    def test_zero_call_path_binds_evidence_hash_and_reuse_ref(self):
        authorization = _auth()
        runtime = AuthorizedTwoLLMRuntime(
            authorization=authorization,
            client_factory=lambda roles: {"diagnostician": object(),
                                          "planner": object()})
        decision = InvocationDecision(
            reason=InvocationReason.NO_SIGNIFICANT_CHANGE,
            llm_calls=0, reuse_plan_ref="plan-001", planned_roles=())
        result = runtime.execute(decision, {"state_id": "s"},
                                 expected_state_id="s")
        assert result["llm_calls"] == 0
        assert result["reuse_plan_ref"] == "plan-001"
        assert result["authorization_id"] == authorization.authorization_id
        assert result["journal"]["entries"] == ()
