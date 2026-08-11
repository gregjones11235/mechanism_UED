"""CC2 follow-up P0-2 tests: the authorized six-role LLM runtime.

TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION / NOT_SCIENTIFIC_EVIDENCE:
NO real API call happens here (or anywhere this round). The
production provider whitelist is EMPTY, so every PRODUCTION
authorization must fail closed; the TEST_ONLY runtime exercises the
contract with a conspicuously-marked synthetic fixture client.

Covered negative matrix:
* production provider (whitelist EMPTY)     -> PROVIDER_UNAUTHORIZED
* replay / mock provider                    -> PROVIDER_FORBIDDEN
* bad grant hash                            -> GRANT_BAD
* empty model_id / source_commit            -> BAD_TYPE
* string client_factory                     -> FACTORY_BAD_TYPE
* wrong role allowlist (count/order)        -> ROLE_ALLOWLIST_MISMATCH
* wrong prompt / schema version             -> VERSION_MISMATCH
* cap != 6                                  -> CAP_BAD
* bad retry / accounting policies           -> RETRY/ACCOUNTING_POLICY_BAD
* TEST_ONLY runtime on production surface   -> TEST_ONLY_REJECTED
* client: order / duplicate / unknown role / cap / window mismatch
* factory producing a client without query  -> CLIENT_BAD_SURFACE
* window invariants (6 calls, COMPLETE/VOID discipline)
"""
from types import SimpleNamespace

import pytest

from dicode.teachers.e1_formal import board_authorization as BA
from dicode.teachers.e1_formal.manifest import (
    BOARD_PROMPT_VERSION,
    BOARD_ROLE_ORDER,
    ROLE_OUTPUT_SCHEMA_VERSION,
)

#: TEST_ONLY / SYNTHETIC grant hash (never a supervisor grant)
_TEST_ONLY_GRANT = "a" * 64
_TEST_ONLY_COMMIT = "TEST_ONLY_SYNTHETIC_SOURCE_COMMIT"


# ---------------------------------------------------------------------------
# SYNTHETIC fixture client/factory — TEST_ONLY, never a real LLM.
# ---------------------------------------------------------------------------
class _FixtureClient:
    """TEST_ONLY / SYNTHETIC / NOT_REAL_EXECUTION reply source."""

    def __init__(self):
        self.queries = []

    def query(self, system_prompt, user_prompts, *, cache_key, role):
        self.queries.append((role, cache_key))
        return [{"content": f"TEST_ONLY fixture reply for {role}"}]


def _fixture_factory(model_id):
    assert model_id  # the authorization must thread the exact model id
    return _FixtureClient()


def _authorize(**overrides):
    kwargs = dict(
        mode=BA.SIX_ROLE_MODE_TEST_ONLY,
        authorization_grant_hash=_TEST_ONLY_GRANT,
        provider=BA.SYNTHETIC_TEST_ONLY_PROVIDER,
        model_id="TEST_ONLY_FIXTURE_MODEL_V1",
        client_factory=_fixture_factory,
        role_allowlist=BOARD_ROLE_ORDER,
        prompt_version=BOARD_PROMPT_VERSION,
        role_output_schema_version=ROLE_OUTPUT_SCHEMA_VERSION,
        source_commit=_TEST_ONLY_COMMIT,
        total_call_cap=6,
    )
    kwargs.update(overrides)
    return BA.authorize_six_role_runtime(**kwargs)


def _full_window_client(runtime, window_id="e1-w000001"):
    """Drive all six roles in the fixed order (TEST_ONLY fixture)."""
    client = runtime.make_client(window_id=window_id, context="test")
    for index, role in enumerate(BOARD_ROLE_ORDER):
        client.query(
            "system", ["user"], cache_key=f"key-{index}", role=role
        )
    return client


# ---------------------------------------------------------------------------
# production authorization fails closed this round
# ---------------------------------------------------------------------------
class TestProductionAuthorizationFailsClosed:
    def test_provider_whitelist_is_empty_this_round(self):
        assert BA.AUTHORIZED_SIX_ROLE_PROVIDERS == ()

    def test_any_production_provider_unauthorized(self):
        for provider in ("anthropic", "openai", "some-real-provider"):
            with pytest.raises(BA.BoardAuthorizationError) as excinfo:
                _authorize(
                    mode=BA.SIX_ROLE_MODE_PRODUCTION,
                    provider=provider,
                    model_id="claude-sonnet-4-5",
                )
            assert excinfo.value.code == BA.SIX_ROLE_PROVIDER_UNAUTHORIZED

    def test_replay_provider_forbidden_in_production(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(
                mode=BA.SIX_ROLE_MODE_PRODUCTION,
                provider="replay",
                model_id="e1-replay-mock-v1",
            )
        assert excinfo.value.code == BA.SIX_ROLE_PROVIDER_FORBIDDEN

    def test_mock_provider_forbidden_in_production(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(
                mode=BA.SIX_ROLE_MODE_PRODUCTION,
                provider="mock",
                model_id="any",
            )
        assert excinfo.value.code == BA.SIX_ROLE_PROVIDER_FORBIDDEN

    def test_bad_grant_hash_refused(self):
        for grant in ("", "abc", "z" * 64, 12345):
            with pytest.raises(BA.BoardAuthorizationError) as excinfo:
                _authorize(authorization_grant_hash=grant)
            assert excinfo.value.code in (
                BA.SIX_ROLE_GRANT_BAD,
                BA.SIX_ROLE_BAD_TYPE,
            )

    def test_empty_model_id_refused(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(model_id="")
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE

    def test_empty_source_commit_refused(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(source_commit="   ")
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE

    def test_string_client_factory_refused(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(client_factory="make_real_llm_client")
        assert excinfo.value.code == BA.SIX_ROLE_FACTORY_BAD_TYPE

    def test_wrong_role_allowlist_refused(self):
        for allowlist in (
            (),
            BOARD_ROLE_ORDER[:5],
            tuple(reversed(BOARD_ROLE_ORDER)),
            BOARD_ROLE_ORDER + ("envcoder",),
        ):
            with pytest.raises(BA.BoardAuthorizationError) as excinfo:
                _authorize(role_allowlist=allowlist)
            assert excinfo.value.code == (
                BA.SIX_ROLE_ROLE_ALLOWLIST_MISMATCH
            )

    def test_wrong_prompt_version_refused(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(prompt_version="e1-board-prompt-v1")
        assert excinfo.value.code == BA.SIX_ROLE_VERSION_MISMATCH

    def test_wrong_schema_version_refused(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(role_output_schema_version="e1-role-output-v0")
        assert excinfo.value.code == BA.SIX_ROLE_VERSION_MISMATCH

    def test_call_cap_must_be_exactly_six_this_round(self):
        for cap in (0, 5, 7, 12, True):
            with pytest.raises(BA.BoardAuthorizationError) as excinfo:
                _authorize(total_call_cap=cap)
            assert excinfo.value.code == BA.SIX_ROLE_CAP_BAD

    def test_unbounded_retry_policy_refused(self):
        for policy in (
            {"max_retries": 99, "retry_on": ()},
            {"max_retries": "3", "retry_on": ()},
            {"retries": 0},
            {"max_retries": True, "retry_on": ()},
        ):
            with pytest.raises(BA.BoardAuthorizationError) as excinfo:
                _authorize(retry_policy=policy)
            assert excinfo.value.code == BA.SIX_ROLE_RETRY_POLICY_BAD

    def test_wrong_accounting_policy_refused(self):
        for policy in (
            {"ledger": "some_other_ledger", "board_unit": "window"},
            {"ledger": BA.TOKEN_ACCOUNTING_LEDGER, "board_unit": "call"},
            {"ledger": BA.TOKEN_ACCOUNTING_LEDGER},
        ):
            with pytest.raises(BA.BoardAuthorizationError) as excinfo:
                _authorize(token_accounting_policy=policy)
            assert excinfo.value.code == (
                BA.SIX_ROLE_ACCOUNTING_POLICY_BAD
            )

    def test_unknown_mode_refused(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(mode="REAL")
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE


# ---------------------------------------------------------------------------
# TEST_ONLY authorization + identity binding
# ---------------------------------------------------------------------------
class TestTestOnlyAuthorization:
    def test_assembles_with_every_field_bound(self):
        runtime = _authorize()
        assert runtime.mode == BA.SIX_ROLE_MODE_TEST_ONLY
        assert runtime.provider == BA.SYNTHETIC_TEST_ONLY_PROVIDER
        assert runtime.model_id == "TEST_ONLY_FIXTURE_MODEL_V1"
        assert runtime.authorization_grant_hash == _TEST_ONLY_GRANT
        assert runtime.role_allowlist == BOARD_ROLE_ORDER
        assert runtime.prompt_version == BOARD_PROMPT_VERSION
        assert runtime.role_output_schema_version == (
            ROLE_OUTPUT_SCHEMA_VERSION
        )
        assert runtime.total_call_cap == 6
        assert len(runtime.client_factory_hash) == 64
        assert len(runtime.runtime_hash) == 64
        # fail-closed default retry: ZERO retries this round
        assert runtime.retry_policy_mapping["max_retries"] == 0
        assert runtime.token_accounting_policy_mapping["ledger"] == (
            BA.TOKEN_ACCOUNTING_LEDGER
        )

    def test_test_only_provider_is_mandatory_in_test_only_mode(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(provider="some-other-fixture")
        assert excinfo.value.code == BA.SIX_ROLE_PROVIDER_FORBIDDEN

    def test_test_only_model_id_must_be_conspicuously_marked(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            _authorize(model_id="fixture-model-v1")
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE

    def test_runtime_hash_binds_the_authorization_fields(self):
        base = _authorize()
        other = _authorize(source_commit="TEST_ONLY_OTHER_COMMIT")
        assert base.runtime_hash != other.runtime_hash
        grant = _authorize(authorization_grant_hash="b" * 64)
        assert base.runtime_hash != grant.runtime_hash

    def test_production_surface_refuses_test_only_runtime(self):
        runtime = _authorize()
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            BA.require_production_six_role_runtime(runtime, "test")
        assert excinfo.value.code == BA.SIX_ROLE_TEST_ONLY_REJECTED

    def test_production_surface_refuses_non_runtime(self):
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            BA.require_production_six_role_runtime(
                {"provider": "x"}, "test"
            )
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE

    def test_test_only_delta_is_zero(self):
        # TEST_ONLY fixtures never count as real execution
        assert BA.six_role_window_delta(_authorize()) == 0


# ---------------------------------------------------------------------------
# the window-scoped client (order / duplicates / cap / journal)
# ---------------------------------------------------------------------------
class TestWindowScopedClient:
    def test_six_in_order_calls_all_journaled(self):
        runtime = _authorize()
        client = _full_window_client(runtime)
        summary = client.window_call_summary()
        assert summary["logical_calls"] == 6
        assert summary["roles_called"] == BOARD_ROLE_ORDER
        assert summary["all_six_roles_called"] is True
        assert len(client.journal) == 6
        assert len(client.journal_hash) == 64
        # the journal binds the authorization identity on every entry
        for entry in client.journal:
            record = dict(entry)
            assert record["window_id"] == "e1-w000001"
            assert record["provider"] == (
                BA.SYNTHETIC_TEST_ONLY_PROVIDER
            )
            assert record["mode"] == BA.SIX_ROLE_MODE_TEST_ONLY

    def test_out_of_order_call_refused(self):
        runtime = _authorize()
        client = runtime.make_client(window_id="e1-w000001", context="test")
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            client.query("s", ["u"], cache_key="k0", role="critic")
        assert excinfo.value.code == BA.SIX_ROLE_ORDER_VIOLATION

    def test_unknown_role_refused(self):
        runtime = _authorize()
        client = runtime.make_client(window_id="e1-w000001", context="test")
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            client.query("s", ["u"], cache_key="k0", role="envcoder")
        assert excinfo.value.code == BA.SIX_ROLE_UNKNOWN_ROLE

    def test_duplicate_role_call_refused(self):
        runtime = _authorize()
        client = runtime.make_client(window_id="e1-w000001", context="test")
        client.query(
            "s", ["u"], cache_key="k0", role=BOARD_ROLE_ORDER[0]
        )
        client.query(
            "s", ["u"], cache_key="k1", role=BOARD_ROLE_ORDER[1]
        )
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            client.query(
                "s", ["u"], cache_key="k2", role=BOARD_ROLE_ORDER[0]
            )
        assert excinfo.value.code == BA.SIX_ROLE_DUPLICATE_CALL

    def test_seventh_call_exceeds_the_grant_cap(self):
        runtime = _authorize()
        client = _full_window_client(runtime)
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            client.query("s", ["u"], cache_key="k6", role="critic")
        assert excinfo.value.code == BA.SIX_ROLE_CALL_CAP_EXCEEDED

    def test_window_mismatch_refused(self):
        runtime = _authorize()
        client = runtime.make_client(window_id="e1-w000001", context="test")
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            client.query(
                "s",
                ["u"],
                cache_key="k0",
                role=BOARD_ROLE_ORDER[0],
                window_id="e1-w000002",
            )
        assert excinfo.value.code == BA.SIX_ROLE_WINDOW_MISMATCH

    def test_factory_without_query_surface_refused(self):
        def _bad_factory(model_id):
            return object()  # no query surface

        runtime = _authorize(client_factory=_bad_factory)
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            runtime.make_client(window_id="e1-w000001", context="test")
        assert excinfo.value.code == BA.SIX_ROLE_CLIENT_BAD_SURFACE


# ---------------------------------------------------------------------------
# six-role window invariants (COMPLETE/VOID discipline)
# ---------------------------------------------------------------------------
def _window(status, parsed_count):
    role_results = tuple(
        (role, {"ok": True} if index < parsed_count else None)
        for index, role in enumerate(BOARD_ROLE_ORDER)
    )
    return SimpleNamespace(status=status, role_results=role_results)


class TestWindowInvariants:
    def test_complete_window_with_six_parsed_outcomes_passes(self):
        runtime = _authorize()
        client = _full_window_client(runtime)
        BA.assert_six_role_window_invariants(
            runtime, client, _window("COMPLETE", 6), "test"
        )

    def test_complete_with_fewer_parsed_outcomes_refused(self):
        runtime = _authorize()
        client = _full_window_client(runtime)
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            BA.assert_six_role_window_invariants(
                runtime, client, _window("COMPLETE", 5), "test"
            )
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE

    def test_any_role_failure_means_void_never_complete(self):
        runtime = _authorize()
        client = _full_window_client(runtime)
        # VOID with five parsed outcomes is the honest shape
        BA.assert_six_role_window_invariants(
            runtime, client, _window("VOID", 5), "test"
        )
        # ...but the same window relabelled non-VOID is refused
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            BA.assert_six_role_window_invariants(
                runtime,
                client,
                _window("SOMETHING_ELSE", 5),
                "test",
            )
        assert excinfo.value.code == BA.SIX_ROLE_BAD_TYPE

    def test_partial_journal_refused(self):
        runtime = _authorize()
        client = runtime.make_client(window_id="e1-w000001", context="test")
        client.query(
            "s", ["u"], cache_key="k0", role=BOARD_ROLE_ORDER[0]
        )
        with pytest.raises(BA.BoardAuthorizationError) as excinfo:
            BA.assert_six_role_window_invariants(
                runtime, client, _window("COMPLETE", 6), "test"
            )
        assert excinfo.value.code == BA.SIX_ROLE_CALL_CAP_EXCEEDED
