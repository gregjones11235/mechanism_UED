"""CC2 follow-up P0-2: the authorized six-role LLM runtime.

The production Board receives its LLM client EXPLICITLY from an
``AuthorizedSixRoleLLMRuntime`` — never ``llm_client=None`` (which
silently falls back to ReplayLLMClient), never a bare provider
string, never a mock/replay client posing as real::

    runtime = authorize_six_role_runtime(mode=PRODUCTION, ...)
    client  = runtime.make_client(window_id=...)
    window  = run_review_board(client, ...)

Authorization is fail-closed on EVERY field:

* ``authorization_grant_hash`` — 64-hex supervisor grant;
* ``provider`` — must be on the supervisor-owned whitelist
  (``AUTHORIZED_SIX_ROLE_PROVIDERS``), EMPTY this round, so NO
  production six-role runtime can authorize yet (honest
  ``SIX_ROLE_PROVIDER_UNAUTHORIZED``); replay/mock providers are
  refused with their own code even if ever whitelisted;
* ``model_id`` — exact, non-empty (no "latest"/auto aliases);
* ``client_factory`` — a real callable (never a string); its identity
  is pinned as ``client_factory_hash``;
* ``role_allowlist`` — EXACTLY the six board roles in the fixed
  order; ``prompt_version`` / ``role_output_schema_version`` — EXACTLY
  the frozen board pins;
* ``retry_policy`` — bounded (max_retries 0..2; the default is 0:
  fail closed, no silent retry);
* ``token_accounting_policy`` — the E1 ledger, window-unit accounting;
* ``total_call_cap`` — EXACTLY 6 this round (one window = six logical
  calls per authorization grant).

Window invariants (mechanical, checked by
``assert_six_role_window_invariants``): one window = six roles in the
fixed order = six logical calls; a COMPLETE window carries six parsed
role outcomes; ANY role failure => VOID (never relabelled COMPLETE).
Under a TEST_ONLY fixture runtime (and under any replay/mock client)
the REAL delta is 0 — only an authorized PRODUCTION runtime
contributes delta 6.

This round performs NO real API calls: only authorization and
TEST_ONLY contract tests. The TEST_ONLY mode is conspicuously marked
and is refused by every production surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Tuple

from .canonical import canonical_sha256
from .manifest import (
    BOARD_PROMPT_VERSION,
    BOARD_ROLE_ORDER,
    E1_REPLAY_MODEL_ID,
    E1_REPLAY_PROVIDER,
    ROLE_OUTPUT_SCHEMA_VERSION,
)
from .schemas import E1SchemaError

#: authorization modes — the ONLY two values ever admitted
SIX_ROLE_MODE_PRODUCTION = "PRODUCTION"
SIX_ROLE_MODE_TEST_ONLY = "TEST_ONLY"

#: the synthetic provider/grant every TEST_ONLY runtime must carry
SYNTHETIC_TEST_ONLY_PROVIDER = "TEST_ONLY_SYNTHETIC_PROVIDER"
SYNTHETIC_TEST_ONLY_GRANT_SIGNER = "SYNTHETIC_TEST_ONLY_GRANT_SIGNER"

#: supervisor-owned production provider whitelist — EMPTY this round.
#: No paid/real LLM is authorized; nothing may call out.
AUTHORIZED_SIX_ROLE_PROVIDERS: Tuple[str, ...] = ()

#: providers that may NEVER pose as the production six-role runtime
FORBIDDEN_SIX_ROLE_PROVIDERS = frozenset(
    {E1_REPLAY_PROVIDER, "mock", E1_REPLAY_MODEL_ID}
)

#: one window = six logical board calls per authorization grant
SIX_ROLE_WINDOW_CALLS = 6

#: the frozen accounting policy surface (E1 ledger, window-unit)
TOKEN_ACCOUNTING_LEDGER = "e1_llm_call_ledger_v1"
TOKEN_ACCOUNTING_BOARD_UNIT = "window"

#: bounded retry surface (fail-closed default: no retry this round)
MAX_RETRY_POLICY_RETRIES = 2

# fail-closed codes (greppable)
SIX_ROLE_BAD_TYPE = "SIX_ROLE_BAD_TYPE"
SIX_ROLE_MISSING_FIELD = "SIX_ROLE_MISSING_FIELD"
SIX_ROLE_GRANT_BAD = "SIX_ROLE_GRANT_BAD"
SIX_ROLE_PROVIDER_UNAUTHORIZED = "SIX_ROLE_PROVIDER_UNAUTHORIZED"
SIX_ROLE_PROVIDER_FORBIDDEN = "SIX_ROLE_PROVIDER_FORBIDDEN"
SIX_ROLE_TEST_ONLY_REJECTED = "SIX_ROLE_TEST_ONLY_REJECTED"
SIX_ROLE_FACTORY_BAD_TYPE = "SIX_ROLE_FACTORY_BAD_TYPE"
SIX_ROLE_CLIENT_BAD_SURFACE = "SIX_ROLE_CLIENT_BAD_SURFACE"
SIX_ROLE_ROLE_ALLOWLIST_MISMATCH = "SIX_ROLE_ROLE_ALLOWLIST_MISMATCH"
SIX_ROLE_VERSION_MISMATCH = "SIX_ROLE_VERSION_MISMATCH"
SIX_ROLE_CAP_BAD = "SIX_ROLE_CAP_BAD"
SIX_ROLE_RETRY_POLICY_BAD = "SIX_ROLE_RETRY_POLICY_BAD"
SIX_ROLE_ACCOUNTING_POLICY_BAD = "SIX_ROLE_ACCOUNTING_POLICY_BAD"
SIX_ROLE_UNKNOWN_ROLE = "SIX_ROLE_UNKNOWN_ROLE"
SIX_ROLE_ORDER_VIOLATION = "SIX_ROLE_ORDER_VIOLATION"
SIX_ROLE_DUPLICATE_CALL = "SIX_ROLE_DUPLICATE_CALL"
SIX_ROLE_CALL_CAP_EXCEEDED = "SIX_ROLE_CALL_CAP_EXCEEDED"
SIX_ROLE_WINDOW_MISMATCH = "SIX_ROLE_WINDOW_MISMATCH"


class BoardAuthorizationError(E1SchemaError):
    """Fail-closed six-role authorization violation; ``code`` is
    greppable."""


def _require_non_empty_str(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BoardAuthorizationError(
            SIX_ROLE_BAD_TYPE,
            f"{ctx}: {name} must be a non-empty str, got {value!r}",
        )
    return value.strip()


def _require_sha64(value: Any, name: str, ctx: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise BoardAuthorizationError(
            SIX_ROLE_GRANT_BAD,
            f"{ctx}: {name} must be a 64-hex hash, got {value!r}",
        )
    try:
        int(value, 16)
    except ValueError:
        raise BoardAuthorizationError(
            SIX_ROLE_GRANT_BAD,
            f"{ctx}: {name} is not hexadecimal: {value!r}",
        )
    return value


def _freeze_mapping(mapping: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((key, mapping[key]) for key in mapping))


def _client_factory_hash(factory: Callable[..., Any]) -> str:
    return canonical_sha256(
        {
            "factory_module": getattr(factory, "__module__", ""),
            "factory_qualname": getattr(factory, "__qualname__", repr(factory)),
        }
    )


@dataclass(frozen=True)
class AuthorizedSixRoleLLMRuntime:
    """The explicit authorization record for the six-role board.

    ``client_factory`` is the ONLY surface that may construct the
    underlying client; it is bound by identity
    (``client_factory_hash``), never by string name. ``runtime_hash``
    binds every authorization field (tamper-evident).
    """

    mode: str  # PRODUCTION | TEST_ONLY
    authorization_grant_hash: str
    provider: str
    model_id: str
    client_factory_hash: str
    role_allowlist: Tuple[str, ...]
    prompt_version: str
    role_output_schema_version: str
    retry_policy: Tuple[Tuple[str, Any], ...]
    token_accounting_policy: Tuple[Tuple[str, Any], ...]
    total_call_cap: int
    source_commit: str
    runtime_hash: str
    client_factory: Callable[..., Any]

    def make_client(self, *, window_id: str, context: str) -> "AuthorizedSixRoleClient":
        """Bind ONE window-scoped client under this authorization.

        The underlying client is constructed by the authorized
        ``client_factory(model_id)`` and must expose the board's
        ``query`` surface. Role order, duplicates and the call cap are
        enforced mechanically on every call.
        """
        _require_non_empty_str(window_id, "window_id", context)
        underlying = self.client_factory(self.model_id)
        if not callable(getattr(underlying, "query", None)):
            raise BoardAuthorizationError(
                SIX_ROLE_CLIENT_BAD_SURFACE,
                f"{context}: the authorized client factory produced a "
                f"{type(underlying).__name__} without a callable query "
                "surface; the board client contract is mandatory",
            )
        return AuthorizedSixRoleClient(
            runtime=self, window_id=window_id, underlying=underlying
        )

    @property
    def retry_policy_mapping(self) -> Dict[str, Any]:
        return dict(self.retry_policy)

    @property
    def token_accounting_policy_mapping(self) -> Dict[str, Any]:
        return dict(self.token_accounting_policy)


def _compute_runtime_hash(
    *,
    mode: str,
    authorization_grant_hash: str,
    provider: str,
    model_id: str,
    client_factory_hash: str,
    role_allowlist: Tuple[str, ...],
    prompt_version: str,
    role_output_schema_version: str,
    retry_policy: Tuple[Tuple[str, Any], ...],
    token_accounting_policy: Tuple[Tuple[str, Any], ...],
    total_call_cap: int,
    source_commit: str,
) -> str:
    return canonical_sha256(
        {
            "mode": mode,
            "authorization_grant_hash": authorization_grant_hash,
            "provider": provider,
            "model_id": model_id,
            "client_factory_hash": client_factory_hash,
            "role_allowlist": list(role_allowlist),
            "prompt_version": prompt_version,
            "role_output_schema_version": role_output_schema_version,
            "retry_policy": [list(pair) for pair in retry_policy],
            "token_accounting_policy": [
                list(pair) for pair in token_accounting_policy
            ],
            "total_call_cap": total_call_cap,
            "source_commit": source_commit,
        }
    )


def _validate_retry_policy(policy: Any, ctx: str) -> Tuple[Tuple[str, Any], ...]:
    if policy is None:
        policy = {"max_retries": 0, "retry_on": ()}
    if not isinstance(policy, Mapping) or set(policy) != {
        "max_retries",
        "retry_on",
    }:
        raise BoardAuthorizationError(
            SIX_ROLE_RETRY_POLICY_BAD,
            f"{ctx}: retry_policy must carry exactly {{max_retries, "
            f"retry_on}}, got {policy!r}",
        )
    max_retries = policy["max_retries"]
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise BoardAuthorizationError(
            SIX_ROLE_RETRY_POLICY_BAD,
            f"{ctx}: max_retries must be an int, got {max_retries!r}",
        )
    if not 0 <= max_retries <= MAX_RETRY_POLICY_RETRIES:
        raise BoardAuthorizationError(
            SIX_ROLE_RETRY_POLICY_BAD,
            f"{ctx}: max_retries must be within "
            f"[0, {MAX_RETRY_POLICY_RETRIES}] (fail-closed bounded "
            f"retry), got {max_retries}",
        )
    retry_on = policy["retry_on"]
    if not isinstance(retry_on, (tuple, list)) or not all(
        isinstance(code, str) for code in retry_on
    ):
        raise BoardAuthorizationError(
            SIX_ROLE_RETRY_POLICY_BAD,
            f"{ctx}: retry_on must be a sequence of error-code strings, "
            f"got {retry_on!r}",
        )
    return _freeze_mapping(
        {"max_retries": max_retries, "retry_on": tuple(retry_on)}
    )


def _validate_accounting_policy(
    policy: Any, ctx: str
) -> Tuple[Tuple[str, Any], ...]:
    if policy is None:
        policy = {
            "ledger": TOKEN_ACCOUNTING_LEDGER,
            "board_unit": TOKEN_ACCOUNTING_BOARD_UNIT,
        }
    if not isinstance(policy, Mapping) or set(policy) != {
        "ledger",
        "board_unit",
    }:
        raise BoardAuthorizationError(
            SIX_ROLE_ACCOUNTING_POLICY_BAD,
            f"{ctx}: token_accounting_policy must carry exactly "
            f"{{ledger, board_unit}}, got {policy!r}",
        )
    if policy["ledger"] != TOKEN_ACCOUNTING_LEDGER:
        raise BoardAuthorizationError(
            SIX_ROLE_ACCOUNTING_POLICY_BAD,
            f"{ctx}: token accounting must use the E1 ledger "
            f"{TOKEN_ACCOUNTING_LEDGER!r}, got {policy['ledger']!r}",
        )
    if policy["board_unit"] != TOKEN_ACCOUNTING_BOARD_UNIT:
        raise BoardAuthorizationError(
            SIX_ROLE_ACCOUNTING_POLICY_BAD,
            f"{ctx}: board accounting unit must be "
            f"{TOKEN_ACCOUNTING_BOARD_UNIT!r}, got {policy['board_unit']!r}",
        )
    return _freeze_mapping(policy)


def authorize_six_role_runtime(
    *,
    mode: str,
    authorization_grant_hash: str,
    provider: str,
    model_id: str,
    client_factory: Any,
    role_allowlist: Any,
    prompt_version: str,
    role_output_schema_version: str,
    source_commit: str,
    total_call_cap: int = SIX_ROLE_WINDOW_CALLS,
    retry_policy: Any = None,
    token_accounting_policy: Any = None,
) -> AuthorizedSixRoleLLMRuntime:
    """Authorize the six-role runtime fail-closed on EVERY field.

    This round: ``AUTHORIZED_SIX_ROLE_PROVIDERS`` is EMPTY, so every
    PRODUCTION authorization fails honestly; only the conspicuously
    marked TEST_ONLY mode can authorize (contract tests / the
    TEST_ONLY closed loop), and no production surface accepts it.
    """
    ctx = "board_authorization.authorize"
    if mode not in (SIX_ROLE_MODE_PRODUCTION, SIX_ROLE_MODE_TEST_ONLY):
        raise BoardAuthorizationError(
            SIX_ROLE_BAD_TYPE,
            f"{ctx}: mode must be one of {[SIX_ROLE_MODE_PRODUCTION, SIX_ROLE_MODE_TEST_ONLY]}, got {mode!r}",
        )
    grant_hash = _require_sha64(
        authorization_grant_hash, "authorization_grant_hash", ctx
    )
    provider = _require_non_empty_str(provider, "provider", ctx)
    model_id = _require_non_empty_str(model_id, "model_id", ctx)
    if mode == SIX_ROLE_MODE_PRODUCTION:
        if provider in FORBIDDEN_SIX_ROLE_PROVIDERS:
            raise BoardAuthorizationError(
                SIX_ROLE_PROVIDER_FORBIDDEN,
                f"{ctx}: provider {provider!r} is a replay/mock "
                "identity; replay and mock clients may NEVER serve the "
                "production six-role board (no silent downgrade)",
            )
        if provider not in AUTHORIZED_SIX_ROLE_PROVIDERS:
            raise BoardAuthorizationError(
                SIX_ROLE_PROVIDER_UNAUTHORIZED,
                f"{ctx}: provider {provider!r} is not on the "
                "supervisor-owned six-role whitelist (EMPTY this "
                "round); no real LLM call is authorized",
            )
    else:  # TEST_ONLY
        if provider != SYNTHETIC_TEST_ONLY_PROVIDER:
            raise BoardAuthorizationError(
                SIX_ROLE_PROVIDER_FORBIDDEN,
                f"{ctx}: TEST_ONLY runtimes must use "
                f"{SYNTHETIC_TEST_ONLY_PROVIDER!r}, got {provider!r}",
            )
        if not model_id.startswith("TEST_ONLY_"):
            raise BoardAuthorizationError(
                SIX_ROLE_BAD_TYPE,
                f"{ctx}: TEST_ONLY model_id must be conspicuously "
                f"marked (prefix TEST_ONLY_), got {model_id!r}",
            )
    if isinstance(client_factory, str) or not callable(client_factory):
        raise BoardAuthorizationError(
            SIX_ROLE_FACTORY_BAD_TYPE,
            f"{ctx}: client_factory must be a real callable (never a "
            f"string contract name), got {client_factory!r}",
        )
    if tuple(role_allowlist or ()) != BOARD_ROLE_ORDER:
        raise BoardAuthorizationError(
            SIX_ROLE_ROLE_ALLOWLIST_MISMATCH,
            f"{ctx}: the role allowlist must be EXACTLY the six board "
            f"roles in the fixed order {list(BOARD_ROLE_ORDER)}, got "
            f"{list(role_allowlist or ())}",
        )
    if prompt_version != BOARD_PROMPT_VERSION:
        raise BoardAuthorizationError(
            SIX_ROLE_VERSION_MISMATCH,
            f"{ctx}: prompt_version must equal the frozen board pin "
            f"{BOARD_PROMPT_VERSION!r}, got {prompt_version!r}",
        )
    if role_output_schema_version != ROLE_OUTPUT_SCHEMA_VERSION:
        raise BoardAuthorizationError(
            SIX_ROLE_VERSION_MISMATCH,
            f"{ctx}: role_output_schema_version must equal the frozen "
            f"pin {ROLE_OUTPUT_SCHEMA_VERSION!r}, got "
            f"{role_output_schema_version!r}",
        )
    if isinstance(total_call_cap, bool) or not isinstance(
        total_call_cap, int
    ):
        raise BoardAuthorizationError(
            SIX_ROLE_CAP_BAD,
            f"{ctx}: total_call_cap must be an int, got "
            f"{total_call_cap!r}",
        )
    if total_call_cap != SIX_ROLE_WINDOW_CALLS:
        raise BoardAuthorizationError(
            SIX_ROLE_CAP_BAD,
            f"{ctx}: total_call_cap must be exactly "
            f"{SIX_ROLE_WINDOW_CALLS} this round (one window = six "
            f"logical calls per grant), got {total_call_cap}",
        )
    frozen_retry = _validate_retry_policy(retry_policy, ctx)
    frozen_accounting = _validate_accounting_policy(
        token_accounting_policy, ctx
    )
    source_commit = _require_non_empty_str(source_commit, "source_commit", ctx)
    factory_hash = _client_factory_hash(client_factory)
    runtime_hash = _compute_runtime_hash(
        mode=mode,
        authorization_grant_hash=grant_hash,
        provider=provider,
        model_id=model_id,
        client_factory_hash=factory_hash,
        role_allowlist=BOARD_ROLE_ORDER,
        prompt_version=prompt_version,
        role_output_schema_version=role_output_schema_version,
        retry_policy=frozen_retry,
        token_accounting_policy=frozen_accounting,
        total_call_cap=total_call_cap,
        source_commit=source_commit,
    )
    return AuthorizedSixRoleLLMRuntime(
        mode=mode,
        authorization_grant_hash=grant_hash,
        provider=provider,
        model_id=model_id,
        client_factory_hash=factory_hash,
        role_allowlist=BOARD_ROLE_ORDER,
        prompt_version=prompt_version,
        role_output_schema_version=role_output_schema_version,
        retry_policy=frozen_retry,
        token_accounting_policy=frozen_accounting,
        total_call_cap=total_call_cap,
        source_commit=source_commit,
        runtime_hash=runtime_hash,
        client_factory=client_factory,
    )


def require_production_six_role_runtime(
    runtime: Any, ctx: str
) -> AuthorizedSixRoleLLMRuntime:
    """Production surfaces accept ONLY a PRODUCTION authorization.

    A TEST_ONLY runtime can never pose as production evidence (the
    mode check is mechanical), and the provider is re-checked against
    the whitelist even after authorization.
    """
    if not isinstance(runtime, AuthorizedSixRoleLLMRuntime):
        raise BoardAuthorizationError(
            SIX_ROLE_BAD_TYPE,
            f"{ctx}: expected an AuthorizedSixRoleLLMRuntime, got "
            f"{type(runtime).__name__}",
        )
    if runtime.mode == SIX_ROLE_MODE_TEST_ONLY:
        raise BoardAuthorizationError(
            SIX_ROLE_TEST_ONLY_REJECTED,
            f"{ctx}: runtime is TEST_ONLY (provider "
            f"{runtime.provider!r}); TEST_ONLY authorizations never "
            "enter a production board, never flip a REAL_* flag and "
            "never grant readiness",
        )
    if runtime.provider not in AUTHORIZED_SIX_ROLE_PROVIDERS:
        raise BoardAuthorizationError(
            SIX_ROLE_PROVIDER_UNAUTHORIZED,
            f"{ctx}: provider {runtime.provider!r} is not on the "
            "supervisor-owned six-role whitelist",
        )
    return runtime


class AuthorizedSixRoleClient:
    """Window-scoped six-role client under an explicit authorization.

    Enforces, mechanically and in this order: the call cap, role
    membership, the FIXED role order and no duplicate role calls.
    Every call is journaled (immutable entries + canonical hash) —
    the journal is the authorization-side record complementing the
    accounting ledger.
    """

    def __init__(
        self,
        *,
        runtime: AuthorizedSixRoleLLMRuntime,
        window_id: str,
        underlying: Any,
    ) -> None:
        self._runtime = runtime
        self._window_id = window_id
        self._underlying = underlying
        self._journal: List[Tuple[Tuple[str, Any], ...]] = []
        self._roles_called: List[str] = []

    # ---- the board's query surface --------------------------------
    def query(
        self,
        system_prompt: str,
        user_prompts: Any,
        *,
        cache_key: str,
        role: str,
        window_id: str = "",
    ) -> Any:
        ctx = f"board_authorization.client[{self._window_id}]"
        if window_id and window_id != self._window_id:
            raise BoardAuthorizationError(
                SIX_ROLE_WINDOW_MISMATCH,
                f"{ctx}: query targets window {window_id!r} but this "
                f"client is bound to {self._window_id!r}",
            )
        if len(self._journal) >= self._runtime.total_call_cap:
            raise BoardAuthorizationError(
                SIX_ROLE_CALL_CAP_EXCEEDED,
                f"{ctx}: the authorization grant covers "
                f"{self._runtime.total_call_cap} logical call(s); the "
                f"{len(self._journal) + 1}th call is refused",
            )
        if role not in self._runtime.role_allowlist:
            raise BoardAuthorizationError(
                SIX_ROLE_UNKNOWN_ROLE,
                f"{ctx}: role {role!r} is not one of the six board "
                f"roles {list(self._runtime.role_allowlist)}",
            )
        if role in self._roles_called:
            raise BoardAuthorizationError(
                SIX_ROLE_DUPLICATE_CALL,
                f"{ctx}: role {role!r} was already called in this "
                "window; each of the six roles runs exactly once",
            )
        expected = BOARD_ROLE_ORDER[len(self._roles_called)]
        if role != expected:
            raise BoardAuthorizationError(
                SIX_ROLE_ORDER_VIOLATION,
                f"{ctx}: the board runs roles in the fixed order "
                f"{list(BOARD_ROLE_ORDER)}; expected {expected!r} at "
                f"position {len(self._roles_called)}, got {role!r}",
            )
        result = self._underlying.query(
            system_prompt, user_prompts, cache_key=cache_key, role=role
        )
        self._roles_called.append(role)
        self._journal.append(
            (
                ("index", len(self._journal)),
                ("window_id", self._window_id),
                ("role", role),
                ("cache_key", cache_key),
                ("provider", self._runtime.provider),
                ("model_id", self._runtime.model_id),
                ("mode", self._runtime.mode),
            )
        )
        return result

    # ---- journal surface -------------------------------------------
    @property
    def window_id(self) -> str:
        return self._window_id

    @property
    def journal(self) -> Tuple[Tuple[Tuple[str, Any], ...], ...]:
        return tuple(self._journal)

    @property
    def journal_hash(self) -> str:
        return canonical_sha256(
            [
                [list(pair) for pair in entry]
                for entry in self._journal
            ]
        )

    def window_call_summary(self) -> Dict[str, Any]:
        """Six logical calls == one complete window (never more)."""
        return {
            "window_id": self._window_id,
            "logical_calls": len(self._journal),
            "roles_called": tuple(self._roles_called),
            "all_six_roles_called": tuple(self._roles_called)
            == BOARD_ROLE_ORDER,
            "mode": self._runtime.mode,
            "provider": self._runtime.provider,
        }


def six_role_window_delta(runtime: AuthorizedSixRoleLLMRuntime) -> int:
    """Real logical LLM calls one COMPLETE window contributes.

    PRODUCTION on the supervisor whitelist => 6. TEST_ONLY fixtures,
    replay and mock runtimes => 0: they prove code paths only and
    never count as real execution.
    """
    if (
        runtime.mode == SIX_ROLE_MODE_PRODUCTION
        and runtime.provider in AUTHORIZED_SIX_ROLE_PROVIDERS
    ):
        return SIX_ROLE_WINDOW_CALLS
    return 0


def assert_six_role_window_invariants(
    runtime: AuthorizedSixRoleLLMRuntime,
    client: AuthorizedSixRoleClient,
    window: Any,
    ctx: str,
) -> None:
    """Mechanical six-role window invariants (P0-2):

    * exactly six logical calls, one per role, in the fixed order;
    * a COMPLETE window carries six parsed role outcomes;
    * ANY failed role => the window is VOID with
      INCOMPLETE_REVIEW_WINDOW (never relabelled COMPLETE).
    """
    summary = client.window_call_summary()
    if summary["logical_calls"] != SIX_ROLE_WINDOW_CALLS:
        raise BoardAuthorizationError(
            SIX_ROLE_CALL_CAP_EXCEEDED,
            f"{ctx}: window {summary['window_id']} journaled "
            f"{summary['logical_calls']} logical call(s); one window "
            f"is exactly {SIX_ROLE_WINDOW_CALLS}",
        )
    if summary["roles_called"] != BOARD_ROLE_ORDER:
        raise BoardAuthorizationError(
            SIX_ROLE_ORDER_VIOLATION,
            f"{ctx}: journaled roles {list(summary['roles_called'])} "
            f"!= the fixed order {list(BOARD_ROLE_ORDER)}",
        )
    role_results = tuple(getattr(window, "role_results", ()))
    status = getattr(window, "status", "")
    parsed_count = sum(1 for _role, obj in role_results if obj is not None)
    if status == "COMPLETE" and parsed_count != SIX_ROLE_WINDOW_CALLS:
        raise BoardAuthorizationError(
            SIX_ROLE_BAD_TYPE,
            f"{ctx}: window is COMPLETE but only {parsed_count}/6 role "
            "outcomes parsed — a COMPLETE window carries six parsed "
            "outcomes by construction",
        )
    if parsed_count < SIX_ROLE_WINDOW_CALLS and status != "VOID":
        raise BoardAuthorizationError(
            SIX_ROLE_BAD_TYPE,
            f"{ctx}: {parsed_count}/6 role outcomes parsed but the "
            "window is not VOID — any role failure voids the whole "
            "window (never relabelled COMPLETE)",
        )
