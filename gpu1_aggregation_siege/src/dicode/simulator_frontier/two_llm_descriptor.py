"""Authorized two-LLM runtime descriptor (director handoff, P0-b1).

Before this contract existed, the signed runtime bundle demanded
``two_llm_runtime: null`` while the E3 preflight simultaneously REQUIRED an
``AuthorizedTwoLLMRuntime`` — a contradiction that made the smoke path
unreachable by construction.  A bundle could never carry a REAL two-LLM
runtime, so preflight could never pass on the bundle path.

``AuthorizedTwoLLMRuntimeDescriptor`` closes that gap: a MINT-ONLY, hash-bound
descriptor that names everything needed to build the real runtime from a
bundle:

* ``authorization_id`` + ``trusted_signer`` — the authorization that allows
  exactly the two logical LLM calls;
* ``provider`` / ``model`` — which real model serves the calls;
* ``client_factory_entrypoint`` — the director-approved, source-hash-bound
  ``module:attr`` that produces the role -> client mapping;
* ``implementation_hash`` — sha256 of the factory's source file + text
  (computed at mint, recomputed by verify: substitution is impossible);
* ``role_allowlist`` == ``LLM_ROLE_SEQUENCE`` and ``exact_call_cap`` ==
  ``TWO_LLM_CALL_CEILING`` — the call contract is structural;
* ``token_cap`` / ``retry_cap`` / ``journal_sink`` — the operational bounds.

``build_authorized_two_llm_runtime`` turns a verified descriptor into the
``AuthorizedTwoLLMRuntime`` the preflight requires.  Building does NOT call
any LLM — a ``--check-only`` run verifies the descriptor and stops.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

from .errors import InvalidEvidenceError, ProductionBlockedError
from .llm_contracts import (
    LLM_ROLE_SEQUENCE,
    TWO_LLM_CALL_CEILING,
    TWO_LLM_AUTHORIZATION_SCHEMA,
    AuthorizedTwoLLMRuntime,
    TwoLLMAuthorization,
    mint_two_llm_authorization,
    verify_two_llm_authorization,
)

TWO_LLM_DESCRIPTOR_SCHEMA = "simulator_frontier.two-llm-runtime-descriptor/v1"
TWO_LLM_DESCRIPTOR_VERSION = "two-llm-runtime-descriptor/v1"

_SYNTHETIC_SIGNATURE_PREFIX = "SYNTHETIC_SIGNATURE_"

DESCRIPTOR_BUNDLE_KEYS = frozenset({
    "descriptor_id", "authorization_id", "provider", "model",
    "client_factory_entrypoint", "client_factory_implementation_hash",
    "token_cap", "retry_cap", "journal_sink", "trusted_signer",
})


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise InvalidEvidenceError(
            f"{name} is not a lowercase sha256 hex digest: {text[:24]!r}…")
    return text


def _import_entrypoint(entrypoint: str, purpose: str) -> Any:
    import importlib
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1 \
            or not all(part.strip() for part in entrypoint.split(":")):
        raise InvalidEvidenceError(
            f"{purpose} entry point must be 'module:attr', got {entrypoint!r}")
    module_name, attr_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise InvalidEvidenceError(
            f"cannot import {purpose} entry point module {module_name!r}: {exc!r}") from exc
    try:
        target = getattr(module, attr_name)
    except AttributeError as exc:
        raise InvalidEvidenceError(
            f"{purpose} entry point attribute {attr_name!r} not found in "
            f"{module_name!r}") from exc
    if not callable(target):
        raise InvalidEvidenceError(
            f"{purpose} entry point resolved to a non-callable "
            f"({type(target).__name__})")
    return target


def _callable_source_sha256(name: str, fn: Any) -> str:
    """sha256 of source file + source text (EOL-normalized), fail-closed."""
    if isinstance(fn, Mapping) or not callable(fn):
        raise InvalidEvidenceError(
            f"{name}: expected a callable, got {type(fn).__name__}")
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise InvalidEvidenceError(
            f"{name}: cannot bind the callable — its source text is unavailable "
            f"({exc!r}); fail closed") from exc
    try:
        source_file = str(inspect.getsourcefile(fn) or "<unknown>")
    except TypeError:
        source_file = "<unknown>"
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(
        f"{source_file}\n::\n{normalized}".encode("utf-8")).hexdigest()


def _require_nonempty_str(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvidenceError(
            f"{where} must be a non-empty string, got {value!r}")
    return value


def _require_nonneg_int(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InvalidEvidenceError(
            f"{where} must be a non-negative int, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class AuthorizedTwoLLMRuntimeDescriptor:
    """One immutable, hash-bound two-LLM runtime descriptor (mint-only).

    ``runtime_hash`` is NOT a constructor argument: it is computed in
    ``__post_init__`` from the descriptor fields only.
    """

    descriptor_id: str
    authorization_id: str
    provider: str
    model: str
    client_factory_entrypoint: str
    client_factory_implementation_hash: str
    token_cap: int
    retry_cap: int
    journal_sink: str
    trusted_signer: str
    role_allowlist: tuple[str, ...] = field(init=False)
    exact_call_cap: int = field(init=False)
    runtime_hash: str = field(init=False)
    descriptor_schema: str = TWO_LLM_DESCRIPTOR_SCHEMA

    def __post_init__(self) -> None:
        for label, value in (("descriptor_id", self.descriptor_id),
                             ("authorization_id", self.authorization_id),
                             ("provider", self.provider),
                             ("model", self.model),
                             ("journal_sink", self.journal_sink),
                             ("trusted_signer", self.trusted_signer)):
            if not str(value).strip():
                raise InvalidEvidenceError(
                    f"AuthorizedTwoLLMRuntimeDescriptor.{label} is empty")
        _require_sha256("client_factory_implementation_hash",
                        self.client_factory_implementation_hash)
        if str(self.trusted_signer).startswith(_SYNTHETIC_SIGNATURE_PREFIX):
            raise InvalidEvidenceError(
                "trusted_signer must be a real director signer id — a synthetic "
                "self-signature can never authorize LLM calls")
        object.__setattr__(self, "role_allowlist", tuple(LLM_ROLE_SEQUENCE))
        object.__setattr__(self, "exact_call_cap", TWO_LLM_CALL_CEILING)
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("runtime_hash", "role_allowlist", "exact_call_cap")
        }
        payload["role_allowlist"] = list(LLM_ROLE_SEQUENCE)
        payload["exact_call_cap"] = TWO_LLM_CALL_CEILING
        object.__setattr__(self, "runtime_hash", _canonical_sha256(payload))


def mint_two_llm_runtime_descriptor(*, descriptor_id: Any,
                                    authorization_id: Any,
                                    provider: Any,
                                    model: Any,
                                    client_factory_entrypoint: Any,
                                    client_factory_implementation_hash: Any,
                                    token_cap: Any,
                                    retry_cap: Any,
                                    journal_sink: Any,
                                    trusted_signer: Any
                                    ) -> AuthorizedTwoLLMRuntimeDescriptor:
    """Mint the descriptor, binding the factory's CURRENT source hash.

    ``client_factory_implementation_hash`` is recomputed from the resolved
    entry point callable and must EQUAL the bundle-declared value — a
    substituted or drifted factory never binds.
    """
    factory = _import_entrypoint(str(client_factory_entrypoint),
                                 "two-LLM client factory")
    actual_hash = _callable_source_sha256("client factory", factory)
    expected = _require_sha256("client_factory_implementation_hash",
                               client_factory_implementation_hash)
    if actual_hash != expected:
        raise InvalidEvidenceError(
            "client factory implementation hash drift: the resolved entry "
            "point callable does not recompute to the declared "
            "implementation hash (fail closed)")
    return AuthorizedTwoLLMRuntimeDescriptor(
        descriptor_id=str(descriptor_id),
        authorization_id=str(authorization_id),
        provider=str(provider),
        model=str(model),
        client_factory_entrypoint=str(client_factory_entrypoint),
        client_factory_implementation_hash=expected,
        token_cap=_require_nonneg_int("token_cap", token_cap),
        retry_cap=_require_nonneg_int("retry_cap", retry_cap),
        journal_sink=str(journal_sink),
        trusted_signer=str(trusted_signer),
    )


def verify_two_llm_runtime_descriptor(descriptor: Any) -> None:
    """Recompute the implementation hash + runtime hash; reject fakes."""
    if isinstance(descriptor, Mapping):
        raise InvalidEvidenceError(
            "verify_two_llm_runtime_descriptor requires a minted "
            "AuthorizedTwoLLMRuntimeDescriptor, not a mapping")
    if not isinstance(descriptor, AuthorizedTwoLLMRuntimeDescriptor):
        raise InvalidEvidenceError(
            f"verify_two_llm_runtime_descriptor requires a minted "
            f"AuthorizedTwoLLMRuntimeDescriptor, got {type(descriptor).__name__}")
    factory = _import_entrypoint(descriptor.client_factory_entrypoint,
                                 "two-LLM client factory")
    current_hash = _callable_source_sha256("client factory", factory)
    if current_hash != descriptor.client_factory_implementation_hash:
        raise InvalidEvidenceError(
            "client factory implementation hash drift: the client factory was "
            "substituted after minting (fail closed)")
    if tuple(descriptor.role_allowlist) != tuple(LLM_ROLE_SEQUENCE):
        raise InvalidEvidenceError(
            "role_allowlist does not equal the fixed LLM role sequence")
    if int(descriptor.exact_call_cap) != TWO_LLM_CALL_CEILING:
        raise InvalidEvidenceError("exact_call_cap must equal 2")
    payload = {
        f.name: getattr(descriptor, f.name)
        for f in fields(descriptor)
        if f.name not in ("runtime_hash", "role_allowlist", "exact_call_cap")
    }
    payload["role_allowlist"] = list(LLM_ROLE_SEQUENCE)
    payload["exact_call_cap"] = TWO_LLM_CALL_CEILING
    if _canonical_sha256(payload) != descriptor.runtime_hash:
        raise InvalidEvidenceError(
            "runtime_hash mismatch: the descriptor was tampered with or "
            "self-reported (fail closed)")


def build_authorized_two_llm_runtime(
        descriptor: Any) -> AuthorizedTwoLLMRuntime:
    """Build the REAL production runtime from a verified descriptor.

    Building never calls any LLM: it resolves the director-approved client
    factory, mints the authorization bound to the trusted signer, and wraps
    both in the ``AuthorizedTwoLLMRuntime`` the preflight demands.  A
    ``--check-only`` run stops here.
    """
    verify_two_llm_runtime_descriptor(descriptor)
    factory = _import_entrypoint(descriptor.client_factory_entrypoint,
                                 "two-LLM client factory")
    authorization = mint_two_llm_authorization(
        authorization_id=str(descriptor.authorization_id),
        authorizer_id=str(descriptor.trusted_signer))
    verify_two_llm_authorization(authorization)
    return AuthorizedTwoLLMRuntime(authorization=authorization,
                                   client_factory=factory)
