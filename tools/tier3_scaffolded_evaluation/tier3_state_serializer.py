#!/usr/bin/env python3
"""CC4 Tier3 — state serializer (ADDITIVE reuse of the CC4 V3 world serializer).

This module does NOT re-implement V3. For the REAL EnvState pytree it delegates
verbatim to the frozen CC4 V3 canonical serializer
(``tools/global_evaluation/materialize_craftax_world_set_twice.py``):
    * ``encode_node``            — deterministic, lossless, pickle-free pytree encoder
    * ``serialize_world_payload``/ ``state_payload_hash`` — SEED-FREE payload hash
    * ``verify_source_identity`` — runtime executed-source identity binding (realpath+SHA)
    * ``FailClosed`` / ``require`` — fail-closed discipline
Importing V3 is mandatory: if it cannot be imported this module FAILS CLOSED rather
than silently diverging into a second serializer (禁止重新实现 V3).

Two payload carriers are distinguished and NEVER conflated:

  1. REAL EnvState payload (JAX host only):
        envstate_payload_hash(envstate_pytree) -> v3mat.state_payload_hash(...)
     This is what a materialized scaffold start's hash is made of. On THIS host
     (no JAX / no craftax) it is BLOCKED_ENVIRONMENT and is never faked.

  2. NORMALIZED state view (pure Python, synthetic-testable):
        normalized_payload_hash(normalized_dict) -> SHA256(canonical_json)
     A plain-dict projection (see tier3_event_predicates docstring) used by the
     predicate / progress / taxonomy layer and by synthetic state-bank protocol
     tests. Its hash is a PROTOCOL self-test carrier only — it is NEVER passed off
     as a real materialized world, and it is NEVER the GLOBAL_WORLD_SET_HASH.

The serialized bytes are treated as OPAQUE: we only ever hash them and compare
hashes. We NEVER edit serialized bytes and then deserialize (禁止直接改序列化字节后反序列化);
tamper detection is hash comparison (NEG05).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys

# Runnable-as-script AND importable-as-package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
# Additive reuse of the frozen V3 serializer (sibling tool package).
_REPO_TOOLS = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, os.path.join(_REPO_TOOLS, "global_evaluation"))

import tier3_source_audit as audit  # noqa: E402

try:
    import materialize_craftax_world_set_twice as v3mat  # noqa: E402
except Exception as exc:  # pragma: no cover - depends on host layout
    raise RuntimeError(
        "FAIL CLOSED: cannot import the CC4 V3 canonical serializer "
        "(materialize_craftax_world_set_twice). Tier3 reuses V3 additively and will "
        "NOT re-implement it. Error: %r" % (exc,))

SCHEMA = "mechanism_UED.tier3_state_payload/v1"


class FailClosed(Exception):
    """Hard stop on any serialization / identity / integrity failure."""


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------------------------------------------------------------------
# Environment capability (honest BLOCKED labels)
# ---------------------------------------------------------------------------
def have_jax() -> bool:
    return importlib.util.find_spec("jax") is not None


def have_craftax() -> bool:
    return importlib.util.find_spec("craftax") is not None


def have_jax_craftax() -> bool:
    return have_jax() and have_craftax()


def environment_status() -> str:
    if have_jax_craftax():
        return "JAX_CRAFTAX_AVAILABLE"
    return "BLOCKED_ENVIRONMENT"


# ---------------------------------------------------------------------------
# Required EnvState fields. A materialized start (real or normalized) MUST carry
# these; a missing one -> FailClosed (NEG04). Names are the audited EnvState
# top-level fields (tier3_source_audit.ENVSTATE_TOP_FIELDS), never invented.
# ---------------------------------------------------------------------------
ALL_ENVSTATE_FIELD_NAMES = [n for n, _ in audit.ENVSTATE_TOP_FIELDS]

# The normalized-view fields every scaffold start must expose to the predicates.
REQUIRED_NORMALIZED_FIELDS = [
    "_normalized", "player_level", "player_health", "player_position", "timestep",
    "achieved", "mobs", "inventory", "down_ladders", "up_ladders",
    "monsters_killed", "boss_progress",
]


def assert_required_envstate_fields(state: dict, required=REQUIRED_NORMALIZED_FIELDS):
    """NEG04: reject a start that is missing any required field."""
    require(isinstance(state, dict),
            "FAIL CLOSED (NEG04): state is not a dict (cannot verify required fields)")
    missing = [f for f in required if f not in state]
    require(not missing,
            "FAIL CLOSED (NEG04): materialized start missing required EnvState field(s): %s"
            % sorted(missing))
    return True


# ---------------------------------------------------------------------------
# Canonical JSON (pure-Python normalized view) — deterministic, sorted keys.
# ---------------------------------------------------------------------------
def _canonicalize(obj):
    """Deep-convert a normalized view into a strictly JSON-serializable form.

    sets -> sorted lists; tuples -> lists; dict keys -> str; everything recursed.
    Determinism comes from json.dumps(sort_keys=True) on the result.
    """
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonicalize(x) for x in obj)
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    raise FailClosed("FAIL CLOSED: normalized view contains non-JSON-canonical type %r "
                     "(refusing repr/pickle fallback)" % type(obj))


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(_canonicalize(obj), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def normalized_payload_hash(normalized_state: dict):
    """Seed-free SHA256 of the canonical JSON of a normalized state view.

    Returns (sha256_hex, canonical_bytes). PROTOCOL self-test carrier only — this
    is NOT a real materialized-world hash.
    """
    b = canonical_json_bytes(normalized_state)
    return sha256_bytes(b), b


# ---------------------------------------------------------------------------
# Integrity verification (NEG05): hash compare only; bytes are opaque.
# ---------------------------------------------------------------------------
def verify_payload_hash(claimed_sha256: str, payload_bytes: bytes):
    """Fail closed if the recomputed hash does not equal the claimed hash.

    We do NOT deserialize/modify the bytes — only re-hash and compare.
    """
    actual = sha256_bytes(payload_bytes)
    require(actual == claimed_sha256,
            "FAIL CLOSED (NEG05): state payload hash tampered: claimed=%s actual=%s"
            % (claimed_sha256[:16], actual[:16]))
    return actual


# ---------------------------------------------------------------------------
# REAL EnvState payload (delegates to V3; BLOCKED_ENVIRONMENT on this host).
# ---------------------------------------------------------------------------
def envstate_payload_hash(envstate_pytree):
    """SHA256 of the seed-free canonical serialization of a REAL EnvState pytree.

    Delegates to the CC4 V3 ``state_payload_hash`` (identical algorithm, no second
    implementation). Requires numpy at minimum and a real EnvState pytree; on a
    JAX+craftax host this is the genuine materialized-world carrier. Caller is
    responsible for first asserting the JAX/craftax capability; here we fail closed
    if numpy (the minimum to run the V3 encoder) is unavailable.
    """
    require(importlib.util.find_spec("numpy") is not None,
            "FAIL CLOSED (BLOCKED_ENVIRONMENT): numpy unavailable; cannot serialize a real "
            "EnvState pytree. Real scaffold materialization is NOT_RUN on this host.")
    payload, manifest = v3mat.serialize_world_payload(envstate_pytree)
    return sha256_bytes(payload), payload, manifest


def verify_executed_source_identity(requested_path, imported_file, label, expected_sha256=None):
    """ADDITIVE reuse of V3 runtime executed-source identity binding (NEG02)."""
    return v3mat.verify_source_identity(requested_path, imported_file, label,
                                        expected_sha256=expected_sha256)


# ---------------------------------------------------------------------------
# Self-test (pure Python; runs on this host).
# ---------------------------------------------------------------------------
def self_test() -> int:
    problems = []

    def check(name, cond):
        if not cond:
            problems.append(name)

    # V3 additive reuse is live (not re-implemented).
    check("v3_serializer_imported", hasattr(v3mat, "encode_node")
          and hasattr(v3mat, "serialize_world_payload")
          and hasattr(v3mat, "verify_source_identity"))

    # Required-field guard (NEG04).
    good = {f: (0 if f not in ("_normalized",) else True) for f in REQUIRED_NORMALIZED_FIELDS}
    check("required_fields_present_ok", assert_required_envstate_fields(good) is True)
    bad = dict(good)
    del bad["player_level"]
    try:
        assert_required_envstate_fields(bad)
        check("required_field_missing_rejected(NEG04)", False)
    except FailClosed:
        check("required_field_missing_rejected(NEG04)", True)

    # Canonical JSON determinism: key/insertion order invariant; set order invariant.
    a = {"b": 1, "a": {"y": [1, 2], "x": {"p", "q"}}}
    b = {"a": {"x": {"q", "p"}, "y": [1, 2]}, "b": 1}
    check("canonical_json_order_invariant", canonical_json_bytes(a) == canonical_json_bytes(b))
    h1, bytes1 = normalized_payload_hash(a)
    h2, bytes2 = normalized_payload_hash(b)
    check("normalized_hash_deterministic", h1 == h2 and bytes1 == bytes2)
    # value change -> hash change
    c = dict(a); c["b"] = 2
    check("normalized_hash_tracks_value", normalized_payload_hash(c)[0] != h1)

    # Integrity verification (NEG05): tamper -> reject; intact -> accept.
    sha, payload = normalized_payload_hash(good)
    check("verify_intact_ok", verify_payload_hash(sha, payload) == sha)
    try:
        verify_payload_hash("0" * 64, payload)
        check("verify_tamper_rejected(NEG05)", False)
    except FailClosed:
        check("verify_tamper_rejected(NEG05)", True)

    # Non-canonical type -> fail closed (no repr/pickle fallback).
    try:
        canonical_json_bytes({"obj": object()})
        check("noncanonical_type_rejected", False)
    except FailClosed:
        check("noncanonical_type_rejected", True)

    # Environment status is honest (this host is BLOCKED_ENVIRONMENT).
    check("environment_status_label",
          environment_status() in ("JAX_CRAFTAX_AVAILABLE", "BLOCKED_ENVIRONMENT"))

    if problems:
        print("TIER3_STATE_SERIALIZER_SELF_TEST_FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print("TIER3_STATE_SERIALIZER_SELF_TEST_PASS (v3_reuse=live, env=%s)" % environment_status())
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in argv:
        return self_test()
    print("usage: tier3_state_serializer.py --self-test")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
