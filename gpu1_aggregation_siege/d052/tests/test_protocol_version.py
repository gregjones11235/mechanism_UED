"""GATE 1 — protocol_version gate + canonical_v2/legacy compatibility.

Verifies the hard versioning discipline:
  * missing protocol_version -> MISSING_PROTOCOL_VERSION (nonzero exit code)
  * unknown / wrong-case value -> UNKNOWN_PROTOCOL_VERSION, no coercion
  * legacy without --allow-legacy-d052 -> LEGACY_NOT_AUTHORIZED
  * legacy WITH the flag -> allowed, warned, never upgraded, training forbidden
  * canonical_v2 -> resolved, frozen fixed config attached
  * non-mapping config -> INVALID_CONFIG_TYPE
"""
import warnings

import pytest

from d052.legacy.canonical_constants import (
    CANONICAL_V2_FIXED_CONFIG,
    assert_canonical_invariants,
)
from d052.legacy.protocol_version import (
    PROTOCOL_GATE_EXIT_CODE,
    D052ProtocolError,
    ProtocolContext,
    ProtocolVersion,
    assert_training_permitted,
    load_protocol_context,
    resolve_protocol_version,
)


def test_error_exit_code_is_nonzero():
    assert PROTOCOL_GATE_EXIT_CODE != 0
    assert D052ProtocolError.exit_code == PROTOCOL_GATE_EXIT_CODE


def test_canonical_invariants_hold():
    # Must not raise.
    assert_canonical_invariants()


# --- MISSING ----------------------------------------------------------------

@pytest.mark.parametrize("cfg", [{}, {"protocol_version": None}])
def test_missing_protocol_version_fails(cfg):
    with pytest.raises(D052ProtocolError) as ei:
        load_protocol_context(cfg)
    assert ei.value.code == D052ProtocolError.MISSING_PROTOCOL_VERSION
    assert ei.value.exit_code != 0


def test_missing_is_not_silently_defaulted_even_though_canonical_is_authoring_default():
    # canonical_v2 is the AUTHORING default, but parsing must still fail when the
    # field is absent -- never silently filled in.
    with pytest.raises(D052ProtocolError):
        resolve_protocol_version(None)


# --- UNKNOWN / NO COERCION --------------------------------------------------

@pytest.mark.parametrize("bad", ["v2", "canonical", "CANONICAL_V2", "Canonical_v2",
                                 " canonical_v2 ", "legacy ", 1, 2.0, True, ["x"]])
def test_unknown_or_nonexact_value_fails_without_coercion(bad):
    # Note: " canonical_v2 " (surrounded by whitespace) is stripped and accepted;
    # wrong CASE and wrong tokens are rejected (no case-folding / no coercion).
    if isinstance(bad, str) and bad.strip() in {"canonical_v2", "legacy"}:
        # whitespace-padded canonical/legacy are accepted after strip, but legacy
        # still needs the flag.
        if bad.strip() == "canonical_v2":
            assert resolve_protocol_version(bad) is ProtocolVersion.CANONICAL_V2
        else:
            with pytest.raises(D052ProtocolError) as ei:
                resolve_protocol_version(bad)  # legacy without flag
            assert ei.value.code == D052ProtocolError.LEGACY_NOT_AUTHORIZED
        return
    with pytest.raises(D052ProtocolError) as ei:
        resolve_protocol_version(bad)
    assert ei.value.code == D052ProtocolError.UNKNOWN_PROTOCOL_VERSION


# --- LEGACY GATING ----------------------------------------------------------

def test_legacy_without_flag_fails():
    with pytest.raises(D052ProtocolError) as ei:
        load_protocol_context({"protocol_version": "legacy"})
    assert ei.value.code == D052ProtocolError.LEGACY_NOT_AUTHORIZED


def test_legacy_with_flag_is_allowed_warned_and_not_upgraded():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ctx = load_protocol_context({"protocol_version": "legacy"},
                                    allow_legacy_d052=True, source="unit-test")
    assert ctx.is_legacy
    assert ctx.version is ProtocolVersion.LEGACY
    assert ctx.fixed_config is None          # NEVER upgraded / no canonical config
    assert ctx.warnings                       # warning recorded for audit
    assert any("LEGACY" in m for m in ctx.warnings)
    assert any("LEGACY" in str(x.message) for x in w)


def test_legacy_training_is_forbidden():
    ctx = load_protocol_context({"protocol_version": "legacy"},
                                allow_legacy_d052=True)
    with pytest.raises(D052ProtocolError) as ei:
        assert_training_permitted(ctx)
    assert ei.value.code == D052ProtocolError.LEGACY_TRAINING_FORBIDDEN


# --- CANONICAL --------------------------------------------------------------

def test_canonical_resolves_and_attaches_frozen_config():
    ctx = load_protocol_context({"protocol_version": "canonical_v2"},
                                source="unit-test")
    assert ctx.is_canonical
    assert ctx.version is ProtocolVersion.CANONICAL_V2
    assert ctx.fixed_config is not None
    # Frozen config matches the module constant in content.
    assert ctx.fixed_config == CANONICAL_V2_FIXED_CONFIG
    # Key frozen values.
    assert ctx.fixed_config["achievement_schema"] == "craftax_67_v1"
    assert ctx.fixed_config["conditioning_type"] == "achievement_multi_hot"
    assert ctx.fixed_config["conditioning_dimension"] == 67
    assert ctx.fixed_config["student_obs_dim"] == 8335
    assert ctx.fixed_config["candidate_pool_mode"] == "shared_frozen"
    assert ctx.fixed_config["score_normalization"] == "rank_percentile_v1"
    assert ctx.fixed_config["unknown_target_policy"] == "error"
    assert ctx.fixed_config["empty_goal_policy"] == "error"
    assert ctx.fixed_config["fallback_policy"] == "error"


def test_canonical_config_copy_is_isolated():
    ctx = load_protocol_context({"protocol_version": "canonical_v2"})
    ctx.fixed_config["student_obs_dim"] = 1  # mutate the copy
    # Module constant unaffected.
    assert CANONICAL_V2_FIXED_CONFIG["student_obs_dim"] == 8335


def test_canonical_training_not_forbidden_at_version_level():
    # Version-level gate passes for canonical; the stronger per-cell authorization
    # gate (d052/cells/) is what actually blocks unauthorized training.
    ctx = load_protocol_context({"protocol_version": "canonical_v2"})
    assert_training_permitted(ctx)  # must not raise


def test_assert_canonical_helper():
    legacy = load_protocol_context({"protocol_version": "legacy"},
                                   allow_legacy_d052=True)
    with pytest.raises(D052ProtocolError):
        legacy.assert_canonical(purpose="selection")
    canon = load_protocol_context({"protocol_version": "canonical_v2"})
    canon.assert_canonical(purpose="selection")  # must not raise


# --- CONFIG TYPE ------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "canonical_v2", ["canonical_v2"], 42])
def test_non_mapping_config_fails(bad):
    with pytest.raises(D052ProtocolError) as ei:
        load_protocol_context(bad)
    assert ei.value.code == D052ProtocolError.INVALID_CONFIG_TYPE


def test_context_is_immutable():
    ctx = load_protocol_context({"protocol_version": "canonical_v2"})
    with pytest.raises(Exception):
        ctx.version = ProtocolVersion.LEGACY  # type: ignore[misc]
