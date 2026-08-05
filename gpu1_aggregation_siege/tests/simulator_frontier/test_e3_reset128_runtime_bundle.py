# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: a Reset128 runtime bundle validates and its carry semantics are
classified as RESET128_CARRY — the two arms differ by construction.
"""

import importlib.util
from pathlib import Path

import pytest

from dicode.simulator_frontier import runtime_bundle as rb
from dicode.simulator_frontier.dual_student import (
    carry_semantics_snapshot,
    memory_carry_rule,
)
from dicode.simulator_frontier.errors import InvalidEvidenceError

R = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def _persistent_module():
    p = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "tpers", p / "test_e3_persistent_runtime_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest():
    m = _persistent_module()._manifest()
    m["student"].update({
        "selected_candidate_id": R,
        "profile_name": "rmt16_reset128_98304",
        "memory_mode": "RESET128",
        "carry_mode": "RESET128",
    })
    m["manifest_hash"] = rb.manifest_canonical_hash(m)
    return m


def test_reset128_bundle_validates():
    rb.validate_runtime_bundle_manifest(_manifest())


def test_reset128_carry_semantics():
    snapshot = carry_semantics_snapshot(R)
    assert snapshot["memory_mode"] == "RESET128"
    assert snapshot["carry_mode"] == "RESET128"
    assert snapshot["carry_rule"] == "RESET128_CARRY"
    assert memory_carry_rule("RESET128") == "RESET128_CARRY"
    assert memory_carry_rule("RESET128") != memory_carry_rule("PERSISTENT")


def test_reset128_bundle_rejects_persistent_memory():
    m = _manifest()
    m["student"]["memory_mode"] = "PERSISTENT"
    m["student"]["carry_mode"] = "PERSISTENT"
    with pytest.raises(InvalidEvidenceError):
        rb.validate_runtime_bundle_manifest(m)
