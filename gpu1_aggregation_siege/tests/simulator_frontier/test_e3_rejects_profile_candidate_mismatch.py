# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: a bundle whose profile_name maps to a different candidate than
selected_candidate_id is refused at the manifest level.
"""

import importlib.util
from pathlib import Path

import pytest

from dicode.simulator_frontier import runtime_bundle as rb
from dicode.simulator_frontier.errors import InvalidEvidenceError


def _persistent_manifest():
    p = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "tpers", p / "test_e3_persistent_runtime_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._manifest()


def test_manifest_rejects_profile_candidate_mismatch():
    # selected=RESET128 while profile=rmt16_persistent_98304 -> fail closed.
    m = _persistent_manifest()
    m["student"]["selected_candidate_id"] = "RESET128_RMT16_ORIGINAL_VTRACE_98304"
    m["student"]["memory_mode"] = "RESET128"
    m["student"]["carry_mode"] = "RESET128"
    with pytest.raises(InvalidEvidenceError):
        rb.validate_runtime_bundle_manifest(m)


def test_manifest_rejects_memory_mode_mismatch():
    m = _persistent_manifest()
    m["student"]["memory_mode"] = "RESET128"
    with pytest.raises(InvalidEvidenceError):
        rb.validate_runtime_bundle_manifest(m)
