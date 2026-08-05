# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: Persistent memory carry — rmt.mem_tokens are PRESERVED across the
128-step segment boundary (segment counter keeps progressing).
"""

from dicode.simulator_frontier.dual_student import (
    carry_semantics_snapshot,
    memory_carry_rule,
)

P = "PERSISTENT_RMT16_ORIGINAL_VTRACE_98304"


def test_persistent_carry_rule_preserves_tokens():
    rule = memory_carry_rule("PERSISTENT")
    assert rule == "PERSISTENT_CARRY"
    # Persistent semantics: the 128-step segment boundary does NOT clear
    # mem_tokens — the segment counter keeps progressing.
    assert rule != "RESET128_CARRY"


def test_persistent_capture_keeps_memory_identity():
    snapshot = carry_semantics_snapshot(P)
    assert snapshot["carry_rule"] == "PERSISTENT_CARRY"
    assert snapshot["memory_mode"] == "PERSISTENT"
    assert snapshot["carry_mode"] == "PERSISTENT"
    assert snapshot["network_family"] == "RMT16"
