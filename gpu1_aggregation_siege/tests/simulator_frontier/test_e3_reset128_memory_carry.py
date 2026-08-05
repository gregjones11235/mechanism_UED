# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS: Reset128 memory carry — rmt.mem_tokens are CLEARED to zero at every
128-step segment boundary (a real segment reset, distinct from persistent).
"""

from dicode.simulator_frontier.dual_student import (
    carry_semantics_snapshot,
    memory_carry_rule,
)

R = "RESET128_RMT16_ORIGINAL_VTRACE_98304"


def test_reset128_carry_rule_clears_tokens():
    assert memory_carry_rule("RESET128") == "RESET128_CARRY"
    # The two arms share the memory TREE but their carry behaviour DIFFERS by
    # construction — same-shape is never the semantics.
    assert memory_carry_rule("RESET128") != memory_carry_rule("PERSISTENT")


def test_reset128_segment_budget():
    snapshot = carry_semantics_snapshot(R)
    assert snapshot["carry_rule"] == "RESET128_CARRY"
    assert snapshot["memory_mode"] == "RESET128"
    assert snapshot["carry_mode"] == "RESET128"
