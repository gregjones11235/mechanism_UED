# -*- coding: utf-8 -*-
"""TEST_ONLY / SYNTHETIC fixtures.  NOT_REAL_EXECUTION.

E3-DS (section 6): the RMT16 adapter is READ-ONLY (load / identity /
policy_step / memory validation / probe).  Its save/restore surfaces are
absent, and without a bound CanonicalDiCodeOneUpdateRuntime for the selected
candidate the consumer never claims it can update.
"""

import pytest

from dicode.simulator_frontier.dual_student import (
    E3_REAL_SMOKE_READY,
    E3_STUDENT_READ_ONLY_MOUNT_READY,
    E3_STUDENT_TRAINING_RUNTIME_READY,
)
from dicode.student_adapters.rmt16_adapter import RMT16StudentAdapter


def test_readonly_mount_is_not_a_training_adapter():
    # Honest readiness: read-only mount alone never implies training.
    assert E3_STUDENT_TRAINING_RUNTIME_READY is False
    assert E3_REAL_SMOKE_READY is False


def test_rmt16_save_restore_are_absent():
    # The read-only adapter must not masquerade as a training adapter.
    with pytest.raises(NotImplementedError):
        RMT16StudentAdapter.save_full_state(None, "", None, {})
    with pytest.raises(NotImplementedError):
        RMT16StudentAdapter.restore_full_state(None, "")
