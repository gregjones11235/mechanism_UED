import numpy as np
import pytest

from dicode.e3_litesim.runtime import recurrent_state as rs


def test_batch_mismatch_raises():
    with pytest.raises(rs.RecurrentStateError):
        rs.assert_state_start_alignment({"mem": np.zeros((3, 4))}, 2)


def _slowgru_mem(zero_longstate=True):
    ls = 0.0 if zero_longstate else 1.0
    return {
        "memories": np.zeros((2, 128, 2, 256)),
        "mem_mask": np.zeros((2, 8, 1, 129), dtype=bool),
        "mem_idx": np.zeros((2,), dtype=np.int32),
        "longstate.h": np.full((2, 4), ls),
        "longstate.buf": np.full((2, 4), ls),
        "longstate.count": np.full((2,), ls),
    }


def test_slowgru_zero_longstate_rejected():
    with pytest.raises(rs.RecurrentStateError):
        rs.assert_state_start_alignment(_slowgru_mem(True), 2,
                                        architecture_family="slowgru")
    rs.assert_state_start_alignment(_slowgru_mem(True), 2,
                                    architecture_family="slowgru",
                                    allow_memory_reset_experiment=True)
    # non-zero longstate + fast-window keys present => OK
    rs.assert_state_start_alignment(_slowgru_mem(False), 2,
                                    architecture_family="slowgru")


def test_slice_ok():
    rs.assert_state_start_alignment({"mem": np.zeros((2, 4))}, 2)


def test_slowgru_key_alias_canonicalization():
    """mem_mask/mem_idx <-> memories_mask/memories_mask_idx must normalize."""
    adapter = {
        "memories": np.zeros((2, 128, 2, 256)),
        "memories_mask": np.zeros((2, 8, 1, 129), dtype=bool),
        "memories_mask_idx": np.zeros((2,), dtype=np.int32),
        "longstate.h": np.ones((2, 256)),
        "longstate.buf": np.ones((2, 32, 256)),
        "longstate.count": np.ones((2,), dtype=np.int32),
    }
    canon = rs.canonicalize_memory(adapter, architecture_family="slowgru")
    assert "mem_mask" in canon and "memories_mask" not in canon
    assert "mem_idx" in canon and "memories_mask_idx" not in canon
    # validation accepts the adapter spelling via canonicalization
    check = rs.validate_memory(adapter, 2, architecture_family="slowgru")
    assert check["ok"], check["reasons"]