import numpy as np
import pytest

from dicode.e3_litesim.runtime import recurrent_state as rs


def test_batch_mismatch_raises():
    with pytest.raises(rs.RecurrentStateError):
        rs.assert_state_start_alignment({"mem": np.zeros((3, 4))}, 2)


def test_slowgru_zero_longstate_rejected():
    mem = {"longstate.h": np.zeros((2, 4)), "longstate.buf": np.zeros((2, 4)),
           "longstate.count": np.zeros((2,))}
    with pytest.raises(rs.RecurrentStateError):
        rs.assert_state_start_alignment(mem, 2, architecture_family="slowgru")
    rs.assert_state_start_alignment(mem, 2, architecture_family="slowgru",
                                    allow_memory_reset_experiment=True)


def test_slice_ok():
    rs.assert_state_start_alignment({"mem": np.zeros((2, 4))}, 2)