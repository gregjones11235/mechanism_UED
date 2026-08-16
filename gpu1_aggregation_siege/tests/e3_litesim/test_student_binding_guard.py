import jax
import pytest

from dicode.e3_litesim.runtime.student_binding import (BindingError,
                                                       StudentBindingGuard)
from helpers import make_setup


def test_verified_and_stale_rejected():
    s = make_setup()
    guard = StudentBindingGuard()
    rec = guard.bind(session_idx=0, global_step=0, student_version="v0",
                     runstate_params=s["params"], probe_params=s["params"],
                     ppo_params=s["params"], checkpoint_params=s["params"])
    assert rec.binding_verified
    guard.verify(rec)

    stale = jax.tree_util.tree_map(lambda x: x * 1.001, s["params"])
    bad = guard.bind(session_idx=0, global_step=0, student_version="v0",
                     runstate_params=s["params"], probe_params=stale,
                     ppo_params=s["params"])
    assert not bad.binding_verified
    with pytest.raises(BindingError):
        guard.verify(bad)
    report = guard.report(bad)
    assert report["schema"] == "e3_litesim.student_binding/v1"