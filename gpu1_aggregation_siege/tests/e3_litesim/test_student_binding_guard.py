import jax
import pytest

from dicode.e3_litesim.runtime.hashing import hash_pytree
from dicode.e3_litesim.runtime.student_binding import (BindingError,
                                                       StudentBindingGuard,
                                                       StudentIdentity)
from helpers import make_setup


def _identity(s, version=0, *, params_hash=None):
    return StudentIdentity(student_id="slice_student", student_version=version,
                           architecture_family="slice",
                           params_hash=params_hash or hash_pytree(s["params"]),
                           checkpoint_step=0)


def _bind(guard, s, identity, *, probe_params=None, ppo_params=None,
          checkpoint_params=None):
    return guard.bind(session_idx=0, global_step=0, identity=identity,
                      runstate_params=s["params"],
                      probe_params=(s["params"] if probe_params is None
                                    else probe_params),
                      ppo_params=(s["params"] if ppo_params is None
                                  else ppo_params),
                      checkpoint_params=checkpoint_params)


def _stale(s):
    return jax.tree_util.tree_map(lambda x: x * 1.001, s["params"])


# 1) correct binding -> PASS
def test_correct_binding_pass():
    s = make_setup()
    guard = StudentBindingGuard()
    rec = _bind(guard, s, _identity(s))
    assert rec.binding_verified
    guard.verify(rec)
    assert guard.report(rec)["schema"] == "e3_litesim.student_binding/v1"


# 2) stale probe -> FAIL
def test_stale_probe_fail():
    s = make_setup()
    rec = _bind(StudentBindingGuard(), s, _identity(s), probe_params=_stale(s))
    assert not rec.binding_verified
    assert "stale_probe" in rec.failure_reasons
    with pytest.raises(BindingError):
        StudentBindingGuard().verify(rec)


# 3) params mismatch -> FAIL
def test_params_mismatch_fail():
    s = make_setup()
    rec = _bind(StudentBindingGuard(), s, _identity(s), ppo_params=_stale(s))
    assert not rec.binding_verified
    assert "params_mismatch" in rec.failure_reasons
    with pytest.raises(BindingError):
        StudentBindingGuard().verify(rec)


# 4) checkpoint mismatch -> FAIL
def test_checkpoint_mismatch_fail():
    s = make_setup()
    rec = _bind(StudentBindingGuard(), s, _identity(s),
                checkpoint_params=_stale(s))
    assert not rec.binding_verified
    assert "checkpoint_mismatch" in rec.failure_reasons
    with pytest.raises(BindingError):
        StudentBindingGuard().verify(rec)


# 5) identity mismatch (recorded params_hash != actual) -> FAIL
def test_identity_mismatch_fail():
    s = make_setup()
    bad = _identity(s, params_hash="deadbeef" * 8)
    rec = _bind(StudentBindingGuard(), s, bad)
    assert not rec.binding_verified
    assert "identity_params_mismatch" in rec.failure_reasons
    with pytest.raises(BindingError):
        StudentBindingGuard().verify(rec)


def test_identity_fields_recorded():
    s = make_setup()
    rec = _bind(StudentBindingGuard(), s, _identity(s, version=3))
    d = rec.to_dict()
    assert d["identity"]["student_id"] == "slice_student"
    assert d["identity"]["student_version"] == 3
    assert d["identity"]["architecture_family"] == "slice"
    assert d["identity"]["params_hash"] == rec.runstate_params_hash
