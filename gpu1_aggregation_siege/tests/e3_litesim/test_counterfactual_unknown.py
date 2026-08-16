import jax
import numpy as np
import pytest

from dicode.e3_litesim.measurement.causal_evidence import (
    CauseRecord, aggregate_causal_evidence)
from dicode.e3_litesim.measurement.counterfactual_runner import (
    InterventionSpec, assert_intervention_isolation,
    run_counterfactual_diagnosis, InterventionIsolationError)
from dicode.e3_litesim.measurement.failure_capsule import restore_capsule
from test_state_restore_replay import _capsule
from helpers import make_setup


def test_unknown_first_class():
    ev = aggregate_causal_evidence([
        CauseRecord("a", "X", 0.02, 0.0, {}),
        CauseRecord("b", "Y", -0.05, 0.0, {})])
    assert ev.unknown and ev.cause == "UNKNOWN"
    ev2 = aggregate_causal_evidence([
        CauseRecord("a", "PERCEPTION", 0.6, 0.2, {})])
    assert not ev2.unknown and ev2.cause == "PERCEPTION"


def test_isolation_guard_rejects_leak():
    s = make_setup()
    cap = _capsule(s)
    state, _mem = restore_capsule(cap)
    leaky = InterventionSpec("leaky", "X", ("player_health",),
                             lambda st: st.replace(player_food=st.player_food + 1))
    with pytest.raises(InterventionIsolationError):
        assert_intervention_isolation(state, leaky.patch(state),
                                      leaky.whitelist)


def test_diagnosis_runs_and_is_isolated():
    s = make_setup()
    cap = _capsule(s)
    ev = run_counterfactual_diagnosis(capsule=cap, env=s["env"],
                                      env_params=s["env_params"],
                                      backend=s["backend"],
                                      params=s["params"],
                                      success_fn=s["registry"].predicate(
                                          "tier1_survive"),
                                      seeds=2, horizon=8)
    assert ev.evidence_hash
    assert isinstance(ev.unknown, bool)