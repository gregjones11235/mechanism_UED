from dicode.e3_litesim.measurement.capability_probe import run_capability_probe
from helpers import make_setup


def test_probe_read_only():
    s = make_setup()
    meas = run_capability_probe(registry=s["registry"], backend=s["backend"],
                                params=s["params"], env_params=s["env_params"],
                                student_id="slice", seeds_per_tier=1,
                                batch_envs=2)
    assert meas.read_only_verified
    assert len(meas.tier_results) == 3
    for tier in meas.tier_results:
        assert tier.status in ("MASTERED", "FRONTIER", "UNSTABLE", "FAILED",
                               "UNKNOWN")
        assert "oscillation_rate" in tier.metrics_aggregate
        assert "stall_rate" in tier.metrics_aggregate