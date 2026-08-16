from dicode.e3_litesim.measurement.capability_probe import (
    CapabilityMeasurementResult, TierProbeResult)
from dicode.e3_litesim.measurement.frontier_locator import locate_frontier
from helpers import small_registry


def _tier(tier_id, family, status, order):
    return TierProbeResult(tier_id=tier_id, skill_family=family,
                           probe_id=f"{tier_id}#x", n_episodes=4, horizon=8,
                           success_rate=0.5, ci_low=0.2, ci_high=0.8,
                           status=status, metrics_aggregate={}, params_hash="h")


def test_frontier_selection():
    meas = CapabilityMeasurementResult(
        student_id="s", checkpoint_step=0, params_hash="h",
        tier_results=[_tier("tier1_survive", "A", "MASTERED", 1),
                      _tier("tier2_combat", "B", "FRONTIER", 2),
                      _tier("tier3_front", "C", "FAILED", 3)],
        read_only_verified=True, probe_wall_s=0.1)
    frontier = locate_frontier(meas, small_registry())
    assert frontier.tier == "tier2_combat"
    assert frontier.mastered_before == "tier1_survive"
    assert frontier.spec_hash