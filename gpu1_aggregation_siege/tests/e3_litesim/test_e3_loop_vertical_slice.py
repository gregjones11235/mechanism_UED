import os

from dicode.e3_litesim.data.frontier_spec import FrontierSpec, finalize
from dicode.e3_litesim.learning.ppo_bridge import PPOConfig
from dicode.e3_litesim.orchestration.e3_loop import E3Loop, E3LoopConfig
from helpers import small_registry


def _cfg(tmp_path, iterations=2, **kw):
    base = dict(iterations=iterations, num_envs=2, rollout_horizon=8,
                seeds_per_tier=1, batch_envs=2,
                ppo=PPOConfig(update_epochs=1),
                artifacts_dir=str(tmp_path), run_id="t")
    base.update(kw)
    return E3LoopConfig(**base)


def test_vertical_slice(tmp_path):
    loop = E3Loop(_cfg(tmp_path), registry=small_registry())
    summary = loop.run()
    gates = summary["gates"]
    assert gates["G1_STUDENT_BINDING"]
    assert gates["G4_READ_ONLY_PROBE"]
    assert gates["G5_ON_POLICY"]
    assert gates["G6_PPO_BRIDGE"]
    assert gates["G7_TRANSITION_ACCOUNTING"]
    assert gates["G9_VERTICAL_SLICE"]

    out = os.path.join(str(tmp_path), "t")
    for name in ("transition_accounting.json", "gates.json",
                 "student_binding_report.json",
                 "frontier_state_bank_manifest.json",
                 "vertical_slice/initial_probe.json",
                 "vertical_slice/frontier.json",
                 "vertical_slice/frontier_after.json"):
        assert os.path.isfile(os.path.join(out, name)), name
    assert summary["accounting"]["total_simulator_transitions"] > 0
    # NoCausal default: no counterfactual evidence, no LLM calls
    assert summary["mode"] == "NoCausal"
    assert summary["accounting"]["llm_calls"] == 0
    assert all(it["causal_evidence"] is None for it in summary["iterations"])
    # evidence-driven G9
    ev = loop._g9_evidence
    assert ev["reprobes_ge_2"] and ev["ppo_updates_ge_2"]
    assert ev["frontier_after_exists"]


def test_adaptive_frontier_chain(tmp_path):
    """Iteration k must consume the frontier located after iteration k-1."""
    loop = E3Loop(_cfg(tmp_path), registry=small_registry())
    summary = loop.run()
    its = summary["iterations"]
    assert len(its) == 2
    # round 2's frontier_used == round 1's frontier_after (NOT the initial A)
    assert its[1]["frontier_used"]["spec_hash"] == its[0]["frontier_after"]["spec_hash"]
    # the chain is recorded with distinct params hashes per iteration
    assert its[0]["identity"]["student_version"] == 0
    assert its[1]["identity"]["student_version"] == 1


def _mk_frontier(tier):
    fam = "BASIC_SURVIVAL" if tier == "tier1_survive" else "THREAT_MANAGEMENT"
    pred = "survived_horizon" if tier == "tier1_survive" else "monster_killed"
    return finalize(FrontierSpec(
        skill_family=fam, tier=tier, probe_id=f"{tier}#p",
        mastered_before=None, failing_here=tier, status="FRONTIER",
        rollout_horizon=8, success_predicate=pred,
        progress_metric="success_rate", priority=1,
        allowed_variations=("frozen", "prefix_variant")))


def test_round2_simulator_uses_frontier_b_not_a(tmp_path, monkeypatch):
    """Probe0 -> Frontier A; PPO; Probe1 -> Frontier B; round 2 must input B."""
    from dicode.e3_litesim.orchestration import e3_loop as el

    calls = {"n": 0}

    def fake_locate(measurement, registry):
        calls["n"] += 1
        return _mk_frontier("tier1_survive" if calls["n"] == 1 else "tier2_combat")

    monkeypatch.setattr(el, "locate_frontier", fake_locate)
    loop = E3Loop(_cfg(tmp_path), registry=small_registry())
    summary = loop.run()
    its = summary["iterations"]
    assert its[0]["frontier_used"]["tier"] == "tier1_survive"      # A
    assert its[0]["frontier_after"]["tier"] == "tier2_combat"     # B located
    assert its[1]["frontier_used"]["tier"] == "tier2_combat"      # round 2 = B
    assert its[1]["frontier_used"]["tier"] != "tier1_survive"     # NOT A
