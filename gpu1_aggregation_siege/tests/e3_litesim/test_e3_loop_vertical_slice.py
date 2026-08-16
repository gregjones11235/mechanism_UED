import os

from dicode.e3_litesim.learning.ppo_bridge import PPOConfig
from dicode.e3_litesim.orchestration.e3_loop import E3Loop, E3LoopConfig
from helpers import small_registry


def test_vertical_slice(tmp_path):
    cfg = E3LoopConfig(iterations=1, num_envs=2, rollout_horizon=8,
                       seeds_per_tier=1, batch_envs=2,
                       ppo=PPOConfig(update_epochs=1),
                       artifacts_dir=str(tmp_path), run_id="t")
    loop = E3Loop(cfg, registry=small_registry())
    summary = loop.run()
    gates = summary["gates"]
    assert gates["G1_STUDENT_BINDING"]
    assert gates["G4_READ_ONLY_PROBE"]
    assert gates["G9_VERTICAL_SLICE"]
    out = os.path.join(str(tmp_path), "t")
    for name in ("transition_accounting.json", "gates.json",
                 "student_binding_report.json",
                 "frontier_state_bank_manifest.json",
                 "vertical_slice/initial_probe.json",
                 "vertical_slice/frontier.json",
                 "vertical_slice/reprobe.json"):
        assert os.path.isfile(os.path.join(out, name)), name
    assert summary["accounting"]["total_simulator_transitions"] > 0