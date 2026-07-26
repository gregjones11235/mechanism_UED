"""Step 2 targeted tests: long-context learner + off-policy gradient updates.

D059 Gates tested here:
  Gate 2  — long-context sequence learner consumes stored trajectory (>128)
  Gate 7  — two optimizer updates with finite nonzero gradients and changed
            treatment parameters
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trajectory_replay import (
    Trajectory,
    ReplaySample,
    TrajectoryReplayBuffer,
)
from hindsight import relabel_sample
from long_context_learner import LongContextLearner


# ---------------------------------------------------------------------------
# Minimal test network (avoids full GTrXL dependency for unit-test isolation)
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """A minimal transformer-like network for gradient testing.

    Has just enough structure to test the learner's gradient flow
    without loading the full GTrXL dependency chain.
    """

    action_dim: int = 10
    hidden: int = 32
    window_mem: int = 16
    num_layers: int = 2
    embed_size: int = 32
    num_heads: int = 4
    window_grad: int = 8

    def setup(self):
        self.encoder = nn.Dense(self.embed_size)
        self.decoder = nn.Dense(self.hidden)
        self.actor = nn.Dense(self.action_dim)
        self.critic = nn.Dense(1)

    def __call__(self, memories, obs, mask):
        # Simple: project obs, ignore memories for test simplicity
        x = nn.relu(self.encoder(obs))
        x = nn.relu(self.decoder(x))
        logits = self.actor(x)
        import distrax
        pi = distrax.Categorical(logits=logits)
        v = jnp.squeeze(self.critic(x), axis=-1)
        mem_out = jnp.mean(x, axis=tuple(range(1, x.ndim)))
        return pi, v, mem_out

    def model_forward_eval(self, memories, obs, mask):
        return self(memories, obs, mask)

    def model_forward_train(self, memories, obs, mask):
        pi, v, _ = self(memories, obs, mask)
        return pi, v


# ---------------------------------------------------------------------------
# Test config (matches D059 gamma=0.999, gae_lambda=0.8)
# ---------------------------------------------------------------------------

@dataclass
class TestConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.8
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.002
    max_grad_norm: float = 1.0
    window_grad: int = 8
    window_mem: int = 16
    num_heads: int = 4
    embed_size: int = 32
    num_layers: int = 2
    num_steps: int = 128
    lr: float = 1e-3


# ---------------------------------------------------------------------------
# Helper: build synthetic ReplaySample for learner
# ---------------------------------------------------------------------------

def _make_replay_sample(
    length: int = 200,
    obs_dim: int = 75,
    n_ach: int = 67,
    achieved_indices: tuple = (41,),
    target_indices: tuple = (0,),
    window_mem: int = 16,
    n_layers: int = 2,
    embed: int = 32,
    embedding_size: int = 67,
) -> ReplaySample:
    """Build a deterministic ReplaySample.

    obs_dim must be >= embedding_size: the trailing ``embedding_size`` dims hold
    the (original) goal/task conditioning so hindsight relabeling can replace
    them.  don[-1]=True so episode_done=True (terminal bootstrap = 0).
    """
    assert obs_dim >= embedding_size, "obs must carry a trailing goal-embedding region"
    r = np.random.RandomState(0)
    obs = r.randn(length, obs_dim).astype(np.float32)

    tgt = np.zeros(n_ach, dtype=np.float32)
    for idx in target_indices:
        tgt[idx] = 1.0
    # Trailing embedding region = original goal conditioning.
    obs[:, obs_dim - embedding_size:] = tgt[:embedding_size]

    act = r.randint(0, 10, size=length).astype(np.int32)
    rew = r.randn(length).astype(np.float32) * 0.1
    don = np.zeros(length, dtype=bool)
    don[-1] = True
    val = r.randn(length).astype(np.float32) * 0.1
    lp = r.randn(length).astype(np.float32)

    mem = r.randn(window_mem, n_layers, embed).astype(np.float32) * 0.01

    ach = np.zeros((length, n_ach), dtype=np.float32)
    for idx in achieved_indices:
        step = min(idx % length, length - 1)
        ach[step, idx] = 1.0

    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0).copy()

    return ReplaySample(
        observations=obs,
        actions=act,
        rewards=rew,
        dones=don,
        values=val,
        log_probs=lp,
        initial_memory=mem,
        achievements=ach,
        target_achievements=tgt,
        source_trajectory_id=0,
        start_step=0,
        length=length,
        next_observations=next_obs,
        next_value=0.0,
        episode_done=True,
    )


def _make_train_state(network, obs_dim, config):
    rng = jax.random.PRNGKey(42)
    init_obs = jnp.zeros((2, obs_dim))
    init_mem = jnp.zeros((2, config.window_mem, config.num_layers, config.embed_size))
    init_mask = jnp.zeros((2, config.num_heads, 1, config.window_mem + 1), dtype=jnp.bool_)
    params = network.init(rng, init_mem, init_obs, init_mask)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


# ---------------------------------------------------------------------------
# Gate 2 (positive): long-context learner consumes >128 step sequences
# ---------------------------------------------------------------------------

def test_gate2_consumes_long_sequence():
    """Learner must accept and process a sequence longer than 128."""
    config = TestConfig()
    network = TinyTransformer(
        action_dim=10, hidden=32, window_mem=config.window_mem,
        num_layers=config.num_layers, embed_size=config.embed_size,
        num_heads=config.num_heads, window_grad=config.window_grad,
    )
    learner = LongContextLearner(network, config, jax.random.PRNGKey(0))
    sample = _make_replay_sample(length=200, window_mem=config.window_mem,
                                 n_layers=config.num_layers, embed=config.embed_size)

    # Must NOT raise on >128 sequence
    learner._validate_sequence_length(sample)

    obs_dim = 75
    ts = _make_train_state(network, obs_dim, config)

    # Run the constrained replay auxiliary loss to verify it compiles and runs.
    loss, diag = learner._replay_aux_loss(ts.params, sample, 0, False)
    assert jnp.isfinite(loss), f"Loss non-finite: {loss}"
    assert np.isfinite(diag["value_loss"]), "value aux loss non-finite"
    # Off-policy diagnostics are recorded even though the actor term is off.
    assert "ess_fraction" in diag and "policy_lag" in diag
    assert "importance_ratio_mean" in diag

    print("  PASS Gate 2: long-context learner consumes 200-step sequence")


def test_gate2_rejects_short_sequence():
    """Learner must REJECT sequences <= 128 steps."""
    config = TestConfig()
    network = TinyTransformer()
    learner = LongContextLearner(network, config, jax.random.PRNGKey(0))
    sample = _make_replay_sample(length=64, window_mem=config.window_mem,
                                 n_layers=config.num_layers, embed=config.embed_size)

    try:
        learner._validate_sequence_length(sample)
        assert False, "Gate 2 FAIL: should have rejected 64-step sequence"
    except ValueError as e:
        msg = str(e)
        assert "128" in msg or "Gate 2" in msg or "long" in msg.lower(), f"Wrong error: {msg}"

    print("  PASS Gate 2: 64-step sequence rejected by long-context learner")


# ---------------------------------------------------------------------------
# Gate 7: two optimizer updates with finite nonzero gradients + changed params
# ---------------------------------------------------------------------------

def test_gate7_two_updates_finite_nonzero():
    """Perform two off-policy updates. Verify:
    - Gradients are finite and nonzero (Gate 7 positive)
    - Parameters change between updates
    - Both updates succeed
    """
    config = TestConfig()
    network = TinyTransformer(
        action_dim=10, hidden=32, window_mem=config.window_mem,
        num_layers=config.num_layers, embed_size=config.embed_size,
        num_heads=config.num_heads, window_grad=config.window_grad,
    )
    learner = LongContextLearner(network, config, jax.random.PRNGKey(1))
    obs_dim = 75
    ts = _make_train_state(network, obs_dim, config)

    # Record initial params
    params_before = jax.tree_util.tree_map(lambda x: x.copy(), ts.params)

    # Update 1
    sample1 = _make_replay_sample(length=200, window_mem=config.window_mem,
                                  n_layers=config.num_layers, embed=config.embed_size)
    ts, metrics1 = learner.update(ts, sample1)

    assert metrics1["params_changed"], "Gate 7 FAIL: params did not change on update 1"
    assert metrics1["grad_norm"] > 1e-12, f"Gate 7 FAIL: grad_norm too small ({metrics1['grad_norm']})"
    assert np.isfinite(metrics1["grad_norm"]), "Gate 7 FAIL: grad_norm non-finite"
    assert np.isfinite(metrics1["total_loss"]), "Gate 7 FAIL: total_loss non-finite"
    assert metrics1["sequence_length"] > 128, "Gate 2/7: sequence not long"

    params_after_1 = jax.tree_util.tree_map(lambda x: x.copy(), ts.params)

    # Verify params actually changed (not just metrics claiming so)
    any_diff_1 = False
    for old, new in zip(
        jax.tree_util.tree_leaves(params_before),
        jax.tree_util.tree_leaves(params_after_1),
    ):
        if jnp.any(old != new):
            any_diff_1 = True
            break
    assert any_diff_1, "Gate 7 FAIL: parameter values unchanged after update 1"

    # Update 2
    sample2 = _make_replay_sample(length=300, window_mem=config.window_mem,
                                  n_layers=config.num_layers, embed=config.embed_size)
    ts, metrics2 = learner.update(ts, sample2)

    assert metrics2["params_changed"], "Gate 7 FAIL: params did not change on update 2"
    assert metrics2["grad_norm"] > 1e-12
    assert np.isfinite(metrics2["grad_norm"])
    assert np.isfinite(metrics2["total_loss"])

    params_after_2 = jax.tree_util.tree_map(lambda x: x.copy(), ts.params)

    # Verify params changed again
    any_diff_2 = False
    for old, new in zip(
        jax.tree_util.tree_leaves(params_after_1),
        jax.tree_util.tree_leaves(params_after_2),
    ):
        if jnp.any(old != new):
            any_diff_2 = True
            break
    assert any_diff_2, "Gate 7 FAIL: parameter values unchanged after update 2"

    print("  PASS Gate 7: two updates with finite, nonzero gradients")
    print(f"    Update 1: loss={metrics1['total_loss']:.4f}, grad_norm={metrics1['grad_norm']:.4f}")
    print(f"    Update 2: loss={metrics2['total_loss']:.4f}, grad_norm={metrics2['grad_norm']:.4f}")


# ---------------------------------------------------------------------------
# Gate 7 extended: treatment parameter change verification
# ---------------------------------------------------------------------------

def test_gate7_hindsight_update():
    """Verify gradient update works on a hindsight-relabeled sample."""
    config = TestConfig()
    network = TinyTransformer(
        action_dim=10, hidden=32, window_mem=config.window_mem,
        num_layers=config.num_layers, embed_size=config.embed_size,
        num_heads=config.num_heads, window_grad=config.window_grad,
    )
    learner = LongContextLearner(network, config, jax.random.PRNGKey(2))
    obs_dim = 75
    ts = _make_train_state(network, obs_dim, config)
    params_before = jax.tree_util.tree_map(lambda x: x.copy(), ts.params)

    # Sample with target=0, achieved=41
    sample = _make_replay_sample(
        length=250,
        achieved_indices=(41,),
        target_indices=(0,),
        window_mem=config.window_mem,
        n_layers=config.num_layers,
        embed=config.embed_size,
    )

    # Relabel to the achieved goal
    relabeled = relabel_sample(sample, goal_index=41, goal_name="DEFEAT_KOBOLD")
    assert relabeled.target_achievements[41] == 1.0
    assert relabeled.target_achievements[0] == 0.0

    ts, metrics = learner.update(ts, relabeled)
    assert metrics["params_changed"]
    assert metrics["grad_norm"] > 1e-12

    any_diff = False
    for old, new in zip(
        jax.tree_util.tree_leaves(params_before),
        jax.tree_util.tree_leaves(ts.params),
    ):
        if jnp.any(old != new):
            any_diff = True
            break
    assert any_diff, "Gate 7 FAIL: params unchanged on hindsight update"

    print("  PASS Gate 7: hindsight-relabeled update with changed params")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    tests = [
        ("Gate 2: consumes long sequence", test_gate2_consumes_long_sequence),
        ("Gate 2: rejects short sequence", test_gate2_rejects_short_sequence),
        ("Gate 7: two updates finite nonzero", test_gate7_two_updates_finite_nonzero),
        ("Gate 7: hindsight update", test_gate7_hindsight_update),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            results[name] = "PASS"
        except Exception as e:
            failed += 1
            results[name] = f"FAIL: {e}"
            import traceback
            traceback.print_exc()
            print(f"  FAIL {name}: {e}")

    print(f"\n{'='*60}")
    print(f"Step 2 Results: {passed}/{passed+failed} tests PASS")
    print(f"{'='*60}")

    gates = {
        "Gate 2 (long-context learner)": all(
            results.get(t, "").startswith("PASS")
            for t in ["Gate 2: consumes long sequence", "Gate 2: rejects short sequence"]
        ),
        "Gate 7 (finite nonzero gradients)": all(
            results.get(t, "").startswith("PASS")
            for t in ["Gate 7: two updates finite nonzero", "Gate 7: hindsight update"]
        ),
    }

    all_gates_pass = all(gates.values())

    report = {
        "step": 2,
        "directive": "D059",
        "description": "long_context_learner + off-policy gradient update tests",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "results": results,
        "gates": {k: v for k, v in gates.items()},
        "all_gates_pass": all_gates_pass,
    }

    os.makedirs(
        os.path.join(os.path.dirname(__file__), "..", "evidence"), exist_ok=True
    )
    report_path = os.path.join(
        os.path.dirname(__file__), "..", "evidence", "step2_test_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nReport written to: {report_path}")
    print(f"All gates PASS: {all_gates_pass}")

    if not all_gates_pass:
        print("\nGATE FAILURES:")
        for k, v in gates.items():
            if not v:
                print(f"  {k}: FAIL")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
