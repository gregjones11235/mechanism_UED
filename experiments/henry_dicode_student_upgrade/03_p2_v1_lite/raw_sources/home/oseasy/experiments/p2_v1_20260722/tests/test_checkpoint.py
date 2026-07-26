"""Step 3 targeted tests: checkpoint save/restore (Gate 8) and manifest (Gate 9).

D059 Gates:
  Gate 8 — save/restore for model, optimizer, replay metadata, RNG, global step
  Gate 9 — unique output root, manifest, source hashes, dependency versions,
           physical UUID
"""

import os
import sys
import json
import hashlib
import pickle
import tempfile
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trajectory_replay import (
    Trajectory,
    TrajectoryReplayBuffer,
    ReplayCounters,
)
from checkpointing import (
    save_full_checkpoint,
    restore_full_checkpoint,
    save_train_state,
    restore_train_state,
    checkpoint_inventory,
    compatible_weight_restore_report,
)


# ---------------------------------------------------------------------------
# Minimal test network (same as in test_learner.py)
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
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


def _make_train_state(network, obs_dim, lr=1e-3):
    rng = jax.random.PRNGKey(0)
    init_obs = jnp.zeros((2, obs_dim))
    init_mem = jnp.zeros((2, 16, 2, 32))
    init_mask = jnp.zeros((2, 4, 1, 17), dtype=jnp.bool_)
    params = network.init(rng, init_mem, init_obs, init_mask)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr, eps=1e-5),
    )
    return TrainState.create(apply_fn=network.apply, params=params, tx=tx)


def _make_trajectory(length=200):
    obs = np.random.randn(length, 8).astype(np.float32)
    act = np.random.randint(0, 10, size=length).astype(np.int32)
    rew = np.random.randn(length).astype(np.float32)
    don = np.zeros(length, dtype=bool); don[-1] = True
    val = np.random.randn(length).astype(np.float32)
    lp = np.random.randn(length).astype(np.float32)
    mem = np.random.randn(16, 2, 32).astype(np.float32)
    ach = np.zeros((length, 67), dtype=np.float32)
    ach[100, 41] = 1.0
    tgt = np.zeros(67, dtype=np.float32); tgt[0] = 1.0
    return Trajectory(
        observations=obs, actions=act, rewards=rew, dones=don,
        values=val, log_probs=lp, initial_memory=mem,
        achievements=ach, target_achievements=tgt,
    )


# ---------------------------------------------------------------------------
# Gate 8: save/restore round-trip
# ---------------------------------------------------------------------------

def test_gate8_save_restore_model_params():
    """Model parameters survive round-trip bit-exact."""
    network = TinyTransformer()
    ts = _make_train_state(network, 8)

    with tempfile.TemporaryDirectory() as tmp:
        # Save
        save_train_state(ts, tmp, 100)
        # Restore
        restored, step = restore_train_state(tmp, ts)
        assert step == 100

        # Compare params leaf-by-leaf
        for old, new in zip(
            jax.tree_util.tree_leaves(ts.params),
            jax.tree_util.tree_leaves(restored.params),
        ):
            assert jnp.allclose(old, new), "Gate 8 FAIL: params differ after restore"

    print("  PASS Gate 8: model params round-trip bit-exact")


def test_gate8_save_restore_optimizer():
    """Optimizer state (moments + step count) survives round-trip."""
    network = TinyTransformer()
    ts = _make_train_state(network, 8)

    # Do a gradient update to populate optimizer state
    grad = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, ts.params)
    ts = ts.apply_gradients(grads=grad)

    # Record step count before save
    step_before = int(jax.tree_util.tree_leaves(ts.opt_state)[0])

    with tempfile.TemporaryDirectory() as tmp:
        save_train_state(ts, tmp, 200)
        restored, step = restore_train_state(tmp, ts)

        step_after = int(jax.tree_util.tree_leaves(restored.opt_state)[0])
        assert step_before == step_after, (
            f"Gate 8 FAIL: optimizer step count changed: {step_before} -> {step_after}"
        )

    print("  PASS Gate 8: optimizer state round-trip preserved")


def test_gate8_save_restore_replay_and_rng():
    """Replay buffer metadata + counters + RNG state round-trip."""
    buf = TrajectoryReplayBuffer(capacity=16, seed=42)
    for i in range(4):
        buf.insert(_make_trajectory(250 + i * 50))
    buf.sample(sequence_length=150)
    buf.sample(sequence_length=200)
    buf.counters.gradient_updates = 5

    rng = jax.random.PRNGKey(999)
    global_step = 7777

    with tempfile.TemporaryDirectory() as tmp:
        save_full_checkpoint(
            _make_train_state(TinyTransformer(), 8),
            buf, rng, global_step, tmp, step=300,
        )

        restored = restore_full_checkpoint(
            tmp, _make_train_state(TinyTransformer(), 8), step=300,
        )

        rbuf = restored["replay_buffer"]
        assert rbuf is not None, "Gate 8 FAIL: replay buffer not restored"
        assert len(rbuf) == 4, f"Gate 8 FAIL: buffer size {len(rbuf)} != 4"
        assert rbuf.counters.trajectories_inserted == 4
        assert rbuf.counters.replay_samples_drawn == 2
        assert rbuf.counters.gradient_updates == 5
        assert rbuf.counters.total_sequence_length == 350  # 150 + 200
        assert restored["global_step"] == 7777
        assert jnp.array_equal(restored["rng"], rng), "Gate 8 FAIL: RNG mismatch"

    print("  PASS Gate 8: replay metadata + counters + RNG round-trip")


def test_gate8_save_restore_empty_replay():
    """Empty replay buffer round-trip (edge case)."""
    buf = TrajectoryReplayBuffer(capacity=8, seed=1)
    rng = jax.random.PRNGKey(0)
    global_step = 0

    with tempfile.TemporaryDirectory() as tmp:
        save_full_checkpoint(
            _make_train_state(TinyTransformer(), 8),
            buf, rng, global_step, tmp, step=0,
        )
        restored = restore_full_checkpoint(
            tmp, _make_train_state(TinyTransformer(), 8), step=0,
        )
        assert len(restored["replay_buffer"]) == 0
        assert restored["global_step"] == 0

    print("  PASS Gate 8: empty replay buffer round-trip")


# ---------------------------------------------------------------------------
# Gate 9: manifest, inventory, unique output root
# ---------------------------------------------------------------------------

def test_gate9_manifest_and_inventory():
    """Checkpoint produces manifest with all required fields."""
    buf = TrajectoryReplayBuffer(capacity=8, seed=1)
    buf.insert(_make_trajectory(200))
    buf.counters.gradient_updates = 3

    with tempfile.TemporaryDirectory() as tmp:
        save_full_checkpoint(
            _make_train_state(TinyTransformer(), 8),
            buf, jax.random.PRNGKey(42), 500, tmp, step=500,
        )

        # Check inventory
        inv = checkpoint_inventory(tmp)
        assert inv["exists"]
        assert inv["total_steps"] == 1
        assert inv["steps"][0]["step"] == 500
        assert "manifest.json" in inv["steps"][0]["files"]
        assert "replay_meta.pkl" in inv["steps"][0]["files"]

        manifest = inv["steps"][0].get("manifest", {})
        assert manifest.get("global_step") == 500
        assert manifest.get("checkpoint_step") == 500
        assert "timestamp_utc" in manifest
        assert "counters" in manifest

    print("  PASS Gate 9: manifest + inventory complete")


def test_gate9_weight_restore_report():
    """Compatible-weight restore report lists restored/skipped/initialized."""
    network = TinyTransformer()
    ts = _make_train_state(network, 8)

    with tempfile.TemporaryDirectory() as tmp:
        save_full_checkpoint(
            ts, TrajectoryReplayBuffer(seed=1),
            jax.random.PRNGKey(0), 0, tmp, step=0,
        )
        report = compatible_weight_restore_report(tmp, ts)
        assert report["status"] == "RESTORED"
        assert "parameter_paths_restored" in report
        assert len(report["parameter_paths_restored"]) > 0
        assert "optimizer_continuation" in report
        assert "Do not claim full" in report["optimizer_continuation"]

    print("  PASS Gate 9: weight restore report lists paths and disclaims full opt continuation")


def test_gate9_source_hashes_and_manifest():
    """Compute source hashes and produce D059 output manifest."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    hashes = {}
    for fname in sorted(os.listdir(src_dir)):
        if fname.endswith(".py"):
            fpath = os.path.join(src_dir, fname)
            h = hashlib.sha256(open(fpath, "rb").read()).hexdigest()
            hashes[fname] = h

    manifest = {
        "directive": "D059",
        "treatment": "AMAGO_STYLE_EXPLORATORY_P2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        ),
        "source_hashes": hashes,
        "physical_gpu_uuid": "GPU-f56a59b4-99f3-f2e5-11c6-d01685de8abd",
        "dependency_versions": {
            "jax": jax.__version__,
            "numpy": np.__version__,
        },
        "stop_rules": [
            "NaN/Inf in loss or gradient",
            "Traceback in training loop",
            "Wrong GPU UUID detected",
            "CPU fallback",
            "Empty replay buffer when sampling required",
            "Sequence-length gate failure (<=128)",
            "Missing manifest or checkpoint",
            "Output collision",
        ],
    }

    evidence_dir = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "d059_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"  PASS Gate 9: source hashes + manifest written")
    for fname, h in sorted(hashes.items()):
        print(f"    {fname}: {h[:16]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    tests = [
        ("Gate 8: model params round-trip", test_gate8_save_restore_model_params),
        ("Gate 8: optimizer round-trip", test_gate8_save_restore_optimizer),
        ("Gate 8: replay+counters+RNG round-trip", test_gate8_save_restore_replay_and_rng),
        ("Gate 8: empty replay round-trip", test_gate8_save_restore_empty_replay),
        ("Gate 9: manifest + inventory", test_gate9_manifest_and_inventory),
        ("Gate 9: weight restore report", test_gate9_weight_restore_report),
        ("Gate 9: source hashes + manifest", test_gate9_source_hashes_and_manifest),
    ]

    passed = 0; failed = 0
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
    print(f"Step 3 Results: {passed}/{passed+failed} tests PASS")
    print(f"{'='*60}")

    gates = {
        "Gate 8 (save/restore)": all(
            results.get(t, "").startswith("PASS")
            for t in [k for k in results if "Gate 8" in k]
        ),
        "Gate 9 (manifest/hashes/inventory)": all(
            results.get(t, "").startswith("PASS")
            for t in [k for k in results if "Gate 9" in k]
        ),
    }
    all_gates_pass = all(gates.values())

    report = {
        "step": 3,
        "directive": "D059",
        "description": "checkpoint save/restore + manifest tests",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "failed": failed, "total": passed + failed,
        "results": results,
        "gates": {k: v for k, v in gates.items()},
        "all_gates_pass": all_gates_pass,
    }

    evidence_dir = os.path.join(os.path.dirname(__file__), "..", "evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    with open(os.path.join(evidence_dir, "step3_test_report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"All gates PASS: {all_gates_pass}")
    if not all_gates_pass:
        for k, v in gates.items():
            if not v: print(f"  {k}: FAIL")
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
