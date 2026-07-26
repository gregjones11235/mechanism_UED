"""P2-v1 deterministic CPU tests.

Required coverage (user directive 2026-07-22):
  - transition alignment      (obs/action/reward/value/log_prob/next_obs)
  - memory alignment          (memory before action; terminal memory not reset)
  - on-policy update exists   (PPO main update runs even with empty replay)
  - relabel changes loss      (goal conditioning + recomputed reward change loss)
  - RNG exact resume          (checkpointable action RNG reproduces actions)
  - replay exact resume       (replay buffer RNG reproduces identical samples)
  - GAE reference             (JAX GAE == numpy reference; terminal bootstrap)
  - nonterminal bootstrap     (terminal -> 0, nonterminal -> V(next) != 0)
  - original PPO equivalence  (replay+hindsight off -> native PPO)
  - trajectory-id sampling    (_get_by_id / sample by trajectory id)

Runs on CPU only (JAX_PLATFORM_NAME=cpu).  No Craftax dependency: uses a
shape-correct test network and a deterministic fake vectorized env.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import distrax

from trajectory_replay import Trajectory, ReplaySample, TrajectoryReplayBuffer
from hindsight import relabel_sample
from long_context_learner import LongContextLearner, RolloutBatch, reference_gae
from rng_utils import (
    make_action_rng, action_rng_state, restore_action_rng, sample_actions,
)
from p2_v1_core import collect_rollout, p2_v1_update


# ---------------------------------------------------------------------------
# Shape-correct test network (matches Henry ActorCriticTransformer signatures)
# ---------------------------------------------------------------------------

class P2V1TestNet(nn.Module):
    action_dim: int = 6
    embed_size: int = 32
    num_layers: int = 2
    num_heads: int = 4
    window_mem: int = 16

    def setup(self):
        self.encoder = nn.Dense(self.embed_size)
        self.actor = nn.Dense(self.action_dim)
        self.critic = nn.Dense(1)

    def _encode(self, obs):
        return nn.relu(self.encoder(obs))

    def model_forward_eval(self, memories, obs, mask):
        x = self._encode(obs)  # [B, E]
        pi = distrax.Categorical(logits=self.actor(x))
        v = jnp.squeeze(self.critic(x), axis=-1)
        mem_out = jnp.broadcast_to(
            x[:, None, :], (x.shape[0], self.num_layers, self.embed_size)
        )
        return pi, v, mem_out

    def model_forward_train(self, memories, obs, mask):
        x = self._encode(obs)  # [B, (W,) E]
        pi = distrax.Categorical(logits=self.actor(x))
        v = jnp.squeeze(self.critic(x), axis=-1)
        return pi, v

    def __call__(self, memories, obs, mask):
        return self.model_forward_eval(memories, obs, mask)


class Cfg:
    gamma = 0.999
    gae_lambda = 0.8
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.002
    max_grad_norm = 1.0
    window_grad = 8
    window_mem = 16
    num_heads = 4
    embed_size = 32
    num_layers = 2
    num_steps = 16
    num_minibatches = 1
    update_epochs = 1
    value_target_clip_min = -50.0
    value_target_clip_max = 300.0
    max_policy_lag = 8


ACTION_DIM = 6
OBS_DIM = 4
NUM_ENVS = 2
ROLLOUT_STEPS = 16  # divisible by window_grad=8


def make_net():
    return P2V1TestNet(
        action_dim=ACTION_DIM, embed_size=32, num_layers=2, num_heads=4, window_mem=16
    )


def make_train_state(net, obs_dim=OBS_DIM, lr=1e-3, seed=0):
    rng = jax.random.PRNGKey(seed)
    init_obs = jnp.zeros((2, obs_dim))
    init_mem = jnp.zeros((2, 16, 2, 32))
    init_mask = jnp.zeros((2, 4, 1, 17), dtype=jnp.bool_)
    params = net.init(rng, init_mem, init_obs, init_mask)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr, eps=1e-5))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


# ---------------------------------------------------------------------------
# Deterministic fake vectorized env (obs encodes a per-env step counter)
# ---------------------------------------------------------------------------

class FakeEnvState:
    def __init__(self, counts):
        self.counts = counts  # np.ndarray [num_envs]


class FakeEnv:
    def __init__(self, num_envs, obs_dim, done_every=10_000):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.done_every = done_every

    def _obs(self, counts):
        counts = np.asarray(counts, dtype=np.float32)
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        for d in range(self.obs_dim):
            obs[:, d] = counts * (d + 1) * 0.1 + d
        return obs

    def reset(self, rng, params=None):
        counts = np.zeros(self.num_envs, dtype=np.int64)
        return self._obs(counts), FakeEnvState(counts)

    def step(self, rng, state, actions):
        actions = np.asarray(actions)
        new_counts = state.counts + 1
        reward = new_counts.astype(np.float32) + actions.astype(np.float32) * 0.01
        done = (new_counts % self.done_every) == 0
        next_obs = self._obs(new_counts)
        return next_obs, FakeEnvState(new_counts), reward, done, {}


def _run_collect(net, ts, env, action_seed=0, rollout_steps=ROLLOUT_STEPS,
                 done_every=10_000, collected_update_count=0, num_envs=NUM_ENVS):
    rng = jax.random.PRNGKey(123)
    obsv, env_state = env.reset(rng)
    memories = jnp.zeros((num_envs, 16, 2, 32))
    mem_mask = jnp.zeros((num_envs, 4, 1, 17), dtype=jnp.bool_)
    mem_idx = jnp.full((num_envs,), 17, dtype=jnp.int32)
    action_rng = make_action_rng(action_seed)
    target = np.zeros(67, dtype=np.float32)
    return collect_rollout(
        ts=ts, network=net, env=env, env_state=env_state, obsv=obsv,
        memories=memories, mem_mask=mem_mask, mem_idx=mem_idx, rng=rng,
        action_rng=action_rng, num_envs=num_envs, rollout_steps=rollout_steps,
        window_mem=16, num_heads=4, target_achievement=target,
        collected_update_count=collected_update_count,
    )


# ---------------------------------------------------------------------------
# 1. transition alignment
# ---------------------------------------------------------------------------

def test_transition_alignment():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=10_000)  # no dones in rollout
    roll = _run_collect(net, ts, env)
    batch = roll["batch"]

    obs = np.asarray(batch.obs)        # [E, T, obs_dim]
    next_obs = np.asarray(batch.next_obs)
    value = np.asarray(batch.value)
    logp = np.asarray(batch.log_prob)
    action = np.asarray(batch.action)

    # next_obs[t] is the decision obs of step t+1 (no off-by-one).
    assert np.allclose(next_obs[:, :-1], obs[:, 1:]), \
        "transition misaligned: next_obs[t] != obs[t+1]"

    # obs[0] is the reset obs (counter 0 -> obs[:,d] = d).
    for d in range(OBS_DIM):
        assert np.allclose(obs[:, 0, d], float(d)), "obs[0] is not the reset obs"

    # value[t] / log_prob[t] correspond to obs[t] (recompute the single-step
    # EVAL forward the collector used, with the stored pre-step memory/mask).
    mem_t = batch.memory            # [E, T, WM, L, Embed]
    mask_t = batch.mask             # [E, T, H, 1, WM+1]
    for t in range(ROLLOUT_STEPS):
        pi, v, _ = net.apply(
            ts.params, mem_t[:, t], obs[:, t], mask_t[:, t],
            method=net.model_forward_eval,
        )
        v = np.asarray(v).reshape(-1)
        assert np.allclose(v, value[:, t], atol=1e-5), \
            f"value[{t}] not aligned with obs[{t}]"
        lp = np.asarray(pi.log_prob(jnp.asarray(action[:, t]))).reshape(-1)
        assert np.allclose(lp, logp[:, t], atol=1e-5), \
            f"log_prob[{t}] not aligned with obs[{t}]/action[{t}]"

    print("  PASS transition alignment: obs/action/value/log_prob/next_obs aligned")


# ---------------------------------------------------------------------------
# 2. memory alignment
# ---------------------------------------------------------------------------

def test_memory_alignment():
    net = make_net()
    ts = make_train_state(net)
    env = FakeEnv(1, OBS_DIM, done_every=5)  # episodes of length 5
    roll = _run_collect(net, ts, env, rollout_steps=16, num_envs=1)
    batch = roll["batch"]
    trajs = roll["trajectories"]
    assert len(trajs) >= 1, "expected at least one completed episode"

    traj = trajs[0]
    L = traj.length
    assert L == 5, f"expected first episode length 5, got {L}"

    mem = np.asarray(batch.memory)[0]  # [T, WM, L, E] mem_pre per step

    # initial_memory is the memory BEFORE step 0.
    assert np.allclose(traj.initial_memory, mem[0]), \
        "initial_memory != memory before step 0"

    # memory before step t (t>=1) == memory AFTER step t-1 (memory_sequence[t-1]).
    for t in range(1, L):
        assert np.allclose(mem[t], traj.memory_sequence[t - 1]), \
            f"memory before step {t} != memory_sequence[{t-1}]"

    # Terminal transition memory is NOT overwritten by the reset:
    # memory_sequence[-1] (terminal post-memory) is nonzero, while the memory
    # carried into the next step (step L) is the reset-zero memory.
    assert np.any(np.asarray(traj.memory_sequence[-1]) != 0.0), \
        "terminal memory was zeroed (reset overwrote terminal transition memory)"
    assert np.allclose(mem[L], 0.0), \
        "memory carried into the post-terminal step is not the reset-zero memory"

    print("  PASS memory alignment: pre-action memory + terminal memory preserved")


# ---------------------------------------------------------------------------
# 3. on-policy update exists
# ---------------------------------------------------------------------------

def test_on_policy_update_exists():
    net = make_net()
    ts = make_train_state(net)
    learner = LongContextLearner(net, Cfg(), jax.random.PRNGKey(0))
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=10_000)
    roll = _run_collect(net, ts, env)
    batch = roll["batch"]

    empty_replay = TrajectoryReplayBuffer(capacity=8, seed=1)  # no samples
    params_before = jax.tree_util.tree_leaves(ts.params)
    ts2, metrics = p2_v1_update(
        ts, learner, batch, empty_replay, update_count=0, replay_aux=True
    )
    params_after = jax.tree_util.tree_leaves(ts2.params)

    assert metrics["on_policy_main_update"] is True
    assert metrics["replay_aux_update"] is False, "aux must not run on empty replay"
    assert metrics["ppo_params_changed"] is True
    assert np.isfinite(metrics["ppo_total_loss"])
    assert metrics["ppo_grad_norm"] > 1e-12
    assert any(bool(jnp.any(o != n)) for o, n in zip(params_before, params_after)), \
        "PPO main update did not change params"

    print("  PASS on-policy update exists: PPO main runs, empty replay skips aux")


# ---------------------------------------------------------------------------
# 4. relabel changes loss (goal conditioning + recomputed reward)
# ---------------------------------------------------------------------------

def _make_long_sample(obs_dim=75, embedding_size=67, length=160, achieved=(41,),
                      target=(0,)):
    raw = obs_dim - embedding_size
    rng = np.random.RandomState(0)
    obs = rng.randn(length, obs_dim).astype(np.float32)
    tgt0 = np.zeros(embedding_size, dtype=np.float32)
    for i in target:
        tgt0[i] = 1.0
    obs[:, raw:] = tgt0  # trailing embedding = original goal conditioning
    act = rng.randint(0, ACTION_DIM, size=length).astype(np.int32)
    rew = rng.randn(length).astype(np.float32) * 0.1
    don = np.zeros(length, dtype=bool); don[-1] = True
    val = rng.randn(length).astype(np.float32) * 0.1
    lp = rng.randn(length).astype(np.float32)
    mem = rng.randn(16, 2, 32).astype(np.float32) * 0.01
    mem_seq = rng.randn(length, 16, 2, 32).astype(np.float32) * 0.01
    next_obs = rng.randn(length, obs_dim).astype(np.float32)
    next_obs[:, raw:] = tgt0
    ach = np.zeros((length, embedding_size), dtype=np.float32)
    ach[min(80, length - 1), 41] = 1.0  # goal 41 literally achieved mid-episode
    tgt_full = np.zeros(embedding_size, dtype=np.float32)
    for i in target:
        tgt_full[i] = 1.0
    return ReplaySample(
        observations=obs, actions=act, rewards=rew, dones=don, values=val,
        log_probs=lp, initial_memory=mem, achievements=ach,
        target_achievements=tgt_full, source_trajectory_id=0, start_step=0,
        length=length, memory_sequence=mem_seq, next_observations=next_obs,
        next_value=0.0, episode_done=True, collected_update_count=0,
    )


def test_relabel_changes_loss():
    obs_dim = 75
    embedding_size = 67
    net = make_net()
    ts = make_train_state(net, obs_dim=obs_dim)
    learner = LongContextLearner(net, Cfg(), jax.random.PRNGKey(0))

    sample = _make_long_sample(obs_dim=obs_dim, embedding_size=embedding_size)

    # Original (target goal 0) auxiliary loss.
    total_orig, diag_orig = learner._replay_aux_loss(
        ts.params, sample, 0, False
    )

    # Relabel to the literally-achieved goal 41.
    relabeled = relabel_sample(sample, goal_index=41, embedding_size=embedding_size)

    # (a) goal conditioning actually changed the observation embedding.
    assert np.allclose(relabeled.observations[:, -embedding_size:],
                       np.eye(embedding_size, dtype=np.float32)[41]), \
        "goal conditioning (obs embedding) not relabeled to goal 41"
    assert not np.allclose(relabeled.observations, sample.observations), \
        "observations unchanged by relabel"

    # (b) reward recomputed under the new goal (differs from env/original reward).
    assert not np.allclose(relabeled.rewards, sample.rewards), \
        "rewards not recomputed for the new goal"
    assert relabeled.rewards.sum() > 0, "goal-progress reward should be positive"

    # (c) target one-hot moved to goal 41.
    assert relabeled.target_achievements[41] == 1.0
    assert relabeled.target_achievements[0] == 0.0

    # (d) the auxiliary loss changes after relabel.
    total_rl, diag_rl = learner._replay_aux_loss(
        ts.params, relabeled, 0, False
    )
    assert not np.allclose(float(total_orig), float(total_rl), atol=1e-8), \
        f"relabel did NOT change loss: {float(total_orig)} == {float(total_rl)}"
    assert diag_rl["hindsight_goal_index"] == 41

    # Gate 6 preserved: fabricated goal rejected.
    try:
        relabel_sample(sample, goal_index=8, embedding_size=embedding_size)
        assert False, "Gate 6 FAIL: fabricated goal accepted"
    except ValueError as e:
        assert "Gate 6" in str(e)

    print(f"  PASS relabel changes loss: {float(total_orig):.5f} -> {float(total_rl):.5f}")


# ---------------------------------------------------------------------------
# 5. RNG exact resume
# ---------------------------------------------------------------------------

def test_rng_exact_resume():
    probs = np.random.RandomState(3).rand(50, ACTION_DIM).astype(np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)

    rng = make_action_rng(seed=777)
    seq_a = [sample_actions(rng, probs[i:i + 1])[0] for i in range(50)]
    state = action_rng_state(rng)
    # continue sampling after the saved state
    seq_b = [sample_actions(rng, probs[i:i + 1])[0] for i in range(50)]

    # restore from the saved state -> must reproduce seq_b exactly
    rng2 = restore_action_rng(state, seed=0)
    seq_b2 = [sample_actions(rng2, probs[i:i + 1])[0] for i in range(50)]
    assert seq_b == seq_b2, "restored action RNG did not reproduce the action sequence"

    # fresh restore from same seed reproduces seq_a
    rng3 = restore_action_rng(None, seed=777)
    seq_a3 = [sample_actions(rng3, probs[i:i + 1])[0] for i in range(50)]
    assert seq_a == seq_a3, "same-seed action RNG did not reproduce the sequence"

    # global np.random is NOT used: perturbing it does not change the sequence
    rng4 = restore_action_rng(state, seed=0)
    np.random.seed(999)
    _ = np.random.rand(1000)
    seq_b4 = [sample_actions(rng4, probs[i:i + 1])[0] for i in range(50)]
    assert seq_b == seq_b4, "action sampling leaked to global np.random"

    print("  PASS RNG exact resume: action RNG checkpointable + independent of global")


# ---------------------------------------------------------------------------
# 6. replay exact resume
# ---------------------------------------------------------------------------

def _make_traj(length, seed, achieved=(41,)):
    r = np.random.RandomState(seed)
    obs = r.randn(length, OBS_DIM).astype(np.float32)
    act = r.randint(0, ACTION_DIM, size=length).astype(np.int32)
    rew = r.randn(length).astype(np.float32)
    don = np.zeros(length, dtype=bool); don[-1] = True
    val = r.randn(length).astype(np.float32)
    lp = r.randn(length).astype(np.float32)
    mem = r.randn(16, 2, 32).astype(np.float32)
    ach = np.zeros((length, 67), dtype=np.float32); ach[10, 41] = 1.0
    tgt = np.zeros(67, dtype=np.float32); tgt[0] = 1.0
    return Trajectory(
        observations=obs, actions=act, rewards=rew, dones=don, values=val,
        log_probs=lp, initial_memory=mem, achievements=ach,
        target_achievements=tgt,
    )


def test_replay_exact_resume():
    buf = TrajectoryReplayBuffer(capacity=16, seed=42)
    for i in range(4):
        buf.insert(_make_traj(200 + i * 30, seed=i))
    buf.sample()  # advance the replay RNG

    state = buf.state_dict()

    # Draw a sequence from the current RNG state.
    s1 = buf.sample()

    # Restore into a fresh buffer from the saved state -> identical next sample.
    buf2 = TrajectoryReplayBuffer.from_state_dict(state)
    s2 = buf2.sample()

    assert s1.source_trajectory_id == s2.source_trajectory_id
    assert s1.start_step == s2.start_step
    assert s1.length == s2.length
    assert np.array_equal(s1.observations, s2.observations)
    assert np.array_equal(s1.actions, s2.actions)

    print("  PASS replay exact resume: replay RNG state reproduces identical samples")


# ---------------------------------------------------------------------------
# 7. GAE reference
# ---------------------------------------------------------------------------

def test_gae_reference():
    learner = LongContextLearner(make_net(), Cfg(), jax.random.PRNGKey(0))
    r = np.random.RandomState(5)
    for L in (1, 7, 64, 129):
        rewards = r.randn(L).astype(np.float32)
        values = r.randn(L).astype(np.float32)
        dones = np.zeros(L, dtype=np.float32)
        dones[-1] = 1.0  # terminal
        next_value = 0.0  # terminal bootstrap
        adv_np, tgt_np = reference_gae(
            rewards, values, dones, next_value, learner.gamma, learner.gae_lambda
        )
        adv_j, tgt_j = learner._gae(
            jnp.asarray(rewards), jnp.asarray(values), jnp.asarray(dones),
            jnp.array(next_value),
        )
        assert np.allclose(np.asarray(adv_j), adv_np, atol=1e-4), \
            f"GAE advantages mismatch at L={L}"
        assert np.allclose(np.asarray(tgt_j), tgt_np, atol=1e-4), \
            f"GAE targets mismatch at L={L}"

    # Determinism: same input -> identical output.
    rewards = r.randn(32).astype(np.float32)
    values = r.randn(32).astype(np.float32)
    dones = np.zeros(32, dtype=np.float32)
    a1, t1 = learner._gae(jnp.asarray(rewards), jnp.asarray(values),
                          jnp.asarray(dones), jnp.array(1.5))
    a2, t2 = learner._gae(jnp.asarray(rewards), jnp.asarray(values),
                          jnp.asarray(dones), jnp.array(1.5))
    assert np.array_equal(np.asarray(a1), np.asarray(a2))
    assert np.array_equal(np.asarray(t1), np.asarray(t2))

    print("  PASS GAE reference: JAX GAE == numpy reference, deterministic")


# ---------------------------------------------------------------------------
# 8. nonterminal bootstrap
# ---------------------------------------------------------------------------

def test_nonterminal_bootstrap():
    buf = TrajectoryReplayBuffer(capacity=8, seed=9)
    traj = _make_traj(300, seed=1)
    buf.insert(traj)

    # Nonterminal slice: ends at step 150 (not done) -> next_value = values[150].
    s_nonterm = buf.sample(sequence_length=150, start_step=0)
    assert s_nonterm.episode_done is False
    assert abs(s_nonterm.next_value - float(traj.values[150])) < 1e-6, \
        "nonterminal bootstrap must equal V(next state)"
    assert s_nonterm.next_value != 0.0 or float(traj.values[150]) == 0.0

    # Terminal slice: ends at the episode end (done=True) -> next_value = 0.
    s_term = buf.sample(sequence_length=150, start_step=150)
    assert s_term.episode_done is True
    assert s_term.next_value == 0.0, "terminal bootstrap must be 0"

    print("  PASS nonterminal bootstrap: terminal->0, nonterminal->V(next)")


# ---------------------------------------------------------------------------
# 9. original PPO equivalence
# ---------------------------------------------------------------------------

def _make_rollout_batch(net, ts, seed=0):
    env = FakeEnv(NUM_ENVS, OBS_DIM, done_every=10_000)
    roll = _run_collect(net, ts, env, action_seed=seed)
    return roll["batch"]


def reference_henry_native_ppo_step(ts, net, learner, batch, advantages, targets):
    """Line-for-line restatement of Henry_work's REAL native on-policy PPO update.

    Mirrors dicode_v7fix58_armB/src/dicode/ppo_tr.py
    (sha256 faa561c0a78c7d7ea733cfb5ee61f2ca745dfd177c2cd6bca9681bb2a9077584),
    inner ``_update_minbatch`` loss at lines 582-616:

        ratio              = exp(log_prob - old_log_prob)            # L595
        gae_r              = (gae_r - mean) / (std + 1e-8)           # L596 (per-minibatch)
        loss_actor1        = ratio * gae_r                           # L597
        loss_actor2        = clip(ratio, 1-eps, 1+eps) * gae_r       # L598-599
        loss_actor         = -min(loss_actor1, loss_actor2).mean()   # L601
        value_pred_clipped = old_value + clip(value - old_value)     # L587-589
        value_losses       = (value - targets)^2                     # L590
        value_losses_clip  = (value_pred_clipped - targets)^2        # L591
        value_loss         = 0.5 * max(...).mean()                   # L592
        entropy            = pi.entropy().mean()                     # L603
        total              = loss_actor + vf_coef*value_loss
                             - ent_coef*entropy                       # L605
        grad_norm          = optax.global_norm(grads)                # L614
        train_state        = train_state.apply_gradients(grads)      # L615

    Optimizer (ppo_tr.py L209-216) is clip_by_global_norm(max_grad_norm) +
    adam(eps=1e-5) — exactly the tx in make_train_state. With num_minibatches =
    update_epochs = 1 (Cfg) this is one deterministic full-batch PPO step, so the
    reference and learner.ppo_update must agree bit-for-bit. Uses
    learner.windowize_batch only for the documented data layout; the loss math is
    restated here (not delegated to learner.native_ppo_loss) so this is a genuine
    independent cross-check of learner.ppo_update against Henry's real path.
    """
    win = learner.windowize_batch(batch)
    B, n_win, W = win["num_envs"], win["n_win"], win["window_grad"]
    rows = B * n_win
    adv = advantages[:, :n_win * W].reshape(rows, W)
    tgt = targets[:, :n_win * W].reshape(rows, W)
    eps = learner.clip_eps

    def loss_fn(params):
        pi, value = net.apply(
            params, win["memories"], win["obs"], win["mask"],
            method=net.model_forward_train,
        )
        log_prob = pi.log_prob(win["action"])
        ratio = jnp.exp(log_prob - win["log_prob"])                       # L595
        gae_r = (adv - adv.mean()) / (adv.std() + 1e-8)                   # L596
        loss_actor1 = ratio * gae_r                                        # L597
        loss_actor2 = jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * gae_r        # L598-599
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()         # L601
        value_pred_clipped = win["value"] + (value - win["value"]).clip(-eps, eps)
        value_losses = jnp.square(value - tgt)                             # L590
        value_losses_clipped = jnp.square(value_pred_clipped - tgt)        # L591
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()  # L592
        entropy = pi.entropy().mean()                                      # L603
        total = loss_actor + learner.vf_coef * value_loss - learner.ent_coef * entropy  # L605
        return total, (loss_actor, value_loss, entropy, gae_r)

    (total, (p_loss, v_loss, ent, adv_norm)), grads = \
        jax.value_and_grad(loss_fn, has_aux=True)(ts.params)
    grad_norm = optax.global_norm(grads)                                   # L614
    ts_after = ts.apply_gradients(grads=grads)                            # L615
    metrics = {
        "policy_loss": float(p_loss), "value_loss": float(v_loss),
        "entropy": float(ent), "total_loss": float(total),
        "grad_norm": float(grad_norm),
    }
    return ts_after, metrics, np.asarray(adv_norm)


def test_original_ppo_equivalence():
    """Directive 七-15: P2-v1 (replay+hindsight OFF) reduces to Henry native PPO.

    Compares learner.ppo_update against a line-for-line restatement of Henry
    ppo_tr.py's REAL update path (reference_henry_native_ppo_step), on identical
    initial params / rollout / advantages / RNG / optimizer (clip_by_global_norm +
    adam eps=1e-5), comparing the full required set: advantage, policy loss, value
    loss, entropy, gradients (grad_norm), and post-update params.
    """
    net = make_net()
    ts = make_train_state(net)
    learner = LongContextLearner(net, Cfg(), jax.random.PRNGKey(0))
    batch = _make_rollout_batch(net, ts, seed=11)

    advantages, targets = learner.compute_on_policy_gae(ts.params, batch)

    # (0) ADVANTAGE: P2-v1 on-policy GAE == independent numpy reference (per env),
    # using the same terminal-zero / V(next_obs) bootstrap the learner computes.
    final_next_obs = batch.next_obs[:, -1]
    final_done = batch.done[:, -1]
    _, boot, _ = net.apply(ts.params, batch.memory_after_final, final_next_obs,
                           batch.mask_after_final, method=net.model_forward_eval)
    boot_np = np.asarray(jnp.where(final_done, 0.0, boot))
    adv_np = np.asarray(advantages); tgt_np = np.asarray(targets)
    for e in range(batch.num_envs):
        ref_adv, ref_tgt = reference_gae(
            np.asarray(batch.reward[e]), np.asarray(batch.value[e]),
            np.asarray(batch.done[e]), float(boot_np[e]), Cfg.gamma, Cfg.gae_lambda)
        ref_tgt = np.clip(ref_tgt, Cfg.value_target_clip_min, Cfg.value_target_clip_max)
        assert np.allclose(adv_np[e], ref_adv, atol=1e-4), \
            f"env{e}: on-policy GAE advantage != numpy reference"
        assert np.allclose(tgt_np[e], ref_tgt, atol=1e-4), \
            f"env{e}: on-policy GAE target != numpy reference"

    # (A) learner PPO main update.
    ts_ppo, m_ppo = learner.ppo_update(ts, batch, advantages, targets)

    # (B) Henry-faithful reference native PPO (ppo_tr.py L582-616).
    ts_ref, ref_m, _ref_adv_norm = reference_henry_native_ppo_step(
        ts, net, learner, batch, advantages, targets)

    # (C) full P2-v1 update with replay + hindsight DISABLED.
    empty = TrajectoryReplayBuffer(capacity=4, seed=2)
    ts_v1, m_v1 = p2_v1_update(ts, learner, batch, empty, update_count=0,
                               replay_aux=False)

    # POLICY LOSS / VALUE LOSS / ENTROPY / GRADIENTS: learner == Henry path.
    assert np.isclose(m_ppo["ppo_policy_loss"], ref_m["policy_loss"], atol=1e-5), \
        f"policy_loss {m_ppo['ppo_policy_loss']} != Henry {ref_m['policy_loss']}"
    assert np.isclose(m_ppo["ppo_value_loss"], ref_m["value_loss"], atol=1e-5), \
        f"value_loss {m_ppo['ppo_value_loss']} != Henry {ref_m['value_loss']}"
    assert np.isclose(m_ppo["ppo_entropy"], ref_m["entropy"], atol=1e-5), \
        f"entropy {m_ppo['ppo_entropy']} != Henry {ref_m['entropy']}"
    assert np.isclose(m_ppo["ppo_total_loss"], ref_m["total_loss"], atol=1e-5), \
        f"total_loss {m_ppo['ppo_total_loss']} != Henry {ref_m['total_loss']}"
    assert np.isclose(m_ppo["ppo_grad_norm"], ref_m["grad_norm"], atol=1e-5), \
        f"grad_norm {m_ppo['ppo_grad_norm']} != Henry {ref_m['grad_norm']}"

    # POST-UPDATE PARAMS: learner == Henry (B), and learner == P2-v1 aux-off (C).
    leaves_ppo = jax.tree_util.tree_leaves(ts_ppo.params)
    leaves_ref = jax.tree_util.tree_leaves(ts_ref.params)
    leaves_v1 = jax.tree_util.tree_leaves(ts_v1.params)
    for a, b in zip(leaves_ppo, leaves_ref):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5), \
            "post-update params: learner.ppo_update != Henry native PPO"
    for a, b in zip(leaves_ppo, leaves_v1):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5), \
            "post-update params: P2-v1 (aux off) != native PPO main update"

    # Metrics confirm the native-PPO-only path.
    assert m_ppo["ppo_params_changed"] is True
    assert m_v1["on_policy_main_update"] is True
    assert m_v1["replay_aux_update"] is False
    assert "replay_aux_total_loss" not in m_v1
    assert "hindsight_relabelled" not in m_v1

    print("  PASS original PPO equivalence: advantage/policy/value/entropy/grad/"
          "post-update params all == Henry native PPO; P2-v1 (aux off) == native PPO")


# ---------------------------------------------------------------------------
# 10. trajectory-id sampling (_get_by_id)
# ---------------------------------------------------------------------------

def test_trajectory_id_sampling():
    buf = TrajectoryReplayBuffer(capacity=8, seed=13)
    ids = []
    for i in range(3):
        ids.append(buf.insert(_make_traj(180 + i * 20, seed=i)))

    # _get_by_id returns the exact stored trajectory.
    for tid in ids:
        t = buf._get_by_id(tid)
        assert t is not None, f"_get_by_id({tid}) returned None"
        assert t.trajectory_id == tid

    # Sampling by trajectory_id stays on that trajectory.
    tid = ids[1]
    s = buf.sample(trajectory_id=tid, sequence_length=150, start_step=0)
    assert s.source_trajectory_id == tid, \
        f"sampled trajectory {s.source_trajectory_id} != requested {tid}"

    # Unknown id raises.
    try:
        buf.sample(trajectory_id=99999, sequence_length=150)
        assert False, "expected error for unknown trajectory_id"
    except RuntimeError:
        pass

    print("  PASS trajectory-id sampling: _get_by_id + sample-by-id correct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TESTS = [
    ("transition alignment", test_transition_alignment),
    ("memory alignment", test_memory_alignment),
    ("on-policy update exists", test_on_policy_update_exists),
    ("relabel changes loss", test_relabel_changes_loss),
    ("RNG exact resume", test_rng_exact_resume),
    ("replay exact resume", test_replay_exact_resume),
    ("GAE reference", test_gae_reference),
    ("nonterminal bootstrap", test_nonterminal_bootstrap),
    ("original PPO equivalence", test_original_ppo_equivalence),
    ("trajectory-id sampling", test_trajectory_id_sampling),
]


def main():
    passed = 0
    failed = 0
    results = {}
    for name, fn in TESTS:
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
    print(f"P2-v1 CPU tests: {passed}/{passed + failed} PASS")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"  [{'OK' if res == 'PASS' else 'XX'}] {name}: {res}")

    if failed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
