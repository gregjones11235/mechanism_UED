# Henry P2 Native Eval Extraction Report (D068)

## Verdict: EXTRACTABLE

Henry P2 (`henry_student_p2_amago_20260721`) training code at
`src/stage_b_launcher.py` contains a complete, extractable native eval/rollout
path suitable for building a read-only evaluation adapter.

## Extracted Interfaces

### 1. ActorCriticTransformer Construction

Source: `stage_b_launcher.py` lines 152-157, `Cfg` class lines 78-96

```python
network = ActorCriticTransformer(
    action_dim=43,          # from env.action_space(env_params).n
    activation="relu",
    hidden_layers=256,
    encoder_size=256,       # = embed_size
    num_heads=8,
    qkv_features=256,
    num_layers=2,
    gating=True,
    gating_bias=2.0,
)
```

### 2. Observation Shape

Source: `stage_b_launcher.py` line 202 runtime

```
obs_dim = 8335  # CraftaxAugObsTrain + embedding conditioning
```

### 3. Memory Initialization (exact training mirror)

Source: `stage_b_launcher.py` lines 208-210

```python
memories  = jnp.zeros((num_envs, window_mem=128, num_layers=2, embed_size=256))
           # shape: (16, 128, 2, 256)
mem_mask  = jnp.zeros((num_envs, num_heads=8, 1, window_mem+1=129), dtype=bool_)
           # shape: (16, 8, 1, 129)
mem_idx   = jnp.full((num_envs,), window_mem + 1, dtype=int32)
           # shape: (16,)
```

### 4. model_forward_eval Signature

Source: `network.py` lines 171-188

```python
def model_forward_eval(self, memories, obs, mask):
    x, memory_out = self.transformer.forward_eval(memories, obs, mask)
    # x: (batch, embed_size)
    # memory_out: (batch, num_layers, embed_size) = (16, 2, 256)
    # ... actor/critic heads ...
    return pi, value, memory_out
```

Call signature (stage_b lines 190-193):
```python
@jax.jit
def jit_forward(params, mem, obs, mask):
    pi, value, mem_out = network.apply(params, mem, obs, mask,
                                       method=network.model_forward_eval)
    return pi.logits, value, mem_out
```

### 5. Memory Update Logic (per-timestep)

Source: `stage_b_launcher.py` lines 236-276 (exact mirror)

```python
# 1. Update memory mask
mem_idx = jnp.clip(mem_idx - 1, 0, window_mem)
ohot = jax.nn.one_hot(mem_idx, window_mem + 1)
ohot = ohot[:, None, None, :].repeat(num_heads, 1)
mem_mask = jnp.logical_or(mem_mask, ohot)

# 2. Forward pass
logits, value, mem_out = jit_forward(params, memories, obs, mem_mask)

# 3. Update memory (sliding window + set last position)
memories = jnp.roll(mem_out, -1, axis=1)  # wait — is this right?

# Actually stage_b does: memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
```

**CRITICAL NOTE**: `mem_out` from `forward_eval` has shape `(batch, num_layers, embed_size)` =
`(16, 2, 256)`, NOT `(16, 128, 2, 256)`. The sliding window update is:
```python
memories = jnp.roll(memories, -1, axis=1).at[:, -1].set(mem_out)
```
This rolls the `(16, 128, 2, 256)` buffer left by 1, then places the new 3D
`(16, 2, 256)` memory at the last window position.

On episode done: memory is reset to zeros via `jnp.where(done, zeros, memories)`.

### 6. Environment Setup

Source: `stage_b_launcher.py` lines 163-179

```python
# Task from evidence/s4_task_code.py (class Env(BaseTask): DEFEAT_KOBOLD)
# Env wraps:
#   MultiTaskMiniCraftaxEnv([Task, ...], ...)
#   → DistributedMultiTaskOptimisticLogWrapper(..., num_envs=16, reset_ratio=16)

static_env_params = StaticEnvParams()
env_params = EnvParams(max_timesteps=4096)
base_env = MultiTaskMiniCraftaxEnv(
    [Task], static_env_params, env_params,
    condition_on_task=True, conditioning_type="embedding",
    embedding_size=67,  # achievement multi-hot dim
    completion_bonus_scale=0.0, completion_bonus_min=0.0,
    bonus_type="none", dynamic_bonus_k=0.0)
env = DistributedMultiTaskOptimisticLogWrapper(
    base_env, PRNGKey(0), num_envs=16, num_tasks=1,
    optimistic_reset_ratio=16,
    task_distribution=jnp.array([1.0]),
    task_embeddings=ach_table)  # shape (1, 67)
```

### 7. Achievement Extraction

Source: `stage_b_launcher.py` lines 278-286

```python
ach_data = np.zeros((num_envs, 67), dtype=np.float32)
est = env_state.env_state  # inner Craftax state
if hasattr(est, 'achievements'):
    ach_data = np.asarray(est.achievements).astype(np.float32)
```

Achievement index for `DEFEAT_KOBOLD`:
```python
from craftax.craftax.constants import Achievement
# Achievement.DEFEAT_KOBOLD index determined by get_achievement_multi_hot
```

### 8. Checkpoint Restore

Source: `stage_b_launcher.py` lines 140-147, `train_state_utils.py`

```python
ts = load_weights_only(CKPT_PATH, dummy_env, dummy_env.default_params, cfg,
                       load_opt_state=False)
# Returns TrainState with params loaded, opt_state fresh (step=0)
# Param leaves: 80
```

### 9. Known Issues

1. **forward_eval squeeze(0) bug**: The transformer's `forward_eval` uses
   `x.squeeze(0)` which eliminates the batch dimension when `num_envs=1`.
   This requires `num_envs >= 2` for the evaluator. Training uses 16 envs
   so never hits this. **Evaluator must use 16 envs.**

2. **optimistic_reset_ratio must divide num_envs**: The wrapper requires
   `num_envs % reset_ratio == 0`. With `reset_ratio=16`, only 16 envs works.

## Extracted Native Eval Path (Pseudocode)

```
for episode in 1..N:
    obs, env_state = env.reset(rng)
    memories = zeros(16, 128, 2, 256)
    mem_mask = zeros(16, 8, 1, 129)
    mem_idx = full(16, 129)
    done = False
    steps = 0
    while not done and steps < 4096:
        # Memory mask update (exact mirror of stage_b)
        mem_idx = clip(mem_idx - 1, 0, 128)
        ohot = one_hot(mem_idx, 129)[:, None, None, :] * (8 heads)
        mem_mask = logical_or(mem_mask, ohot)
        # Forward
        logits, value, mem_out = model_forward_eval(params, memories, obs, mem_mask)
        # Action (greedy for eval)
        action = argmax(logits, axis=-1)
        # Memory update (sliding window)
        memories = roll(memories, -1, axis=1).at[:, -1].set(mem_out)
        # Env step
        obs, env_state, reward, done, info = env.step(rng, env_state, action)
        # Reset memory for done envs
        memories = where(done[:,None,None,None], zeros, memories)
        mem_mask = where(done[:,None,None,None], zeros, mem_mask)
        mem_idx = where(done, 128, mem_idx)
        steps += 1
    # Achievement check
    achievements = env_state.env_state.achievements
    tier3_success = achievements[DEFEAT_KOBOLD_INDEX] > 0
```
