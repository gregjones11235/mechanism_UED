# LONG_CONTEXT_ARCHITECTURE_BAKEOFF_PHASE1 – Frozen Design

**Status: FROZEN – 2026-07-25**
**Arms: GPU0 LC-W512-PPO, GPU1 LC-RMT16-PPO**

> 冻结后不得修改架构、超参或融合方式。

---

## 1. LC-W512 (GPU0)

### Architecture
```
obs → encoder(Dense 8335→256) → GTrXL(128-step, 2 layers) → h_t
                                                               │
                     ┌─────────────────────────────────────────┘
                     ▼
           cross-attention(query=h_t, kv=long_buf+posenc)
                     │
                     ▼
           z_t = h_t + tanh(w512_gate) × long_ctx_t
                     │
             ┌───────┴───────┐
             ▼               ▼
        Actor head      Value head
```

- **GTrXL**: 128-step window, UNCHANGED from ckpt17500
- **Long buffer**: 384-step ring buffer of raw h_t (GTrXL output)
- **Delay line**: 128-step FIFO; h_t enters delay, exits into long buffer after 128 steps
  - Ensures long buffer contains h_{t-512}..h_{t-129} (NO overlap with GTrXL window)
- **Positional encoding**: fixed sinusoidal (384, 256), added to buffer entries before cross-attention
- **Cross-attention**: nn.MultiHeadDotProductAttention(8 heads, 256 qkv, 256 out)
- **LayerNorm**: after cross-attention, before fusion
- **Gate**: scalar, zero-init → z_t = h_t at init (bit-exact)
- **No summary compression** – raw hidden states only

### New parameters
| Module | Shape | Init |
|---|---|---|
| w512_cross_attn (MHA) | QKV proj + out proj | Flax default |
| w512_ln (LayerNorm) | scale(256) + bias(256) | ones, zeros |
| w512_gate (scalar) | (1,) | **zeros** |
| w512_posenc (fixed) | (384, 256) | sinusoidal (non-trainable) |

### State per env
- delay_buf (128, 256), delay_idx, delay_count
- long_buf (384, 256), long_mask (384,), long_idx
- Rollout boundaries: NO reset. True done: full reset.

---

## 2. LC-RMT16 (GPU1)

### Architecture
```
obs → encoder → GTrXL(128-step) → h_t
                                     │
              ┌──────────────────────┘
              ▼
    cross-attention(query=h_t, kv=mem_tokens)    ← read (every step)
              │
              ▼
    z_t = h_t + tanh(rmt_gate) × rmt_ctx_t
              │
      ┌───────┴───────┐
      ▼               ▼
  Actor head      Value head

Every 128 steps (segment boundary):
  mem_tokens ← mem_tokens + LN(attn(query=mem_tokens, kv=seg_buf))
```

- **GTrXL**: 128-step window, UNCHANGED
- **Memory tokens**: 16 persistent tokens (256-dim) per env
- **Read**: cross-attention(query=h_t, kv=mem_tokens) at every step
- **Update**: at segment boundary (128 steps), residual attention update
  - query=mem_tokens (16, 256), kv=segment_h_buf (128, 256)
  - new_tokens = old_tokens + LN(attn_out)
  - No fixed-period averaging
- **Gate**: scalar, zero-init → bit-exact at init

### New parameters
| Module | Shape | Init |
|---|---|---|
| rmt_read_attn (MHA) | QKV proj + out proj | Flax default |
| rmt_read_ln (LayerNorm) | scale(256) + bias(256) | ones, zeros |
| rmt_update_attn (MHA) | QKV proj + out proj | Flax default |
| rmt_update_ln (LayerNorm) | scale(256) + bias(256) | ones, zeros |
| rmt_gate (scalar) | (1,) | **zeros** |

### State per env
- mem_tokens (16, 256)
- seg_buf (128, 256), seg_count
- Rollout boundaries: NO reset of mem_tokens. True done: full reset.

---

## 3. Shared Training Configuration

- Start: ckpt17500 (base params loaded, new modules initialized, gate=0)
- Seed: MASTER_SEED = 42
- XLA: `--xla_gpu_deterministic_ops=true`
- Stage4-native, goal = DEFEAT_KOBOLD
- **total_steps = 24576 updates** (= 24576 × 16 × 128 env steps)
- Checkpoint schedule: **0 / 4096 / 24576**
- Original PPO only: update_epochs=1, num_minibatches=2, lr=2e-5, Adam eps=1e-5
- gamma=0.999, gae_lambda=0.8, clip=0.2, ent_coef=0.002, vf_coef=0.5
- max_grad_norm=1.0, num_envs=16, num_steps=128
- optimistic_reset_ratio=16, mode=score
- **NOT enabled**: replay, V-trace, AWR, hindsight, novelty, NavAux, EgoMap

---

## 4. Engineering Gates (10 per arm)

| Gate | Description |
|---|---|
| G1 | feature-off bit-exact (gate=0 → identical to ckpt17500) |
| G2 | vector env isolation (no cross-env contamination) |
| G3 | rollout state continuity (state persists across rollout boundary) |
| G4 | true done reset (done clears state for that env only) |
| G5 | checkpoint roundtrip (pickle bit-exact) |
| G6 | exact resume (same seed → same params after N updates) |
| G7 | deterministic 4096 smoke (finite, no NaN/Inf, entropy > 0.1) |
| G8 | memory path has finite non-zero gradients |
| G9 | no NaN/Inf and no entropy collapse |
| G10 | zeroed long-context memory → action KL non-zero (with gate=1) |

**失败候选停止，不影响另一候选。**

---

## 5. Evaluation Protocol

### Frozen anchors (reproduce first)
- Baseline ckpt17500
- Canonical Control @24576

### Arm evaluation
- LC-W512 @24576, LC-RMT16 @24576
- 256 worlds, seed_base=100000, stochastic, max_steps=4096/world
- Stage4-native, spawn_floor=2, DEFEAT_KOBOLD ever-set

### Metrics
- DK SR, floor3, conditional kill, death/timeout, episode length
- Paired CI and McNemar

### Ablation (4 conditions)
| Condition | Description |
|---|---|
| A | Full long-context memory (intact) |
| B | Long-context memory zeroed |
| C | History/token order shuffled |
| D | Only last 128 steps (short-term only) |

Output: action KL, top-action flip rate, value diff, behavioral results.

---

## 6. Candidate Qualification

Must satisfy ALL:
- SR relative to Control drops ≤ 5pp
- floor3 ≥ 90%
- No significant death rate worsening
- At least one directional positive performance signal
- Long-context memory demonstrably used by Actor (ablation A vs B/C/D)
- All 10 engineering gates passed

### Final output
```
CANDIDATE_GPU0=QUALIFIED/REJECTED
CANDIDATE_GPU1=QUALIFIED/REJECTED
```

**不得自行决定LC-WINNER。不得训练到98304。不得启动P2-Full-B。**

---

## 7. File Inventory

| File | Purpose |
|---|---|
| w512_memory.py | W512 state: delay line + long buffer |
| rmt16_memory.py | RMT16 state: memory tokens + segment buffer |
| network_w512.py | ActorCriticTransformerW512 |
| network_rmt16.py | ActorCriticTransformerRMT16 |
| ppo_tr_w512.py | W512-augmented PPO trainer |
| ppo_tr_rmt16.py | RMT16-augmented PPO trainer |
| launcher_bakeoff.py | Unified launcher (--arm w512/rmt16) |
| gate_tests_bakeoff.py | 10 engineering gates (--arm w512/rmt16) |
| bakeoff_frozen_design.md | This document |
