# CC1 A-Side Long Memory Corrected Evaluation & Causal Carry Report

**Date**: 2026-07-25
**Director**: CC1 (GPU0 + GPU1 only)
**Label**: LONG_MEMORY_BAKEOFF_A_SIDE_CORRECTED_EVAL

---

## 1. Executive Summary

Five-arm (Baseline, Control, W512-Persistent, W512-Reset128, RMT16-Persistent, RMT16-Reset128) frozen 256-world evaluation completed. All anchors pass. Inference-time ablations A–E completed for both Persistent architectures.

**Verdict: LONG_MEMORY_WINNER_A_SIDE = NONE**

Both long-memory architectures show a **significant positive carry effect** (persistent cross-rollout memory helps), but the **architecture modifications themselves hurt so much** that the net effect is negative. Neither architecture beats the Control baseline.

---

## 2. Evaluation Protocol (Frozen)

| Parameter | Value |
|-----------|-------|
| Worlds | 256 |
| Seed | 42 |
| Policy | stochastic |
| Spawn floor | 2 (S4_dark native start) |
| Max steps | 4096 |
| Pre-step achievements | yes |
| Wrapper | DistributedMultiTaskOptimisticLogWrapper |
| GPU | GPU-3c7a2864-755b-7045-b293-6f80e748283f (GPU1) |
| Evaluator SHA256 | dcf7fe207bb485c47b2669e6c0eb187556d1a4724dd3417a81a83fc88abe5828 |
| Eval step | 24576 |

### Anchors

| Anchor | n_success / 256 | Expected | Status |
|--------|----------------|----------|--------|
| Baseline (ckpt17500) | 101 | 101 | **PASS** |
| Control @24576 | 93 | 93 | **PASS** |

---

## 3. Five-Arm Results (256 worlds, paired by world index)

| Arm | DK SR | n_success | Floor3 Rate | n_floor3 | Cond. Kill | Death | Timeout | Sewers | Ep Len |
|-----|-------|-----------|-------------|----------|------------|-------|---------|--------|--------|
| Baseline (ckpt17500) | 39.45% | 101/256 | 43.36% | 111 | 91.0% | 147 | 8 | 111 | 986 |
| Control @24576 | 36.33% | 93/256 | 43.36% | 111 | 83.8% | 156 | 7 | 111 | 862 |
| W512-Persistent (on) | 10.94% | 28/256 | 17.58% | 45 | 62.2% | 226 | 2 | 45 | 622 |
| W512-Persistent (off) | 11.33% | 29/256 | 16.80% | 43 | 67.4% | 226 | 1 | 43 | 663 |
| W512-Reset128 (on) | **2.73%** | **7/256** | 6.64% | 17 | 41.2% | 249 | 0 | 17 | 575 |
| RMT16-Persistent (on) | 27.34% | 70/256 | 32.03% | 82 | 85.4% | 182 | 4 | 82 | 893 |
| RMT16-Persistent (off) | 25.78% | 66/256 | 29.69% | 76 | 86.8% | 187 | 3 | 76 | 864 |
| RMT16-Reset128 (on) | **11.33%** | **29/256** | 19.14% | 49 | 59.2% | 224 | 3 | 49 | 897 |

---

## 4. Causal Decomposition (Paired, + = first arm better)

### W512

| Quantity | dSR (pp) | dFloor3 (pp) | McNemar p | 95% CI |
|----------|----------|-------------|-----------|--------|
| **Carry** = Persistent − Reset128 | **+8.20** | +10.94 | 6.3e-05 | [+4.69, +12.11] |
| **Arch** = Reset128 − Control | **−33.59** | −36.72 | <1e-10 | [−39.45, −27.73] |
| **Total** = Persistent − Control | **−25.39** | −25.78 | <1e-10 | [−31.64, −19.14] |

### RMT16

| Quantity | dSR (pp) | dFloor3 (pp) | McNemar p | 95% CI |
|----------|----------|-------------|-----------|--------|
| **Carry** = Persistent − Reset128 | **+16.02** | +12.89 | 2e-06 | [+10.16, +22.27] |
| **Arch** = Reset128 − Control | **−25.00** | −24.22 | <1e-10 | [−31.64, −18.36] |
| **Total** = Persistent − Control | **−8.98** | −11.33 | 0.017 | [−16.02, −1.95] |

### Interpretation

- **Carry is real and significant** for both architectures: persistent cross-rollout memory improves SR by +8.2pp (W512) and +16.0pp (RMT16) over resetting every 128 steps.
- **Architecture cost dominates**: both W512 and RMT16 modifications to the base GTrXL-128 architecture impose a massive performance penalty (−33.6pp and −25.0pp respectively vs Control), even when trained WITHOUT carry (Reset128 mode).
- **Net effect is negative**: even with carry benefit, neither architecture approaches Control performance.
- **RMT16 is the better architecture**: its total deficit (−8.98pp) is much smaller than W512's (−25.39pp), and its carry benefit is larger (+16.02pp vs +8.20pp).

---

## 5. Inference-Time Ablations (A–E)

### W512-Persistent

| Mode | Description | DK SR | n_success | Floor3 |
|------|-------------|-------|-----------|--------|
| A: on | Full persistent long state | 10.94% | 28/256 | 17.58% |
| B: off | Zeroed long state input | 11.33% | 29/256 | 16.80% |
| C: reset128 | Zero long state every 128 steps | 11.33% | 29/256 | 16.80% |
| D: gtrxl | Same as off (GTrXL-128 only) | 11.33% | 29/256 | 16.80% |
| E: shuffle | Permuted long state across batch | 10.94% | 28/256 | 16.02% |

**Finding**: W512's long_buf/long_mask provide **no useful information at inference time**. All ablation modes give ~11% SR, same as "off". The carry benefit is a **training-time effect** (the network learned differently with carry during training), not an inference-time information effect.

### RMT16-Persistent

| Mode | Description | DK SR | n_success | Floor3 |
|------|-------------|-------|-----------|--------|
| A: on | Full persistent mem_tokens | 27.34% | 70/256 | 32.03% |
| B: off | Zeroed mem_tokens input | 25.78% | 66/256 | 29.69% |
| C: reset128 | Zero mem_tokens every 128 steps | 25.78% | 66/256 | 29.69% |
| D: gtrxl | Same as off (GTrXL-128 only) | 25.78% | 66/256 | 29.69% |
| E: shuffle | Permuted mem_tokens across batch | 23.83% | 61/256 | 28.12% |

**Finding**: RMT16's mem_tokens provide a **small but measurable inference-time benefit** (+1.56pp SR, on vs off). Shuffling hurts slightly more (−3.51pp vs on), suggesting the network partially uses world-specific memory token content at inference. However, the majority of RMT16's carry benefit (+16.02pp) is still a **training-time effect**.

---

## 6. Qualification Assessment

| Condition | W512 | RMT16 | Description |
|-----------|------|-------|-------------|
| c1: carry > 0 | ✅ | ✅ | Persistent > Reset128 |
| c2: carry significant | ✅ | ✅ | McNemar p < 0.05 |
| c3: total < 0 | ✅ | ✅ | Persistent < Control |
| c4: arch < 0 | ✅ | ✅ | Reset128 < Control |
| c5: total > 0 | ❌ | ❌ | Neither beats Control |
| c6: arch ≥ 0 | ❌ | ❌ | Architecture hurts |

**W512: W512_NO_SIGNAL** (carry helps, but architecture cost dominates → net negative)
**RMT16: RMT16_NO_SIGNAL** (carry helps, but architecture cost dominates → net negative)

---

## 7. Reset128 Training Summary

### W512-Reset128 (GPU0)

| Step | Actor Loss | Entropy | Grad Norm Max | Time (s) |
|------|-----------|---------|---------------|----------|
| 4096 | 0.0100 | 0.8512 | 417.19 | 2817 |
| 8192 | 0.0039 | 0.8065 | 195.48 | 2760 |
| 12288 | 0.0036 | 0.8797 | 451.01 | 2775 |
| 16384 | 0.0021 | 0.8235 | 130.05 | 2797 |
| 20480 | 0.0022 | 0.8355 | 1090.30 | 2886 |
| 24576 | 0.0023 | 0.4492 | 434.19 | 2853 |

### RMT16-Reset128 (GPU1)

| Step | Actor Loss | Entropy | Grad Norm Max | Time (s) |
|------|-----------|---------|---------------|----------|
| 4096 | 0.0096 | 0.8753 | 194.56 | 2867 |
| 8192 | 0.0049 | 0.5495 | 179.72 | 2727 |
| 12288 | 0.0046 | 0.7374 | 887.34 | 2771 |
| 16384 | 0.0040 | 0.8750 | 156.02 | 2795 |
| 20480 | 0.0042 | 0.7877 | 153.71 | 2760 |
| 24576 | 0.0046 | 0.8800 | 4475.41 | 2731 |

---

## 8. RMT16 Variable Name Fix Record

**Bug**: `make_reset128_variants.py` generated RMT16-Reset128 code using variable name `rst`, but the actual carry unpacking variable in `ppo_tr_rmt16.py` is `rmt_st`.

**Symptom**: `UnboundLocalError: local variable 'rst' referenced before assignment` at training start.

**Fix**: Direct sed replacement on server:
- `rst = {**rst,` → `rmt_st = {**rmt_st,`
- `rst["mem_tokens"]` → `rmt_st["mem_tokens"]`

**Impact assessment**: This was **purely a variable name error** in the generated Reset128 insertion code. The fix:
- Did NOT change the network architecture
- Did NOT change the state lifecycle (mem_tokens still cleared at 128-step boundaries)
- Did NOT change the loss function
- Did NOT change the optimizer
- Did NOT change any hyperparameters
- Only corrected the Python variable reference to match the existing carry unpacking

---

## 9. Frozen Artifacts

### Code SHA256

| File | SHA256 |
|------|--------|
| ppo_tr_w512_reset128.py | 6c7c0e36802de1c25a149d061cabf5381b256a90bd7c1b2abfb59449cadb9693 |
| ppo_tr_rmt16_reset128.py | 2820f8b7bc89a5c64dc5820ec4aed061a9003a60af13cf6fdc31fb1f46673f13 |
| launcher_w512_reset128.py | c17313ad9cfcfab29ad8de9832f196ddcb495eb7c964c4714bf844f9bb1f0dce |
| launcher_rmt16_reset128.py | 8039125468483fc7b1d0d2e7095732da4beb271b3a8933de8e5c9b5326c817cf |
| eval_a_side_unified.py | dcf7fe207bb485c47b2669e6c0eb187556d1a4724dd3417a81a83fc88abe5828 |
| ppo_tr_w512.py (Persistent) | b8590c48a8e3f5d0778c023f1693c53fbe9b0976d8919f446f7a537d8aca8f95 |
| ppo_tr_rmt16.py (Persistent) | d01e23d42d5e33a7019e24da2404bf7daadd3f704e01bd17a5276709bd744dd1 |

### Checkpoint SHA256

| Checkpoint | params.pkl SHA256 | Manifest SHA256 (in params_sha256 field) |
|------------|-------------------|----------------------------------------|
| W512-Reset128 @24576 | 9290d066b030bc18bc42d3eec65e9e3f9e995c68cbf115bd805a7c9fbbbfdccb | 51388e219719d52ec2bfe36e5a15488d19284c1571d48a1fe30021a12fc9dfbc |
| RMT16-Reset128 @24576 | cad939bb56cf89aae8e948f657deb780e68e459e213335b13f934c19da63dca2 | b4ef22bceb93c537938107bc11d12cc5fcd4026745c294311998ce22fb0b5010 |
| W512-Persistent @24576 | (frozen earlier) | 8c84287c4bff2032... |
| RMT16-Persistent @24576 | (frozen earlier) | 102f30f9186957f0... |

### Memory Schema

| Architecture | Long State Fields | Reset128 Clear Target |
|-------------|-------------------|----------------------|
| W512 | long_buf (384-dim), long_mask, delay_buf | long_buf, long_mask (delay_buf preserved) |
| RMT16 | mem_tokens (16 tokens), seg_buf | mem_tokens (seg_buf preserved) |

### Config Diff: Persistent vs Reset128

The ONLY difference between Persistent and Reset128 training code is the insertion of a boundary-reset block after the carry unpacking in `_env_step`:

```python
# RESET128: clear [long_buf/long_mask | mem_tokens] at rollout boundary
at_boundary = jnp.logical_and(jnp.greater(step_loop, 0),
                               jnp.equal(jnp.mod(step_loop, config.num_steps), 0))
[arch_state] = {**[arch_state],
    "[field]": jnp.where(at_boundary, jnp.zeros_like(...), ...)}
```

All other code (network, loss, optimizer, hyperparameters, data pipeline) is identical.

---

## 10. Key Scientific Conclusions

1. **Cross-rollout memory carry is causally beneficial** for both W512 (+8.2pp) and RMT16 (+16.0pp), with high statistical significance (McNemar p < 1e-4).

2. **The benefit is primarily a training-time effect**: the network learns better representations when trained with persistent carry, even though the carried information provides little to no additional signal at inference time (especially for W512).

3. **Architecture modifications impose a large cost**: both W512 and RMT16 architectural changes to the base GTrXL-128 dramatically reduce performance (−33.6pp and −25.0pp respectively), even without carry. This suggests the long-memory mechanisms themselves interfere with the base architecture's learned representations.

4. **RMT16 is the more promising architecture**: smaller total deficit (−8.98pp vs −25.39pp), larger carry benefit (+16.02pp vs +8.20pp), and some inference-time utility from mem_tokens.

5. **Neither architecture beats Control at 24576 steps**: the carry benefit does not compensate for the architecture cost within the training budget tested.

6. **The carry × architecture interaction is the key open question for Part B**: can P2 Replay rescue the architecture cost by providing off-policy correction? This motivates the W512 × P2 Replay 2×2 experiment.

---

## 11. Frozen Status

- **TRAINING_TO_98304_AUTHORIZED = false**
- **P2_FULL_B_FINAL_AUTHORIZED = false**
- **UPDATE_HORIZON_PHASE_AUTHORIZED = false**
- **Part A: COMPLETE AND FROZEN**
- **Part B: AUTHORIZED TO PROCEED** (after this freeze)
