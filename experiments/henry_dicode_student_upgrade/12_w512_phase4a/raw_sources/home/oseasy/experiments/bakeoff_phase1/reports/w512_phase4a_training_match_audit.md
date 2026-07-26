# W512 Phase4A Training Match Audit

**Verdict: W512_PHASE4A_MATCHED_DESIGN=PASS**

## Summary

20/20 checks passed. Two cosmetic differences noted (SAVE_STEPS frequency, GPU assignment).

## Core Matched Design

| Property | Persistent | Reset128 | Match? |
|----------|-----------|----------|--------|
| Init SHA (P2Replay step0) | 5942526301c66766... | 5942526301c66766... | ✓ |
| Network | ActorCriticTransformerW512 | same | ✓ |
| Param count | 5,268,013 | 5,268,013 | ✓ |
| Init source | ckpt17500 + W512 zero-init | same | ✓ |
| Seed | 42 | 42 | ✓ |
| LR | 2e-5 | 2e-5 | ✓ |
| num_envs | 16 | 16 | ✓ |
| rollout_steps | 128 | 128 | ✓ |
| Replay capacity | 64 | 64 | ✓ |
| L_SEQ | 129 | 129 | ✓ |
| K_BATCH | 4 | 4 | ✓ |
| FullP2Config | imported from P2-Full-A | same | ✓ |
| **Only difference** | **carry long_buf across boundary** | **clear long_buf at boundary** | **designed** |

## Cosmetic Differences (non-experimental)
1. SAVE_STEPS: P2Replay saves at {0,4096,8192,12288,16384,20480,24576} vs P2-Full-A {0,4096,8192,12288,24576}
2. GPU assignment: P2Replay Persistent=GPU0, Reset128=GPU1 (parallel). PPO both on GPU0 (sequential).
