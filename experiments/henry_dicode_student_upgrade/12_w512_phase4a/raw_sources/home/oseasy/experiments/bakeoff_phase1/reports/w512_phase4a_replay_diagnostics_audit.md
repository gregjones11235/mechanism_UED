# W512 Phase4A Replay Diagnostics Audit

## Summary

Replay训练诊断确认：P2 Replay作为训练稳定化机制运作，而非长程信用分配机制。

## Key Diagnostics

| Metric | Persistent | Reset128 |
|--------|-----------|----------|
| Rollouts | 12 | 12 |
| Replay updates | 11 | 11 |
| Accepted | 11 | 11 |
| KL rejected | 0 | 0 |
| Episodes collected | 21 | 23 |
| Hindsight eligible | 44/44 | 44/44 |
| Final KL | 0.001 | 0.001 |
| ESS range | 0.39-1.00 | 0.39-1.00 |
| ratio_max range | 1.0-26.6 | 1.0-26.7 |
| NaN | No | No |
| Time | 938.7s | 955.8s |

## Why only 11 updates (not 12)

Rollout 0 completes with 2 episodes in buffer. can_sample() requires at least one
trajectory >= 129 steps. Short episodes from rollout 0 don't meet this threshold.
First update occurs after rollout 1.

## KL trajectory (both arms nearly identical)

Rollout 1: kl=0.025 → Rollout 5: kl=0.009 → Rollout 11: kl=0.001

All updates accepted at scale=1.0 (first trial). No scale reduction ever needed.

## ESS trajectory

ESS drops from 1.0 to ~0.39 (rollout 9) then recovers to ~0.96 (rollout 11).
This indicates moderate off-policy divergence mid-training, recovering as policy converges.

## Characterization

**Supported:** Replay stabilized W512 training and recovered control-level performance.
**Not supported:** Replay solved long-range credit assignment or enhanced cross-rollout memory.
