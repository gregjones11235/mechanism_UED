# P2-v0 KNOWN ISSUES (9 confirmed defects + provenance caveats)

These defects are why P2-v0 is labeled `P2-v0-exploratory-invalid-for-attribution`
and `SEVERE_REGRESSION_FROM_HEALTHY_STAGE4_BASELINE`. They are being fixed in P2-v1
(see the single-director mandate sections 六/七). P2-v0 checkpoints must NEVER be
resumed, and their results must NEVER be used to attribute effectiveness to replay /
hindsight / the P2 mechanism.

## Confirmed algorithmic / engineering defects

1. obs/action/value/log_prob time-step misalignment
   - stage4_continue_launcher.py rollout loop appends obs AFTER env.step (next-step obs)
     but action/value/log_prob are computed from the pre-step obs.
   - Stored transition is (obs[t+1], action[t], value[t], log_prob[t], reward[t], done[t]) — misaligned.

2. Transformer memory window0 misalignment
   - buf["init_mem"] is copied AFTER the step-0 memory update, so it holds post-step-0
     memory, not pre-step-0 memory. The learner's window0 uses this as the pre-window
     memory -> off by one. (_build_memory_carry is dead code.)

3. Missing native on-policy PPO main update
   - Each rollout only does: replay.sample() -> relabel_sample() -> learner.update()
     (an off-policy replay update). There is NO native on-policy PPO main update on the
     freshly collected rollout batch. training_log confirms: only off_loss is logged.

4. Hindsight relabel is a no-op w.r.t. the loss
   - hindsight.relabel_* changes target_achievements, but long_context_learner._off_policy_loss
     never reads target_achievements/achievements. Relabel changes fields/logs only — not
     network input, reward, or loss.

5. Global np.random action sampling — unrecoverable RNG
   - Action sampling uses global np.random.choice(...) (not saved, not restorable).
   - JAX a_rng is split then unused. trajectory_replay._get_by_id is broken (_traj_id
     never assigned; id-based sampling always returns None).

6. GAE double-reversal
   - GAE uses a reverse scan plus manual reverse indexing (double reversal) — fragile and
     a source of indexing error.

7. Nonterminal bootstrap error
   - Bootstrap uses lv = v[-1] (the window's last value), not V(next_obs) for nonterminal
     segments -> bootstrap bias.

8. replay_meta + RNG incomplete restore
   - replay_meta.pkl + RNG states are not fully saved/restored. The 122880 checkpoint has
     NO replay_meta.pkl at all -> NOT_FULLY_RESUMABLE; detect_latest_checkpoint_step skips
     122880 and falls back to 98304.

9. Trajectory-id lookup broken
   - (Linked to #5) _get_by_id never matches because _traj_id is never assigned at insert.

## Provenance caveats

- NOT a git repository. Source-file SHA256 (SHA256SUMS) + run-time source_hashes.json are
  the authoritative provenance.
- SOURCE-HASH DISCREPANCY: current on-disk src/stage4_continue_launcher.py (55058e06...)
  differs from the run-time hash that produced the 122880 checkpoint (2c13d267...). That one
  file was edited after 122880 was produced; its exact producing bytes are NOT recoverable.
  The other 6 src files match their run-time hashes.

## Result interpretation

- 98304  SR_STAGE4_NATIVE = 8/512  = 1.5625%
- 122880 SR_STAGE4_NATIVE = 20/512 = 3.90625%
- Healthy session175 baseline (same Stage4-native caliber) = 25/64 = 39.0625%
- => BOTH P2-v0 results are a SEVERE REGRESSION vs the healthy baseline.
- 122880 > 98304 is ONLY local recovery within a degraded state — NOT proof the P2
  mechanism works. See RESULTS_LIMITATIONS.md.
