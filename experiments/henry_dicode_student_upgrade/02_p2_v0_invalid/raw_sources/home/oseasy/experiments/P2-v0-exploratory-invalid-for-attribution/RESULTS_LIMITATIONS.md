# P2-v0 RESULTS_LIMITATIONS (mandatory interpretation constraints)

These results are FROZEN historical artifacts. Any reuse MUST carry these limitations.

1. CALIBER: Both success rates are STAGE4-NATIVE (S4_dark scaffold: floor-2 spawn, gate
   open, winner kit, floor-2 up-ladder removed, needs_depletion_multiplier=0.3,
   monsters_killed[2]=8, target=DEFEAT_KOBOLD, kobold on floor 3). They are NOT Official
   FULL (natural floor-0 spawn) results.

   - 98304  SR_STAGE4_NATIVE_DEFEAT_KOBOLD = 8/512  = 1.5625%
   - 122880 SR_STAGE4_NATIVE_DEFEAT_KOBOLD = 20/512 = 3.90625%

2. HEALTHY BASELINE: The healthy v7 base checkpoint
   (base_ckpt_v7fix55_armA_s0/rl_checkpoints/17500, "session175"), under the SAME frozen
   Stage4-native caliber (stochastic, seed 42, 4096 steps, 64-ep pilot), achieves
   SR_STAGE4_NATIVE_DEFEAT_KOBOLD = 25/64 = 39.0625%
   (pilot evaluator SHA256 06221187ac06d7da59dac64e6273abfc865b3baafdc75615e0808fc5065d26e2).

3. VERDICT: P2-v0 (1.56% / 3.91%) is a SEVERE REGRESSION from the healthy 39.06% baseline
   under the same caliber.

4. NOT FOR ATTRIBUTION: These results are INVALID for attributing effectiveness to replay,
   hindsight, or the P2 mechanism. The run contained the 9 confirmed defects in
   KNOWN_ISSUES.md (transition misalignment, missing native on-policy PPO main update,
   no-op hindsight, unrecoverable RNG, etc.). A regression measured under broken machinery
   cannot isolate any mechanism's contribution.

5. 122880 > 98304 IS NOT PROOF: The improvement 20/512 > 8/512 is ONLY local recovery
   within an already-degraded state. It must NEVER be cited as evidence that P2 works, that
   replay helps, or that hindsight helps. (Also: 122880 restarted the optimizer — fresh
   Adam, moments reset — and is NOT_FULLY_RESUMABLE, missing replay_meta.pkl.)

6. DO NOT:
   - resume training from 98304 or 122880;
   - compare these SRs against D052 (different conditioning family — D052 is 32-slot one-hot,
     obs 8300; P2-v0/session175 is achievement-embedding-67, obs 8335);
   - mix P2-v0 and D052 results;
   - record any missing metric as 0.

7. The 64-ep pilot's 39.06% is itself a pilot estimate, not an exact truth; formal
   comparisons require the same larger evaluation sample (single-director mandate section 九).
