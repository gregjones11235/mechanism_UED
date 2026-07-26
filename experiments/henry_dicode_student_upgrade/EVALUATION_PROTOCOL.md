# Evaluation Protocol

The archived Stage4-native / S4_dark screen uses:

- goal: `DEFEAT_KOBOLD`
- fixed 256 paired worlds
- stochastic policy
- max 4096 steps per world
- one record per world
- 256 unique seeds
- achievement ever-set before auto-reset
- floor3 reach
- `ENTER_SEWERS`
- death
- timeout
- conditional kill
- mean/median episode length
- paired McNemar
- paired 95% confidence interval

Stage4-native DK SR is not Official FULL Tier3 success rate. Stage4-native is a late-stage bottleneck screen. Official FULL Tier3 remains the end-to-end goal including natural progression to `ENTER_SEWERS` and `DEFEAT_KOBOLD`.

Do not mix control anchors from different evaluator protocols.

