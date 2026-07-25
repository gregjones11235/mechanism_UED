import re
p = "/workspace/mechanism_UED/dicode_src/src/dicode/utils/general/train_state_utils.py"
src = open(p).read()
pat = re.compile(
    r"(?P<i>[ \t]*)def linear_schedule\(count\):\n"
    r"(?P<j>[ \t]*)frac = 1\.0 - \(count // \(config\.num_minibatches \* config\.update_epochs\)\) / TOTAL_GLOBAL_UPDATES\n"
    r"(?P<k>[ \t]*)return config\.min_lr \+ \(config\.lr - config\.min_lr\) \* frac"
)
ms = list(pat.finditer(src))
assert len(ms) == 1, f"expected exactly 1 unclamped live-path schedule, found {len(ms)}"
g = ms[0]
new = (
    f"{g['i']}def linear_schedule(count):\n"
    f"{g['j']}# [CRASH FIX v2] Clamp anneal. Past horizon (TOTAL=15300 global updates for 2e9)\n"
    f"{g['j']}# frac goes negative -> lr crosses zero at TOTAL*(1+min_lr/(lr-min_lr)) -> Adam does\n"
    f"{g['j']}# gradient ascent (idx 15454 @2e-4, 17000 @2e-5; matches all 7 crash sites).\n"
    f"{g['j']}# ff6b956 clamped ppo_tr.py only, which is UNREACHABLE on resume: the restored\n"
    f"{g['j']}# TrainState keeps THIS tx. Mirror the clamp here (bit-identical in-horizon).\n"
    f"{g['j']}frac = jnp.maximum(\n"
    f"{g['j']}    0.0,\n"
    f"{g['j']}    1.0 - (count // (config.num_minibatches * config.update_epochs)) / TOTAL_GLOBAL_UPDATES,\n"
    f"{g['j']})\n"
    f"{g['k']}return config.min_lr + (config.lr - config.min_lr) * frac"
)
open(p, "w").write(src[:g.start()] + new + src[g.end():])
print("patched OK ->", p)
