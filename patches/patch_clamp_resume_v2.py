import re
p = "/workspace/mechanism_UED/dicode_src/src/dicode/utils/general/train_state_utils.py"
src = open(p, newline='').read()          # 保留原始行尾,不翻译
nl = "\r\n" if "\r\n" in src else "\n"
pat = re.compile(
    r"(?P<i>[ \t]*)def linear_schedule\(count\):\r?\n"
    r"(?P<j>[ \t]*)frac = 1\.0 - \(count // \(config\.num_minibatches \* config\.update_epochs\)\) / TOTAL_GLOBAL_UPDATES\r?\n"
    r"(?P<k>[ \t]*)return config\.min_lr \+ \(config\.lr - config\.min_lr\) \* frac"
)
ms = list(pat.finditer(src))
assert len(ms) == 1, f"expected exactly 1 unclamped live-path schedule, found {len(ms)}"
g = ms[0]
body = [
    f"{g['i']}def linear_schedule(count):",
    f"{g['j']}# [CRASH FIX v2] Clamp anneal. Past horizon (TOTAL=15300 global updates for 2e9)",
    f"{g['j']}# frac goes negative -> lr crosses zero at TOTAL*(1+min_lr/(lr-min_lr)) -> Adam does",
    f"{g['j']}# gradient ascent (idx 15454 @2e-4, 17000 @2e-5; matches all 7 crash sites).",
    f"{g['j']}# ff6b956 clamped ppo_tr.py only, which is UNREACHABLE on resume: the restored",
    f"{g['j']}# TrainState keeps THIS tx. Mirror the clamp here (bit-identical in-horizon).",
    f"{g['j']}frac = jnp.maximum(",
    f"{g['j']}    0.0,",
    f"{g['j']}    1.0 - (count // (config.num_minibatches * config.update_epochs)) / TOTAL_GLOBAL_UPDATES,",
    f"{g['j']})",
    f"{g['k']}return config.min_lr + (config.lr - config.min_lr) * frac",
]
open(p, "w", newline='').write(src[:g.start()] + nl.join(body) + src[g.end():])
print(f"patched OK (nl={'CRLF' if nl=='\r\n' else 'LF'}) ->", p)
