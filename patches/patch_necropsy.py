import re
p = "/workspace/mechanism_UED/dicode_src/src/dicode/craftax_evaluation.py"
src = open(p, newline='').read()
def sub1(pat, rep, s, n=1):
    r, cnt = re.subn(pat, rep, s)
    assert cnt == n, f"anchor {pat!r}: expected {n}, got {cnt}"
    return r
# 1) import
src = sub1(r"from minicraftax\.envs\.craftax import CraftaxAugObsTrain",
           "from minicraftax.envs.craftax import CraftaxAugObsTrain\nfrom dicode.necropsy import necro_init, necro_step", src)
# 2) init after max_floor zeros
src = sub1(r"(?P<i>[ \t]*)max_floor = jnp\.zeros\(\(num_envs,\), dtype=jnp\.int32\)",
           lambda m: m.group(0) + "\n" + m.group("i") + "necro = necro_init(num_envs, _state_core(env_state), detail)", src)
# 3) carry tuples x3 (init, unpack, return) -- indices 8-14 downstream stay valid
src = sub1(r"max_floor,(\s+)rng,", lambda m: f"max_floor,{m.group(1)}necro,{m.group(1)}rng,", src, n=3)
# 4) update line after forensics block
src = sub1(r"(?P<i>[ \t]*)max_floor = jnp\.where\(finished_mask, max_floor, jnp\.maximum\(max_floor, lvl\)\)",
           lambda m: m.group(0) + "\n" + m.group("i") + "necro = necro_step(necro, _state_core(env_state), core, active_mask, first_done_now, detail)", src)
# 5) unpack after final_carry[14]
src = sub1(r"(?P<i>[ \t]*)_max_floor = final_carry\[14\]",
           lambda m: m.group(0) + "\n" + m.group("i") + "_necro = final_carry[15]", src)
# 6) details dict extension
src = sub1(r'"floor_at_done": _floor_at_done, "max_floor": _max_floor,',
           '"floor_at_done": _floor_at_done, "max_floor": _max_floor, **_necro,', src)
# 7) smoke lever: flag-gated max_timesteps (default 8192 = identical)
src = sub1(r"max_timesteps=8192,",
           'max_timesteps=int(config.eval.get("max_timesteps", 8192)) if hasattr(config, "eval") else 8192,', src)
open(p, "w", newline='').write(src)
print("necropsy patched OK (7 anchors)")
