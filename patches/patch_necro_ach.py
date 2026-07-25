p = "/workspace/mechanism_UED/dicode_src/src/dicode/necropsy.py"
src = open(p, newline='').read()
def sub1(old, new):
    global src
    assert src.count(old) == 1, f"anchor x{src.count(old)}: {old[:50]!r}"
    src = src.replace(old, new)
sub1('        food_at_death=z(), drink_at_death=z(),\n',
     '        food_at_death=z(), drink_at_death=z(),\n'
     '        ach_at_done=jnp.zeros((num_envs, core.achievements.shape[1]), dtype=jnp.float32),\n')
sub1('    out["drink_at_death"] = jnp.where(first_done, nxt.player_drink.astype(jnp.float32), necro["drink_at_death"])\n',
     '    out["drink_at_death"] = jnp.where(first_done, nxt.player_drink.astype(jnp.float32), necro["drink_at_death"])\n'
     '    out["ach_at_done"] = jnp.where(first_done[:, None], nxt.achievements.astype(jnp.float32), necro["ach_at_done"])\n')
open(p, "w", newline='').write(src)
print("ach patch OK (2 anchors)")
