p = "/workspace/mechanism_UED/dicode_src/experiments/analysis/sil_collect.py"
src = open(p, newline='').read()
def sub1(old, new):
    global src
    assert src.count(old) == 1, f"anchor x{src.count(old)}: {old[:60]!r}"
    src = src.replace(old, new)
sub1('def make_collect(config, env, env_params, num_envs, num_steps, mode="descend"):\n',
     'def make_collect(config, env, env_params, num_envs, num_steps, mode="descend", skill_idx=0):\n')
sub1('    collect_jit = jax.jit(make_collect(config, env, env_params, num_envs, num_steps, mode))\n',
     '    collect_jit = jax.jit(make_collect(config, env, env_params, num_envs, num_steps, mode, skill_idx))\n')
sub1('    mode = str(sil.get("mode", "descend"))\n',
     '    mode = str(sil.get("mode", "descend"))\n'
     '    skill_name = str(sil.get("skill", "MAKE_IRON_ARMOUR"))\n'
     '    if mode == "skill":\n'
     '        from craftax.craftax.constants import Achievement\n'
     '        skill_idx = int(Achievement[skill_name].value)\n'
     '        print(f"[SIL-COLLECT] skill target: {skill_name} (idx {skill_idx})")\n'
     '    else:\n'
     '        skill_idx = 0\n')
sub1('            prev_drink=jnp.full((num_envs,), 9.0),\n',
     '            prev_drink=jnp.full((num_envs,), 9.0),\n'
     '            skill_prev=jnp.zeros((num_envs,), dtype=jnp.bool_),\n')
sub1('            if mode == "descend":\n',
     '            skill_prev = c["skill_prev"]\n'
     '            if mode == "descend":\n')
sub1('            else:\n                raise ValueError(f"unknown sil.mode {mode!r}")\n',
     '            elif mode == "skill":\n'
     '                # snapshot the 64-step window ending at the target\n'
     '                # achievement\'s first flip (prep -> craft execution)\n'
     '                bit = core.achievements[:, skill_idx] > 0.5\n'
     '                snap = active & (~c["captured"]) & bit & (~c["skill_prev"])\n'
     '                skill_prev = c["skill_prev"] | (bit & active)\n'
     '                crossed2 = c["crossed2"] | (active & (lvl >= 2))\n'
     '                postcnt = c["postcnt"]; stay_cnt = c["stay_cnt"]; prev_drink = c["prev_drink"]; mark = snap\n'
     '            else:\n                raise ValueError(f"unknown sil.mode {mode!r}")\n')
sub1('                      stay_cnt=stay_cnt, prev_drink=prev_drink,\n',
     '                      stay_cnt=stay_cnt, prev_drink=prev_drink, skill_prev=skill_prev,\n')
open(p, "w", newline='').write(src)
print("skill-mode patch OK (7 anchors)")
