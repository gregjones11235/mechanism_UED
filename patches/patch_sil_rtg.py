p = "/workspace/mechanism_UED/dicode_src/experiments/analysis/sil_collect.py"
src = open(p, newline='').read()
def sub1(old, new):
    global src
    assert src.count(old) == 1, f"anchor x{src.count(old)}: {old[:50]!r}"
    src = src.replace(old, new)
sub1('            ring_act=jnp.zeros((num_envs, K), dtype=jnp.int32),\n',
     '            ring_act=jnp.zeros((num_envs, K), dtype=jnp.int32),\n'
     '            ring_cum=jnp.zeros((num_envs, K), dtype=jnp.float32),\n')
sub1('            cap_act=jnp.zeros((num_envs, K), dtype=jnp.int32),\n',
     '            cap_act=jnp.zeros((num_envs, K), dtype=jnp.int32),\n'
     '            cap_cum=jnp.zeros((num_envs, K), dtype=jnp.float32),\n')
sub1('            wptr = c["wptr"] + active.astype(jnp.int32)\n',
     '            ring_cum = c["ring_cum"].at[ar, slot].set(\n'
     '                jnp.where(active, c["ret"], c["ring_cum"][ar, slot]))\n'
     '            wptr = c["wptr"] + active.astype(jnp.int32)\n')
sub1('            cap_wptr = jnp.where(snap, wptr, c["cap_wptr"])\n',
     '            cap_cum = jnp.where(snap[:, None], ring_cum, c["cap_cum"])\n'
     '            cap_wptr = jnp.where(snap, wptr, c["cap_wptr"])\n')
sub1('                      ring_obs=ring_obs, ring_act=ring_act, wptr=wptr,\n',
     '                      ring_obs=ring_obs, ring_act=ring_act, ring_cum=ring_cum, wptr=wptr,\n')
sub1('                      cap_obs=cap_obs, cap_act=cap_act, cap_wptr=cap_wptr,\n',
     '                      cap_obs=cap_obs, cap_act=cap_act, cap_cum=cap_cum, cap_wptr=cap_wptr,\n')
sub1('        keys = ["cap_obs", "cap_act", "cap_wptr", "captured", "finished", "ret",\n',
     '        keys = ["cap_obs", "cap_act", "cap_cum", "cap_wptr", "captured", "finished", "ret",\n')
sub1('        segs_o, segs_a, metas = [], [], []\n',
     '        segs_o, segs_a, segs_r, metas = [], [], [], []\n')
sub1('            segs_a.append(out["cap_act"][e][order].astype(np.int16))\n',
     '            segs_a.append(out["cap_act"][e][order].astype(np.int16))\n'
     '            segs_r.append((float(rets[e]) - out["cap_cum"][e][order]).astype(np.float16))\n')
sub1('                                obs=np.stack(segs_o), act=np.stack(segs_a))\n',
     '                                obs=np.stack(segs_o), act=np.stack(segs_a),\n'
     '                                rtg=np.stack(segs_r),\n'
     '                                ret=np.array([m["ret"] for m in metas], dtype=np.float32))\n')
open(p, "w", newline='').write(src)
print("RTG+meta patch OK (10 anchors)")
