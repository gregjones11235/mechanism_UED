p = "/workspace/mechanism_UED/dicode_src/experiments/analysis/sil_collect.py"
src = open(p, newline='').read()
def sub1(old, new):
    global src
    assert src.count(old) == 1, f"anchor x{src.count(old)}: {old[:60]!r}"
    src = src.replace(old, new)

# 1) 签名 + 2) 调用点带 mode
sub1("def make_collect(config, env, env_params, num_envs, num_steps):\n",
     'def make_collect(config, env, env_params, num_envs, num_steps, mode="descend"):\n')
sub1("    collect_jit = jax.jit(make_collect(config, env, env_params, num_envs, num_steps))\n",
     "    collect_jit = jax.jit(make_collect(config, env, env_params, num_envs, num_steps, mode))\n")
# 3) main 解析 mode
sub1('    tag = str(sil.get("tag", "run"))\n',
     '    tag = str(sil.get("tag", "run"))\n    mode = str(sil.get("mode", "descend"))\n')
# 4) carry 初始化加 stay_cnt / prev_drink
sub1('            postcnt=jnp.zeros((num_envs,), dtype=jnp.int32),\n',
     '            postcnt=jnp.zeros((num_envs,), dtype=jnp.int32),\n'
     '            stay_cnt=jnp.zeros((num_envs,), dtype=jnp.int32),\n'
     '            prev_drink=jnp.full((num_envs,), 9.0),\n')
# 5) 触发块按模式分支(mode 为编译期常量,零运行时开销)
sub1('            x2 = active & (~c["crossed2"]) & (lvl >= 2)\n'
     '            crossed2 = c["crossed2"] | x2\n'
     '            postcnt = jnp.where(x2, K_POST, c["postcnt"])\n'
     '            counting = crossed2 & (~c["captured"]) & active\n'
     '            postcnt = jnp.where(counting & (~x2), postcnt - 1, postcnt)\n'
     '            snap = counting & ((postcnt <= 0) | ndone)\n',
     '            if mode == "descend":\n'
     '                x2 = active & (~c["crossed2"]) & (lvl >= 2)\n'
     '                crossed2 = c["crossed2"] | x2\n'
     '                postcnt = jnp.where(x2, K_POST, c["postcnt"])\n'
     '                counting = crossed2 & (~c["captured"]) & active\n'
     '                postcnt = jnp.where(counting & (~x2), postcnt - 1, postcnt)\n'
     '                snap = counting & ((postcnt <= 0) | ndone)\n'
     '                stay_cnt = c["stay_cnt"]; prev_drink = c["prev_drink"]; mark = x2\n'
     '            elif mode == "stay":\n'
     '                # 64-step continuous floor-2 residence window\n'
     '                on2 = active & (lvl >= 2)\n'
     '                stay_cnt = jnp.where(on2, c["stay_cnt"] + 1, 0)\n'
     '                crossed2 = c["crossed2"] | on2\n'
     '                snap = active & (~c["captured"]) & (stay_cnt >= K)\n'
     '                postcnt = c["postcnt"]; prev_drink = c["prev_drink"]; mark = snap\n'
     '            elif mode == "resource":\n'
     '                # thirst-refill event: drink rose while previously < 3\n'
     '                drink = core.player_drink.astype(jnp.float32)\n'
     '                snap = active & (~c["captured"]) & (c["prev_drink"] < 3.0) & (drink > c["prev_drink"] + 0.5)\n'
     '                prev_drink = jnp.where(active, drink, c["prev_drink"])\n'
     '                crossed2 = c["crossed2"] | (active & (lvl >= 2))\n'
     '                postcnt = c["postcnt"]; stay_cnt = c["stay_cnt"]; mark = snap\n'
     '            else:\n'
     '                raise ValueError(f"unknown sil.mode {mode!r}")\n')
# 6) cross_step 记 mark(descend=入层步,stay/resource=快照步)
sub1('            cross_step = jnp.where(x2, c["step"], c["cross_step"])\n',
     '            cross_step = jnp.where(mark, c["step"], c["cross_step"])\n')
# 7) carry 回写
sub1('                      crossed2=crossed2, postcnt=postcnt, captured=captured,\n',
     '                      crossed2=crossed2, postcnt=postcnt, captured=captured,\n'
     '                      stay_cnt=stay_cnt, prev_drink=prev_drink,\n')
# 8) keep 过滤按模式(资源健康度双筛仅 descend 有意义)
sub1('        keep = capd & (rets >= thr) & (out["e1food"] >= fmin) & (out["e1drink"] >= dmin)\n',
     '        keep = capd & (rets >= thr)\n'
     '        if mode == "descend":\n'
     '            keep = keep & (out["e1food"] >= fmin) & (out["e1drink"] >= dmin)\n')
# 9) 打印带 mode
sub1('    print(f"[SIL-COLLECT] donor={tag} ckpt={ckpt} envs={num_envs} steps={num_steps} "\n',
     '    print(f"[SIL-COLLECT] donor={tag} mode={mode} ckpt={ckpt} envs={num_envs} steps={num_steps} "\n')
open(p, "w", newline='').write(src)
print("v1.1 patch OK (9 anchors)")
