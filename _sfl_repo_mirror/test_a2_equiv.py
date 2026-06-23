#!/usr/bin/env python
"""A2 数值等价单测：gen_ppo_loss_batched vs 原 vmap(gen_ppo_loss) over L levels。

验证 flatten nested vmap 后 loss + grads 与原版逐元素一致(rtol 1e-5)。
不改方法、纯工程 → 必须 bit/数值级等价(允许浮点重排级 1e-5)。
"""
import sys
sys.path.insert(0, "sfl/train")
import jax, jax.numpy as jnp
import numpy as np
from pcgrl_generator import (
    make_pcgrl_env, make_generator_network, make_generator_state,
    gen_rollout_traj, assign_terminal_reward, compute_gae,
    gen_ppo_loss, gen_ppo_loss_batched,
)

def main():
    rng = jax.random.PRNGKey(7)
    env, _ = make_pcgrl_env(map_size=11, max_board_scans=3.0)
    net, _ = make_generator_network(env)
    rng, r = jax.random.split(rng)
    gstate, _ = make_generator_state(r, env, net, ["difficulty"])
    params = gstate["params"][0]
    rep = env.rep
    n_agents = 1
    act_shape = tuple(int(s) for s in rep.act_shape)
    action_dim = int(rep.action_space.n) // (act_shape[0] * act_shape[1])

    # 造 L=8 条真实轨迹(够验证等价,不用64省时)
    L = 8
    rng, rr = jax.random.split(rng)
    roll_rngs = jax.random.split(rr, L)
    trajs, _maps, last_vals = jax.vmap(lambda k: gen_rollout_traj(env, net, params, k))(roll_rngs)
    # 假 terminal reward + GAE
    rng, rt = jax.random.split(rng)
    trew = jax.random.normal(rt, (L,))
    def _bg(traj, tr, lv):
        traj = assign_terminal_reward(traj, tr)
        adv, ret = compute_gae(traj, lv, 0.99, 0.95)
        return traj, adv, ret
    trajs, advs, rets = jax.vmap(_bg)(trajs, trew, last_vals)

    # --- 原版: vmap(gen_ppo_loss) over L, 再 mean ---
    def orig_loss(p):
        def _per(traj, adv, ret):
            return gen_ppo_loss(p, net, traj, adv, ret, n_agents, act_shape, action_dim)
        ls, auxs = jax.vmap(_per)(trajs, advs, rets)
        return ls.mean(), jax.tree_map(lambda x: x.mean(), auxs)
    # --- A2: batched ---
    def new_loss(p):
        return gen_ppo_loss_batched(p, net, trajs, advs, rets, n_agents, act_shape, action_dim)

    (lo, auxo), go = jax.value_and_grad(orig_loss, has_aux=True)(params)
    (ln, auxn), gn = jax.value_and_grad(new_loss, has_aux=True)(params)

    print(f"loss  orig={float(lo):.8f}  batched={float(ln):.8f}  |diff|={abs(float(lo)-float(ln)):.2e}")
    for name, a, b in zip(["actor", "value", "entropy"], auxo, auxn):
        print(f"  {name:8s} orig={float(a):.8f} batched={float(b):.8f} |diff|={abs(float(a)-float(b)):.2e}")
    # grads 等价
    gflat_o = jax.tree_util.tree_leaves(go)
    gflat_n = jax.tree_util.tree_leaves(gn)
    max_gdiff = max(float(jnp.max(jnp.abs(x - y))) for x, y in zip(gflat_o, gflat_n))
    print(f"\nmax |grad diff| over all params = {max_gdiff:.2e}")

    ok_loss = np.allclose(float(lo), float(ln), rtol=1e-5, atol=1e-6)
    ok_grad = max_gdiff < 1e-4
    print(f"\nloss 等价(rtol1e-5)={ok_loss}  grad 等价(<1e-4)={ok_grad}")
    print("A2_EQUIV_PASS" if (ok_loss and ok_grad) else "A2_EQUIV_FAIL")

if __name__ == "__main__":
    main()
