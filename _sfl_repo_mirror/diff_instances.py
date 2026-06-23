#!/usr/bin/env python
"""逐字段对比 generator instances vs 海选 instances 的 dtype/shape/weak_type。
定位为何 generator instances 让 train_and_eval_step 编译爆炸而海选不会。"""
import sys
sys.path.insert(0, "sfl/train")
import jax, jax.numpy as jnp
from hydra import compose, initialize
from omegaconf import OmegaConf
from jaxmarl.environments.jaxnav.jaxnav_env import JaxNav
from pcgrl_generator import (
    make_pcgrl_env, make_generator_network, make_generator_state,
    gen_rollout_batch, gen_batch_to_env_instances,
)

def sig(inst, name):
    print(f"\n=== {name} ===")
    for f in inst.__dataclass_fields__ if hasattr(inst, "__dataclass_fields__") else inst._fields:
        v = getattr(inst, f)
        a = jax.eval_shape(lambda x: x, v) if not hasattr(v, "dtype") else v
        wt = getattr(v, "weak_type", "?")
        print(f"  {f:12s} dtype={v.dtype} shape={v.shape} weak_type={wt}")

def main():
    with initialize(config_path="sfl/train/config", version_base=None):
        cfg = compose(config_name="jaxnav-sfl")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    ep = dict(cfg["env"]["env_params"])
    rng = jax.random.PRNGKey(0)

    # 海选 env (valid_path_check=False) + generator place_env (valid_path_check=True)
    sel_env = JaxNav(num_agents=cfg["env"]["num_agents"], **ep)
    ep2 = {**ep, "map_params": {**dict(ep["map_params"]), "valid_path_check": True}}
    place_env = JaxNav(num_agents=cfg["env"]["num_agents"], **ep2)

    # generator instances
    genv, _ = make_pcgrl_env(map_size=11, max_board_scans=3.0)
    gnet, _ = make_generator_network(genv)
    rng, r = jax.random.split(rng)
    gstate, _ = make_generator_state(r, genv, gnet, ["difficulty"])
    gp = gstate["params"][0]
    rng, ra, rb = jax.random.split(rng, 3)
    emaps = gen_rollout_batch(genv, gnet, gp, ra, 8)
    gen_insts, _ = gen_batch_to_env_instances(emaps, place_env, rb)
    sig(gen_insts, "GENERATOR instances (gen_batch_to_env_instances, place_env valid_path=True)")

    # 海选 instances: SFL 海选用 env.sample_test_case 或类似;这里直接 reset 一批看 EnvInstance 结构
    # 用 sel_env 采样 8 个 instance(海选路径的 instance 来源)
    rng, rc = jax.random.split(rng)
    rcs = jax.random.split(rc, 8)
    try:
        sel_insts = jax.vmap(lambda k: sel_env.sample_test_case(k))(rcs)
        sig(sel_insts, "海选 instances (sel_env.sample_test_case)")
    except Exception as e:
        print(f"\n海选 sample_test_case 失败({e}); 改用 reset 看 EnvInstance")
    print("\nDIFF_DONE")

if __name__ == "__main__":
    main()
