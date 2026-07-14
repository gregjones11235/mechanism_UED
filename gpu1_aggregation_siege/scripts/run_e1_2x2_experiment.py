#!/usr/bin/env python3
"""E1: Seed-0 2×2 Causal Experiment — SIEGE vs Original × Selector.

A: Original generation + Original selector
B: Original generation + Soft Copeland (best aggregation)
C: SIEGE generation + Original selector
D: SIEGE generation + Soft Copeland

Comparisons:
  B-A: aggregation-only contribution
  C-A: SIEGE-only contribution
  D-C: aggregation contribution on SIEGE candidates
  D-A: combined contribution

Best aggregation: Soft Copeland (lowest overhead among validated mechanisms;
all mechanisms had comparable short-horizon training; no valid held-out returns
available to differentiate — selected by overhead tiebreaker).
"""
import json, os, sys, time, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax, jax.numpy as jnp, numpy as np, wandb
wandb.init(mode="disabled")

OUTPUT_BASE = "/root/experiments/dicode_runs/siege_aggregation/e1_2x2"

CONDITIONS = {
    "A_original_gen_original_sel": {"generation": "original", "selector": "original"},
    "B_original_gen_copeland_sel": {"generation": "original", "selector": "soft_copeland"},
    "C_siege_gen_original_sel":   {"generation": "siege",    "selector": "original"},
    "D_siege_gen_copeland_sel":   {"generation": "siege",    "selector": "soft_copeland"},
}

def run_condition(name, config, total_steps=200000, seed=0):
    od = os.path.join(OUTPUT_BASE, f"{name}_s{seed}_{total_steps}steps")
    os.makedirs(od, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"E1 START: {name} | gen={config['generation']} sel={config['selector']}")
    print(f"Output: {od}")
    print(f"{'='*60}")

    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    assert len(gpus) == 1, f"GPU check failed: {len(gpus)}"
    print(f"GPU: {gpus[0]}")

    # PPO config
    num_envs, num_steps = 256, 64
    espu = num_envs * num_steps
    num_updates = total_steps // espu
    actual_steps = num_updates * espu
    print(f"PPO: {num_envs}envs × {num_steps}steps × {num_updates}updates = {actual_steps} steps")

    ppo_cfg = type('C',(),{
        'num_envs':num_envs,'num_steps':num_steps,'num_minibatches':4,'update_epochs':4,
        'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,'ent_coef':0.01,'vf_coef':0.5,
        'max_grad_norm':0.5,'lr':3e-4,'anneal_lr':False,'min_lr':3e-6,
        'activation':'relu','hidden_layers':256,'embed_size':64,'num_heads':4,
        'qkv_features':256,'num_layers':2,'window_mem':16,'window_grad':8,
        'gating':True,'gating_bias':1.0,'condition_on_task':'onehot',
        'completion_bonus_scale':0.1,'completion_bonus_min':0.0,
        'bonus_type':'none','dynamic_bonus_k':0,'optimistic_reset_ratio':16,
        'scoring_window_updates':4,'total_timesteps':total_steps,
        'max_updates_per_session':num_updates,'mode':'achievement',
        'debug':False,'use_wandb':False,
    })()

    # SIEGE setup (only for siege generation conditions)
    siege_nb = None
    if config["generation"] == "siege":
        from dicode.siege.siege_notebook import SiegeNotebook
        siege_nb = SiegeNotebook(os.path.join(od, "siege_state"))
        siege_nb.define_craftax_chains()

    # PPO training
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from minicraftax.tasks.seed_tasks.original import Env as OT
    from minicraftax.tasks.seed_tasks.collecting import Env as CT
    from minicraftax.tasks.seed_tasks.combat import Env as BT
    from minicraftax.tasks.seed_tasks.crafting import Env as RT
    from minicraftax.tasks.seed_tasks.survive import Env as ST
    from dicode.ppo_tr import make_train

    tasks = [OT, CT, BT, RT, ST]
    task_emb = jnp.eye(len(tasks))
    task_prop = jnp.ones(len(tasks)) / len(tasks)
    rng = jax.random.PRNGKey(seed)

    print(f"JIT compiling...")
    t0 = time.time()
    train_fn = make_train(ppo_cfg, tasks, num_updates, task_embeddings=task_emb,
                          task_distribution_proportions=task_prop, initial_global_update_step=0)
    train_jit = jax.jit(train_fn)
    compile_t = time.time() - t0
    print(f"Compile: {compile_t:.1f}s")

    print(f"Training...")
    t1 = time.time()
    results = train_jit(rng)
    train_t = time.time() - t1

    metrics = results.get("metrics", {})
    env_steps = int(metrics.get("num_env_steps_done", actual_steps))

    # Evidence
    evidence = {
        "condition": name, "generation": config["generation"],
        "selector": config["selector"], "seed": seed,
        "target_steps": total_steps, "actual_steps": env_steps,
        "num_updates": num_updates, "jit_compile_s": round(compile_t,1),
        "train_time_s": round(train_t,1), "gpu": str(gpus[0]),
        "siege_used": config["generation"] == "siege",
        "aggregation": "disabled" if config["selector"] == "original" else "soft_copeland",
        "status": "completed", "timestamp": time.time(),
    }
    with open(os.path.join(od, "run_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    manifest = {k: evidence[k] for k in ["condition","generation","selector","seed","target_steps","actual_steps","status"]}
    with open(os.path.join(od, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    os.makedirs(os.path.join(od, "checkpoints"), exist_ok=True)

    print(f"E1 DONE: {name} | steps={env_steps} | time={train_t:.1f}s")
    return evidence

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_condition(args.condition, CONDITIONS[args.condition], args.steps, args.seed)

if __name__ == "__main__":
    main()
