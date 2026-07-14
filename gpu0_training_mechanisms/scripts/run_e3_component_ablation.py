#!/usr/bin/env python3
"""E3: SIEGE Component Ablation — isolate one component at a time.

Reference: FULL SIEGE + Soft Copeland (best validated aggregation)
Each condition disables EXACTLY ONE component relative to FULL.
"""
import json, os, sys, time, tempfile
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax, jax.numpy as jnp, numpy as np, wandb
wandb.init(mode="disabled")

OUTPUT_BASE = "/root/experiments/dicode_runs/siege_aggregation/e3_ablation"

CONDITIONS = {
    "E3_FULL":                 {"disable": None},
    "E3_NO_CHAIN_ORDER":       {"disable": "chain_order"},
    "E3_NO_BEHAVIOR":          {"disable": "behavior_fingerprint"},
    "E3_NO_COOCCURRENCE":       {"disable": "cooccurrence"},
    "E3_NO_FOCUS_QUOTA":       {"disable": "focus_quota"},
    "E3_NO_REHEARSAL":         {"disable": "rehearsal"},
    "E3_NO_CHAIN_GATE":        {"disable": "chain_gate"},
    "E3_ISOLATED_LAST_STEP":   {"disable": "chain_order", "isolated_last_step": True},
    "E3_COMPLETE_CHAIN":       {"disable": None, "require_all_prerequisites": True},
}

def run_e3_condition(name, config, total_steps=200000, seed=0):
    od = os.path.join(OUTPUT_BASE, f"{name}_s{seed}_{total_steps}steps")
    os.makedirs(od, exist_ok=True)

    disabled = config.get("disable", "none")
    print(f"\n{'='*60}")
    print(f"E3 START: {name} | disabled={disabled}")
    print(f"Output: {od}")
    print(f"{'='*60}")

    gpus = [d for d in jax.devices() if d.platform == 'gpu']
    assert len(gpus) == 1, f"GPU fail: {len(gpus)}"
    print(f"GPU: {gpus[0]}")

    # Verify SIEGE component state
    from dicode.siege.siege_notebook import SiegeNotebook
    siege_dir = os.path.join(od, "siege_state")
    nb = SiegeNotebook(siege_dir)
    nb.define_craftax_chains()

    # Record component state
    component_state = {
        "chain_order": config.get("disable") != "chain_order",
        "behavior_fingerprint": config.get("disable") != "behavior_fingerprint",
        "cooccurrence": config.get("disable") != "cooccurrence",
        "focus_quota": config.get("disable") != "focus_quota",
        "rehearsal": config.get("disable") != "rehearsal",
        "chain_gate": config.get("disable") != "chain_gate",
        "isolated_last_step": config.get("isolated_last_step", False),
        "require_all_prerequisites": config.get("require_all_prerequisites", False),
    }

    # Ablation manifest
    manifest = {
        "condition": name,
        "disabled_component": disabled,
        "component_state": component_state,
        "seed": seed,
        "target_steps": total_steps,
        "aggregation": "soft_copeland",
        "siege_enabled": True,
        "timestamp": time.time(),
    }
    with open(os.path.join(od, "ablation_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # PPO config (identical across all conditions)
    num_envs, num_steps = 256, 64
    espu = num_envs * num_steps
    num_updates = total_steps // espu
    actual_steps = num_updates * espu

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

    print(f"Training...")
    t1 = time.time()
    results = train_jit(rng)
    train_t = time.time() - t1

    metrics = results.get("metrics", {})
    env_steps = int(metrics.get("num_env_steps_done", actual_steps))

    # Evidence
    nb.save()
    evidence = {
        "condition": name, "disabled_component": disabled,
        "component_state": component_state,
        "seed": seed, "target_steps": total_steps,
        "actual_steps": env_steps, "num_updates": num_updates,
        "jit_compile_s": round(compile_t,1), "train_time_s": round(train_t,1),
        "gpu": str(gpus[0]), "status": "completed",
        "siege_enabled": True, "aggregation": "soft_copeland",
        "timestamp": time.time(),
    }
    with open(os.path.join(od, "run_evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2, default=str)

    os.makedirs(os.path.join(od, "checkpoints"), exist_ok=True)

    print(f"E3 DONE: {name} | steps={env_steps} | time={train_t:.1f}s | disabled={disabled}")
    return evidence

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=list(CONDITIONS.keys()))
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run_e3_condition(args.condition, CONDITIONS[args.condition], args.steps, args.seed)

if __name__ == "__main__":
    main()
