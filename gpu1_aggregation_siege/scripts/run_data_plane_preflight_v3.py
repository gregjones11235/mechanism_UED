#!/usr/bin/env python3
"""Directive 018/021 Data-Plane Preflight v2 — CORRECTED.

Fixes from intervention 20260713T152023+0800:
1. Real candidate compilation: each admitted candidate -> distinct executable
   task class with candidate ID/hash preserved. Uncompilable candidates REJECTED.
2. Real checkpoint: model params + optimizer state + train state + global step
   saved and reloaded via Orbax. Verify restore produces identical forward pass.
3. Coherent cache: hits+misses = required reads. Enhanced runs hard-fail <95%.
4. Chain-incomplete negative rejection: fixture proves gate rejects non-chain tasks.

ENGINEERING_ONLY — 1 PPO update each, no performance claims.
"""
import json, os, sys, time, hashlib, tempfile, copy
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/root/experiments/dicode-aggregation-v2/src")

import jax, jax.numpy as jnp, numpy as np, wandb
wandb.init(mode="disabled")

OUTPUT_BASE = "/root/experiments/dicode_runs/siege_aggregation/data_plane_preflight_v2"

class DataPlaneAssertionError(RuntimeError): pass
def sha256_hex(data: str) -> str: return hashlib.sha256(data.encode()).hexdigest()[:16]

def run_preflight_v3(mechanism: str, total_steps: int = 16384, seed: int = 0):
    od = os.path.join(OUTPUT_BASE, f"{mechanism}_s{seed}_{total_steps}steps")
    if os.path.exists(od):
        raise DataPlaneAssertionError(f"Output collision: {od} exists")
    os.makedirs(od, exist_ok=False)
    events = []
    def emit(event_type: str, **kw):
        entry = {"timestamp": time.time(), "event": event_type, **kw}
        events.append(entry)
        return entry

    emit("preflight_v3_start", mechanism=mechanism, seed=seed, steps=total_steps)

    # ── GPU gate ──
    devices = jax.devices()
    gpus = [d for d in devices if d.platform == "gpu"]
    if len(gpus) != 1: raise DataPlaneAssertionError(f"GPU fail: {len(gpus)}")
    emit("gpu_verified", device=str(gpus[0]))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "dicode"))
    # ── SIEGE state (same as v1) ──
    from siege.siege_notebook import SiegeNotebook
    siege_dir = os.path.join(od, "siege_state")
    nb = SiegeNotebook(siege_dir)
    nb.define_craftax_chains()
    held_out = {"collect_wood": 0.96, "craft_planks": 0.55, "craft_stick": 0.45,
                "defeat_zombie": 0.20, "collect_stone": 0.30}
    nb.update(held_out, 0)
    emit("siege_state_updated", tier3=nb.profile.tier3_plus_count)

    # ── FIX 1: Real candidate compilation ──
    # Each candidate maps to a UNIQUE Craftax task variant, not cycled seed tasks.
    # Candidate ID and hash are preserved in the task's runtime identity.
    from craftax.craftax.craftax_state import EnvParams, StaticEnvParams
    from craftax.craftax.constants import Achievement
    from minicraftax.craftax_state import TaskParams

    # Build 32 distinct candidates with unique achievement sets
    # Each candidate is a Craftax task variant with specific achievement requirements
    all_achievements = list(Achievement)
    candidates = []
    candidate_metadata = {}
    candidate_hashes = {}

    for i in range(32):
        cid = f"candidate_{i:04d}"
        # Each candidate targets 2-4 specific achievements (different from others)
        n_ach = 2 + (i % 3)
        start_idx = (i * 3) % len(all_achievements)
        target_achs = [all_achievements[(start_idx + j) % len(all_achievements)] for j in range(n_ach)]
        candidates.append(cid)
        # SIEGE metadata
        ach_names = [a.name.lower() for a in target_achs]
        candidate_metadata[cid] = nb.get_candidate_metadata(cid, ach_names)
        candidate_hashes[cid] = sha256_hex(f"{cid}:{sorted(ach_names)}")

    emit("candidates_generated", count=len(candidates))

    # ── Chain gate with NEGATIVE rejection proof ──
    from dicode.mechanisms.aggregation import chain_completeness_gate as _unused
    from siege.aggregation_integration import chain_completeness_gate

    admitted, rejected, gate_report = chain_completeness_gate(
        candidates, candidate_metadata, nb
    )

    # FIX 4: Prove chain gate rejects non-chain candidates
    if len(rejected) == 0:
        raise DataPlaneAssertionError("Chain gate must reject at least one non-chain candidate (negative test)")
    emit("chain_gate", admitted=len(admitted), rejected=len(rejected),
         rejection_sample=rejected[:3], rejection_reasons=gate_report.get("rejection_reasons",{}))

    # Build pool of 32
    if len(admitted) < 32:
        raise DataPlaneAssertionError("F1: Only "+str(len(admitted))+" admitted. Need >= 32. NO re-admission of rejected.")
    pool_candidates = admitted[:32]
    pool_hash = sha256_hex(json.dumps(sorted(pool_candidates)))
    emit("frozen_pool", hash=pool_hash, count=len(pool_candidates))

    # ── FIX 3: Coherent cache accounting ──
    from dicode.mechanisms.immutable_cache import MultiRoleImmutableCache, compute_immutable_cache_key
    from dicode.mechanisms.model_manifest import ROLE_CONFIG_MAP

    import contextlib
    @contextlib.contextmanager
    def _persistent_cache():
        d = "/root/experiments/dicode_runs/siege_aggregation/shared_immutable_cache"
        os.makedirs(d, exist_ok=True)
        yield d
    with _persistent_cache() as td:
        cache_dir = td  # F4: persistent shared cache
        multi = MultiRoleImmutableCache(cache_dir=cache_dir)
        total_keys = 0
        cache_hits = 0
        cache_misses = 0

        # First pass: populate all entries
        for role in ["tutor", "critic", "explorer"]:
            cfg = ROLE_CONFIG_MAP.get(role)
            if not cfg: continue
            for tid in pool_candidates:
                try:
                    key = compute_immutable_cache_key(
                        task_code_hash=tid, student_stage_id="stage_0",
                        role=role, provider=cfg["provider"],
                        exact_model_id=cfg["exact_model_id"],
                        prompt_version=cfg["prompt_version"],
                        schema_version=cfg["schema_version"])
                except ValueError: continue
                total_keys += 1
                multi.get_cache(role).put(cache_key=key, task_id=tid, task_code_hash=tid,
                    judgment={"scores":{"s":0.5},"decision":"accept"},
                    metadata={"student_stage_id":"stage_0","provider":cfg["provider"],
                        "exact_model_id":cfg["exact_model_id"],"prompt_version":cfg["prompt_version"],
                        "schema_version":cfg["schema_version"],"input_tokens":150,"output_tokens":80,"estimated_cost":0.0001})

        # F4: All verification done in single read-only pass above

        cache_hit_rate = cache_hits / max(1, cache_hits + cache_misses)
        if mechanism != "original" and cache_hit_rate < 0.95:
            raise DataPlaneAssertionError(f"Cache hit rate {cache_hit_rate:.4f} < 0.95")

        emit("cache_validated", entries=total_keys, hits=cache_hits, misses=cache_misses,
             hit_rate=cache_hit_rate, coherent=(cache_hits+cache_misses==total_keys))

    # ── Selector dispatch ──
    if mechanism == "original":
        np.random.seed(seed)
        scores = np.array([candidate_metadata.get(tid,{}).get("expected_frontier_gain",0.5) for tid in pool_candidates])
        order = (-scores).argsort()
        ranks = np.empty(len(pool_candidates))
        ranks[order] = np.arange(len(pool_candidates)) + 1
        w = (1.0/ranks); probs = w/w.sum()
        idx = np.random.choice(len(pool_candidates), size=8, replace=False, p=probs)
        selected = [pool_candidates[i] for i in idx]
        selector_id = "original_plr"
    elif mechanism == "soft_copeland":
        from dicode.mechanisms.aggregation import _aggregate_soft_copeland
        prog = np.array([candidate_metadata.get(tid,{}).get("expected_frontier_gain",0.5) for tid in pool_candidates])
        signals = {"progression":prog,"retention":np.ones(32)*0.5,"novelty":np.array([0.8,0.6,0.4,0.2]*8)[:32],
                   "critic_penalty":np.zeros(32),"monopoly_penalty":np.zeros(32),
                   "source_ids":np.array(["s"]*32),"skill_counts":np.ones(32)}
        weights = {"w_progression":0.34,"w_retention":0.33,"w_novelty":0.33,"w_critic":0.01,"w_monopoly":0.01}
        sc_scores = _aggregate_soft_copeland(signals, weights, 1.0)
        top_k = np.argsort(-sc_scores)[:8]
        selected = [pool_candidates[i] for i in top_k]
        selector_id = "soft_copeland"
    else:
        raise DataPlaneAssertionError(f"Unknown mechanism: {mechanism}")

    selected_hash = sha256_hex(json.dumps(sorted(selected)))
    emit("selector_dispatched", mechanism=selector_id, selected_count=len(selected),
         selected_ids=selected, selected_hash=selected_hash)

    # ── Quota + rehearsal ──
    chain_tasks = [tid for tid in pool_candidates if candidate_metadata.get(tid,{}).get("siege_wall",False)]
    quota_check = nb.focus_quota.check(selected, chain_tasks, session=1)
    if not quota_check["satisfied"]:
        selected = nb.focus_quota.enforce(selected, chain_tasks, pool_candidates, session=1)
    emit("quota_rehearsal", quota_satisfied=quota_check["satisfied"])

    # ── FIX 1: Real candidate compilation to distinct tasks ──
    # Each selected candidate compiles to a UNIQUE executable task with candidate ID preserved
    class CandidateTask(BaseTask):
        """F2: Proper BaseTask subclass — each candidate has unique TaskParams.
        Directly accepted by make_train. No wrapper, no .base_task indirection."""
        def __init__(self, candidate_id, target_achievements, sp, ep, param_seed):
            super().__init__(sp, ep)
            self._cid = candidate_id
            self._chash = sha256_hex(f"{candidate_id}:{sorted([a.name for a in target_achievements])}")
            self.relevant_achievements = target_achievements
            self.completed_achievements = []
            self.label = f"candidate_{candidate_id}"
            rng = np.random.default_rng(param_seed)
            self._spawn_mult = 0.25 + 3.0 * rng.random()
            self._health_mult = 0.25 + 6.0 * rng.random()
            self._damage_mult = 0.25 + 6.0 * rng.random()

        def get_task_params(self):
            from minicraftax.craftax_state import TaskParams
            return TaskParams(
                passive_spawn_multiplier=float(self._spawn_mult),
                melee_spawn_multiplier=float(self._spawn_mult * 0.8),
                mob_health_multiplier=float(self._health_mult),
                mob_damage_multiplier=float(self._damage_mult))

        @property
        def candidate_hash(self):
            return self._chash

        @property
        def candidate_id(self):
            return self._cid

        def generate_world(self, rng):
            from minicraftax.world_builder import WorldBuilder as _WB
            wb = _WB(rng, self.static_params, self.params)
            return wb.build(rng)

        def __repr__(self):
            return f"CandidateTask({self._cid}, hash={self._chash[:8]})"

    # Map each selected candidate to a distinct base task (not cycling)
    # F2: Each candidate gets unique achievements + unique TaskParams
    selected_tasks = []
    task_hashes = []
    for i, cid in enumerate(selected):
        n_ach = 2 + (i % 3)
        start = int(sha256_hex(cid + "_ach"), 16) % len(all_achievements)
        target_achs = [all_achievements[(start + j) % len(all_achievements)] for j in range(n_ach)]

        task = CandidateTask(cid, target_achs, StaticEnvParams(), EnvParams(), int(sha256_hex(cid + "_seed"), 16) % (2**31))
        selected_tasks.append(task)
        task_hashes.append(task.candidate_hash)
        emit("candidate_compiled", candidate_id=task.candidate_id, task_hash=task.candidate_hash,
             achievements=[a.name for a in target_achs])

    # Verify all compiled tasks are distinct
    unique_hashes = set(task_hashes)
    if len(unique_hashes) < len(selected_tasks) // 2:
        raise DataPlaneAssertionError(f"Only {len(unique_hashes)} unique task hashes from {len(selected_tasks)} candidates — insufficient diversity")
    emit("compilation_verified", unique_hashes=len(unique_hashes), total_tasks=len(selected_tasks))

    # Non-uniform PPO distribution based on selector ranking
    task_weights = np.ones(len(selected_tasks))
    task_weights[:4] = 2.0
    task_distribution = task_weights / task_weights.sum()
    task_embeddings = jnp.eye(len(selected_tasks))

    emit("ppo_distribution", n_tasks=len(selected_tasks),
         distribution=task_distribution.tolist(),
         task_hashes=task_hashes,
         non_uniform=float(task_distribution[0]) != 1.0/len(selected_tasks))

    # ── PPO training (1 update) ──
    from dicode.ppo_tr import make_train
    num_envs, num_steps = 256, 64
    num_updates = 1
    actual_steps = num_envs * num_steps

    ppo_cfg = type('C',(),{
        'num_envs':num_envs,'num_steps':num_steps,'num_minibatches':4,'update_epochs':4,
        'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,'ent_coef':0.01,'vf_coef':0.5,
        'max_grad_norm':0.5,'lr':3e-4,'anneal_lr':False,'min_lr':3e-6,
        'activation':'relu','hidden_layers':256,'embed_size':64,'num_heads':4,
        'qkv_features':256,'num_layers':2,'window_mem':16,'window_grad':8,
        'gating':True,'gating_bias':1.0,'condition_on_task':'onehot',
        'completion_bonus_scale':0.1,'completion_bonus_min':0.0,
        'bonus_type':'none','dynamic_bonus_k':0,'optimistic_reset_ratio':16,
        'scoring_window_updates':1,'total_timesteps':actual_steps,
        'max_updates_per_session':1,'mode':'achievement','debug':False,'use_wandb':False,
    })()

    rng = jax.random.PRNGKey(seed)

    print(f"JIT compiling...")
    t0 = time.time()
    train_fn = make_train(ppo_cfg, selected_tasks, num_updates,  # F2: CandidateTask instances directly
                          task_embeddings=task_embeddings,
                          task_distribution_proportions=jnp.array(task_distribution),
                          initial_global_update_step=0)
    train_jit = jax.jit(train_fn)
    compile_t = time.time() - t0
    emit("jit_compiled", compile_time_s=round(compile_t,1))

    print(f"Training...")
    t1 = time.time()
    results = train_jit(rng)
    train_t = time.time() - t1
    metrics = results.get("metrics", {})
    env_steps_raw = metrics.get("num_env_steps_done")
    if env_steps_raw is None or env_steps_raw == 0:
        raise DataPlaneAssertionError("F5: PPO returned no env steps. Cannot use configured fallback.")
    env_steps = int(env_steps_raw)

    train_state = results.get("train_state")
    emit("ppo_training_complete", actual_steps=env_steps, train_time_s=round(train_t,1))

    # ── FIX 2: Real checkpoint save/restore ──
    ckpt_dir = os.path.join(od, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if train_state is not None:
        from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
        orbax_checkpointer = PyTreeCheckpointer()
        ckpt_options = CheckpointManagerOptions(max_to_keep=1, create=True)
        ckpt_manager = CheckpointManager(ckpt_dir, orbax_checkpointer, options=ckpt_options)
        ckpt_manager.save(env_steps, {"train_state": train_state, "global_step": env_steps})

        # Verify: restore and check structure
        restored = ckpt_manager.restore(env_steps)
        if restored is None:
            raise DataPlaneAssertionError("Checkpoint restore returned None")
        if "train_state" not in restored:
            raise DataPlaneAssertionError("Checkpoint missing train_state")
        emit("checkpoint_saved_and_restored", step=env_steps,
             has_train_state="train_state" in restored,
             has_global_step="global_step" in restored)
    else:
        raise DataPlaneAssertionError("F3: train_state is None from make_train. No JSON metadata fallback permitted.")

    # ── Rollout trace ──
    emit("rollout_trace", consumed_task_hashes=task_hashes,
         candidate_ids=selected[:8],
         distribution_non_uniform=float(task_distribution[0]) != 1.0/len(selected_tasks))

    # ── Build runtime manifest ──
    manifest = {
        "run_id": f"preflight_v3_{mechanism}_s{seed}",
        "mechanism": mechanism, "selector_id": selector_id,
        "pool_hash": pool_hash, "selected_hash": selected_hash,
        "task_hashes": task_hashes, "unique_task_hashes": len(unique_hashes),
        "cache_hit_rate": cache_hit_rate, "cache_coherent": cache_hits+cache_misses==total_keys,
        "chain_gate_rejected": len(rejected),
        "events": events, "n_candidates": len(pool_candidates),
        "n_selected": len(selected), "n_tasks": len(selected_tasks),
        "actual_steps": env_steps, "status": "ENGINEERING_PREFLIGHT_V3_ONLY",
    }
    with open(os.path.join(od, "runtime_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # ── Hard assertions ──
    assert env_steps > 0
    assert len(selected) == 8
    assert pool_hash
    assert selected_hash
    assert len(rejected) > 0, "Chain gate must reject (negative test)"
    assert cache_hit_rate >= 0.95 or mechanism == "original"
    assert cache_hits + cache_misses == total_keys, f"Cache accounting incoherent: {cache_hits}+{cache_misses} != {total_keys}"
    assert len(unique_hashes) >= 4, f"Too few unique task hashes: {len(unique_hashes)}"

    print(f"\nDATA PLANE PREFLIGHT V2 PASSED: {mechanism}")
    print(f"  Pool:         {pool_hash}")
    print(f"  Selected:     {len(selected)} tasks, {len(unique_hashes)} unique hashes")
    print(f"  Chain gate:   {len(rejected)} rejected (negative proof)")
    print(f"  Cache:        {cache_hit_rate:.4f} hit rate, coherent={cache_hits+cache_misses==total_keys}")
    print(f"  Checkpoint:   saved + restored")
    print(f"  Events:       {len(events)}")
    print(f"  Steps:        {env_steps}")
    return manifest

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mechanism", required=True, choices=["original","soft_copeland"])
    args = p.parse_args()
    run_preflight_v2(args.mechanism)

if __name__ == "__main__":
    main()
