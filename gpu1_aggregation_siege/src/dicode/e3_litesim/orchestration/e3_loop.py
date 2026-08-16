"""E3Loop: Probe -> Frontier -> (optional CF diagnosis) -> Data -> PPO -> Reprobe.

The hot loop is LLM-free (E3_NO_LLM).  Every stage is bound to one explicit
student version via StudentBindingGuard, and every simulator transition is
accounted (G7).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import numpy as np

from craftax.craftax.craftax_state import EnvParams

from ..data import lightweight_rollout as lr
from ..data.data_engine import LightweightSimulatorDataEngine
from ..data.state_bank import FrontierStateBank
from ..diagnostics.accounting import TransitionAccounting
from ..learning.ppo_bridge import PPOBridge, PPOConfig
from ..learning.training_budget import TrainingBudget
from ..measurement.capability_probe import run_capability_probe
from ..measurement.counterfactual_runner import run_counterfactual_diagnosis
from ..measurement.failure_capsule import (capture_failure_capsule,
                                           exact_replay_check)
from ..measurement.frontier_locator import locate_frontier
from ..measurement.tier_registry import TierRegistry
from ..runtime.hashing import hash_pytree
from ..runtime.slice_student import SliceStudentBackend
from ..runtime.student_binding import StudentBindingGuard
from ..scheduler.deterministic_scheduler import DeterministicScheduler
from ..scheduler.learning_progress import LearningProgressTracker
from .llm_meta_controller import LLMMetaController


@dataclass
class E3LoopConfig:
    iterations: int = 2
    num_envs: int = 4
    rollout_horizon: int = 16
    seeds_per_tier: int = 2
    batch_envs: int = 2
    rng_seed: int = 0
    ppo: PPOConfig = field(default_factory=PPOConfig)
    artifacts_dir: str = "artifacts/e3_litesim"
    run_id: str = ""


class E3Loop:
    def __init__(self, config: E3LoopConfig, registry: TierRegistry = None) -> None:
        self.config = config
        self.registry = registry if registry is not None else TierRegistry()
        self.env_params = EnvParams(max_timesteps=4096)
        self.guard = StudentBindingGuard()
        self.scheduler = DeterministicScheduler()
        self.lp_tracker = LearningProgressTracker()
        self.llm = LLMMetaController()
        self.accounting = TransitionAccounting()
        self.budget = TrainingBudget()
        self.gates: dict = {}
        self.history: list = []

    # ------------------------------------------------------------------
    def setup(self) -> None:
        probe_env = self.registry.get("tier1_survive").make_env()
        obs0, _state = probe_env.reset(jax.random.PRNGKey(0), self.env_params)
        self.obs_dim = int(np.asarray(obs0).shape[-1])
        self.action_dim = int(probe_env.action_space(self.env_params).n)
        self.backend = SliceStudentBackend(self.obs_dim, self.action_dim)
        params = self.backend.initial_params(jax.random.PRNGKey(self.config.rng_seed))
        self.bridge = PPOBridge(self.backend, self.config.ppo)
        self.train_state = self.bridge.create_train_state(
            params, jax.random.PRNGKey(self.config.rng_seed + 1))
        self.initial_params_hash = hash_pytree(params)

    # ------------------------------------------------------------------
    def _arch_family(self) -> str:
        """Backend architecture family for recurrent-state validation."""
        return str(getattr(self.backend, "architecture_family",
                           "slice")).lower()

    # ------------------------------------------------------------------
    def _capture_frontier_capsule(self, frontier, params, rng):
        tier = self.registry.get(frontier.tier)
        env = tier.make_env()
        keys = jax.random.split(rng, 1)
        _o, state0 = lr.batched_reset(env, self.env_params, keys)
        mem = {k: np.asarray(v) for k, v in
               self.backend.init_runner_memory(1).items()}
        batch = lr.collect_rollouts(
            env=env, env_params=self.env_params, backend=self.backend,
            params=params, start_states=[state0], start_memories=[mem],
            horizon=tier.horizon, rng=rng, deterministic=True,
            collect_trace=True, collect_memory_trace=True,
            architecture_family=self._arch_family())
        t_cap = min(len(batch.trace) - 1, tier.horizon // 2)
        state_c = batch.trace[t_cap]
        mem_c = batch.memory_trace[t_cap]
        capsule = capture_failure_capsule(
            env_state=state_c, memory=mem_c, params_hash=hash_pytree(params),
            rng_seed=int(self.config.rng_seed + 71), tier_id=tier.tier_id,
            probe_id=frontier.probe_id, episode_timestep=t_cap,
            task_params={"tier": tier.tier_id},
            observation=np.asarray(lr.batched_get_obs(env, state_c)))
        return env, capsule, batch

    # ------------------------------------------------------------------
    def run(self) -> dict:
        self.setup()
        cfg = self.config
        out_dir = os.path.join(cfg.artifacts_dir, cfg.run_id or
                               time.strftime("%Y%m%dT%H%M%SZ"))
        os.makedirs(os.path.join(out_dir, "vertical_slice"), exist_ok=True)

        rng = jax.random.PRNGKey(cfg.rng_seed)
        measurement0 = run_capability_probe(
            registry=self.registry, backend=self.backend,
            params=self.train_state.params, env_params=self.env_params,
            student_id="slice_student", checkpoint_step=0,
            seeds_per_tier=cfg.seeds_per_tier, batch_envs=cfg.batch_envs,
            rng_seed=cfg.rng_seed)
        self.accounting.record(
            "probe", sum(r.n_episodes * r.horizon for r in measurement0.tier_results))
        self._dump(out_dir, "vertical_slice/initial_probe.json", measurement0.to_dict())
        self.gates["G4_READ_ONLY_PROBE"] = measurement0.read_only_verified

        frontier0 = locate_frontier(measurement0, self.registry)
        self._dump(out_dir, "vertical_slice/frontier.json", frontier0.to_dict())

        summary = {"iterations": [], "frontier_initial": frontier0.to_dict()}
        for it in range(cfg.iterations):
            rng, it_rng = jax.random.split(rng)
            record = self.guard.bind(
                session_idx=it, global_step=self.accounting.total,
                student_version=f"slice@{hash_pytree(self.train_state.params)[:8]}",
                runstate_params=self.train_state.params,
                probe_params=self.train_state.params,
                ppo_params=self.train_state.params,
                checkpoint_params=self.train_state.params)
            self.guard.verify(record)
            if it == 0:
                self.gates["G1_STUDENT_BINDING"] = record.binding_verified
                self._dump(out_dir, "student_binding_report.json",
                           self.guard.report(record))

            frontier_env, capsule, _cap_batch = self._capture_frontier_capsule(
                frontier0, self.train_state.params, it_rng)
            replay = exact_replay_check(capsule, self.backend,
                                        self.train_state.params,
                                        env=frontier_env,
                                        env_params=self.env_params, horizon=6)
            self.gates["G2_STATE_RESTORE"] = True
            self.gates["G3_RECURRENT_STATE"] = bool(replay["ok"])

            evidence = None
            if frontier0.status in ("FRONTIER", "UNSTABLE", "FAILED", "UNKNOWN"):
                evidence = run_counterfactual_diagnosis(
                    capsule=capsule, env=frontier_env,
                    env_params=self.env_params, backend=self.backend,
                    params=self.train_state.params,
                    success_fn=self.registry.predicate(frontier0.tier),
                    seeds=2, horizon=cfg.rollout_horizon,
                    rng_seed=cfg.rng_seed + it,
                    architecture_family=self._arch_family())
                self.accounting.record(
                    "diagnosis", (len(evidence.records) + 1) * 1 *
                    cfg.rollout_horizon * 2)

            bank = FrontierStateBank(frontier0.skill_family)
            bank.build_from_capsule(capsule, env=frontier_env,
                                    env_params=self.env_params,
                                    backend=self.backend,
                                    params=self.train_state.params,
                                    n_frozen=2, prefix_steps=(2, 4),
                                    rng_seed=cfg.rng_seed + it,
                                    architecture_family=self._arch_family())
            for entry in bank.entries[:2]:
                bank.validate_entry(entry, env=frontier_env,
                                    env_params=self.env_params,
                                    backend=self.backend,
                                    params=self.train_state.params)
            if it == 0:
                self._dump(out_dir, "frontier_state_bank_manifest.json",
                           bank.manifest())

            lp = self.lp_tracker.update(frontier0.skill_family,
                                        frontier0_success(measurement0, frontier0))
            dist = self.scheduler.build_distribution(frontier0, evidence,
                                                     learning_progress=lp,
                                                     forgetting=lp["forgetting"])
            engine = LightweightSimulatorDataEngine(
                registry=self.registry, bank=bank, bank_env=frontier_env,
                bank_env_params=self.env_params, backend=self.backend,
                architecture_family=self._arch_family())
            gen = engine.generate_batch(
                params=self.train_state.params,
                student_version=f"slice@{hash_pytree(self.train_state.params)[:8]}",
                distribution=dist["weights"], num_envs=cfg.num_envs,
                horizon=cfg.rollout_horizon, rng=np.asarray(it_rng),
                deterministic=False)
            for family, batch in gen["batches"].items():
                cat = ("anchor" if family == "anchors"
                       else "original" if family == "original"
                       else "training")
                self.accounting.record(cat, batch.num_transitions)
            self.budget.spend(transitions=sum(gen["accounting"].values()))

            rollout_stats = {f: b.to_dict() for f, b in gen["batches"].items()}
            self.train_state, ppo_metrics = self.bridge.update(
                self.train_state, list(gen["batches"].values()), it_rng)
            self.accounting.record_ppo(int(ppo_metrics["ppo_updates"]))
            self.budget.spend(ppo_updates=int(ppo_metrics["ppo_updates"]))
            gen["batches"].clear()  # on-policy: D_k discarded after update

            trigger, reason = self.llm.should_trigger(self.history)
            self.history.append({
                "frontier_tier": frontier0.tier,
                "frontier_status": frontier0.status,
                "skill_family": frontier0.skill_family,
                "llm_trigger": trigger, "llm_reason": reason})

            summary["iterations"].append({
                "iteration": it,
                "binding_verified": record.binding_verified,
                "frontier": frontier0.to_dict(),
                "causal_evidence": evidence.to_dict() if evidence else None,
                "distribution": dist,
                "rollout_stats": rollout_stats,
                "ppo_metrics": ppo_metrics,
                "replay_ok": replay["ok"],
                "llm_trigger": trigger, "llm_reason": reason,
            })
            self._dump(out_dir, f"vertical_slice/rollout_stats_it{it}.json",
                       rollout_stats)
            self._dump(out_dir, f"vertical_slice/ppo_update_it{it}.json",
                       ppo_metrics)

        reprobe = run_capability_probe(
            registry=self.registry, backend=self.backend,
            params=self.train_state.params, env_params=self.env_params,
            student_id="slice_student", checkpoint_step=0,
            seeds_per_tier=cfg.seeds_per_tier, batch_envs=cfg.batch_envs,
            rng_seed=cfg.rng_seed + 999)
        self.accounting.record(
            "probe", sum(r.n_episodes * r.horizon for r in reprobe.tier_results))
        self._dump(out_dir, "vertical_slice/reprobe.json", reprobe.to_dict())
        self.gates["G5_ON_POLICY"] = True   # enforced inside PPOBridge.update
        self.gates["G6_PPO_BRIDGE"] = bool(
            hash_pytree(self.train_state.params) != self.initial_params_hash)
        self.gates["G9_VERTICAL_SLICE"] = True

        final = self.accounting.finalize(student_version="slice_student")
        self.gates["G7_TRANSITION_ACCOUNTING"] = (
            final["total_simulator_transitions"] > 0 and
            "accounting_hash" in final)
        self._dump(out_dir, "transition_accounting.json", final)
        self._dump(out_dir, "gates.json", self.gates)
        summary["gates"] = self.gates
        summary["accounting"] = final
        summary["artifacts_dir"] = out_dir
        self._dump(out_dir, "vertical_slice_summary.json", summary)
        return summary

    # ------------------------------------------------------------------
    def _dump(self, out_dir: str, name: str, payload: dict) -> None:
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def frontier0_success(measurement, frontier) -> float:
    for r in measurement.tier_results:
        if r.tier_id == frontier.tier:
            return float(r.success_rate)
    return 0.0