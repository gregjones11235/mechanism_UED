"""E3Loop: per-iteration Probe -> Frontier -> Data -> PPO -> Reprobe -> New Frontier.

The hot loop is LLM-free (E3_NO_LLM).  Every iteration closes the adaptive
frontier loop: it probes with the CURRENT student params, locates the frontier
from that probe, builds frontier data, runs one PPO update, then RE-PROBES the
updated student and locates the NEXT frontier, which is the frontier the next
iteration actually consumes.  ``frontier_after`` is persisted as an artifact.

Every stage is bound to one explicit student version via StudentBindingGuard,
and every simulator transition is accounted (G7).
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
from ..learning.ppo_bridge import PPOBridge, PPOConfig, OnPolicyViolation
from ..learning.training_budget import TrainingBudget
from ..measurement.capability_probe import run_capability_probe
from ..measurement.counterfactual_runner import run_counterfactual_diagnosis
from ..measurement.failure_capsule import (capture_failure_capsule,
                                           exact_replay_check, restore_capsule)
from ..measurement.frontier_locator import locate_frontier
from ..measurement.tier_registry import TierRegistry
from ..runtime import recurrent_state
from ..runtime.hashing import hash_pytree
from ..runtime.slice_student import SliceStudentBackend
from ..runtime.student_binding import (StudentBindingGuard, StudentIdentity)
from ..scheduler.deterministic_scheduler import DeterministicScheduler
from ..scheduler.learning_progress import LearningProgressTracker
from .llm_meta_controller import LLMMetaController

# G9 minimums: the vertical slice must close the loop at least twice.
_MIN_PPO_UPDATES = 2
_MIN_REPROBES = 2


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
    # "NoCausal" (default, E3_NO_LLM) or "CF" (counterfactual evidence affects
    # the scheduler distribution).  Both share the same Probe/Frontier/
    # StateBank/Simulator/PPO/Accounting machinery.
    mode: str = "NoCausal"
    # Student identity override.  Empty => derive from the backend
    # (student_id / candidate_id).  checkpoint_step < 0 => derive from backend.
    student_id: str = ""
    checkpoint_step: int = -1
    # Allow a state-start with zero recurrent state (SlowGRU longstate) when it
    # is a legitimate fresh-episode start / memory-reset intervention.  Set
    # True for the reduced-horizon slowgru slice (horizons < SLOW_INTERVAL mean
    # the frontier capsule may still carry zero longstate); keep False (strict)
    # for the full-horizon formal experiment.
    allow_memory_reset_experiment: bool = False


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
        # adaptive-loop bookkeeping (evidence for G9)
        self.student_version: int = 0
        self.n_reprobes: int = 0
        self.n_ppo_update_calls: int = 0
        self.n_ppo_grad_steps: int = 0
        self.no_stale_data: bool = True
        self.no_binding_mismatch: bool = True

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

    def _student_id(self) -> str:
        if self.config.student_id:
            return str(self.config.student_id)
        return str(getattr(self.backend, "student_id",
                           getattr(self.backend, "candidate_id", "slice_student")))

    def _checkpoint_step(self) -> int:
        if self.config.checkpoint_step >= 0:
            return int(self.config.checkpoint_step)
        return int(getattr(self.backend, "checkpoint_step", 0))

    def _identity(self, *, session_idx: int) -> StudentIdentity:
        return StudentIdentity(
            student_id=self._student_id(),
            student_version=int(self.student_version),
            architecture_family=self._arch_family(),
            params_hash=hash_pytree(self.train_state.params),
            checkpoint_step=self._checkpoint_step())

    def _probe(self, *, rng_seed: int):
        measurement = run_capability_probe(
            registry=self.registry, backend=self.backend,
            params=self.train_state.params, env_params=self.env_params,
            student_id=self._student_id(),
            checkpoint_step=self._checkpoint_step(),
            seeds_per_tier=self.config.seeds_per_tier,
            batch_envs=self.config.batch_envs, rng_seed=rng_seed)
        self.accounting.record(
            "probe", sum(r.n_episodes * r.horizon
                         for r in measurement.tier_results))
        self.gates["G4_READ_ONLY_PROBE"] = measurement.read_only_verified
        return measurement

    # ------------------------------------------------------------------
    def _capture_frontier_capsule(self, frontier, params, rng,
                                  allow_memory_reset_experiment: bool = False):
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
            architecture_family=self._arch_family(),
            allow_memory_reset_experiment=allow_memory_reset_experiment)
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
        student_id = self._student_id()

        rng = jax.random.PRNGKey(cfg.rng_seed)

        # Probe_0 -> Frontier_0 (loop preamble)
        probe = self._probe(rng_seed=cfg.rng_seed)
        self._dump(out_dir, "vertical_slice/initial_probe.json", probe.to_dict())
        frontier = locate_frontier(probe, self.registry)
        self._dump(out_dir, "vertical_slice/frontier.json", frontier.to_dict())

        summary = {"iterations": [], "frontier_initial": frontier.to_dict(),
                   "mode": cfg.mode}
        g5_all = True
        g6_all = True

        for it in range(cfg.iterations):
            rng, it_rng = jax.random.split(rng)
            identity = self._identity(session_idx=it)
            record = self.guard.bind(
                session_idx=it, global_step=self.accounting.total,
                identity=identity,
                runstate_params=self.train_state.params,
                probe_params=self.train_state.params,
                ppo_params=self.train_state.params,
                checkpoint_params=self.train_state.params)
            self.guard.verify(record)
            self.no_binding_mismatch = (self.no_binding_mismatch
                                        and record.binding_verified)
            if it == 0:
                self.gates["G1_STUDENT_BINDING"] = record.binding_verified
                self._dump(out_dir, "student_binding_report.json",
                           self.guard.report(record))

            frontier_env, capsule, _cap_batch = self._capture_frontier_capsule(
                frontier, self.train_state.params, it_rng,
                allow_memory_reset_experiment=self.config.allow_memory_reset_experiment)
            replay = exact_replay_check(capsule, self.backend,
                                        self.train_state.params,
                                        env=frontier_env,
                                        env_params=self.env_params, horizon=6)
            # G2: exact-replay determinism across two independent restores.
            self.gates["G2_STATE_RESTORE"] = bool(replay["ok"])
            # G3: recurrent state present + aligned (longstate NOT zeroed for
            # SlowGRU; fast-window keys canonicalized).
            _cap_state, _cap_mem = restore_capsule(capsule)
            recurrent = recurrent_state.validate_memory(
                _cap_mem, 1, architecture_family=self._arch_family())
            self.gates["G3_RECURRENT_STATE"] = bool(replay["ok"]) and recurrent["ok"]

            evidence = None
            if cfg.mode == "CF" and frontier.status in (
                    "FRONTIER", "UNSTABLE", "FAILED", "UNKNOWN"):
                evidence = run_counterfactual_diagnosis(
                    capsule=capsule, env=frontier_env,
                    env_params=self.env_params, backend=self.backend,
                    params=self.train_state.params,
                    success_fn=self.registry.predicate(frontier.tier),
                    seeds=2, horizon=cfg.rollout_horizon,
                    rng_seed=cfg.rng_seed + it,
                    architecture_family=self._arch_family())
                self.accounting.record(
                    "diagnosis", (len(evidence.records) + 1) * 1 *
                    cfg.rollout_horizon * 2)

            bank = FrontierStateBank(frontier.skill_family)
            _manifest, build_transitions = bank.build_from_capsule(
                capsule, env=frontier_env, env_params=self.env_params,
                backend=self.backend, params=self.train_state.params,
                n_frozen=2, prefix_steps=(2, 4), rng_seed=cfg.rng_seed + it,
                architecture_family=self._arch_family(),
                student_version=self.student_version,
                allow_memory_reset_experiment=self.config.allow_memory_reset_experiment)
            self.accounting.record("state_bank_build", build_transitions)
            for entry in bank.entries[:2]:
                chk = bank.validate_entry(entry, env=frontier_env,
                                          env_params=self.env_params,
                                          backend=self.backend,
                                          params=self.train_state.params)
                self.accounting.record("state_bank_validation",
                                       int(chk.get("transitions", 1)))
            if it == 0:
                self._dump(out_dir, "frontier_state_bank_manifest.json",
                           bank.manifest())

            lp = self.lp_tracker.update(frontier.skill_family,
                                        frontier0_success(probe, frontier))
            dist = self.scheduler.build_distribution(frontier, evidence,
                                                     learning_progress=lp,
                                                     forgetting=lp["forgetting"])
            engine = LightweightSimulatorDataEngine(
                registry=self.registry, bank=bank, bank_env=frontier_env,
                bank_env_params=self.env_params, backend=self.backend,
                architecture_family=self._arch_family())
            gen = engine.generate_batch(
                params=self.train_state.params,
                student_version=student_id,
                distribution=dist["weights"], num_envs=cfg.num_envs,
                horizon=cfg.rollout_horizon, rng=np.asarray(it_rng),
                deterministic=False,
                allow_memory_reset_experiment=self.config.allow_memory_reset_experiment)
            for family, batch in gen["batches"].items():
                cat = ("anchor" if family == "anchors"
                       else "original" if family == "original"
                       else "training")
                self.accounting.record(cat, batch.num_transitions)
            self.budget.spend(transitions=sum(gen["accounting"].values()))

            # G5: strict on-policy — every batch must be tagged with the
            # CURRENT student params hash (fail closed before update).
            g5_it = all(b.policy_hash == identity.params_hash
                        for b in gen["batches"].values())
            g5_all = g5_all and g5_it
            if not g5_it:
                self.no_stale_data = False
                raise OnPolicyViolation(
                    "G5 ON_POLICY violation: batch generated by a policy "
                    "other than the current student (stale batch)")

            rollout_stats = {f: b.to_dict() for f, b in gen["batches"].items()}
            self.train_state, ppo_metrics = self.bridge.update(
                self.train_state, list(gen["batches"].values()), it_rng)
            self.accounting.record_ppo(int(ppo_metrics["ppo_updates"]))
            self.budget.spend(ppo_updates=int(ppo_metrics["ppo_updates"]))
            self.n_ppo_update_calls += 1
            self.n_ppo_grad_steps += int(ppo_metrics["ppo_updates"])
            self.student_version += 1
            gen["batches"].clear()  # on-policy: D_k discarded after update

            g6_it = _g6_ok(ppo_metrics)
            g6_all = g6_all and g6_it

            # Reprobe the updated student -> Frontier_{k+1}
            reprobe = self._probe(rng_seed=cfg.rng_seed + 1000 + it)
            self.n_reprobes += 1
            frontier_after = locate_frontier(reprobe, self.registry)
            self._dump(out_dir, f"vertical_slice/reprobe_it{it}.json",
                       reprobe.to_dict())

            trigger, reason = self.llm.should_trigger(self.history)
            self.history.append({
                "frontier_tier": frontier.tier,
                "frontier_status": frontier.status,
                "skill_family": frontier.skill_family,
                "llm_trigger": trigger, "llm_reason": reason})

            summary["iterations"].append({
                "iteration": it,
                "identity": identity.to_dict(),
                "binding_verified": record.binding_verified,
                "frontier_used": frontier.to_dict(),
                "frontier_after": frontier_after.to_dict(),
                "causal_evidence": evidence.to_dict() if evidence else None,
                "distribution": dist,
                "rollout_stats": rollout_stats,
                "ppo_metrics": ppo_metrics,
                "g5_on_policy": g5_it,
                "g6_bridge": g6_it,
                "replay_ok": replay["ok"],
                "recurrent_ok": recurrent["ok"],
                "llm_trigger": trigger, "llm_reason": reason,
            })
            self._dump(out_dir, f"vertical_slice/rollout_stats_it{it}.json",
                       rollout_stats)
            self._dump(out_dir, f"vertical_slice/ppo_update_it{it}.json",
                       ppo_metrics)

            # next iteration MUST consume the new frontier, not the old one
            probe = reprobe
            frontier = frontier_after

        # Frontier_after artifact (post-final-PPO frontier)
        frontier_after_path = os.path.join(out_dir,
                                           "vertical_slice/frontier_after.json")
        self._dump(out_dir, "vertical_slice/frontier_after.json",
                   frontier.to_dict())
        # final reprobe measurement (backward-compatible artifact name)
        self._dump(out_dir, "vertical_slice/reprobe.json", probe.to_dict())

        self.gates["G5_ON_POLICY"] = g5_all
        self.gates["G6_PPO_BRIDGE"] = g6_all

        final = self.accounting.finalize(student_version=student_id)
        self.gates["G7_TRANSITION_ACCOUNTING"] = (
            final["total_simulator_transitions"] > 0
            and bool(final.get("conservation_ok"))
            and "accounting_hash" in final)

        self.gates["G9_VERTICAL_SLICE"] = self._compute_g9(
            frontier_after_path=frontier_after_path)

        self._dump(out_dir, "transition_accounting.json", final)
        self._dump(out_dir, "gates.json", self.gates)
        summary["gates"] = self.gates
        summary["accounting"] = final
        summary["artifacts_dir"] = out_dir
        self._dump(out_dir, "vertical_slice_summary.json", summary)
        return summary

    # ------------------------------------------------------------------
    def _compute_g9(self, *, frontier_after_path: str) -> bool:
        required = {
            "G1_STUDENT_BINDING": bool(self.gates.get("G1_STUDENT_BINDING")),
            "G2_STATE_RESTORE": bool(self.gates.get("G2_STATE_RESTORE")),
            "G3_RECURRENT_STATE": bool(self.gates.get("G3_RECURRENT_STATE")),
            "G4_READ_ONLY_PROBE": bool(self.gates.get("G4_READ_ONLY_PROBE")),
            "G5_ON_POLICY": bool(self.gates.get("G5_ON_POLICY")),
            "G6_PPO_BRIDGE": bool(self.gates.get("G6_PPO_BRIDGE")),
            "G7_TRANSITION_ACCOUNTING": bool(
                self.gates.get("G7_TRANSITION_ACCOUNTING")),
        }
        checks = {
            "ppo_updates_ge_2": self.n_ppo_update_calls >= _MIN_PPO_UPDATES,
            "reprobes_ge_2": self.n_reprobes >= _MIN_REPROBES,
            "student_version_advanced": self.student_version >= self.config.iterations,
            "frontier_after_exists": os.path.isfile(frontier_after_path),
            "no_stale_data": self.no_stale_data,
            "no_binding_mismatch": self.no_binding_mismatch,
            "all_required_gates": all(required.values()),
        }
        self._g9_evidence = {**checks, "required_gates": required,
                             "ppo_update_calls": self.n_ppo_update_calls,
                             "reprobes": self.n_reprobes,
                             "student_version": self.student_version}
        return all(checks.values())

    # ------------------------------------------------------------------
    def _dump(self, out_dir: str, name: str, payload: dict) -> None:
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _g6_ok(ppo_metrics: dict) -> bool:
    """G6 PPO bridge: real update, finite everything, params actually changed."""
    return bool(
        int(ppo_metrics.get("ppo_updates", 0)) > 0
        and ppo_metrics.get("loss_finite", False)
        and ppo_metrics.get("grad_finite", False)
        and ppo_metrics.get("params_finite", False)
        and ppo_metrics.get("params_changed", False)
    )


def frontier0_success(measurement, frontier) -> float:
    for r in measurement.tier_results:
        if r.tier_id == frontier.tier:
            return float(r.success_rate)
    return 0.0
