"""End-to-end deterministic pipeline (training-free).

build shared frozen pool -> attach normalized role signals -> unified selector ->
execution-mapping certificate per selected candidate. Runs 0 timesteps.

Run:  PYTHONPATH=gpu1_aggregation_siege python -m d052.examples.example_pipeline
"""
from __future__ import annotations

import json

from d052.execution import build_execution_certificate, canonical_compiled_spec
from d052.generation import build_pool
from d052.schemas.selector import SelectorConfig, SelectorType
from d052.selectors import CandidateSignals, SelectorSignals, select

_TP = {"passive_spawn_multiplier": 1.0, "melee_spawn_multiplier": 1.0,
       "mob_health_multiplier": 1.0, "mob_damage_multiplier": 1.0}

# three raw candidates over canonical achievements
RAW = [
    {"task_id": "cand_a", "task_params": dict(_TP),
     "target_achievements": ["collect_wood", "place_table"]},
    {"task_id": "cand_b", "task_params": dict(_TP),
     "target_achievements": ["eat_cow"]},
    {"task_id": "cand_c", "task_params": dict(_TP),
     "target_achievements": ["defeat_kobold"]},
]

# normalized role signals (already rank_percentile_v1-scaled to [0,1]) per candidate
SIGNALS = {
    "cand_a": {"role_scores": {"tutor": 0.9, "explorer": 0.4, "critic": 0.6},
               "critic_reject": False, "critic_penalty": 0.1},
    "cand_b": {"role_scores": {"tutor": 0.5, "explorer": 0.8, "critic": 0.7},
               "critic_reject": False, "critic_penalty": 0.2},
    "cand_c": {"role_scores": {"tutor": 0.2, "explorer": 0.6, "critic": 0.3},
               "critic_reject": True, "critic_penalty": 0.9},  # critic veto
}


def main() -> int:
    pool = build_pool("shared_pool_demo", RAW)
    print(f"pool_id={pool.pool_id} count={pool.candidate_count} "
          f"pool_hash={pool.pool_hash[:16]}... frozen={pool.frozen}")

    signals = SelectorSignals(
        pool_hash=pool.pool_hash,
        candidates=[CandidateSignals(candidate_id=c.task_id, **SIGNALS[c.task_id])
                    for c in pool.candidates])

    config = SelectorConfig(selector=SelectorType.SOFT_COPELAND, k=2, seed=1234)
    result = select(config, pool, signals)
    print(f"\nselector={result.selector.value} status={result.selection_status.value}")
    print(f"selected_ids={result.selected_ids} "
          f"rejected_by_critic={result.rejected_by_critic}")
    print(f"selection_hash={result.selection_hash[:16]}...")

    by_id = {c.task_id: c for c in pool.candidates}
    print("\nexecution-mapping certificates:")
    for cid in result.selected_ids:
        cand = by_id[cid]
        cert = build_execution_certificate(
            cand, canonical_compiled_spec(cand, training_task_id=f"train-{cid}"))
        print(f"  {cid}: targets={cert.canonical_names} ids={cert.canonical_ids} "
              f"dim={cert.goal_vector_dim} ones={cert.goal_vector_ones} "
              f"obs={cert.student_obs_dim} "
              f"executed_as_intended={cert.executed_as_intended}")

    print("\nD052_LONG_TRAINING_RUNS=0 (this pipeline launches nothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
