"""EnvironmentCandidateGenerator — expands plan_{k} into raw candidates (§4).

Deterministic expansion of a reconciled CurriculumPlan into exactly
RAW_CANDIDATES (64) environment-level TaskParams CANDIDATES, distributed over
families proportionally to their slot budget (largest-remainder rounding).
Candidates are mock-namespaced (the real TaskParams adapter is BLOCKED) and
each one declares which ledger hypotheses it is meant to DISTINGUISH — that
binding is what lets probe feedback flow back to the right hypothesis.

No randomness: everything derives from (window, plan_id, family, index), so a
replay reproduces the identical candidate batch and hashes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from d052.feedback_llm_ued import constants as C
from d052.feedback_llm_ued.feedback_contracts import (
    CandidateEnvironment,
    CurriculumPlan,
)

#: natural induction knobs per environment family (environment-level ONLY)
FAMILY_AXES: Dict[str, tuple] = {
    "threat_distance_family": ("threat_distance_grading", "threat_count"),
    "resource_pressure_family": ("resource_pressure",),
    "day_night_rest_need_family": ("day_night_rest_need",
                                   "rest_need_pressure",
                                   "safe_rest_area_availability"),
    "visibility_family": ("visibility", "view_occlusion"),
    "multi_threat_interference_family": ("multi_threat_interference",),
    "long_term_memory_family": ("long_term_memory_requirement",),
    "global_task_conflict_family": ("global_task_conflict",),
}

AXIS_LEVELS = ("low", "medium", "high")


def _split_counts(total: int, weights: Sequence[int]) -> List[int]:
    """Largest-remainder apportionment of ``total`` over integer weights."""
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("EMPTY_PLAN_BUDGET: no slots to expand")
    raw = [total * w / wsum for w in weights]
    floors = [int(x) for x in raw]
    left = total - sum(floors)
    order = sorted(range(len(raw)),
                   key=lambda i: (-(raw[i] - floors[i]), i))
    for i in order[:left]:
        floors[i] += 1
    return floors


def generate_candidates(plan: CurriculumPlan, *,
                        hypothesis_families: Dict[str, List[str]],
                        raw_cap: int = C.RAW_CANDIDATES
                        ) -> List[CandidateEnvironment]:
    """Expand one plan into exactly ``raw_cap`` candidates.

    ``hypothesis_families`` maps environment_family -> sorted hypothesis ids,
    so each candidate can declare which hypotheses it distinguishes (rotated
    by index for within-family diversity).
    """
    if plan.allocations:
        families = [a.environment_family for a in plan.allocations
                    if a.slots > 0]
        weights = [a.slots for a in plan.allocations if a.slots > 0]
    else:
        families = list(C.ENVIRONMENT_FAMILIES)
        weights = [1] * len(families)
    counts = _split_counts(raw_cap, weights)

    out: List[CandidateEnvironment] = []
    for family, count in zip(families, counts):
        axes = FAMILY_AXES.get(family, (C.MUTATION_AXES[0],))
        hyp_ids = list(hypothesis_families.get(family, []))
        for i in range(count):
            axis = axes[i % len(axes)]
            level = AXIS_LEVELS[i % len(AXIS_LEVELS)]
            held = {a: "medium" for a in axes if a != axis}
            distinguishes = []
            if hyp_ids:
                distinguishes = [hyp_ids[i % len(hyp_ids)]]
            out.append(CandidateEnvironment(
                candidate_id=f"cand-w{plan.window:02d}-{family}-{i:02d}",
                environment_family=family,
                axis_values={axis: level},
                held_constant_axes=held,
                variant_id=f"var-{family}-{i:02d}",
                variant_kind="perturb",
                mutation_axes=[axis],
                distinguishes_hypothesis_ids=distinguishes,
                provenance=dict(
                    source=C.SOURCE_CANDIDATE_PROBE,
                    plan_id=plan.plan_id,
                    window=plan.window,
                    generator=C.FEEDBACK_LOOP_VERSION),
            ))
    if len(out) != raw_cap:                      # pragma: no cover - guarded
        raise ValueError(
            f"CANDIDATE_COUNT_MISMATCH: {len(out)} != {raw_cap}")
    # deterministic order: family, then index (already built in order)
    return out
