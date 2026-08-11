"""EnvironmentCandidateGenerator — expands plan_k + AxisDirectives into raw
candidates (§4, rewritten for the directive-driven loop, C8).

Deterministic expansion of a reconciled CurriculumPlan into exactly
RAW_CANDIDATES (64) environment-level TaskParams CANDIDATES, distributed over
the FUNDED families proportionally to their slot budget (largest-remainder
rounding), and within each family across the board's AxisDirectives for that
family. The historical ``i % len(axes)`` / ``i % 3`` index rotation is
ABOLISHED: a candidate's axis configuration derives ONLY from a directive
(``axis_directive.candidate_axis_config``) — the board's controlled-
experiment specification is the single source of truth for what gets probed.

Directives for families the plan does not fund are coded by the EnvCoder but
NEVER probed — that is honest and recorded in the window bookkeeping. Every
FUNDED family must carry at least one directive, else generation fails
closed (``FUNDED_FAMILY_WITHOUT_DIRECTIVE``): probing a funded family
without a board specification would be an uncontrolled mutation.

Candidates are mock-namespaced (the real TaskParams adapter is BLOCKED) and
each one declares which ledger hypotheses it is meant to DISTINGUISH (ALL
ledger hypotheses of its family — the probe is meant to settle the family's
line of inquiry, not a single index-rotated pick). That binding is what lets
probe feedback flow back to the right hypotheses.

No randomness: everything derives from (window, plan_id, directive content,
index), so a replay reproduces the identical candidate batch and hashes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Sequence

if TYPE_CHECKING:
    from d052.feedback_llm_ued.axis_directive import AxisDirective

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


def generate_candidates_from_directives(
        plan: CurriculumPlan, *,
        directives: Sequence[AxisDirective],
        hypothesis_families: Dict[str, List[str]],
        raw_cap: int = C.RAW_CANDIDATES) -> List[CandidateEnvironment]:
    """Expand one funded plan + its directive batch into exactly ``raw_cap``
    candidates.

    * every funded allocation (slots > 0) MUST have >= 1 directive for its
      family, otherwise generation fails closed;
    * family candidate counts follow the slot budget (largest remainder);
    * within a family, counts are spread over that family's directives in
      deterministic directive_id order (equal weights, largest remainder);
    * each candidate's (axis_values, held_constant_axes) come from
      ``candidate_axis_config(directive)`` — treatment applies ``new_level``,
      control re-measures ``old_level``;
    * ``distinguishes_hypothesis_ids`` = ALL ledger hypotheses of the family.
    """
    # lazy import: axis_directive imports FAMILY_AXES/AXIS_LEVELS from this
    # module, so a module-level import here would be circular
    from d052.feedback_llm_ued.axis_directive import candidate_axis_config

    funded = [a for a in plan.allocations if a.slots > 0]
    if not funded:
        raise ValueError(
            "EMPTY_PLAN_BUDGET: plan carries no funded allocation to expand")

    dir_by_family: Dict[str, List[AxisDirective]] = {}
    for d in directives:
        dir_by_family.setdefault(d.environment_family, []).append(d)
    for fam in dir_by_family:
        dir_by_family[fam].sort(key=lambda d: d.directive_id)

    for a in funded:
        if a.environment_family not in dir_by_family:
            raise ValueError(
                f"FUNDED_FAMILY_WITHOUT_DIRECTIVE: plan funds "
                f"{a.environment_family!r} (slots={a.slots}) but the board "
                f"emitted no AxisDirective for it — probing a funded family "
                f"without a board specification is an uncontrolled mutation "
                f"and is refused")

    counts = _split_counts(raw_cap, [a.slots for a in funded])
    out: List[CandidateEnvironment] = []
    for a, count in zip(funded, counts):
        family = a.environment_family
        fam_directives = dir_by_family[family]
        per_directive = _split_counts(count, [1] * len(fam_directives))
        hyp_ids = sorted(hypothesis_families.get(family, []))
        idx = 0
        for directive, rep_count in zip(fam_directives, per_directive):
            axis_values, held = candidate_axis_config(directive)
            for rep in range(rep_count):
                out.append(CandidateEnvironment(
                    candidate_id=(f"cand-w{plan.window:02d}-{family}-"
                                  f"{idx:02d}"),
                    environment_family=family,
                    axis_values=dict(axis_values),
                    held_constant_axes=dict(held),
                    variant_id=(f"var-{family}-{directive.directive_id}-"
                                f"{rep:02d}"),
                    variant_kind="directive",
                    mutation_axes=[directive.axis],
                    distinguishes_hypothesis_ids=list(hyp_ids),
                    provenance=dict(
                        source=C.SOURCE_CANDIDATE_PROBE,
                        plan_id=plan.plan_id,
                        window=plan.window,
                        generator=C.FEEDBACK_LOOP_VERSION,
                        directive_id=directive.directive_id,
                        directive_hash=directive.directive_hash),
                ))
                idx += 1
    if len(out) != raw_cap:                      # pragma: no cover - guarded
        raise ValueError(
            f"CANDIDATE_COUNT_MISMATCH: {len(out)} != {raw_cap}")
    # deterministic order: plan allocation order, then within-family index
    return out
